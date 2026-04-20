"""
Deep learning model library for FBSEM PET reconstruction.

Implements:
  - ResUnit_v2       : residual CNN block used as the learned regularizer
  - FBSEMnet_v3      : full model-based network (EM iterations + CNN regularization)
  - WeightedMSELoss  : MSE with boosted foreground signal
  - Trainer          : training loop with validation, LR scheduling, early stopping
  - fbsemInference   : load a checkpoint and run inference

Reference:
    Mehranian et al., "Model-Based Deep Learning PET Image Reconstruction Using
    Forward–Backward Splitting Expectation Maximization", IEEE TRPMS 2020.
    https://doi.org/10.1109/TRPMS.2020.3004408
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn

from models.deeplib import zeroNanInfs, crop, uncrop

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class WeightedMSELoss(nn.Module):
    """MSE with higher weight on foreground (non-background) voxels.

    Voxels whose target value exceeds ``threshold_fraction * target.max()``
    are given ``foreground_weight`` × more gradient signal.  This prevents the
    loss from being dominated by the large number of zero-valued background
    voxels, giving lesions and brain tissue a stronger learning signal.
    """

    def __init__(self, foreground_weight: float = 10.0, threshold_fraction: float = 0.05):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.threshold_fraction = threshold_fraction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        threshold = target.detach().max() * self.threshold_fraction
        weight = torch.where(
            target > threshold,
            torch.full_like(target, self.foreground_weight),
            torch.ones_like(target),
        )
        return (weight * (pred - target).pow(2)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ssim_psnr(pred_t: torch.Tensor, target_t: torch.Tensor):
    """Return mean SSIM and PSNR over a batch of reconstructed images.

    Parameters
    ----------
    pred_t, target_t : torch.Tensor, shape (B, 1, H, W)

    Returns
    -------
    ssim_val : float or None  — None when skimage is not installed
    psnr_val : float
    """
    pred   = pred_t.detach().cpu().numpy().squeeze(1).astype("float32")   # (B, H, W)
    target = target_t.detach().cpu().numpy().squeeze(1).astype("float32")

    mse      = np.mean((pred - target) ** 2, axis=(1, 2))
    max_val  = target.max(axis=(1, 2))
    psnr_val = float(np.mean(20.0 * np.log10(max_val / (np.sqrt(mse) + 1e-8))))

    ssim_val = None
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_val = float(np.mean([
            _ssim(pred[i], target[i], data_range=float(target[i].max()))
            for i in range(pred.shape[0])
        ]))
    except ImportError:
        logger.debug("skimage not installed — SSIM will not be tracked.")

    return ssim_val, psnr_val


# ─────────────────────────────────────────────────────────────────────────────
# CNN regularizer
# ─────────────────────────────────────────────────────────────────────────────

class ResUnit_v2(nn.Module):
    """Residual CNN block used as the learned regularizer inside FBSEMnet_v3.

    Architecture (2-D example, depth=5):
        Conv2d(in_ch → K) → BN → ReLU
        Conv2d(K → K)     → BN → ReLU   (repeated depth-2 times)
        Conv2d(K → 1)     → BN
        + residual (identity = input PET channel)
        ReLU

    When ``in_channels=2``, the input PET image ``x`` and the MR guidance
    image ``y`` are concatenated before the first convolution.  The residual
    connection always uses only the PET channel (``x`` before concatenation),
    so the network predicts a correction to the PET image rather than fitting
    the MR signal.

    Parameters
    ----------
    depth       : total number of convolutional layers (≥ 2)
    num_kernels : number of feature maps in hidden layers
    kernel_size : spatial kernel size (3 is standard)
    in_channels : 1 = PET only, 2 = PET + MR guidance
    is3d        : use 3-D convolutions when True
    """

    def __init__(
        self,
        depth: int,
        num_kernels: int,
        kernel_size: int,
        in_channels: int,
        is3d: bool,
    ):
        super().__init__()
        self.in_channels = in_channels

        Conv = nn.Conv3d if is3d else nn.Conv2d
        BN   = nn.BatchNorm3d if is3d else nn.BatchNorm2d

        layers = [
            Conv(in_channels, num_kernels, kernel_size, padding=1),
            BN(num_kernels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                Conv(num_kernels, num_kernels, kernel_size, padding=1),
                BN(num_kernels),
                nn.ReLU(inplace=True),
            ]
        layers += [
            Conv(num_kernels, 1, kernel_size, padding=1),
            BN(1),
        ]
        self.dcnn = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        identity = x  # single-channel PET image; used as the residual shortcut
        if y is not None:
            x = torch.cat((x, y), dim=1)  # concat MR guidance along channel axis
        out = self.dcnn(x)
        out = self.relu(out + identity)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Model-based reconstruction network
# ─────────────────────────────────────────────────────────────────────────────

class FBSEMnet_v3(nn.Module):
    """Fedorov–Bregman EM network for PET image reconstruction.

    Combines physics-based OSEM iterations with a learned CNN regularizer.
    At each sub-iteration the EM update and CNN regularization are fused via
    the closed-form FBSEM update:

        img_new = 2 * img_em / (
            (1 - γ·S·r) + sqrt((1 - γ·S·r)² + 4·γ·S·img_em)
        )

    where  img_em = EM update,  r = CNN residual,  S = inverse sensitivity,
    and γ is a learnable scalar that balances data fidelity vs regularization.

    Parameters
    ----------
    depth         : conv layers in the ResUnit regularizer
    num_kernels   : feature maps in the regularizer
    kernel_size   : kernel size (3 is standard)
    in_channels   : 1 = PET only, 2 = PET + MR guidance
    is3d          : 3-D reconstruction mode
    reg_cnn_model : regularizer architecture name (only 'resUnit' supported)
    """

    def __init__(
        self,
        depth: int,
        num_kernels: int,
        kernel_size: int,
        in_channels: int = 1,
        is3d: bool = False,
        reg_cnn_model: str = "resUnit",
    ):
        super().__init__()
        if reg_cnn_model.lower() != "resunit":
            raise ValueError(
                f"Unknown reg_cnn_model: {reg_cnn_model!r}. Only 'resUnit' is supported."
            )
        self.regularize = ResUnit_v2(depth, num_kernels, kernel_size, in_channels, is3d)
        # γ: learnable regularization strength; constrained > 0 during training
        self.gamma = nn.Parameter(torch.rand(1), requires_grad=True)
        self.is3d  = is3d

    def forward(
        self,
        PET,
        prompts,
        img=None,
        RS=None,
        AN=None,
        iSensImg=None,
        mrImg=None,
        niters: int = 10,
        nsubs: int = 1,
        tof: bool = False,
        psf: float = 0.0,
        device: str = "cuda",
        crop_factor: float = 0.0,
    ) -> torch.Tensor:
        batch_size  = prompts.shape[0]
        device_obj  = torch.device(device)
        matrixSize  = PET.image.matrixSize

        # Crop/uncrop helpers applied in image space
        if 0 < crop_factor < 1:
            Crop   = lambda x: crop(x, crop_factor, is3d=self.is3d)
            unCrop = lambda x: uncrop(x, matrixSize[0], is3d=self.is3d)
        else:
            Crop   = lambda x: x
            unCrop = lambda x: x

        if self.is3d:
            toTorch = lambda x: zeroNanInfs(
                Crop(torch.from_numpy(x).unsqueeze(1).to(device=device_obj, dtype=torch.float32))
            )
            toNumpy = lambda x: zeroNanInfs(
                unCrop(x)
            ).detach().cpu().numpy().squeeze(1).astype("float32")

            if iSensImg is None:
                iSensImg = PET.iSensImageBatch3D(AN, nsubs, psf).astype("float32")
            if img is None:
                img = np.ones(
                    [batch_size, matrixSize[0], matrixSize[1], matrixSize[2]],
                    dtype="float32",
                )
            if batch_size == 1:
                if iSensImg.ndim == 4:
                    iSensImg = iSensImg[None].astype("float32")
                if img.ndim == 3:
                    img = img[None].astype("float32")
        else:
            reShape = lambda x: x.reshape([batch_size, matrixSize[0], matrixSize[1]], order="F")
            Flatten = lambda x: x.reshape([batch_size, matrixSize[0] * matrixSize[1]], order="F")
            toTorch = lambda x: zeroNanInfs(
                Crop(
                    torch.from_numpy(reShape(x)).unsqueeze(1).to(device=device_obj, dtype=torch.float32)
                )
            )
            toNumpy = lambda x: zeroNanInfs(
                Flatten(unCrop(x).detach().cpu().numpy().squeeze(1))
            )

            if iSensImg is None:
                iSensImg = PET.iSensImageBatch2D(AN, nsubs, psf)
            if img is None:
                img = np.ones([batch_size, matrixSize[0] * matrixSize[1]], dtype="float32")
            if batch_size == 1:
                if iSensImg.ndim == 2:
                    iSensImg = iSensImg[None].astype("float32")
                if img.ndim == 1:
                    img = img[None].astype("float32")

        if mrImg is not None:
            mrImg = Crop(mrImg)

        imgt = toTorch(img)

        # If mrImg was already at the post-crop PET size and got double-cropped,
        # resize it back to match imgt's spatial dims.
        if mrImg is not None and mrImg.shape[2:] != imgt.shape[2:]:
            mrImg = torch.nn.functional.interpolate(
                mrImg, size=imgt.shape[2:], mode="bilinear", align_corners=False
            )

        for i in range(niters):
            for s in range(nsubs):
                if self.is3d:
                    img_em  = img * PET.forwardDivideBackwardBatch3D(
                        img, prompts, RS, AN, nsubs, s, psf
                    ) * iSensImg[:, s, :, :, :]
                    img_emt  = toTorch(img_em)
                    img_regt = zeroNanInfs(self.regularize(imgt, mrImg))
                    S        = toTorch(iSensImg[:, s, :, :, :])
                else:
                    img_em  = img * PET.forwardDivideBackwardBatch2D(
                        img, prompts, RS, AN, nsubs, s, tof, psf
                    ) * iSensImg[:, s, :]
                    img_emt  = toTorch(img_em)
                    img_regt = zeroNanInfs(self.regularize(imgt, mrImg))
                    S        = toTorch(iSensImg[:, s, :])

                # FBSEM closed-form combination (Eq. 6 in Mehranian et al. 2020):
                #   derived from minimising the Bregman divergence between
                #   the EM update and the regularized estimate.
                discriminant = (1.0 - self.gamma * S * img_regt) ** 2 + 4.0 * self.gamma * S * img_emt
                imgt = 2.0 * img_emt / (
                    (1.0 - self.gamma * S * img_regt) + torch.sqrt(discriminant)
                )
                img = toNumpy(imgt)
                del img_em, img_emt, img_regt, S

        del iSensImg, prompts, RS, AN, PET, img
        return imgt


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _try_get_tb_writer(log_dir: str | None):
    """Return a SummaryWriter if tensorboard is installed, else None."""
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logging to: %s", log_dir)
        return writer
    except ImportError:
        logger.warning(
            "tensorboard not installed — run `pip install tensorboard` to enable it. "
            "Training will continue without TensorBoard."
        )
        return None


def _tb_log_images(writer, tag: str, images: torch.Tensor, step: int,
                   max_images: int = 4) -> None:
    """Write a grid of images to TensorBoard (no-op if writer is None)."""
    if writer is None:
        return
    n = min(images.shape[0], max_images)
    imgs = images[:n].clamp(min=0)
    imgs = imgs / (imgs.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    from torchvision.utils import make_grid
    writer.add_image(tag, make_grid(imgs, nrow=n), step)


def Trainer(PET, model, opts, train_loader, valid_loader=None) -> dict:
    """Train FBSEMnet_v3 and return a results dict including the checkpoint path.

    Parameters
    ----------
    PET          : BuildGeometry_v4 — scanner geometry (used inside model.forward)
    model        : FBSEMnet_v3
    opts         : dotstruct (or any object) with training hyperparameters.
                   See train_2d.yaml for a full list of supported fields.
    train_loader : DataLoader for training split
    valid_loader : DataLoader for validation split (optional)

    Returns
    -------
    dict with keys:
        train_losses, valid_losses, valid_psnrs, valid_ssims, gamma,
        final_checkpoint (str path of the last saved .pth file)
    """
    from models.deeplib import dotstruct, setOptions, imShowBatch, crop
    import torch.optim as optim

    # ── Default hyperparameters (overridden by opts) ──────────────────────────
    g = dotstruct()
    g.psf_cm             = 0.15
    g.niters             = 10
    g.nsubs              = 6
    g.lr                 = 0.001
    g.epochs             = 100
    g.in_channels        = 1
    g.save_dir           = os.getcwd()
    g.model_name         = "fbsem-pm-01"
    g.display            = False
    g.disp_figsize       = (20, 10)
    g.save_from_epoch    = None
    g.crop_factor        = 0.3
    g.do_validation      = True
    g.device             = "cpu"
    g.mr_scale           = 5
    g.loss_type          = "mse"           # 'mse' or 'weighted_mse'
    g.foreground_weight  = 10.0
    g.lr_scheduler       = "plateau"       # 'plateau' or None
    g.lr_patience        = 5
    g.lr_factor          = 0.5
    g.lr_min             = 1e-6
    g.early_stop_patience = 15             # None to disable
    g.tensorboard_dir    = None            # set to a path to enable TensorBoard
    g.tb_image_interval  = 5              # log sample images every N epochs

    g = setOptions(g, opts)
    os.makedirs(g.save_dir, exist_ok=True)

    tb = _try_get_tb_writer(getattr(g, "tensorboard_dir", None))

    # ── Loss and optimiser ────────────────────────────────────────────────────
    if g.loss_type == "weighted_mse":
        loss_fn = WeightedMSELoss(g.foreground_weight)
    else:
        loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=g.lr)

    scheduler = None
    if g.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=g.lr_factor, patience=g.lr_patience, min_lr=g.lr_min,
        )

    use_cuda = str(g.device).startswith("cuda") and torch.cuda.is_available()

    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype("float32")

    train_losses  : list[float] = []
    valid_losses  : list[float] = []
    valid_ssims   : list[float] = []
    valid_psnrs   : list[float] = []
    gamma_history : list[float] = []
    best_valid_loss  = float("inf")
    early_stop_count = 0
    final_checkpoint : str | None = None

    model.train()

    for e in range(g.epochs):
        running_loss = 0.0

        for sinoLD, imgHD, AN, _, _, _, mrImg, _, _, index in train_loader:
            AN     = _to_numpy(AN)
            sinoLD = _to_numpy(sinoLD)
            imgHD  = imgHD.to(g.device, dtype=torch.float32).unsqueeze(1)

            mr_input = None
            if g.in_channels == 2:
                mr_input = (g.mr_scale * mrImg / mrImg.max()).to(
                    g.device, dtype=torch.float32
                ).unsqueeze(1)

            optimizer.zero_grad()
            img  = model.forward(
                PET, prompts=sinoLD, AN=AN, mrImg=mr_input,
                niters=g.niters, nsubs=g.nsubs,
                psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor,
            )
            loss = loss_fn(img, imgHD)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Keep gamma positive; if it goes negative or NaN, reset it softly
            if torch.isnan(model.gamma).item() or model.gamma.item() < 0:
                logger.warning("gamma became invalid (%.4f); resetting to 0.01.", model.gamma.item())
                model.gamma.data.fill_(0.01)

            gam = model.gamma.item()
            gamma_history.append(gam)

            if g.display:
                imShowBatch(crop(_to_numpy(img).squeeze(), 0.3), figsize=g.disp_figsize)
                logger.debug("gamma: %.6f", gam)

            del sinoLD, AN, mrImg, index

        # ── End of epoch ─────────────────────────────────────────────────────
        train_losses.append(running_loss / len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  γ=%.6f  lr=%.2e",
            e + 1, g.epochs, train_losses[-1], model.gamma.item(), current_lr,
        )

        if tb is not None:
            tb.add_scalar("Loss/train", train_losses[-1], e)
            tb.add_scalar("Gamma", model.gamma.item(), e)
            tb.add_scalar("LR", current_lr, e)

        # ── Validation ───────────────────────────────────────────────────────
        if g.do_validation and valid_loader is not None:
            valid_loss   = 0.0
            ssim_sum     = 0.0
            psnr_sum     = 0.0
            ssim_tracked = False

            model.eval()
            with torch.no_grad():
                for sinoLD, imgHD, AN, _, _, _, mrImg, _, _, index in valid_loader:
                    AN     = _to_numpy(AN)
                    sinoLD = _to_numpy(sinoLD)
                    imgHD  = imgHD.to(g.device, dtype=torch.float32).unsqueeze(1)

                    mr_input = None
                    if g.in_channels == 2:
                        mr_input = (g.mr_scale * mrImg / mrImg.max()).to(
                            g.device, dtype=torch.float32
                        ).unsqueeze(1)

                    img = model.forward(
                        PET, prompts=sinoLD, AN=AN, mrImg=mr_input,
                        niters=g.niters, nsubs=g.nsubs,
                        psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor,
                    )
                    valid_loss += loss_fn(img, imgHD).item()

                    ssim_val, psnr_val = _compute_ssim_psnr(img, imgHD)
                    if ssim_val is not None:
                        ssim_sum    += ssim_val
                        ssim_tracked = True
                    psnr_sum += psnr_val

            model.train()
            n_val = len(valid_loader)
            valid_losses.append(valid_loss / n_val)
            valid_psnrs.append(psnr_sum / n_val)
            if ssim_tracked:
                valid_ssims.append(ssim_sum / n_val)

            ssim_str = f"  ssim={valid_ssims[-1]:.4f}" if ssim_tracked else ""
            logger.info(
                "           val_loss=%.4f  psnr=%.2f dB%s",
                valid_losses[-1], valid_psnrs[-1], ssim_str,
            )

            if tb is not None:
                tb.add_scalar("Loss/valid", valid_losses[-1], e)
                tb.add_scalar("Metrics/PSNR", valid_psnrs[-1], e)
                if ssim_tracked:
                    tb.add_scalar("Metrics/SSIM", valid_ssims[-1], e)

        # ── LR scheduler ─────────────────────────────────────────────────────
        monitor = (valid_losses[-1] if (g.do_validation and valid_losses)
                   else train_losses[-1])
        if scheduler is not None:
            scheduler.step(monitor)

        # ── Early stopping ────────────────────────────────────────────────────
        stop_early = False
        if g.early_stop_patience is not None and g.do_validation and valid_losses:
            if monitor < best_valid_loss:
                best_valid_loss  = monitor
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= g.early_stop_patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        e + 1, g.early_stop_patience,
                    )
                    stop_early = True

        # ── TensorBoard image grid ────────────────────────────────────────────
        tb_interval = getattr(g, "tb_image_interval", 5)
        if tb is not None and (e % tb_interval == 0 or stop_early or e == g.epochs - 1):
            model.eval()
            with torch.no_grad():
                try:
                    sample_sino, sample_hd, sample_an, *_, sample_mr, _, _, _ = next(
                        iter(valid_loader if valid_loader is not None else train_loader)
                    )
                    sample_an   = _to_numpy(sample_an)
                    sample_sino = _to_numpy(sample_sino)
                    sample_hd   = sample_hd.to(g.device, dtype=torch.float32).unsqueeze(1)
                    mr_in = None
                    if g.in_channels == 2:
                        mr_in = (g.mr_scale * sample_mr / sample_mr.max()).to(
                            g.device, dtype=torch.float32
                        ).unsqueeze(1)
                    pred = model.forward(
                        PET, prompts=sample_sino, AN=sample_an, mrImg=mr_in,
                        niters=g.niters, nsubs=g.nsubs,
                        psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor,
                    )
                    _tb_log_images(tb, "Images/prediction", pred, e)
                    _tb_log_images(tb, "Images/target",     sample_hd, e)
                except Exception as exc:
                    logger.debug("TensorBoard image logging skipped: %s", exc)
            model.train()

        # ── Checkpoint ───────────────────────────────────────────────────────
        is_last = stop_early or (e == g.epochs - 1)
        if ((g.save_from_epoch is not None) and (g.save_from_epoch <= e)) or is_last:
            g.state_dict    = model.state_dict()
            g.train_losses  = train_losses
            g.valid_losses  = valid_losses
            g.valid_psnrs   = valid_psnrs
            g.valid_ssims   = valid_ssims
            g.training_idx  = train_loader.sampler.indices
            g.gamma         = gamma_history

            save_path        = os.path.join(g.save_dir, f"{g.model_name}-epo-{e}.pth")
            torch.save(g.as_dict(), save_path)
            final_checkpoint = save_path
            logger.info("Checkpoint saved: %s", save_path)

        if use_cuda:
            torch.cuda.empty_cache()

        if stop_early:
            break

    if tb is not None:
        tb.close()
        logger.info("TensorBoard writer closed.")

    return {
        "train_losses":      train_losses,
        "valid_losses":      valid_losses,
        "valid_psnrs":       valid_psnrs,
        "valid_ssims":       valid_ssims,
        "gamma":             gamma_history,
        "final_checkpoint":  final_checkpoint,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def fbsemInference(
    dl_model_flname: str,
    PET,
    sinoLD: torch.Tensor,
    AN: torch.Tensor,
    mrImg: torch.Tensor | None,
    niters: int = None,
    nsubs: int = None,
    device: str = "cpu",
) -> np.ndarray:
    """Load a checkpoint and run a forward pass (no gradients).

    Parameters
    ----------
    dl_model_flname : path to a .pth checkpoint saved by Trainer
    PET             : BuildGeometry_v4 geometry object
    sinoLD          : low-dose sinogram tensor, shape (B, R, A)
    AN              : attenuation × normalisation tensor
    mrImg           : MR image tensor (or None for PET-only mode)
    niters, nsubs   : override the reconstruction iterations/subsets stored
                      in the checkpoint (None = use checkpoint values)
    device          : torch device string

    Returns
    -------
    np.ndarray — reconstructed image(s), shape (B, H, W) or (H, W) if B=1
    """
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype("float32")

    logger.info("Loading checkpoint: %s", dl_model_flname)
    g = torch.load(dl_model_flname, map_location=torch.device(device))

    reg_cnn_model = g.get("reg_cnn_model", g.get("reg_ccn_model", "resUnit"))
    is3d          = g.get("is3d", g.get("is_3d", False))

    model = FBSEMnet_v3(
        g["depth"], g["num_kernels"], g["kernel_size"],
        g["in_channels"], is3d, reg_cnn_model,
    ).to(device)
    model.load_state_dict(g["state_dict"])
    model.eval()

    AN     = _to_numpy(AN)
    sinoLD = _to_numpy(sinoLD)

    mr_input = None
    if g["in_channels"] == 2 and mrImg is not None:
        mr_input = (g["mr_scale"] * mrImg / mrImg.max()).to(
            device, dtype=torch.float32
        ).unsqueeze(1)

    niters = niters or g["niters"]
    nsubs  = nsubs  or g["nsubs"]

    with torch.no_grad():
        img = model.forward(
            PET, prompts=sinoLD, AN=AN, mrImg=mr_input,
            niters=niters, nsubs=nsubs,
            psf=g["psf_cm"], device=device, crop_factor=g["crop_factor"],
        )

    result = _to_numpy(img).squeeze()
    logger.info("Inference complete; output shape: %s", result.shape)
    return result
