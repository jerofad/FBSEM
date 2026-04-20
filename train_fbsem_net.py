"""
Train a 2D (or 3D) FBSEM network for PET image reconstruction.

Usage
-----
    python train_fbsem_net.py --config configs/train_2d.yaml
    python train_fbsem_net.py --config configs/train_2d.yaml --device cuda --epochs 100

All parameters live in the YAML config file.  CLI flags override individual
values at runtime without modifying the file on disk.

Quick-start
-----------
1. Edit configs/train_2d.yaml — set system_matrix_path, training_data_dir, save_dir.
2. Run build_training_sets.py first (if no .npy files exist yet).
3. Run this script.
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for headless servers
from matplotlib import pyplot as plt

import torch
import yaml

from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import PETMrDataset, dotstruct, toNumpy, crop
from models.modellib import FBSEMnet_v3, Trainer, fbsemInference

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a timestamped format written to stdout."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _dict_to_dotstruct(d: dict) -> dotstruct:
    g = dotstruct()
    for k, v in d.items():
        setattr(g, k, v)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train FBSEM network for PET reconstruction")
    parser.add_argument("--config", default="configs/train_2d.yaml",
                        help="Path to YAML config file")
    # Per-run CLI overrides (any YAML key can also be supplied here)
    parser.add_argument("--device",             default=None,
                        help="PyTorch device string, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--epochs",             type=int,   default=None)
    parser.add_argument("--lr",                 type=float, default=None,
                        help="Initial learning rate")
    parser.add_argument("--save_dir",           default=None,
                        help="Directory to save checkpoints and figures")
    parser.add_argument("--model_name",         default=None,
                        help="Checkpoint base name, e.g. 'fbsem-pm-01'")
    parser.add_argument("--system_matrix_path", default=None,
                        help="Directory containing the scanner system-matrix files")
    parser.add_argument("--training_data_dir",  default=None,
                        help="Directory containing data-N.npy training files")
    parser.add_argument("--log_level",          default="INFO",
                        help="Logging verbosity: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--tensorboard_dir",    default=None,
                        help="Directory for TensorBoard event files (null = disabled)")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    logger.info("Loading config: %s", args.config)

    cfg = _load_config(args.config)

    # Apply CLI overrides
    for key in ("device", "epochs", "lr", "save_dir", "model_name",
                "system_matrix_path", "training_data_dir", "tensorboard_dir"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
            logger.debug("CLI override: %s = %s", key, val)

    # Fail early with a clear message if required paths were not configured
    _PLACEHOLDERS = {None, "/path/to/system_matrix",
                     "/path/to/training_data", "/path/to/output/"}
    for field in ("system_matrix_path", "training_data_dir", "save_dir"):
        if cfg.get(field) in _PLACEHOLDERS:
            raise ValueError(
                f"Config field '{field}' must be set to a real path.\n"
                f"  Edit configs/train_2d.yaml or pass --{field} on the command line."
            )

    g = _dict_to_dotstruct(cfg)
    logger.info(
        "Scanner: %s | 3-D: %s | Device: %s | Epochs: %d | LR: %.2e",
        g.scanner, g.is_3d, g.device, g.epochs, g.lr,
    )

    # ── PET geometry ──────────────────────────────────────────────────────────
    logger.info("Building PET geometry ...")
    PET = BuildGeometry_v4(g.scanner, g.radial_bin_crop_factor)
    PET.loadSystemMatrix(g.system_matrix_path, is3d=g.is_3d)

    # ── Data loaders ──────────────────────────────────────────────────────────
    logger.info("Building DataLoaders from: %s", g.training_data_dir)
    training_flname = [g.training_data_dir + os.sep, g.data_prefix]
    train_loader, valid_loader, test_loader = PETMrDataset(
        training_flname,
        num_train=g.num_train,
        is3d=g.is_3d,
        batch_size=g.batch_size,
        test_size=g.test_size,
        valid_size=g.valid_size,
        num_workers=g.num_workers,
        crop_factor=g.crop_factor,
        augment=getattr(g, "augment", False),
    )
    logger.info(
        "Splits — train batches: %d | valid batches: %s | test batches: %d",
        len(train_loader),
        len(valid_loader) if valid_loader is not None else "N/A",
        len(test_loader),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(
        "Initialising FBSEMnet_v3 — depth=%d, kernels=%d, in_channels=%d, 3D=%s",
        g.depth, g.num_kernels, g.in_channels, g.is_3d,
    )
    model = FBSEMnet_v3(
        g.depth, g.num_kernels, g.kernel_size,
        g.in_channels, g.is_3d, g.reg_cnn_model,
    ).to(g.device, dtype=torch.float32)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: {:,}".format(n_params))

    # ── Train ─────────────────────────────────────────────────────────────────
    # logger.info("Starting training ...")
    # results = Trainer(PET, model, g, train_loader, valid_loader)

    # final_ckpt = results.get("final_checkpoint")
    # if final_ckpt is None:
    #     logger.warning("No checkpoint was saved during training; skipping inference.")
    #     return

    # ── Test-set inference ────────────────────────────────────────────────────
    final_ckpt = "/home/jeremiah/Datasets/PicoPET/Model/fbsem-pm-01-epo-1.pth"
    logger.info("Running test-set inference from: %s", final_ckpt)
    sinoLD, imgHD, AN, _, imgLD, imgLD_psf, mrImg, _, imgGT, _ = next(iter(test_loader))

    img_fbsem = fbsemInference(
        final_ckpt, PET, sinoLD, AN, mrImg,
        niters=g.niters, nsubs=g.nsubs,
    )

    # Convert tensors to numpy for comparison / visualisation
    sinoLD    = toNumpy(sinoLD)
    imgHD     = toNumpy(imgHD)
    AN        = toNumpy(AN)
    imgLD     = toNumpy(imgLD)
    imgLD_psf = toNumpy(imgLD_psf)
    mrImg     = toNumpy(mrImg)
    imgGT     = toNumpy(imgGT)

    # Traditional MAPEM baseline for comparison
    logger.info("Running MAPEM baseline ...")
    img_mapem = PET.mrMAPEM2DBatch(
        sinoLD, AN, mrImg, beta=0.06,
        niters=g.niters, nsubs=g.nsubs, psf=g.psf_cm,
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    cf          = g.crop_factor
    num_batches = sinoLD.shape[0]
    fig, ax     = plt.subplots(
        num_batches, 5,
        figsize=tuple(g.disp_figsize),
        sharex=True, sharey=True,
    )
    if num_batches == 1:
        ax = ax[None]   # ensure 2-D indexing for single-batch case

    col_titles = ["MR", "HD (target)", "LD+PSF", "MAPEM", "FBSEM"]
    for i in range(num_batches):
        for j, (col_img, title) in enumerate(zip(
            [mrImg[i], imgHD[i], imgLD_psf[i], img_mapem[i], img_fbsem[i]],
            col_titles,
        )):
            ax[i, j].imshow(crop(col_img, cf), cmap="gist_gray_r")
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(title, fontsize=20)

    fig.subplots_adjust(hspace=0, wspace=0)
    out_path = os.path.join(g.save_dir, "test_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    logger.info("Saved comparison figure: %s", out_path)


if __name__ == "__main__":
    main()
