"""
Data loading, dataset building, and utility functions for FBSEM PET reconstruction.

Public API:
  buildBrainPhantomDataset  — generate .npy training files from BrainWeb phantoms
  DatasetPetMr_v2           — PyTorch Dataset backed by those .npy files
  PETMrDataset              — convenience wrapper returning train/valid/test loaders
  noise_realizations        — generate multiple noise realisations of one phantom slice
  noise_levels_realizations — sweep count levels for one phantom slice
  dotstruct                 — dict-like attribute-access object (used for config)
  setOptions                — merge two dotstruct objects
  toNumpy                   — detach a tensor and convert to float32 numpy array
  crop / uncrop             — spatial crop / zero-pad utilities (tensor or numpy)
  gaussFilterBatch          — apply a Gaussian PSF to a batch of images
  zeroNanInfs               — replace NaN/Inf with 0 (tensor or numpy)
  imShowBatch               — quick matplotlib visualisation helper
"""

import gc
import logging
import os

import numpy as np
from numpy import ceil, load
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility structures
# ─────────────────────────────────────────────────────────────────────────────

class dotstruct:
    """Thin wrapper giving attribute-style access to a plain dict.

    Useful for passing config dicts through APIs that expect named attributes.

    Examples
    --------
    >>> g = dotstruct()
    >>> g.lr = 0.001
    >>> g["lr"]
    0.001
    >>> g.as_dict()
    {'lr': 0.001}
    """

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getitem__(self, name):
        return self.__dict__[name]

    def __contains__(self, name):
        return name in self.__dict__

    def get(self, name, default=None):
        return self.__dict__.get(name, default)

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"dotstruct({items})"


def setOptions(arg: dotstruct, opt, transfer: bool = True) -> dotstruct:
    """Copy matching fields from *opt* into *arg*; optionally transfer new fields too.

    Parameters
    ----------
    arg      : destination dotstruct (modified in-place and returned)
    opt      : source object — any object with a ``__dict__``
    transfer : when True, fields present in *opt* but not in *arg* are also copied
    """
    opt_dict = opt.__dict__ if hasattr(opt, "__dict__") else {}
    for key in arg.__dict__:
        if key in opt_dict:
            arg.__dict__[key] = opt_dict[key]
    if transfer:
        for key, val in opt_dict.items():
            if key not in arg.__dict__:
                arg.__dict__[key] = val
    return arg


# ─────────────────────────────────────────────────────────────────────────────
# Tensor / array helpers
# ─────────────────────────────────────────────────────────────────────────────

def toNumpy(x) -> np.ndarray:
    """Detach a PyTorch tensor and return a float32 numpy array."""
    return x.detach().cpu().numpy().astype("float32")


def zeroNanInfs(x):
    """Replace NaN and Inf with 0 in-place.  Works for tensors and numpy arrays."""
    import torch
    if torch.is_tensor(x):
        # Use scalar 0 rather than Tensor([0]) to avoid device mismatches
        x.data[torch.isnan(x)] = 0
        x.data[torch.isinf(x)] = 0
    else:
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
    return x


def crop(img, crop_factor: float = 0.0, is3d: bool = False):
    """Crop the spatial (W, H) dimensions of *img* by *crop_factor* (fraction of each side).

    Handles tensors (B, C, W, H[, D]) and numpy arrays of various shapes.
    """
    if crop_factor == 0:
        return img

    import torch
    round_int = lambda x: int(ceil(x / 2.0) * 2)

    if torch.is_tensor(img):
        i = round_int(img.shape[2] * crop_factor) // 2
        j = round_int(img.shape[3] * crop_factor) // 2
        return img[:, :, i: img.shape[2] - i, j: img.shape[3] - j]

    # numpy
    if img.ndim == 4 or (img.ndim == 3 and not is3d):
        i = round_int(img.shape[1] * crop_factor) // 2
        j = round_int(img.shape[2] * crop_factor) // 2
        return img[:, i: img.shape[1] - i, j: img.shape[2] - j]
    # 2-D or 3-D volume (W, H[, D])
    i = round_int(img.shape[0] * crop_factor) // 2
    j = round_int(img.shape[1] * crop_factor) // 2
    return img[i: img.shape[0] - i, j: img.shape[1] - j]


def uncrop(img, W: int, H: int = None, is3d: bool = False):
    """Zero-pad the spatial dimensions of *img* back to (W, H).

    Handles tensors (B, C, W0, H0[, D]) and numpy arrays of various shapes.
    """
    import torch

    if H is None:
        H = W

    if torch.is_tensor(img):
        if img.shape[2] == W and img.shape[3] == H:
            return img
        i = (W - img.shape[2]) // 2
        j = (H - img.shape[3]) // 2
        dims = [img.shape[0], img.shape[1], W, H]
        if img.dim() == 5:
            dims.append(img.shape[4])
        out = torch.zeros(dims, dtype=img.dtype, device=img.device)
        out[:, :, i: W - i, j: H - j] = img
        return out

    # numpy
    if img.ndim == 4 and (img.shape[1] != W or img.shape[2] != H):          # (B, W, H, D)
        i, j = (W - img.shape[1]) // 2, (H - img.shape[2]) // 2
        out  = np.zeros((img.shape[0], W, H, img.shape[3]), dtype=img.dtype)
        out[:, i: W - i, j: H - j, :] = img
        return out
    if (img.ndim == 3 and not is3d) and (img.shape[1] != W or img.shape[2] != H):  # (B, W, H)
        i, j = (W - img.shape[1]) // 2, (H - img.shape[2]) // 2
        out  = np.zeros((img.shape[0], W, H), dtype=img.dtype)
        out[:, i: W - i, j: H - j] = img
        return out
    if (img.ndim == 3 and is3d) and (img.shape[0] != W or img.shape[1] != H):     # (W, H, D)
        i, j = (W - img.shape[0]) // 2, (H - img.shape[1]) // 2
        out  = np.zeros((W, H, img.shape[2]), dtype=img.dtype)
        out[i: W - i, j: H - j, :] = img
        return out
    if img.ndim == 2 and (img.shape[0] != W or img.shape[1] != H):                 # (W, H)
        i, j = (W - img.shape[0]) // 2, (H - img.shape[1]) // 2
        out  = np.zeros((W, H), dtype=img.dtype)
        out[i: W - i, j: H - j] = img
        return out
    return img


def gaussFilterBatch(img, voxelSizeCm, fwhm, is3d: bool = True):
    """Apply a Gaussian PSF to a batch of images.

    Parameters
    ----------
    img         : np.ndarray — (B, W, H[, D]) or (W, H[, D])
    voxelSizeCm : array-like — voxel size in cm along each spatial axis
    fwhm        : scalar or array-like — FWHM in cm (same for all axes, or per-axis)
    is3d        : True for 3-D batches
    """
    from scipy import ndimage

    fwhm = np.asarray(fwhm, dtype=float)
    if np.all(fwhm == 0):
        return img

    voxelSizeCm = np.asarray(voxelSizeCm)
    if not is3d:
        voxelSizeCm = voxelSizeCm[:2]

    # Expand scalar FWHM to match spatial dimensions
    if fwhm.ndim == 0:
        fwhm = fwhm * np.ones(3 if is3d else 2)

    sigma = fwhm / voxelSizeCm / np.sqrt(8.0 * np.log(2))

    def _filter(x):
        return ndimage.gaussian_filter(x, sigma)

    if is3d:
        if img.ndim == 3:
            return _filter(img)
        out = np.zeros_like(img)
        for b in range(img.shape[0]):
            out[b] = _filter(img[b])
        return out
    else:
        if img.ndim == 2:
            return _filter(img)
        out = np.zeros_like(img)
        for b in range(img.shape[0]):
            out[b] = _filter(img[b])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def imShowBatch(
    x,
    batch_size=None,
    is3d: bool = False,
    slice_num=None,
    vmax=None,
    cmap=None,
    title=None,
    coronal: bool = False,
    figsize=(20, 10),
    caption=None,
    rotation: int = 0,
):
    """Quick matplotlib display of a batch of 2-D or 3-D images."""
    from matplotlib import pyplot as plt

    if batch_size is None:
        batch_size = x.shape[0]
    if cmap is None:
        cmap = ["gist_yarg"] * batch_size
    if not isinstance(cmap, list):
        cmap = [cmap] * batch_size
    if vmax is None:
        vmax = [None] * batch_size
    elif np.isscalar(vmax):
        vmax = [vmax] * batch_size

    fig, ax = plt.subplots(1, batch_size, sharex=True, sharey=True, figsize=figsize)
    if batch_size == 1:
        ax = [ax]

    for i in range(batch_size):
        if is3d:
            if coronal:
                sl = slice_num if slice_num is not None else x.shape[1] // 2
                img = np.rot90(x[i, sl, :, :], 1)
            else:
                sl = slice_num if slice_num is not None else x.shape[3] // 2
                img = x[i, :, :, sl]
        else:
            img = x[i, :, :]

        ax[i].imshow(img, vmin=0, vmax=vmax[i], cmap=cmap[i])
        ax[i].axis("off")
        if caption is not None:
            ax[i].set_title(caption[i], fontsize=22, va="bottom", rotation=rotation)

    if title is not None:
        fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.pause(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DatasetPetMr_v2(Dataset):
    """PyTorch Dataset backed by .npy files produced by :func:`buildBrainPhantomDataset`.

    Each file is a dict with keys: sinoLD, imgHD, AN, RS (optional), imgLD,
    imgLD_psf, imgGT, mrImg, counts, plus simulation metadata.

    Parameters
    ----------
    filename     : ['save_dir/', 'prefix'] — base path; files are prefix+N+'.npy'
    num_train    : total number of dataset files
    augment      : enable online augmentation (left-right flip + MR jitter).
                   Should be True only for the training split.
    crop_factor  : fraction of radial sinogram bins / image pixels to discard
                   from each side (0 = no crop).
    """

    def __init__(
        self,
        filename,
        num_train: int,
        transform=None,
        target_transform=None,
        is3d: bool = False,
        imgLD_flname=None,
        crop_factor: float = 0.0,
        allow_pickle: bool = True,
        augment: bool = False,
    ):
        self.transform        = transform
        self.target_transform = target_transform
        self.is3d             = is3d
        self.filename         = filename
        self.num_train        = num_train
        self.imgLD_flname     = imgLD_flname
        self.crop_factor      = crop_factor
        self.allow_pickle     = allow_pickle
        self.augment          = augment

    # ── internal crop helpers ─────────────────────────────────────────────────

    def _crop_sino(self, sino):
        if self.crop_factor == 0:
            return sino
        i = int(ceil(sino.shape[0] * self.crop_factor / 2.0) * 2) // 2
        return sino[i: sino.shape[0] - i]

    def _crop_img(self, img):
        if self.crop_factor == 0:
            return img
        i = int(ceil(img.shape[0] * self.crop_factor / 2.0) * 2) // 2
        return img[i: img.shape[0] - i, i: img.shape[1] - i]

    def __len__(self) -> int:
        return self.num_train

    def __getitem__(self, index: int):
        path = self.filename[0] + self.filename[1] + str(index) + ".npy"
        dset = load(path, allow_pickle=self.allow_pickle).item()

        sinoLD = dset["sinoLD"]
        AN     = dset["AN"]
        imgHD  = self._crop_img(dset["imgHD"])
        mrImg  = self._crop_img(dset["mrImg"])
        counts = dset["counts"]

        RS = dset["RS"] if ("RS" in dset and not isinstance(dset["RS"], list)) else 0

        imgGT = (
            self._crop_img(dset["imgGT"])
            if ("imgGT" in dset and not isinstance(dset["imgGT"], list))
            else 0
        )
        imgLD = (
            self._crop_img(dset["imgLD"])
            if ("imgLD" in dset and not isinstance(dset["imgLD"], list))
            else 0
        )
        imgLD_psf = (
            self._crop_img(dset["imgLD_psf"])
            if ("imgLD_psf" in dset and not isinstance(dset["imgLD_psf"], list))
            else 0
        )

        # ── Online augmentation (training split only) ────────────────────────
        if self.augment:
            # Left-right flip (50 % probability).
            # In 2-D PET, flipping image columns corresponds to reversing sinogram
            # radial bins (axis 0), preserving the forward-model consistency.
            if np.random.rand() > 0.5:
                sinoLD = sinoLD[::-1, :].copy()
                AN     = AN[::-1, :].copy()
                imgHD  = imgHD[:, ::-1].copy()
                mrImg  = mrImg[:, ::-1].copy()
                if not np.isscalar(RS):
                    RS = RS[::-1, :].copy()
                if not np.isscalar(imgLD):
                    imgLD = imgLD[:, ::-1].copy()
                if not np.isscalar(imgLD_psf):
                    imgLD_psf = imgLD_psf[:, ::-1].copy()
                if not np.isscalar(imgGT):
                    imgGT = imgGT[:, ::-1].copy()

            # MR intensity jitter: uniform scale in [0.8, 1.2].
            mr_scale = 0.8 + 0.4 * np.random.rand()
            mrImg    = (mrImg * mr_scale).astype(mrImg.dtype)

        # ── External transforms ───────────────────────────────────────────────
        if self.transform is not None:
            sinoLD = self.transform(sinoLD)
            AN     = self.transform(AN)
            if not np.isscalar(RS):
                RS = self.transform(RS)

        if self.target_transform is not None:
            imgHD = self.target_transform(imgHD)
            mrImg = self.target_transform(mrImg)
            if not np.isscalar(imgLD):
                imgLD = self.target_transform(imgLD)
            if not np.isscalar(imgLD_psf):
                imgLD_psf = self.target_transform(imgLD_psf)
            if not np.isscalar(imgGT):
                imgGT = self.target_transform(imgGT)

        return sinoLD, imgHD, AN, RS, imgLD, imgLD_psf, mrImg, counts, imgGT, index


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split(
    train_dset,
    eval_dset,
    num_train: int,
    batch_size: int,
    test_size: float,
    valid_size: float = 0.0,
    num_workers: int = 0,
    shuffle: bool = True,
):
    """Split indices into train / valid / test and return DataLoaders.

    ``train_dset`` is used for the training loader (may have augmentation enabled).
    ``eval_dset``  is used for validation and test loaders (augmentation off).

    Returns
    -------
    train_loader, test_loader, valid_loader (valid_loader is None if valid_size=0)
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler

    indices = list(range(num_train))
    if shuffle:
        np.random.shuffle(indices)

    n_test  = int(np.floor(num_train * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    valid_loader = None
    if valid_size > 0:
        if shuffle:
            np.random.shuffle(train_idx)
        n_valid = int(np.floor(len(train_idx) * valid_size))
        valid_idx, train_idx = train_idx[:n_valid], train_idx[n_valid:]
        valid_loader = DataLoader(
            eval_dset, batch_size=batch_size,
            sampler=SubsetRandomSampler(valid_idx),
            num_workers=num_workers, pin_memory=False,
        )

    train_loader = DataLoader(
        train_dset, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        eval_dset, batch_size=batch_size,
        sampler=SubsetRandomSampler(test_idx),
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader, valid_loader


def PETMrDataset(
    filename,
    num_train: int,
    batch_size: int,
    test_size: float,
    valid_size: float = 0.0,
    num_workers: int = 0,
    transform=None,
    target_transform=None,
    is3d: bool = False,
    imgLD_flname=None,
    shuffle: bool = True,
    crop_factor: float = 0.0,
    augment: bool = False,
):
    """Create train / validation / test DataLoaders from a .npy dataset directory.

    The training DataLoader uses online augmentation when ``augment=True``.
    Validation and test loaders never augment.

    Returns
    -------
    train_loader, valid_loader, test_loader
    """
    train_dset = DatasetPetMr_v2(
        filename, num_train, transform, target_transform,
        is3d, imgLD_flname, crop_factor, augment=augment,
    )
    eval_dset = DatasetPetMr_v2(
        filename, num_train, transform, target_transform,
        is3d, imgLD_flname, crop_factor, augment=False,
    )
    train_loader, test_loader, valid_loader = train_test_split(
        train_dset, eval_dset, num_train, batch_size,
        test_size, valid_size, num_workers, shuffle,
    )
    return train_loader, valid_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def buildBrainPhantomDataset(
    PET,
    save_training_dir: str,
    phanPath: str,
    phanType: str = "brainweb",
    phanNumber=None,
    is3d: bool = True,
    num_rand_rotations: int = 1,
    rot_angle_degrees: float = 15.0,
    psf_hd: float = 0.25,
    psf_ld: float = 0.40,
    niter_hd: int = 15,
    niter_ld: int = 10,
    nsubs_hd: int = 14,
    nsubs_ld: int = 14,
    counts_hd: float = 1e10,
    count_ld_window_3d=(90e6, 120e6),
    count_ld_window_2d=(1e6, 10e6),
    slices_2d=None,
    pet_lesion: bool = True,
    t1_lesion: bool = True,
    num_lesions: int = 15,
    lesion_size_mm=(2, 8),
    hot_cold_ratio: float = 0.6,
) -> None:
    """Generate .npy training files from BrainWeb digital brain phantoms.

    For each phantom and each random in-plane rotation:
    - Simulates a high-dose (HD) and a low-dose (LD) sinogram.
    - Reconstructs both with OSEM to produce the target (HD) and network
      input (LD, LD+PSF) images.
    - Saves all arrays as a single .npy dict for fast DataLoader access.

    Parameters
    ----------
    PET                  : BuildGeometry_v4 — must already have system matrix loaded
    save_training_dir    : directory where data-N.npy files will be written
    phanPath             : directory containing BrainWeb subject_XX.raws files
                           (downloaded automatically if missing)
    phanType             : phantom library — only 'brainweb' is currently supported
    phanNumber           : scalar, list, or np.ndarray of BrainWeb subject indices
                           (None = all 20 subjects)
    is3d                 : True for 3-D volumes, False for 2-D slice batches
    num_rand_rotations   : random in-plane rotations applied per phantom
    rot_angle_degrees    : max rotation magnitude (±degrees)
    psf_hd / psf_ld      : PSF FWHM (cm) for HD / LD OSEM reconstructions
    niter_hd / niter_ld  : OSEM iterations for HD / LD
    nsubs_hd / nsubs_ld  : OSEM subsets for HD / LD
    counts_hd            : total coincidences for the HD sinogram
    count_ld_window_2d/3d: [min, max] range for randomly sampled LD count level
    slices_2d            : axial slice indices to extract (2-D mode only).
                           Default: every other slice from index 65 to 84.
    pet_lesion / t1_lesion : add random spherical lesions to PET / T1 images
    num_lesions          : number of lesions per phantom
    lesion_size_mm       : [min, max] lesion diameter in mm
    hot_cold_ratio       : fraction of lesions that are hot (vs cold)

    Example
    -------
    >>> from geometry.BuildGeometry_v4 import BuildGeometry_v4
    >>> from models.deeplib import buildBrainPhantomDataset
    >>> import numpy as np
    >>> PET = BuildGeometry_v4('mmr', 0.5)
    >>> PET.loadSystemMatrix('/data/system_matrix', is3d=False)
    >>> buildBrainPhantomDataset(
    ...     PET, save_training_dir='/data/output/', phanPath='/data/brainweb/',
    ...     phanNumber=np.arange(5), is3d=False,
    ... )
    """
    from phantoms.phantomlib import imRotation

    bar = PET.engine.bar

    if phanType.lower() != "brainweb":
        raise ValueError(f"Unknown phanType: {phanType!r}. Only 'brainweb' is supported.")

    from phantoms.brainweb import PETbrainWebPhantom

    if phanNumber is None:
        phanNumber = np.arange(20)
    elif np.isscalar(phanNumber):
        phanNumber = np.array([phanNumber])
    else:
        phanNumber = np.asarray(phanNumber)

    if PET.is3d != is3d:
        mode = "3D" if is3d else "2D"
        raise ValueError(f"PET object geometry is not {mode}.")

    os.makedirs(save_training_dir, exist_ok=True)

    if slices_2d is None:
        slices_2d = np.arange(65, 85, 2)  # 10 brain-containing axial slices
    num_slices = len(slices_2d)

    voxel_size = np.array(PET.image.voxelSizeCm) * 10  # cm → mm
    image_size = PET.image.matrixSize

    # Metadata stored once per file alongside the image arrays
    meta = dict(
        psf_hd=psf_hd, psf_ld=psf_ld,
        niter_hd=niter_hd, nsubs_hd=nsubs_hd,
        niter_ld=niter_ld, nsubs_ld=nsubs_ld,
        num_lesions=num_lesions, lesion_size_mm=lesion_size_mm,
        hot_cold_ratio=hot_cold_ratio,
        rot_angle_degrees=rot_angle_degrees,
        counts_hd=counts_hd,
        count_ld_window_3d=count_ld_window_3d,
        count_ld_window_2d=count_ld_window_2d,
        pet_lesion=pet_lesion, t1_lesion=t1_lesion,
        phanType=phanType, phanPath=[],
    )

    file_idx = 0

    for i in phanNumber:
        logger.info("Processing phantom %d ...", i)
        img, mumap, t1, _ = PETbrainWebPhantom(
            phanPath, i, voxel_size, image_size,
            num_lesions, lesion_size_mm, pet_lesion, t1_lesion,
            False, hot_cold_ratio,
        )

        angles = 2 * rot_angle_degrees * np.random.rand(num_rand_rotations) - rot_angle_degrees
        angles[0] = 0.0  # always include the un-rotated phantom

        for j, angle in enumerate(angles):
            logger.info("  Phantom %d — rotation %d/%d (%.1f°) ...",
                        i, j + 1, num_rand_rotations, angle)

            if is3d:
                imgr    = np.clip(imRotation(img,   angle), 0, None)
                mumapr  = np.clip(imRotation(mumap, angle), 0, None)
                t1r     = np.clip(imRotation(t1,    angle), 0, None)

                counts_ld = (
                    count_ld_window_3d[0]
                    + (count_ld_window_3d[1] - count_ld_window_3d[0]) * np.random.rand()
                )
                logger.info("    Simulating 3-D sinograms (LD counts=%.2e) ...", counts_ld)
                prompts_hd, AF, NF, _ = PET.simulateSinogramData(imgr, mumap=mumapr,
                                                                  counts=counts_hd, psf=psf_hd)
                prompts_ld, *_        = PET.simulateSinogramData(imgr, AF=AF, NF=NF,
                                                                  counts=counts_ld, psf=psf_ld)
                AN_arr = AF * NF

                logger.info("    Reconstructing 3-D images ...")
                img_hd     = PET.OSEM3D(prompts_hd, AN=AN_arr, niter=niter_hd, nsubs=nsubs_hd, psf=psf_hd)
                img_ld     = PET.OSEM3D(prompts_ld, AN=AN_arr, niter=niter_ld,  nsubs=nsubs_ld,  psf=psf_hd)
                img_ld_psf = PET.OSEM3D(prompts_ld, AN=AN_arr, niter=niter_ld,  nsubs=nsubs_ld,  psf=psf_ld)

                dset = dict(meta,
                            sinoLD=prompts_ld, imgHD=img_hd, imgLD=img_ld,
                            imgLD_psf=img_ld_psf, AN=AN_arr,
                            imgGT=imgr, mrImg=t1r, counts=counts_ld,
                            RS=[])

                save_path = _next_save_path(save_training_dir, bar, file_idx)
                np.save(save_path, dset)
                file_idx += 1
                logger.info("    Saved: %s", save_path)

                del imgr, mumapr, t1r, prompts_hd, AF, NF, prompts_ld, AN_arr
                del img_hd, img_ld, img_ld_psf

            else:
                # Extract 2-D slices from the 3-D phantom
                imgr   = np.clip(np.transpose(imRotation(img[:, :, slices_2d],   angle), (2, 0, 1)), 0, None)
                mumapr = np.clip(np.transpose(imRotation(mumap[:, :, slices_2d], angle), (2, 0, 1)), 0, None)
                t1r    = np.clip(np.transpose(imRotation(t1[:, :, slices_2d],    angle), (2, 0, 1)), 0, None)

                counts_ld = (
                    count_ld_window_2d[0]
                    + (count_ld_window_2d[1] - count_ld_window_2d[0]) * np.random.rand()
                )
                logger.info("    Simulating 2-D sinograms (LD counts=%.2e) ...", counts_ld)
                prompts_hd, AF, NF, _ = PET.simulateSinogramData(imgr, mumap=mumapr,
                                                                  counts=counts_hd, psf=psf_hd)
                prompts_ld, *_        = PET.simulateSinogramData(imgr, AF=AF, NF=NF,
                                                                  counts=counts_ld, psf=psf_ld)
                AN_arr = AF * NF

                logger.info("    Reconstructing 2-D images ...")
                img_hd     = PET.OSEM2D(prompts_hd, AN=AN_arr, niter=niter_hd, nsubs=nsubs_hd, psf=psf_hd)
                img_ld     = PET.OSEM2D(prompts_ld, AN=AN_arr, niter=niter_ld,  nsubs=nsubs_ld,  psf=psf_hd)
                img_ld_psf = PET.OSEM2D(prompts_ld, AN=AN_arr, niter=niter_ld,  nsubs=nsubs_ld,  psf=psf_ld)

                logger.info("    Saving %d slices ...", num_slices)
                for k in range(num_slices):
                    dset = dict(meta,
                                sinoLD=prompts_ld[k], imgHD=img_hd[k], imgLD=img_ld[k],
                                imgLD_psf=img_ld_psf[k], AN=AN_arr[k],
                                imgGT=imgr[k], mrImg=t1r[k], counts=counts_ld,
                                RS=[])
                    save_path = _next_save_path(save_training_dir, bar, file_idx)
                    np.save(save_path, dset)
                    file_idx += 1

                logger.info("    Saved slices to %s (indices up to %d).", save_training_dir, file_idx - 1)
                del imgr, mumapr, t1r, prompts_hd, AF, NF, prompts_ld, AN_arr
                del img_hd, img_ld, img_ld_psf

        del img, mumap, t1
        gc.collect()

    logger.info("Dataset build complete. %d files written to %s.", file_idx, save_training_dir)


def _next_save_path(save_dir: str, bar: str, start_idx: int) -> str:
    """Return 'save_dir/data-N.npy' where N is the first index >= start_idx not on disk."""
    idx = start_idx
    while os.path.isfile(os.path.join(save_dir, f"data-{idx}.npy")):
        idx += 1
    return os.path.join(save_dir, f"data-{idx}.npy")


# ─────────────────────────────────────────────────────────────────────────────
# Noise-replication utilities
# ─────────────────────────────────────────────────────────────────────────────

def noise_realizations(
    img,
    mumap,
    t1,
    save_dir: str,
    pet_geometry_dir: str = None,
    PET=None,
    psf_hd: float = 0.25,
    psf_ld: float = 0.40,
    niter_hd: int = 15,
    niter_ld: int = 10,
    nsubs_hd: int = 14,
    nsubs_ld: int = 14,
    counts_hd: float = 1e10,
    counts_ld: float = None,
    num_noise_realizations: int = 10,
) -> None:
    """Generate multiple noise realisations of the same phantom slice.

    Parameters
    ----------
    img, mumap, t1  : 2-D arrays — PET activity, attenuation, and T1 maps for one slice
    save_dir        : output directory (data-nrN.npy files)
    PET             : BuildGeometry_v4 (created with defaults if not supplied)
    counts_ld       : fixed LD count level (random if None)

    Example
    -------
    >>> from geometry.BuildGeometry_v4 import BuildGeometry_v4
    >>> from phantoms.brainweb import PETbrainWebPhantom
    >>> from models.deeplib import noise_realizations
    >>> PET = BuildGeometry_v4('mmr', 0.5)
    >>> PET.loadSystemMatrix('/data/system_matrix', is3d=False)
    >>> img_vol, mumap_vol, t1_vol, _ = PETbrainWebPhantom('/data/brainweb/', subject=4,
    ...     voxel_size=np.array(PET.image.voxelSizeCm)*10, image_size=PET.image.matrixSize)
    >>> noise_realizations(img_vol[:, :, 74], mumap_vol[:, :, 74], t1_vol[:, :, 74],
    ...     save_dir='/data/output/', PET=PET, counts_ld=0.1e6)
    """
    os.makedirs(save_dir, exist_ok=True)

    if PET is None:
        from geometry.BuildGeometry_v4 import BuildGeometry_v4
        PET = BuildGeometry_v4("mmr", 0.5)
        PET.loadSystemMatrix(pet_geometry_dir)

    if counts_ld is None:
        counts_ld = 1e6 + 9e6 * np.random.rand()

    meta = dict(
        psf_hd=psf_hd, psf_ld=psf_ld,
        niter_hd=niter_hd, nsubs_hd=nsubs_hd,
        niter_ld=niter_ld, nsubs_ld=nsubs_ld,
        counts_hd=counts_hd, counts_ld=counts_ld,
    )

    prompts_hd, AF, NF, _ = PET.simulateSinogramData(img, mumap=mumap, counts=counts_hd, psf=psf_hd)
    AN_arr = AF * NF
    img_hd = PET.OSEM2D(prompts_hd, AN=AN_arr, niter=niter_hd, nsubs=nsubs_hd, psf=psf_hd)

    for j in range(num_noise_realizations):
        prompts_ld, *_ = PET.simulateSinogramData(img, AF=AF, NF=NF, counts=counts_ld, psf=psf_ld)
        img_ld     = PET.OSEM2D(prompts_ld, AN=AN_arr, niter=niter_ld, nsubs=nsubs_ld, psf=psf_hd)
        img_ld_psf = PET.OSEM2D(prompts_ld, AN=AN_arr, niter=niter_ld, nsubs=nsubs_ld, psf=psf_ld)

        dset = dict(meta,
                    sinoLD=prompts_ld, imgHD=img_hd, AN=AN_arr,
                    imgGT=img, mrImg=t1, counts=counts_ld,
                    imgLD=img_ld, imgLD_psf=img_ld_psf, RS=[])
        np.save(os.path.join(save_dir, f"data-nr{j}.npy"), dset)

    logger.info("Saved %d noise realisations to %s.", num_noise_realizations, save_dir)


def noise_levels_realizations(
    img,
    mumap,
    t1,
    save_dir: str,
    pet_geometry_dir: str = None,
    PET=None,
    psf_hd: float = 0.15,
    psf_ld: float = 0.15,
    counts_hd: float = 100e6,
    ld_count_window=(50e3, 1e6),
    num_count_levels: int = 5,
    num_noise_realizations: int = 1,
) -> None:
    """Sweep a range of count levels for one phantom slice.

    Produces files named ``data-cl{level}nr{realization}.npy``.

    Parameters
    ----------
    img, mumap, t1        : 2-D arrays — one axial slice
    ld_count_window       : [min, max] count range to sweep linearly
    num_count_levels      : number of equally-spaced count levels
    num_noise_realizations: noise draws per count level

    Example
    -------
    >>> noise_levels_realizations(
    ...     img[:, :, 50], mumap[:, :, 50], t1[:, :, 50],
    ...     save_dir='/data/output/', PET=PET,
    ...     ld_count_window=[1e5, 1e6], num_count_levels=5,
    ...     num_noise_realizations=10,
    ... )
    """
    os.makedirs(save_dir, exist_ok=True)

    if PET is None:
        from geometry.BuildGeometry_v4 import BuildGeometry_v4
        PET = BuildGeometry_v4("mmr", 0.5)
        PET.loadSystemMatrix(pet_geometry_dir)

    prompts_hd, AF, *_ = PET.simulateDataBatch2D(img, mumap, counts=counts_hd, psf=psf_hd)
    prompts_hd = prompts_hd[None, :, :]
    AF         = AF[None, :, :]
    img_hd     = PET.OsemBatch2D(prompts_hd, AN=AF, psf=psf_hd, nsubs=14, niter=10)

    counts_ld = np.linspace(ld_count_window[0], ld_count_window[1], num_count_levels)

    for j in range(num_noise_realizations):
        for i in range(num_count_levels):
            prompts_ld, *_ = PET.simulateDataBatch2D(img, mumap, counts=counts_ld[i], psf=psf_ld)
            prompts_ld = prompts_ld[None, :, :]
            img_ld     = PET.OsemBatch2D(prompts_ld, AN=AF, psf=psf_ld, nsubs=6, niter=10)

            dset = dict(
                psf=psf_ld,
                sinoLD=prompts_ld[0], imgHD=img_hd[0], AN=AF[0],
                imgGT=img, mrImg=t1, counts=counts_ld[i], imgOsem=img_ld[0],
            )
            np.save(os.path.join(save_dir, f"data-cl{i}nr{j}.npy"), dset)

    logger.info("Saved count-level sweep to %s.", save_dir)
