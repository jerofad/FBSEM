"""
Utility mixin for BuildGeometry_v4.

Provides shared helpers used by all other mixins:
  - Gaussian PSF filtering (1-D flattened image space)
  - OSEM angular subset generation (bit-reversed ordering)
  - Circular FOV mask computation
  - Sinogram / image crop utilities
  - NaN / Inf sanitisation
"""

import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class _UtilsMixin:
    """Utility methods shared across geometry, projector, reconstruction, and simulation."""

    # ── PSF filtering ──────────────────────────────────────────────────────────

    def gaussFilter(self, img: np.ndarray, fwhm, is3d: bool = False) -> np.ndarray:
        """Apply an isotropic Gaussian PSF to a flattened image vector.

        Parameters
        ----------
        img  : 1-D or (batch, pixels) array in scanner-native Fortran order
        fwhm : scalar FWHM in cm (same along all axes)
        is3d : True for 3-D images

        Returns
        -------
        Filtered array of the same shape.
        """
        fwhm = np.asarray(fwhm, dtype=float)
        if np.all(fwhm == 0):
            return img

        if is3d:
            img = img.reshape(self.image.matrixSize, order="F")
            voxelSizeCm = self.image.voxelSizeCm
        else:
            img = img.reshape(self.image.matrixSize[:2], order="F")
            voxelSizeCm = self.image.voxelSizeCm[:2]

        # Expand scalar FWHM to match spatial dimensions
        if fwhm.ndim == 0:
            fwhm = fwhm * np.ones(3 if is3d else 2)

        sigma = fwhm / np.asarray(voxelSizeCm) / np.sqrt(8.0 * np.log(2))
        return ndimage.gaussian_filter(img, sigma).flatten("F")

    def gaussFilterBatch(self, img: np.ndarray, fwhm) -> np.ndarray:
        """Apply a Gaussian PSF to a batch of images (respects self.is3d)."""
        voxelSizeCm = np.asarray(self.image.voxelSizeCm)
        is3d        = self.is3d
        fwhm        = np.asarray(fwhm, dtype=float)

        if np.all(fwhm == 0):
            return img

        if not is3d:
            voxelSizeCm = voxelSizeCm[:2]

        if fwhm.ndim == 0:
            fwhm = fwhm * np.ones(3 if is3d else 2)

        sigma = fwhm / voxelSizeCm / np.sqrt(8.0 * np.log(2))

        def _filt(x):
            return ndimage.gaussian_filter(x, sigma)

        if is3d:
            if img.ndim == 3:
                return _filt(img)
            out = np.zeros_like(img)
            for b in range(img.shape[0]):
                out[b] = _filt(img[b])
            return out
        else:
            if img.ndim == 2:
                return _filt(img)
            out = np.zeros_like(img)
            for b in range(img.shape[0]):
                out[b] = _filt(img[b])
            return out

    # ── Angular subset generation ──────────────────────────────────────────────

    def bit_reverse(self, mm: int) -> np.ndarray:
        """Bit-reversal permutation of indices 0..mm-1 (used for OSEM ordering)."""
        dec2bin       = lambda x, y: [np.binary_repr(x[i], width=y) for i in range(len(x))]
        bin2dec_rev   = lambda x: np.array([int(x[i][::-1], 2) for i in range(len(x))])
        nn = 2 ** int(np.ceil(np.log2(mm)))
        y  = len(np.binary_repr(nn - 1))
        ii = bin2dec_rev(dec2bin(np.arange(nn), y))
        return ii[ii < mm]

    def check_nsubs(self, nsub: int) -> None:
        """Raise ValueError if *nsub* does not evenly divide nAngularBins."""
        nAngles = self.sinogram.nAngularBins
        if (nAngles % nsub) != 0:
            valid = np.arange(1, nAngles)
            valid = valid[(nAngles % valid) == 0]
            raise ValueError(f"nsub={nsub} does not divide nAngularBins={nAngles}. "
                             f"Valid choices: {valid.tolist()}")

    def angular_subsets(self, nsub: int):
        """Return (subsets, subsize) arrays for bit-reversed OSEM subset ordering.

        Parameters
        ----------
        nsub : number of OSEM subsets

        Returns
        -------
        subsets : (subsize, nsub) int16 — angular indices per subset
        subsize : int — number of angles per subset
        """
        nAngles = self.sinogram.nAngularBins
        if (nAngles // nsub) % 2 != 0:
            valid = np.arange(1, nAngles / 2)
            valid = valid[(np.mod(nAngles / 2 / valid, 1)) == 0]
            raise ValueError(f"nsub={nsub} yields odd subsize. Valid choices: {valid.tolist()}")

        subsize = int(nAngles / nsub)
        subsets = np.zeros((subsize, nsub), dtype="int16")
        for j in range(nsub):
            k = 0
            for i in np.arange(j, nAngles // 2, nsub):
                subsets[k, j]              = i
                subsets[k + subsize // 2, j] = i + nAngles // 2
                k += 1

        # Apply bit-reversed ordering across subsets for better convergence
        st = self.bit_reverse(nsub)
        ordered = np.zeros_like(subsets)
        for i in range(nsub):
            ordered[:, i] = subsets[:, st[i]]
        return ordered, subsize

    # ── FOV mask ───────────────────────────────────────────────────────────────

    def mask_fov(self, reconFovRadious: float = None) -> np.ndarray:
        """Return a boolean FOV mask (flattened Fortran order for 2-D).

        The mask is cached after first computation.
        """
        if self.fov_mask is not None:
            return self.fov_mask

        if reconFovRadious is not None:
            r = reconFovRadious * 0.96
        elif hasattr(self.image, "reconFovRadious"):
            r = self.image.reconFovRadious * 0.96
        else:
            r = self.sinogram.nRadialBins * self.scanner.xCrystalDimCm / 4.0 * 0.96

        x  = self.image.voxelSizeCm[0] * np.arange(-self.image.matrixSize[0] // 2,
                                                     self.image.matrixSize[0] // 2)
        y  = self.image.voxelSizeCm[1] * np.arange(-self.image.matrixSize[1] // 2,
                                                     self.image.matrixSize[1] // 2)
        xx, yy = np.meshgrid(x, y)

        if self.is3d:
            mask = np.zeros(self.image.matrixSize)
            for i in range(mask.shape[2]):
                mask[:, :, i] = (xx ** 2 + yy ** 2) < r ** 2
        else:
            mask = ((xx ** 2 + yy ** 2) < r ** 2).flatten("F")

        self.fov_mask = mask
        return mask

    # ── Sinogram / image crop utilities ───────────────────────────────────────

    def crop_sino(self, sino: np.ndarray) -> np.ndarray:
        """Crop radial bins from both ends using radialBinCropfactor."""
        cf = self.sinogram.radialBinCropfactor
        if cf == 0:
            return sino
        i = int(np.ceil(sino.shape[0] * cf / 2.0) * 2) // 2
        return sino[i: sino.shape[0] - i]

    def crop_img(self, img: np.ndarray, crop_factor: float = None) -> np.ndarray:
        """Crop spatial pixels from all sides by *crop_factor* fraction."""
        cf = self.sinogram.radialBinCropfactor if crop_factor is None else crop_factor
        if cf == 0:
            return img
        i = int(np.ceil(img.shape[0] * cf / 2.0) * 2) // 2
        return img[i: img.shape[0] - i, i: img.shape[1] - i]

    def uncrop_img(self, img: np.ndarray) -> np.ndarray:
        """Zero-pad a cropped image batch back to the original radial bin size."""
        W = self.sinogram.nRadialBins_orig
        i = (W - img.shape[1]) // 2
        out = np.zeros((W, W, img.shape[2]), dtype=img.dtype)
        out[i: W - i, i: W - i, :] = img
        return out

    # ── NaN / Inf sanitisation ─────────────────────────────────────────────────

    def zeroNanInfs(self, x: np.ndarray) -> np.ndarray:
        """Replace NaN and Inf with 0 in-place."""
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x

    # ── Synthetic phantom (for testing) ───────────────────────────────────────

    def buildPhantom(self, model: int = 0, display: bool = False) -> np.ndarray:
        """Generate a simple Shepp–Logan-like 3-D test phantom.

        Parameters
        ----------
        model   : 0 = default ellipsoid phantom
        display : show four axial slices with matplotlib

        Returns
        -------
        img : np.ndarray shaped like self.image.matrixSize
        """
        if model != 0:
            raise ValueError(f"Unknown phantom model {model}. Only model=0 is supported.")

        x  = np.arange(-self.image.matrixSize[0] // 2, self.image.matrixSize[0] // 2)
        y  = np.arange(-self.image.matrixSize[1] // 2, self.image.matrixSize[1] // 2)
        z  = np.arange(-self.image.matrixSize[2] // 2, self.image.matrixSize[2] // 2)
        xx, yy, zz = np.meshgrid(x, y, z)

        img = (
            ((xx ** 2 + (yy / 1.3) ** 2 + (zz / 0.8) ** 2) < 80 ** 2).astype(float)
            + 2.5 * (((xx + 20) ** 2 + (yy + 20) ** 2 + (zz / 3) ** 2) < 10 ** 2)
            + 3.0 * (((xx / 0.45 - 60) ** 2 + (yy + 20) ** 2 + (zz / 3) ** 2) < 60 ** 2)
            + 2.0 * ((xx ** 2 + (yy / 0.8 - 30) ** 2 + (zz / 3) ** 2) < 15 ** 2)
        )

        if display:
            from matplotlib import pyplot as plt
            slices = np.sort(np.random.randint(self.image.matrixSize[2], size=4))
            fig, axes = plt.subplots(2, 2)
            for ax, sl in zip(axes.flat, slices):
                ax.imshow(img[:, :, sl])
                ax.set_title(f"Slice {sl}", fontsize=15)
            plt.show()

        return img
