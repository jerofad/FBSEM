"""
Projection operator mixin for BuildGeometry_v4.

Provides the 2-D (primary focus) forward and back-projection operators,
plus the inverse sensitivity image computation needed by OSEM / FBSEM.

All methods operate on batches of images / sinograms to amortise the
Python loop overhead over the system matrix.

Notation
--------
  B  = batch size
  R  = nRadialBins
  A  = nAngularBins
  W  = matrixSize[0]  (number of image pixels per side)
  q  = A // 2         (half the angular bins; symmetry doubles coverage)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class _ProjectorMixin:
    """2-D (and stub 3-D) forward / back-projection and sensitivity operators."""

    # ── 2-D inverse sensitivity image ─────────────────────────────────────────

    def iSensImageBatch2D(self, AN: np.ndarray = None,
                          nsubs: int = 1, psf: float = 0) -> np.ndarray:
        """Compute the (normalised) inverse sensitivity image for each OSEM subset.

        The sensitivity image S_s = A^T * AN_s is the backprojection of the
        attenuation-normalisation factors restricted to subset *s*.  We return
        1/S_s (clamped to zero where S_s = 0) for use in the EM update:
            img_new = img * (A^T (y / (A img))) * (1/S_s)

        Parameters
        ----------
        AN   : (B, R, A) or (R, A) — attenuation × normalisation factors;
               ones if None (uniform sensitivity)
        nsubs : number of OSEM subsets
        psf   : PSF FWHM (cm) — applied after backprojection

        Returns
        -------
        iSensImg : (B, nsubs, W*W) — inverse sensitivity per subset per sample
                   Shape collapses to (nsubs, W*W) when B=1.
        """
        if AN is None:
            AN = np.ones([1, self.sinogram.nRadialBins, self.sinogram.nAngularBins],
                         dtype="float32")
        elif AN.ndim == 2:
            AN = AN[None, :, :]
        batch_size = AN.shape[0]

        nVoxels     = int(np.prod(self.image.matrixSize[:2]))
        matrixSize  = self.image.matrixSize
        q           = self.sinogram.nAngularBins // 2
        numAng, subSize = self.angular_subsets(nsubs)

        sensImg    = np.zeros([batch_size, nVoxels], dtype="float64")
        sensSubBatch = np.zeros([batch_size, nsubs, nVoxels], dtype="float64")

        for sub in range(nsubs):
            sensImg[:] = 0.0
            for ii in range(subSize // 2):
                i  = numAng[ii, sub]
                for j in range(self.sinogram.nRadialBins):
                    M0 = self.geoMatrix[0][i, j]
                    if np.isscalar(M0):
                        continue
                    M    = M0[:, :3].astype("int32")
                    G    = M0[:, 3] / 1e4
                    idx1 = M[:, 0] + M[:, 1] * matrixSize[0]
                    idx2 = M[:, 1] + matrixSize[0] * (matrixSize[0] - 1 - M[:, 0])

                    if self.scanner.isTof:
                        W   = self.tofMatrix[0][i, j] / 1e4
                        GW  = G * np.sum(W, axis=1)
                        for b in range(batch_size):
                            sensImg[b, idx1] += GW * AN[b, j, i]
                            sensImg[b, idx2] += GW * AN[b, j, i + q]
                    else:
                        for b in range(batch_size):
                            sensImg[b, idx1] += G * AN[b, j, i]
                            sensImg[b, idx2] += G * AN[b, j, i + q]

            for b in range(batch_size):
                s = self.gaussFilter(sensImg[b, :], psf) * self.mask_fov()
                sensSubBatch[b, sub, :] = s

        iSens = np.zeros_like(sensSubBatch)
        mask  = sensSubBatch > 0
        iSens[mask] = 1.0 / sensSubBatch[mask]

        if batch_size == 1:
            iSens = iSens[0]   # (nsubs, W*W)
        return iSens.astype("float32")

    # ── 2-D forward projection ─────────────────────────────────────────────────

    def forwardProjectBatch2D(self, img: np.ndarray,
                               tof: bool = False, psf: float = 0) -> np.ndarray:
        """Project a batch of 2-D images into sinogram space.

        Parameters
        ----------
        img : (B, W, W) or (W, W) — image(s) in standard (non-Fortran) layout
        psf : PSF FWHM (cm) applied before projection
        tof : use TOF weights (requires a TOF-capable scanner)

        Returns
        -------
        y : (B, R, A) or (R, A) — forward-projected sinogram(s)
        """
        if tof and not self.scanner.isTof:
            raise ValueError("Scanner does not support TOF.")

        if img.ndim == 2:
            batch_size = 1;  img = img[None, :, :]
        else:
            batch_size = img.shape[0]

        img_flat = img.reshape([batch_size, int(np.prod(img.shape[1:3]))], order="F")

        if psf:
            for b in range(batch_size):
                img_flat[b, :] = self.gaussFilter(img_flat[b, :], psf)

        si     = self.sinogram
        dims   = [batch_size, si.nRadialBins, si.nAngularBins]
        if tof: dims.append(si.nTofBins)
        y      = np.zeros(dims, dtype="float32")
        mS     = self.image.matrixSize
        q      = si.nAngularBins // 2

        for i in range(si.nAngularBins // 2):
            for j in range(si.nRadialBins):
                M0 = self.geoMatrix[0][i, j]
                if np.isscalar(M0):
                    continue
                M    = M0[:, :3].astype("int32")
                G    = M0[:, 3] / 1e4
                idx1 = M[:, 0] + M[:, 1] * mS[0]
                idx2 = M[:, 1] + mS[0] * (mS[0] - 1 - M[:, 0])
                if tof:
                    W = self.tofMatrix[0][i, j] / 1e4
                    for b in range(batch_size):
                        y[b, j, i, :]     = (G * img_flat[b, idx1]).dot(W)
                        y[b, j, i + q, :] = (G * img_flat[b, idx2]).dot(W)
                else:
                    for b in range(batch_size):
                        y[b, j, i]     = G.dot(img_flat[b, idx1])
                        y[b, j, i + q] = G.dot(img_flat[b, idx2])

        logger.debug("forwardProjectBatch2D: %d batch(es)", batch_size)
        if batch_size == 1:
            y = y[0]
        return y

    # ── 2-D back-projection ────────────────────────────────────────────────────

    def backProjectBatch2D(self, sinodata: np.ndarray = None,
                            tof: bool = False, psf: float = 0) -> np.ndarray:
        """Back-project a batch of sinograms into image space.

        Parameters
        ----------
        sinodata : (B, R, A) or (R, A) — sinogram(s)
        psf      : PSF FWHM (cm) applied after backprojection

        Returns
        -------
        img : (B, W, W) or (W, W)
        """
        if tof and not self.scanner.isTof:
            raise ValueError("Scanner does not support TOF.")

        si = self.sinogram
        if sinodata is None:
            batch_size = 1
            sinodata = (np.ones([1, si.nRadialBins, si.nAngularBins, si.nTofBins], dtype="float32")
                        if tof else
                        np.ones([1, si.nRadialBins, si.nAngularBins], dtype="float32"))
        else:
            if (tof and sinodata.ndim == 4) or (not tof and sinodata.ndim == 3):
                batch_size = sinodata.shape[0]
            else:
                batch_size = 1;  sinodata = sinodata[None]

        mS  = self.image.matrixSize
        img = np.zeros([batch_size, int(np.prod(mS[:2]))], dtype="float32")
        q   = si.nAngularBins // 2

        for i in range(si.nAngularBins // 2):
            for j in range(si.nRadialBins):
                M0 = self.geoMatrix[0][i, j]
                if np.isscalar(M0):
                    continue
                M    = M0[:, :3].astype("int32")
                G    = M0[:, 3] / 1e4
                idx1 = M[:, 0] + M[:, 1] * mS[0]
                idx2 = M[:, 1] + mS[0] * (mS[0] - 1 - M[:, 0])
                if tof:
                    W = self.tofMatrix[0][i, j] / 1e4
                    for b in range(batch_size):
                        img[b, idx1] += G * W.dot(sinodata[b, j, i, :])
                        img[b, idx2] += G * W.dot(sinodata[b, j, i + q, :])
                else:
                    for b in range(batch_size):
                        img[b, idx1] += G * sinodata[b, j, i]
                        img[b, idx2] += G * sinodata[b, j, i + q]

        if np.any(psf != 0):
            for b in range(batch_size):
                img[b, :] = self.gaussFilter(img[b, :], psf)

        img = np.reshape(img, [batch_size, mS[0], mS[1]], order="F")
        if batch_size == 1:
            img = img[0]
        return img

    # ── 2-D EM numerator: A^T [ y / (A f) ] ─────────────────────────────────

    def forwardDivideBackwardBatch2D(self, imgb: np.ndarray,
                                      prompts: np.ndarray,
                                      RS=None, AN=None,
                                      nsubs: int = 1, subset_i: int = 0,
                                      tof: bool = False, psf: float = 0) -> np.ndarray:
        """Compute the EM correction image: A^T [ y / (A f + RS) ] for one subset.

        This is the numerator of the OSEM update used inside FBSEMnet_v3.

        Parameters
        ----------
        imgb     : (B, W*W) — current image estimate (flattened Fortran order)
        prompts  : (B, R, A) — measured sinogram (low-dose)
        RS       : (B, R, A) or None — scatter + randoms estimate
        AN       : (B, R, A) or None — attenuation × normalisation
        nsubs    : total number of OSEM subsets
        subset_i : which subset to process (0-indexed)
        psf      : PSF FWHM (cm)

        Returns
        -------
        backProjImage : (B, W*W) — EM correction image (un-normalised)
        """
        if tof and not self.scanner.isTof:
            raise ValueError("Scanner does not support TOF.")
        if nsubs > 1 and subset_i >= nsubs:
            raise ValueError(f"subset_i must be in [0, {nsubs - 1}].")

        numAng, subSize = self.angular_subsets(nsubs)
        batch_size      = prompts.shape[0]
        mS              = self.image.matrixSize
        nVoxels         = int(np.prod(mS[:2]))
        q               = self.sinogram.nAngularBins // 2

        img = imgb.copy().reshape([batch_size, nVoxels], order="F")

        if prompts.ndim != 4:
            tof = False
        if RS is None:
            RS = np.zeros_like(prompts)
        if AN is None:
            AN = np.ones([batch_size, self.sinogram.nRadialBins,
                          self.sinogram.nAngularBins], dtype="float32")

        if np.any(psf != 0):
            for b in range(batch_size):
                img[b, :] = self.gaussFilter(img[b, :], psf)

        backProj = np.zeros_like(img)

        for ii in range(subSize // 2):
            i = numAng[ii, subset_i]
            for j in range(self.sinogram.nRadialBins):
                M0 = self.geoMatrix[0][i, j]
                if np.isscalar(M0):
                    continue
                M    = M0[:, :3].astype("int32")
                G    = M0[:, 3] / 1e4
                idx1 = M[:, 0] + M[:, 1] * mS[0]
                idx2 = M[:, 1] + mS[0] * (mS[0] - 1 - M[:, 0])

                if tof:
                    W = self.tofMatrix[0][i, j] / 1e4
                    for b in range(batch_size):
                        backProj[b, idx1] += G * AN[b, j, i] * W.dot(
                            prompts[b, j, i, :] / (AN[b, j, i] * (G * img[b, idx1]).dot(W)
                                                    + RS[b, j, i, :] + 1e-5))
                        backProj[b, idx2] += G * AN[b, j, i + q] * W.dot(
                            prompts[b, j, i + q, :] / (AN[b, j, i + q] * (G * img[b, idx2]).dot(W)
                                                        + RS[b, j, i + q, :] + 1e-5))
                else:
                    for b in range(batch_size):
                        backProj[b, idx1] += G * AN[b, j, i] * (
                            prompts[b, j, i] / (AN[b, j, i] * G.dot(img[b, idx1])
                                                + RS[b, j, i] + 1e-5))
                        backProj[b, idx2] += G * AN[b, j, i + q] * (
                            prompts[b, j, i + q] / (AN[b, j, i + q] * G.dot(img[b, idx2])
                                                    + RS[b, j, i + q] + 1e-5))

        if np.any(psf != 0):
            for b in range(batch_size):
                backProj[b, :] = self.gaussFilter(backProj[b, :], psf) * self.mask_fov()

        return backProj

    # ── 2-D A^T A f  (Fisher information denominator) ─────────────────────────

    def forwardBackwardBatch2D(self, imgb: np.ndarray,
                                RS=None, AN=None,
                                nsubs: int = 1, subset_i: int = 0,
                                tof: bool = False, psf: float = 0) -> np.ndarray:
        """Compute A^T [ AN * (A f + RS) ] for one subset (Fisher denominator).

        Used for curvature-based step-size computation.

        Returns
        -------
        result : (B, W, W) or (W, W) reshaped image
        """
        if tof and not self.scanner.isTof:
            raise ValueError("Scanner does not support TOF.")

        numAng, subSize = self.angular_subsets(nsubs)
        img = imgb.copy()
        if img.ndim == 2:
            batch_size = 1;  img = img[None]
        else:
            batch_size = img.shape[0]

        mS      = self.image.matrixSize
        nVoxels = int(np.prod(mS[:2]))
        img     = img.reshape([batch_size, nVoxels], order="F")
        q       = self.sinogram.nAngularBins // 2

        if RS is None:
            dims = [batch_size, self.sinogram.nRadialBins, self.sinogram.nAngularBins]
            if tof and self.scanner.isTof: dims.append(self.sinogram.nTofBins)
            RS = np.zeros(dims, dtype="float32")
        if AN is None:
            AN = np.ones([batch_size, self.sinogram.nRadialBins,
                          self.sinogram.nAngularBins], dtype="float32")

        if np.any(psf != 0):
            for b in range(batch_size):
                img[b, :] = self.gaussFilter(img[b, :], psf)

        backProj = np.zeros_like(img)

        for ii in range(subSize // 2):
            i = numAng[ii, subset_i]
            for j in range(self.sinogram.nRadialBins):
                M0 = self.geoMatrix[0][i, j]
                if np.isscalar(M0):
                    continue
                M    = M0[:, :3].astype("int32")
                G    = M0[:, 3] / 1e4
                idx1 = M[:, 0] + M[:, 1] * mS[0]
                idx2 = M[:, 1] + mS[0] * (mS[0] - 1 - M[:, 0])

                if tof:
                    W = self.tofMatrix[0][i, j] / 1e4
                    for b in range(batch_size):
                        backProj[b, idx1] += G * AN[b, j, i] * W.dot(
                            AN[b, j, i] * (G * img[b, idx1]).dot(W) + RS[b, j, i, :])
                        backProj[b, idx2] += G * AN[b, j, i + q] * W.dot(
                            AN[b, j, i + q] * (G * img[b, idx2]).dot(W) + RS[b, j, i + q, :])
                else:
                    for b in range(batch_size):
                        backProj[b, idx1] += G * AN[b, j, i] * (
                            AN[b, j, i] * G.dot(img[b, idx1]) + RS[b, j, i])
                        backProj[b, idx2] += G * AN[b, j, i + q] * (
                            AN[b, j, i + q] * G.dot(img[b, idx2]) + RS[b, j, i + q])

        if np.any(psf != 0):
            for b in range(batch_size):
                backProj[b, :] = self.gaussFilter(backProj[b, :], psf) * self.mask_fov()

        backProj = backProj.reshape([batch_size, mS[0], mS[1]], order="F")
        if batch_size == 1:
            backProj = backProj[0]
        return backProj

    # ── 2-D multi-subset weighted backprojection ───────────────────────────────

    def backwardBatch2D_i(self, prompts: np.ndarray,
                           AN=None, nsubs: int = 1,
                           tof: bool = False, psf: float = 0) -> np.ndarray:
        """Backproject AN-weighted prompts for all subsets simultaneously.

        Returns
        -------
        result : (B, W, W, nsubs) or (W, W, nsubs)
        """
        if tof and not self.scanner.isTof:
            raise ValueError("Scanner does not support TOF.")

        numAng, subSize = self.angular_subsets(nsubs)
        if (tof and prompts.ndim == 4) or (not tof and prompts.ndim == 3):
            batch_size = prompts.shape[0]
        else:
            batch_size = 1;  prompts = prompts[None]

        mS      = self.image.matrixSize
        nVoxels = int(np.prod(mS[:2]))
        q       = self.sinogram.nAngularBins // 2

        if AN is None:
            AN = np.ones([batch_size, self.sinogram.nRadialBins,
                          self.sinogram.nAngularBins], dtype="float32")

        backProj = np.zeros([batch_size, nVoxels, nsubs], dtype="float32")

        for s in range(nsubs):
            for ii in range(subSize // 2):
                i = numAng[ii, s]
                for j in range(self.sinogram.nRadialBins):
                    M0 = self.geoMatrix[0][i, j]
                    if np.isscalar(M0):
                        continue
                    M    = M0[:, :3].astype("int32")
                    G    = M0[:, 3] / 1e4
                    idx1 = M[:, 0] + M[:, 1] * mS[0]
                    idx2 = M[:, 1] + mS[0] * (mS[0] - 1 - M[:, 0])

                    if tof:
                        W = self.tofMatrix[0][i, j] / 1e4
                        for b in range(batch_size):
                            backProj[b, idx1, s] += G * AN[b, j, i] * W.dot(prompts[b, j, i, :])
                            backProj[b, idx2, s] += G * AN[b, j, i + q] * W.dot(prompts[b, j, i + q, :])
                    else:
                        for b in range(batch_size):
                            backProj[b, idx1, s] += G * AN[b, j, i]     * prompts[b, j, i]
                            backProj[b, idx2, s] += G * AN[b, j, i + q] * prompts[b, j, i + q]

            if np.any(psf != 0):
                for b in range(batch_size):
                    backProj[b, :, s] = self.gaussFilter(backProj[b, :, s], psf) * self.mask_fov()

        backProj = backProj.reshape([batch_size, mS[0], mS[1], nsubs], order="F")
        if batch_size == 1:
            backProj = backProj[0]
        return backProj
