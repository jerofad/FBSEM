"""
Iterative reconstruction algorithm mixin for BuildGeometry_v4.

Provides:
  OSEM2D         — Ordered Subset Expectation-Maximisation (2-D, batch)
  MAPEM2D        — MR-guided MAP-EM with Bowsher prior (2-D)
  mrMAPEM2DBatch — batch wrapper around MAPEM2D using an MR image as prior
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class _ReconstructionMixin:
    """2-D iterative PET reconstruction algorithms."""

    # ── OSEM ──────────────────────────────────────────────────────────────────

    def OSEM2D(self, prompts: np.ndarray, img: np.ndarray = None,
               RS=None, AN=None, iSensImg=None,
               niter: int = 100, nsubs: int = 1,
               tof: bool = False, psf: float = 0) -> np.ndarray:
        """2-D Ordered Subset Expectation-Maximisation reconstruction.

        Parameters
        ----------
        prompts  : (B, R, A) or (R, A) — measured / simulated sinogram
        img      : initial image estimate; uniform ones if None
        RS       : scatter + randoms sinogram; zeros if None
        AN       : attenuation × normalisation; ones if None
        iSensImg : pre-computed inverse sensitivity; computed if None
        niter    : number of full iterations (each processes all subsets)
        nsubs    : number of OSEM subsets
        tof      : use time-of-flight weights
        psf      : PSF FWHM (cm)

        Returns
        -------
        img : (B, W, W) or (W, W) — reconstructed image(s)
        """
        tic = time.time()
        numAng, subSize = self.angular_subsets(nsubs)

        if tof and not self.scanner.isTof:
            raise ValueError("Scanner does not support TOF.")

        # Normalise input dimensions
        if not tof and prompts.ndim == 2:
            batch_size = 1;  prompts = prompts[None, :, :]
        elif tof and prompts.ndim == 3:
            batch_size = 1;  prompts = prompts[None, :, :, :]
        else:
            batch_size = prompts.shape[0]

        mS      = self.image.matrixSize
        nVoxels = int(np.prod(mS[:2]))
        q       = self.sinogram.nAngularBins // 2

        if img is None:
            img = np.ones([batch_size, nVoxels], dtype="float32")
        else:
            if batch_size > 1 and img.shape[0] != batch_size:
                raise ValueError("img batch dimension does not match prompts.")
        img = img.reshape([batch_size, nVoxels], order="F")

        if RS is None:
            RS = np.zeros_like(prompts)
        if AN is None:
            AN = np.ones([batch_size, self.sinogram.nRadialBins,
                          self.sinogram.nAngularBins], dtype="float32")
        elif AN.ndim == 2:
            AN = AN[None, :, :]

        if iSensImg is None:
            iSensImg = self.iSensImageBatch2D(AN, nsubs, psf)
        if iSensImg.ndim == 2:
            iSensImg = iSensImg[None, :, :]   # (1, nsubs, W*W)

        for n in range(niter):
            for sub in range(nsubs):
                img_ = img.copy()
                if np.any(psf != 0):
                    for b in range(batch_size):
                        img_[b, :] = self.gaussFilter(img_[b, :], psf)

                backProj = np.zeros_like(img_)
                for ii in range(subSize // 2):
                    i = numAng[ii, sub]
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
                                    prompts[b, j, i, :] / (
                                        (AN[b, j, i] * G * img_[b, idx1]).dot(W)
                                        + RS[b, j, i, :] + 1e-5))
                                backProj[b, idx2] += G * AN[b, j, i + q] * W.dot(
                                    prompts[b, j, i + q, :] / (
                                        (AN[b, j, i + q] * G * img_[b, idx2]).dot(W)
                                        + RS[b, j, i + q, :] + 1e-5))
                        else:
                            for b in range(batch_size):
                                backProj[b, idx1] += G * AN[b, j, i] * (
                                    prompts[b, j, i] / (
                                        AN[b, j, i] * G.dot(img_[b, idx1])
                                        + RS[b, j, i] + 1e-5))
                                backProj[b, idx2] += G * AN[b, j, i + q] * (
                                    prompts[b, j, i + q] / (
                                        AN[b, j, i + q] * G.dot(img_[b, idx2])
                                        + RS[b, j, i + q] + 1e-5))

                if np.any(psf != 0):
                    for b in range(batch_size):
                        backProj[b, :] = self.gaussFilter(backProj[b, :], psf)

                img = img * backProj * iSensImg[:, sub, :]

        img = img.reshape([batch_size, mS[0], mS[1]], order="F")
        if batch_size == 1:
            img = img[0]

        logger.info("OSEM2D: %d batch(es) reconstructed in %.2f s.",
                    batch_size, time.time() - tic)
        return img

    # ── MR-guided MAP-EM ───────────────────────────────────────────────────────

    def MAPEM2D(self, prompts: np.ndarray, img: np.ndarray = None,
                RS=None, niter: int = 100, nsubs: int = 1,
                AN=None, tof: bool = False, psf: float = 0,
                beta: float = 1.0, prior=None,
                prior_weights=1.0) -> np.ndarray:
        """2-D MR-guided MAP-EM reconstruction with a Bowsher-type prior.

        The FBSEM update rule is:
            img_new = 2 * img_em / [
                sqrt((1 - β·wj·reg)² + 4·β·wj·img_em) + (1 - β·wj·reg)
            ]

        where img_em is the standard OSEM correction and reg = Div(W * Div(img)).

        Parameters
        ----------
        prompts       : (R, A) — single-slice sinogram (non-batch)
        beta          : regularisation strength
        prior         : ``Prior`` object; default Bowsher prior created if None
        prior_weights : per-voxel prior weights (from MR anatomy)

        Returns
        -------
        img : (W, W) — reconstructed image
        """
        tic = time.time()
        numAng, subSize = self.angular_subsets(nsubs)

        if tof and not self.scanner.isTof:
            raise ValueError("Scanner does not support TOF.")
        if img is None:
            img = np.ones(self.image.matrixSize[:2], dtype="float32")
        img = img.flatten("F")

        mS      = self.image.matrixSize
        nVoxels = int(np.prod(mS[:2]))
        q       = self.sinogram.nAngularBins // 2

        sensImg     = np.zeros(nVoxels, dtype="float64")
        sensImgSubs = np.zeros([nVoxels, nsubs], dtype="float64")

        if np.ndim(prompts) != 3:
            tof = False
        if RS is None:
            RS = np.zeros_like(prompts)
        if AN is None:
            AN = np.ones([self.sinogram.nRadialBins, self.sinogram.nAngularBins],
                         dtype="float32")

        if prior is None:
            from geometry.Prior import Prior
            prior = Prior(list(mS[:2]))

        W   = prior.Wd * prior_weights
        wj  = prior.imCropUndo(W.sum(axis=1))

        first_iter = True

        for n in range(niter):
            for sub in range(nsubs):
                imgOld = self.gaussFilter(img.copy(), psf) if np.any(psf != 0) else img.copy()
                backProj  = np.zeros(nVoxels, dtype="float64")
                sensImg[:] = 0.0

                for ii in range(subSize // 2):
                    i = numAng[ii, sub]
                    for j in range(self.sinogram.nRadialBins):
                        M0 = self.geoMatrix[0][i, j]
                        if np.isscalar(M0):
                            continue
                        M    = M0[:, :3].astype("int32")
                        G    = M0[:, 3] / 1e4
                        idx1 = M[:, 0] + M[:, 1] * mS[0]
                        idx2 = M[:, 1] + mS[0] * (mS[0] - 1 - M[:, 0])

                        if tof:
                            Wt  = self.tofMatrix[0][i, j] / 1e4
                            backProj[idx1] += G * AN[j, i] * Wt.dot(
                                prompts[j, i, :] / (AN[j, i] * (G * imgOld[idx1]).dot(Wt)
                                                    + RS[j, i, :] + 1e-5))
                            backProj[idx2] += G * AN[j, i + q] * Wt.dot(
                                prompts[j, i + q, :] / (AN[j, i + q] * (G * imgOld[idx2]).dot(Wt)
                                                        + RS[j, i + q, :] + 1e-5))
                            if first_iter:
                                GW = G * np.sum(Wt, axis=1)
                                sensImg[idx1] += GW * AN[j, i]
                                sensImg[idx2] += GW * AN[j, i + q]
                        else:
                            backProj[idx1] += G * AN[j, i] * (
                                prompts[j, i] / (AN[j, i] * G.dot(imgOld[idx1])
                                                 + RS[j, i] + 1e-5))
                            backProj[idx2] += G * AN[j, i + q] * (
                                prompts[j, i + q] / (AN[j, i + q] * G.dot(imgOld[idx2])
                                                     + RS[j, i + q] + 1e-5))
                            if first_iter:
                                sensImg[idx1] += G * AN[j, i]
                                sensImg[idx2] += G * AN[j, i + q]

                if first_iter:
                    s = self.gaussFilter(sensImg.copy(), psf) + 1e-5 if np.any(psf != 0) else sensImg + 1e-5
                    sensImgSubs[:, sub] = s

                if np.any(psf != 0):
                    backProj = self.gaussFilter(backProj.copy(), psf)

                img_sens = sensImgSubs[:, sub]
                betaj    = beta * wj / img_sens
                img_em   = img * backProj / img_sens

                # Bowsher / FBSEM regularisation term
                img_reg = (1.0 / (2.0 * wj + 1e-8)
                           * prior.imCropUndo((W * prior.Div(img)).sum(axis=1)))

                # FBSEM closed-form combination
                disc  = (1.0 - betaj * img_reg) ** 2 + 4.0 * betaj * img_em
                img   = 2.0 * img_em / (np.sqrt(disc) + (1.0 - betaj * img_reg) + 1e-5)

            first_iter = False

        img = img.reshape(mS[:2], order="F")
        logger.info("MAPEM2D: reconstructed in %.2f min.", (time.time() - tic) / 60)
        return img

    # ── Batch MR-guided MAPEM ─────────────────────────────────────────────────

    def mrMAPEM2DBatch(self, prompts: np.ndarray, AN: np.ndarray,
                       mrImg: np.ndarray, RS=None,
                       beta: float = 1.0, niters: int = 15,
                       nsubs: int = 14, psf: float = 0) -> np.ndarray:
        """Apply MR-guided MAPEM2D to a batch of sinograms.

        Each sinogram slice is reconstructed independently using the
        corresponding MR slice as the anatomical prior.

        Parameters
        ----------
        prompts : (B, R, A) — batch of sinograms
        AN      : (B, R, A) — attenuation × normalisation factors
        mrImg   : (B, W, W) — MR images (normalised internally to [0, 1])
        beta    : regularisation strength
        niters  : MAPEM iterations per slice
        nsubs   : OSEM subsets per iteration
        psf     : PSF FWHM (cm)

        Returns
        -------
        img_mapem : (B, W, W) — reconstructed images
        """
        from geometry.Prior import Prior
        from skimage.transform import resize as sk_resize
        W           = int(self.image.matrixSize[0])
        H           = int(self.image.matrixSize[1])
        prior       = Prior([W, H], sWindowSize=3)
        num_batches = int(prompts.shape[0])
        img_mapem   = np.zeros((num_batches, W, H), dtype="float32")

        for i in range(num_batches):
            mr_i = mrImg[i]
            if mr_i.shape[0] != W or mr_i.shape[1] != H:
                mr_i = sk_resize(mr_i, (W, H), order=1,
                                 preserve_range=True, anti_aliasing=True).astype("float32")
            mr_norm  = mr_i / (mr_i.max() + 1e-8)
            weights  = prior.BowshserWeights(mr_norm, prior.nS // 2)
            result   = self.MAPEM2D(
                prompts[i], niter=niters, nsubs=nsubs,
                AN=AN[i], psf=psf, beta=beta,
                prior=prior, prior_weights=weights,
            )
            img_mapem[i] = result.reshape(W, H)
            logger.debug("mrMAPEM2DBatch: slice %d/%d done.", i + 1, num_batches)

        return img_mapem
