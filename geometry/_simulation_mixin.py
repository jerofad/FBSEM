"""
Sinogram simulation mixin for BuildGeometry_v4.

Provides simulateSinogramData — the only entry point needed for generating
training data from digital phantoms.

Physics model
-------------
1.  Forward-project the activity image to get true coincidences y.
2.  Multiply by the attenuation factor AF = exp(-∫ μ dl) to get y_att.
3.  Scale y_att to the requested total count level and apply Poisson noise.
4.  Multiply by the normalisation factor NF (detector efficiency, gaps).
5.  Optionally add Poisson-distributed randoms.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class _SimulationMixin:
    """Sinogram simulation from activity / attenuation maps."""

    def simulateSinogramData(
        self,
        img: np.ndarray,
        mumap: np.ndarray = None,
        AF: np.ndarray = None,
        NF: np.ndarray = None,
        counts: float = 1e7,
        psf: float = 0.0,
        tof: bool = False,
        randomsFraction: float = 0.0,
    ):
        """Simulate a noisy PET sinogram from an activity image.

        Parameters
        ----------
        img             : (B, W, W[, D]) or (W, W[, D]) — PET activity map
        mumap           : attenuation map (same shape as *img*); used to compute
                          AF if AF is not provided.  Units: cm⁻¹.
        AF              : pre-computed attenuation factor sinogram;
                          computed from *mumap* if None.
        NF              : normalisation factors sinogram (detector efficiency,
                          gap mask); simulated from LOR geometry if None.
        counts          : total true coincidences (scalar or per-batch array).
                          Scaled to match the requested count level before
                          Poisson sampling.
        psf             : PSF FWHM (cm) applied during forward projection.
        tof             : use TOF forward projector.
        randomsFraction : fraction of counts attributed to random coincidences
                          (0 = no randoms).

        Returns
        -------
        prompts : sinogram with Poisson noise, attenuation, normalisation,
                  and randoms applied.
        AF      : attenuation factor sinogram used (useful for reconstruction).
        NF      : normalisation factor sinogram used.
        Randoms : random-coincidence sinogram (0 if randomsFraction == 0).
        """
        if self.is3d:
            batch_size = 1 if img.ndim == 3 else img.shape[0]
            projector  = lambda x: self.forwardProjectBatch3D(x, psf=psf)
        else:
            batch_size = 1 if img.ndim == 2 else img.shape[0]
            projector  = lambda x: self.forwardProjectBatch2D(x, psf=psf)

        # ── Attenuation factor ─────────────────────────────────────────────────
        if AF is None:
            if mumap is None:
                AF = 1.0
            else:
                if mumap.ndim != img.ndim:
                    raise ValueError("mumap must have the same number of dimensions as img.")
                # Integrate attenuation along LORs: AF = exp(-∫ μ dl)
                AF = np.exp(-projector(mumap * self.image.voxelSizeCm[0]))
                AF[np.isinf(AF)] = 0.0

        # ── Normalisation factor ───────────────────────────────────────────────
        if NF is None:
            _, _, gaps = self.LorsTransaxialCoor()
            valid_lors = ~gaps.astype(bool)   # True where detector pair is real (not gap)
            NF         = np.zeros_like(AF if not np.isscalar(AF)
                                       else np.zeros([self.sinogram.nRadialBins,
                                                      self.sinogram.nAngularBins]))
            if batch_size == 1:
                NF = NF[None]
            if self.is3d:
                for b in range(batch_size):
                    for i in range(self.sinogram.totalNumberOfSinogramPlanes):
                        NF[b, :, :, i] = (valid_lors
                                          * np.exp(-np.random.rand(
                                              self.sinogram.nRadialBins,
                                              self.sinogram.nAngularBins)))
            else:
                for b in range(batch_size):
                    NF[b, :, :] = (valid_lors
                                   * np.exp(-np.random.rand(
                                       self.sinogram.nRadialBins,
                                       self.sinogram.nAngularBins)))
            if batch_size == 1:
                NF = NF[0]

        # ── True coincidences + Poisson noise ─────────────────────────────────
        counts_arr   = np.full(batch_size, counts) if np.isscalar(counts) else np.asarray(counts)
        trues_frac   = 1.0 - randomsFraction

        y       = projector(img)         # noiseless forward projection
        y_att   = y * AF                 # apply attenuation
        y_noisy = np.zeros_like(y_att)

        if batch_size > 1:
            for b in range(batch_size):
                if self.is3d:
                    sf = counts_arr[b] * trues_frac / y_att[b].sum()
                    y_noisy[b] = np.random.poisson(y_att[b] * sf) / sf
                else:
                    sf = counts_arr[b] * trues_frac / y_att[b].sum()
                    y_noisy[b] = np.random.poisson(y_att[b] * sf) / sf
        else:
            sf      = float(counts_arr[0]) * trues_frac / y_att.sum()
            y_noisy = np.random.poisson(y_att * sf) / sf

        # ── Randoms ───────────────────────────────────────────────────────────
        Randoms = 0
        if randomsFraction > 0:
            r_flat = np.ones_like(y)
            if batch_size > 1:
                for b in range(batch_size):
                    if self.is3d:
                        sfr = counts_arr[b] * randomsFraction / r_flat[b].sum()
                        r_flat[b] = np.random.poisson(r_flat[b] * sfr) / sfr
                    else:
                        sfr = counts_arr[b] * randomsFraction / r_flat[b].sum()
                        r_flat[b] = np.random.poisson(r_flat[b] * sfr) / sfr
            else:
                sfr    = float(counts_arr[0]) * randomsFraction / r_flat.sum()
                r_flat = np.random.poisson(r_flat * sfr) / sfr
            Randoms = r_flat

        prompts = y_noisy * NF + Randoms

        logger.debug(
            "simulateSinogramData: B=%d, counts=%.2e, psf=%.2f cm, randoms=%.0f%%",
            batch_size, float(counts_arr[0]), psf, randomsFraction * 100,
        )
        return prompts, AF, NF, Randoms
