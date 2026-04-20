"""
BuildGeometry_v4 — PET scanner geometry and reconstruction.

This module is a thin facade that composes the following mixin classes:

    _GeometryMixin      — scanner setup, LOR coordinates, system matrix I/O
    _ProjectorMixin     — 2-D forward / back-projection, iSensImageBatch2D
    _ReconstructionMixin— OSEM2D, MAPEM2D, mrMAPEM2DBatch
    _SimulationMixin    — simulateSinogramData
    _UtilsMixin         — Gaussian PSF filter, angular subsets, FOV mask,
                          crop helpers, NaN/Inf sanitisation

The public API is identical to the original monolithic version — all existing
calling code continues to work without modification.

Supported scanners
------------------
  'mmr'  — Siemens Biograph mMR (model 2008)
  'mct'  — Siemens Biograph mCT (model 1104)

Quick start (2-D)
-----------------
    from geometry.BuildGeometry_v4 import BuildGeometry_v4

    PET = BuildGeometry_v4('mmr', radialBinCropfactor=0.5)
    PET.loadSystemMatrix('/path/to/system_matrix', is3d=False)

    # Simulate training data
    prompts, AF, NF, _ = PET.simulateSinogramData(img, mumap=mumap, counts=5e6)

    # Reconstruct
    img_recon = PET.OSEM2D(prompts, AN=AF*NF, niter=10, nsubs=6)

Reference
---------
    Mehranian et al., "Model-Based Deep Learning PET Image Reconstruction
    Using Forward–Backward Splitting Expectation Maximization",
    IEEE TRPMS 2020. https://doi.org/10.1109/TRPMS.2020.3004408
"""

import logging
from sys import platform

import numpy as np

from geometry.scanner_params import SUPPORTED_SCANNERS
from geometry._utils_mixin          import _UtilsMixin
from geometry._geometry_mixin       import _GeometryMixin
from geometry._projector_mixin      import _ProjectorMixin
from geometry._reconstruction_mixin import _ReconstructionMixin
from geometry._simulation_mixin     import _SimulationMixin

np.seterr(divide="ignore")

logger = logging.getLogger(__name__)


class _Struct:
    """Minimal attribute-access wrapper (mirrors dotstruct in models/deeplib.py)."""
    def __setattr__(self, name, value): self.__dict__[name] = value
    def __getitem__(self, name):        return self.__dict__[name]
    def __contains__(self, name):       return name in self.__dict__
    def get(self, name, default=None):  return self.__dict__.get(name, default)
    def as_dict(self):                  return dict(self.__dict__)


class BuildGeometry_v4(
    _GeometryMixin,
    _ProjectorMixin,
    _ReconstructionMixin,
    _SimulationMixin,
    _UtilsMixin,
):
    """PET scanner geometry, projection operators, and iterative reconstruction.

    Parameters
    ----------
    scannerModel         : 'mmr' or 'mct' (case-insensitive)
    radialBinCropfactor  : fraction of radial bins trimmed from each end (0–1).
                           0.5 keeps the central 50 % of radial bins.
    """

    def __init__(self, scannerModel: str, radialBinCropfactor: float = 0.0):
        self.scanner  = _Struct()
        self.sinogram = _Struct()
        self.engine   = _Struct()
        self.image    = _Struct()

        self.sinogram.radialBinCropfactor = radialBinCropfactor
        self.fov_mask = None
        self.gaps     = None

        self.engine.bar = "\\" if platform == "win32" else "/"

        key = scannerModel.lower()
        if key not in SUPPORTED_SCANNERS:
            raise ValueError(
                f"Unknown scanner model: {scannerModel!r}. "
                f"Supported: {list(SUPPORTED_SCANNERS)}"
            )
        self._computeGantryInfo(SUPPORTED_SCANNERS[key])
        logger.debug("BuildGeometry_v4 initialised: scanner=%s, cropFactor=%.2f",
                     key, radialBinCropfactor)
