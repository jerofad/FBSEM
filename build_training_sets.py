"""
Build FBSEM training datasets from BrainWeb digital brain phantoms.

For each phantom subject and each random in-plane rotation the script:
  1. Loads (and auto-downloads) the BrainWeb .raws file.
  2. Simulates a high-dose and a low-dose PET sinogram.
  3. Reconstructs both with OSEM to produce reference (HD) and input (LD, LD+PSF) images.
  4. Saves all arrays as a single .npy dict consumed by DatasetPetMr_v2.

Usage
-----
    python build_training_sets.py --config configs/build_dataset.yaml
    python build_training_sets.py --config configs/build_dataset.yaml \\
        --phantom_data_dir /data/brainweb --save_training_dir /data/output

All parameters live in the YAML config.  CLI flags override individual values.

Note
----
This script can take several hours (each phantom × rotation × 3 reconstructions
runs CPU-bound OSEM).  Consider running with ``nohup`` or in a ``screen`` session.
"""

import argparse
import logging
import sys

import numpy as np
import yaml

from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import buildBrainPhantomDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FBSEM training datasets from BrainWeb phantoms"
    )
    parser.add_argument("--config", default="configs/build_dataset.yaml",
                        help="Path to YAML config file")
    # Per-run CLI overrides
    parser.add_argument("--system_matrix_path", default=None,
                        help="Directory containing the scanner system-matrix files")
    parser.add_argument("--phantom_data_dir",   default=None,
                        help="Directory containing (or to download) subject_XX.raws files")
    parser.add_argument("--save_training_dir",  default=None,
                        help="Output directory for data-N.npy training files")
    parser.add_argument("--phantom_numbers",    nargs="+", type=int, default=None,
                        help="Phantom subject indices, e.g. --phantom_numbers 0 1 2")
    parser.add_argument("--log_level",          default="INFO",
                        help="Logging verbosity: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    logger.info("Loading config: %s", args.config)
    cfg = _load_config(args.config)

    # Apply CLI overrides
    for key in ("system_matrix_path", "phantom_data_dir", "save_training_dir"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
            logger.debug("CLI override: %s = %s", key, val)
    if args.phantom_numbers is not None:
        cfg["phantom_numbers"] = args.phantom_numbers

    # Fail early with a clear message if required paths were not configured
    _PLACEHOLDERS = {None, "/path/to/system_matrix",
                     "/path/to/brainweb_raws", "/path/to/output/"}
    for field in ("system_matrix_path", "phantom_data_dir", "save_training_dir"):
        if cfg.get(field) in _PLACEHOLDERS:
            raise ValueError(
                f"Config field '{field}' must be set to a real path.\n"
                f"  Edit configs/build_dataset.yaml or pass --{field} on the command line."
            )

    # ── PET geometry ──────────────────────────────────────────────────────────
    logger.info("Building PET geometry (scanner=%s, is_3d=%s) ...",
                cfg["scanner"], cfg["is_3d"])
    PET = BuildGeometry_v4(cfg["scanner"], cfg["radial_bin_crop_factor"])
    PET.loadSystemMatrix(cfg["system_matrix_path"], is3d=cfg["is_3d"])

    # ── Phantom selection ─────────────────────────────────────────────────────
    phantom_numbers = cfg.get("phantom_numbers")
    if phantom_numbers is not None:
        phantom_numbers = np.array(phantom_numbers)
        logger.info("Phantom subjects: %s", phantom_numbers.tolist())
    else:
        logger.info("Phantom subjects: all 20 (default)")

    slices_2d = cfg.get("slices_2d")
    if slices_2d is not None:
        slices_2d = np.array(slices_2d)

    logger.info("Output directory: %s", cfg["save_training_dir"])
    logger.info(
        "Rotations per phantom: %d | LD count window (2D): %s",
        cfg["num_rand_rotations"], cfg["count_ld_window_2d"],
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    buildBrainPhantomDataset(
        PET,
        save_training_dir=cfg["save_training_dir"],
        phanPath=cfg["phantom_data_dir"],
        phanType=cfg["phantom_type"],
        phanNumber=phantom_numbers,
        is3d=cfg["is_3d"],
        num_rand_rotations=cfg["num_rand_rotations"],
        rot_angle_degrees=cfg["rot_angle_degrees"],
        psf_hd=cfg["psf_hd_cm"],
        psf_ld=cfg["psf_ld_cm"],
        niter_hd=cfg["niter_hd"],
        niter_ld=cfg["niter_ld"],
        nsubs_hd=cfg["nsubs_hd"],
        nsubs_ld=cfg["nsubs_ld"],
        counts_hd=cfg["counts_hd"],
        count_ld_window_3d=cfg["count_ld_window_3d"],
        count_ld_window_2d=cfg["count_ld_window_2d"],
        slices_2d=slices_2d,
        pet_lesion=cfg["add_pet_lesion"],
        t1_lesion=cfg["add_t1_lesion"],
        num_lesions=cfg["num_lesions"],
        lesion_size_mm=cfg["lesion_size_mm"],
        hot_cold_ratio=cfg["hot_cold_ratio"],
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
