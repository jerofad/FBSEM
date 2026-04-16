"""
Build FBSEM training datasets from BrainWeb phantoms.

Usage:
    python build_training_sets.py --config configs/build_dataset.yaml
    python build_training_sets.py --config configs/build_dataset.yaml \\
        --phantom_data_dir /data/brainweb --save_training_dir /data/output

All parameters live in the YAML config. CLI flags override individual values.
"""

import argparse
import yaml
import numpy as np

from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import buildBrainPhantomDataset


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build FBSEM training datasets")
    parser.add_argument("--config", default="configs/build_dataset.yaml",
                        help="Path to YAML config file")
    # Per-run overrides
    parser.add_argument("--system_matrix_path", default=None)
    parser.add_argument("--phantom_data_dir",   default=None)
    parser.add_argument("--save_training_dir",  default=None)
    parser.add_argument("--phantom_numbers",    nargs="+", type=int, default=None,
                        help="Phantom indices to use, e.g. --phantom_numbers 0 1 2 3 4")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Apply CLI overrides
    for key in ("system_matrix_path", "phantom_data_dir", "save_training_dir"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.phantom_numbers is not None:
        cfg["phantom_numbers"] = args.phantom_numbers

    # Validate required paths
    for field in ("system_matrix_path", "phantom_data_dir", "save_training_dir"):
        if cfg.get(field) in (None, "/path/to/system_matrix",
                              "/path/to/brainweb_raws", "/path/to/output/"):
            raise ValueError(
                f"Config field '{field}' must be set to a real path. "
                f"Edit configs/build_dataset.yaml or pass --{field} on the command line."
            )

    # ── PET geometry ──────────────────────────────────────────────────────────
    PET = BuildGeometry_v4(cfg["scanner"], cfg["radial_bin_crop_factor"])
    PET.loadSystemMatrix(cfg["system_matrix_path"], is3d=cfg["is_3d"])

    print("PET scanner:", cfg["scanner"])
    print("is_3d:", cfg["is_3d"])
    print("Saving datasets to:", cfg["save_training_dir"])

    # ── Phantom numbers ───────────────────────────────────────────────────────
    phantom_numbers = cfg.get("phantom_numbers")
    if phantom_numbers is not None:
        phantom_numbers = np.array(phantom_numbers)

    # ── Build datasets ────────────────────────────────────────────────────────
    slices_2d = cfg.get("slices_2d")
    if slices_2d is not None:
        slices_2d = np.array(slices_2d)

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

    print("Done.")


if __name__ == "__main__":
    main()
