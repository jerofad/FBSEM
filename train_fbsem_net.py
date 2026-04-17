"""
Train a 2D FBSEM network.

Usage:
    python train_fbsem_net.py --config configs/train_2d.yaml
    python train_fbsem_net.py --config configs/train_2d.yaml --device cuda --epochs 100

All parameters live in the YAML config. CLI flags override individual values.
"""

import argparse
import os
import yaml
import numpy as np
import torch

from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import PETMrDataset, dotstruct, toNumpy, crop
from models.modellib import FBSEMnet_v3, Trainer, fbsemInference


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _dict_to_dotstruct(d):
    g = dotstruct()
    for k, v in d.items():
        setattr(g, k, v)
    return g


def main():
    parser = argparse.ArgumentParser(description="Train FBSEM network")
    parser.add_argument("--config", default="configs/train_2d.yaml",
                        help="Path to YAML config file")
    # Per-run overrides — anything in the YAML can also be overridden here
    parser.add_argument("--device",          default=None)
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--save_dir",        default=None)
    parser.add_argument("--model_name",      default=None)
    parser.add_argument("--system_matrix_path", default=None)
    parser.add_argument("--training_data_dir",  default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Apply any CLI overrides
    for key in ("device", "epochs", "lr", "save_dir", "model_name",
                "system_matrix_path", "training_data_dir"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    # Validate required paths
    for field in ("system_matrix_path", "training_data_dir", "save_dir"):
        if cfg.get(field) in (None, "/path/to/system_matrix",
                              "/path/to/training_data", "/path/to/output/"):
            raise ValueError(
                f"Config field '{field}' must be set to a real path. "
                f"Edit configs/train_2d.yaml or pass --{field} on the command line."
            )

    g = _dict_to_dotstruct(cfg)

    # ── PET geometry ──────────────────────────────────────────────────────────
    PET = BuildGeometry_v4(g.scanner, g.radial_bin_crop_factor)
    PET.loadSystemMatrix(g.system_matrix_path, is3d=g.is_3d)

    # ── Data loaders ──────────────────────────────────────────────────────────
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
        augment=getattr(g, 'augment', False),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FBSEMnet_v3(
        g.depth, g.num_kernels, g.kernel_size,
        g.in_channels, g.is_3d, g.reg_cnn_model,
    ).to(g.device, dtype=torch.float32)

    # ── Train ─────────────────────────────────────────────────────────────────
    # results = Trainer(PET, model, g, train_loader, valid_loader)

    # ── Test-set inference (optional) ─────────────────────────────────────────
    dl_model_flname = os.path.join(g.save_dir, g.model_name + f"-epo-{g.epochs - 1}.pth")
    if not os.path.isfile(dl_model_flname):
        print(f"Trained model not found at {dl_model_flname}, skipping inference.")
        return

    sinoLD, imgHD, AN, RS, imgLD, imgLD_psf, mrImg, counts, imgGT, _ = next(iter(test_loader))
    img_fbsem = fbsemInference(dl_model_flname, PET, sinoLD, AN, mrImg,
                               niters=g.niters, nsubs=g.nsubs)

    sinoLD   = toNumpy(sinoLD)
    imgHD    = toNumpy(imgHD)
    AN       = toNumpy(AN)
    imgLD    = toNumpy(imgLD)
    imgLD_psf = toNumpy(imgLD_psf)
    mrImg    = toNumpy(mrImg)
    imgGT    = toNumpy(imgGT)

    img_mapem = PET.mrMAPEM2DBatch(sinoLD, AN, mrImg, beta=0.06,
                                   niters=g.niters, nsubs=g.nsubs, psf=g.psf_cm)

    # ── Visualise ─────────────────────────────────────────────────────────────
    from matplotlib import pyplot as plt

    cf = g.crop_factor
    num_batches = sinoLD.shape[0]
    fig, ax = plt.subplots(num_batches, 5, figsize=g.disp_figsize, sharex=True, sharey=True)
    if num_batches == 1:
        ax = ax[None]  # ensure 2-D indexing

    titles = ["MR", "HD (target)", "LD+PSF", "MAPEM", "FBSEM"]
    for i in range(num_batches):
        for j, (col_img, title) in enumerate(zip(
            [mrImg[i], imgHD[i], imgLD_psf[i], img_mapem[i], img_fbsem[i]], titles
        )):
            ax[i, j].imshow(crop(col_img, cf), cmap="gist_gray_r")
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(title, fontsize=20)

    fig.subplots_adjust(hspace=0, wspace=0)
    out_path = os.path.join(g.save_dir, "test_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved comparison figure to {out_path}")


if __name__ == "__main__":
    main()
