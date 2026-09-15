#!/usr/bin/env python3
"""Train a defocus model.

Three model kinds, deliberately comparable:

    --kind image    ResNet on the frame.  Most capacity, most exposed to
                    learning the synthetic sample's statistics.
    --kind edge     1-D CNN on geometry-invariant physics features (edge spread
                    function, its integral, encircled energy).  Should transfer
                    across sample geometries; blind where no boundary or spot
                    is resolvable.
    --kind hybrid   Both, with the edge branch gated by its own validity.

``--hold-out`` is the experiment that matters: train with one geometry family
excluded, then test on it.  A model that learned the optics keeps its accuracy;
one that memorised sample statistics does not.

Examples
--------
    python scripts/train.py --data data/train --kind image --epochs 20
    python scripts/train.py --data data/train --kind edge --hold-out networks
    python scripts/train.py --data data/train --kind hybrid --out runs/hybrid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afocus.models.preprocess import AugmentConfig
from afocus.sim.dataset import ShardedArrays
from afocus.sim.geometry import FAMILIES
from afocus.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kind", default="image", choices=("image", "edge", "hybrid"))
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bins", type=int, default=81)
    ap.add_argument("--span", type=float, default=16.0, help="defocus bin range, in DoF")
    ap.add_argument("--target-sigma", type=float, default=0.8,
                    help="width of the soft defocus target, in DoF")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-out", nargs="*", default=[],
                    help=f"geometry families to exclude from training: {' '.join(FAMILIES)}")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--all-frames", action="store_true",
                    help="for edge/hybrid, keep frames with no physics features "
                         "(they will have an all-zero input)")
    args = ap.parse_args()

    unknown = set(args.hold_out) - set(FAMILIES)
    if unknown:
        ap.error(f"unknown families: {sorted(unknown)}")

    arrays = ShardedArrays(args.data)
    print(arrays.summary())

    cfg = TrainConfig(
        kind=args.kind, arch=args.arch, pretrained=not args.no_pretrained,
        image_size=args.image_size, n_bins=args.bins, span_dof=args.span,
        target_sigma=args.target_sigma, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, num_workers=args.workers,
        device=args.device, seed=args.seed,
        held_out_families=tuple(args.hold_out),
        require_features=False if args.all_frames else "auto",
        augment=AugmentConfig(flip=False, rot90=False, crop=0.0, extra_noise=0.0,
                              intensity_jitter=0.0) if args.no_augment
        else AugmentConfig(flip=True, rot90=True, crop=0.06,
                           extra_noise=0.01, intensity_jitter=0.1),
    )
    out = Path(args.out) if args.out else Path("runs") / (
        args.kind + ("_hold_" + "_".join(args.hold_out) if args.hold_out else ""))
    print(f"\ntraining kind={args.kind} -> {out}")
    if args.hold_out:
        print(f"  holding out families: {args.hold_out} (cross-geometry test)")

    res = train(args.data, cfg, out, arrays=arrays)
    print(f"\nframes used: train={res['n_train']} val={res['n_val']} test={res['n_test']}"
          + ("  (restricted to frames with physics features)" if res["require_features"] else ""))

    print("\n=== test set ===")
    o = res["test"]["overall"]
    for k in ("n", "mae_dof", "median_dof", "p90_dof", "median_nm", "within_1dof",
              "within_0.25dof", "sign_accuracy", "coverage_1sigma", "validity_auc"):
        if k in o:
            print(f"  {k:18s} {o[k]:.4f}" if isinstance(o[k], float) else f"  {k:18s} {o[k]}")
    if "family" in res["test"]:
        print("\n  per family (MAE DoF / sign acc / n):")
        for fam, m in sorted(res["test"]["family"].items()):
            print(f"    {fam:14s} {m.get('mae_dof', float('nan')):7.3f}  "
                  f"{m.get('sign_accuracy', float('nan')):6.3f}  {m.get('n', 0):6d}")
    print(f"\nwrote {out / 'result.json'}")


if __name__ == "__main__":
    main()
