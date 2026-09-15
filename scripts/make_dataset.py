#!/usr/bin/env python3
"""Generate a synthetic autofocus dataset.

Each scene contributes one focal series; every plane is one record.  Scenes are
independent, so this parallelises cleanly over cores.

Examples
--------
    python scripts/make_dataset.py --out data/train --scenes 800
    python scripts/make_dataset.py --out data/spheres --scenes 200 \
        --geometries solid_sphere hollow_shell sphere_size_series
    python scripts/make_dataset.py --out data/60x --scenes 200 --system 60x_oil
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afocus.sim.dataset import DatasetConfig, generate
from afocus.sim.geometry import FAMILIES, GENERATORS
from afocus.sim.scene import RandomisationConfig
from afocus.optics.system import PRESETS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--fov", type=int, default=128, help="field of view, camera pixels")
    ap.add_argument("--planes", type=int, default=11, help="training planes per scene")
    ap.add_argument("--span", type=float, default=14.0, help="scan half-range, in DoF")
    ap.add_argument("--workers", type=int, default=None, help="scene-level processes")
    ap.add_argument("--fft-threads", type=int, default=1,
                    help="FFT threads per scene; keep at 1 when workers > 1")
    ap.add_argument("--shard-size", type=int, default=32, help="scenes per shard")
    ap.add_argument("--system", default=None, choices=list(PRESETS),
                    help="fix the objective instead of randomising over presets")
    ap.add_argument("--geometries", nargs="*", default=None,
                    help=f"restrict geometries; any of: {' '.join(sorted(GENERATORS))}")
    ap.add_argument("--families", nargs="*", default=None,
                    help=f"restrict to families: {' '.join(FAMILIES)}")
    ap.add_argument("--aberration-scale", type=float, nargs=2, default=None,
                    metavar=("LOW", "HIGH"), help="range multiplying the Zernike sigmas")
    args = ap.parse_args()

    geoms = args.geometries
    if args.families:
        unknown = set(args.families) - set(FAMILIES)
        if unknown:
            ap.error(f"unknown families: {sorted(unknown)}")
        from_fams = [g for f in args.families for g in FAMILIES[f]]
        geoms = sorted(set(geoms or []) | set(from_fams))
    if geoms:
        unknown = set(geoms) - set(GENERATORS)
        if unknown:
            ap.error(f"unknown geometries: {sorted(unknown)}")

    rc = RandomisationConfig(geometries=tuple(geoms) if geoms else None)
    if args.aberration_scale:
        rc = RandomisationConfig(geometries=rc.geometries,
                                 aberration_scale=tuple(args.aberration_scale))

    cfg = DatasetConfig(fov_px=args.fov, n_planes=args.planes,
                        scan_span_dof=args.span, randomisation=rc,
                        fft_workers=args.fft_threads)

    print(f"generating {args.scenes} scenes -> {args.out}")
    print(f"  fov={args.fov}px  planes/scene={args.planes}  span=+-{args.span} DoF")
    print(f"  system={args.system or 'randomised'}  geometries={geoms or 'all'}")
    generate(args.out, args.scenes, cfg, seed0=args.seed0,
             shard_size=args.shard_size, workers=args.workers,
             system_name=args.system)


if __name__ == "__main__":
    main()
