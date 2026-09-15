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
    ap.add_argument("--system-config", default=None, metavar="YAML",
                    help="model one specific instrument; see configs/lab_template.yaml")
    ap.add_argument("--jitter-optics", action="store_true",
                    help="with --system-config, also randomise NA, wavelength and "
                         "refractive indices around the given values")
    ap.add_argument("--geometries", nargs="*", default=None,
                    help=f"restrict geometries; any of: {' '.join(sorted(GENERATORS))}")
    ap.add_argument("--families", nargs="*", default=None,
                    help=f"restrict to families: {' '.join(FAMILIES)}")
    ap.add_argument("--aberration-scale", type=float, nargs=2, default=None,
                    metavar=("LOW", "HIGH"), help="range multiplying the Zernike sigmas")
    ap.add_argument("--slab-thickness", type=float, nargs=2, default=None,
                    metavar=("LOW", "HIGH"),
                    help="axial extent of sample kept, um (log-uniform). Focus is "
                         "only well defined for a slab thin enough that the focal "
                         "shift does not spread it over several depths of field; "
                         "run scripts/show_system.py for the limit")
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

    custom = None
    if args.system_config:
        from afocus.optics.config import describe, load_system
        custom = load_system(args.system_config)
        print(describe(custom))
        print()

    kw = {"geometries": tuple(geoms) if geoms else None,
          "custom_system": custom, "lock_optics": not args.jitter_optics}
    if args.aberration_scale:
        kw["aberration_scale"] = tuple(args.aberration_scale)
    if args.slab_thickness:
        kw["slab_thickness"] = tuple(args.slab_thickness)
    rc = RandomisationConfig(**kw)

    # A sample thicker than the focal shift can hold in one depth of field has
    # no single best-focus plane, and the label gates will reject it.  Rejection
    # is correct but it is paid for in rendering time, so say so up front.
    if custom is not None:
        from afocus.optics.psf import PSFEngine
        from afocus.sim.render import FocalShift
        probe = PSFEngine(custom, fov_px=args.fov, pad=2, workers=1, cache_size=2)
        slope = abs(FocalShift(probe, depth_max=10.0).slope)
        if slope > 1e-6:
            limit = 2.0 * custom.depth_of_field / slope
            if rc.slab_thickness[1] > limit:
                print(f"NOTE: sample slabs up to {rc.slab_thickness[1]:.1f} um are being "
                      f"drawn, but on this instrument focus stays defined within "
                      f"2 DoF only up to about {limit:.2f} um "
                      f"(focal-shift slope {slope:.2f} um/um, DoF "
                      f"{custom.depth_of_field * 1000:.0f} nm).")
                print(f"      Thicker scenes will be rejected by the label gates as "
                      f"ambiguous, which costs rendering time. Consider "
                      f"--slab-thickness {max(0.1, limit / 8):.2f} {limit:.2f}")
                print()

    cfg = DatasetConfig(fov_px=args.fov, n_planes=args.planes,
                        scan_span_dof=args.span, randomisation=rc,
                        fft_workers=args.fft_threads)

    print(f"generating {args.scenes} scenes -> {args.out}")
    print(f"  fov={args.fov}px  planes/scene={args.planes}  span=+-{args.span} DoF")
    print(f"  system={custom.name if custom else (args.system or 'randomised')}"
          f"{'' if custom is None else (' (optics locked)' if not args.jitter_optics else ' (optics jittered)')}"
          f"  geometries={geoms or 'all'}")
    generate(args.out, args.scenes, cfg, seed0=args.seed0,
             shard_size=args.shard_size, workers=args.workers,
             system_name=args.system)


if __name__ == "__main__":
    main()
