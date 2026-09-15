#!/usr/bin/env python3
"""Compare focus policies on freshly rendered scenes.

This is the end-to-end test: not "how accurate is the regressor" but "how close
to focus does the microscope end up, and how many frames did it cost".  Those
are different questions -- a policy can waste a good estimator, and a good
policy can rescue a mediocre one.

Classical metric-only policies need no model, so they always run.  Model-based
policies run if ``--checkpoint`` is given; with ``--oracle`` they run against a
ground-truth estimator with configurable noise instead, which separates policy
failures from model failures.

Examples
--------
    python scripts/evaluate.py --scenes 40 --oracle 0.5
    python scripts/evaluate.py --scenes 40 --checkpoint runs/image/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afocus.optics.psf import PSFEngine
from afocus.search import policies as P
from afocus.sim import metrics as M
from afocus.sim.camera import Sensor, auto_exposure
from afocus.sim.dataset import DatasetConfig
from afocus.sim.render import FocalShift, Renderer
from afocus.sim.scene import RandomisationConfig, Stage, draw_scene, draw_system

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build_scene(seed: int, fov: int, rc: RandomisationConfig, system: str | None):
    rng = np.random.default_rng(seed)
    probe_sys, _ = draw_system(np.random.default_rng(seed), rc, system)
    probe = PSFEngine(probe_sys, fov_px=fov, pad=2, workers=1)
    grid_um = probe.n_grid * probe.dx
    fov_um = fov * probe_sys.pixel_size_sample
    scene = draw_scene(rng, rc, fov, grid_um, fov_um, (fov, fov), system)
    if scene.is_empty or len(scene.emitters) == 0:
        return None
    eng = PSFEngine(scene.system, fov_px=fov, pad=2,
                    aberration=scene.aberration, workers=1)
    rend = Renderer(eng, depth_bin=0.5,
                    focal_shift=FocalShift(eng, depth_max=20.0),
                    illumination_field=scene.illumination_field)
    return scene, eng, rend, rng


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=900_000,
                    help="offset from the training seeds, so scenes are unseen")
    ap.add_argument("--fov", type=int, default=128)
    ap.add_argument("--system", default=None)
    ap.add_argument("--start-offset", type=float, default=6.0,
                    help="initial defocus, in DoF")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--oracle", type=float, default=None,
                    help="use a ground-truth estimator with this noise, in DoF")
    ap.add_argument("--oracle-sign-error", type=float, default=0.05)
    ap.add_argument("--out", default="outputs/policy_comparison.json")
    args = ap.parse_args()

    rc = RandomisationConfig()
    results = defaultdict(list)
    used = 0

    for k in range(args.scenes * 3):
        if used >= args.scenes:
            break
        built = build_scene(args.seed0 + k, args.fov, rc, args.system)
        if built is None:
            continue
        scene, eng, rend, rng = built
        em = scene.emitters
        splats = rend._splat(em)
        if not splats:
            continue
        dof = scene.system.depth_of_field
        best, info = rend.best_stage(em)
        if not np.isfinite(best):
            continue
        used += 1

        flux0 = rend.render_flux(em, best, splats=splats)
        if flux0.max() <= 0:
            continue
        exposure = auto_exposure(flux0, scene.system.camera,
                                 target_fill=scene.exposure_fill)
        sensor = Sensor(scene.system.camera, (args.fov, args.fov), rng)
        holder = {}

        def acquire(z: float) -> np.ndarray:
            achieved = holder["stage"].move_to(z)
            f = rend.render_flux(em, achieved, splats=splats)
            return sensor.expose(f, exposure,
                                 background=scene.system.illumination.background
                                 ).astype(np.float64)

        estimator = None
        if args.checkpoint:
            from afocus.models.estimator import ModelEstimator
            estimator = ModelEstimator(args.checkpoint, scene.system)
        elif args.oracle is not None:
            from afocus.models.estimator import OracleEstimator
            estimator = OracleEstimator(best, sigma_um=args.oracle * dof,
                                        sign_error_rate=args.oracle_sign_error, rng=rng)

        tests = [
            ("full_scan", P.FullScan(span=8 * dof, n_planes=21)),
            ("coarse_to_fine", P.CoarseToFine(span=8 * dof, n_coarse=9, n_fine=7)),
            ("hill_climb", P.HillClimb(step=2 * dof, min_step=0.05 * dof,
                                       max_frames=30, travel=25 * dof)),
        ]
        if estimator is not None:
            tests += [
                ("single_shot", P.SingleShot(estimator, tolerance=0.15 * dof)),
                ("iterative", P.Iterative(estimator, max_iters=5, tolerance=0.1 * dof)),
                ("dual_plane", P.DualPlane(estimator, offset=1.5 * dof,
                                           tolerance=0.1 * dof)),
                ("model_guided_scan", P.ModelGuidedScan(estimator, span=6 * dof,
                                                        n_planes=5)),
            ]

        start = best + args.start_offset * dof
        for name, pol in tests:
            holder["stage"] = Stage(rng, jitter=scene.stage_jitter,
                                    backlash=scene.stage_backlash, position=start)
            try:
                res = pol.run(acquire, start=start)
            except RuntimeError:
                continue
            results[name].append({
                "err_dof": abs(res.error(best)) / dof,
                "signed_dof": res.error(best) / dof,
                "frames": res.n_frames, "converged": bool(res.converged),
                "geometry": scene.params["geometry"], "dof_um": dof,
                "runaway": bool(res.info.get("runaway", False)),
            })
        print(f"  scene {used}/{args.scenes}: {scene.params['geometry']:20s} "
              f"DoF={dof:.3f}um", flush=True)

    print(f"\n=== {used} scenes, start {args.start_offset} DoF from focus ===")
    print(f"{'policy':20s} {'median':>8s} {'p90':>8s} {'mean nm':>9s} "
          f"{'frames':>7s} {'conv':>6s} {'runaway':>8s}")
    summary = {}
    for name, rows in results.items():
        e = np.array([r["err_dof"] for r in rows])
        f = np.array([r["frames"] for r in rows])
        nm = np.array([r["err_dof"] * r["dof_um"] for r in rows]) * 1000
        conv = np.mean([r["converged"] for r in rows])
        run = np.mean([r["runaway"] for r in rows])
        summary[name] = {"n": len(rows), "median_dof": float(np.median(e)),
                         "p90_dof": float(np.percentile(e, 90)),
                         "mean_nm": float(nm.mean()),
                         "mean_frames": float(f.mean()),
                         "converged": float(conv), "runaway": float(run)}
        print(f"{name:20s} {np.median(e):8.3f} {np.percentile(e, 90):8.3f} "
              f"{nm.mean():9.0f} {f.mean():7.1f} {conv:6.2f} {run:8.2f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary,
                               "per_scene": {k: v for k, v in results.items()},
                               "args": vars(args)}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
