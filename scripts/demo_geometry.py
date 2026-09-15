#!/usr/bin/env python3
"""Render every sample geometry through the same optics and compare focus curves.

Produces two figures:

``geometry_gallery.png``
    Each geometry at best focus and at two defocus levels, so the out-of-focus
    signature of each shape is visible side by side.
``focus_curves.png``
    Classical focus metrics versus stage position for each geometry.  The
    interesting content is where the curves are *flat*, *double-peaked* or
    *offset* from the true focus -- those are the cases a scalar z-scan gets
    wrong and a learned model has to handle.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afocus.optics.psf import PSFEngine
from afocus.optics.system import preset
from afocus.sim import geometry as G
from afocus.sim import metrics as M
from afocus.sim.camera import Sensor
from afocus.sim.render import FocalShift, Renderer

warnings.filterwarnings("ignore", category=RuntimeWarning)

GEOMS = [
    ("solid_sphere",       dict(radius=0.8,  density=2000.)),
    ("hollow_shell",       dict(radius=1.6,  thickness=0.15, density=3000.)),
    ("sphere_size_series", dict(radii=(0.15, 0.3, 0.6, 1.2, 2.4), spacing=7.0, density=1500.)),
    ("ellipsoid",          dict(semi_axes=(2.0, 0.6, 0.6), density=2000., orient=False)),
    ("superellipsoid",     dict(semi_axes=(1.0, 1.0, 1.0), e1=0.15, e2=0.15, density=2000., orient=False)),
    ("rod",                dict(length=5.0, radius=0.3, density=2500., orient=False)),
    ("lumpy_particle",     dict(radius=1.5, roughness=0.35, n_modes=7, density=1500.)),
    ("raspberry_cluster",  dict(n_sub=14, r_sub=0.45, r_core=1.4, density=2000.)),
    ("fractal_aggregate",  dict(n_sub=50, r_sub=0.35, mode="ballistic", density=1200.)),
    ("fractal_aggregate",  dict(n_sub=70, r_sub=0.3,  mode="chain", density=1200.)),
    ("worm_like_chains",   dict(n_chains=30, length=25., persistence=6., radius=0.08,
                                extent=(16., 16., 0.6), lin_density=500.)),
    ("strut_network",      dict(n_nodes=70, k_neighbours=3, radius=0.15,
                                extent=(16., 16., 0.8), lin_density=400.)),
    ("spinodal_texture",   dict(extent=(16., 16., 0.8), correlation=1.6, fill=0.4, density=300.)),
    ("point_emitters",     dict(n=60, extent=(16., 16., 0.1))),
    ("thin_sheet",         dict(extent=(16., 16.), tilt=0.0, thickness=0.3, density=200.)),
    ("thin_sheet",         dict(extent=(16., 16.), tilt=6.0, thickness=0.3, density=200.)),
]


def label_for(name: str, kw: dict) -> str:
    if name == "fractal_aggregate":
        return f"aggregate ({kw['mode']})"
    if name == "thin_sheet":
        return f"thin sheet (tilt {kw['tilt']:.0f} deg)"
    return name.replace("_", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="20x_air")
    ap.add_argument("--fov", type=int, default=128)
    ap.add_argument("--depth", type=float, default=1.5, help="sample depth above the coverglass, um")
    ap.add_argument("--nz", type=int, default=31)
    ap.add_argument("--span", type=float, default=6.0, help="stage scan half-range, in DoF")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--noise", action="store_true", help="show camera frames instead of noiseless flux")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sysm = preset(args.system)
    eng = PSFEngine(sysm, fov_px=args.fov, pad=2)
    dof = sysm.depth_of_field
    print(f"{sysm.name}: dx={sysm.pixel_size_sample:.4f} um  FOV={args.fov * sysm.pixel_size_sample:.1f} um  "
          f"DoF={dof:.3f} um  res={sysm.abbe_resolution:.3f} um")

    t0 = time.time()
    shift = FocalShift(eng, depth_max=max(3.0 * args.depth, 4.0), n_knots=4)
    rend = Renderer(eng, depth_bin=0.5, focal_shift=shift)
    print(f"focal-shift model built in {time.time() - t0:.1f}s; shift({args.depth:.1f} um) = {shift(args.depth):+.3f} um")

    rng = np.random.default_rng(args.seed)
    span = args.span * dof
    stage_grid = np.linspace(-span, span, args.nz)

    rows = []
    for name, kw in GEOMS:
        em = G.build(name, rng, **kw)
        em = em.translate([0.0, 0.0, args.depth - float(em.z.min())])
        t = time.time()
        best, _ = rend.best_stage(em, metric="brenner")
        stack = rend.render_stack(em, best + stage_grid)
        rows.append(dict(name=name, kw=kw, label=label_for(name, kw), em=em,
                         best=best, stack=stack))
        print(f"  {label_for(name, kw):26s} N={len(em):7d}  best_stage={best:+7.3f} um  "
              f"(depth-only prediction {shift(args.depth):+.3f})  [{time.time() - t:.1f}s]")

    # ---------------- figure 1: gallery ----------------
    picks = [0.0, 1.5 * dof, 5.0 * dof]
    ncol = len(picks) + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.05 * ncol, 1.95 * len(rows)))
    sensor = Sensor(sysm.camera, (args.fov, args.fov), np.random.default_rng(0))
    for i, row in enumerate(rows):
        em = row["em"]
        ax = axes[i, 0]
        ax.scatter(em.x, em.y, s=0.12, c=em.z, cmap="viridis", linewidths=0)
        lim = args.fov * sysm.pixel_size_sample / 2
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(row["label"], fontsize=7.5, rotation=0, ha="right", va="center")
        if i == 0:
            ax.set_title("emitters (colour = depth)", fontsize=8)
        for j, dz in enumerate(picks):
            k = int(np.argmin(np.abs(stage_grid - dz)))
            img = row["stack"][k]
            if args.noise:
                img = sensor.expose(img, exposure=0.02, background=sysm.illumination.background)
            a = axes[i, j + 1]
            a.imshow(img, cmap="magma", vmin=0, vmax=np.percentile(row["stack"][0], 99.8))
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(f"defocus {dz / dof:+.1f} DoF", fontsize=8)
    fig.suptitle(f"Sample geometries through {sysm.name}  "
                 f"(depth {args.depth} um, DoF {dof:.2f} um)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out / "geometry_gallery.png", dpi=135)
    print(f"\nwrote {out / 'geometry_gallery.png'}")

    # ---------------- figure 2: focus curves ----------------
    mets = ["brenner", "laplacian", "normalised_variance", "hf_ratio", "dct_entropy", "vollath4"]
    ncol = 4
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow), squeeze=False)
    summary = []
    for i, row in enumerate(rows):
        ax = axes[i // ncol][i % ncol]
        for m in mets:
            c = M.focus_curve(row["stack"], m)
            c = (c - c.min()) / max(c.max() - c.min(), 1e-30)
            ax.plot(stage_grid / dof, c, lw=1.1, label=m)
            summary.append((row["label"], m, M.argmax_parabolic(stage_grid, c) / dof))
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_title(row["label"], fontsize=8.5)
        ax.set_xlabel("defocus (DoF)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, ncol=2)
    for k in range(len(rows), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Normalised focus metrics vs defocus  (0 = label, dotted line)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "focus_curves.png", dpi=135)
    print(f"wrote {out / 'focus_curves.png'}")

    # ---------------- table: metric bias per geometry ----------------
    print(f"\nmetric peak offset from the label, in units of DoF "
          f"(step size {np.diff(stage_grid)[0] / dof:.2f} DoF):")
    print(f"{'geometry':26s} " + " ".join(f"{m[:11]:>12s}" for m in mets))
    worst = []
    for row in rows:
        vals = [o for (lab, m, o) in summary if lab == row["label"]]
        print(f"{row['label']:26s} " + " ".join(f"{v:+12.2f}" for v in vals))
        worst.append((max(abs(v) for v in vals), row["label"]))
    worst.sort(reverse=True)
    print("\nlargest metric disagreement (DoF):", ", ".join(f"{l} {v:.1f}" for v, l in worst[:5]))


if __name__ == "__main__":
    main()
