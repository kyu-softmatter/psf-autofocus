#!/usr/bin/env python3
"""Recover the PSF ring structure from an extended object by edge averaging.

The rings are invisible in a raw frame of an extended sample: neighbouring
emitters' rings overlap, they sit at 1e-3 to 1e-2 of the peak, and camera noise
covers them. Averaging the intensity profile *along the object's boundary*
removes all three obstacles at once -- per-emitter brightness and local sample
structure average away, and the noise falls as sqrt(n_points) over hundreds of
boundary points -- leaving the optics' own response.

What is and is not recovered
----------------------------
The averaging works: the "spread of a single point" band in the ESF row is
barely visible against the 197-point average, and the edge width grows
monotonically with defocus (0.79 -> 1.45 -> 2.57 -> 4.62 um over 0 -> 4 DoF at
20x/0.75), as does the cumulative profile's leakage (0.18 -> 0.73).

But the *ring pattern itself* does not come back, and the LSF row shows why:
compare the recovered LSF against the true PSF profile at 1-4 DoF. The PSF has
strong radial oscillations; the LSF is a single smooth hump. This is arithmetic,
not a bug -- the line spread function is the PSF integrated along the direction
parallel to the edge, and projecting a set of concentric rings onto one axis
superimposes different radii and washes the oscillations out.

So an edge recovers the rings' *consequences* (width, overshoot, MTF zeros) but
not their shape. To see the rings themselves the object has to be a point, not
an edge, which is the encircled-energy route in
:mod:`afocus.features.radial`. The two methods are complementary in what they
measure, not only in the object size they need.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afocus.features import edge as E
from afocus.features import radial as RD
from afocus.optics.psf import PSFEngine
from afocus.optics.system import SampleStack, preset
from afocus.sim import geometry as G
from afocus.sim.camera import Sensor
from afocus.sim.render import FocalShift, Renderer

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="20x_air")
    ap.add_argument("--fov", type=int, default=192)
    ap.add_argument("--radius", type=float, default=8.0,
                    help="object radius, um; must be >> the PSF for an edge to exist")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--exposure", type=float, default=0.02)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # index-matched so the response is symmetric and the rings are clean
    base = preset(args.system)
    sysm = base.evolve(stack=SampleStack(
        n_sample=base.objective.n_immersion, n_glass=base.stack.n_glass,
        n_immersion=base.objective.n_immersion,
        n_glass_design=base.stack.n_glass_design,
        n_immersion_design=base.stack.n_immersion_design))
    eng = PSFEngine(sysm, fov_px=args.fov, pad=2, workers=-1, cache_size=24)
    dof = sysm.depth_of_field
    px = sysm.pixel_size_sample
    rend = Renderer(eng, depth_bin=0.5, focal_shift=FocalShift(eng, depth_max=4.0))

    # A uniform thin *cylinder*, not a flattened sphere.  This matters: a sphere's
    # column density falls off as sqrt(R^2 - r^2) towards the rim, so its rim is
    # not a step and the measured edge width is dominated by the object's own
    # projection rather than the PSF.  Using a squashed sphere here gave a 10-90
    # width of 3.33 um in focus against a true PSF width near 0.3 um -- an order
    # of magnitude of pure geometry.  A cylinder has a genuine step in column
    # density, which is what the slanted-edge method assumes.
    n_em = int(np.pi * args.radius ** 2 * 900)
    ang = rng.uniform(0, 2 * np.pi, n_em)
    rad = args.radius * np.sqrt(rng.uniform(0, 1, n_em))
    thickness = 0.1
    xyz = np.column_stack([rad * np.cos(ang), rad * np.sin(ang),
                           rng.uniform(-thickness / 2, thickness / 2, n_em)])
    disc = G.Emitters(xyz, np.ones(n_em), "uniform_disc",
                      {"radius": args.radius, "thickness": thickness})
    disc = disc.translate([0.0, 0.0, args.depth - float(disc.z.min())])
    splats = rend._splat(disc)
    sensor = Sensor(sysm.camera, (args.fov, args.fov), rng)
    best, _ = rend.best_stage(disc)
    print(f"{sysm.name}: DoF {dof:.3f} um, disc radius {args.radius} um "
          f"({args.radius / sysm.abbe_resolution:.0f} x the resolution limit), "
          f"N={len(disc)} emitters, best focus {best:+.3f} um")

    dzs = [0.0, 1.0, 2.0, 4.0]
    fig, axes = plt.subplots(4, len(dzs), figsize=(3.05 * len(dzs), 10.6))

    for j, d in enumerate(dzs):
        flux = rend.render_flux(disc, best + d * dof, splats=splats)
        img = sensor.expose(flux, args.exposure,
                            background=sysm.illumination.background).astype(np.float64)

        # --- row 0: the raw frame, linear, as a detector delivers it ---
        ax = axes[0, j]
        ax.imshow(img, cmap="magma", vmin=np.percentile(img, 1),
                  vmax=np.percentile(img, 99.7))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"defocus {d:g} DoF", fontsize=10)
        if j == 0:
            ax.set_ylabel("raw camera frame\n(rings invisible)", fontsize=8.5)

        prof = E.extract(img, px, half_width=4.0 * max(dof, 0.4), n_samples=161,
                         max_curvature_um=args.radius * 0.5)
        if not prof.valid:
            for r in (1, 2, 3):
                axes[r, j].text(0.5, 0.5, f"no edge:\n{prof.reason}", fontsize=7,
                                ha="center", va="center", transform=axes[r, j].transAxes)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            continue

        # --- row 1: the averaged ESF, with the single-point spread for contrast ---
        ax = axes[1, j]
        ax.plot(prof.offsets, prof.esf, lw=1.6, color="#C44E52",
                label=f"averaged over {prof.n_points} edge points")
        ax.fill_between(prof.offsets, prof.esf - prof.esf_std,
                        prof.esf + prof.esf_std, color="#C44E52", alpha=0.18,
                        label="spread of a single point")
        ax.axhline(0, color="k", lw=0.5, ls=":"); ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.set_xlabel("distance from edge (um)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.22)
        if j == 0:
            ax.set_ylabel("edge spread function\n(averaging kills the noise)", fontsize=8.5)
            ax.legend(fontsize=6.5, loc="upper left")

        # --- row 2: the LSF against the true PSF radial profile ---
        ax = axes[2, j]
        lsf = prof.lsf / max(np.abs(prof.lsf).max(), 1e-30)
        ax.plot(prof.offsets, lsf, lw=1.6, color="#4C72B0", label="LSF from the frame")
        psf = eng.psf_fine(best_rel := d * dof, args.depth)
        mid = psf.shape[0] // 2
        r_psf = (np.arange(psf.shape[0]) - mid) * eng.dx
        ax.plot(r_psf, psf[mid] / max(psf[mid].max(), 1e-30), lw=1.1, ls="--",
                color="#55A868", label="true PSF profile")
        ax.set_xlim(prof.offsets[0], prof.offsets[-1])
        ax.set_xlabel("distance (um)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.22)
        if j == 0:
            ax.set_ylabel("line spread function\nvs true PSF: rings are\nlost to the line integral",
                          fontsize=8)
            ax.legend(fontsize=6.5)

        # --- row 3: the cumulative profile P(r) ---
        ax = axes[3, j]
        p_cum, desc = RD.cumulative_edge_profile(prof.offsets, prof.esf)
        ax.plot(prof.offsets, p_cum, lw=1.6, color="#8172B3")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("distance from edge (um)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.22)
        ax.set_title(f"leak {desc['leak_total']:.3f}   "
                     f"fill deficit {desc['fill_deficit']:.3f}", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("cumulative P(r) = $\\int_0^r$ I dr\n(um)", fontsize=8.5)
        print(f"  {d:g} DoF: {prof.n_points:4d} edge points, "
              f"w10-90 {prof.descriptors['width_10_90']:.3f} um, "
              f"overshoot {prof.descriptors['overshoot_bright']:+.4f}, "
              f"leak {desc['leak_total']:.3f}, fill deficit {desc['fill_deficit']:.3f}")

    fig.suptitle(
        f"What edge averaging recovers from an extended object -- and what it does not\n"
        f"{sysm.name}, disc of radius {args.radius} um at depth {args.depth} um, "
        f"index matched, with camera noise", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out / "edge_rings.png", dpi=145)
    print(f"\nwrote {out / 'edge_rings.png'}")


if __name__ == "__main__":
    main()
