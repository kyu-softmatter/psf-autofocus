#!/usr/bin/env python3
"""Look at the PSF itself: rings, axial structure, aberrations.

The ring structure is invisible in the dataset frames for three reasons, all of
them expected: the samples are extended objects rather than point sources, so
rings from neighbouring emitters overlap; the rings sit at 1e-3 to 1e-2 of the
peak, which a linear display crushes; and camera noise covers them. Shown here
on a single point emitter, noiseless, on a log scale, they are unmistakable.

This matters beyond being pretty. The axial *asymmetry* visible in the
mismatched-index panels is the only reason the sign of defocus is recoverable
from one frame, and the ``gaussian`` panel shows exactly what a model trained on
a ring-free approximation would never learn.

Figures
-------
``psf_rings.png``     lateral slices through focus, log scale, three objectives
``psf_axial.png``     x-z sections: index-matched vs mismatched vs aberrated
``psf_models.png``    vectorial / scalar / gaussian, and named aberrations
``psf_profiles.png``  radial profiles and encircled energy against defocus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afocus.optics.psf import PSFEngine
from afocus.optics.system import SampleStack, preset
from afocus.sim.render import FocalShift


def log_show(ax, img, dx, floor=1e-5, extent_um=None):
    a = np.asarray(img, dtype=np.float64)
    a = a / max(a.max(), 1e-300)
    a = np.clip(a, floor, 1.0)
    n = a.shape[0]
    half = n * dx / 2 if extent_um is None else extent_um
    ax.imshow(a, cmap="inferno", norm=LogNorm(vmin=floor, vmax=1.0),
              extent=[-half, half, -half, half], origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    return half


def crop(img, n_keep):
    c = img.shape[0] // 2
    h = n_keep // 2
    return img[c - h:c + h, c - h:c + h]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--fov", type=int, default=96, help="camera pixels for the engine")
    ap.add_argument("--oversampling", type=int, default=4,
                    help="finer than the camera, so the rings are actually sampled")
    ap.add_argument("--floor", type=float, default=2e-3,
                    help="log display floor, as a fraction of each panel's own peak; "
                         "1e-5 crushes the contrast and shows numerical noise")
    ap.add_argument("--axial-floor", type=float, default=1e-4,
                    help="log floor for the axial sections, relative to the global peak")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---------------- figure 1: lateral slices through focus ----------------
    systems = ["20x_air", "40x_water", "60x_oil"]
    dz_dof = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    fig, axes = plt.subplots(len(systems), len(dz_dof),
                             figsize=(1.75 * len(dz_dof), 1.95 * len(systems)))
    for i, name in enumerate(systems):
        s = preset(name)
        # index-matched stack: a clean, symmetric PSF isolates the ring structure
        s = s.evolve(stack=SampleStack(
            n_sample=s.objective.n_immersion, n_glass=s.stack.n_glass,
            n_immersion=s.objective.n_immersion,
            n_glass_design=s.stack.n_glass_design,
            n_immersion_design=s.stack.n_immersion_design))
        eng = PSFEngine(s, fov_px=args.fov, oversampling=args.oversampling,
                        pad=2, workers=-1, cache_size=8)
        dof = s.depth_of_field
        # scale the field with the objective: 6 um is most of a 20x ring pattern
        # and a small corner of a 60x one
        keep = min(eng.n_fine, int(max(2.5, 12.0 * s.abbe_resolution) / eng.dx))
        for j, d in enumerate(dz_dof):
            p = eng.psf_fine(d * dof, 0.5)
            half = log_show(axes[i, j], crop(p, keep), eng.dx, args.floor)
            if i == 0:
                axes[i, j].set_title(f"{d:g} DoF", fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(f"{s.name}\nDoF {dof:.2f} um",
                                      fontsize=7.5, rotation=0, ha="right", va="center")
            if i == len(systems) - 1 and j == 0:
                axes[i, j].set_xlabel(f"{2 * half:.1f} um", fontsize=7)
    fig.suptitle("Lateral PSF through focus, index-matched, log scale "
                 f"(floor {args.floor:g} of peak) — rings are the diffraction "
                 "structure a Gaussian model has none of", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out / "psf_rings.png", dpi=150)
    print(f"wrote {out / 'psf_rings.png'}")

    # ---------------- figure 2: axial x-z sections ----------------
    base = preset("60x_oil")
    cases = [
        ("index matched\n(n_s = n_imm = 1.518)",
         base.evolve(stack=SampleStack(n_sample=1.518, n_immersion=1.518,
                                       n_immersion_design=1.518)), {}, 0.5),
        ("oil into water, 1 um deep\n(depth-induced spherical)",
         base, {}, 1.0),
        ("oil into water, 5 um deep",
         base, {}, 5.0),
        ("matched + 0.12 waves\nspherical aberration",
         base.evolve(stack=SampleStack(n_sample=1.518, n_immersion=1.518,
                                       n_immersion_design=1.518)),
         {"spherical": 0.12}, 0.5),
        ("matched + 0.10 waves\nastigmatism",
         base.evolve(stack=SampleStack(n_sample=1.518, n_immersion=1.518,
                                       n_immersion_design=1.518)),
         {"astig_vertical": 0.10}, 0.5),
    ]
    fig, axes = plt.subplots(1, len(cases), figsize=(2.5 * len(cases), 4.4))
    for k, (label, sysm, ab, depth) in enumerate(cases):
        eng = PSFEngine(sysm, fov_px=args.fov, oversampling=args.oversampling,
                        pad=2, aberration=ab, workers=-1, cache_size=4)
        dof = sysm.depth_of_field
        # The paraxial ratio n_imm/n_sample is not accurate enough to centre
        # these panels: measured best focus for a 1 um deep emitter at 60x/1.40
        # sits 1.44 um out, where paraxial predicts 1.14 um.  Use the same
        # least-squares focal-shift model the renderer uses.
        centre = float(FocalShift(eng, depth_max=max(2 * depth, 4.0))(depth))
        zs = centre + np.linspace(-6 * dof, 6 * dof, 61)
        keep = min(eng.n_fine, int(4.0 / eng.dx))
        section = np.stack([crop(eng.psf_fine(float(z), depth), keep)[keep // 2]
                            for z in zs])
        a = section / section.max()
        a = np.clip(a, args.axial_floor, 1.0)
        x_half = keep * eng.dx / 2
        axes[k].imshow(a.T, cmap="inferno", norm=LogNorm(vmin=args.axial_floor, vmax=1.0),
                       extent=[(zs[0] - centre) / dof, (zs[-1] - centre) / dof,
                               -x_half, x_half],
                       aspect="auto", origin="lower")
        axes[k].axvline(0, color="white", ls=":", lw=0.7, alpha=0.6)
        axes[k].set_title(label, fontsize=8)
        axes[k].set_xlabel("defocus from nominal (DoF)", fontsize=7.5)
        axes[k].tick_params(labelsize=7)
        if k == 0:
            axes[k].set_ylabel("x (um)", fontsize=8)
    fig.suptitle("Axial (x-z) PSF sections, 60x/1.40 oil, log scale.  "
                 "Left panel is symmetric about focus — so a single frame cannot "
                 "give the sign.  The others are not.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "psf_axial.png", dpi=150)
    print(f"wrote {out / 'psf_axial.png'}")

    # ---------------- figure 3: models and aberrations ----------------
    s = preset("60x_oil").evolve(stack=SampleStack(
        n_sample=1.518, n_immersion=1.518, n_immersion_design=1.518))
    panels = [("vectorial", {}, "vectorial"), ("scalar", {}, "scalar"),
              ("gaussian", {}, "gaussian"),
              ("vectorial", {"astig_vertical": 0.12}, "astigmatism 0.12"),
              ("vectorial", {"coma_x": 0.10}, "coma 0.10"),
              ("vectorial", {"spherical": 0.12}, "spherical 0.12"),
              ("vectorial", {"trefoil_x": 0.10}, "trefoil 0.10")]
    dz_show = [0.0, 1.5, 4.0]
    fig, axes = plt.subplots(len(panels), len(dz_show),
                             figsize=(2.0 * len(dz_show), 1.85 * len(panels)))
    for i, (model, ab, label) in enumerate(panels):
        eng = PSFEngine(s, fov_px=args.fov, oversampling=args.oversampling, pad=2,
                        model=model, aberration=ab, workers=-1, cache_size=4)
        dof = s.depth_of_field
        keep = min(eng.n_fine, int(4.0 / eng.dx))
        for j, d in enumerate(dz_show):
            log_show(axes[i, j], crop(eng.psf_fine(d * dof, 0.5), keep),
                     eng.dx, args.floor)
            if i == 0:
                axes[i, j].set_title(f"{d:g} DoF", fontsize=9)
        axes[i, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Forward models and aberrations, 60x/1.40 oil, log scale.  The "
                 "Gaussian row has no rings at all — that is what it costs.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out / "psf_models.png", dpi=150)
    print(f"wrote {out / 'psf_models.png'}")

    # ---------------- figure 4: radial profiles and encircled energy ----------
    eng = PSFEngine(s, fov_px=args.fov, oversampling=args.oversampling, pad=2,
                    workers=-1, cache_size=16)
    dof = s.depth_of_field
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ds = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(ds)))
    for d, c in zip(ds, colours):
        p = eng.psf_fine(d * dof, 0.5)
        n = p.shape[0]; mid = n // 2
        r = np.arange(mid) * eng.dx
        prof = p[mid, mid:] / p.max()
        ax[0].semilogy(r, np.maximum(prof, 1e-8), color=c, lw=1.3,
                       label=f"{d:g} DoF")
        # encircled energy
        yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
        rad = np.hypot(yy, xx) * eng.dx
        edges = np.linspace(0, r[-1], 80)
        cum = np.array([p[rad <= e].sum() for e in edges])
        ax[1].plot(edges, cum / max(cum[-1], 1e-30), color=c, lw=1.3)
    ax[0].set_xlabel("radius (um)"); ax[0].set_ylabel("intensity / peak")
    ax[0].set_title("radial profile — the rings, on a log axis", fontsize=9)
    ax[0].set_xlim(0, 3.0); ax[0].set_ylim(1e-5, 1.5)
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.25)
    ax[1].set_xlabel("radius (um)"); ax[1].set_ylabel("encircled energy")
    ax[1].set_title("encircled energy — monotone in defocus at fixed radius",
                    fontsize=9)
    ax[1].set_xlim(0, 3.0); ax[1].grid(alpha=0.25)

    fine = np.linspace(-8, 8, 65)
    peak = np.array([eng.peak_intensity(z * dof, 0.5) for z in fine])
    e_at = []
    for z in fine:
        p = eng.psf_fine(z * dof, 0.5)
        n = p.shape[0]
        yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
        rad = np.hypot(yy, xx) * eng.dx
        e_at.append(p[rad <= 0.3].sum() / max(p.sum(), 1e-30))
    ax[2].plot(fine, peak / peak.max(), lw=1.5, label="peak intensity (Strehl)")
    ax[2].plot(fine, np.array(e_at) / max(e_at), lw=1.5,
               label="energy within 0.3 um")
    ax[2].axvline(0, color="k", ls=":", lw=0.8)
    ax[2].set_xlabel("defocus (DoF)")
    ax[2].set_title("both are symmetric here — index matched, no aberration",
                    fontsize=9)
    ax[2].legend(fontsize=7.5); ax[2].grid(alpha=0.25)
    fig.suptitle("60x/1.40 oil, index matched: ring structure, encircled energy, "
                 "and the axial response", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "psf_profiles.png", dpi=150)
    print(f"wrote {out / 'psf_profiles.png'}")


if __name__ == "__main__":
    main()
