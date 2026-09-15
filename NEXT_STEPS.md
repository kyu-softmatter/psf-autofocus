# Where this left off, and what to run next

The dataset is generated and validated; **no model has been trained at scale
yet**. Everything below is a single command.

## State

| item | status |
|---|---|
| optical forward model, geometries, camera, renderer | built, validated against theory (see README §2–5) |
| label pipeline | validated: repeatability 0.006 DoF median, 11% of scenes correctly rejected |
| feature extractors (ESF/MTF, cumulative P(r), encircled energy, spectral ratio) | built, validated |
| three models + multi-task head + losses | built; `image` passes a 64-sample overfit test at 0.30 DoF |
| search policies + classical baselines | built, compared against an oracle estimator |
| dataset | **generated: 11,000 records / 1000 scenes, 237 MB in `data/train`** |
| **training at scale** | **not done** |
| **sim-to-real validation** | **not started — no real microscope data has been through this** |

## What the generated dataset looks like

```
11000 records from 1000 scenes
  family:  nonspherical=3058  networks=2354  spheres=2079
           aggregates=1606    reference=1529  empty=374
  system:  20x_air=3597  40x_water=3509  60x_oil=3894
  valid:   7610 (69.2%)
  defocus: -14.0 to +14.0 DoF, balanced (3829 negative / 3781 positive)
  SNR:     median 40.3
  physics features: edge 36.2%, encircled energy 44.6%, either 63.4%
```

Rejections (they overlap): 13.8% ambiguous focus peak, 11.6% below the SNR
floor, 3.2% peak on the search-window edge. The defocus balance matters: an
unbalanced set would let the model learn a sign prior instead of reading one.

Note `label_residual_dof` has a median of 0.175 DoF on the usable records, but
that is the resolution of the *check* (a parabola fit through 11 planes spaced
~2.5 DoF apart), not the label's precision. Label precision was measured
separately at 0.006 DoF median by relabelling scenes from shifted search
windows — see README §6.

## Run the training

```bash
cd ~/Desktop/autofocus

# 1. sanity: what is in the dataset
python3 -c "import sys; sys.path.insert(0,'.'); \
from afocus.sim.dataset import ShardedArrays; print(ShardedArrays('data/train').summary())"

# 2. the three models, same data, same split
python3 scripts/train.py --data data/train --kind image  --epochs 25 --out runs/image
python3 scripts/train.py --data data/train --kind edge   --epochs 25 --out runs/edge
python3 scripts/train.py --data data/train --kind hybrid --epochs 25 --out runs/hybrid
```

`edge` and `hybrid` restrict themselves to frames whose physics features exist
(~36% have an edge profile, ~40% an encircled-energy curve, ~52% at least one).
The comparison against `image` is only fair on that subset — `result.json`
records the realised counts under `n_train` / `n_val` / `n_test`.

## The experiment that actually decides the approach

Train with one geometry family held out, test on it. A model that learned the
optics keeps its accuracy; one that memorised the synthetic sample does not.

```bash
for fam in spheres nonspherical aggregates networks; do
  for kind in image edge; do
    python3 scripts/train.py --data data/train --kind $kind --hold-out $fam \
      --epochs 25 --out runs/${kind}_hold_${fam}
  done
done
```

Then compare `test.overall.mae_dof` against the same model trained without a
hold-out. **The prediction to test: `edge` loses less accuracy than `image`**,
because it measures the optics rather than the sample. If it does not, the
geometry-invariance argument in README §8 is wrong and the image model is the
right choice.

## End-to-end, on the microscope's terms

```bash
# policies against a noisy ground-truth estimator: isolates policy from model
python3 scripts/evaluate.py --scenes 40 --oracle 0.5

# policies driven by a trained model
python3 scripts/evaluate.py --scenes 40 --checkpoint runs/image/best.pt
```

This reports final focus error *and frames spent*, which is the comparison that
matters: `coarse_to_fine` reaches 0.005 DoF in 16 frames, so a learned model
earns its place only by being close to that in 2–5.

## Two findings from inspecting the generated dataset

Run `python3 scripts/preview_dataset.py` to reproduce the three figures in
`outputs/`. Two things they showed are worth knowing *before* trusting a
cross-geometry result.

### The edge profile is least available exactly where it matters

Fraction of frames with a usable feature, against defocus:

| defocus | edge profile | encircled energy | either |
|---|---|---|---|
| +-13 DoF | 0.60 | 0.33 | 0.69 |
| **0 DoF** | **0.15** | 0.62 | 0.69 |

The edge extractor fires four times more often on heavily defocused frames than
on in-focus ones. The cause is its own screening: in focus an object is small,
so its boundary is short and tightly curved, and `min_length=24` px plus
`max_curvature_um=4.0` reject it; defocus smears the same object into a large,
gently curved blob that passes.

This matters for the interpretation, not just the coverage. It means the ESF
branch is largely measuring **the blur of a blob, not a true step edge**, which
weakens the geometry-invariance argument in README §8 -- a blob's profile does
depend on the object. Before concluding anything from an `edge` vs `image`
hold-out comparison, either:

- relax the screening (`max_curvature_um` down to ~1.5, `min_length` to ~12) and
  let the model condition on the curvature, which `EdgeProfile` already
  returns; or
- restrict the comparison to scenes with genuinely large objects
  (`hollow_shell` with radius > 2 um, `thin_sheet`, the network family), where
  a real edge exists at every defocus.

Encircled energy runs the opposite way (0.62 in focus, 0.33 far out), so "either
feature available" is flat at 0.55-0.70 across the whole range. That is good
news for `hybrid` and is the strongest argument for using both.

### Compact objects mostly land outside the crop

`draw_sample` scatters compact geometries uniformly over the *padded* raster so
that out-of-field material contributes its defocused haze. With `pad=2` the crop
is a quarter of that area, so only ~25% of objects land in frame. The visible
consequence, in `dataset_geometries.png`, is that many `ellipsoid`,
`raspberry_cluster` and `fractal_aggregate` frames are nearly empty or show the
object clipped at the edge -- and it is a large part of the 11.6% of records
rejected by the SNR floor.

Fix (then regenerate, ~40 min): draw most object positions inside the crop and
keep a minority outside, e.g. in `afocus/sim/scene.py::draw_sample` replace the
uniform `rng.uniform(-half, half)` with a 70/30 mixture of `crop_half` and
`half`. Keep some outside -- the haze is real and a model trained only on
isolated crops never learns to ignore it.

### What did validate cleanly

In `dataset_stats.png`, normalised sharpness against the defocus label collapses
onto one curve for all six geometry families, peaking at 0. That is an
independent confirmation that the labels are consistent and that the focus
response really is geometry-independent -- which is the premise the whole
approach rests on.

## Known open items

1. **Sign of defocus.** Currently relies on depth-induced spherical aberration.
   The robust options are implemented but untested end-to-end: `DualPlane`
   (two frames at a known offset) and the spectral ratio, whose sign flips
   exactly when the two frames are swapped. Deliberate astigmatism is the third
   option and would need a cylindrical-lens term added to the pupil.
2. **`depth_bin` is unmeasured.** Set to 0.5 µm, which is conservative and
   therefore slow for thick samples (bins scale with slab thickness). The
   residual after removing the analytic focal shift is depth-dependent spherical
   aberration, which should vary slowly enough to allow coarser bins — but that
   was never measured, so it was left conservative rather than guessed. The
   study to run: compare renders and labels at `depth_bin` 0.25 → 8.0 µm on a
   thick sample and find where the label moves by more than 0.05 DoF.
3. **Generation parallel efficiency is poor** — 2.06 s/scene wall clock on 6
   workers against 2.96 s single-threaded. Memory-bound, not CPU-bound.
4. **Saturation is modelled but never exercised.** The camera model clips in
   electrons before the ADC, and README §5 argues this matters because the
   in-focus plane concentrates photons and is the frame most likely to clip.
   But the generated dataset contains **zero** saturated frames: `target_fill`
   is drawn over (0.05, 0.75) of full well, so `auto_exposure` never pushes a
   frame into clipping. The argument is therefore untested. Either widen
   `RandomisationConfig.target_fill` past 1.0 for a fraction of scenes, or drop
   the claim.
5. **No real data.** Domain randomisation is a hypothesis about what matters,
   not a measurement. A few hundred real z-stacks with a known stage position
   would let the sim-to-real gap be measured and fine-tuned against.
