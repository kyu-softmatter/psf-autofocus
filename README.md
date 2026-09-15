# psf-autofocus

**Physics-based synthetic data and defocus-regression models for fluorescence
microscope autofocus.** Vectorial (Richards–Wolf) PSF simulation, 14 sample
geometries, a full sCMOS/EMCCD sensor model, geometry-invariant focus features,
and the classical z-scan baselines the learned model has to beat.

[![ci](https://github.com/kyu-softmatter/psf-autofocus/actions/workflows/ci.yml/badge.svg)](https://github.com/kyu-softmatter/psf-autofocus/actions/workflows/ci.yml)
![status](https://img.shields.io/badge/status-simulation%20validated%2C%20training%20pending-yellow)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

> **Status.** The forward model, labels and features are built and validated
> against theory (numbers throughout). An 11,000-frame dataset generates in
> ~40 min. **No model has been trained at scale yet, and no real microscope data
> has been through this** — see [`NEXT_STEPS.md`](NEXT_STEPS.md).

It exists to answer one question honestly: **can a network trained on simulated
images focus a real microscope, and does it beat the classical z-scan it would
replace?**

Everything here is measured, not assumed. Numbers quoted below come from the
validation runs described in each section, and the failure modes are documented
because they are the useful part.

---

## 1. What the pipeline does

```
sample geometry  ──►  emitter point cloud  ──►  3-D convolution with a
(14 generators)       (positions + weights)     vectorial PSF
                                                      │
                            ┌─────────────────────────┴───────────┐
                            ▼                                     ▼
                     camera model                          focus label
              (shot, read, dark, PRNU,              (global scan + parabolic
               EM gain, well saturation)             refinement, self-checked)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   raw frame          edge profile        encircled energy
   (ResNet)           (ESF/LSF/MTF,       (spots smaller
                       integral P(r))      than the PSF)
        └───────────────────┼───────────────────┘
                            ▼
                multi-task head: defocus distribution,
                heteroscedastic scalar, validity, sharpness
                            │
                            ▼
                   search policy  ──►  stage
```

## 2. Optical forward model

`afocus/optics/psf.py` implements Richards–Wolf vectorial diffraction in the
back-focal-plane form, so high NA, dipole emission, Zernike aberrations, a
stratified sample/coverglass/immersion stack and supercritical-angle rays all
fall out of one expression. Two cheaper models (`scalar` Gibson–Lanni,
`gaussian`) share the interface for ablations.

**Validated against theory:**

| check | measured | expected |
|---|---|---|
| lateral FWHM, 60x/1.40 oil into water | 0.2167 µm | 0.51·λ/NA_eff = 0.199 µm (0.054 µm grid) |
| energy conservation, 20x/0.75 | 99.8% | 100% (Parseval) |
| OTF at zero frequency | 1.000000 | = total PSF energy |
| Zernike orthonormality (Gram − I) | 6×10⁻³ | 0 (discretisation) |
| Noll and ANSI index tables | exact match | Noll 1976 |

**Conventions that matter.** `defocus` is the signed axial distance from the
nominal focal plane to the emitter; `depth` is the emitter's height above the
coverglass. They are *not* interchangeable: with an oil objective on an aqueous
sample, best focus for an emitter 1 µm deep sits 1.44 µm away in stage travel,
and the ratio itself drifts with depth (1.44 → 1.25 over 1 → 8 µm) because the
depth-induced spherical aberration grows. That asymmetry is also the only thing
that makes the *sign* of defocus recoverable from a single frame.

## 3. Sample geometries

14 generators in four families (`afocus/sim/geometry.py`), all reduced to
emitter point clouds so one renderer handles every shape:

- **spheres** — solid, hollow shell, and a size series for a direct size sweep
- **non-spherical** — ellipsoid, superellipsoid (sphere → cube → octahedron on
  one knob), rod/spherocylinder, irregular "lumpy" particle
- **aggregates** — raspberry cluster, and accretion aggregates in two
  morphologies (ballistic, d_f ≈ 2.6–2.8; self-avoiding chain, d_f ≈ 1.9–2.0)
- **networks** — worm-like-chain filaments, strut networks, spinodal texture
- **reference** — sub-diffraction beads, textured thin sheet with optional tilt

**Validated by construction:** solid-sphere emitter count matches 4πρ/3 (12536
vs 12566 per µm³); an ellipsoid with semi-axes 2.0/0.5/0.5 recovers PCA σ-ratio
4.06 against a true 4.0; a superellipsoid at e=0.1 gives σ = 0.576 = 1/√3, which
is a cube and not a sphere (0.447). Fractal dimensions are **measured** from the
monomer centres and stored, not assumed.

## 4. Rendering

`flux(stage) = Σ_depths density_d ⊛ psf(stage, depth_d)`, exact for incoherent
fluorescence. Two things make it tractable and one bug made it wrong for a
while:

**The depth-induced focus shift is removed analytically.** Best focus is the
least-squares projection, over the pupil, of everything else in the phase onto
the stage term. It agrees with a full peak-intensity PSF scan to 0.3–4.4% while
being 1000–2500× faster (1 ms vs 2.3 s).

> **Bug worth recording.** The first version projected only the *depth* term,
> since only that term contains `depth`. But the coverglass/immersion mismatch
> contributes a large *depth-independent* focus offset, and domain randomisation
> varies coverglass thickness by ±8 µm. With a −8 µm error, best focus moves by
> **+8.0 µm at 60x** — so for a sample 3.6 µm deep the slope-only model predicted
> −3.5 µm where the truth was −10.0 µm, and the label search window did not
> contain focus at all. Including the constant term brought the analytic model
> back to within 0.02–0.23 DoF of the PSF scan.

**Wraparound is bounded and measured.** Rendering is an FFT convolution on a
padded raster. Against a 2× larger grid, the relative error on a realistic
full-field sample stays below 8.5×10⁻⁴; it reaches ~5% only for an isolated
emitter 1 µm from the crop edge at 20 µm defocus, where the blur radius
(22.7 µm) exceeds the padding margin (20.8 µm). `Renderer.max_safe_defocus`
bounds the *relative* defocus (distance from that emitter's own best focus), not
the absolute stage coordinate — bounding the latter wrongly declares the
in-focus plane of a deep sample unreachable.

## 5. Camera

Full sCMOS/EMCCD chain in `afocus/sim/camera.py`. Validated: dark-frame read
noise recovers the 1.60 e⁻ spec exactly; the photon-transfer slope is 2.06 =
1/gain; saturation clips at 62500 ADU = full well / gain; EMCCD output variance
is 2.02× the mean, the expected √2 excess-noise factor.

Two details chosen deliberately. **Well saturation** happens in electrons before
the ADC, because the in-focus plane concentrates the same photons into fewer
pixels and is therefore the one most likely to clip — a model trained without
clipping will confidently mis-rank the sharpest frame. **Fixed-pattern noise** is
re-drawn per field of view, because PRNU and hot pixels do not move when the
stage does and a network will otherwise learn them as landmarks.

## 6. Labels

A 3-D sample has no single in-focus plane, so "in focus" is defined as the stage
position maximising a focus functional on the **noiseless** render of the actual
scene. Getting this right took three attempts and is the most important part of
the repository.

| attempt | label repeatability (median) | scenes rejected |
|---|---|---|
| local bracket walk | 0.007 DoF when it worked, **16–19 DoF** in 4/14 scenes | — |
| + prominence and rival-peak gates | 0.003 DoF on survivors | 61% |
| + constant focal-shift term (§4) | **0.006 DoF**, 14/16 within 0.1 DoF | **11%** |

The diagnostic that exposed the first failure: label the same scene twice from
different search-window positions. A local walk converges to whichever local
maximum it started nearest, and focus curves genuinely have several — from
structure at different depths, and from shot noise reading as sharpness on dim
defocused frames. The fix is a **global** scan over the whole wrap-free domain,
which is also *cheaper* (≈21 renders against up to 65 for the walk).

Three gates reject scenes whose label cannot be trusted, and each caught a real
failure mode:

- **peak prominence** — a flat focus curve has no focus to find, and
  `argmax` then returns a value tied to wherever the window sat. Relabelling
  such scenes from a window shifted by 4.9 DoF moved the label by exactly 4.9 DoF.
- **rival peak** — a second comparable maximum far from the first means the
  scene has no unique best-focus plane.
- **edge peak** — the maximum on the window boundary means the truth is outside
  the domain. This fired 8/18 times before the §4 fix and **0/18** after.

Every record also stores `label_residual_dof`, an independent check of the label
against the emitted planes, so a training run can filter on label quality rather
than trust it.

## 7. Three models, and the experiment that distinguishes them

All three share a head and an output contract (`afocus/models/nets.py`):

- **defocus distribution** over bins, soft-argmax to a continuous estimate.
  Better than scalar regression because the near-focus likelihood is genuinely
  bimodal in sign, and the distribution's spread is a free confidence.
- **heteroscedastic scalar** with Gaussian NLL — a controller needs to know when
  *not* to trust a prediction more than it needs another decimal place.
- **validity** — blank, saturated or signal-free frames carry no focus
  information. Without an explicit "I cannot tell", a regression head emits a
  confident zero on a blank frame and a search policy believes it.
- **sharpness**, as an auxiliary task.

Targets are in **depths of field, not micrometres**: `dz/DoF` means the same
physical blur at 20x/0.75 and 60x/1.40, so the network does not have to infer
the objective's scale from the sample. NA, wavelength, pixel size, DoF and photon
level are supplied as conditioning scalars — all known at acquisition time.

| kind | input | transfers across geometry? | blind when |
|---|---|---|---|
| `image` | raw frame, ResNet | least — most exposed to sample statistics | never |
| `edge` | ESF/LSF/P(r)/encircled energy, 1-D CNN | most — measures the optics | no resolvable boundary *and* no detectable spot |
| `hybrid` | both, edge branch gated by validity | between | never |

**The headline experiment is `--hold-out`:** train with one geometry family
excluded, test on it. A model that learned the optics keeps its accuracy; one
that memorised the synthetic sample does not.

## 8. Geometry-invariant features

Why bother, when a CNN on the raw frame has more capacity? Because an image's
power spectrum is `|O(f)|²·|OTF(f)|²` — the sample's own spectrum multiplies the
optics'. A single frame's spectrum is not a property of the microscope, and the
dominant sim-to-real risk is a model that learned the synthetic `O(f)`.

**Edge spread function** (`afocus/features/edge.py`). Slanted-edge MTF
measurement (ISO 12233) repurposed as a focus sensor: across a large, sharp
boundary the observed profile is the true boundary convolved with the LSF, so its
derivative is the LSF and its transform is the MTF. Averaging along the boundary
buys √n in SNR. Validated against analytic Gaussian blur — 10–90% widths of
0.266/0.510/1.030/2.039 µm against exact 0.256/0.513/1.025/2.051.

The profile is deliberately **not symmetrised**: an aberrated system rings
differently on either side of focus, and that asymmetry is the single-frame sign
information. Curvature screening is explicit — a 1 µm-radius disc is rejected, a
3 µm one accepted, because a boundary curving on the scale of the profile mixes
curvature into the blur estimate.

**Cumulative profile `P(r) = ∫₀ʳ I dr`** (`afocus/features/radial.py`).
Integrating once more than the LSF instead of differentiating: on a noisy camera
frame the descriptors matched the noiseless render to within 2% (0.4153 vs
0.4180 in focus, 0.9285 vs 0.9227 at 3 DoF), where a gradient metric on the same
frames is noise-dominated. Use `fill_deficit`, which is monotone over 0–6 DoF
(0.065 → 0.472). Do *not* use a half-radius of a `P` normalised by its endpoint —
that turns `P` into a shape measure of an already-normalised ramp and barely
moves (1.25, 1.25, 1.18, 1.11 µm, not even monotone).

**Encircled energy `E(r)`.** Covers the regime the edge method cannot: a
sub-diffraction bead has no boundary, so its image *is* the PSF. Measured on
40 beads at 20x/0.75, the edge extractor returned `valid=False` at every defocus
while `E(0.3 µm)` fell monotonically 0.158 → 0.077 → 0.018 over 0 → 2 DoF. Its
own limit is spot detection: at ≥4 DoF nothing exceeds 5σ and it reports invalid.

**Spectral ratio.** The ratio of two frames' power spectra cancels `|O(f)|²`
exactly. Measured on two completely different samples through identical optics:

| band (of f_Nyquist) | thin sheet | filaments |
|---|---|---|
| 0.05–0.2 | 3.269 | 3.512 |
| 0.2–0.4 | 6.102 | 6.094 |
| 0.4–0.7 | 7.256 | 7.424 |

Agreement to 1–7% across unrelated geometries, and swapping the two frames flips
every sign exactly — so it carries **sign** information, unlike any single-frame
cue. This is the most promising geometry-invariant signal in the repository and
pairs naturally with a two-plane acquisition.

**An edge cannot show the rings themselves.** Averaging along a boundary
suppresses noise dramatically -- over 197 edge points the single-point scatter
becomes invisible, and the edge width tracks defocus cleanly (0.79 → 1.45 →
2.57 → 4.62 µm over 0 → 4 DoF at 20x/0.75). But the recovered LSF is a single
smooth hump where the true PSF has strong radial oscillations, because the line
spread function is the PSF integrated *along* the edge and that projection
superimposes different ring radii. So an edge gives the rings' consequences
(width, overshoot, MTF zeros) and not their shape; only a point-like object
does, via encircled energy. `scripts/demo_edge_rings.py` shows the two side by
side.

Also worth knowing before choosing a test object: a uniformly labelled sphere's
column density falls as sqrt(R² − r²) towards the rim, so its edge is not a
step. Measured with a squashed sphere, the in-focus 10–90 width came out at
3.33 µm against ~0.4 µm expected — an order of magnitude of pure geometry. Use a
uniform cylinder (or any flat-topped object) when the point is to measure optics.

**All of these are sign-degenerate on their own.** For an unaberrated,
index-matched system every one is symmetric about focus. Two planes, a
deliberately asymmetric pupil (astigmatism), or off-axis illumination are what
supply the sign.

## 9. Search policies and the classical baseline

`afocus/search/policies.py`. The bar is not the regressor's accuracy but the
microscope's final error *per frame*, since frames cost time and bleaching. The
stage is commanded, not set — `Stage` adds jitter and direction-dependent
backlash — and every frame is counted, including verification frames.

Measured against an oracle estimator with 0.6 DoF noise, 20 repeats:

| policy | median error | p90 | frames |
|---|---|---|---|
| `full_scan` (21 planes) | 0.016 DoF | — | 21 |
| `coarse_to_fine` | 0.005 DoF | — | 16 |
| `hill_climb` | **+33 DoF, runaway** | — | 18 |
| `single_shot` | 0.744 DoF | 1.604 | 2 |
| `iterative` | **0.271 DoF** | 0.599 | 4 |
| `dual_plane` | 0.634 DoF | 1.029 | 3 |
| `model_guided_scan` | 0.275 DoF | 0.447 | 5 |

Two findings from this table:

**`hill_climb` fails catastrophically and used to lie about it.** Gradient focus
metrics rise with shot noise, so a dim, heavily defocused frame can score
*higher* than a mildly defocused one; the climb then finds a genuine local
maximum 48 µm from focus and satisfies its own step criterion. It now requires
the endpoint to beat its neighbours and reports `runaway` and distance
travelled. This is exactly why the classical baseline must be in the repository:
it is what the learned model has to beat, and it is not a strong baseline
everywhere.

**`iterative` was worse than `single_shot` until it stopped throwing frames
away.** Walking to the newest prediction and stopping there carries the full
noise of one estimate (0.645 DoF from 5 frames, against 0.117 from 2).
Combining every frame's *implied* focus position with inverse-variance weights
made extra frames pay: 0.271 DoF.

## 10. Classical focus metrics

Ten metrics in `afocus/sim/metrics.py`, all normalised so scaling an image does
not change the score — otherwise "brighter" reads as "sharper". `RELIABLE` lists
the seven that peaked within 0.1 DoF of the label on every geometry tested.
`CAVEATS` records what the others do:

- `dct_entropy` had an **inverted sign** and peaked at the scan edge on every
  sample. Blur concentrates DCT energy at low frequency, so entropy *falls*
  with blur; sharper is higher entropy. Fixed, now +0.003 DoF.
- `vollath5` peaks at the scan edge on extended textures with both median- and
  exact-mean subtraction, while behaving correctly (+0.01 DoF) on a compact
  sphere. Blur raises the lag-1 correlation while lowering the variance and
  which wins is sample-dependent. Excluded from `RELIABLE`; `vollath4`, a
  difference of two lags, does not have this failure mode.
- `hf_ratio` peaked 0.7 DoF off on a spinodal texture; `laplacian` 0.36 DoF off
  on an axially oriented rod.

## 11. Performance, and two wrong guesses about it

Generation started at 11.7 s per scene. Two plausible-sounding fixes were tried
and neither was the problem:

- **Pinning FFT threads.** With scene-level parallelism, letting each scene's
  FFTs also spawn threads looked like obvious oversubscription. Pinning them to
  one thread each made it *slower*, and the measurement was confounded by
  another job on the same cores anyway.
- **Narrowing the label search window.** Real, safe (`edge_peak` stayed at 0/12)
  and worth keeping, but a small effect.

The actual answer came from `cProfile` on one 18.1 s scene:

| | time | calls |
|---|---|---|
| Zernike wavefront (`radial` + `zernike`) | 3.8 s | 1728 |
| `cos_theta` | 2.6 s | 1446 |
| **FFTs** | **3.1 s** | 468 |
| Fresnel coefficients | 1.0 s | 288 |
| apodization | 0.7 s | 144 |

The FFTs — the only part doing real work — were 17% of the runtime. Everything
above them is fixed by the pupil grid, the system and the aberration state, and
was being rebuilt for every one of the ~144 PSFs a single scene evaluates.
Caching them in `PupilTerms` left one PSF costing a phase ramp, one `exp` and
the FFTs: **3.93× faster overall**, 11.66 s → 2.96 s per scene, with the
validated numbers unchanged (FWHM 0.2167 µm, OTF(0) = 1.000000,
`best_focus(1.0)` = −1.4161 against −1.416 before).

A separate memory fix was also needed. The PSF cache was unbounded, and a
scene's label search touches ~100 distinct keys, each used once: with 9 worker
processes this drove the machine to 17.2 GB of swap and degraded generation from
3.0 to 9.1 s/scene. `PSFEngine(cache_size=...)` now bounds it to an LRU of 16
entries; peak RSS per worker settled around 600 MB.

The lesson worth keeping: *profile before optimising*, and treat a timing
measurement taken while another job shares the cores as no measurement at all.

## 12. Tests

141 tests, 20 seconds, no GPU. They exist because the bugs this code had were
exactly the kind a test catches, and writing them caught two more:

- `dct_entropy` had an **inverted sign** and peaked at the edge of every scan.
  `test_reliable_metrics_peak_at_focus` is one assertion and would have caught
  it the day it was written.
- The focal-shift model **omitted the constant coverglass term**, putting best
  focus 6.5 µm from the truth. `test_focal_shift_includes_the_coverglass_term`
  pins it.
- Writing the suite then found that `FocalShift` **short-circuited the
  index-matched case to a zero shift**. Matched indices remove the depth-induced
  *aberration*, not the focal shift: with n_sample = n_immersion best focus sits
  at exactly −depth, because the emitter has moved. This never bit the
  randomised dataset (indices are never exactly equal) but would have bitten
  the first lab instrument configured as matched.
- And that `DefocusBins.soft_target` **silently produced an all-zero target**
  for any label outside the bin range: every Gaussian weight underflowed, so the
  frame contributed nothing to the loss — dropped by the binning rather than by
  the validity mask, and invisible either way. Targets are now clamped to the
  edge bin.

```bash
pip install -r requirements-dev.txt
python3 -m pytest                       # everything
python3 -m pytest tests/test_psf.py -v  # one module
```

CI runs three jobs on every push: the physics suite on Python 3.10 and 3.12
*without* torch (so it finishes in under a minute), the full suite with CPU-only
torch, and an end-to-end smoke job that generates a tiny dataset, trains one
epoch of each model kind and runs the policy comparison — because unit tests
exercise the modules but only that catches the CLIs drifting away from the
library underneath them.

## 13. Pointing it at a real instrument

The presets are common objectives, but no microscope is exactly one of them, and
the differences move best focus by micrometres. Copy
[`configs/lab_template.yaml`](configs/lab_template.yaml), fill in the numbers,
and check what they imply *before* generating anything:

```bash
python3 scripts/show_system.py configs/mylab.yaml
```

That prints the sampling ratio, the wrap-free defocus range, the focal-shift
offset and slope, and — most useful — the sample thickness beyond which focus
stops being well defined on that instrument. For the template system that limit
is 0.84 µm; generating with the default 8 µm slabs sent 50% of scenes to the
label gates as ambiguous, and following the printed suggestion took the usable
fraction from 50% to 70%.

```bash
python3 scripts/make_dataset.py --out data/mylab --scenes 1000     --system-config configs/mylab.yaml --slab-thickness 0.1 0.84 --workers 6
```

With `--system-config` the optics are held exactly as specified while
aberrations, photon budget, illumination field and stage error stay randomised —
those genuinely vary session to session on one instrument. Pass
`--jitter-optics` to also vary NA, wavelength and the indices, which is right
when training a model that must work on *any* microscope and wrong when the
instrument is known.

## 14. Usage

```bash
# geometry gallery and focus-curve comparison
python scripts/demo_geometry.py --system 20x_air --fov 128

# inspect an instrument before committing compute to it
python scripts/show_system.py --preset 60x_oil

# dataset (parallel over scenes; keep --fft-threads 1 when --workers > 1)
python scripts/make_dataset.py --out data/train --scenes 1200 --workers 9

# one family at a time, for the cross-geometry test
python scripts/make_dataset.py --out data/spheres --scenes 300 --families spheres

# train
python scripts/train.py --data data/train --kind image  --epochs 20
python scripts/train.py --data data/train --kind edge   --hold-out networks
python scripts/train.py --data data/train --kind hybrid --out runs/hybrid

# end-to-end policy comparison on unseen scenes
python scripts/evaluate.py --scenes 40 --oracle 0.5              # policy check
python scripts/evaluate.py --scenes 40 --checkpoint runs/image/best.pt
```

## 15. Honest limits

- **Sim-to-real is unvalidated.** No real microscope data has been through this.
  Domain randomisation (aberrations, index mismatch, coverglass error,
  illumination field, photon budget, stage error) is a hypothesis about what
  matters, not a measurement. A small real fine-tuning set is the missing piece.
- **The forward model ignores** multiple scattering, sample-induced aberration
  beyond the stratified stack, and spatially varying aberration across the
  field. Out-of-field emitters *are* included (they contribute defocused haze).
- **31% of generated frames are rejected**, not 11%: the label gates reject
  13.8% (ambiguous focus peak) plus 3.2% (peak on the search-window edge), and
  an SNR floor removes a further 11.6%. The 11% figure in §6 is the *scene*
  rejection rate from the repeatability study, which used brighter scenes.
- **Saturation is modelled but never exercised.** §5 argues clipping matters
  because the in-focus plane concentrates photons; the generated dataset
  nonetheless contains zero saturated frames, because `target_fill` is drawn
  over (0.05, 0.75) of full well. The argument stands but is untested — see
  `NEXT_STEPS.md`.
- **`hill_climb`'s runaway is real physics**, not a simulation artefact, and it
  will happen on hardware with the same metric.
- **YOLO is the wrong tool** for this task — it is an object detector, and
  defocus is a per-frame regression. It would only make sense for a per-object
  focus map, which is a different problem.
