"""Autofocus search policies.

Every policy drives the same interface -- move the stage, take a frame, decide --
so simulation and real hardware are interchangeable and, more importantly, so
the comparison is fair.  The thing a learned autofocus has to beat is not just
the accuracy of a classical z-scan but its accuracy *per frame*, because frames
cost time and photobleaching.

Policies are grouped by what they need:

*Metric only* (no model): ``FullScan``, ``CoarseToFine``, ``HillClimb``.  These
are the baselines a real microscope ships with.

*Model* : ``SingleShot``, ``Iterative``, ``DualPlane``, ``ModelGuidedScan``.

Two accounting rules keep the comparison honest.  First, the stage is commanded,
not set: :class:`afocus.sim.scene.Stage` adds jitter and backlash, so the
reported error includes stage error exactly as it would on hardware.  Second,
every frame is counted, including the ones a policy takes to verify its own
answer -- a single-shot method that needs a confirmation frame is a two-frame
method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..sim import metrics as M


# ---------------------------------------------------------------------------
# interfaces
# ---------------------------------------------------------------------------

class Acquirer(Protocol):
    """Moves the stage to a commanded position and returns one frame."""

    def __call__(self, stage: float) -> np.ndarray: ...


class DefocusEstimator(Protocol):
    """Maps frames to a defocus estimate in um, with a confidence in [0, 1]."""

    def __call__(self, frames: Sequence[np.ndarray],
                 stages: Sequence[float]) -> Tuple[float, float]: ...


@dataclass
class FocusResult:
    final_stage: float
    n_frames: int
    trace: List[Tuple[float, float]] = field(default_factory=list)  # (stage, score or estimate)
    converged: bool = False
    confidence: float = float("nan")
    info: Dict = field(default_factory=dict)

    def error(self, truth: float) -> float:
        return float(self.final_stage - truth)


class Camera:
    """Counts frames and records the trace, wrapping a raw acquirer."""

    def __init__(self, acquire: Acquirer, budget: Optional[int] = None) -> None:
        self._acquire = acquire
        self.budget = budget
        self.n_frames = 0
        self.stages: List[float] = []

    def grab(self, stage: float) -> np.ndarray:
        if self.budget is not None and self.n_frames >= self.budget:
            raise RuntimeError(f"frame budget of {self.budget} exhausted")
        img = self._acquire(float(stage))
        self.n_frames += 1
        self.stages.append(float(stage))
        return img


# ---------------------------------------------------------------------------
# metric-only baselines
# ---------------------------------------------------------------------------

@dataclass
class FullScan:
    """Scan a fixed grid, fit a parabola to the metric peak.  The textbook method."""

    span: float = 10.0
    n_planes: int = 21
    metric: str = "brenner"

    def run(self, acquire: Acquirer, start: float = 0.0,
            budget: Optional[int] = None) -> FocusResult:
        cam = Camera(acquire, budget)
        stages = start + np.linspace(-self.span, self.span, self.n_planes)
        scores = np.array([M.focus_score(cam.grab(z), self.metric) for z in stages])
        best = M.argmax_parabolic(stages, scores)
        j = int(scores.argmax())
        interior = 0 < j < self.n_planes - 1
        return FocusResult(
            final_stage=float(best), n_frames=cam.n_frames,
            trace=list(zip(stages.tolist(), scores.tolist())),
            converged=interior,
            info={"peak_interior": interior, "step": float(stages[1] - stages[0]),
                  "contrast": float(scores.max() / max(scores.min(), 1e-30))},
        )


@dataclass
class CoarseToFine:
    """Coarse scan, then a narrow scan around the coarse peak."""

    span: float = 12.0
    n_coarse: int = 9
    n_fine: int = 7
    metric: str = "brenner"

    def run(self, acquire: Acquirer, start: float = 0.0,
            budget: Optional[int] = None) -> FocusResult:
        cam = Camera(acquire, budget)
        trace: List[Tuple[float, float]] = []
        zs = start + np.linspace(-self.span, self.span, self.n_coarse)
        sc = np.array([M.focus_score(cam.grab(z), self.metric) for z in zs])
        trace += list(zip(zs.tolist(), sc.tolist()))
        j = int(sc.argmax())
        step = float(zs[1] - zs[0])
        lo, hi = zs[max(j - 1, 0)], zs[min(j + 1, self.n_coarse - 1)]
        zf = np.linspace(lo, hi, self.n_fine)
        sf = np.array([M.focus_score(cam.grab(z), self.metric) for z in zf])
        trace += list(zip(zf.tolist(), sf.tolist()))
        best = M.argmax_parabolic(zf, sf)
        return FocusResult(float(best), cam.n_frames, trace,
                           converged=0 < j < self.n_coarse - 1,
                           info={"coarse_step": step})


@dataclass
class HillClimb:
    """Greedy climb with step halving on reversal -- what many controllers do.

    Cheap when it starts near focus, and unreliable when it does not.  Two
    failure modes, both observed in this simulator rather than hypothesised:

    1. On a monotone flank the climb has no way to tell "nearly there" from
       "far away", so it can walk the entire travel range.
    2. Gradient focus metrics rise with shot noise, so a heavily defocused, dim
       frame can score *higher* than a mildly defocused one.  The climb then
       finds a genuine local maximum tens of micrometres from focus.  Measured
       on a thin-sheet sample at 20x/0.75 starting 6 um out, it settled 48.5 um
       away -- and the step criterion was satisfied, so a naive convergence
       flag called that a success.

    ``travel`` bounds the search the way a real stage does, and ``converged``
    additionally requires the endpoint to actually beat both of its neighbours.
    """

    step: float = 2.0
    min_step: float = 0.05
    max_frames: int = 30
    travel: float = 25.0
    metric: str = "brenner"

    def run(self, acquire: Acquirer, start: float = 0.0,
            budget: Optional[int] = None) -> FocusResult:
        cap = min(self.max_frames, budget) if budget else self.max_frames
        cam = Camera(acquire, cap)
        z0 = float(start)
        z = z0
        step = float(self.step)
        s_here = M.focus_score(cam.grab(z), self.metric)
        trace = [(z, s_here)]
        direction = 1.0
        hit_limit = False
        reversals = 0
        while cam.n_frames < cap and step >= self.min_step:
            z_try = z + direction * step
            if abs(z_try - z0) > self.travel:
                hit_limit = True
                direction = -direction
                step *= 0.5
                reversals += 1
                continue
            s_try = M.focus_score(cam.grab(z_try), self.metric)
            trace.append((z_try, s_try))
            if s_try > s_here:
                z, s_here = z_try, s_try
            else:
                direction = -direction
                step *= 0.5
                reversals += 1

        # A satisfied step criterion only proves the steps got small, and a
        # local maximum only proves the neighbours are lower.  Neither rules out
        # having settled on a noise-driven maximum far from focus: measured on a
        # rendered scene the climb landed 8 DoF out having reversed once, with
        # both of those conditions met.  So also require the peak to have been
        # bracketed from both sides -- at least two reversals -- and report
        # everything a caller needs to disbelieve the answer.
        near = [(zz, ss) for zz, ss in trace if abs(zz - z) <= 2.0 * self.step]
        is_local_max = all(ss <= s_here + 1e-12 for _, ss in near)
        travelled = abs(z - z0)
        bracketed = reversals >= 2
        return FocusResult(
            float(z), cam.n_frames, trace,
            converged=bool(step < self.min_step and is_local_max
                           and bracketed and not hit_limit),
            info={"final_step": step, "travelled_um": travelled,
                  "hit_travel_limit": hit_limit, "local_max": is_local_max,
                  "reversals": reversals, "bracketed": bracketed,
                  "runaway": travelled > 0.8 * self.travel},
        )


# ---------------------------------------------------------------------------
# model-driven policies
# ---------------------------------------------------------------------------

@dataclass
class SingleShot:
    """One frame, one prediction, one move.  Optionally one frame to verify.

    ``verify`` is not free and is reported in ``n_frames``.  It is worth it: the
    verification frame is what turns a point estimate into a bounded error, and
    without it a confident wrong sign is undetectable.
    """

    estimator: DefocusEstimator
    verify: bool = True
    tolerance: float = 0.15

    def run(self, acquire: Acquirer, start: float = 0.0,
            budget: Optional[int] = None) -> FocusResult:
        cam = Camera(acquire, budget)
        z = float(start)
        img = cam.grab(z)
        dz, conf = self.estimator([img], [z])
        z_new = z - dz
        trace = [(z, dz)]
        converged = True
        if self.verify:
            img2 = cam.grab(z_new)
            dz2, conf2 = self.estimator([img2], [z_new])
            trace.append((z_new, dz2))
            converged = abs(dz2) <= self.tolerance
            if abs(dz2) < abs(dz):
                z_new = z_new - dz2
                conf = conf2
        return FocusResult(float(z_new), cam.n_frames, trace, converged, conf,
                           info={"first_estimate": float(dz)})


@dataclass
class Iterative:
    """Predict, move, repeat -- accumulating every estimate rather than the last.

    Each frame k at stage ``z_k`` predicting defocus ``dz_k`` independently
    implies best focus at ``z_k - dz_k``.  Walking to the newest prediction and
    stopping there throws away every earlier frame, so the final answer carries
    the full noise of a single estimate: measured against an estimator with
    0.6 DoF noise, a 5-frame walk landed 0.65 DoF out while a 2-frame single
    shot managed 0.12 DoF.  Combining the implied positions with
    inverse-variance weights instead makes extra frames actually pay.

    Damping still matters for where the *next* frame is taken: near focus the
    estimate's noise is comparable to the residual error, so stepping the full
    correction wanders instead of settling.
    """

    estimator: DefocusEstimator
    max_iters: int = 5
    damping: float = 0.85
    tolerance: float = 0.1
    min_confidence: float = 0.0

    def run(self, acquire: Acquirer, start: float = 0.0,
            budget: Optional[int] = None) -> FocusResult:
        cam = Camera(acquire, budget)
        z = float(start)
        trace: List[Tuple[float, float]] = []
        implied: List[float] = []
        weights: List[float] = []
        conf = float("nan")
        converged = False
        for _ in range(self.max_iters):
            img = cam.grab(z)
            dz, conf = self.estimator([img], [z])
            trace.append((z, dz))
            if not np.isfinite(dz) or conf < self.min_confidence:
                break
            implied.append(z - dz)
            weights.append(max(float(conf), 1e-12))
            if abs(dz) <= self.tolerance:
                converged = True
                break
            z = z - self.damping * dz

        if not implied:
            return FocusResult(float(z), cam.n_frames, trace, False, conf,
                               info={"iters": len(trace), "n_used": 0})
        e = np.asarray(implied); w = np.asarray(weights)
        # later frames sit closer to focus, where the estimator is more
        # accurate, so weight by recency as well as by reported confidence
        w = w * np.linspace(0.5, 1.0, len(w))
        best = float(np.sum(e * w) / np.sum(w))
        spread = float(np.sqrt(np.average((e - best) ** 2, weights=w))) if len(e) > 1 else 0.0
        return FocusResult(best, cam.n_frames, trace, converged, conf,
                           info={"iters": len(trace), "n_used": int(len(e)),
                                 "consensus_spread": spread,
                                 "last_position": float(z)})


@dataclass
class DualPlane:
    """Two frames at a known offset, estimated jointly.

    This is the robust answer to the sign problem.  For an unaberrated,
    index-matched system the axial response is symmetric, so one frame cannot
    distinguish above from below focus at all; two frames at a known separation
    always can, because the pair's relative sharpness is monotone in position.
    The price is exactly one extra frame.
    """

    estimator: DefocusEstimator
    offset: float = 1.0
    refine: bool = True
    tolerance: float = 0.1

    def run(self, acquire: Acquirer, start: float = 0.0,
            budget: Optional[int] = None) -> FocusResult:
        cam = Camera(acquire, budget)
        z0 = float(start) - self.offset / 2.0
        z1 = float(start) + self.offset / 2.0
        f0, f1 = cam.grab(z0), cam.grab(z1)
        dz, conf = self.estimator([f0, f1], [z0, z1])
        z = float(start) - dz
        trace = [(z0, dz), (z1, dz)]
        converged = True
        if self.refine:
            f = cam.grab(z)
            dz2, conf2 = self.estimator([f, f], [z, z])
            trace.append((z, dz2))
            converged = abs(dz2) <= self.tolerance
            z = z - dz2
            conf = conf2
        return FocusResult(float(z), cam.n_frames, trace, converged, conf)


@dataclass
class ModelGuidedScan:
    """Coarse scan, then a confidence-weighted consensus of per-frame estimates.

    Each frame k at stage z_k predicts a defocus dz_k, so each independently
    implies best focus at ``z_k - dz_k``.  Averaging those with inverse-variance
    weights uses the whole scan instead of just its peak, which is what makes
    this beat a parabolic fit on sparse or dim samples -- and it needs no
    assumption that the metric curve is quadratic.
    """

    estimator: DefocusEstimator
    span: float = 8.0
    n_planes: int = 5
    trim: float = 0.2

    def run(self, acquire: Acquirer, start: float = 0.0,
            budget: Optional[int] = None) -> FocusResult:
        cam = Camera(acquire, budget)
        zs = start + np.linspace(-self.span, self.span, self.n_planes)
        est, conf = [], []
        for z in zs:
            d, c = self.estimator([cam.grab(z)], [z])
            est.append(z - d); conf.append(c)
        est = np.asarray(est); conf = np.asarray(conf, dtype=np.float64)
        keep = np.isfinite(est) & np.isfinite(conf) & (conf > 0)
        if not keep.any():
            return FocusResult(float(start), cam.n_frames,
                               list(zip(zs.tolist(), est.tolist())), False, 0.0)
        e, w = est[keep], conf[keep]
        if self.trim > 0 and e.size >= 5:
            # drop the most discordant tail: one badly wrong sign would
            # otherwise drag a plain weighted mean across focus
            med = np.median(e)
            order = np.argsort(np.abs(e - med))
            k = max(int(np.ceil(e.size * (1 - self.trim))), 3)
            e, w = e[order[:k]], w[order[:k]]
        best = float(np.sum(e * w) / np.sum(w))
        spread = float(np.sqrt(np.average((e - best) ** 2, weights=w)))
        return FocusResult(best, cam.n_frames, list(zip(zs.tolist(), est.tolist())),
                           converged=spread < self.span / 4,
                           confidence=float(w.mean()),
                           info={"consensus_spread": spread, "n_used": int(e.size)})


POLICIES = {
    "full_scan": FullScan, "coarse_to_fine": CoarseToFine, "hill_climb": HillClimb,
    "single_shot": SingleShot, "iterative": Iterative,
    "dual_plane": DualPlane, "model_guided_scan": ModelGuidedScan,
}
METRIC_ONLY = ("full_scan", "coarse_to_fine", "hill_climb")
MODEL_BASED = ("single_shot", "iterative", "dual_plane", "model_guided_scan")
