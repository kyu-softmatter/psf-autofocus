"""Search policies, driven against a synthetic microscope with a known answer."""
from __future__ import annotations

import numpy as np
import pytest

from afocus.models.estimator import OracleEstimator
from afocus.search import policies as P
from afocus.sim.scene import Stage


@pytest.fixture
def acquirer():
    """A one-dimensional stand-in: a Gaussian blob whose width is minimal at a
    known stage position.  Focus metrics on it behave like they do on a real
    frame, without the cost of rendering one."""
    truth = 1.7

    def acquire(stage: float) -> np.ndarray:
        sigma = 1.5 + 6.0 * abs(stage - truth)
        n = 64
        y, x = np.mgrid[0:n, 0:n] - n / 2
        img = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return 1000.0 * img / img.max() + 5.0

    acquire.truth = truth
    return acquire


@pytest.mark.parametrize("policy", [
    P.FullScan(span=8.0, n_planes=21),
    P.CoarseToFine(span=8.0, n_coarse=9, n_fine=7),
])
def test_metric_policies_find_the_optimum(policy, acquirer):
    res = policy.run(acquirer, start=acquirer.truth + 5.0)
    assert abs(res.error(acquirer.truth)) < 0.3
    assert res.n_frames > 0
    assert res.converged


def test_frames_are_counted_including_verification(acquirer):
    est = OracleEstimator(acquirer.truth, sigma_um=0.2,
                          rng=np.random.default_rng(0))
    one = P.SingleShot(est, verify=False).run(acquirer, start=5.0)
    two = P.SingleShot(est, verify=True).run(acquirer, start=5.0)
    assert one.n_frames == 1
    assert two.n_frames == 2


def test_budget_is_enforced(acquirer):
    with pytest.raises(RuntimeError, match="budget"):
        P.FullScan(span=8.0, n_planes=21).run(acquirer, start=0.0, budget=5)


@pytest.mark.parametrize("name,make", [
    ("single_shot", lambda e: P.SingleShot(e, tolerance=0.2)),
    ("iterative", lambda e: P.Iterative(e, max_iters=5, tolerance=0.15)),
    ("dual_plane", lambda e: P.DualPlane(e, offset=1.0, tolerance=0.15)),
    ("model_guided_scan", lambda e: P.ModelGuidedScan(e, span=6.0, n_planes=5)),
])
def test_model_policies_converge_with_a_noisy_estimator(name, make, acquirer):
    rng = np.random.default_rng(7)
    errors = []
    for _ in range(12):
        est = OracleEstimator(acquirer.truth, sigma_um=0.5, rng=rng)
        res = make(est).run(acquirer, start=acquirer.truth + rng.uniform(-5, 5))
        errors.append(abs(res.error(acquirer.truth)))
    assert np.median(errors) < 1.2, f"{name} median error {np.median(errors):.3f}"


def test_iterative_beats_single_shot_by_using_every_frame(acquirer):
    """Walking to the newest prediction and stopping there throws away every
    earlier frame, so the answer carries the noise of one estimate.  Combining
    the implied focus positions is what makes extra frames pay."""
    rng = np.random.default_rng(11)
    med = {}
    for name, make in (("single", lambda e: P.SingleShot(e)),
                       ("iterative", lambda e: P.Iterative(e, max_iters=5))):
        errs = []
        for _ in range(24):
            est = OracleEstimator(acquirer.truth, sigma_um=0.8, rng=rng)
            res = make(est).run(acquirer, start=acquirer.truth + rng.uniform(-5, 5))
            errs.append(abs(res.error(acquirer.truth)))
        med[name] = float(np.median(errs))
    assert med["iterative"] < med["single"]


def test_hill_climb_reports_a_runaway_instead_of_claiming_success():
    """Gradient focus metrics rise with shot noise, so a dim, heavily defocused
    frame can outscore a mildly defocused one and the climb settles tens of
    micrometres away with its step criterion satisfied.  The failure is real
    physics; the policy must not call it convergence."""
    def adversarial(stage: float) -> np.ndarray:
        rng = np.random.default_rng(int(abs(stage) * 1000) % 100000)
        # score increases with |stage|: a monotone flank with no true peak
        return rng.normal(100.0, 1.0 + abs(stage), size=(48, 48))

    res = P.HillClimb(step=2.0, min_step=0.05, max_frames=30,
                      travel=25.0).run(adversarial, start=0.0)
    assert res.info["travelled_um"] > 0
    if res.info["travelled_um"] > 0.8 * 25.0:
        assert res.info["runaway"] is True
        assert res.converged is False


def test_stage_backlash_appears_only_on_reversal():
    stage = Stage(np.random.default_rng(3), jitter=0.0, backlash=0.1)
    forward = [stage.move_to(t) - t for t in (0.0, 1.0, 2.0, 3.0)]
    reversed_ = stage.move_to(2.0) - 2.0
    assert all(abs(e) < 1e-9 for e in forward[1:])
    assert abs(reversed_) == pytest.approx(0.1, abs=1e-9)


def test_policy_registry_is_consistent():
    assert set(P.METRIC_ONLY) | set(P.MODEL_BASED) == set(P.POLICIES)
    assert not set(P.METRIC_ONLY) & set(P.MODEL_BASED)
