"""Networks, losses and the output contract.  Requires torch."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from afocus.models.nets import (DefocusBins, EdgeNet1D, HybridNet, ImageNet2D,
                                LossWeights, build_model, multitask_loss)
from afocus.models.preprocess import (AugmentConfig, augment, cond_from_system,
                                      normalise_cond, normalise_image, resize)
from afocus.optics.system import preset


@pytest.fixture(scope="module")
def bins():
    return DefocusBins(n_bins=41, span=16.0)


def test_soft_target_is_a_distribution(bins):
    dz = torch.tensor([0.0, 3.5, -7.2, 15.0])
    t = bins.soft_target(dz, sigma=1.0)
    assert torch.allclose(t.sum(dim=1), torch.ones(4), atol=1e-6)
    assert (t >= 0).all()


def test_soft_argmax_inverts_the_soft_target(bins):
    """Round-tripping a target through the head's own decoder must return the
    value; otherwise the regression is biased by the binning itself."""
    dz = torch.tensor([0.0, 3.5, -7.2, 12.0])
    logits = torch.log(bins.soft_target(dz, sigma=1.0) + 1e-20)
    assert torch.allclose(bins.soft_argmax(logits), dz, atol=1e-3)


@pytest.mark.parametrize("kind,call", [
    ("image", lambda m: m(torch.randn(3, 1, 64, 64), torch.randn(3, 5))),
    ("edge", lambda m: m(torch.randn(3, 4, 65), torch.randn(3, 5), torch.randn(3, 12))),
    ("hybrid", lambda m: m(torch.randn(3, 1, 64, 64), torch.randn(3, 4, 65),
                           torch.tensor([1.0, 0.0, 1.0]), torch.randn(3, 5),
                           torch.randn(3, 12))),
])
def test_forward_shapes_and_contract(kind, call, bins):
    kw = {"bins": bins, "n_cond": 5}
    if kind == "image":
        kw |= {"in_channels": 1, "pretrained": False}
    elif kind == "edge":
        kw |= {"in_channels": 4, "n_descriptors": 12}
    else:
        kw |= {"in_channels": 1, "edge_channels": 4, "n_descriptors": 12,
               "pretrained": False}
    m = build_model(kind, **kw).eval()
    with torch.no_grad():
        p = call(m)
    assert p.dz.shape == (3,)
    assert p.logits.shape == (3, bins.n_bins)
    assert (p.sigma > 0).all()
    assert torch.isfinite(p.fuse()).all()
    assert (p.entropy() >= 0).all()
    assert torch.isfinite(p.confidence()).all()
    assert torch.allclose(p.to_um(2.0), p.dz * 2.0)


def test_edge_branch_is_gated_by_validity(bins):
    """A zeroed feature block and a genuinely flat one are not the same thing;
    the gate lets the hybrid fall back to the image branch instead of consuming
    whatever an empty extraction produced."""
    m = HybridNet(bins=bins, in_channels=1, edge_channels=4, n_cond=5,
                  n_descriptors=12, pretrained=False).eval()
    img = torch.randn(2, 1, 64, 64)
    cond, desc = torch.randn(2, 5), torch.randn(2, 12)
    prof_a, prof_b = torch.randn(2, 4, 65), torch.randn(2, 4, 65)
    off = torch.zeros(2)
    with torch.no_grad():
        a = m(img, prof_a, off, cond, desc).dz
        b = m(img, prof_b, off, cond, desc).dz
    assert torch.allclose(a, b, atol=1e-6)      # edge input ignored when gated off


def test_loss_masks_invalid_frames(bins):
    """A blank frame has no true defocus, so training a regression against one
    teaches the model to invent a number.  Only the validity head may learn
    from those frames."""
    m = ImageNet2D(bins=bins, in_channels=1, n_cond=5, pretrained=False)
    img, cond = torch.randn(4, 1, 64, 64), torch.randn(4, 5)
    dz = torch.tensor([1.0, -2.0, 6.0, -9.0])
    p = m(img, cond)
    all_valid = torch.ones(4)
    some_valid = torch.tensor([1.0, 1.0, 0.0, 0.0])
    _, parts_all = multitask_loss(p, dz, all_valid, bins=bins)
    _, parts_some = multitask_loss(p, dz, some_valid, bins=bins)
    # the masked loss is the mean over the valid rows only, so it must equal
    # what those two rows alone contribute -- the invalid rows are absent, not
    # merely down-weighted
    tgt = bins.soft_target(dz, sigma=1.0)        # multitask_loss default
    ce = -(tgt * torch.log_softmax(p.logits, dim=-1)).sum(-1).detach()
    assert parts_some["distribution"] == pytest.approx(
        float(ce[:2].mean()), rel=1e-5)
    assert parts_all["distribution"] == pytest.approx(float(ce.mean()), rel=1e-5)


def test_out_of_range_targets_are_clamped_not_dropped(bins):
    """A target beyond the bin range must still supervise the edge bin.  Left
    unclamped, every Gaussian weight underflows to zero and the row becomes an
    all-zero target that contributes nothing to the loss -- a frame silently
    dropped by the binning rather than by the validity mask."""
    far = bins.soft_target(torch.tensor([1e3, -1e3]), sigma=1.0)
    assert torch.allclose(far.sum(dim=1), torch.ones(2), atol=1e-6)
    assert far[0].argmax().item() == bins.n_bins - 1
    assert far[1].argmax().item() == 0


def test_gradients_reach_every_head(bins):
    m = ImageNet2D(bins=bins, in_channels=1, n_cond=5, pretrained=False)
    p = m(torch.randn(4, 1, 64, 64), torch.randn(4, 5))
    loss, parts = multitask_loss(p, torch.randn(4) * 4, torch.ones(4),
                                 sharpness=torch.rand(4), bins=bins)
    loss.backward()
    for name in ("logits", "scalar", "log_var", "validity", "sharpness"):
        w = getattr(m.head, name).weight
        assert w.grad is not None and w.grad.abs().sum() > 0, f"{name} got no gradient"
    assert set(parts) >= {"distribution", "nll", "validity", "sharpness", "total"}


def test_model_can_fit_a_learnable_mapping(bins):
    """Sanity check that the head and loss can actually learn.  The profile
    encodes the target linearly, so any working model fits it quickly; failure
    here means a wiring bug, not a hard problem."""
    torch.manual_seed(0)
    n = 96
    dz = torch.empty(n).uniform_(-8.0, 8.0)
    ramp = torch.linspace(-1.0, 1.0, 65)[None, None, :]
    profile = (dz[:, None, None] / 8.0) * ramp.repeat(n, 4, 1)
    cond, desc = torch.zeros(n, 5), torch.zeros(n, 12)
    m = EdgeNet1D(bins=bins, in_channels=4, n_cond=5, n_descriptors=12)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(150):
        idx = torch.randperm(n)[:32]
        p = m(profile[idx], cond[idx], desc[idx])
        loss, _ = multitask_loss(p, dz[idx], torch.ones(32), bins=bins)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        mae = (m(profile, cond, desc).fuse() - dz).abs().mean().item()
    assert mae < 2.0, f"MAE {mae:.3f} DoF on a linearly encoded target"


def test_log_var_starts_wide():
    """An over-confident initialisation is worse than an inaccurate one: a
    search policy gates on this number."""
    m = ImageNet2D(bins=DefocusBins(41, 16.0), in_channels=1, n_cond=5,
                   pretrained=False).eval()
    with torch.no_grad():
        p = m(torch.randn(2, 1, 64, 64), torch.randn(2, 5))
    assert (p.sigma > 0.8).all()


def test_missing_conditioning_raises(bins):
    m = ImageNet2D(bins=bins, in_channels=1, n_cond=5, pretrained=False).eval()
    with pytest.raises(ValueError, match="conditioning"):
        m(torch.randn(2, 1, 64, 64), None)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def test_image_normalisation_is_scale_invariant(rng):
    """Absolute brightness on a real microscope is not reproducible, so any
    dependence on it is a dependence on something that will not transfer."""
    img = (rng.random((64, 64)) * 1000 + 100).astype(np.uint16)
    a, _, _ = normalise_image(img)
    b, _, scale = normalise_image(img.astype(np.float64) * 37.0)
    assert np.allclose(a, b, atol=1e-5)
    assert np.median(a) == pytest.approx(0.0, abs=1e-6)
    assert np.percentile(a, 99.5) == pytest.approx(1.0, abs=1e-6)


def test_conditioning_uses_fixed_constants_not_dataset_statistics():
    """Normalising with dataset statistics at training time and per-image ones
    at deployment is the classic way to ruin a model like this."""
    c = normalise_cond(np.array([[0.75, 0.52, 0.325, 0.92, 2.5]]))
    again = normalise_cond(np.array([[0.75, 0.52, 0.325, 0.92, 2.5],
                                     [1.4, 0.6, 0.1, 0.4, 4.0]]))
    assert np.allclose(c[0], again[0])
    live = cond_from_system(preset("20x_air"), signal_photons=300.0)
    assert live.shape == (1, 5) and np.isfinite(live).all()


def test_augmentation_preserves_the_label_symmetries(rng):
    """Defocus is axial, so in-plane flips and rotations leave it unchanged --
    and an axial flip, which would invert the sign, is deliberately absent."""
    img = rng.random((32, 32)).astype(np.float32)
    out = augment(img, rng, AugmentConfig(flip=True, rot90=True, crop=0.0,
                                          intensity_jitter=0.0, extra_noise=0.0))
    assert out.shape == img.shape
    assert sorted(out.ravel().tolist()) == pytest.approx(
        sorted(img.ravel().tolist()), rel=1e-6)


def test_area_resize_preserves_the_mean(rng):
    img = rng.random((128, 128)).astype(np.float32)
    assert resize(img, 32).mean() == pytest.approx(img.mean(), rel=1e-4)
    assert resize(img, 224).shape == (224, 224)
