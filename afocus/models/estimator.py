"""Adapter from a trained model to the search policies' estimator interface.

This is the only place where a frame coming off a camera meets a network, so it
is also the only place where training/inference preprocessing can diverge.  It
therefore reuses :mod:`afocus.models.preprocess` rather than re-deriving
anything, and it takes the optical system as an explicit argument: NA,
wavelength, pixel size and depth of field are needed both to build the
conditioning vector and to convert the model's DoF-normalised output back into
micrometres of stage travel.

Multi-frame handling.  A single-channel model still benefits from more than one
frame: each frame k at stage ``z_k`` predicting ``dz_k`` implies best focus at
``z_k - dz_k``, and combining those with inverse-variance weights is what lets
:class:`afocus.search.policies.DualPlane` resolve the sign.  For an
index-matched, unaberrated system the sign is genuinely unrecoverable from one
frame, so this is not a refinement -- it is the difference between working and
not working.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..features import edge as E
from ..optics.system import ImagingSystem

if TYPE_CHECKING:            # pragma: no cover - typing only
    import torch


@dataclass
class EstimatorConfig:
    image_size: int = 224
    esf_samples: int = 65
    esf_half_width_dof: float = 2.5
    max_sigma_dof: float = 8.0       # above this the estimate is called worthless
    min_validity: float = 0.5


class ModelEstimator:
    """Callable matching :class:`afocus.search.policies.DefocusEstimator`.

    torch is imported here rather than at module scope so that
    :class:`OracleEstimator` -- and therefore the whole of
    :mod:`afocus.search.policies`, including the classical z-scan baselines --
    can be used on a machine with no deep-learning stack installed.  The
    baselines are the thing a learned model has to beat, and needing a 2 GB
    dependency to run them would be absurd.
    """

    def __init__(
        self,
        checkpoint: "str | Path | Dict",
        system: ImagingSystem,
        device: "Optional[torch.device]" = None,
        cfg: Optional[EstimatorConfig] = None,
    ) -> None:
        import torch

        from ..sim.dataset import DESCRIPTOR_NAMES, N_COND, N_DESCRIPTORS
        from .nets import DefocusBins, build_model
        from .preprocess import cond_from_system, normalise_image, resize

        self._torch = torch
        self._helpers = (DESCRIPTOR_NAMES, N_COND, N_DESCRIPTORS,
                         cond_from_system, normalise_image, resize)

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False) \
            if not isinstance(checkpoint, dict) else checkpoint
        self.cfg = cfg or EstimatorConfig()
        self.system = system
        self.train_cfg = ck["config"]
        self.kind = self.train_cfg["kind"]
        self.bins = DefocusBins(ck["bins"]["n_bins"], ck["bins"]["span"])
        self.device = device or torch.device("cpu")

        kw: Dict = {"bins": self.bins, "n_cond": N_COND}
        if self.kind == "image":
            kw |= {"in_channels": 1, "arch": self.train_cfg["arch"], "pretrained": False}
        elif self.kind == "edge":
            kw |= {"in_channels": 3, "n_descriptors": N_DESCRIPTORS}
        else:
            kw |= {"in_channels": 1, "edge_channels": 3,
                   "arch": self.train_cfg["arch"], "pretrained": False}
        self.model = build_model(self.kind, **kw).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.image_size = int(self.train_cfg.get("image_size", self.cfg.image_size))
        self.calls = 0

    # -- single frame ------------------------------------------------------
    def predict_one(self, frame: np.ndarray) -> Dict[str, float]:
        """Defocus for one frame, in micrometres, plus diagnostics."""
        torch = self._torch
        with torch.no_grad():
            return self._predict_one(frame)

    def _predict_one(self, frame: np.ndarray) -> Dict[str, float]:
        torch = self._torch
        (DESCRIPTOR_NAMES, N_COND, N_DESCRIPTORS,
         cond_from_system, normalise_image, resize) = self._helpers
        sysm = self.system
        dof = sysm.depth_of_field
        img_n, _, scale = normalise_image(frame)
        # scale is in ADU; convert to a rough detected-photon level for the
        # conditioning vector, the same quantity the dataset recorded
        signal = max(scale * sysm.camera.gain, 1e-6)
        cond = torch.from_numpy(cond_from_system(sysm, signal)).to(self.device)

        feed: Dict[str, torch.Tensor] = {"cond": cond}
        if self.kind in ("image", "hybrid"):
            feed["image"] = torch.from_numpy(
                np.ascontiguousarray(resize(img_n, self.image_size))
            )[None, None].float().to(self.device)
        if self.kind in ("edge", "hybrid"):
            prof = E.extract(np.asarray(frame, dtype=np.float64), sysm.pixel_size_sample,
                             half_width=self.cfg.esf_half_width_dof * dof,
                             n_samples=self.cfg.esf_samples)
            if prof.valid:
                stack = np.stack([prof.esf, prof.lsf, prof.esf_std])
                desc = np.array([prof.descriptors.get(n, 0.0) for n in DESCRIPTOR_NAMES])
            else:
                stack = np.zeros((3, self.cfg.esf_samples))
                desc = np.zeros(N_DESCRIPTORS)
            feed["profile"] = torch.from_numpy(
                np.nan_to_num(stack).astype(np.float32))[None].to(self.device)
            feed["descriptors"] = torch.from_numpy(
                np.tanh(np.nan_to_num(desc)).astype(np.float32))[None].to(self.device)
            feed["edge_valid"] = torch.tensor([float(prof.valid)]).to(self.device)

        if self.kind == "image":
            p = self.model(feed["image"], feed["cond"])
        elif self.kind == "edge":
            p = self.model(feed["profile"], feed["cond"], feed["descriptors"])
        else:
            p = self.model(feed["image"], feed["profile"], feed["edge_valid"], feed["cond"])

        self.calls += 1
        dz_dof = float(p.fuse().item())
        sigma_dof = float(p.sigma.item())
        validity = float(torch.sigmoid(p.validity).item())
        return {
            "dz_um": dz_dof * dof, "dz_dof": dz_dof,
            "sigma_um": sigma_dof * dof, "sigma_dof": sigma_dof,
            "validity": validity, "entropy": float(p.entropy().item()),
            "edge_valid": float(feed.get("edge_valid", torch.tensor([1.0])).item()),
        }

    # -- policy interface --------------------------------------------------
    def __call__(self, frames: Sequence[np.ndarray],
                 stages: Sequence[float]) -> Tuple[float, float]:
        """Return (defocus at the mean stage position, confidence in [0, 1])."""
        if len(frames) == 0:
            return float("nan"), 0.0
        preds = [self.predict_one(f) for f in frames]
        z = np.asarray(stages, dtype=np.float64)

        implied, weights = [], []
        for zk, pr in zip(z, preds):
            if not np.isfinite(pr["dz_um"]):
                continue
            if pr["validity"] < self.cfg.min_validity:
                continue
            if pr["sigma_dof"] > self.cfg.max_sigma_dof:
                continue
            implied.append(zk - pr["dz_um"])
            weights.append(pr["validity"] / max(pr["sigma_um"] ** 2, 1e-9))

        if not implied:
            return float("nan"), 0.0
        e = np.asarray(implied); w = np.asarray(weights)
        focus = float(np.sum(e * w) / np.sum(w))
        dz = float(np.mean(z) - focus)

        # confidence: mean validity, discounted when the frames disagree
        val = float(np.mean([p["validity"] for p in preds]))
        sig = float(np.mean([p["sigma_dof"] for p in preds]))
        spread = float(np.std(e)) if len(e) > 1 else 0.0
        dof = self.system.depth_of_field
        agree = 1.0 / (1.0 + (spread / max(dof, 1e-9)) ** 2)
        conf = val * agree / (1.0 + sig)
        return dz, float(np.clip(conf, 0.0, 1.0))


class OracleEstimator:
    """Ground-truth estimator with configurable noise -- for testing policies.

    Useful for separating "the policy is wrong" from "the model is wrong": run a
    policy against this first, and any remaining failure is the policy's.
    """

    def __init__(self, best_stage: float, sigma_um: float = 0.3,
                 sign_error_rate: float = 0.0, rng: Optional[np.random.Generator] = None) -> None:
        self.best = float(best_stage)
        self.sigma = float(sigma_um)
        self.sign_error_rate = float(sign_error_rate)
        self.rng = rng or np.random.default_rng()

    def __call__(self, frames: Sequence[np.ndarray],
                 stages: Sequence[float]) -> Tuple[float, float]:
        z = float(np.mean(stages))
        dz = (z - self.best) + self.rng.normal(0.0, self.sigma)
        if len(frames) == 1 and self.rng.random() < self.sign_error_rate:
            dz = -dz
        return dz, float(1.0 / (1.0 + self.sigma))
