"""Multi-task defocus networks.

Three architectures, all with the same head and the same output contract, so
they can be compared on equal terms:

``ImageNet2D``  -- ResNet backbone on the raw frame.  Most capacity, most prone
                   to learning the synthetic sample's statistics.
``EdgeNet1D``   -- 1-D CNN on an averaged edge spread function.  Sees only the
                   optics, so it should transfer across geometries; unusable
                   where no resolvable boundary exists.
``HybridNet``   -- both, fused.  Falls back on the image branch when the edge
                   branch reports no valid profile.

Design choices that matter
--------------------------
**Targets live in depth-of-field units, not micrometres.**  ``dz / DoF`` is
scale-free: the same numeric target means the same physical blur at 20x/0.75 and
60x/1.40.  Training in micrometres forces the network to learn the objective's
scale factor from the image, which is both unnecessary and a generalisation
trap.  Micrometres are recovered at the end by multiplying by the known DoF.

**Defocus is predicted as a distribution, not a number.**  The head emits
logits over ``n_bins`` defocus bins and takes a soft-argmax.  Compared with
direct scalar regression this (a) handles the near-focus regime where the
likelihood is genuinely bimodal in *sign*, (b) converges faster under an L2-like
loss because soft targets do not punish sign confusion as a huge error, and
(c) hands back a usable confidence for free from the distribution's own spread.

**A separate heteroscedastic scalar head.**  Gaussian NLL over ``(mu, log_var)``
gives a calibrated per-frame error bar.  A focus controller needs to know when
*not* to trust a prediction far more than it needs another decimal place.

**A validity head.**  Empty fields, fields with sample only outside the crop,
and saturated frames carry no focus information.  Without an explicit "I cannot
tell" output a regression head will emit a confident zero on a blank frame, and
a search policy will happily believe it.

**Physics conditioning.**  NA, wavelength, pixel size and photon level are fed
in as scalars.  They are known at acquisition time on any real microscope, and
withholding them forces the network to infer the imaging scale from the sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Everything the model says about one frame.  Units: DoF unless stated."""

    dz: torch.Tensor              # (B,) soft-argmax defocus estimate
    dz_scalar: torch.Tensor       # (B,) direct regression estimate
    log_var: torch.Tensor         # (B,) log predictive variance of dz_scalar
    logits: torch.Tensor          # (B, n_bins) defocus distribution
    validity: torch.Tensor        # (B,) logit: does this frame carry focus info
    sharpness: torch.Tensor       # (B,) predicted normalised focus score

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(0.5 * self.log_var)

    def confidence(self) -> torch.Tensor:
        """P(valid) divided by the predictive spread -- what a policy should gate on."""
        return torch.sigmoid(self.validity) / (self.sigma + 1e-3)

    def entropy(self) -> torch.Tensor:
        p = F.softmax(self.logits, dim=-1)
        return -(p * torch.log(p + 1e-12)).sum(-1)

    def fuse(self) -> torch.Tensor:
        """Inverse-variance blend of the two defocus estimates."""
        w = 1.0 / (self.sigma ** 2 + 1e-6)
        return (self.dz + w * self.dz_scalar) / (1.0 + w)

    def to_um(self, dof: torch.Tensor | float) -> torch.Tensor:
        return self.dz * (dof if torch.is_tensor(dof) else float(dof))


# ---------------------------------------------------------------------------
# defocus binning
# ---------------------------------------------------------------------------

class DefocusBins:
    """Bin centres over [-span, span] depths of field, and soft-target builder."""

    def __init__(self, n_bins: int = 81, span: float = 20.0) -> None:
        if n_bins < 3:
            raise ValueError("need at least 3 bins")
        self.n_bins = int(n_bins)
        self.span = float(span)
        self.centres = torch.linspace(-self.span, self.span, self.n_bins)
        self.width = float(self.centres[1] - self.centres[0])

    def soft_target(self, dz: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Gaussian-smoothed one-hot target.

        ``sigma`` is in DoF.  Smoothing is what makes this behave like ordinal
        regression rather than unordered classification: a prediction one bin
        away is penalised far less than one at the wrong sign.
        """
        c = self.centres.to(dz.device)
        d = (dz[:, None] - c[None, :]) / max(sigma, 1e-6)
        t = torch.exp(-0.5 * d ** 2)
        return t / t.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)
        return (p * self.centres.to(logits.device)[None, :]).sum(-1)


# ---------------------------------------------------------------------------
# shared head
# ---------------------------------------------------------------------------

class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, bins: DefocusBins, n_cond: int = 0,
                 hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.bins = bins
        self.n_cond = n_cond
        d = in_dim + n_cond
        self.trunk = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.logits = nn.Linear(hidden, bins.n_bins)
        self.scalar = nn.Linear(hidden, 1)
        self.log_var = nn.Linear(hidden, 1)
        self.validity = nn.Linear(hidden, 1)
        self.sharpness = nn.Linear(hidden, 1)
        # start with a wide, honest error bar rather than a confident wrong one
        nn.init.constant_(self.log_var.bias, 1.0)
        nn.init.zeros_(self.log_var.weight)

    def forward(self, feat: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Prediction:
        if self.n_cond:
            if cond is None:
                raise ValueError(f"model expects {self.n_cond} conditioning scalars, got none")
            feat = torch.cat([feat, cond], dim=-1)
        h = self.trunk(feat)
        logits = self.logits(h)
        return Prediction(
            dz=self.bins.soft_argmax(logits),
            dz_scalar=self.scalar(h).squeeze(-1),
            log_var=self.log_var(h).squeeze(-1).clamp(-8.0, 6.0),
            logits=logits,
            validity=self.validity(h).squeeze(-1),
            sharpness=self.sharpness(h).squeeze(-1),
        )


# ---------------------------------------------------------------------------
# 2-D image model
# ---------------------------------------------------------------------------

class ImageNet2D(nn.Module):
    """ResNet backbone on the frame (or a small stack of frames as channels)."""

    def __init__(self, bins: Optional[DefocusBins] = None, in_channels: int = 1,
                 arch: str = "resnet18", pretrained: bool = False, n_cond: int = 5,
                 dropout: float = 0.1) -> None:
        super().__init__()
        import torchvision

        self.bins = bins or DefocusBins()
        self.in_channels = in_channels
        fn = getattr(torchvision.models, arch, None)
        if fn is None:
            raise ValueError(f"unknown torchvision arch {arch!r}")
        weights = "DEFAULT" if pretrained else None
        net = fn(weights=weights)

        # Adapt the stem to `in_channels`.  When starting from ImageNet weights,
        # averaging the RGB filters preserves the learned edge detectors instead
        # of discarding them, which is worth a lot on small datasets.
        old = net.conv1
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                w = old.weight.mean(dim=1, keepdim=True)
                new.weight.copy_(w.repeat(1, in_channels, 1, 1) / in_channels)
        net.conv1 = new
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = MultiTaskHead(feat_dim, self.bins, n_cond=n_cond, dropout=dropout)

    def forward(self, img: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Prediction:
        return self.head(self.backbone(img), cond)


# ---------------------------------------------------------------------------
# 1-D edge-profile model
# ---------------------------------------------------------------------------

class _Res1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, k: int = 5) -> None:
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride, p, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, 1, p, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = (nn.Identity() if (c_in == c_out and stride == 1) else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                                   nn.BatchNorm1d(c_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.silu(y + self.skip(x))


class EdgeNet1D(nn.Module):
    """1-D residual CNN on stacked edge-profile channels (ESF, LSF, spread)."""

    def __init__(self, bins: Optional[DefocusBins] = None, in_channels: int = 4,
                 widths: Sequence[int] = (32, 64, 128, 128), n_cond: int = 5,
                 n_descriptors: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.bins = bins or DefocusBins()
        layers: List[nn.Module] = []
        c = in_channels
        for i, w in enumerate(widths):
            layers.append(_Res1d(c, w, stride=1 if i == 0 else 2))
            c = w
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = MultiTaskHead(c + n_descriptors, self.bins, n_cond=n_cond, dropout=dropout)
        self.n_descriptors = n_descriptors

    def forward(self, profile: torch.Tensor, cond: Optional[torch.Tensor] = None,
                descriptors: Optional[torch.Tensor] = None) -> Prediction:
        f = self.pool(self.body(profile)).squeeze(-1)
        if self.n_descriptors:
            if descriptors is None:
                raise ValueError(f"model expects {self.n_descriptors} descriptors, got none")
            f = torch.cat([f, descriptors], dim=-1)
        return self.head(f, cond)


# ---------------------------------------------------------------------------
# hybrid
# ---------------------------------------------------------------------------

class HybridNet(nn.Module):
    """Image and edge branches fused, with an explicit edge-validity gate.

    ``edge_valid`` zeroes the edge features when no usable boundary was found,
    so the fusion degrades to the image branch instead of consuming whatever
    garbage an empty extraction produced.
    """

    def __init__(self, bins: Optional[DefocusBins] = None, in_channels: int = 1,
                 edge_channels: int = 4, arch: str = "resnet18", pretrained: bool = False,
                 n_cond: int = 5, n_descriptors: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        import torchvision
        self.bins = bins or DefocusBins()
        fn = getattr(torchvision.models, arch)
        net = fn(weights="DEFAULT" if pretrained else None)
        old = net.conv1
        net.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                              old.stride, old.padding, bias=False)
        img_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.image_branch = net

        edge = EdgeNet1D(self.bins, in_channels=edge_channels, n_cond=0)
        self.edge_body = edge.body
        self.edge_pool = edge.pool
        edge_dim = 128
        self.n_descriptors = int(n_descriptors)
        self.head = MultiTaskHead(img_dim + edge_dim + self.n_descriptors + 1,
                                  self.bins, n_cond=n_cond, dropout=dropout)

    def forward(self, img: torch.Tensor, profile: torch.Tensor,
                edge_valid: torch.Tensor, cond: Optional[torch.Tensor] = None,
                descriptors: Optional[torch.Tensor] = None) -> Prediction:
        fi = self.image_branch(img)
        fe = self.edge_pool(self.edge_body(profile)).squeeze(-1)
        fe = fe * edge_valid[:, None].to(fe.dtype)
        parts = [fi, fe]
        if self.n_descriptors:
            if descriptors is None:
                raise ValueError(f"model expects {self.n_descriptors} descriptors, got none")
            parts.append(descriptors)
        parts.append(edge_valid[:, None].to(fe.dtype))
        return self.head(torch.cat(parts, dim=-1), cond)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    distribution: float = 1.0
    nll: float = 0.5
    validity: float = 0.3
    sharpness: float = 0.1
    sign: float = 0.2


def multitask_loss(
    pred: Prediction,
    dz: torch.Tensor,
    valid: torch.Tensor,
    sharpness: Optional[torch.Tensor] = None,
    bins: Optional[DefocusBins] = None,
    weights: Optional[LossWeights] = None,
    target_sigma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined loss.  ``valid`` is 1 where a defocus target is meaningful.

    Every defocus term is masked by ``valid``: a blank frame has no true
    defocus, so training a regression against one would teach the model to
    invent a number.  Only the validity head learns from those frames.
    """
    w = weights or LossWeights()
    bins = bins or DefocusBins(pred.logits.shape[-1])
    m = valid.float()
    denom = m.sum().clamp_min(1.0)
    parts: Dict[str, float] = {}
    total = torch.zeros((), device=dz.device, dtype=pred.dz.dtype)

    # soft-label cross-entropy over the defocus distribution
    tgt = bins.soft_target(dz, sigma=target_sigma)
    ce = -(tgt * F.log_softmax(pred.logits, dim=-1)).sum(-1)
    l_dist = (ce * m).sum() / denom
    total = total + w.distribution * l_dist
    parts["distribution"] = float(l_dist.detach())

    # heteroscedastic Gaussian NLL on the scalar head
    inv = torch.exp(-pred.log_var)
    nll = 0.5 * (inv * (pred.dz_scalar - dz) ** 2 + pred.log_var)
    l_nll = (nll * m).sum() / denom
    total = total + w.nll * l_nll
    parts["nll"] = float(l_nll.detach())

    # validity, on every frame
    l_val = F.binary_cross_entropy_with_logits(pred.validity, valid.float())
    total = total + w.validity * l_val
    parts["validity"] = float(l_val.detach())

    # sign: an explicit term, because getting the sign right matters more than
    # the magnitude for a controller and the distribution loss under-weights it
    near = (dz.abs() > 0.5 * bins.width).float() * m
    if near.sum() > 0:
        sgn = torch.sign(dz)
        margin = F.softplus(-sgn * pred.dz_scalar * 4.0)
        l_sign = (margin * near).sum() / near.sum().clamp_min(1.0)
        total = total + w.sign * l_sign
        parts["sign"] = float(l_sign.detach())

    if sharpness is not None:
        l_sh = ((pred.sharpness - sharpness) ** 2 * m).sum() / denom
        total = total + w.sharpness * l_sh
        parts["sharpness"] = float(l_sh.detach())

    parts["total"] = float(total.detach())
    return total, parts


def build_model(kind: str, **kw) -> nn.Module:
    kinds = {"image": ImageNet2D, "edge": EdgeNet1D, "hybrid": HybridNet}
    try:
        return kinds[kind](**kw)
    except KeyError:
        raise KeyError(f"unknown model {kind!r}; available: {sorted(kinds)}") from None
