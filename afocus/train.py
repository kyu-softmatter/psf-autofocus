"""Training and evaluation for the defocus models.

Reported metrics, and why these
-------------------------------
``mae_dof`` / ``median_dof``
    Mean and median absolute error in depths of field.  DoF rather than
    micrometres because it is the unit that means the same thing on every
    objective; micrometres are reported alongside for whichever objective the
    record came from.
``sign_accuracy``
    Fraction of clearly-defocused frames whose sign is right.  Reported
    separately because it is not interchangeable with magnitude error: a
    controller that knows "1 DoF, direction unknown" is useless, while one that
    knows "somewhere between 1 and 3 DoF, downwards" converges fine.
``within_1dof``
    Fraction landing inside one depth of field -- the practical success rate.
``coverage_1sigma``
    Fraction of true errors inside the predicted +-1 sigma.  Should be 0.68.
    An over-confident model is worse than an inaccurate one here, because a
    search policy gates on that confidence.
``validity_auc``
    How well the model recognises frames that carry no focus information.

Everything is additionally broken down per geometry family, since the headline
question is whether a model trained on one family transfers to another.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .models.nets import (DefocusBins, LossWeights, Prediction, build_model,
                          multitask_loss)
from .models.preprocess import (AugmentConfig, augment, normalise_cond,
                                normalise_image, resize)
from .sim.dataset import (N_COND, N_PROFILE_CHANNELS, PROFILE_CHANNELS,
                          ShardedArrays, n_scalar_features)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Records from sharded npz files, preprocessed for one of the model kinds."""

    def __init__(self, arrays: ShardedArrays, indices: np.ndarray,
                 kind: str = "image", image_size: int = 224,
                 augment_cfg: Optional[AugmentConfig] = None,
                 only_valid: bool = False, seed: int = 0,
                 require_features: bool = False) -> None:
        self.a = arrays
        self.kind = kind
        self.image_size = image_size
        self.aug = augment_cfg
        idx = np.asarray(indices)
        if only_valid:
            idx = idx[arrays["valid"][idx]]
        if require_features and kind in ("edge", "hybrid"):
            # The physics features do not exist for every frame: an edge profile
            # needs a resolvable boundary and encircled energy needs a detectable
            # spot, and in a mixed dataset only ~36% and ~40% of frames have one.
            # Training the edge model on frames where its whole input is zero
            # asks it to predict from nothing: measured on 64 mixed records it
            # reached only 2.45 DoF, against 0.61 DoF on 64 records that
            # actually had a profile -- comparable to the image model's 0.49 DoF
            # on the same subset.  The apparent "edge model cannot learn" was
            # entirely this.
            have = arrays["edge_valid"][idx].astype(bool)
            if "encircled_valid" in arrays:
                have |= arrays["encircled_valid"][idx].astype(bool)
            idx = idx[have]
        self.idx = idx
        self.seed = seed
        self.cond = normalise_cond(arrays["cond"])
        # Descriptors are unbounded and full of NaN-derived zeros; a fixed,
        # dataset-independent squash keeps inference identical to training.
        def squash(key: str) -> np.ndarray:
            return np.tanh(np.nan_to_num(arrays[key], nan=0.0,
                                         posinf=0.0, neginf=0.0)).astype(np.float32)
        blocks = [squash("descriptors")]
        if "radial_descriptors" in arrays:
            blocks.append(squash("radial_descriptors"))
        if "encircled" in arrays:
            blocks.append(np.nan_to_num(arrays["encircled"]).astype(np.float32))
        flags = [arrays["edge_valid"].astype(np.float32)[:, None]]
        if "encircled_valid" in arrays:
            flags.append(arrays["encircled_valid"].astype(np.float32)[:, None])
        self.scalars = np.concatenate(blocks + flags, axis=1).astype(np.float32)
        self.n_scalars = int(self.scalars.shape[1])

    def __len__(self) -> int:
        return int(self.idx.size)

    def _rng(self, i: int) -> np.random.Generator:
        return np.random.default_rng((self.seed * 1_000_003 + int(i)) % (2 ** 63))

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        i = int(self.idx[k])
        rng = self._rng(i)
        out: Dict[str, torch.Tensor] = {}

        if self.kind in ("image", "hybrid"):
            img, _, _ = normalise_image(self.a["image"][i])
            if self.aug is not None:
                img = augment(img, rng, self.aug)
            img = resize(img, self.image_size)
            out["image"] = torch.from_numpy(np.ascontiguousarray(img))[None].float()

        if self.kind in ("edge", "hybrid"):
            prof = np.stack([self.a[c][i] for c in PROFILE_CHANNELS
                             if c in self.a] )
            out["profile"] = torch.from_numpy(np.nan_to_num(prof).astype(np.float32))
            out["edge_valid"] = torch.tensor(float(self.a["edge_valid"][i]))
            out["descriptors"] = torch.from_numpy(self.scalars[i])

        out["cond"] = torch.from_numpy(self.cond[i])
        dz = self.a["dz_dof"][i]
        out["dz"] = torch.tensor(0.0 if not np.isfinite(dz) else float(dz))
        out["valid"] = torch.tensor(float(self.a["valid"][i]))
        out["sharpness"] = torch.tensor(float(self.a["sharpness"][i]))
        out["dof_um"] = torch.tensor(float(self.a["dof_um"][i]))
        out["row"] = torch.tensor(i)
        return out


def forward_batch(model: torch.nn.Module, batch: Dict[str, torch.Tensor],
                  kind: str) -> Prediction:
    if kind == "image":
        return model(batch["image"], batch["cond"])
    if kind == "edge":
        return model(batch["profile"], batch["cond"], batch["descriptors"])
    if kind == "hybrid":
        return model(batch["image"], batch["profile"], batch["edge_valid"],
                     batch["cond"], batch["descriptors"])
    raise ValueError(f"unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def evaluate_arrays(dz_true: np.ndarray, dz_pred: np.ndarray, sigma: np.ndarray,
                    valid_true: np.ndarray, valid_logit: np.ndarray,
                    dof_um: np.ndarray, sign_threshold: float = 1.0) -> Dict[str, float]:
    m = valid_true.astype(bool)
    out: Dict[str, float] = {"n": int(m.sum())}
    if m.sum() == 0:
        return out
    err = dz_pred[m] - dz_true[m]
    out["mae_dof"] = float(np.mean(np.abs(err)))
    out["median_dof"] = float(np.median(np.abs(err)))
    out["p90_dof"] = float(np.percentile(np.abs(err), 90))
    out["rmse_dof"] = float(np.sqrt(np.mean(err ** 2)))
    out["bias_dof"] = float(np.mean(err))
    out["mae_nm"] = float(np.mean(np.abs(err * dof_um[m])) * 1000.0)
    out["median_nm"] = float(np.median(np.abs(err * dof_um[m])) * 1000.0)
    out["within_1dof"] = float(np.mean(np.abs(err) <= 1.0))
    out["within_0.25dof"] = float(np.mean(np.abs(err) <= 0.25))

    far = np.abs(dz_true[m]) >= sign_threshold
    if far.any():
        out["sign_accuracy"] = float(np.mean(
            np.sign(dz_pred[m][far]) == np.sign(dz_true[m][far])))
        out["n_sign"] = int(far.sum())

    s = np.maximum(sigma[m], 1e-6)
    out["coverage_1sigma"] = float(np.mean(np.abs(err) <= s))
    out["coverage_2sigma"] = float(np.mean(np.abs(err) <= 2 * s))
    out["mean_sigma_dof"] = float(np.mean(s))
    out["gauss_nll"] = float(np.mean(0.5 * (err / s) ** 2 + np.log(s)))

    # validity head: rank-based AUC, no sklearn dependency
    if valid_true.min() == 0 and valid_true.max() == 1:
        pos = valid_logit[valid_true.astype(bool)]
        neg = valid_logit[~valid_true.astype(bool)]
        if pos.size and neg.size:
            order = np.argsort(np.concatenate([pos, neg]))
            ranks = np.empty(order.size); ranks[order] = np.arange(1, order.size + 1)
            auc = (ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
            out["validity_auc"] = float(auc)
    return out


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, kind: str,
            device: torch.device, bins: DefocusBins) -> Dict[str, np.ndarray]:
    model.eval()
    acc: Dict[str, List[np.ndarray]] = {}
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        p = forward_batch(model, b, kind)
        chunk = {
            "dz_true": b["dz"], "dz_pred": p.dz, "dz_scalar": p.dz_scalar,
            "dz_fused": p.fuse(), "sigma": p.sigma, "entropy": p.entropy(),
            "valid_true": b["valid"], "valid_logit": p.validity,
            "sharp_pred": p.sharpness, "dof_um": b["dof_um"], "row": b["row"],
        }
        for k, v in chunk.items():
            acc.setdefault(k, []).append(v.detach().float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def report(pred: Dict[str, np.ndarray], arrays: Optional[ShardedArrays] = None,
           head: str = "dz_fused", group_by: Optional[str] = "family") -> Dict:
    res = {"overall": evaluate_arrays(
        pred["dz_true"], pred[head], pred["sigma"], pred["valid_true"],
        pred["valid_logit"], pred["dof_um"])}
    for alt in ("dz_pred", "dz_scalar", "dz_fused"):
        if alt != head:
            res[f"head_{alt}"] = evaluate_arrays(
                pred["dz_true"], pred[alt], pred["sigma"], pred["valid_true"],
                pred["valid_logit"], pred["dof_um"])
    if arrays is not None and group_by is not None and group_by in arrays:
        rows = pred["row"].astype(int)
        labels = arrays[group_by].astype(str)[rows]
        res[group_by] = {}
        for g in sorted(set(labels)):
            s = labels == g
            res[group_by][g] = evaluate_arrays(
                pred["dz_true"][s], pred[head][s], pred["sigma"][s],
                pred["valid_true"][s], pred["valid_logit"][s], pred["dof_um"][s])
    return res


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    kind: str = "image"
    arch: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    n_bins: int = 81
    span_dof: float = 16.0
    target_sigma: float = 0.8
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    ema_decay: float = 0.999
    seed: int = 0
    #: Restrict training and evaluation to frames whose physics features exist.
    #: ``"auto"`` enables it for the edge and hybrid models and leaves the image
    #: model on the full set.  Comparisons between kinds are only fair on the
    #: same subset, so the realised counts are reported in the result.
    require_features: str = "auto"
    augment: AugmentConfig = field(default_factory=lambda: AugmentConfig(
        flip=True, rot90=True, crop=0.06, extra_noise=0.01, intensity_jitter=0.1))
    loss: LossWeights = field(default_factory=LossWeights)
    held_out_families: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return asdict(self)


class EMA:
    """Exponential moving average of weights, evaluated instead of the raw model.

    The decay is warmed up as ``min(decay, (1 + step) / (10 + step))`` rather
    than held at its final value from step zero.  Without that, a short run
    evaluates something very close to the initialisation: at decay 0.999 the
    shadow weights retain 96% of their initial value after 39 steps, which
    showed up as a validation MAE frozen at exactly 4.363 DoF across every
    epoch of a 3-epoch smoke test -- easy to mistake for a dead gradient.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


def train(
    data_root: str | Path,
    cfg: Optional[TrainConfig] = None,
    out_dir: Optional[str | Path] = None,
    arrays: Optional[ShardedArrays] = None,
    progress: bool = True,
) -> Dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    a = arrays if arrays is not None else ShardedArrays(data_root)

    if cfg.held_out_families:
        tr_idx, te_idx = a.family_split(cfg.held_out_families)
        # carve a validation slice out of the training scenes, by scene
        sub = a.scene_split((0.9, 0.1), seed=cfg.seed)
        tr_idx = np.intersect1d(tr_idx, sub[0])
        va_idx = np.intersect1d(np.flatnonzero(np.isin(np.arange(len(a)), sub[1])), 
                                np.setdiff1d(np.arange(len(a)), te_idx))
    else:
        tr_idx, va_idx, te_idx = a.scene_split((0.8, 0.1, 0.1), seed=cfg.seed)

    probe = FrameDataset(a, tr_idx[:1], cfg.kind, cfg.image_size, None, seed=cfg.seed)

    n_profile = sum(1 for c in PROFILE_CHANNELS if c in a)
    bins = DefocusBins(cfg.n_bins, cfg.span_dof)
    kw: Dict = {"bins": bins, "n_cond": N_COND}
    if cfg.kind == "image":
        kw |= {"in_channels": 1, "arch": cfg.arch, "pretrained": cfg.pretrained}
    elif cfg.kind == "edge":
        kw |= {"in_channels": n_profile, "n_descriptors": probe.n_scalars}
    else:
        kw |= {"in_channels": 1, "edge_channels": n_profile,
               "n_descriptors": probe.n_scalars,
               "arch": cfg.arch, "pretrained": cfg.pretrained}
    model = build_model(cfg.kind, **kw).to(device)

    need = (cfg.require_features is True or
            (cfg.require_features == "auto" and cfg.kind in ("edge", "hybrid")))
    ds = lambda idx, aug, ov: FrameDataset(a, idx, cfg.kind, cfg.image_size,
                                           aug, only_valid=ov, seed=cfg.seed,
                                           require_features=need)
    dl = lambda d, sh: DataLoader(d, batch_size=cfg.batch_size, shuffle=sh,
                                  num_workers=cfg.num_workers, drop_last=sh)
    train_ds, val_ds, test_ds = (ds(tr_idx, cfg.augment, False),
                                 ds(va_idx, None, False),
                                 ds(te_idx, None, False))
    empty = [n for n, d in (("train", train_ds), ("val", val_ds), ("test", test_ds))
             if len(d) == 0]
    if empty:
        extra = ("  The physics features do not exist for every frame -- an edge "
                 "profile needs a resolvable boundary and encircled energy a "
                 "detectable spot -- and this dataset has none in the affected "
                 "split. Either generate a larger field of view, or pass "
                 "require_features=False (scripts/train.py --all-frames) to "
                 "train on frames whose feature block is all zeros."
                 if need else
                 "  Check that the dataset is large enough to split, and that it "
                 "contains records for the requested geometry families.")
        raise ValueError(
            f"model kind {cfg.kind!r}: the {', '.join(empty)} split is empty "
            f"after filtering ({len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
            f"train/val/test frames from {a and len(a)} records).{extra}")

    train_dl = dl(train_ds, True)
    val_dl = dl(val_ds, False)
    test_dl = dl(test_ds, False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(len(train_dl) * cfg.epochs, 1)
    warm = max(int(steps * cfg.warmup_frac), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + np.cos(np.pi * (s - warm) / max(steps - warm, 1))))
    ema = EMA(model, cfg.ema_decay)

    history: List[Dict] = []
    best = {"mae_dof": float("inf")}
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        sums: Dict[str, float] = {}
        n = 0
        for batch in train_dl:
            b = {k: v.to(device) for k, v in batch.items()}
            p = forward_batch(model, b, cfg.kind)
            loss, parts = multitask_loss(p, b["dz"], b["valid"], b["sharpness"],
                                         bins, cfg.loss, cfg.target_sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step(); ema.update(model)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1

        eval_model = build_model(cfg.kind, **kw).to(device)
        eval_model.load_state_dict(model.state_dict())
        ema.copy_to(eval_model)
        val = report(predict(eval_model, val_dl, cfg.kind, device, bins), a)
        row = {"epoch": epoch, "lr": sched.get_last_lr()[0],
               "train": {k: v / max(n, 1) for k, v in sums.items()},
               "val": val["overall"], "seconds": time.time() - t0}
        history.append(row)
        if progress:
            v = val["overall"]
            print(f"epoch {epoch:3d}  loss {row['train'].get('total', float('nan')):7.4f}  "
                  f"val MAE {v.get('mae_dof', float('nan')):6.3f} DoF "
                  f"({v.get('median_nm', float('nan')):6.0f} nm median)  "
                  f"sign {v.get('sign_accuracy', float('nan')):.3f}  "
                  f"cov1s {v.get('coverage_1sigma', float('nan')):.2f}  "
                  f"AUC {v.get('validity_auc', float('nan')):.3f}  "
                  f"{row['seconds']:.0f}s", flush=True)
        if val["overall"].get("mae_dof", float("inf")) < best["mae_dof"]:
            best = dict(val["overall"]); best["epoch"] = epoch
            if out:
                torch.save({"state_dict": eval_model.state_dict(),
                            "config": cfg.to_dict(), "bins": {"n_bins": bins.n_bins,
                                                              "span": bins.span}},
                           out / "best.pt")

    eval_model = build_model(cfg.kind, **kw).to(device)
    eval_model.load_state_dict(model.state_dict())
    ema.copy_to(eval_model)
    test = report(predict(eval_model, test_dl, cfg.kind, device, bins), a)

    result = {"config": cfg.to_dict(), "history": history, "best_val": best,
              "test": test, "device": str(device),
              "require_features": bool(need),
              "n_train": len(train_dl.dataset), "n_val": len(val_dl.dataset),
              "n_test": len(test_dl.dataset),
              "n_train_scenes_pool": int(tr_idx.size),
              "n_params": sum(p.numel() for p in model.parameters())}
    if out:
        (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
        torch.save({"state_dict": eval_model.state_dict(), "config": cfg.to_dict(),
                    "bins": {"n_bins": bins.n_bins, "span": bins.span}}, out / "final.pt")
    return result
