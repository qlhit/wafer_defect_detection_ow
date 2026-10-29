#!/usr/bin/env python3
"""RDP-gated atomic explainer for unknown components in MixedWM38.

Put this file in ``rdp_net_repro/scripts/`` and run it from the repository
root.  It freezes a leave-one-defect-out RDP-Net and trains an eight-channel
pixel explainer using exact atomic masks from online mixtures of singleton
wafer maps.  At inference, only RDP-predicted known channels may explain
observed defect pixels; the remaining defect probability is the unknown map.
Composition dropout keeps an atom visible in the input while removing it from
the permission condition, so the model learns to leave unlisted atoms residual.

Default output is intentionally minimal:
  * best_atomic_explainer.pt
  * metrics.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdpnet.models.rdpnet import RDPNet  # noqa: E402


FULL_CLASSES: Tuple[str, ...] = ("C", "D", "EL", "ER", "L", "NF", "S", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an RDP-gated pixel-level atomic explainer"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--rdp-checkpoint", required=True)
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="S")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=16000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument(
        "--truth-condition-prob",
        type=float,
        default=0.50,
        help="During training, probability of using exact composition instead of frozen RDP output",
    )
    parser.add_argument(
        "--component-dropout-prob",
        type=float,
        default=0.50,
        help="Probability of hiding one present atom from the explanation condition",
    )
    parser.add_argument("--pseudo-unknown-weight", type=float, default=1.0)
    parser.add_argument("--atomic-dice-weight", type=float, default=1.0)
    parser.add_argument("--recomposition-weight", type=float, default=0.50)
    parser.add_argument("--rdp-threshold", type=float, default=0.50)
    parser.add_argument("--map-threshold", type=float, default=0.50)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/atomic_explainer_<unknown>_heldout",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_torch_checkpoint(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_mixedwm38(path: str) -> Tuple[Tensor, Tensor]:
    raw = np.load(path)
    if "arr_0" not in raw.files:
        raise KeyError(f"{path} has no arr_0; found keys {raw.files}")
    x = torch.from_numpy(raw["arr_0"].astype(np.float32))

    # Original repository: arr_1 is full [N,8].  Earlier open-world exports:
    # arr_1 may be known [N,7] and arr_2 is the full [N,8] label.
    if "arr_2" in raw.files and raw["arr_2"].ndim == 2 and raw["arr_2"].shape[1] == 8:
        y_full_np = raw["arr_2"]
    elif "arr_1" in raw.files and raw["arr_1"].ndim == 2 and raw["arr_1"].shape[1] == 8:
        y_full_np = raw["arr_1"]
    else:
        shapes = {key: tuple(raw[key].shape) for key in raw.files}
        raise ValueError(
            "A full 8-column label array is required in arr_1 or arr_2; "
            f"found {shapes}"
        )
    y_full = torch.from_numpy(y_full_np.astype(np.float32))

    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"arr_0 must have shape (N,H,W), got {tuple(x.shape)}")
    if len(x) != len(y_full):
        raise ValueError("wafer maps and full labels have different sample counts")
    return x, y_full


def label_codes(y: Tensor) -> np.ndarray:
    values = y.to(torch.int64).cpu().numpy()
    return np.asarray(["".join(map(str, row.tolist())) for row in values])


def make_open_set_split(
    y_full: Tensor,
    unknown_idx: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("known_train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("known_val_ratio must be positive and train+val must be < 1")

    indices = np.arange(len(y_full))
    is_unknown = y_full[:, unknown_idx].cpu().numpy() > 0.5
    known_indices = indices[~is_unknown]
    unknown_indices = indices[is_unknown]
    active = [i for i in range(8) if i != unknown_idx]

    known_codes = label_codes(y_full[known_indices][:, active])
    train_idx, remainder_idx = train_test_split(
        known_indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=known_codes,
    )
    val_fraction = val_ratio / (1.0 - train_ratio)
    remainder_codes = label_codes(y_full[remainder_idx][:, active])
    val_idx, known_test_idx = train_test_split(
        remainder_idx,
        train_size=val_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=remainder_codes,
    )
    return {
        "known_train": np.asarray(train_idx),
        "known_val": np.asarray(val_idx),
        "known_test": np.asarray(known_test_idx),
        "unknown_test": np.asarray(unknown_indices),
    }


def defect_and_valid_masks(x_raw: Tensor) -> Tuple[Tensor, Tensor]:
    """Support raw 0/1/2 maps and normalized 0/0.5/1 maps."""
    maximum = float(x_raw.max().item()) if x_raw.numel() else 2.0
    if maximum <= 1.01:
        valid = x_raw > 0.25
        defect = x_raw > 0.75
    else:
        valid = x_raw > 0.5
        defect = x_raw > 1.5
    return valid, defect


def normalize_wafer(x_raw: Tensor) -> Tensor:
    maximum = x_raw.amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(maximum > 1.01, torch.full_like(maximum, 2.0), torch.ones_like(maximum))
    return (x_raw / scale).clamp(0.0, 1.0)


def unique_known_combinations(
    y_full: Tensor,
    indices: Sequence[int],
    unknown_idx: int,
) -> List[Tuple[int, ...]]:
    active = set(i for i in range(8) if i != unknown_idx)
    combinations = set()
    for row in y_full[torch.as_tensor(indices, dtype=torch.long)].to(torch.int64):
        combination = tuple(i for i in range(8) if int(row[i]) == 1)
        if combination and set(combination).issubset(active):
            combinations.add(combination)
    return sorted(combinations, key=lambda item: (len(item), item))


class SyntheticAtomicDataset(Dataset):
    """Online exact-label mixtures made only from known singleton wafers."""

    def __init__(
        self,
        x: Tensor,
        y_full: Tensor,
        source_indices: Sequence[int],
        unknown_idx: int,
        combinations: Sequence[Tuple[int, ...]],
        length: int,
        seed: int,
    ) -> None:
        self.x = x
        self.y_full = y_full
        self.unknown_idx = unknown_idx
        self.combinations = list(combinations)
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0

        source = torch.as_tensor(source_indices, dtype=torch.long)
        labels = y_full[source]
        singleton = labels.sum(dim=1) == 1
        source = source[singleton]
        labels = y_full[source]
        self.by_class: Dict[int, np.ndarray] = {}
        for class_idx in range(8):
            if class_idx == unknown_idx:
                continue
            members = source[labels[:, class_idx] > 0.5].cpu().numpy()
            if len(members) == 0:
                raise ValueError(
                    f"No known singleton samples found for class {FULL_CLASSES[class_idx]}"
                )
            self.by_class[class_idx] = members

        usable = [
            combo
            for combo in self.combinations
            if all(class_idx in self.by_class for class_idx in combo)
        ]
        if not usable:
            raise ValueError("No usable known composition templates were found")
        self.combinations = usable

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        rng = np.random.default_rng(
            self.seed + self.epoch * 1000003 + int(index)
        )
        combo = self.combinations[int(rng.integers(len(self.combinations)))]
        selected = {
            class_idx: int(rng.choice(self.by_class[class_idx]))
            for class_idx in combo
        }

        base_idx = selected[combo[0]]
        base_raw = self.x[base_idx]
        base_valid, _ = defect_and_valid_masks(base_raw)
        target = torch.zeros(
            (8, base_raw.shape[-2], base_raw.shape[-1]), dtype=torch.float32
        )
        for class_idx, sample_idx in selected.items():
            _, atom_defect = defect_and_valid_masks(self.x[sample_idx])
            target[class_idx] = (atom_defect & base_valid).to(torch.float32)

        union = target.amax(dim=0) > 0.5
        composite = torch.zeros_like(base_raw, dtype=torch.float32)
        composite[base_valid] = 1.0
        composite[union] = 2.0
        composition = torch.zeros(8, dtype=torch.float32)
        composition[list(combo)] = 1.0
        return composite, target, composition


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConditionalAtomicUNet(nn.Module):
    def __init__(self, base_channels: int = 32, output_channels: int = 8) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(2, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.bottleneck = ConvBlock(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)
        self.condition = nn.Sequential(
            nn.Linear(8, 2 * c),
            nn.SiLU(inplace=True),
            nn.Linear(2 * c, 8 * c),
        )
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = ConvBlock(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = ConvBlock(2 * c, c)
        self.out = nn.Conv2d(c, output_channels, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        gamma, beta = self.condition(condition).chunk(2, dim=1)
        b = b * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def model_input(x_raw: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    valid, defect = defect_and_valid_masks(x_raw)
    normalized = normalize_wafer(x_raw)
    inputs = torch.stack([normalized, valid.to(normalized.dtype)], dim=1)
    return inputs, valid, defect


@torch.no_grad()
def frozen_rdp_probabilities(
    rdp: RDPNet,
    x_raw: Tensor,
    unknown_idx: int,
) -> Tensor:
    probabilities = torch.sigmoid(rdp(x_raw.unsqueeze(1)))
    probabilities[:, unknown_idx] = 0.0
    return probabilities


def atomic_losses(
    logits: Tensor,
    explain_targets: Tensor,
    full_targets: Tensor,
    valid: Tensor,
    known_indices: Sequence[int],
    atomic_dice_weight: float,
    recomposition_weight: float,
    pseudo_unknown_weight: float,
) -> Tuple[Tensor, Dict[str, float]]:
    logits_known = logits[:, known_indices]
    targets_known = explain_targets[:, known_indices]
    full_targets_known = full_targets[:, known_indices]
    valid_expanded = valid[:, None].to(logits.dtype)

    positives = (targets_known * valid_expanded).sum(dim=(0, 2, 3))
    total_valid = valid_expanded.sum() * targets_known.shape[1]
    per_class_valid = valid_expanded.sum()
    negatives = per_class_valid - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    bce_map = F.binary_cross_entropy_with_logits(
        logits_known,
        targets_known,
        pos_weight=pos_weight[None, :, None, None],
        reduction="none",
    )
    atomic_bce = (bce_map * valid_expanded).sum() / (
        total_valid.clamp_min(1.0)
    )

    probs = torch.sigmoid(logits_known) * valid_expanded
    target_valid = targets_known * valid_expanded
    intersection = (probs * target_valid).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + target_valid.sum(dim=(0, 2, 3))
    atomic_dice_loss = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()

    union_prob = 1.0 - torch.prod(1.0 - probs.clamp(0.0, 1.0), dim=1)
    union_target = target_valid.amax(dim=1)
    union_bce_map = F.binary_cross_entropy(
        union_prob.clamp(1e-6, 1.0 - 1e-6), union_target, reduction="none"
    )
    union_bce = (union_bce_map * valid.to(logits.dtype)).sum() / valid.sum().clamp_min(1)
    union_intersection = (union_prob * union_target).sum(dim=(1, 2))
    union_denominator = union_prob.sum(dim=(1, 2)) + union_target.sum(dim=(1, 2))
    union_dice_loss = 1.0 - (
        (2.0 * union_intersection + 1.0) / (union_denominator + 1.0)
    ).mean()

    recomposition_loss = union_bce + union_dice_loss
    full_union_target = (full_targets_known * valid_expanded).amax(dim=1)
    pseudo_unknown_target = (
        full_union_target * (1.0 - union_target)
    ).clamp(0.0, 1.0)
    pseudo_support = pseudo_unknown_target.sum()
    if float(pseudo_support.detach().item()) > 0.0:
        pseudo_unknown_loss = (
            union_prob * pseudo_unknown_target
        ).sum() / pseudo_support
    else:
        pseudo_unknown_loss = logits.new_zeros(())
    total = (
        atomic_bce
        + atomic_dice_weight * atomic_dice_loss
        + recomposition_weight * recomposition_loss
        + pseudo_unknown_weight * pseudo_unknown_loss
    )
    return total, {
        "atomic_bce": float(atomic_bce.detach().item()),
        "atomic_dice_loss": float(atomic_dice_loss.detach().item()),
        "recomposition_loss": float(recomposition_loss.detach().item()),
        "pseudo_unknown_loss": float(pseudo_unknown_loss.detach().item()),
    }


@torch.no_grad()
def segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    known_indices: Sequence[int],
    threshold: float,
) -> Tuple[float, float]:
    prediction = (
        torch.sigmoid(logits[:, known_indices]) >= threshold
    ).to(torch.float32)
    truth = targets[:, known_indices]
    valid_expanded = valid[:, None].to(torch.float32)
    prediction *= valid_expanded
    truth *= valid_expanded

    support = truth.sum(dim=(0, 2, 3)) > 0
    intersection = (prediction * truth).sum(dim=(0, 2, 3))
    denominator = prediction.sum(dim=(0, 2, 3)) + truth.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    atomic_dice = float(dice[support].mean().item()) if support.any() else 0.0

    union_prediction = prediction.amax(dim=1)
    union_truth = truth.amax(dim=1)
    union_intersection = (union_prediction * union_truth).sum(dim=(1, 2))
    union_denominator = union_prediction.sum(dim=(1, 2)) + union_truth.sum(dim=(1, 2))
    recomposition_dice = float(
        ((2.0 * union_intersection + 1.0) / (union_denominator + 1.0)).mean().item()
    )
    return atomic_dice, recomposition_dice


def build_training_task(
    rdp_probabilities: Tensor,
    exact_composition: Tensor,
    full_targets: Tensor,
    rdp_threshold: float,
    truth_probability: float,
    component_dropout_probability: float,
    unknown_idx: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Create an explanation permission list and aligned pixel targets.

    The input wafer remains unchanged.  For selected samples, one truly
    present atom is removed from the condition and becomes pseudo-unknown.
    """
    if truth_probability <= 0.0:
        condition = rdp_probabilities.clone()
    elif truth_probability >= 1.0:
        condition = exact_composition.clone()
    else:
        selector = torch.rand(
            (len(exact_composition), 1), device=exact_composition.device
        ) < truth_probability
        condition = torch.where(
            selector, exact_composition, rdp_probabilities
        ).clone()
    condition[:, unknown_idx] = 0.0
    allowed = condition >= rdp_threshold
    allowed[:, unknown_idx] = False
    dropped = torch.zeros_like(allowed)

    apply_dropout = torch.rand(
        len(exact_composition), device=exact_composition.device
    ) < component_dropout_probability
    for sample_idx in range(len(exact_composition)):
        if not bool(apply_dropout[sample_idx]):
            continue
        present = torch.nonzero(
            exact_composition[sample_idx] > 0.5, as_tuple=False
        ).flatten()
        present = present[present != unknown_idx]
        if present.numel() == 0:
            continue
        currently_allowed = present[allowed[sample_idx, present]]
        candidates = currently_allowed if currently_allowed.numel() else present
        choice = int(
            torch.randint(
                candidates.numel(),
                (1,),
                device=exact_composition.device,
            ).item()
        )
        dropped_idx = int(candidates[choice].item())
        condition[sample_idx, dropped_idx] = 0.0
        allowed[sample_idx, dropped_idx] = False
        dropped[sample_idx, dropped_idx] = True

    explain_targets = full_targets * allowed[:, :, None, None].to(
        full_targets.dtype
    )
    return condition, explain_targets, dropped


def run_synthetic_epoch(
    explainer: ConditionalAtomicUNet,
    rdp: RDPNet,
    loader: DataLoader,
    device: torch.device,
    known_indices: Sequence[int],
    unknown_idx: int,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    explainer.train(training)
    rdp.eval()
    sums = {
        "loss": 0.0,
        "atomic_bce": 0.0,
        "atomic_dice_loss": 0.0,
        "recomposition_loss": 0.0,
        "pseudo_unknown_loss": 0.0,
        "atomic_dice": 0.0,
        "recomposition_dice": 0.0,
    }
    count = 0

    for x_raw, targets, exact_composition in loader:
        x_raw = x_raw.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        exact_composition = exact_composition.to(device, non_blocking=True)
        inputs, valid, _ = model_input(x_raw)
        with torch.no_grad():
            rdp_probs = frozen_rdp_probabilities(rdp, x_raw, unknown_idx)
            if training:
                condition, explain_targets, _ = build_training_task(
                    rdp_probabilities=rdp_probs,
                    exact_composition=exact_composition,
                    full_targets=targets,
                    rdp_threshold=args.rdp_threshold,
                    truth_probability=args.truth_condition_prob,
                    component_dropout_probability=args.component_dropout_prob,
                    unknown_idx=unknown_idx,
                )
            else:
                # Stable full-composition validation measures ordinary atomic
                # decomposition; pseudo-unknown behavior is evaluated separately.
                condition = exact_composition
                explain_targets = targets

        with torch.set_grad_enabled(training):
            logits = explainer(inputs, condition)
            loss, parts = atomic_losses(
                logits,
                explain_targets,
                targets,
                valid,
                known_indices,
                args.atomic_dice_weight,
                args.recomposition_weight,
                args.pseudo_unknown_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        atomic_dice, recomposition_dice = segmentation_metrics(
            logits.detach(),
            explain_targets,
            valid,
            known_indices,
            args.map_threshold,
        )
        batch_size = len(x_raw)
        sums["loss"] += float(loss.detach().item()) * batch_size
        for key, value in parts.items():
            sums[key] += value * batch_size
        sums["atomic_dice"] += atomic_dice * batch_size
        sums["recomposition_dice"] += recomposition_dice * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_explainer(
    explainer: ConditionalAtomicUNet,
    rdp: RDPNet,
    train_dataset: SyntheticAtomicDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    known_indices: Sequence[int],
    unknown_idx: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Tensor], int, Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state = copy.deepcopy(explainer.state_dict())
    best_epoch = 0
    best_metrics: Dict[str, float] = {"loss": float("inf")}
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = run_synthetic_epoch(
            explainer,
            rdp,
            train_loader,
            device,
            known_indices,
            unknown_idx,
            args,
            optimizer,
        )
        val_metrics = run_synthetic_epoch(
            explainer,
            rdp,
            val_loader,
            device,
            known_indices,
            unknown_idx,
            args,
            optimizer=None,
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_atomic_dice={val_metrics['atomic_dice']:.4f} "
            f"val_recomp_dice={val_metrics['recomposition_dice']:.4f}"
        )
        if val_metrics["loss"] < best_metrics["loss"]:
            best_state = copy.deepcopy(explainer.state_dict())
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
                break
    return best_state, best_epoch, best_metrics


def make_real_loader(
    x: Tensor,
    y_full: Tensor,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    base = TensorDataset(x, y_full)
    subset = Subset(base, [int(i) for i in indices])
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def top_fraction_score(
    unknown_heatmap: Tensor,
    defect: Tensor,
    top_fraction: float,
) -> Tensor:
    scores: List[Tensor] = []
    for index in range(len(unknown_heatmap)):
        values = unknown_heatmap[index][defect[index]]
        if values.numel() == 0:
            scores.append(torch.zeros((), device=unknown_heatmap.device))
            continue
        k = max(1, int(math.ceil(values.numel() * top_fraction)))
        scores.append(values.topk(k).values.mean())
    return torch.stack(scores)


@torch.no_grad()
def evaluate_synthetic_pseudo_unknown(
    explainer: ConditionalAtomicUNet,
    loader: DataLoader,
    device: torch.device,
    known_indices: Sequence[int],
    unknown_idx: int,
    map_threshold: float,
    top_fraction: float,
) -> Dict[str, float]:
    """Drop one known atom from each condition and verify it remains residual."""
    explainer.eval()
    no_drop_scores: List[Tensor] = []
    drop_scores: List[Tensor] = []
    residual_dice_values: List[Tensor] = []
    global_index = 0

    for x_raw, full_targets, exact_composition in loader:
        x_raw = x_raw.to(device, non_blocking=True)
        full_targets = full_targets.to(device, non_blocking=True)
        exact_composition = exact_composition.to(device, non_blocking=True)
        inputs, _, defect = model_input(x_raw)

        full_condition = exact_composition.clone()
        full_condition[:, unknown_idx] = 0.0
        full_probs = torch.sigmoid(explainer(inputs, full_condition))
        full_allowed = full_condition > 0.5
        full_allowed[:, unknown_idx] = False
        full_coverage = (
            full_probs
            * full_allowed[:, :, None, None].to(full_probs.dtype)
        ).amax(dim=1)
        full_residual = defect.to(full_probs.dtype) * (1.0 - full_coverage)
        no_drop_scores.append(
            top_fraction_score(full_residual, defect, top_fraction).cpu()
        )

        drop_condition = exact_composition.clone()
        for sample_idx in range(len(exact_composition)):
            present = torch.nonzero(
                exact_composition[sample_idx] > 0.5, as_tuple=False
            ).flatten()
            present = present[present != unknown_idx]
            if present.numel() == 0:
                continue
            selected = int(present[global_index % present.numel()].item())
            global_index += 1
            drop_condition[sample_idx, selected] = 0.0
        drop_condition[:, unknown_idx] = 0.0

        drop_probs = torch.sigmoid(explainer(inputs, drop_condition))
        drop_allowed = drop_condition > 0.5
        drop_allowed[:, unknown_idx] = False
        drop_coverage = (
            drop_probs
            * drop_allowed[:, :, None, None].to(drop_probs.dtype)
        ).amax(dim=1)
        drop_residual = defect.to(drop_probs.dtype) * (1.0 - drop_coverage)
        drop_scores.append(
            top_fraction_score(drop_residual, defect, top_fraction).cpu()
        )

        kept_targets = full_targets * drop_allowed[:, :, None, None].to(
            full_targets.dtype
        )
        full_union = full_targets[:, known_indices].amax(dim=1)
        kept_union = kept_targets[:, known_indices].amax(dim=1)
        pseudo_unknown = full_union * (1.0 - kept_union)
        prediction = (drop_residual >= map_threshold).to(torch.float32)
        intersection = (prediction * pseudo_unknown).sum(dim=(1, 2))
        denominator = prediction.sum(dim=(1, 2)) + pseudo_unknown.sum(dim=(1, 2))
        residual_dice_values.append(
            ((2.0 * intersection + 1.0) / (denominator + 1.0)).cpu()
        )

    known_scores = torch.cat(no_drop_scores).numpy()
    pseudo_scores = torch.cat(drop_scores).numpy()
    labels = np.concatenate(
        [
            np.zeros(len(known_scores), dtype=np.int64),
            np.ones(len(pseudo_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate([known_scores, pseudo_scores])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "residual_dice": float(torch.cat(residual_dice_values).mean().item()),
        "full_condition_score_mean": float(np.mean(known_scores)),
        "drop_condition_score_mean": float(np.mean(pseudo_scores)),
    }


@torch.no_grad()
def score_real_loader(
    explainer: ConditionalAtomicUNet,
    rdp: RDPNet,
    loader: DataLoader,
    device: torch.device,
    known_indices: Sequence[int],
    unknown_idx: int,
    rdp_threshold: float,
    top_fraction: float,
) -> Tuple[np.ndarray, float]:
    explainer.eval()
    rdp.eval()
    all_scores: List[Tensor] = []
    exact_correct = 0
    sample_count = 0

    for x_raw, y_full in loader:
        x_raw = x_raw.to(device, non_blocking=True)
        y_full = y_full.to(device, non_blocking=True)
        inputs, _, defect = model_input(x_raw)
        rdp_probs = frozen_rdp_probabilities(rdp, x_raw, unknown_idx)
        map_probs = torch.sigmoid(explainer(inputs, rdp_probs))

        active = rdp_probs >= rdp_threshold
        active[:, unknown_idx] = False
        gated_maps = map_probs * active[:, :, None, None].to(map_probs.dtype)
        known_coverage = gated_maps.amax(dim=1)
        unknown_heatmap = defect.to(map_probs.dtype) * (1.0 - known_coverage)
        all_scores.append(
            top_fraction_score(unknown_heatmap, defect, top_fraction).cpu()
        )

        predicted_known = active[:, known_indices]
        true_known = y_full[:, known_indices] > 0.5
        exact_correct += int((predicted_known == true_known).all(dim=1).sum().item())
        sample_count += len(x_raw)
    return torch.cat(all_scores).numpy(), exact_correct / max(sample_count, 1)


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    return float(fpr[valid[0]]) if len(valid) else 1.0


def validate_checkpoint(
    checkpoint: Dict,
    unknown_class: str,
    unknown_idx: int,
) -> None:
    checkpoint_unknown = checkpoint.get("unknown_class")
    if checkpoint_unknown is not None and checkpoint_unknown != unknown_class:
        raise ValueError(
            f"RDP checkpoint is {checkpoint_unknown}-held-out, not {unknown_class}-held-out"
        )
    checkpoint_index = checkpoint.get("inactive_unknown_index")
    if checkpoint_index is not None and int(checkpoint_index) != unknown_idx:
        raise ValueError("RDP checkpoint inactive index does not match unknown class")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.truth_condition_prob <= 1.0:
        raise ValueError("truth_condition_prob must be between 0 and 1")
    if not 0.0 <= args.component_dropout_prob <= 1.0:
        raise ValueError("component_dropout_prob must be between 0 and 1")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0,1]")
    if args.samples_per_epoch < 1 or args.validation_samples < 1:
        raise ValueError("synthetic sample counts must be positive")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    unknown_idx = FULL_CLASSES.index(args.unknown_class)
    known_indices = [i for i in range(8) if i != unknown_idx]
    output_dir = Path(
        args.output_dir
        or f"outputs/atomic_explainer_{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y_full = load_mixedwm38(args.data)
    split = make_open_set_split(
        y_full,
        unknown_idx,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    split_counts = {name: int(len(indices)) for name, indices in split.items()}
    combinations = unique_known_combinations(
        y_full, split["known_train"], unknown_idx
    )

    rdp_checkpoint = load_torch_checkpoint(args.rdp_checkpoint, device)
    validate_checkpoint(rdp_checkpoint, args.unknown_class, unknown_idx)
    rdp = RDPNet(output_dim=8, in_channels=1, apply_sigmoid=False).to(device)
    rdp.load_state_dict(rdp_checkpoint.get("model_state_dict", rdp_checkpoint))
    rdp.eval()
    for parameter in rdp.parameters():
        parameter.requires_grad_(False)

    train_dataset = SyntheticAtomicDataset(
        x,
        y_full,
        split["known_train"],
        unknown_idx,
        combinations,
        args.samples_per_epoch,
        args.seed,
    )
    val_dataset = SyntheticAtomicDataset(
        x,
        y_full,
        split["known_train"],
        unknown_idx,
        combinations,
        args.validation_samples,
        args.seed + 7000001,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("device:", device)
    print("unknown_class:", args.unknown_class)
    print("split:", split_counts)
    print("known composition templates:", len(combinations))
    print(
        "known singleton counts:",
        {
            FULL_CLASSES[class_idx]: int(len(train_dataset.by_class[class_idx]))
            for class_idx in known_indices
        },
    )

    explainer = ConditionalAtomicUNet(
        base_channels=args.base_channels, output_channels=8
    ).to(device)
    best_state, best_epoch, best_synthetic_metrics = train_explainer(
        explainer,
        rdp,
        train_dataset,
        train_loader,
        val_loader,
        device,
        known_indices,
        unknown_idx,
        args,
    )
    explainer.load_state_dict(best_state)

    print("evaluating synthetic composition dropout")
    pseudo_unknown_metrics = evaluate_synthetic_pseudo_unknown(
        explainer,
        val_loader,
        device,
        known_indices,
        unknown_idx,
        args.map_threshold,
        args.top_fraction,
    )

    real_loaders = {
        name: make_real_loader(
            x,
            y_full,
            indices,
            args.batch_size,
            args.num_workers,
            pin_memory,
        )
        for name, indices in split.items()
        if name != "known_train"
    }
    print("scoring known validation set")
    val_scores, _ = score_real_loader(
        explainer,
        rdp,
        real_loaders["known_val"],
        device,
        known_indices,
        unknown_idx,
        args.rdp_threshold,
        args.top_fraction,
    )
    threshold = float(np.quantile(val_scores, 1.0 - args.known_reject_rate))

    print("scoring known test set")
    known_scores, known_component_exact = score_real_loader(
        explainer,
        rdp,
        real_loaders["known_test"],
        device,
        known_indices,
        unknown_idx,
        args.rdp_threshold,
        args.top_fraction,
    )
    print("scoring unknown test set")
    unknown_scores, unknown_known_component_exact = score_real_loader(
        explainer,
        rdp,
        real_loaders["unknown_test"],
        device,
        known_indices,
        unknown_idx,
        args.rdp_threshold,
        args.top_fraction,
    )

    labels = np.concatenate(
        [
            np.zeros(len(known_scores), dtype=np.int64),
            np.ones(len(unknown_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate([known_scores, unknown_scores])
    known_accept = float(np.mean(known_scores <= threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))
    metrics = {
        "configuration": {
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_idx,
            "known_classes": [FULL_CLASSES[i] for i in known_indices],
            "rdp_checkpoint": str(Path(args.rdp_checkpoint)),
            "seed": args.seed,
            "rdp_threshold": args.rdp_threshold,
            "map_threshold": args.map_threshold,
            "top_fraction": args.top_fraction,
            "known_reject_rate": args.known_reject_rate,
            "truth_condition_prob": args.truth_condition_prob,
            "component_dropout_prob": args.component_dropout_prob,
            "pseudo_unknown_weight": args.pseudo_unknown_weight,
        },
        "split_counts": split_counts,
        "synthetic_training": {
            "best_epoch": best_epoch,
            "known_composition_template_count": len(combinations),
            "best_validation": best_synthetic_metrics,
            "pseudo_unknown_validation": pseudo_unknown_metrics,
        },
        "rdp_known_component_exact_match": {
            "known_test": known_component_exact,
            "unknown_test_known_projection": unknown_known_component_exact,
        },
        "unknown_detection": {
            "auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores)),
            "fpr95": fpr_at_tpr95(labels, scores),
            "threshold": threshold,
            "known_accept_rate": known_accept,
            "unknown_recall": unknown_recall,
            "balanced_accuracy_at_threshold": 0.5 * (known_accept + unknown_recall),
            "known_score_mean": float(np.mean(known_scores)),
            "unknown_score_mean": float(np.mean(unknown_scores)),
        },
    }

    payload = {
        "model_state_dict": explainer.state_dict(),
        "model_name": "ConditionalAtomicUNet",
        "base_channels": args.base_channels,
        "full_class_order": list(FULL_CLASSES),
        "known_indices": known_indices,
        "unknown_class": args.unknown_class,
        "unknown_index": unknown_idx,
        "rdp_threshold": args.rdp_threshold,
        "map_threshold": args.map_threshold,
        "top_fraction": args.top_fraction,
        "unknown_threshold": threshold,
        "rdp_checkpoint": str(Path(args.rdp_checkpoint)),
    }
    torch.save(payload, output_dir / "best_atomic_explainer.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved:", output_dir / "best_atomic_explainer.pt")
    print("saved:", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
