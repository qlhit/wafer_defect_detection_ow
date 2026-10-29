#!/usr/bin/env python3
"""Label-wise reciprocal rejection for a frozen leave-one-out RDP-Net.

This is a deep, wafer-specific adaptation of the reciprocal/open-space idea.
The original RDP-Net and its multi-label logits are frozen.  A small verifier
learns one label-specific spatial token and a positive/negative reciprocal
pair for every active known class, using known training wafers only.

At inference, class k is reported only when both conditions hold:

    sigmoid(rdp_logit[k]) >= label_threshold
    verifier says the label-specific feature lies in class k's known region

The held-out class is never used for training, validation, early stopping, or
threshold calibration.  It is read only once for final diagnostic evaluation.

Only two files are written:
  - best_reciprocal_verifier.pt
  - metrics.json

Place this file in rdp_net_repro/scripts/ and run it from the repository root.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


FULL_CLASSES: Tuple[str, ...] = ("C", "D", "EL", "ER", "L", "NF", "S", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate label-wise reciprocal rejection for frozen RDP-Net"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Default: checkpoint/rdp_leave_one_out/<class>_heldout/best_open_set.pt",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--logit-scale", type=float, default=10.0)
    parser.add_argument("--reciprocal-margin", type=float, default=0.20)
    parser.add_argument("--compactness-weight", type=float, default=0.10)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--label-threshold", type=float, default=0.50)
    parser.add_argument(
        "--class-accept-rate",
        type=float,
        default=0.995,
        help="Per-gate acceptance quantile on true known-validation positives",
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def locate_rdp_source() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (here / "src", here.parent / "src")
    for source in candidates:
        if source.is_dir():
            return source
    raise FileNotFoundError(
        "Cannot find RDP src/. Put this file in rdp_net_repro/ or rdp_net_repro/scripts/."
    )


SOURCE = locate_rdp_source()
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from rdpnet.models.rdpnet import RDPNet  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_arrays(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = np.load(path)
    if "arr_0" not in raw:
        raise ValueError(f"{path} has no arr_0")
    x = torch.from_numpy(raw["arr_0"].astype(np.float32))
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    label_key = next(
        (
            key
            for key in ("arr_1", "arr_2")
            if key in raw
            and raw[key].ndim == 2
            and raw[key].shape[1] == len(FULL_CLASSES)
        ),
        None,
    )
    if x.ndim != 3 or label_key is None:
        raise ValueError("Expected arr_0=[N,H,W] and an [N,8] label array")
    y = torch.from_numpy(raw[label_key].astype(np.float32))
    if len(x) != len(y):
        raise ValueError("Wafer and label sample counts differ")
    return x, (y > 0.5).float()


def label_codes(labels: torch.Tensor) -> np.ndarray:
    values = labels.to(torch.int64).cpu().numpy()
    return np.asarray(["".join(map(str, row.tolist())) for row in values])


def make_split(
    y: torch.Tensor,
    unknown_index: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    indices = np.arange(len(y))
    unknown_mask = y[:, unknown_index].cpu().numpy() > 0.5
    known_indices = indices[~unknown_mask]
    unknown_indices = indices[unknown_mask]
    active = [index for index in range(len(FULL_CLASSES)) if index != unknown_index]
    codes = label_codes(y[known_indices][:, active])
    train_indices, remainder = train_test_split(
        known_indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=codes,
    )
    remainder_codes = label_codes(y[remainder][:, active])
    val_fraction = val_ratio / (1.0 - train_ratio)
    val_indices, test_indices = train_test_split(
        remainder,
        train_size=val_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=remainder_codes,
    )
    return {
        "known_train": np.asarray(train_indices),
        "known_val": np.asarray(val_indices),
        "known_test": np.asarray(test_indices),
        "unknown_test": np.asarray(unknown_indices),
    }


def load_frozen_rdp(
    checkpoint_path: Path,
    unknown_class: str,
    device: torch.device,
) -> Tuple[nn.Module, List[int], Dict[str, object]]:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    unknown_index = FULL_CLASSES.index(unknown_class)
    expected_active = [i for i in range(len(FULL_CLASSES)) if i != unknown_index]
    stored_unknown = checkpoint.get("unknown_class")
    if stored_unknown is not None and stored_unknown != unknown_class:
        raise ValueError(
            f"Checkpoint holds out {stored_unknown}, requested {unknown_class}"
        )
    active = [
        int(i) for i in checkpoint.get("active_known_indices", expected_active)
    ]
    if active != expected_active:
        raise ValueError(f"Expected active indices {expected_active}, got {active}")
    model = RDPNet(
        output_dim=len(FULL_CLASSES), in_channels=1, apply_sigmoid=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, active, checkpoint


@torch.inference_mode()
def forward_frozen_rdp(
    model: nn.Module, wafer: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = model.conv1(wafer)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    z = model.block_1_1(z)
    z = model.block_1_2(z)
    z = model.block_2_1(z)
    z = model.block_2_2(z)
    z = model.block_3_1(z)
    block3 = model.block_3_2(z)
    z = model.block_4_1(block3)
    z = model.block_4_2(z)
    features = torch.flatten(model.avgpool(z), 1)
    logits = model.fc(features)
    return logits, block3


@torch.inference_mode()
def extract_frozen_features(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maps: List[torch.Tensor] = []
    probabilities: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    print(f"extracting frozen RDP features: {name} ({len(indices)})")
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        wafer = x[batch_indices].unsqueeze(1).to(device, non_blocking=True)
        logits, feature_map = forward_frozen_rdp(model, wafer)
        maps.append(feature_map.cpu().to(torch.float16))
        probabilities.append(
            torch.sigmoid(logits[:, active_indices]).cpu().to(torch.float32)
        )
        targets.append(y[batch_indices][:, active_indices].cpu().to(torch.float32))
    return torch.cat(maps), torch.cat(probabilities), torch.cat(targets)


class ReciprocalComponentVerifier(nn.Module):
    """Label-specific attention tokens with positive/negative reciprocal anchors."""

    def __init__(
        self,
        class_count: int,
        feature_channels: int,
        embedding_dim: int,
        logit_scale: float,
    ) -> None:
        super().__init__()
        self.class_count = class_count
        self.feature_channels = feature_channels
        self.embedding_dim = embedding_dim
        self.logit_scale = float(logit_scale)

        self.queries = nn.Parameter(torch.empty(class_count, feature_channels))
        self.projection = nn.Parameter(
            torch.empty(class_count, feature_channels, embedding_dim)
        )
        self.projection_bias = nn.Parameter(torch.zeros(class_count, embedding_dim))
        self.positive_anchor = nn.Parameter(torch.empty(class_count, embedding_dim))
        self.negative_anchor = nn.Parameter(torch.empty(class_count, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, std=0.02)
        nn.init.xavier_uniform_(self.projection)
        nn.init.normal_(self.positive_anchor, std=0.02)
        nn.init.normal_(self.negative_anchor, std=0.02)

    def forward(
        self, feature_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # feature_map: [B,C,H,W]. Each class query extracts its own spatial token.
        feature_map = F.normalize(feature_map.float(), dim=1)
        queries = F.normalize(self.queries, dim=1)
        attention_logits = torch.einsum(
            "bchw,kc->bkhw", feature_map, queries
        ) / math.sqrt(self.feature_channels)
        attention = torch.softmax(attention_logits.flatten(2), dim=-1).view_as(
            attention_logits
        )
        tokens = torch.einsum("bkhw,bchw->bkc", attention, feature_map)
        embeddings = torch.einsum(
            "bkc,kce->bke", tokens, self.projection
        ) + self.projection_bias.unsqueeze(0)
        embeddings = F.normalize(embeddings, dim=-1)

        positive = F.normalize(self.positive_anchor, dim=-1)
        negative = F.normalize(self.negative_anchor, dim=-1)
        positive_similarity = torch.einsum("bke,ke->bk", embeddings, positive)
        negative_similarity = torch.einsum("bke,ke->bk", embeddings, negative)
        relative_logits = self.logit_scale * (
            positive_similarity - negative_similarity
        )
        positive_distance = 1.0 - positive_similarity
        negative_distance = 1.0 - negative_similarity
        return (
            relative_logits,
            positive_distance,
            negative_distance,
            attention,
        )


def balanced_binary_mean(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average positive and negative losses equally within every class."""
    positive_count = targets.sum(dim=0).clamp_min(1.0)
    negative_count = (1.0 - targets).sum(dim=0).clamp_min(1.0)
    positive = (values * targets).sum(dim=0) / positive_count
    negative = (values * (1.0 - targets)).sum(dim=0) / negative_count
    return 0.5 * (positive + negative).mean()


def verifier_loss(
    verifier: ReciprocalComponentVerifier,
    feature_maps: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
    compactness_weight: float,
    margin_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits, positive_distance, negative_distance, _ = verifier(feature_maps)
    bce_values = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    bce = balanced_binary_mean(bce_values, targets)

    signed = targets * 2.0 - 1.0
    relative_similarity = logits / verifier.logit_scale
    margin_values = F.relu(margin - signed * relative_similarity)
    margin_loss = balanced_binary_mean(margin_values, targets)

    correct_anchor_distance = (
        targets * positive_distance + (1.0 - targets) * negative_distance
    )
    compactness = balanced_binary_mean(correct_anchor_distance, targets)
    total = bce + margin_weight * margin_loss + compactness_weight * compactness
    return total, {
        "total": float(total.detach().item()),
        "bce": float(bce.detach().item()),
        "margin": float(margin_loss.detach().item()),
        "compactness": float(compactness.detach().item()),
    }


@torch.inference_mode()
def evaluate_loss(
    verifier: ReciprocalComponentVerifier,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    compactness_weight: float,
    margin_weight: float,
) -> Dict[str, float]:
    verifier.eval()
    sums = {"total": 0.0, "bce": 0.0, "margin": 0.0, "compactness": 0.0}
    count = 0
    for feature_maps, targets in loader:
        feature_maps = feature_maps.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, parts = verifier_loss(
            verifier,
            feature_maps,
            targets,
            margin,
            compactness_weight,
            margin_weight,
        )
        batch_n = len(feature_maps)
        for key in sums:
            sums[key] += parts[key] * batch_n
        count += batch_n
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_verifier(
    verifier: ReciprocalComponentVerifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        verifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    best_state = copy.deepcopy(verifier.state_dict())
    best_epoch = 0
    best_val = {"total": float("inf")}
    stale = 0

    for epoch in range(1, args.epochs + 1):
        verifier.train()
        running = 0.0
        sample_count = 0
        for feature_maps, targets in train_loader:
            feature_maps = feature_maps.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss, _ = verifier_loss(
                verifier,
                feature_maps,
                targets,
                args.reciprocal_margin,
                args.compactness_weight,
                args.margin_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item()) * len(feature_maps)
            sample_count += len(feature_maps)
        scheduler.step()

        validation = evaluate_loss(
            verifier,
            val_loader,
            device,
            args.reciprocal_margin,
            args.compactness_weight,
            args.margin_weight,
        )
        improved = validation["total"] < best_val["total"] - 1e-6
        if improved:
            best_state = copy.deepcopy(verifier.state_dict())
            best_epoch = epoch
            best_val = validation
            stale = 0
        else:
            stale += 1
        marker = " *" if improved else ""
        print(
            f"epoch {epoch:03d} train={running / max(sample_count, 1):.6f} "
            f"val={validation['total']:.6f} bce={validation['bce']:.6f} "
            f"margin={validation['margin']:.6f} "
            f"compact={validation['compactness']:.6f} "
            f"patience={stale}/{args.patience}{marker}"
        )
        if args.patience > 0 and stale >= args.patience:
            break
    return best_state, best_epoch, best_val


@torch.inference_mode()
def collect_verifier_scores(
    verifier: ReciprocalComponentVerifier,
    feature_maps: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    verifier.eval()
    relative_parts: List[np.ndarray] = []
    distance_parts: List[np.ndarray] = []
    for start in range(0, len(feature_maps), batch_size):
        batch = feature_maps[start : start + batch_size].to(
            device, non_blocking=True
        )
        relative, positive_distance, _, _ = verifier(batch)
        relative_parts.append(relative.cpu().numpy())
        distance_parts.append(positive_distance.cpu().numpy())
    return np.concatenate(relative_parts), np.concatenate(distance_parts)


def calibrate_open_thresholds(
    relative_logits: np.ndarray,
    positive_distances: np.ndarray,
    targets: np.ndarray,
    accept_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    relative_floors = []
    distance_ceilings = []
    for class_index in range(targets.shape[1]):
        positive = targets[:, class_index].astype(bool)
        if not positive.any():
            raise ValueError(
                f"Known validation contains no positive sample for local class {class_index}"
            )
        relative_floors.append(
            float(np.quantile(relative_logits[positive, class_index], 1.0 - accept_rate))
        )
        distance_ceilings.append(
            float(np.quantile(positive_distances[positive, class_index], accept_rate))
        )
    return (
        np.asarray(relative_floors, dtype=np.float32),
        np.asarray(distance_ceilings, dtype=np.float32),
    )


def predictions_for_schemes(
    probabilities: np.ndarray,
    relative_logits: np.ndarray,
    positive_distances: np.ndarray,
    label_threshold: float,
    relative_floors: np.ndarray,
    distance_ceilings: np.ndarray,
) -> Dict[str, np.ndarray]:
    base = probabilities >= label_threshold
    relative = relative_logits >= relative_floors.reshape(1, -1)
    distance = positive_distances <= distance_ceilings.reshape(1, -1)
    return {
        "rdp_only": base,
        "rdp_plus_distance": base & distance,
        "rdp_plus_relative": base & relative,
        "rdp_plus_both": base & distance & relative,
    }


def multilabel_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    predictions = predictions.astype(bool)
    targets = targets.astype(bool)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="micro", zero_division=0
    )
    false_positive = np.sum(predictions & ~targets, axis=0)
    false_negative = np.sum(~predictions & targets, axis=0)
    negative = np.sum(~targets, axis=0)
    pure = targets.sum(axis=1) == 0
    supported = support > 0
    return {
        "samples": int(len(targets)),
        "exact_match": float(np.mean(np.all(predictions == targets, axis=1))),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "supported_class_macro_f1": (
            float(np.mean(f1[supported])) if supported.any() else None
        ),
        "mean_false_positives_per_sample": float(false_positive.sum() / len(targets)),
        "mean_false_negatives_per_sample": float(false_negative.sum() / len(targets)),
        "pure_unknown_samples": int(pure.sum()),
        "pure_unknown_false_known_rate": (
            float(np.mean(predictions[pure].any(axis=1))) if pure.any() else None
        ),
        "per_class": {
            name: {
                "positive_samples": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "false_positive": int(false_positive[index]),
                "false_positive_rate": (
                    float(false_positive[index] / negative[index])
                    if negative[index]
                    else None
                ),
            }
            for index, name in enumerate(class_names)
        },
    }


def evaluate_all_schemes(
    probabilities: np.ndarray,
    relative_logits: np.ndarray,
    positive_distances: np.ndarray,
    targets: np.ndarray,
    label_threshold: float,
    relative_floors: np.ndarray,
    distance_ceilings: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    schemes = predictions_for_schemes(
        probabilities,
        relative_logits,
        positive_distances,
        label_threshold,
        relative_floors,
        distance_ceilings,
    )
    return {
        name: multilabel_metrics(prediction, targets, class_names)
        for name, prediction in schemes.items()
    }


def main() -> None:
    args = parse_args()
    if not 0.5 < args.class_accept_rate < 1.0:
        raise ValueError("--class-accept-rate must be in (0.5,1.0)")
    if args.embedding_dim <= 0 or args.batch_size <= 0:
        raise ValueError("Embedding and batch sizes must be positive")
    set_seed(args.seed)

    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [i for i in range(len(FULL_CLASSES)) if i != unknown_index]
    class_names = [FULL_CLASSES[i] for i in active_indices]
    base_checkpoint = Path(
        args.base_checkpoint
        or f"checkpoint/rdp_leave_one_out/{args.unknown_class.lower()}_heldout/best_open_set.pt"
    ).resolve()
    output_dir = Path(
        args.output_dir
        or f"checkpoint/rdp_reciprocal_rejection/{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"unknown={args.unknown_class}; active known={class_names}")
    print(f"loading frozen RDP: {base_checkpoint}")
    x, y = load_arrays(args.data)
    split = make_split(
        y,
        unknown_index,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    base_model, stored_active, _ = load_frozen_rdp(
        base_checkpoint, args.unknown_class, device
    )
    if stored_active != active_indices:
        raise ValueError(f"Active class mismatch: {stored_active} vs {active_indices}")

    # Unknown data is intentionally not extracted until training, model selection,
    # and known-only threshold calibration have all finished.
    train_maps, train_probabilities, train_targets = extract_frozen_features(
        base_model,
        x,
        y,
        split["known_train"],
        active_indices,
        args.extract_batch_size,
        device,
        "known_train",
    )
    val_maps, val_probabilities, val_targets = extract_frozen_features(
        base_model,
        x,
        y,
        split["known_val"],
        active_indices,
        args.extract_batch_size,
        device,
        "known_val",
    )
    print(f"cached block_3_2 feature shape: {list(train_maps.shape[1:])}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_maps, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(val_maps, val_targets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    verifier = ReciprocalComponentVerifier(
        class_count=len(active_indices),
        feature_channels=int(train_maps.shape[1]),
        embedding_dim=args.embedding_dim,
        logit_scale=args.logit_scale,
    ).to(device)
    best_state, best_epoch, best_validation = train_verifier(
        verifier, train_loader, val_loader, device, args
    )
    verifier.load_state_dict(best_state)

    val_relative, val_distances = collect_verifier_scores(
        verifier, val_maps, args.batch_size, device
    )
    relative_floors, distance_ceilings = calibrate_open_thresholds(
        val_relative,
        val_distances,
        val_targets.numpy(),
        args.class_accept_rate,
    )

    # Only now is held-out data touched.
    test_maps, test_probabilities, test_targets = extract_frozen_features(
        base_model,
        x,
        y,
        split["known_test"],
        active_indices,
        args.extract_batch_size,
        device,
        "known_test",
    )
    unknown_maps, unknown_probabilities, unknown_targets = extract_frozen_features(
        base_model,
        x,
        y,
        split["unknown_test"],
        active_indices,
        args.extract_batch_size,
        device,
        "unknown_test",
    )
    test_relative, test_distances = collect_verifier_scores(
        verifier, test_maps, args.batch_size, device
    )
    unknown_relative, unknown_distances = collect_verifier_scores(
        verifier, unknown_maps, args.batch_size, device
    )

    known_results = evaluate_all_schemes(
        test_probabilities.numpy(),
        test_relative,
        test_distances,
        test_targets.numpy(),
        args.label_threshold,
        relative_floors,
        distance_ceilings,
        class_names,
    )
    unknown_results = evaluate_all_schemes(
        unknown_probabilities.numpy(),
        unknown_relative,
        unknown_distances,
        unknown_targets.numpy(),
        args.label_threshold,
        relative_floors,
        distance_ceilings,
        class_names,
    )

    metrics = {
        "configuration": {
            "method": "frozen_rdp_labelwise_reciprocal_rejection",
            "full_class_order": list(FULL_CLASSES),
            "known_classes": class_names,
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_index,
            "base_checkpoint": str(base_checkpoint),
            "base_rdp_frozen": True,
            "heldout_data_used_for_training_selection_or_calibration": False,
            "feature_layer": "block_3_2",
            "embedding_dim": args.embedding_dim,
            "logit_scale": args.logit_scale,
            "reciprocal_margin": args.reciprocal_margin,
            "compactness_weight": args.compactness_weight,
            "margin_weight": args.margin_weight,
            "label_threshold": args.label_threshold,
            "class_accept_rate_per_gate": args.class_accept_rate,
            "seed": args.seed,
        },
        "split_counts": {name: int(len(value)) for name, value in split.items()},
        "training": {
            "best_epoch": best_epoch,
            "best_known_validation_loss": best_validation,
        },
        "calibration": {
            name: {
                "relative_logit_floor": float(relative_floors[index]),
                "positive_distance_ceiling": float(distance_ceilings[index]),
            }
            for index, name in enumerate(class_names)
        },
        "known_test_components": known_results,
        "unknown_test_known_projection": unknown_results,
        "recommended_primary_scheme": "rdp_plus_both",
        "notes": {
            "unknown_gate": "Dinomaly remains the final wafer-level unknown detector",
            "verifier_role": "suppress unsupported known-component predictions, not name the unknown class",
            "scheme_selection": "all schemes are reported; do not choose using held-out unknown labels",
        },
    }

    checkpoint = {
        "verifier_state_dict": verifier.state_dict(),
        "verifier_configuration": {
            "class_count": len(active_indices),
            "feature_channels": int(train_maps.shape[1]),
            "embedding_dim": args.embedding_dim,
            "logit_scale": args.logit_scale,
            "feature_layer": "block_3_2",
        },
        "base_checkpoint": str(base_checkpoint),
        "full_class_order": list(FULL_CLASSES),
        "active_known_indices": active_indices,
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "label_threshold": args.label_threshold,
        "relative_logit_floors": torch.from_numpy(relative_floors),
        "positive_distance_ceilings": torch.from_numpy(distance_ceilings),
        "primary_scheme": "rdp_plus_both",
    }
    torch.save(checkpoint, output_dir / "best_reciprocal_verifier.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved:", output_dir / "best_reciprocal_verifier.pt")
    print("saved:", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
