#!/usr/bin/env python3
"""Train contamination-robust RDP-Net for atomic leave-one-out evaluation.

The RDP architecture and eight-output physical head are unchanged. Wafers
containing the held-out class are never used for training, validation, model
selection, pseudo-unknown generation, or threshold calibration. During known
training, irregular synthetic contamination is added online while retaining
the original known-component target. Pure synthetic contamination is trained
against an all-zero active-known target.

Only two files are written:
  - best_model.pt
  - metrics.json
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
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


FULL_CLASSES: Tuple[str, ...] = ("C", "D", "EL", "ER", "L", "NF", "S", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contamination-robust leave-one-out RDP-Net"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3529)

    parser.add_argument("--pseudo-batch-fraction", type=float, default=0.50)
    parser.add_argument("--pseudo-min-die-fraction", type=float, default=0.02)
    parser.add_argument("--pseudo-max-die-fraction", type=float, default=0.12)
    parser.add_argument("--contamination-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--pure-reject-loss-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: checkpoint/rdp_contamination_robust/<class>_heldout",
    )
    return parser.parse_args()


def locate_rdp_source() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (here / "src", here.parent / "src")
    for source in candidates:
        if source.is_dir():
            return source
    raise FileNotFoundError(
        "Cannot find RDP src/. Place this file in rdp_net_repro/ or rdp_net_repro/scripts/."
    )


SOURCE = locate_rdp_source()
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from rdpnet.losses import AsymmetricLoss  # noqa: E402
from rdpnet.models.rdpnet import RDPNet  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(x[index_tensor], y[index_tensor])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def forward_features(
    model: RDPNet, wafer: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = model.conv1(wafer)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.block_1_1(x)
    x = model.block_1_2(x)
    x = model.block_2_1(x)
    x = model.block_2_2(x)
    x = model.block_3_1(x)
    x = model.block_3_2(x)
    x = model.block_4_1(x)
    x = model.block_4_2(x)
    features = torch.flatten(model.avgpool(x), 1)
    return model.fc(features), features


def normalized_asl(
    criterion: nn.Module, logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return criterion(logits, targets) / max(1, targets.numel())


def data_values(x: torch.Tensor) -> Tuple[float, float, float]:
    maximum = float(x.max().item())
    if maximum > 1.5:
        return 0.5, 1.0, 2.0
    return 0.25, 0.5, 1.0


def irregular_masks(
    wafer: torch.Tensor,
    eligible: torch.Tensor,
    min_fraction: float,
    max_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size, _, height, width = wafer.shape
    device = wafer.device
    noise = torch.rand(
        (batch_size, 1, height, width), device=device, generator=generator
    )
    smooth3 = F.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
    smooth7 = F.avg_pool2d(noise, kernel_size=7, stride=1, padding=3)

    yy = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width)
    center_x1 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        -0.75, 0.75, generator=generator
    )
    center_y1 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        -0.75, 0.75, generator=generator
    )
    center_x2 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        -0.75, 0.75, generator=generator
    )
    center_y2 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        -0.75, 0.75, generator=generator
    )
    scale_x1 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        0.12, 0.50, generator=generator
    )
    scale_y1 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        0.12, 0.50, generator=generator
    )
    scale_x2 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        0.10, 0.40, generator=generator
    )
    scale_y2 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
        0.10, 0.40, generator=generator
    )
    envelope1 = torch.exp(
        -0.5
        * (
            ((xx - center_x1) / scale_x1).square()
            + ((yy - center_y1) / scale_y1).square()
        )
    )
    envelope2 = torch.exp(
        -0.5
        * (
            ((xx - center_x2) / scale_x2).square()
            + ((yy - center_y2) / scale_y2).square()
        )
    )
    envelope = torch.maximum(envelope1, envelope2)
    score = (0.45 * smooth3 + 0.35 * smooth7 + 0.20 * noise) * (
        0.25 + 0.75 * envelope
    )
    score = score.masked_fill(~eligible, float("-inf"))

    fractions = torch.empty(batch_size, device=device).uniform_(
        min_fraction, max_fraction, generator=generator
    )
    masks = torch.zeros_like(eligible)
    for index in range(batch_size):
        eligible_positions = torch.nonzero(
            eligible[index].flatten(), as_tuple=False
        ).flatten()
        if len(eligible_positions) == 0:
            continue
        valid_count = int((wafer[index] > 0).sum().item())
        keep = max(1, int(round(valid_count * float(fractions[index].item()))))
        keep = min(keep, len(eligible_positions))
        selected_values = score[index].flatten()[eligible_positions]
        chosen = eligible_positions[
            torch.topk(selected_values, k=keep).indices
        ]
        masks[index].view(-1)[chosen] = True
    return masks


def synthesize_pseudo_unknown(
    wafer: torch.Tensor,
    min_fraction: float,
    max_fraction: float,
    generator: torch.Generator,
    valid_threshold: float,
    normal_value: float,
    defect_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = wafer > valid_threshold
    existing_defect = wafer > (normal_value + defect_value) / 2.0
    normal = valid & ~existing_defect
    contaminated_mask = irregular_masks(
        wafer, normal, min_fraction, max_fraction, generator
    )
    contaminated = wafer.clone()
    contaminated[contaminated_mask] = defect_value

    pure = wafer.clone()
    pure[valid] = normal_value
    pure_eligible = valid
    pure_mask = irregular_masks(
        pure, pure_eligible, min_fraction, max_fraction, generator
    )
    pure[pure_mask] = defect_value
    return contaminated, pure


@torch.inference_mode()
def validation_metrics(
    model: RDPNet,
    loader: DataLoader,
    active_indices: Sequence[int],
    device: torch.device,
    label_threshold: float,
    args: argparse.Namespace,
    valid_threshold: float,
    normal_value: float,
    defect_value: float,
) -> Dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 10001)
    clean_correct = contaminated_correct = 0
    pure_false_known = 0
    count = 0
    for wafer, labels in loader:
        wafer = wafer.unsqueeze(1).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        contaminated, pure = synthesize_pseudo_unknown(
            wafer,
            args.pseudo_min_die_fraction,
            args.pseudo_max_die_fraction,
            generator,
            valid_threshold,
            normal_value,
            defect_value,
        )
        clean_logits, _ = forward_features(model, wafer)
        contaminated_logits, _ = forward_features(model, contaminated)
        pure_logits, _ = forward_features(model, pure)
        truth = labels[:, active_indices] > 0.5
        clean_prediction = (
            torch.sigmoid(clean_logits[:, active_indices]) >= label_threshold
        )
        contaminated_prediction = (
            torch.sigmoid(contaminated_logits[:, active_indices]) >= label_threshold
        )
        pure_prediction = (
            torch.sigmoid(pure_logits[:, active_indices]) >= label_threshold
        )
        clean_correct += int((clean_prediction == truth).all(dim=1).sum().item())
        contaminated_correct += int(
            (contaminated_prediction == truth).all(dim=1).sum().item()
        )
        pure_false_known += int(pure_prediction.any(dim=1).sum().item())
        count += len(wafer)
    clean_exact = clean_correct / max(1, count)
    contaminated_exact = contaminated_correct / max(1, count)
    pure_false_rate = pure_false_known / max(1, count)
    selection_score = (
        0.5 * clean_exact + 0.5 * contaminated_exact - 0.25 * pure_false_rate
    )
    return {
        "clean_exact_match": clean_exact,
        "contaminated_exact_match": contaminated_exact,
        "pure_pseudo_false_known_rate": pure_false_rate,
        "selection_score": selection_score,
    }


@torch.inference_mode()
def collect_predictions(
    model: RDPNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    logits_parts: List[torch.Tensor] = []
    feature_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    for wafer, labels in loader:
        wafer = wafer.unsqueeze(1).to(device, non_blocking=True)
        logits, features = forward_features(model, wafer)
        logits_parts.append(logits.cpu())
        feature_parts.append(features.cpu())
        target_parts.append(labels.cpu())
    return (
        torch.cat(logits_parts),
        torch.cat(feature_parts),
        torch.cat(target_parts),
    )


def multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    active_indices: Sequence[int],
    class_names: Sequence[str],
    threshold: float,
) -> Dict[str, object]:
    probabilities = torch.sigmoid(logits[:, active_indices]).numpy()
    predictions = probabilities >= threshold
    truth = targets[:, active_indices].numpy() > 0.5
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        truth, predictions, average="micro", zero_division=0
    )
    supported = support > 0
    false_positive = predictions & ~truth
    false_negative = ~predictions & truth
    pure = ~truth.any(axis=1)
    return {
        "exact_match": float(np.mean(np.all(predictions == truth, axis=1))),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "supported_class_macro_f1": (
            float(np.mean(f1[supported])) if supported.any() else None
        ),
        "mean_false_positives_per_sample": float(
            false_positive.sum(axis=1).mean()
        ),
        "mean_false_negatives_per_sample": float(
            false_negative.sum(axis=1).mean()
        ),
        "pure_unknown_false_known_rate": (
            float(predictions[pure].any(axis=1).mean()) if pure.any() else None
        ),
        "per_class": {
            name: {
                "positive_samples": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "false_positive": int(false_positive[:, index].sum()),
                "false_positive_rate": float(
                    false_positive[:, index].sum()
                    / max(1, (~truth[:, index]).sum())
                ),
            }
            for index, name in enumerate(class_names)
        },
    }


def build_prototypes(
    features: torch.Tensor,
    targets: torch.Tensor,
    active_indices: Sequence[int],
) -> Tuple[torch.Tensor, List[str]]:
    normalized = F.normalize(features, dim=1)
    codes = label_codes(targets[:, active_indices])
    prototypes = []
    prototype_codes = []
    for code in sorted(np.unique(codes).tolist()):
        mask = torch.from_numpy(codes == code)
        prototype = F.normalize(
            normalized[mask].mean(dim=0, keepdim=True), dim=1
        )
        prototypes.append(prototype.squeeze(0))
        prototype_codes.append(code)
    return torch.stack(prototypes), prototype_codes


def prototype_score(features: torch.Tensor, prototypes: torch.Tensor) -> np.ndarray:
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    return (1.0 - (features @ prototypes.T).max(dim=1).values).numpy()


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    indices = np.flatnonzero(tpr >= 0.95)
    return float(fpr[indices[0]]) if len(indices) else 1.0


def main() -> None:
    args = parse_args()
    if not 0.0 < args.pseudo_batch_fraction <= 1.0:
        raise ValueError("--pseudo-batch-fraction must be in (0,1]")
    if not 0.0 < args.pseudo_min_die_fraction <= args.pseudo_max_die_fraction < 1.0:
        raise ValueError("Invalid pseudo die-fraction range")
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    x, y = load_arrays(args.data)
    valid_threshold, normal_value, defect_value = data_values(x)
    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    class_names = [FULL_CLASSES[index] for index in active_indices]
    split = make_split(
        y,
        unknown_index,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    loaders = {
        name: make_loader(
            x,
            y,
            indices,
            args.batch_size,
            shuffle=name == "known_train",
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            seed=args.seed + offset,
        )
        for offset, (name, indices) in enumerate(split.items())
    }
    output_dir = Path(
        args.output_dir
        or f"checkpoint/rdp_contamination_robust/{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"

    print(f"device: {device}")
    print(
        f"unknown={args.unknown_class}; active known={class_names}; "
        + ", ".join(f"{name}={len(indices)}" for name, indices in split.items()),
        flush=True,
    )
    model = RDPNet(
        output_dim=len(FULL_CLASSES), in_channels=1, apply_sigmoid=False
    ).to(device)
    criterion = AsymmetricLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_generator = torch.Generator(device=device).manual_seed(args.seed + 5001)

    best_score = float("-inf")
    best_epoch = 0
    best_validation: Dict[str, float] = {}
    best_state = copy.deepcopy(model.state_dict())
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"total": 0.0, "clean": 0.0, "contam": 0.0, "cons": 0.0, "reject": 0.0}
        sample_count = 0
        for wafer, labels in loaders["known_train"]:
            wafer = wafer.unsqueeze(1).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pseudo_count = max(
                1, min(len(wafer), int(round(len(wafer) * args.pseudo_batch_fraction)))
            )
            contaminated, pure = synthesize_pseudo_unknown(
                wafer[:pseudo_count],
                args.pseudo_min_die_fraction,
                args.pseudo_max_die_fraction,
                train_generator,
                valid_threshold,
                normal_value,
                defect_value,
            )

            clean_logits, _ = forward_features(model, wafer)
            pseudo_inputs = torch.cat([contaminated, pure], dim=0)
            pseudo_logits, _ = forward_features(model, pseudo_inputs)
            contaminated_logits = pseudo_logits[:pseudo_count]
            pure_logits = pseudo_logits[pseudo_count:]
            clean_active = clean_logits[:, active_indices]
            contaminated_active = contaminated_logits[:, active_indices]
            pure_active = pure_logits[:, active_indices]
            clean_targets = labels[:, active_indices]
            contaminated_targets = clean_targets[:pseudo_count]

            clean_loss = normalized_asl(criterion, clean_active, clean_targets)
            contamination_loss = normalized_asl(
                criterion, contaminated_active, contaminated_targets
            )
            consistency_loss = F.mse_loss(
                torch.sigmoid(contaminated_active),
                torch.sigmoid(clean_active[:pseudo_count]).detach(),
            )
            pure_reject_loss = F.softplus(pure_active).mean()
            loss = (
                clean_loss
                + args.contamination_loss_weight * contamination_loss
                + args.consistency_loss_weight * consistency_loss
                + args.pure_reject_loss_weight * pure_reject_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = len(wafer)
            sample_count += batch_size
            for key, value in (
                ("total", loss),
                ("clean", clean_loss),
                ("contam", contamination_loss),
                ("cons", consistency_loss),
                ("reject", pure_reject_loss),
            ):
                totals[key] += float(value.item()) * batch_size

        validation = validation_metrics(
            model,
            loaders["known_val"],
            active_indices,
            device,
            args.label_threshold,
            args,
            valid_threshold,
            normal_value,
            defect_value,
        )
        improved = validation["selection_score"] > best_score
        if improved:
            best_score = validation["selection_score"]
            best_epoch = epoch
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
            # Keep a recoverable best weight file during long runs. The same
            # path is overwritten with the complete prototype checkpoint after
            # final evaluation, so no extra output file is created.
            torch.save(
                {
                    "model_state_dict": best_state,
                    "active_known_indices": active_indices,
                    "inactive_unknown_index": unknown_index,
                    "known_classes": class_names,
                    "unknown_class": args.unknown_class,
                    "full_class_order": list(FULL_CLASSES),
                    "label_threshold": args.label_threshold,
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "intermediate_training_checkpoint": True,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
        means = {key: value / max(1, sample_count) for key, value in totals.items()}
        print(
            f"epoch {epoch:03d} total={means['total']:.6f} "
            f"clean={means['clean']:.6f} contam={means['contam']:.6f} "
            f"cons={means['cons']:.6f} reject={means['reject']:.6f} | "
            f"val_clean={validation['clean_exact_match']:.6f} "
            f"val_contam={validation['contaminated_exact_match']:.6f} "
            f"val_pure_false={validation['pure_pseudo_false_known_rate']:.6f} "
            f"score={validation['selection_score']:.6f} "
            f"patience={patience_count}/{args.patience}"
            + (" *" if improved else ""),
            flush=True,
        )
        if args.patience > 0 and patience_count >= args.patience:
            print("early stopping", flush=True)
            break

    model.load_state_dict(best_state, strict=True)
    collected = {}
    for name in ("known_train", "known_val", "known_test", "unknown_test"):
        print(f"scoring {name}", flush=True)
        collected[name] = collect_predictions(model, loaders[name], device)
    train_logits, train_features, train_targets = collected["known_train"]
    val_logits, val_features, val_targets = collected["known_val"]
    known_logits, known_features, known_targets = collected["known_test"]
    unknown_logits, unknown_features, unknown_targets = collected["unknown_test"]
    prototypes, prototype_codes = build_prototypes(
        train_features, train_targets, active_indices
    )
    val_scores = prototype_score(val_features, prototypes)
    known_scores = prototype_score(known_features, prototypes)
    unknown_scores = prototype_score(unknown_features, prototypes)
    unknown_threshold = float(
        np.quantile(val_scores, 1.0 - args.known_reject_rate)
    )
    detection_labels = np.concatenate(
        [
            np.zeros(len(known_scores), dtype=np.int64),
            np.ones(len(unknown_scores), dtype=np.int64),
        ]
    )
    detection_scores = np.concatenate([known_scores, unknown_scores])
    known_reject = float(np.mean(known_scores > unknown_threshold))
    unknown_recall = float(np.mean(unknown_scores > unknown_threshold))

    known_metrics = multilabel_metrics(
        known_logits,
        known_targets,
        active_indices,
        class_names,
        args.label_threshold,
    )
    unknown_metrics = multilabel_metrics(
        unknown_logits,
        unknown_targets,
        active_indices,
        class_names,
        args.label_threshold,
    )
    configuration = {
        "full_class_order": list(FULL_CLASSES),
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "unknown_index": unknown_index,
        "label_threshold": args.label_threshold,
        "known_reject_rate": args.known_reject_rate,
        "use_asl": True,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pseudo_batch_fraction": args.pseudo_batch_fraction,
        "pseudo_min_die_fraction": args.pseudo_min_die_fraction,
        "pseudo_max_die_fraction": args.pseudo_max_die_fraction,
        "contamination_loss_weight": args.contamination_loss_weight,
        "consistency_loss_weight": args.consistency_loss_weight,
        "pure_reject_loss_weight": args.pure_reject_loss_weight,
        "pseudo_unknown_source": "known-training wafers plus online irregular masks; held-out data unused",
    }
    result = {
        "configuration": configuration,
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "training": {
            "best_epoch": best_epoch,
            "best_validation": best_validation,
        },
        "known_test_components": known_metrics,
        "unknown_test_known_projection": unknown_metrics,
        "unknown_detection": {
            "auroc": float(roc_auc_score(detection_labels, detection_scores)),
            "aupr": float(average_precision_score(detection_labels, detection_scores)),
            "fpr95": fpr_at_tpr95(detection_labels, detection_scores),
            "threshold": unknown_threshold,
            "known_reject_rate": known_reject,
            "unknown_recall": unknown_recall,
            "balanced_accuracy_at_threshold": float(
                0.5 * ((1.0 - known_reject) + unknown_recall)
            ),
        },
    }
    checkpoint_payload = {
        "model_state_dict": best_state,
        "output_classes": list(FULL_CLASSES),
        "active_known_indices": active_indices,
        "inactive_unknown_index": unknown_index,
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "full_class_order": list(FULL_CLASSES),
        "prototypes": prototypes,
        "prototype_codes": prototype_codes,
        "unknown_threshold": unknown_threshold,
        "label_threshold": args.label_threshold,
        "training_configuration": configuration,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {checkpoint_path}")
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()
