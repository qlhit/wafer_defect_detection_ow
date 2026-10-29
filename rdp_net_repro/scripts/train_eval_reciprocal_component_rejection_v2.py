#!/usr/bin/env python3
"""Train a spatial, multi-prototype reciprocal verifier for frozen RDP-Net.

The held-out class is excluded from training, validation, checkpoint selection,
and threshold calibration.  It is loaded only for the final diagnostic test.

Compared with the first reciprocal verifier, this version adds:
  * class-specific tokens from both block_2_2 and block_3_2;
  * explicit flattened spatial-attention descriptors;
  * multiple positive and negative reciprocal prototypes per known class;
  * RDP-guided hard-negative mining;
  * one joint support score and a per-class known-validation threshold that
    retains a requested fraction of RDP's true-positive candidates.

Only two output files are written:
  * best_spatial_reciprocal_verifier.pt
  * metrics.json

Place this file beside train_eval_reciprocal_component_rejection.py in
rdp_net_repro/scripts/ and run it from the rdp_net_repro repository root.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_eval_reciprocal_component_rejection import (
    FULL_CLASSES,
    load_arrays,
    load_frozen_rdp,
    make_split,
    multilabel_metrics,
)
from train_eval_contamination_robust_rdp import (
    data_values,
    synthesize_pseudo_unknown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate spatial multi-prototype reciprocal RDP verifier"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help=(
            "Default: contamination-robust checkpoint when it exists, otherwise "
            "the original leave-one-out RDP checkpoint"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--prototype-count", type=int, default=3)
    parser.add_argument("--prototype-temperature", type=float, default=0.10)
    parser.add_argument("--logit-scale", type=float, default=10.0)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--compactness-weight", type=float, default=0.05)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.25,
        help="Fraction of absent-label samples retained per class in every batch",
    )
    parser.add_argument(
        "--hard-rdp-weight",
        type=float,
        default=0.70,
        help="Hard-negative ranking weight assigned to frozen RDP probability",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=2.0,
        help="Joint support = relative_logit - distance_weight * positive_distance",
    )
    parser.add_argument(
        "--pseudo-feature-fraction",
        type=float,
        default=0.50,
        help=(
            "Known-training fraction receiving one contaminated and one pure "
            "pseudo-unknown feature example"
        ),
    )
    parser.add_argument("--pseudo-min-die-fraction", type=float, default=0.02)
    parser.add_argument("--pseudo-max-die-fraction", type=float, default=0.12)
    parser.add_argument(
        "--positive-retention",
        type=float,
        default=0.99,
        help="Retain this fraction of known-validation RDP true-positive candidates",
    )
    parser.add_argument("--label-threshold", type=float, default=0.50)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_base_checkpoint(explicit: str | None, unknown_class: str) -> Path:
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RDP checkpoint not found: {path}")
        return path
    heldout = f"{unknown_class.lower()}_heldout"
    candidates = (
        Path("checkpoint/rdp_contamination_robust") / heldout / "best_model.pt",
        Path("checkpoint/rdp_leave_one_out") / heldout / "best_open_set.pt",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    rendered = "\n  ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError("No compatible RDP checkpoint found. Checked:\n  " + rendered)


@torch.inference_mode()
def forward_frozen_multiscale(
    model: nn.Module, wafer: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = model.conv1(wafer)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    z = model.block_1_1(z)
    z = model.block_1_2(z)
    z = model.block_2_1(z)
    block2 = model.block_2_2(z)
    z = model.block_3_1(block2)
    block3 = model.block_3_2(z)
    z = model.block_4_1(block3)
    z = model.block_4_2(z)
    logits = model.fc(torch.flatten(model.avgpool(z), 1))
    return logits, block2, block3


@torch.inference_mode()
def extract_multiscale_features(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    name: str,
    pseudo_fraction: float = 0.0,
    pseudo_generator: torch.Generator | None = None,
    pseudo_values: Tuple[float, float, float] | None = None,
    pseudo_min_fraction: float = 0.02,
    pseudo_max_fraction: float = 0.12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block2_parts: List[torch.Tensor] = []
    block3_parts: List[torch.Tensor] = []
    probability_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    print(f"extracting frozen multiscale RDP features: {name} ({len(indices)})")
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        wafer = x[batch_indices].unsqueeze(1).to(device, non_blocking=True)
        logits, block2, block3 = forward_frozen_multiscale(model, wafer)
        block2_parts.append(block2.cpu().to(torch.float16))
        block3_parts.append(block3.cpu().to(torch.float16))
        probability_parts.append(
            torch.sigmoid(logits[:, active_indices]).cpu().to(torch.float32)
        )
        target_parts.append(
            y[batch_indices][:, active_indices].cpu().to(torch.float32)
        )
        if pseudo_fraction > 0.0:
            if pseudo_generator is None or pseudo_values is None:
                raise ValueError("Pseudo feature extraction requires generator and data values")
            pseudo_count = max(
                1,
                min(
                    len(wafer),
                    int(round(len(wafer) * pseudo_fraction)),
                ),
            )
            valid_threshold, normal_value, defect_value = pseudo_values
            contaminated, pure = synthesize_pseudo_unknown(
                wafer[:pseudo_count],
                pseudo_min_fraction,
                pseudo_max_fraction,
                pseudo_generator,
                valid_threshold,
                normal_value,
                defect_value,
            )
            pseudo_wafer = torch.cat((contaminated, pure), dim=0)
            pseudo_logits, pseudo_block2, pseudo_block3 = forward_frozen_multiscale(
                model, pseudo_wafer
            )
            block2_parts.append(pseudo_block2.cpu().to(torch.float16))
            block3_parts.append(pseudo_block3.cpu().to(torch.float16))
            probability_parts.append(
                torch.sigmoid(pseudo_logits[:, active_indices]).cpu().to(torch.float32)
            )
            contaminated_targets = y[batch_indices[:pseudo_count]][:, active_indices]
            pure_targets = torch.zeros_like(contaminated_targets)
            target_parts.append(
                torch.cat((contaminated_targets, pure_targets), dim=0).to(torch.float32)
            )
    return (
        torch.cat(block2_parts),
        torch.cat(block3_parts),
        torch.cat(probability_parts),
        torch.cat(target_parts),
    )


class SpatialMultiPrototypeVerifier(nn.Module):
    """Class-specific multiscale attention with reciprocal prototype banks."""

    def __init__(
        self,
        class_count: int,
        block2_shape: Sequence[int],
        block3_shape: Sequence[int],
        embedding_dim: int,
        prototype_count: int,
        prototype_temperature: float,
        logit_scale: float,
    ) -> None:
        super().__init__()
        c2, h2, w2 = (int(value) for value in block2_shape)
        c3, h3, w3 = (int(value) for value in block3_shape)
        self.class_count = int(class_count)
        self.block2_shape = (c2, h2, w2)
        self.block3_shape = (c3, h3, w3)
        self.embedding_dim = int(embedding_dim)
        self.prototype_count = int(prototype_count)
        self.prototype_temperature = float(prototype_temperature)
        self.logit_scale = float(logit_scale)

        self.query2 = nn.Parameter(torch.empty(class_count, c2))
        self.query3 = nn.Parameter(torch.empty(class_count, c3))
        descriptor_dim = c2 + c3 + h2 * w2 + h3 * w3
        self.projection = nn.Parameter(
            torch.empty(class_count, descriptor_dim, embedding_dim)
        )
        self.projection_bias = nn.Parameter(
            torch.zeros(class_count, embedding_dim)
        )
        self.positive_prototypes = nn.Parameter(
            torch.empty(class_count, prototype_count, embedding_dim)
        )
        self.negative_prototypes = nn.Parameter(
            torch.empty(class_count, prototype_count, embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query2, std=0.02)
        nn.init.normal_(self.query3, std=0.02)
        nn.init.xavier_uniform_(self.projection)
        nn.init.normal_(self.positive_prototypes, std=0.02)
        nn.init.normal_(self.negative_prototypes, std=0.02)

    @staticmethod
    def attention_token(
        feature_map: torch.Tensor, query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = F.normalize(feature_map.float(), dim=1)
        query = F.normalize(query, dim=1)
        logits = torch.einsum("bchw,kc->bkhw", feature_map, query)
        attention = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)
        token = torch.einsum("bkhw,bchw->bkc", attention, feature_map)
        # Multiplication by spatial size keeps a uniform map centred at one.
        spatial = attention.flatten(2) * attention.shape[-2] * attention.shape[-1]
        return token, spatial

    def forward(
        self, block2: torch.Tensor, block3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token2, spatial2 = self.attention_token(block2, self.query2)
        token3, spatial3 = self.attention_token(block3, self.query3)
        descriptor = torch.cat((token2, token3, spatial2, spatial3), dim=-1)
        embedding = torch.einsum(
            "bkd,kde->bke", descriptor, self.projection
        ) + self.projection_bias.unsqueeze(0)
        embedding = F.normalize(embedding, dim=-1)

        positive = F.normalize(self.positive_prototypes, dim=-1)
        negative = F.normalize(self.negative_prototypes, dim=-1)
        positive_similarity_all = torch.einsum(
            "bke,kpe->bkp", embedding, positive
        )
        negative_similarity_all = torch.einsum(
            "bke,kpe->bkp", embedding, negative
        )
        temperature = self.prototype_temperature
        normalizer = math.log(self.prototype_count)
        positive_similarity = temperature * (
            torch.logsumexp(positive_similarity_all / temperature, dim=-1)
            - normalizer
        )
        negative_similarity = temperature * (
            torch.logsumexp(negative_similarity_all / temperature, dim=-1)
            - normalizer
        )
        relative_logits = self.logit_scale * (
            positive_similarity - negative_similarity
        )
        positive_distance = 1.0 - positive_similarity
        negative_distance = 1.0 - negative_similarity
        return relative_logits, positive_distance, negative_distance, embedding


def prototype_diversity(prototypes: torch.Tensor) -> torch.Tensor:
    count = prototypes.shape[1]
    if count <= 1:
        return prototypes.new_zeros(())
    normalized = F.normalize(prototypes, dim=-1)
    cosine = torch.einsum("kpe,kqe->kpq", normalized, normalized)
    mask = ~torch.eye(count, dtype=torch.bool, device=prototypes.device)
    off_diagonal = cosine.masked_select(mask.unsqueeze(0).expand_as(cosine))
    # Only penalize near-collapse; moderately similar modes are allowed.
    return F.relu(off_diagonal - 0.90).mean()


def hard_negative_mask(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    relative_logits: torch.Tensor,
    fraction: float,
    rdp_weight: float,
) -> torch.Tensor:
    negative = targets < 0.5
    selected = torch.zeros_like(negative)
    model_probability = torch.sigmoid(relative_logits.detach())
    hardness = rdp_weight * probabilities + (1.0 - rdp_weight) * model_probability
    for class_index in range(targets.shape[1]):
        indices = torch.nonzero(
            negative[:, class_index], as_tuple=False
        ).flatten()
        if len(indices) == 0:
            continue
        keep = max(1, int(math.ceil(len(indices) * fraction)))
        local = hardness[indices, class_index]
        chosen = indices[torch.topk(local, k=min(keep, len(indices))).indices]
        selected[chosen, class_index] = True
    return selected


def verifier_loss(
    verifier: SpatialMultiPrototypeVerifier,
    block2: torch.Tensor,
    block3: torch.Tensor,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    hard_negative_fraction: float,
    hard_rdp_weight: float,
    margin: float,
    margin_weight: float,
    compactness_weight: float,
    diversity_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    relative, positive_distance, negative_distance, _ = verifier(block2, block3)
    positives = targets >= 0.5
    hard_negatives = hard_negative_mask(
        targets,
        probabilities,
        relative,
        hard_negative_fraction,
        hard_rdp_weight,
    )
    positive_margin = F.relu(margin + positive_distance - negative_distance)
    negative_margin = F.relu(margin + negative_distance - positive_distance)
    class_bce: List[torch.Tensor] = []
    class_margin: List[torch.Tensor] = []
    class_compactness: List[torch.Tensor] = []
    for class_index in range(targets.shape[1]):
        positive = positives[:, class_index]
        hard_negative = hard_negatives[:, class_index]
        terms_bce: List[torch.Tensor] = []
        terms_margin: List[torch.Tensor] = []
        terms_compact: List[torch.Tensor] = []
        if positive.any():
            terms_bce.append(
                F.binary_cross_entropy_with_logits(
                    relative[positive, class_index],
                    torch.ones_like(relative[positive, class_index]),
                )
            )
            terms_margin.append(positive_margin[positive, class_index].mean())
            terms_compact.append(positive_distance[positive, class_index].mean())
        if hard_negative.any():
            terms_bce.append(
                F.binary_cross_entropy_with_logits(
                    relative[hard_negative, class_index],
                    torch.zeros_like(relative[hard_negative, class_index]),
                )
            )
            terms_margin.append(negative_margin[hard_negative, class_index].mean())
            terms_compact.append(negative_distance[hard_negative, class_index].mean())
        if terms_bce:
            class_bce.append(torch.stack(terms_bce).mean())
            class_margin.append(torch.stack(terms_margin).mean())
            class_compactness.append(torch.stack(terms_compact).mean())
    classification = torch.stack(class_bce).mean()
    margin_loss = torch.stack(class_margin).mean()
    compactness = torch.stack(class_compactness).mean()
    diversity = 0.5 * (
        prototype_diversity(verifier.positive_prototypes)
        + prototype_diversity(verifier.negative_prototypes)
    )
    total = (
        classification
        + margin_weight * margin_loss
        + compactness_weight * compactness
        + diversity_weight * diversity
    )
    return total, {
        "classification": classification.detach(),
        "margin": margin_loss.detach(),
        "compactness": compactness.detach(),
        "diversity": diversity.detach(),
    }


def make_loader(
    block2: torch.Tensor,
    block3: torch.Tensor,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(block2, block3, probabilities, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def evaluate_loss(
    verifier: SpatialMultiPrototypeVerifier,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    verifier.eval()
    totals = {
        "total": 0.0,
        "classification": 0.0,
        "margin": 0.0,
        "compactness": 0.0,
        "diversity": 0.0,
    }
    count = 0
    for block2, block3, probabilities, targets in loader:
        block2 = block2.to(device, non_blocking=True)
        block3 = block3.to(device, non_blocking=True)
        probabilities = probabilities.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss, parts = verifier_loss(
            verifier,
            block2,
            block3,
            probabilities,
            targets,
            args.hard_negative_fraction,
            args.hard_rdp_weight,
            args.margin,
            args.margin_weight,
            args.compactness_weight,
            args.diversity_weight,
        )
        batch_count = len(targets)
        totals["total"] += float(loss.item()) * batch_count
        for name, value in parts.items():
            totals[name] += float(value.item()) * batch_count
        count += batch_count
    return {name: value / max(count, 1) for name, value in totals.items()}


def train_verifier(
    verifier: SpatialMultiPrototypeVerifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        verifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
    )
    best_state = copy.deepcopy(verifier.state_dict())
    best_epoch = 0
    best_validation = {"total": float("inf")}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        verifier.train()
        totals = {"total": 0.0, "classification": 0.0, "margin": 0.0}
        count = 0
        for block2, block3, probabilities, targets in train_loader:
            block2 = block2.to(device, non_blocking=True)
            block3 = block3.to(device, non_blocking=True)
            probabilities = probabilities.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss, parts = verifier_loss(
                verifier,
                block2,
                block3,
                probabilities,
                targets,
                args.hard_negative_fraction,
                args.hard_rdp_weight,
                args.margin,
                args.margin_weight,
                args.compactness_weight,
                args.diversity_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 5.0)
            optimizer.step()
            batch_count = len(targets)
            totals["total"] += float(loss.item()) * batch_count
            totals["classification"] += float(parts["classification"].item()) * batch_count
            totals["margin"] += float(parts["margin"].item()) * batch_count
            count += batch_count
        scheduler.step()
        validation = evaluate_loss(verifier, val_loader, device, args)
        improved = validation["total"] < best_validation["total"] - 1e-6
        if improved:
            best_state = copy.deepcopy(verifier.state_dict())
            best_epoch = epoch
            best_validation = validation
            stale = 0
        else:
            stale += 1
        marker = " *" if improved else ""
        print(
            f"epoch {epoch:03d} "
            f"train={totals['total'] / max(count, 1):.6f} "
            f"train_bce={totals['classification'] / max(count, 1):.6f} "
            f"train_margin={totals['margin'] / max(count, 1):.6f} "
            f"val={validation['total']:.6f} "
            f"val_bce={validation['classification']:.6f} "
            f"patience={stale}/{args.patience}{marker}"
        )
        if args.patience > 0 and stale >= args.patience:
            break
    return best_state, best_epoch, best_validation


@torch.inference_mode()
def collect_scores(
    verifier: SpatialMultiPrototypeVerifier,
    block2: torch.Tensor,
    block3: torch.Tensor,
    batch_size: int,
    device: torch.device,
    distance_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    verifier.eval()
    support_parts: List[np.ndarray] = []
    relative_parts: List[np.ndarray] = []
    distance_parts: List[np.ndarray] = []
    for start in range(0, len(block2), batch_size):
        local2 = block2[start : start + batch_size].to(device, non_blocking=True)
        local3 = block3[start : start + batch_size].to(device, non_blocking=True)
        relative, distance, _, _ = verifier(local2, local3)
        support = relative - distance_weight * distance
        support_parts.append(support.cpu().numpy())
        relative_parts.append(relative.cpu().numpy())
        distance_parts.append(distance.cpu().numpy())
    return (
        np.concatenate(support_parts).astype(np.float32),
        np.concatenate(relative_parts).astype(np.float32),
        np.concatenate(distance_parts).astype(np.float32),
    )


def calibrate_support_thresholds(
    support: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    label_threshold: float,
    positive_retention: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    candidate = probabilities >= label_threshold
    thresholds: List[float] = []
    diagnostics: List[Dict[str, float]] = []
    for class_index in range(targets.shape[1]):
        positive = targets[:, class_index].astype(bool)
        retained_pool = positive & candidate[:, class_index]
        if not retained_pool.any():
            raise ValueError(
                f"No RDP-positive validation sample for local class {class_index}"
            )
        threshold = float(
            np.quantile(
                support[retained_pool, class_index],
                1.0 - positive_retention,
            )
        )
        prediction = candidate[:, class_index] & (
            support[:, class_index] >= threshold
        )
        tp = int(np.sum(prediction & positive))
        fp = int(np.sum(prediction & ~positive))
        positive_count = int(positive.sum())
        negative_count = int((~positive).sum())
        thresholds.append(threshold)
        diagnostics.append(
            {
                "threshold": threshold,
                "positive_samples": positive_count,
                "recall": tp / max(positive_count, 1),
                "false_positive_rate": fp / max(negative_count, 1),
            }
        )
    return np.asarray(thresholds, dtype=np.float32), diagnostics


def predict(
    probabilities: np.ndarray,
    support: np.ndarray,
    label_threshold: float,
    support_thresholds: np.ndarray,
) -> np.ndarray:
    return (probabilities >= label_threshold) & (
        support >= support_thresholds.reshape(1, -1)
    )


def main() -> None:
    args = parse_args()
    if args.prototype_count <= 0 or args.embedding_dim <= 0:
        raise ValueError("Prototype count and embedding dimension must be positive")
    if args.prototype_temperature <= 0.0:
        raise ValueError("--prototype-temperature must be positive")
    if not 0.0 < args.hard_negative_fraction <= 1.0:
        raise ValueError("--hard-negative-fraction must be in (0,1]")
    if not 0.0 <= args.hard_rdp_weight <= 1.0:
        raise ValueError("--hard-rdp-weight must be in [0,1]")
    if not 0.5 < args.positive_retention < 1.0:
        raise ValueError("--positive-retention must be in (0.5,1.0)")
    if not 0.0 <= args.pseudo_feature_fraction <= 1.0:
        raise ValueError("--pseudo-feature-fraction must be in [0,1]")
    if not (
        0.0 < args.pseudo_min_die_fraction
        <= args.pseudo_max_die_fraction
        < 1.0
    ):
        raise ValueError("Invalid pseudo die-fraction range")
    set_seed(args.seed)

    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [i for i in range(len(FULL_CLASSES)) if i != unknown_index]
    class_names = [FULL_CLASSES[i] for i in active_indices]
    base_path = resolve_base_checkpoint(args.base_checkpoint, args.unknown_class)
    output_dir = Path(
        args.output_dir
        or f"checkpoint/rdp_reciprocal_v2/{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"unknown={args.unknown_class}; active known={class_names}")
    print(f"loading frozen RDP: {base_path}")

    x, y = load_arrays(args.data)
    split = make_split(
        y,
        unknown_index,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    print(
        "split: "
        + ", ".join(f"{name}={len(indices)}" for name, indices in split.items())
    )
    base_model, stored_active, _ = load_frozen_rdp(
        base_path, args.unknown_class, device
    )
    if stored_active != active_indices:
        raise ValueError(f"RDP active classes mismatch: {stored_active}")

    extracted: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    pseudo_generator = torch.Generator(device=device).manual_seed(args.seed + 7103)
    pseudo_values = data_values(x)
    for name in ("known_train", "known_val", "known_test"):
        extracted[name] = extract_multiscale_features(
            base_model,
            x,
            y,
            split[name],
            active_indices,
            args.extract_batch_size,
            device,
            name,
            pseudo_fraction=(
                args.pseudo_feature_fraction if name == "known_train" else 0.0
            ),
            pseudo_generator=pseudo_generator,
            pseudo_values=pseudo_values,
            pseudo_min_fraction=args.pseudo_min_die_fraction,
            pseudo_max_fraction=args.pseudo_max_die_fraction,
        )
    train2, train3, train_probability, train_target = extracted["known_train"]
    val2, val3, val_probability, val_target = extracted["known_val"]

    verifier = SpatialMultiPrototypeVerifier(
        class_count=len(active_indices),
        block2_shape=train2.shape[1:],
        block3_shape=train3.shape[1:],
        embedding_dim=args.embedding_dim,
        prototype_count=args.prototype_count,
        prototype_temperature=args.prototype_temperature,
        logit_scale=args.logit_scale,
    ).to(device)
    train_loader = make_loader(
        train2,
        train3,
        train_probability,
        train_target,
        args.batch_size,
        True,
        args.num_workers,
    )
    val_loader = make_loader(
        val2,
        val3,
        val_probability,
        val_target,
        args.batch_size,
        False,
        args.num_workers,
    )
    best_state, best_epoch, best_validation = train_verifier(
        verifier, train_loader, val_loader, device, args
    )
    verifier.load_state_dict(best_state, strict=True)

    val_support, _, _ = collect_scores(
        verifier,
        val2,
        val3,
        args.batch_size,
        device,
        args.distance_weight,
    )
    support_thresholds, calibration = calibrate_support_thresholds(
        val_support,
        val_probability.numpy(),
        val_target.numpy(),
        args.label_threshold,
        args.positive_retention,
    )

    result_metrics: Dict[str, object] = {}
    for name in ("known_test", "unknown_test"):
        if name == "unknown_test":
            features = extract_multiscale_features(
                base_model,
                x,
                y,
                split[name],
                active_indices,
                args.extract_batch_size,
                device,
                name,
            )
        else:
            features = extracted[name]
        local2, local3, probabilities, targets = features
        support, _, _ = collect_scores(
            verifier,
            local2,
            local3,
            args.batch_size,
            device,
            args.distance_weight,
        )
        predictions = predict(
            probabilities.numpy(),
            support,
            args.label_threshold,
            support_thresholds,
        )
        result_metrics[name] = multilabel_metrics(
            predictions,
            targets.numpy(),
            class_names,
        )
        del features, local2, local3

    metrics = {
        "configuration": {
            "method": "spatial_multiscale_multiprototype_hard_negative_rcr_v2",
            "full_class_order": list(FULL_CLASSES),
            "known_classes": class_names,
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_index,
            "base_checkpoint": str(base_path),
            "base_rdp_frozen": True,
            "heldout_data_used_for_training_selection_or_calibration": False,
            "feature_layers": ["block_2_2", "block_3_2"],
            "embedding_dim": args.embedding_dim,
            "prototype_count_per_polarity": args.prototype_count,
            "prototype_temperature": args.prototype_temperature,
            "hard_negative_fraction": args.hard_negative_fraction,
            "hard_rdp_weight": args.hard_rdp_weight,
            "distance_weight": args.distance_weight,
            "positive_retention": args.positive_retention,
            "pseudo_feature_fraction": args.pseudo_feature_fraction,
            "pseudo_min_die_fraction": args.pseudo_min_die_fraction,
            "pseudo_max_die_fraction": args.pseudo_max_die_fraction,
            "pseudo_unknown_source": "known-training wafers only",
            "label_threshold": args.label_threshold,
            "seed": args.seed,
        },
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "training": {
            "best_epoch": best_epoch,
            "best_known_validation_loss": best_validation,
        },
        "known_validation_calibration": {
            name: calibration[index] for index, name in enumerate(class_names)
        },
        "known_test_components": result_metrics["known_test"],
        "unknown_test_known_projection": result_metrics["unknown_test"],
        "notes": {
            "unknown_gate": "Dinomaly remains the wafer-level unknown detector",
            "primary_goal": "reduce false known components under unknown-component interference",
            "checkpoint_selection": "known-validation loss only",
        },
    }
    checkpoint = {
        "model_state_dict": verifier.state_dict(),
        "model_configuration": {
            "class_count": len(active_indices),
            "block2_shape": list(train2.shape[1:]),
            "block3_shape": list(train3.shape[1:]),
            "embedding_dim": args.embedding_dim,
            "prototype_count": args.prototype_count,
            "prototype_temperature": args.prototype_temperature,
            "logit_scale": args.logit_scale,
        },
        "base_checkpoint": str(base_path),
        "full_class_order": list(FULL_CLASSES),
        "active_known_indices": active_indices,
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "label_threshold": args.label_threshold,
        "distance_weight": args.distance_weight,
        "support_thresholds": torch.from_numpy(support_thresholds),
    }
    torch.save(checkpoint, output_dir / "best_spatial_reciprocal_verifier.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved:", output_dir / "best_spatial_reciprocal_verifier.pt")
    print("saved:", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
