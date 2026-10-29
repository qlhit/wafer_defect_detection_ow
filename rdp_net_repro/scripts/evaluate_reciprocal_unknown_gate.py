#!/usr/bin/env python3
"""Evaluate wafer-level unknown scores from a trained reciprocal verifier.

No model is trained and no held-out sample is used for calibration.  The
known-validation split determines every score threshold and normalization
statistic.  The script reports several fixed, interpretable scores instead of
selecting one with held-out labels:

* reciprocal_topk_distance (primary): distance to the nearest reciprocal
  anchor, aggregated over the K most confident label relations;
* reciprocal_topk_ambiguity: uncertainty of positive-vs-negative reciprocal
  assignment;
* candidate_open_violation: open-space violation of RDP-positive candidates;
* rdp_composition_prototype: the original V1 composition-prototype score;
* known_only_equal_fusion: equal fusion after robust known-validation scaling.

Only one JSON file is written.  Place this file beside
train_eval_reciprocal_component_rejection.py in rdp_net_repro/scripts/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from train_eval_reciprocal_component_rejection import (
    FULL_CLASSES,
    ReciprocalComponentVerifier,
    load_arrays,
    load_frozen_rdp,
    make_split,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CREM-inspired reciprocal wafer-level unknown scores"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument(
        "--verifier-checkpoint",
        default=None,
        help=(
            "Optional explicit checkpoint. By default, search "
            "checkpoint/rdp_reciprocal_rejection and the legacy "
            "checkpoint/rdp_contamination_reciprocal directory."
        ),
    )
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Default: use the base_checkpoint recorded by the verifier",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="0: round the mean known-training label cardinality",
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


@torch.inference_mode()
def forward_rdp_with_features(
    model: torch.nn.Module, wafer: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    final_features = torch.flatten(model.avgpool(z), 1)
    logits = model.fc(final_features)
    return logits, block3, final_features


@torch.inference_mode()
def collect_outputs(
    base_model: torch.nn.Module,
    verifier: ReciprocalComponentVerifier,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    name: str,
) -> Dict[str, np.ndarray]:
    probability_parts: List[np.ndarray] = []
    relative_parts: List[np.ndarray] = []
    positive_distance_parts: List[np.ndarray] = []
    negative_distance_parts: List[np.ndarray] = []
    final_feature_parts: List[np.ndarray] = []
    print(f"scoring {name}: {len(indices)}")
    base_model.eval()
    verifier.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        wafer = x[batch_indices].unsqueeze(1).to(device, non_blocking=True)
        logits, block3, final_features = forward_rdp_with_features(base_model, wafer)
        relative, positive_distance, negative_distance, _ = verifier(block3)
        probability_parts.append(
            torch.sigmoid(logits[:, active_indices]).cpu().numpy()
        )
        relative_parts.append(relative.cpu().numpy())
        positive_distance_parts.append(positive_distance.cpu().numpy())
        negative_distance_parts.append(negative_distance.cpu().numpy())
        final_feature_parts.append(final_features.cpu().numpy())
    target_indices = torch.as_tensor(indices, dtype=torch.long)
    return {
        "probabilities": np.concatenate(probability_parts).astype(np.float32),
        "relative_logits": np.concatenate(relative_parts).astype(np.float32),
        "positive_distances": np.concatenate(positive_distance_parts).astype(np.float32),
        "negative_distances": np.concatenate(negative_distance_parts).astype(np.float32),
        "final_features": np.concatenate(final_feature_parts).astype(np.float32),
        "targets": y[target_indices][:, active_indices].cpu().numpy().astype(np.float32),
    }


def topk_smallest_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 1), values.shape[1])
    selected = np.partition(values, kth=k - 1, axis=1)[:, :k]
    return selected.mean(axis=1)


def topk_largest_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 1), values.shape[1])
    selected = np.partition(values, kth=values.shape[1] - k, axis=1)[:, -k:]
    return selected.mean(axis=1)


def reciprocal_scores(outputs: Dict[str, np.ndarray], top_k: int) -> Dict[str, np.ndarray]:
    relative = outputs["relative_logits"]
    nearest_anchor_distance = np.minimum(
        outputs["positive_distances"], outputs["negative_distances"]
    )
    # CREM-style intuition: known semantics should be close to at least one
    # reciprocal relation.  The K closest relations form the sample knownness.
    topk_distance = topk_smallest_mean(nearest_anchor_distance, top_k)

    # A known relation should confidently choose either the positive or the
    # negative side.  Low absolute margins indicate reciprocal ambiguity.
    relation_confidence = 1.0 / (1.0 + np.exp(-np.abs(relative)))
    topk_confidence = topk_largest_mean(relation_confidence, top_k)
    topk_ambiguity = 1.0 - topk_confidence
    return {
        "reciprocal_topk_distance": topk_distance.astype(np.float64),
        "reciprocal_topk_ambiguity": topk_ambiguity.astype(np.float64),
    }


def robust_center_scale(values: np.ndarray) -> Tuple[float, float]:
    center = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        mad = float(np.median(np.abs(values - center)) * 1.4826)
        scale = mad if mad >= 1e-6 else max(float(np.std(values)), 1e-6)
    return center, scale


def candidate_violation_scores(
    outputs: Dict[str, np.ndarray],
    label_threshold: float,
    relative_floors: np.ndarray,
    distance_ceilings: np.ndarray,
    distance_scales: np.ndarray,
    relative_scales: np.ndarray,
) -> np.ndarray:
    candidates = outputs["probabilities"] >= label_threshold
    distance_excess = (
        outputs["positive_distances"] - distance_ceilings.reshape(1, -1)
    ) / distance_scales.reshape(1, -1)
    relative_excess = (
        relative_floors.reshape(1, -1) - outputs["relative_logits"]
    ) / relative_scales.reshape(1, -1)
    violation = np.maximum(distance_excess, relative_excess)
    violation = np.where(candidates, violation, -np.inf)
    scores = violation.max(axis=1)
    # No positive candidate means there is no positive-label open violation.
    scores[~np.isfinite(scores)] = -5.0
    return scores.astype(np.float64)


def validate_prototypes(
    base_checkpoint: Dict[str, object], active_class_count: int
) -> np.ndarray | None:
    prototypes = base_checkpoint.get("prototypes")
    codes = base_checkpoint.get("prototype_codes")
    if prototypes is None or codes is None:
        return None
    if isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.detach().cpu().numpy()
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if prototypes.ndim != 2 or len(prototypes) != len(codes):
        return None
    if any(len(str(code)) != active_class_count for code in codes):
        return None
    return prototypes


def composition_prototype_score(
    features: np.ndarray, prototypes: np.ndarray
) -> np.ndarray:
    feature_tensor = F.normalize(torch.from_numpy(features).float(), dim=1)
    prototype_tensor = F.normalize(torch.from_numpy(prototypes).float(), dim=1)
    similarity = feature_tensor @ prototype_tensor.T
    return (1.0 - similarity.max(dim=1).values.numpy()).astype(np.float64)


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    return float(fpr[valid[0]]) if len(valid) else 1.0


def score_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
    }


def evaluate_detection(
    known_validation_scores: np.ndarray,
    known_test_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_reject_rate: float,
) -> Dict[str, object]:
    threshold = float(
        np.quantile(known_validation_scores, 1.0 - known_reject_rate)
    )
    labels = np.concatenate(
        [np.zeros(len(known_test_scores)), np.ones(len(unknown_scores))]
    )
    scores = np.concatenate([known_test_scores, unknown_scores])
    known_reject = float(np.mean(known_test_scores > threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr_at_tpr95(labels, scores),
        "threshold_from_known_validation": threshold,
        "known_test_reject_rate": known_reject,
        "unknown_recall_at_threshold": unknown_recall,
        "balanced_accuracy_at_threshold": float(
            0.5 * ((1.0 - known_reject) + unknown_recall)
        ),
        "known_scores": score_summary(known_test_scores),
        "unknown_scores": score_summary(unknown_scores),
    }


def robust_z(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (values - center) / scale


def resolve_verifier_path(explicit_path: str | None, unknown_class: str) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Verifier checkpoint not found: {path}")
        return path
    heldout = f"{unknown_class.lower()}_heldout"
    candidates = (
        Path("checkpoint/rdp_reciprocal_rejection")
        / heldout
        / "best_reciprocal_verifier.pt",
        Path("checkpoint/rdp_contamination_reciprocal")
        / heldout
        / "best_reciprocal_verifier.pt",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    rendered = "\n  ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "No reciprocal verifier checkpoint was found. Checked:\n  " + rendered
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.known_reject_rate < 0.5:
        raise ValueError("--known-reject-rate must be in (0,0.5)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [i for i in range(len(FULL_CLASSES)) if i != unknown_index]
    class_names = [FULL_CLASSES[i] for i in active_indices]
    verifier_path = resolve_verifier_path(
        args.verifier_checkpoint, args.unknown_class
    )
    verifier_checkpoint = torch_load(verifier_path, map_location="cpu")
    stored_unknown = verifier_checkpoint.get("unknown_class")
    if stored_unknown != args.unknown_class:
        raise ValueError(
            f"Verifier holds out {stored_unknown}, requested {args.unknown_class}"
        )
    stored_active = [int(i) for i in verifier_checkpoint["active_known_indices"]]
    if stored_active != active_indices:
        raise ValueError(f"Active indices mismatch: {stored_active} vs {active_indices}")
    base_path = Path(
        args.base_checkpoint or verifier_checkpoint["base_checkpoint"]
    ).resolve()
    output_path = Path(
        args.output
        or verifier_path.parent / "reciprocal_unknown_gate_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"unknown gate: {args.unknown_class}; known classes={class_names}")
    print(f"loading verifier: {verifier_path}")
    print(f"loading frozen RDP: {base_path}")

    x, y = load_arrays(args.data)
    split = make_split(
        y,
        unknown_index,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    base_model, _, base_checkpoint = load_frozen_rdp(
        base_path, args.unknown_class, device
    )
    verifier_config = verifier_checkpoint["verifier_configuration"]
    verifier = ReciprocalComponentVerifier(
        class_count=int(verifier_config["class_count"]),
        feature_channels=int(verifier_config["feature_channels"]),
        embedding_dim=int(verifier_config["embedding_dim"]),
        logit_scale=float(verifier_config["logit_scale"]),
    ).to(device)
    verifier.load_state_dict(verifier_checkpoint["verifier_state_dict"], strict=True)
    verifier.eval()

    train_targets = (
        y[torch.as_tensor(split["known_train"], dtype=torch.long)][:, active_indices]
        .cpu()
        .numpy()
    )
    automatic_top_k = int(np.clip(np.rint(train_targets.sum(axis=1).mean()), 1, len(active_indices)))
    top_k = args.top_k if args.top_k > 0 else automatic_top_k
    top_k = int(np.clip(top_k, 1, len(active_indices)))
    print(f"top-k reciprocal aggregation: {top_k}")

    known_val = collect_outputs(
        base_model,
        verifier,
        x,
        y,
        split["known_val"],
        active_indices,
        args.batch_size,
        device,
        "known_val",
    )
    known_test = collect_outputs(
        base_model,
        verifier,
        x,
        y,
        split["known_test"],
        active_indices,
        args.batch_size,
        device,
        "known_test",
    )
    unknown_test = collect_outputs(
        base_model,
        verifier,
        x,
        y,
        split["unknown_test"],
        active_indices,
        args.batch_size,
        device,
        "unknown_test",
    )

    score_sets: Dict[str, Dict[str, np.ndarray]] = {
        "known_val": reciprocal_scores(known_val, top_k),
        "known_test": reciprocal_scores(known_test, top_k),
        "unknown_test": reciprocal_scores(unknown_test, top_k),
    }

    relative_floors = verifier_checkpoint["relative_logit_floors"]
    distance_ceilings = verifier_checkpoint["positive_distance_ceilings"]
    if isinstance(relative_floors, torch.Tensor):
        relative_floors = relative_floors.cpu().numpy()
    if isinstance(distance_ceilings, torch.Tensor):
        distance_ceilings = distance_ceilings.cpu().numpy()
    relative_floors = np.asarray(relative_floors, dtype=np.float32)
    distance_ceilings = np.asarray(distance_ceilings, dtype=np.float32)

    distance_scales = np.empty(len(active_indices), dtype=np.float32)
    relative_scales = np.empty(len(active_indices), dtype=np.float32)
    for class_index in range(len(active_indices)):
        _, distance_scales[class_index] = robust_center_scale(
            known_val["positive_distances"][:, class_index]
        )
        _, relative_scales[class_index] = robust_center_scale(
            known_val["relative_logits"][:, class_index]
        )
    for split_name, outputs in (
        ("known_val", known_val),
        ("known_test", known_test),
        ("unknown_test", unknown_test),
    ):
        score_sets[split_name]["candidate_open_violation"] = candidate_violation_scores(
            outputs,
            float(verifier_checkpoint["label_threshold"]),
            relative_floors,
            distance_ceilings,
            distance_scales,
            relative_scales,
        )

    prototypes = validate_prototypes(base_checkpoint, len(active_indices))
    if prototypes is not None:
        for split_name, outputs in (
            ("known_val", known_val),
            ("known_test", known_test),
            ("unknown_test", unknown_test),
        ):
            score_sets[split_name]["rdp_composition_prototype"] = composition_prototype_score(
                outputs["final_features"], prototypes
            )

    component_names = list(score_sets["known_val"].keys())
    centers: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for name in component_names:
        centers[name], scales[name] = robust_center_scale(
            score_sets["known_val"][name]
        )
    for split_name in score_sets:
        z_parts = [
            robust_z(score_sets[split_name][name], centers[name], scales[name])
            for name in component_names
        ]
        score_sets[split_name]["known_only_equal_fusion"] = np.mean(
            np.stack(z_parts, axis=1), axis=1
        )

    all_results: Dict[str, object] = {}
    pure_mask = unknown_test["targets"].sum(axis=1) == 0
    mixed_mask = ~pure_mask
    for score_name in score_sets["known_val"]:
        result = {
            "all_unknown": evaluate_detection(
                score_sets["known_val"][score_name],
                score_sets["known_test"][score_name],
                score_sets["unknown_test"][score_name],
                args.known_reject_rate,
            )
        }
        if pure_mask.any():
            result["pure_unknown_only"] = evaluate_detection(
                score_sets["known_val"][score_name],
                score_sets["known_test"][score_name],
                score_sets["unknown_test"][score_name][pure_mask],
                args.known_reject_rate,
            )
        if mixed_mask.any():
            result["mixed_known_plus_unknown"] = evaluate_detection(
                score_sets["known_val"][score_name],
                score_sets["known_test"][score_name],
                score_sets["unknown_test"][score_name][mixed_mask],
                args.known_reject_rate,
            )
        all_results[score_name] = result

    metrics = {
        "configuration": {
            "method": "crem_inspired_reciprocal_unknown_gate_diagnostic",
            "unknown_class": args.unknown_class,
            "known_classes": class_names,
            "data": str(Path(args.data).resolve()),
            "verifier_checkpoint": str(verifier_path),
            "base_checkpoint": str(base_path),
            "top_k": top_k,
            "automatic_top_k_from_known_train_cardinality": automatic_top_k,
            "known_reject_rate": args.known_reject_rate,
            "seed": args.seed,
        },
        "split_counts": {name: int(len(values)) for name, values in split.items()},
        "unknown_subsets": {
            "pure_unknown": int(pure_mask.sum()),
            "mixed_known_plus_unknown": int(mixed_mask.sum()),
        },
        "scores": all_results,
        "known_only_fusion": {
            "components": component_names,
            "normalization": "known-validation median and robust scale",
            "unknown_labels_used_for_weights_or_normalization": False,
        },
        "interpretation": {
            "primary_score": "reciprocal_topk_distance",
            "selection_warning": "do not select a score using held-out metrics",
            "replacement_rule": "replace Dinomaly only if the predeclared primary reciprocal score matches or exceeds Dinomaly on AUROC and recall at the same known reject rate",
            "scope": "diagnostic deep adaptation inspired by CREM; not an exact reproduction of kernel CREM",
        },
    }
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nscore comparison: all unknown")
    for name, result in all_results.items():
        values = result["all_unknown"]
        print(
            f"  {name:<30} AUROC={values['auroc']:.6f} "
            f"recall@KR={values['unknown_recall_at_threshold']:.6f} "
            f"known_reject={values['known_test_reject_rate']:.6f}"
        )
    print("saved:", output_path)


if __name__ == "__main__":
    main()
