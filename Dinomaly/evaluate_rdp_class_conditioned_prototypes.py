#!/usr/bin/env python3
"""Evaluate class-conditioned RDP prototype verification.

For every RDP candidate class, the 512-D feature is compared only with stored
training prototypes whose composition contains that class. Per-class distance
thresholds are calibrated on true known-validation samples, then frozen before
known-test and held-out-unknown evaluation. No Dinomaly model is loaded and no
training is performed. Only one JSON file is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from calibrate_evaluate_rdp_thresholds import (
    make_split,
    probability_summary,
    summary_metrics,
)
from diagnose_rdp_unknown_projection import (
    FULL_CLASSES,
    load_arrays,
    load_rdp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class-conditioned prototype verification for RDP-Net"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument("--rdp-repo", default="../rdp_net_repro")
    parser.add_argument("--rdp-checkpoint", default=None)
    parser.add_argument(
        "--rdp-threshold-json",
        default=None,
        help="Default: checkpoint/rdp_conditioned_dinomaly/<class>_heldout/rdp_threshold_calibration.json",
    )
    parser.add_argument(
        "--rdp-threshold-scheme",
        choices=("fixed_0_5", "known_val_max_f1", "known_val_fpr_control"),
        default="known_val_max_f1",
    )
    parser.add_argument(
        "--known-class-accept-rate",
        type=float,
        default=0.95,
        help="Fraction of true known-validation class samples accepted by distance",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def forward_features(
    model: torch.nn.Module, wafer: torch.Tensor
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
    logits = model.fc(features)
    return logits, features


@torch.inference_mode()
def collect(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability_parts = []
    feature_parts = []
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        wafer = x[batch_indices].unsqueeze(1).to(device, non_blocking=True)
        logits, features = forward_features(model, wafer)
        probability_parts.append(
            torch.sigmoid(logits[:, active_indices]).cpu().numpy()
        )
        feature_parts.append(features.cpu().numpy())
    probabilities = np.concatenate(probability_parts, axis=0)
    features = np.concatenate(feature_parts, axis=0)
    targets = (
        y[torch.as_tensor(indices, dtype=torch.long)][:, active_indices]
        .cpu()
        .numpy()
        > 0.5
    )
    return probabilities, features, targets


def load_probability_thresholds(
    path: Path,
    scheme: str,
    class_names: Sequence[str],
) -> np.ndarray:
    if not path.is_file():
        if scheme != "fixed_0_5":
            raise FileNotFoundError(f"Threshold calibration not found: {path}")
        return np.full(len(class_names), 0.5, dtype=np.float64)
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["threshold_schemes"][scheme]["thresholds"]
    return np.asarray([values[name] for name in class_names], dtype=np.float64)


def validate_prototypes(
    checkpoint: Dict[str, object],
    active_count: int,
) -> Tuple[np.ndarray, Sequence[str]]:
    if "prototypes" not in checkpoint or "prototype_codes" not in checkpoint:
        raise KeyError(
            "RDP checkpoint has no prototypes/prototype_codes; use the leave-one-out best_open_set.pt"
        )
    prototypes = checkpoint["prototypes"]
    if isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.detach().cpu().numpy()
    prototypes = np.asarray(prototypes, dtype=np.float32)
    codes = [str(code) for code in checkpoint["prototype_codes"]]
    if prototypes.ndim != 2 or len(prototypes) != len(codes):
        raise ValueError("Invalid prototype tensor/code list")
    if any(len(code) != active_count for code in codes):
        raise ValueError(
            f"Prototype codes must have {active_count} bits, got {codes[:3]}"
        )
    return prototypes, codes


def class_conditioned_distances(
    features: np.ndarray,
    prototypes: np.ndarray,
    prototype_codes: Sequence[str],
    class_count: int,
) -> np.ndarray:
    feature_tensor = F.normalize(torch.from_numpy(features).float(), dim=1)
    prototype_tensor = F.normalize(
        torch.from_numpy(prototypes).float(), dim=1
    )
    similarity = (feature_tensor @ prototype_tensor.T).numpy()
    distances = np.empty((len(features), class_count), dtype=np.float32)
    for class_index in range(class_count):
        allowed = np.asarray(
            [code[class_index] == "1" for code in prototype_codes], dtype=bool
        )
        if not allowed.any():
            raise ValueError(
                f"No stored prototype contains local class index {class_index}"
            )
        distances[:, class_index] = 1.0 - similarity[:, allowed].max(axis=1)
    return distances


def calibrate_distance_thresholds(
    distances: np.ndarray,
    targets: np.ndarray,
    accept_rate: float,
) -> np.ndarray:
    thresholds = []
    for class_index in range(targets.shape[1]):
        true_distances = distances[targets[:, class_index], class_index]
        if len(true_distances) == 0:
            raise ValueError(
                f"Known validation has no positive sample for class {class_index}"
            )
        thresholds.append(float(np.quantile(true_distances, accept_rate)))
    return np.asarray(thresholds, dtype=np.float64)


def evaluate_predictions(
    probabilities: np.ndarray,
    distances: np.ndarray,
    targets: np.ndarray,
    probability_thresholds: np.ndarray,
    distance_thresholds: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    probability_predictions = probabilities >= probability_thresholds.reshape(1, -1)
    verified_predictions = probability_predictions & (
        distances <= distance_thresholds.reshape(1, -1)
    )
    # summary_metrics thresholds its first argument. Passing binary 0/1 values
    # with a 0.5 threshold lets us reuse exactly the same multilabel reporting.
    fixed_binary_threshold = np.full(len(class_names), 0.5, dtype=np.float64)
    return {
        "probability_only": summary_metrics(
            probability_predictions.astype(np.float32),
            targets,
            fixed_binary_threshold,
            class_names,
        ),
        "probability_plus_prototype": summary_metrics(
            verified_predictions.astype(np.float32),
            targets,
            fixed_binary_threshold,
            class_names,
        ),
    }


def class_distance_diagnostics(
    class_index: int,
    class_name: str,
    known_test_probabilities: np.ndarray,
    known_test_distances: np.ndarray,
    known_test_targets: np.ndarray,
    unknown_probabilities: np.ndarray,
    unknown_distances: np.ndarray,
    unknown_targets: np.ndarray,
    probability_threshold: float,
    distance_threshold: float,
) -> Dict[str, object]:
    true_known = known_test_targets[:, class_index]
    false_unknown_candidate = (
        (unknown_probabilities[:, class_index] >= probability_threshold)
        & ~unknown_targets[:, class_index]
    )
    true_distances = known_test_distances[true_known, class_index]
    false_distances = unknown_distances[false_unknown_candidate, class_index]
    result: Dict[str, object] = {
        "class_name": class_name,
        "probability_threshold": float(probability_threshold),
        "distance_threshold_from_known_validation": float(distance_threshold),
        "true_known_test": probability_summary(true_distances),
        "false_unknown_candidates": probability_summary(false_distances),
        "true_known_accept_rate": (
            float(np.mean(true_distances <= distance_threshold))
            if len(true_distances)
            else None
        ),
        "false_unknown_reject_rate": (
            float(np.mean(false_distances > distance_threshold))
            if len(false_distances)
            else None
        ),
    }
    if len(true_distances) and len(false_distances):
        labels = np.concatenate(
            [
                np.zeros(len(true_distances), dtype=np.int64),
                np.ones(len(false_distances), dtype=np.int64),
            ]
        )
        scores = np.concatenate([true_distances, false_distances])
        result["false_candidate_auroc"] = float(roc_auc_score(labels, scores))
        result["false_candidate_aupr"] = float(
            average_precision_score(labels, scores)
        )
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.known_class_accept_rate < 1.0:
        raise ValueError("--known-class-accept-rate must be in (0,1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo = Path(args.rdp_repo).resolve()
    checkpoint_path = (
        Path(args.rdp_checkpoint).resolve()
        if args.rdp_checkpoint
        else repo
        / "checkpoint"
        / "rdp_leave_one_out"
        / f"{args.unknown_class.lower()}_heldout"
        / "best_open_set.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RDP checkpoint not found: {checkpoint_path}")
    threshold_json = Path(
        args.rdp_threshold_json
        or "checkpoint/rdp_conditioned_dinomaly/"
        f"{args.unknown_class.lower()}_heldout/rdp_threshold_calibration.json"
    ).resolve()

    x, y = load_arrays(args.data)
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
    model, loaded_active = load_rdp(
        repo, checkpoint_path, args.unknown_class, device
    )
    if loaded_active != active_indices:
        raise ValueError("Unexpected RDP active class order")
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    prototypes, prototype_codes = validate_prototypes(
        checkpoint, len(active_indices)
    )
    probability_thresholds = load_probability_thresholds(
        threshold_json, args.rdp_threshold_scheme, class_names
    )

    collected = {}
    for name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {name}: {len(split[name])}", flush=True)
        probabilities, features, targets = collect(
            model,
            x,
            y,
            split[name],
            active_indices,
            args.batch_size,
            device,
        )
        distances = class_conditioned_distances(
            features,
            prototypes,
            prototype_codes,
            len(active_indices),
        )
        collected[name] = {
            "probabilities": probabilities,
            "distances": distances,
            "targets": targets,
        }

    distance_thresholds = calibrate_distance_thresholds(
        collected["known_val"]["distances"],
        collected["known_val"]["targets"],
        args.known_class_accept_rate,
    )
    evaluation = {
        name: evaluate_predictions(
            values["probabilities"],
            values["distances"],
            values["targets"],
            probability_thresholds,
            distance_thresholds,
            class_names,
        )
        for name, values in collected.items()
    }
    per_class_diagnostics = {
        class_name: class_distance_diagnostics(
            class_index,
            class_name,
            collected["known_test"]["probabilities"],
            collected["known_test"]["distances"],
            collected["known_test"]["targets"],
            collected["unknown_test"]["probabilities"],
            collected["unknown_test"]["distances"],
            collected["unknown_test"]["targets"],
            probability_thresholds[class_index],
            distance_thresholds[class_index],
        )
        for class_index, class_name in enumerate(class_names)
    }

    result = {
        "configuration": {
            "data": args.data,
            "unknown_class": args.unknown_class,
            "known_classes": class_names,
            "rdp_checkpoint": str(checkpoint_path),
            "rdp_threshold_json": str(threshold_json),
            "rdp_threshold_scheme": args.rdp_threshold_scheme,
            "known_class_accept_rate_target": args.known_class_accept_rate,
            "device": str(device),
            "prototype_count": int(len(prototypes)),
            "feature_dim": int(prototypes.shape[1]),
            "threshold_selection_uses_unknown_labels": False,
        },
        "thresholds": {
            class_name: {
                "probability": float(probability_thresholds[class_index]),
                "class_conditioned_cosine_distance": float(
                    distance_thresholds[class_index]
                ),
            }
            for class_index, class_name in enumerate(class_names)
        },
        "split_counts": {name: int(len(values)) for name, values in split.items()},
        "evaluation": evaluation,
        "per_class_distance_diagnostics": per_class_diagnostics,
        "notes": {
            "distance": "1 - maximum cosine similarity to a stored composition prototype containing the candidate class",
            "distance_threshold": "class-specific quantile of true known-validation distances",
            "unknown_labels": "used only for final diagnostic metrics, never threshold calibration",
        },
    }
    output_path = Path(
        args.output
        or "checkpoint/rdp_conditioned_dinomaly/"
        f"{args.unknown_class.lower()}_heldout/"
        "rdp_class_conditioned_prototypes.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("class-conditioned prototype comparison:", flush=True)
    for class_name, values in per_class_diagnostics.items():
        false_samples = values["false_unknown_candidates"]["samples"]
        if false_samples == 0:
            continue
        print(
            f"  {class_name}: false_candidates={false_samples} "
            f"AUROC={values.get('false_candidate_auroc')} "
            f"true_accept={values['true_known_accept_rate']} "
            f"false_reject={values['false_unknown_reject_rate']}",
            flush=True,
        )
    before = evaluation["unknown_test"]["probability_only"]
    after = evaluation["unknown_test"]["probability_plus_prototype"]
    print(
        "unknown_test known projection: "
        f"exact {before['exact_match']:.6f}->{after['exact_match']:.6f}, "
        f"micro_f1 {before['micro_f1']:.6f}->{after['micro_f1']:.6f}, "
        f"FP/sample {before['mean_false_positives_per_sample']:.6f}"
        f"->{after['mean_false_positives_per_sample']:.6f}",
        flush=True,
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
