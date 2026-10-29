#!/usr/bin/env python3
"""Evaluate a Dinomaly-gated two-expert RDP system.

Routing rule:

  Dinomaly score <= known-only threshold -> original known-only RDP
  Dinomaly score >  known-only threshold -> contamination-robust V1 RDP

The gate threshold is calibrated exclusively from the known validation split.
Held-out samples are used only for final reporting.  No model is trained and
only one JSON file is written.

Place this file in the Dinomaly repository root beside:
  train_eval_dinomaly_atomic_open_set.py
  train_eval_rdp_conditioned_dinomaly.py
  evaluate_rdp_conditioned_dinomaly_full.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from train_eval_dinomaly_atomic_open_set import (
    FULL_CLASSES,
    MixedWM38DinomalyDataset,
    build_model,
    load_arrays,
    make_loader,
    make_open_set_split,
    set_seed,
    torch_load,
)
from train_eval_rdp_conditioned_dinomaly import (
    RDPConditionedDinomaly,
    collect_scores,
    detection_metrics,
    load_frozen_rdp,
)
from evaluate_rdp_conditioned_dinomaly_full import collect_rdp_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dinomaly-gated original/robust RDP expert routing"
    )
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--robust-rdp-checkpoint", required=True)
    parser.add_argument("--original-rdp-checkpoint", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default=None)
    parser.add_argument("--rdp-repo", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rdp-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--rdp-label-threshold", type=float, default=0.5)
    parser.add_argument("--gate-known-reject-rate", type=float, default=0.05)
    parser.add_argument(
        "--sweep-known-reject-rates",
        default="0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50",
        help="Known-validation quantiles used for diagnostic operating points.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def parse_rates(text: str, main_rate: float) -> Sequence[float]:
    rates = [float(item.strip()) for item in text.split(",") if item.strip()]
    rates.append(float(main_rate))
    unique = sorted(set(rates))
    if any(rate <= 0.0 or rate >= 1.0 for rate in unique):
        raise ValueError("All known reject rates must be in (0,1)")
    return unique


def component_metrics(
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
    supported = support > 0
    false_positive = predictions & ~targets
    false_negative = ~predictions & targets
    pure = ~targets.any(axis=1)
    exact_per_sample = np.all(predictions == targets, axis=1)
    return {
        "samples": int(len(targets)),
        "exact_match": float(exact_per_sample.mean()),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "supported_class_macro_f1": (
            float(f1[supported].mean()) if supported.any() else None
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
                    / max(1, (~targets[:, index]).sum())
                ),
            }
            for index, name in enumerate(class_names)
        },
    }


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    d_metrics = metrics["per_class"].get("D", {})
    return {
        "exact_match": metrics["exact_match"],
        "micro_f1": metrics["micro_f1"],
        "mean_false_positives_per_sample": metrics[
            "mean_false_positives_per_sample"
        ],
        "mean_false_negatives_per_sample": metrics[
            "mean_false_negatives_per_sample"
        ],
        "pure_unknown_false_known_rate": metrics[
            "pure_unknown_false_known_rate"
        ],
        "D_false_positive_rate": d_metrics.get("false_positive_rate"),
    }


def routed_predictions(
    original: np.ndarray,
    robust: np.ndarray,
    gate_positive: np.ndarray,
) -> np.ndarray:
    if original.shape != robust.shape:
        raise ValueError("Original and robust RDP prediction shapes differ")
    return np.where(gate_positive[:, None], robust, original)


def exact_array(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.all(predictions.astype(bool) == targets.astype(bool), axis=1)


def route_at_threshold(
    threshold: float,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_original: np.ndarray,
    known_robust: np.ndarray,
    unknown_original: np.ndarray,
    unknown_robust: np.ndarray,
    known_targets: np.ndarray,
    unknown_targets: np.ndarray,
    class_names: Sequence[str],
    include_full: bool,
) -> Dict[str, object]:
    known_gate = known_scores > threshold
    unknown_gate = unknown_scores > threshold
    known_routed = routed_predictions(known_original, known_robust, known_gate)
    unknown_routed = routed_predictions(
        unknown_original, unknown_robust, unknown_gate
    )
    known_metrics = component_metrics(known_routed, known_targets, class_names)
    unknown_metrics = component_metrics(
        unknown_routed, unknown_targets, class_names
    )
    known_exact = exact_array(known_routed, known_targets)
    unknown_exact = exact_array(unknown_routed, unknown_targets)
    known_joint = (~known_gate) & known_exact
    unknown_joint = unknown_gate & unknown_exact
    overall_composition_exact = float(
        (known_exact.sum() + unknown_exact.sum())
        / (len(known_exact) + len(unknown_exact))
    )
    overall_joint_exact = float(
        (known_joint.sum() + unknown_joint.sum())
        / (len(known_joint) + len(unknown_joint))
    )
    result: Dict[str, object] = {
        "threshold": float(threshold),
        "gate": {
            "known_false_route_rate": float(known_gate.mean()),
            "unknown_correct_route_rate": float(unknown_gate.mean()),
        },
        "known_test_components": (
            known_metrics if include_full else compact_metrics(known_metrics)
        ),
        "unknown_test_known_projection": (
            unknown_metrics if include_full else compact_metrics(unknown_metrics)
        ),
        "overall_composition_exact_match": overall_composition_exact,
        "joint_unknown_flag_and_composition_exact_match": {
            "known_test": float(known_joint.mean()),
            "unknown_test": float(unknown_joint.mean()),
            "overall": overall_joint_exact,
        },
    }
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.rdp_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if not 0.0 < args.gate_known_reject_rate < 1.0:
        raise ValueError("--gate-known-reject-rate must be in (0,1)")
    rates = parse_rates(
        args.sweep_known_reject_rates, args.gate_known_reject_rate
    )

    gate_checkpoint_path = Path(args.gate_checkpoint).resolve()
    gate_checkpoint = torch_load(gate_checkpoint_path, map_location="cpu")
    configuration = gate_checkpoint["configuration"]
    unknown_class = str(configuration["unknown_class"])
    if args.unknown_class is not None and args.unknown_class != unknown_class:
        raise ValueError(
            f"Gate checkpoint holds out {unknown_class}, not {args.unknown_class}"
        )
    unknown_index = FULL_CLASSES.index(unknown_class)
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    class_names = [FULL_CLASSES[index] for index in active_indices]
    stored_active = [int(index) for index in configuration["active_known_indices"]]
    if stored_active != active_indices:
        raise ValueError("Gate checkpoint has an unexpected active class order")

    data_path = args.data or str(configuration["data"])
    rdp_repo = Path(
        args.rdp_repo or str(configuration["rdp_repo"])
    ).resolve()
    original_checkpoint_path = Path(
        args.original_rdp_checkpoint or str(configuration["rdp_checkpoint"])
    ).resolve()
    robust_checkpoint_path = Path(args.robust_rdp_checkpoint).resolve()
    top_fraction = float(
        args.top_fraction
        if args.top_fraction is not None
        else configuration["top_fraction"]
    )
    seed = int(configuration["seed"])
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x, y = load_arrays(data_path)
    split = make_open_set_split(y, unknown_index, 0.70, 0.15, seed)
    evaluation_indices = {
        "known_val": split["known_val"],
        "known_test": split["known_test"],
        "unknown_test": split["unknown_test"],
    }
    loaders = {
        name: make_loader(
            MixedWM38DinomalyDataset(
                x,
                y,
                indices,
                image_size=int(configuration["image_size"]),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=seed + offset,
        )
        for offset, (name, indices) in enumerate(evaluation_indices.items())
    }

    print(f"device: {device}")
    print(f"unknown routing experiment: {unknown_class} held out")
    print(f"gate checkpoint: {gate_checkpoint_path}")
    print(f"original expert: {original_checkpoint_path}")
    print(f"robust expert: {robust_checkpoint_path}", flush=True)

    original_rdp, original_active = load_frozen_rdp(
        rdp_repo,
        original_checkpoint_path,
        unknown_class,
        device,
    )
    robust_rdp, robust_active = load_frozen_rdp(
        rdp_repo,
        robust_checkpoint_path,
        unknown_class,
        device,
    )
    if original_active != active_indices or robust_active != active_indices:
        raise ValueError("RDP expert active class orders differ")

    base, bottleneck, decoder, _ = build_model(
        str(configuration["encoder"]),
        int(configuration["decoder_depth"]),
        device,
    )
    feature_dim = bottleneck[0].fc2.out_features
    gate_model = RDPConditionedDinomaly(
        base,
        condition_dim=len(active_indices),
        condition_hidden_dim=int(configuration["condition_hidden_dim"]),
        feature_dim=feature_dim,
        condition_scale=float(configuration["condition_scale"]),
    ).to(device)
    bottleneck.load_state_dict(
        gate_checkpoint["bottleneck_state_dict"], strict=True
    )
    decoder.load_state_dict(gate_checkpoint["decoder_state_dict"], strict=True)
    gate_model.conditioner.load_state_dict(
        gate_checkpoint["conditioner_state_dict"], strict=True
    )
    gate_model.eval()

    scores: Dict[str, np.ndarray] = {}
    indices_by_split: Dict[str, np.ndarray] = {}
    original_predictions: Dict[str, np.ndarray] = {}
    robust_predictions: Dict[str, np.ndarray] = {}
    targets: Dict[str, np.ndarray] = {}
    for name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {name}", flush=True)
        split_scores, split_indices, _ = collect_scores(
            gate_model,
            original_rdp,
            loaders[name],
            x,
            active_indices,
            device,
            int(configuration["image_size"]),
            top_fraction,
            args.rdp_label_threshold,
        )
        _, original_pred, split_targets = collect_rdp_predictions(
            original_rdp,
            x,
            y,
            split_indices,
            active_indices,
            args.rdp_batch_size,
            args.rdp_label_threshold,
            device,
        )
        _, robust_pred, robust_targets = collect_rdp_predictions(
            robust_rdp,
            x,
            y,
            split_indices,
            active_indices,
            args.rdp_batch_size,
            args.rdp_label_threshold,
            device,
        )
        if not np.array_equal(split_targets, robust_targets):
            raise RuntimeError("Expert target arrays are misaligned")
        scores[name] = split_scores
        indices_by_split[name] = split_indices
        original_predictions[name] = original_pred
        robust_predictions[name] = robust_pred
        targets[name] = split_targets

    gate_detection = detection_metrics(
        scores["known_val"],
        scores["known_test"],
        scores["unknown_test"],
        args.gate_known_reject_rate,
    )
    gate_labels = np.concatenate(
        [
            np.zeros(len(scores["known_test"]), dtype=np.int64),
            np.ones(len(scores["unknown_test"]), dtype=np.int64),
        ]
    )
    gate_scores = np.concatenate(
        [scores["known_test"], scores["unknown_test"]]
    )
    gate_detection["auroc"] = float(roc_auc_score(gate_labels, gate_scores))
    gate_detection["aupr"] = float(
        average_precision_score(gate_labels, gate_scores)
    )

    original_known_metrics = component_metrics(
        original_predictions["known_test"],
        targets["known_test"],
        class_names,
    )
    original_unknown_metrics = component_metrics(
        original_predictions["unknown_test"],
        targets["unknown_test"],
        class_names,
    )
    robust_known_metrics = component_metrics(
        robust_predictions["known_test"],
        targets["known_test"],
        class_names,
    )
    robust_unknown_metrics = component_metrics(
        robust_predictions["unknown_test"],
        targets["unknown_test"],
        class_names,
    )
    oracle_known_exact = exact_array(
        original_predictions["known_test"], targets["known_test"]
    )
    oracle_unknown_exact = exact_array(
        robust_predictions["unknown_test"], targets["unknown_test"]
    )
    oracle_overall_exact = float(
        (oracle_known_exact.sum() + oracle_unknown_exact.sum())
        / (len(oracle_known_exact) + len(oracle_unknown_exact))
    )

    operating_points = {}
    main_result = None
    for rate in rates:
        threshold = float(
            np.quantile(scores["known_val"], 1.0 - rate)
        )
        is_main = abs(rate - args.gate_known_reject_rate) < 1e-12
        routed = route_at_threshold(
            threshold,
            scores["known_test"],
            scores["unknown_test"],
            original_predictions["known_test"],
            robust_predictions["known_test"],
            original_predictions["unknown_test"],
            robust_predictions["unknown_test"],
            targets["known_test"],
            targets["unknown_test"],
            class_names,
            include_full=is_main,
        )
        operating_points[f"known_reject_{rate:.3f}"] = routed
        if is_main:
            main_result = routed
    if main_result is None:
        raise RuntimeError("Main gate operating point was not evaluated")

    result = {
        "configuration": {
            "method": "dinomaly_gated_original_and_robust_rdp",
            "unknown_class": unknown_class,
            "known_classes": class_names,
            "gate_checkpoint": str(gate_checkpoint_path),
            "gate_condition_rdp_checkpoint": str(original_checkpoint_path),
            "original_rdp_checkpoint": str(original_checkpoint_path),
            "robust_rdp_checkpoint": str(robust_checkpoint_path),
            "data": data_path,
            "device": str(device),
            "top_fraction": top_fraction,
            "rdp_label_threshold": args.rdp_label_threshold,
            "main_gate_known_reject_rate": args.gate_known_reject_rate,
            "threshold_source": "known validation only",
            "heldout_labels_used_for_threshold_selection": False,
        },
        "split_counts": {
            name: int(len(values)) for name, values in indices_by_split.items()
        },
        "gate_unknown_detection": gate_detection,
        "expert_baselines": {
            "always_original": {
                "known_test_components": original_known_metrics,
                "unknown_test_known_projection": original_unknown_metrics,
            },
            "always_robust_v1": {
                "known_test_components": robust_known_metrics,
                "unknown_test_known_projection": robust_unknown_metrics,
            },
            "oracle_gate_upper_bound": {
                "description": (
                    "Original RDP for every known wafer and robust V1 RDP for "
                    "every held-out wafer; uses true routing only as an upper bound."
                ),
                "known_test_exact_match": float(oracle_known_exact.mean()),
                "unknown_test_projection_exact_match": float(
                    oracle_unknown_exact.mean()
                ),
                "overall_composition_exact_match": oracle_overall_exact,
            },
        },
        "main_operating_point": main_result,
        "known_only_threshold_sweep": operating_points,
        "notes": {
            "routing": (
                "Gate negative uses original RDP; gate positive uses robust V1 RDP."
            ),
            "joint_exact": (
                "Requires both the unknown flag and every known-component label "
                "to be correct."
            ),
            "sweep_warning": (
                "Sweep rows are diagnostic. Do not choose a paper threshold from "
                "held-out results; predeclare the known-validation reject rate."
            ),
        },
    }
    output_path = Path(
        args.output
        or f"checkpoint/rdp_expert_router/{unknown_class.lower()}_heldout/metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "expert baselines: "
        f"original known={original_known_metrics['exact_match']:.6f} "
        f"unknown={original_unknown_metrics['exact_match']:.6f}; "
        f"robust known={robust_known_metrics['exact_match']:.6f} "
        f"unknown={robust_unknown_metrics['exact_match']:.6f}",
        flush=True,
    )
    print("routing sweep:", flush=True)
    for key, row in operating_points.items():
        known_metrics = row["known_test_components"]
        unknown_metrics = row["unknown_test_known_projection"]
        print(
            f"  {key}: gate_known={row['gate']['known_false_route_rate']:.4f} "
            f"gate_unknown={row['gate']['unknown_correct_route_rate']:.4f} "
            f"known_exact={known_metrics['exact_match']:.4f} "
            f"unknown_exact={unknown_metrics['exact_match']:.4f} "
            f"C_to_D={unknown_metrics['D_false_positive_rate'] if 'D_false_positive_rate' in unknown_metrics else unknown_metrics['per_class'].get('D', {}).get('false_positive_rate')} "
            f"joint={row['joint_unknown_flag_and_composition_exact_match']['overall']:.4f}",
            flush=True,
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
