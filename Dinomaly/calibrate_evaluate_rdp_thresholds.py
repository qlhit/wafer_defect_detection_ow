#!/usr/bin/env python3
"""Calibrate per-class RDP thresholds using known validation data only.

Requires ``diagnose_rdp_unknown_projection.py`` in the same directory. The
held-out unknown split is used only once for final diagnostics, never for
threshold selection. No Dinomaly model is loaded and no training is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from diagnose_rdp_unknown_projection import (
    FULL_CLASSES,
    diagnostic_metrics,
    load_arrays,
    load_rdp,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-only threshold calibration for RDP-Net"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument("--rdp-repo", default="../rdp_net_repro")
    parser.add_argument("--rdp-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--per-class-known-fpr",
        type=float,
        default=0.01,
        help="Target validation false-positive rate for the conservative scheme",
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


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
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("known-train-ratio must be in (0,1)")
    if not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("known-val-ratio must be positive and train+val < 1")
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


def max_f1_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    if targets.sum() == 0:
        return 1.0
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best = np.flatnonzero(f1 == np.nanmax(f1))
    # Prefer the largest equally good threshold to reduce false positives.
    return float(thresholds[best[-1]])


def controlled_fpr_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_fpr: float,
) -> float:
    negatives = probabilities[~targets.astype(bool)]
    if len(negatives) == 0:
        return 0.5
    try:
        threshold = float(
            np.quantile(negatives, 1.0 - target_fpr, method="higher")
        )
    except TypeError:
        threshold = float(
            np.quantile(negatives, 1.0 - target_fpr, interpolation="higher")
        )
    # Predictions use >=. Move above a tied quantile if needed.
    if float(np.mean(negatives >= threshold)) > target_fpr:
        threshold = float(np.nextafter(threshold, np.inf))
    return threshold


def calibrate_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_fpr: float,
) -> Dict[str, np.ndarray]:
    class_count = targets.shape[1]
    return {
        "fixed_0_5": np.full(class_count, 0.5, dtype=np.float64),
        "known_val_max_f1": np.asarray(
            [
                max_f1_threshold(probabilities[:, index], targets[:, index])
                for index in range(class_count)
            ],
            dtype=np.float64,
        ),
        "known_val_fpr_control": np.asarray(
            [
                controlled_fpr_threshold(
                    probabilities[:, index], targets[:, index], target_fpr
                )
                for index in range(class_count)
            ],
            dtype=np.float64,
        ),
    }


def summary_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    predictions = probabilities >= thresholds.reshape(1, -1)
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
    true_known_present = targets.any(axis=1)
    predicted_known_present = predictions.any(axis=1)
    recovered_true_component = (predictions & targets).any(axis=1)

    pure_mask = ~true_known_present
    mixed_mask = true_known_present
    result: Dict[str, object] = {
        "samples": int(len(targets)),
        "exact_match": float(np.mean(np.all(predictions == targets, axis=1))),
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
        "known_presence_accuracy": float(
            np.mean(predicted_known_present == true_known_present)
        ),
        "per_class": {
            name: {
                "threshold": float(thresholds[index]),
                "positive_samples": int(support[index]),
                "predicted_positive_samples": int(predictions[:, index].sum()),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "false_positive": int(false_positive[:, index].sum()),
                "false_negative": int(false_negative[:, index].sum()),
                "false_positive_rate": (
                    float(
                        false_positive[:, index].sum()
                        / max(1, (~targets[:, index]).sum())
                    )
                ),
            }
            for index, name in enumerate(class_names)
        },
    }
    if pure_mask.any():
        result["pure_unknown"] = {
            "samples": int(pure_mask.sum()),
            "false_known_presence_rate": float(
                predicted_known_present[pure_mask].mean()
            ),
            "mean_predicted_component_count": float(
                predictions[pure_mask].sum(axis=1).mean()
            ),
        }
    if mixed_mask.any():
        result["mixed_known_plus_unknown"] = {
            "samples": int(mixed_mask.sum()),
            "predicted_known_presence_rate": float(
                predicted_known_present[mixed_mask].mean()
            ),
            "at_least_one_true_component_recovered": float(
                recovered_true_component[mixed_mask].mean()
            ),
        }
    return result


def probability_summary(values: np.ndarray) -> Dict[str, float | None]:
    if len(values) == 0:
        return {"samples": 0, "mean": None, "q05": None, "q50": None, "q95": None, "q99": None}
    return {
        "samples": int(len(values)),
        "mean": float(values.mean()),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.per_class_known_fpr < 1.0:
        raise ValueError("--per-class-known-fpr must be in [0,1)")
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

    probabilities = {}
    targets = {}
    for name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {name}: {len(split[name])}", flush=True)
        probabilities[name] = predict(
            model,
            x,
            split[name],
            active_indices,
            args.batch_size,
            device,
        )
        targets[name] = (
            y[torch.as_tensor(split[name], dtype=torch.long)][:, active_indices]
            .cpu()
            .numpy()
            > 0.5
        )

    schemes = calibrate_thresholds(
        probabilities["known_val"],
        targets["known_val"],
        args.per_class_known_fpr,
    )
    evaluation = {}
    for scheme_name, thresholds in schemes.items():
        evaluation[scheme_name] = {
            "thresholds": {
                name: float(thresholds[index])
                for index, name in enumerate(class_names)
            },
            "known_validation": summary_metrics(
                probabilities["known_val"],
                targets["known_val"],
                thresholds,
                class_names,
            ),
            "known_test": summary_metrics(
                probabilities["known_test"],
                targets["known_test"],
                thresholds,
                class_names,
            ),
            "unknown_test_known_projection": summary_metrics(
                probabilities["unknown_test"],
                targets["unknown_test"],
                thresholds,
                class_names,
            ),
        }

    probability_distributions = {}
    for index, name in enumerate(class_names):
        probability_distributions[name] = {
            "known_validation_when_present": probability_summary(
                probabilities["known_val"][targets["known_val"][:, index], index]
            ),
            "known_validation_when_absent": probability_summary(
                probabilities["known_val"][~targets["known_val"][:, index], index]
            ),
            "known_test_when_present": probability_summary(
                probabilities["known_test"][targets["known_test"][:, index], index]
            ),
            "unknown_test_when_present": probability_summary(
                probabilities["unknown_test"][targets["unknown_test"][:, index], index]
            ),
            "unknown_test_when_absent": probability_summary(
                probabilities["unknown_test"][~targets["unknown_test"][:, index], index]
            ),
        }

    result = {
        "configuration": {
            "data": args.data,
            "unknown_class": args.unknown_class,
            "known_classes": class_names,
            "rdp_checkpoint": str(checkpoint_path),
            "device": str(device),
            "per_class_known_fpr_target": args.per_class_known_fpr,
            "threshold_selection_uses_unknown_labels": False,
        },
        "split_counts": {name: int(len(values)) for name, values in split.items()},
        "threshold_schemes": evaluation,
        "probability_distributions": probability_distributions,
        "notes": {
            "known_val_max_f1": "Each threshold maximizes that class's known-validation F1.",
            "known_val_fpr_control": "Each threshold targets the requested negative-class FPR on known validation.",
            "unknown_test": "Used for final diagnosis only, never threshold calibration.",
        },
    }
    output_path = Path(
        args.output
        or "checkpoint/rdp_conditioned_dinomaly/"
        f"{args.unknown_class.lower()}_heldout/"
        "rdp_threshold_calibration.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("scheme comparison on unknown_test:", flush=True)
    for scheme_name, scheme_result in evaluation.items():
        unknown_metrics = scheme_result["unknown_test_known_projection"]
        d_unknown = unknown_metrics["per_class"].get("D")
        d_known = scheme_result["known_test"]["per_class"].get("D")
        pure_metrics = unknown_metrics.get("pure_unknown", {})
        print(
            f"  {scheme_name}: exact={unknown_metrics['exact_match']:.6f} "
            f"micro_f1={unknown_metrics['micro_f1']:.6f} "
            f"FP/sample={unknown_metrics['mean_false_positives_per_sample']:.6f} "
            f"pure_false_known={pure_metrics.get('false_known_presence_rate', float('nan')):.6f} "
            f"C_to_D_FPR={d_unknown['false_positive_rate'] if d_unknown else float('nan'):.6f} "
            f"true_D_recall={d_known['recall'] if d_known else float('nan'):.6f}",
            flush=True,
        )
    print("thresholds:", flush=True)
    for name in class_names:
        values = " ".join(
            f"{scheme}={evaluation[scheme]['thresholds'][name]:.6f}"
            for scheme in schemes
        )
        print(f"  {name}: {values}", flush=True)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
