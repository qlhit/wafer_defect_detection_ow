#!/usr/bin/env python3
"""Re-score a trained spatial decomposer without retraining it.

The original spatial verifier used:

    support = presence_logit - shape_distance_weight * shape_distance

For difficult held-out defects, a highly confident presence logit can hide an
abnormal mask shape.  This evaluator calibrates and compares four candidate
filters using known validation only:

  * current_support: the original combined score;
  * presence_only: the image-level presence logit;
  * shape_distance_only: an upper bound on class-mask shape distance;
  * dual_gate: independent presence lower bound and shape-distance upper bound.

The held-out split is evaluated only after every threshold is fixed.  No model
is trained and only one JSON file is written.

Place this file beside ``train_eval_spatial_decomposition_verifier.py`` in the
RDP repository's ``scripts/`` directory and run it from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from train_eval_reciprocal_component_rejection import (
    FULL_CLASSES,
    load_arrays,
    load_frozen_rdp,
    make_split,
    multilabel_metrics,
    torch_load,
)
from train_eval_spatial_decomposition_verifier import (
    AtomicMaskNet,
    collect_decomposition_outputs,
    load_v2_model,
    rcr_predictions,
    spatial_support,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare distance-only and combined spatial RCR gates"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="ER")
    parser.add_argument(
        "--target-class",
        choices=FULL_CLASSES,
        default=None,
        help=(
            "Apply the new spatial gates only to this known class. "
            "Example: --target-class EL for the ER-to-EL confusion. "
            "Default: gate every active known class."
        ),
    )
    parser.add_argument(
        "--spatial-checkpoint",
        default=None,
        help=(
            "Default: checkpoint/rdp_spatial_decomposition/"
            "<class>_heldout/best_spatial_decomposer.pt"
        ),
    )
    parser.add_argument("--v2-checkpoint", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--positive-retention",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--max-known-val-em-drop",
        type=float,
        default=0.005,
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Default: same directory as spatial checkpoint, "
            "distance_gate_metrics.json"
        ),
    )
    return parser.parse_args()


def resolve_spatial_checkpoint(
    explicit: str | None, unknown_class: str
) -> Path:
    path = Path(
        explicit
        or (
            "checkpoint/rdp_spatial_decomposition/"
            f"{unknown_class.lower()}_heldout/best_spatial_decomposer.pt"
        )
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Spatial checkpoint not found: {path}")
    return path


def as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def retention_grid(initial: float) -> List[float]:
    return sorted(
        {
            float(initial),
            max(float(initial), 0.9975),
            max(float(initial), 0.9990),
            max(float(initial), 0.9995),
            1.0,
        }
    )


def calibrate_high_score_gate(
    baseline: np.ndarray,
    targets: np.ndarray,
    score: np.ndarray,
    class_names: Sequence[str],
    initial_retention: float,
    max_em_drop: float,
    gated_indices: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Calibrate a gate for which larger scores mean more known-like."""
    baseline = baseline.astype(bool)
    truth = targets > 0.5
    baseline_em = float(np.mean(np.all(baseline == truth, axis=1)))
    selected_thresholds = None
    selected_prediction = baseline
    selected_retention = 1.0
    attempts: List[Dict[str, float]] = []

    for retention in retention_grid(initial_retention):
        thresholds = np.empty(score.shape[1], dtype=np.float32)
        accepted = np.ones_like(baseline, dtype=bool)
        for class_index, class_name in enumerate(class_names):
            if class_index not in gated_indices:
                thresholds[class_index] = -np.inf
                accepted[:, class_index] = True
                continue
            pool = baseline[:, class_index] & truth[:, class_index]
            if not pool.any():
                raise ValueError(
                    f"No true RCR-v2 validation candidate for {class_name}"
                )
            thresholds[class_index] = float(
                np.quantile(score[pool, class_index], 1.0 - retention)
            )
            accepted[:, class_index] = (
                score[:, class_index] >= thresholds[class_index]
            )
        prediction = baseline & accepted
        em = float(np.mean(np.all(prediction == truth, axis=1)))
        attempts.append(
            {
                "positive_retention": float(retention),
                "known_validation_exact_match": em,
                "change_from_baseline": em - baseline_em,
            }
        )
        if em >= baseline_em - max_em_drop:
            selected_thresholds = thresholds
            selected_prediction = prediction
            selected_retention = float(retention)
            break
    if selected_thresholds is None:
        raise RuntimeError("High-score gate calibration failed")

    per_class = {}
    for class_index, class_name in enumerate(class_names):
        gated = class_index in gated_indices
        true_pool = baseline[:, class_index] & truth[:, class_index]
        false_pool = baseline[:, class_index] & ~truth[:, class_index]
        accepted = score[:, class_index] >= selected_thresholds[class_index]
        per_class[class_name] = {
            "gated": gated,
            "threshold": (
                float(selected_thresholds[class_index]) if gated else None
            ),
            "true_candidate_retention": float(
                accepted[true_pool].mean() if true_pool.any() else 0.0
            ),
            "false_candidate_rejection": (
                float(1.0 - accepted[false_pool].mean())
                if false_pool.any()
                else None
            ),
            "true_candidates": int(true_pool.sum()),
            "false_candidates": int(false_pool.sum()),
        }
    return selected_thresholds, {
        "baseline_known_validation_exact_match": baseline_em,
        "gated_known_validation_exact_match": float(
            np.mean(np.all(selected_prediction == truth, axis=1))
        ),
        "selected_positive_retention": selected_retention,
        "retention_attempts": attempts,
        "per_class": per_class,
    }


def calibrate_dual_gate(
    baseline: np.ndarray,
    targets: np.ndarray,
    presence: np.ndarray,
    distance: np.ndarray,
    class_names: Sequence[str],
    initial_retention: float,
    max_em_drop: float,
    gated_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Calibrate presence >= floor and distance <= ceiling separately."""
    baseline = baseline.astype(bool)
    truth = targets > 0.5
    baseline_em = float(np.mean(np.all(baseline == truth, axis=1)))
    selected_presence = None
    selected_distance = None
    selected_prediction = baseline
    selected_retention = 1.0
    attempts: List[Dict[str, float]] = []

    for retention in retention_grid(initial_retention):
        presence_floors = np.empty(presence.shape[1], dtype=np.float32)
        distance_ceilings = np.empty(distance.shape[1], dtype=np.float32)
        accepted = np.ones_like(baseline, dtype=bool)
        for class_index, class_name in enumerate(class_names):
            if class_index not in gated_indices:
                presence_floors[class_index] = -np.inf
                distance_ceilings[class_index] = np.inf
                accepted[:, class_index] = True
                continue
            pool = baseline[:, class_index] & truth[:, class_index]
            if not pool.any():
                raise ValueError(
                    f"No true RCR-v2 validation candidate for {class_name}"
                )
            presence_floors[class_index] = float(
                np.quantile(presence[pool, class_index], 1.0 - retention)
            )
            distance_ceilings[class_index] = float(
                np.quantile(distance[pool, class_index], retention)
            )
            accepted[:, class_index] = (
                presence[:, class_index] >= presence_floors[class_index]
            ) & (
                distance[:, class_index] <= distance_ceilings[class_index]
            )
        prediction = baseline & accepted
        em = float(np.mean(np.all(prediction == truth, axis=1)))
        attempts.append(
            {
                "per_gate_positive_retention": float(retention),
                "known_validation_exact_match": em,
                "change_from_baseline": em - baseline_em,
            }
        )
        if em >= baseline_em - max_em_drop:
            selected_presence = presence_floors
            selected_distance = distance_ceilings
            selected_prediction = prediction
            selected_retention = float(retention)
            break
    if selected_presence is None or selected_distance is None:
        raise RuntimeError("Dual gate calibration failed")

    per_class = {}
    for class_index, class_name in enumerate(class_names):
        gated = class_index in gated_indices
        true_pool = baseline[:, class_index] & truth[:, class_index]
        false_pool = baseline[:, class_index] & ~truth[:, class_index]
        accepted = (
            presence[:, class_index] >= selected_presence[class_index]
        ) & (
            distance[:, class_index] <= selected_distance[class_index]
        )
        per_class[class_name] = {
            "gated": gated,
            "presence_floor": (
                float(selected_presence[class_index]) if gated else None
            ),
            "distance_ceiling": (
                float(selected_distance[class_index]) if gated else None
            ),
            "true_candidate_retention": float(
                accepted[true_pool].mean() if true_pool.any() else 0.0
            ),
            "false_candidate_rejection": (
                float(1.0 - accepted[false_pool].mean())
                if false_pool.any()
                else None
            ),
            "true_candidates": int(true_pool.sum()),
            "false_candidates": int(false_pool.sum()),
        }
    return selected_presence, selected_distance, {
        "baseline_known_validation_exact_match": baseline_em,
        "gated_known_validation_exact_match": float(
            np.mean(np.all(selected_prediction == truth, axis=1))
        ),
        "selected_per_gate_positive_retention": selected_retention,
        "retention_attempts": attempts,
        "per_class": per_class,
    }


def score_dataset(
    model: AtomicMaskNet,
    base_model: torch.nn.Module,
    verifier: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    defect_value: float,
    valid_threshold: float,
    descriptor_args: argparse.Namespace,
    centers: np.ndarray,
    scales: np.ndarray,
    shape_distance_weight: float,
    rcr_distance_weight: float,
    rcr_label_threshold: float,
    rcr_support_thresholds: np.ndarray,
    name: str,
) -> Dict[str, np.ndarray]:
    baseline, targets = rcr_predictions(
        base_model,
        verifier,
        x,
        y,
        indices,
        active_indices,
        batch_size,
        device,
        rcr_distance_weight,
        rcr_label_threshold,
        rcr_support_thresholds,
        name,
    )
    presence, descriptors, spatial_targets = collect_decomposition_outputs(
        model,
        x,
        y,
        indices,
        active_indices,
        batch_size,
        defect_value,
        valid_threshold,
        device,
        descriptor_args,
        name,
    )
    if not np.array_equal(targets > 0.5, spatial_targets > 0.5):
        raise RuntimeError(f"{name} target order mismatch")
    combined, distance = spatial_support(
        presence,
        descriptors,
        centers,
        scales,
        shape_distance_weight,
    )
    return {
        "baseline": baseline.astype(bool),
        "targets": targets.astype(np.float32),
        "presence": presence.astype(np.float32),
        "distance": distance.astype(np.float32),
        "combined": combined.astype(np.float32),
    }


def apply_high_score_gate(
    baseline: np.ndarray,
    score: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    return baseline & (score >= thresholds.reshape(1, -1))


def apply_dual_gate(
    baseline: np.ndarray,
    presence: np.ndarray,
    distance: np.ndarray,
    presence_floors: np.ndarray,
    distance_ceilings: np.ndarray,
) -> np.ndarray:
    return baseline & (
        presence >= presence_floors.reshape(1, -1)
    ) & (
        distance <= distance_ceilings.reshape(1, -1)
    )


def concise(metrics: Dict[str, object]) -> str:
    return (
        f"EM={metrics['exact_match']:.6f} "
        f"micro_F1={metrics['micro_f1']:.6f} "
        f"FP/sample={metrics['mean_false_positives_per_sample']:.6f} "
        f"pure_false={metrics['pure_unknown_false_known_rate']}"
    )


def quantiles(values: np.ndarray) -> Dict[str, float] | None:
    if len(values) == 0:
        return None
    return {
        "mean": float(np.mean(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.5 < args.positive_retention <= 1.0:
        raise ValueError("--positive-retention must be in (0.5,1.0]")
    if not 0.0 <= args.max_known_val_em_drop < 1.0:
        raise ValueError("--max-known-val-em-drop must be in [0,1)")

    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    class_names = [FULL_CLASSES[index] for index in active_indices]
    if args.target_class == args.unknown_class:
        raise ValueError("--target-class must be an active known class")
    if args.target_class is None:
        gated_indices = list(range(len(class_names)))
    else:
        if args.target_class not in class_names:
            raise ValueError(
                f"--target-class {args.target_class} is not in {class_names}"
            )
        gated_indices = [class_names.index(args.target_class)]
    spatial_path = resolve_spatial_checkpoint(
        args.spatial_checkpoint, args.unknown_class
    )
    spatial_checkpoint = torch_load(spatial_path, map_location="cpu")
    if spatial_checkpoint.get("unknown_class") != args.unknown_class:
        raise ValueError(
            f"Spatial checkpoint holds out "
            f"{spatial_checkpoint.get('unknown_class')}, "
            f"requested {args.unknown_class}"
        )
    stored_active = [
        int(value) for value in spatial_checkpoint["active_known_indices"]
    ]
    if stored_active != active_indices:
        raise ValueError(f"Active class mismatch: {stored_active}")

    v2_path = Path(
        args.v2_checkpoint or spatial_checkpoint["v2_checkpoint"]
    ).resolve()
    v2_checkpoint = torch_load(v2_path, map_location="cpu")
    base_path = Path(
        args.base_checkpoint or spatial_checkpoint["base_checkpoint"]
    ).resolve()
    output_path = Path(
        args.output or spatial_path.parent / "distance_gate_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"unknown={args.unknown_class}; known classes={class_names}")
    print(
        "spatial gates applied to:",
        [class_names[index] for index in gated_indices],
    )
    print(f"loading spatial decomposer: {spatial_path}")
    print(f"loading RCR-v2: {v2_path}")

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
        + ", ".join(f"{name}={len(value)}" for name, value in split.items())
    )

    model_config = spatial_checkpoint["model_configuration"]
    model = AtomicMaskNet(
        class_count=int(model_config["class_count"]),
        base_channels=int(model_config["base_channels"]),
    ).to(device)
    model.load_state_dict(spatial_checkpoint["model_state_dict"], strict=True)
    model.eval()

    descriptor_config = spatial_checkpoint["descriptor_configuration"]
    descriptor_args = argparse.Namespace(
        mask_threshold=float(descriptor_config["mask_threshold"]),
        mask_top_fraction=float(descriptor_config["mask_top_fraction"]),
        radial_bins=int(descriptor_config["radial_bins"]),
        angular_bins=int(descriptor_config["angular_bins"]),
    )
    shape_distance_weight = float(
        descriptor_config.get("shape_distance_weight", 1.0)
    )
    centers = as_numpy(spatial_checkpoint["shape_centers"])
    scales = as_numpy(spatial_checkpoint["shape_scales"])
    data_config = spatial_checkpoint["data_values"]
    valid_threshold = float(data_config["valid_threshold"])
    defect_value = float(data_config["defect_value"])

    base_model, base_active, _ = load_frozen_rdp(
        base_path, args.unknown_class, device
    )
    if base_active != active_indices:
        raise ValueError(f"Frozen RDP active class mismatch: {base_active}")
    verifier = load_v2_model(v2_checkpoint, device)
    rcr_distance_weight = float(v2_checkpoint["distance_weight"])
    rcr_label_threshold = float(v2_checkpoint["label_threshold"])
    rcr_support_thresholds = as_numpy(v2_checkpoint["support_thresholds"])

    print("calibrating all gates on known_val")
    validation = score_dataset(
        model,
        base_model,
        verifier,
        x,
        y,
        split["known_val"],
        active_indices,
        device,
        args.batch_size,
        defect_value,
        valid_threshold,
        descriptor_args,
        centers,
        scales,
        shape_distance_weight,
        rcr_distance_weight,
        rcr_label_threshold,
        rcr_support_thresholds,
        "known_val",
    )

    current_thresholds, current_calibration = calibrate_high_score_gate(
        validation["baseline"],
        validation["targets"],
        validation["combined"],
        class_names,
        args.positive_retention,
        args.max_known_val_em_drop,
        gated_indices,
    )
    presence_thresholds, presence_calibration = calibrate_high_score_gate(
        validation["baseline"],
        validation["targets"],
        validation["presence"],
        class_names,
        args.positive_retention,
        args.max_known_val_em_drop,
        gated_indices,
    )
    distance_thresholds_as_score, distance_calibration = (
        calibrate_high_score_gate(
            validation["baseline"],
            validation["targets"],
            -validation["distance"],
            class_names,
            args.positive_retention,
            args.max_known_val_em_drop,
            gated_indices,
        )
    )
    distance_ceilings = -distance_thresholds_as_score
    (
        dual_presence_floors,
        dual_distance_ceilings,
        dual_calibration,
    ) = calibrate_dual_gate(
        validation["baseline"],
        validation["targets"],
        validation["presence"],
        validation["distance"],
        class_names,
        args.positive_retention,
        args.max_known_val_em_drop,
        gated_indices,
    )

    calibrations = {
        "current_support": current_calibration,
        "presence_only": presence_calibration,
        "shape_distance_only": distance_calibration,
        "dual_gate": dual_calibration,
    }
    thresholds = {
        "current_support": current_thresholds,
        "presence_only": presence_thresholds,
        "shape_distance_only": distance_ceilings,
        "dual_presence_floors": dual_presence_floors,
        "dual_distance_ceilings": dual_distance_ceilings,
    }

    def evaluate(name: str) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
        scored = score_dataset(
            model,
            base_model,
            verifier,
            x,
            y,
            split[name],
            active_indices,
            device,
            args.batch_size,
            defect_value,
            valid_threshold,
            descriptor_args,
            centers,
            scales,
            shape_distance_weight,
            rcr_distance_weight,
            rcr_label_threshold,
            rcr_support_thresholds,
            name,
        )
        predictions = {
            "rcr_v2": scored["baseline"],
            "current_support": apply_high_score_gate(
                scored["baseline"], scored["combined"], current_thresholds
            ),
            "presence_only": apply_high_score_gate(
                scored["baseline"], scored["presence"], presence_thresholds
            ),
            "shape_distance_only": scored["baseline"] & (
                scored["distance"] <= distance_ceilings.reshape(1, -1)
            ),
            "dual_gate": apply_dual_gate(
                scored["baseline"],
                scored["presence"],
                scored["distance"],
                dual_presence_floors,
                dual_distance_ceilings,
            ),
        }
        metrics = {
            method: multilabel_metrics(
                prediction, scored["targets"], class_names
            )
            for method, prediction in predictions.items()
        }
        return metrics, scored

    print("scoring known_test")
    known_metrics, known_scored = evaluate("known_test")
    print("scoring unknown_test (final diagnostic only)")
    unknown_metrics, unknown_scored = evaluate("unknown_test")

    per_class_distance = {}
    for class_index, class_name in enumerate(class_names):
        known_true = known_scored["targets"][:, class_index] > 0.5
        unknown_false_candidate = (
            unknown_scored["baseline"][:, class_index]
            & ~(unknown_scored["targets"][:, class_index] > 0.5)
        )
        per_class_distance[class_name] = {
            "known_true": quantiles(
                known_scored["distance"][known_true, class_index]
            ),
            "unknown_false_rcr_candidate": quantiles(
                unknown_scored["distance"][
                    unknown_false_candidate, class_index
                ]
            ),
            "distance_only_ceiling": float(distance_ceilings[class_index]),
        }

    result = {
        "configuration": {
            "method": "known_only_calibrated_spatial_gate_comparison",
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_index,
            "known_classes": class_names,
            "target_class": args.target_class,
            "gated_classes": [
                class_names[index] for index in gated_indices
            ],
            "data": str(Path(args.data).resolve()),
            "spatial_checkpoint": str(spatial_path),
            "v2_checkpoint": str(v2_path),
            "base_checkpoint": str(base_path),
            "positive_retention": args.positive_retention,
            "max_known_val_em_drop": args.max_known_val_em_drop,
            "heldout_data_used_for_training_selection_or_calibration": False,
            "primary_predeclared_scheme": "shape_distance_only",
            "seed": args.seed,
        },
        "split_counts": {
            name: int(len(value)) for name, value in split.items()
        },
        "known_validation_calibration": calibrations,
        "thresholds": {
            name: {
                class_name: (
                    float(value[class_index])
                    if np.isfinite(value[class_index])
                    else None
                )
                for class_index, class_name in enumerate(class_names)
            }
            for name, value in thresholds.items()
        },
        "known_test": known_metrics,
        "unknown_test": unknown_metrics,
        "shape_distance_diagnostics": per_class_distance,
        "notes": {
            "unknown_gate": "Dinomaly remains the wafer-level unknown detector",
            "model_retrained": False,
            "unknown_test_role": "final diagnostic only",
            "selection_warning": (
                "Do not select a paper method using held-out-test performance; "
                "shape_distance_only is the predeclared primary comparison."
            ),
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("\ncomparison")
    print("known_test")
    for method, values in known_metrics.items():
        print(f"  {method}: {concise(values)}")
    print("unknown_test")
    for method, values in unknown_metrics.items():
        print(f"  {method}: {concise(values)}")
    if args.unknown_class == "ER" and "EL" in class_names:
        print("ER-like confusion: EL false-positive rate")
        for method, values in unknown_metrics.items():
            rate = values["per_class"]["EL"]["false_positive_rate"]
            print(f"  {method}: {rate:.6f}")
        el_index = class_names.index("EL")
        print(
            "EL distance ceiling:",
            f"{distance_ceilings[el_index]:.6f}",
        )
    print("saved:", output_path)


if __name__ == "__main__":
    main()
