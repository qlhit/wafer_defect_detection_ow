#!/usr/bin/env python3
"""Fit and evaluate an input-topology gate on top of a trained RCR-v2.

The frozen RDP and trained RCR-v2 propose known labels.  This script adds an
independent gate derived directly from the wafer defect mask.  Its prototypes
are grouped by known composition code and fitted from known training wafers.
Every class threshold is calibrated on known validation positives only.  The
held-out class is loaded only for final diagnostic evaluation.

Only two files are written beside the supplied RCR-v2 checkpoint:
  * best_topology_gate.pt
  * topology_metrics.json

Place this file beside train_eval_reciprocal_component_rejection_v2.py in
rdp_net_repro/scripts/ and run it from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from train_eval_contamination_robust_rdp import data_values
from train_eval_reciprocal_component_rejection import (
    FULL_CLASSES,
    load_arrays,
    load_frozen_rdp,
    make_split,
    multilabel_metrics,
    torch_load,
)
from train_eval_reciprocal_component_rejection_v2 import (
    SpatialMultiPrototypeVerifier,
    collect_scores,
    extract_multiscale_features,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-only raw-wafer topology gate for RCR-v2"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="ER")
    parser.add_argument(
        "--v2-checkpoint",
        default=None,
        help=(
            "Default: checkpoint/rdp_reciprocal_v2/"
            "<class>_heldout/best_spatial_reciprocal_verifier.pt"
        ),
    )
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--descriptor-batch-size", type=int, default=2048)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--topology-positive-retention",
        type=float,
        default=0.995,
        help="Retain this fraction of true known-validation v2 candidates",
    )
    parser.add_argument(
        "--minimum-prototype-samples",
        type=int,
        default=10,
        help="Minimum train samples for one exact-composition prototype",
    )
    parser.add_argument(
        "--variance-shrinkage",
        type=float,
        default=0.50,
        help="Blend each composition variance with its class-wide variance",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=0.05,
        help="Minimum standardized descriptor variance",
    )
    parser.add_argument("--radial-bins", type=int, default=8)
    parser.add_argument("--angular-bins", type=int, default=12)
    parser.add_argument("--projection-bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_v2_checkpoint(explicit: str | None, unknown_class: str) -> Path:
    path = Path(
        explicit
        or (
            f"checkpoint/rdp_reciprocal_v2/{unknown_class.lower()}_heldout/"
            "best_spatial_reciprocal_verifier.pt"
        )
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RCR-v2 checkpoint not found: {path}")
    return path


def coordinate_bins(
    height: int,
    width: int,
    radial_bins: int,
    angular_bins: int,
    projection_bins: int,
) -> Dict[str, np.ndarray]:
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    x_grid = np.broadcast_to(xx, (height, width))
    y_grid = np.broadcast_to(yy, (height, width))
    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid) / math.sqrt(2.0)
    angle = (np.arctan2(y_grid, x_grid) + math.pi) / (2.0 * math.pi)
    radial_index = np.minimum((radius * radial_bins).astype(np.int64), radial_bins - 1)
    angular_index = np.minimum(
        (angle * angular_bins).astype(np.int64), angular_bins - 1
    )
    x_index = np.minimum(
        (((x_grid + 1.0) * 0.5) * projection_bins).astype(np.int64),
        projection_bins - 1,
    )
    y_index = np.minimum(
        (((y_grid + 1.0) * 0.5) * projection_bins).astype(np.int64),
        projection_bins - 1,
    )
    return {
        "x": x_grid.reshape(-1),
        "y": y_grid.reshape(-1),
        "radius": radius.reshape(-1),
        "radial_index": radial_index.reshape(-1),
        "angular_index": angular_index.reshape(-1),
        "x_index": x_index.reshape(-1),
        "y_index": y_index.reshape(-1),
    }


def normalized_histogram(
    defect: np.ndarray,
    bin_index: np.ndarray,
    bin_count: int,
    denominator: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(defect), bin_count), dtype=np.float32)
    for index in range(bin_count):
        output[:, index] = defect[:, bin_index == index].sum(axis=1)
    return output / denominator[:, None]


def topology_descriptors(
    wafers: torch.Tensor,
    indices: np.ndarray,
    valid_threshold: float,
    normal_value: float,
    defect_value: float,
    radial_bins: int,
    angular_bins: int,
    projection_bins: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    height, width = int(wafers.shape[-2]), int(wafers.shape[-1])
    coordinates = coordinate_bins(
        height, width, radial_bins, angular_bins, projection_bins
    )
    descriptor_parts: List[np.ndarray] = []
    cutoff = (normal_value + defect_value) / 2.0
    names = (
        ["defect_ratio", "radial_mean", "radial_std", "x_mean", "y_mean", "x_std", "y_std", "xy_cov"]
        + [f"radial_{i}" for i in range(radial_bins)]
        + [f"angular_{i}" for i in range(angular_bins)]
        + [f"x_projection_{i}" for i in range(projection_bins)]
        + [f"y_projection_{i}" for i in range(projection_bins)]
        + ["radial_coverage", "angular_coverage", "x_coverage", "y_coverage", "row_peak", "column_peak"]
    )
    for start in range(0, len(indices), batch_size):
        local_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        local = wafers[local_indices].cpu().numpy().astype(np.float32)
        flat = local.reshape(len(local), -1)
        valid = flat > valid_threshold
        defect = flat > cutoff
        valid_count = np.maximum(valid.sum(axis=1).astype(np.float32), 1.0)
        defect_count = np.maximum(defect.sum(axis=1).astype(np.float32), 1.0)
        defect_float = defect.astype(np.float32)

        x = coordinates["x"][None, :]
        y = coordinates["y"][None, :]
        radius = coordinates["radius"][None, :]
        x_mean = (defect_float * x).sum(axis=1) / defect_count
        y_mean = (defect_float * y).sum(axis=1) / defect_count
        radial_mean = (defect_float * radius).sum(axis=1) / defect_count
        radial_std = np.sqrt(
            np.maximum(
                (defect_float * (radius - radial_mean[:, None]) ** 2).sum(axis=1)
                / defect_count,
                0.0,
            )
        )
        x_std = np.sqrt(
            np.maximum(
                (defect_float * (x - x_mean[:, None]) ** 2).sum(axis=1)
                / defect_count,
                0.0,
            )
        )
        y_std = np.sqrt(
            np.maximum(
                (defect_float * (y - y_mean[:, None]) ** 2).sum(axis=1)
                / defect_count,
                0.0,
            )
        )
        xy_cov = (
            defect_float
            * (x - x_mean[:, None])
            * (y - y_mean[:, None])
        ).sum(axis=1) / defect_count

        radial = normalized_histogram(
            defect_float,
            coordinates["radial_index"],
            radial_bins,
            defect_count,
        )
        angular = normalized_histogram(
            defect_float,
            coordinates["angular_index"],
            angular_bins,
            defect_count,
        )
        x_projection = normalized_histogram(
            defect_float,
            coordinates["x_index"],
            projection_bins,
            defect_count,
        )
        y_projection = normalized_histogram(
            defect_float,
            coordinates["y_index"],
            projection_bins,
            defect_count,
        )
        coverage = np.stack(
            (
                (radial > 0).mean(axis=1),
                (angular > 0).mean(axis=1),
                (x_projection > 0).mean(axis=1),
                (y_projection > 0).mean(axis=1),
                y_projection.max(axis=1),
                x_projection.max(axis=1),
            ),
            axis=1,
        )
        scalar = np.stack(
            (
                defect.sum(axis=1) / valid_count,
                radial_mean,
                radial_std,
                x_mean,
                y_mean,
                x_std,
                y_std,
                xy_cov,
            ),
            axis=1,
        )
        descriptor_parts.append(
            np.concatenate(
                (scalar, radial, angular, x_projection, y_projection, coverage),
                axis=1,
            ).astype(np.float32)
        )
    descriptors = np.concatenate(descriptor_parts)
    if descriptors.shape[1] != len(names):
        raise RuntimeError(
            f"Descriptor width mismatch: {descriptors.shape[1]} vs {len(names)}"
        )
    return descriptors, names


def robust_standardizer(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0).astype(np.float32)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    scale = ((q75 - q25) / 1.349).astype(np.float32)
    standard = values.std(axis=0).astype(np.float32)
    scale = np.where(scale >= 1e-4, scale, standard)
    scale = np.maximum(scale, 1e-4).astype(np.float32)
    return center, scale


def label_codes(targets: np.ndarray) -> np.ndarray:
    return np.asarray(
        ["".join(str(int(value)) for value in row) for row in targets],
        dtype=object,
    )


def fit_topology_prototypes(
    descriptors: np.ndarray,
    targets: np.ndarray,
    minimum_samples: int,
    variance_shrinkage: float,
    variance_floor: float,
) -> List[Dict[str, object]]:
    codes = label_codes(targets)
    models: List[Dict[str, object]] = []
    for class_index in range(targets.shape[1]):
        positive = targets[:, class_index].astype(bool)
        class_values = descriptors[positive]
        if len(class_values) == 0:
            raise ValueError(f"No training positive for local class {class_index}")
        class_variance = np.maximum(
            class_values.var(axis=0).astype(np.float32), variance_floor
        )
        centers: List[np.ndarray] = []
        variances: List[np.ndarray] = []
        prototype_codes: List[str] = []
        for code in sorted(set(codes[positive].tolist())):
            member = positive & (codes == code)
            values = descriptors[member]
            if len(values) < minimum_samples:
                continue
            local_variance = values.var(axis=0).astype(np.float32)
            variance = (
                (1.0 - variance_shrinkage) * local_variance
                + variance_shrinkage * class_variance
            )
            centers.append(values.mean(axis=0).astype(np.float32))
            variances.append(np.maximum(variance, variance_floor))
            prototype_codes.append(str(code))
        if not centers:
            centers.append(class_values.mean(axis=0).astype(np.float32))
            variances.append(class_variance)
            prototype_codes.append("ALL")
        models.append(
            {
                "centers": np.stack(centers),
                "variances": np.stack(variances),
                "composition_codes": prototype_codes,
            }
        )
    return models


def topology_distances(
    descriptors: np.ndarray,
    models: Sequence[Dict[str, object]],
) -> np.ndarray:
    output = np.empty((len(descriptors), len(models)), dtype=np.float32)
    for class_index, model in enumerate(models):
        centers = np.asarray(model["centers"], dtype=np.float32)
        variances = np.asarray(model["variances"], dtype=np.float32)
        difference = descriptors[:, None, :] - centers[None, :, :]
        distance = np.mean(
            difference * difference / variances[None, :, :], axis=-1
        )
        output[:, class_index] = distance.min(axis=1)
    return output


def calibrate_topology_thresholds(
    distances: np.ndarray,
    v2_predictions: np.ndarray,
    targets: np.ndarray,
    retention: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    thresholds: List[float] = []
    diagnostics: List[Dict[str, float]] = []
    for class_index in range(targets.shape[1]):
        positive = targets[:, class_index].astype(bool)
        pool = positive & v2_predictions[:, class_index]
        if not pool.any():
            raise ValueError(
                f"No true-positive v2 validation candidate for local class {class_index}"
            )
        threshold = float(np.quantile(distances[pool, class_index], retention))
        topology_accept = distances[:, class_index] <= threshold
        combined = v2_predictions[:, class_index] & topology_accept
        tp = int(np.sum(combined & positive))
        fp = int(np.sum(combined & ~positive))
        thresholds.append(threshold)
        diagnostics.append(
            {
                "threshold": threshold,
                "v2_true_positive_candidates": int(pool.sum()),
                "combined_recall": tp / max(int(positive.sum()), 1),
                "combined_false_positive_rate": fp / max(int((~positive).sum()), 1),
            }
        )
    return np.asarray(thresholds, dtype=np.float32), diagnostics


def load_v2_model(
    checkpoint: Dict[str, object], device: torch.device
) -> SpatialMultiPrototypeVerifier:
    config = checkpoint["model_configuration"]
    model = SpatialMultiPrototypeVerifier(
        class_count=int(config["class_count"]),
        block2_shape=config["block2_shape"],
        block3_shape=config["block3_shape"],
        embedding_dim=int(config["embedding_dim"]),
        prototype_count=int(config["prototype_count"]),
        prototype_temperature=float(config.get("prototype_temperature", 0.10)),
        logit_scale=float(config["logit_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.descriptor_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if not 0.5 < args.topology_positive_retention < 1.0:
        raise ValueError("--topology-positive-retention must be in (0.5,1.0)")
    if not 0.0 <= args.variance_shrinkage <= 1.0:
        raise ValueError("--variance-shrinkage must be in [0,1]")
    if args.variance_floor <= 0.0 or args.minimum_prototype_samples <= 0:
        raise ValueError("Variance floor and minimum prototype samples must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [i for i in range(len(FULL_CLASSES)) if i != unknown_index]
    class_names = [FULL_CLASSES[i] for i in active_indices]
    v2_path = resolve_v2_checkpoint(args.v2_checkpoint, args.unknown_class)
    v2_checkpoint = torch_load(v2_path, map_location="cpu")
    if v2_checkpoint.get("unknown_class") != args.unknown_class:
        raise ValueError(
            f"RCR-v2 holds out {v2_checkpoint.get('unknown_class')}, "
            f"requested {args.unknown_class}"
        )
    stored_active = [int(value) for value in v2_checkpoint["active_known_indices"]]
    if stored_active != active_indices:
        raise ValueError(f"Active class mismatch: {stored_active} vs {active_indices}")
    base_path = Path(
        args.base_checkpoint or v2_checkpoint["base_checkpoint"]
    ).resolve()
    output_dir = Path(args.output_dir or v2_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"unknown={args.unknown_class}; known classes={class_names}")
    print(f"loading RCR-v2: {v2_path}")
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
    base_model, base_active, _ = load_frozen_rdp(
        base_path, args.unknown_class, device
    )
    if base_active != active_indices:
        raise ValueError(f"Frozen RDP active class mismatch: {base_active}")
    v2_model = load_v2_model(v2_checkpoint, device)
    label_threshold = float(v2_checkpoint["label_threshold"])
    distance_weight = float(v2_checkpoint["distance_weight"])
    support_thresholds = v2_checkpoint["support_thresholds"]
    if isinstance(support_thresholds, torch.Tensor):
        support_thresholds = support_thresholds.cpu().numpy()
    support_thresholds = np.asarray(support_thresholds, dtype=np.float32)

    valid_threshold, normal_value, defect_value = data_values(x)
    print("extracting known-train raw topology descriptors")
    train_descriptors, descriptor_names = topology_descriptors(
        x,
        split["known_train"],
        valid_threshold,
        normal_value,
        defect_value,
        args.radial_bins,
        args.angular_bins,
        args.projection_bins,
        args.descriptor_batch_size,
    )
    train_targets = (
        y[torch.as_tensor(split["known_train"], dtype=torch.long)][:, active_indices]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    standard_center, standard_scale = robust_standardizer(train_descriptors)
    train_standard = (train_descriptors - standard_center) / standard_scale
    topology_models = fit_topology_prototypes(
        train_standard,
        train_targets,
        args.minimum_prototype_samples,
        args.variance_shrinkage,
        args.variance_floor,
    )
    del train_descriptors, train_standard

    cached: Dict[str, Dict[str, np.ndarray]] = {}
    for name in ("known_val", "known_test"):
        print(f"scoring {name}")
        block2, block3, probabilities, targets = extract_multiscale_features(
            base_model,
            x,
            y,
            split[name],
            active_indices,
            args.batch_size,
            device,
            name,
        )
        support, _, _ = collect_scores(
            v2_model,
            block2,
            block3,
            args.batch_size,
            device,
            distance_weight,
        )
        v2_predictions = predict(
            probabilities.numpy(),
            support,
            label_threshold,
            support_thresholds,
        )
        descriptors, _ = topology_descriptors(
            x,
            split[name],
            valid_threshold,
            normal_value,
            defect_value,
            args.radial_bins,
            args.angular_bins,
            args.projection_bins,
            args.descriptor_batch_size,
        )
        topology_distance = topology_distances(
            (descriptors - standard_center) / standard_scale,
            topology_models,
        )
        cached[name] = {
            "v2_predictions": v2_predictions,
            "targets": targets.numpy(),
            "topology_distances": topology_distance,
        }
        del block2, block3, probabilities, descriptors

    topology_thresholds, calibration = calibrate_topology_thresholds(
        cached["known_val"]["topology_distances"],
        cached["known_val"]["v2_predictions"],
        cached["known_val"]["targets"],
        args.topology_positive_retention,
    )

    print("scoring unknown_test")
    block2, block3, probabilities, targets = extract_multiscale_features(
        base_model,
        x,
        y,
        split["unknown_test"],
        active_indices,
        args.batch_size,
        device,
        "unknown_test",
    )
    support, _, _ = collect_scores(
        v2_model,
        block2,
        block3,
        args.batch_size,
        device,
        distance_weight,
    )
    unknown_v2 = predict(
        probabilities.numpy(), support, label_threshold, support_thresholds
    )
    unknown_descriptors, _ = topology_descriptors(
        x,
        split["unknown_test"],
        valid_threshold,
        normal_value,
        defect_value,
        args.radial_bins,
        args.angular_bins,
        args.projection_bins,
        args.descriptor_batch_size,
    )
    unknown_distance = topology_distances(
        (unknown_descriptors - standard_center) / standard_scale,
        topology_models,
    )
    unknown_targets = targets.numpy()

    known_v2 = cached["known_test"]["v2_predictions"]
    known_distance = cached["known_test"]["topology_distances"]
    known_targets = cached["known_test"]["targets"]
    known_v3 = known_v2 & (
        known_distance <= topology_thresholds.reshape(1, -1)
    )
    unknown_v3 = unknown_v2 & (
        unknown_distance <= topology_thresholds.reshape(1, -1)
    )

    results = {
        "rcr_v2": {
            "known_test_components": multilabel_metrics(
                known_v2, known_targets, class_names
            ),
            "unknown_test_known_projection": multilabel_metrics(
                unknown_v2, unknown_targets, class_names
            ),
        },
        "rcr_v2_plus_topology": {
            "known_test_components": multilabel_metrics(
                known_v3, known_targets, class_names
            ),
            "unknown_test_known_projection": multilabel_metrics(
                unknown_v3, unknown_targets, class_names
            ),
        },
    }
    metrics = {
        "configuration": {
            "method": "rcr_v3_known_composition_topology_gate",
            "unknown_class": args.unknown_class,
            "known_classes": class_names,
            "data": str(Path(args.data).resolve()),
            "v2_checkpoint": str(v2_path),
            "base_checkpoint": str(base_path),
            "heldout_data_used_for_fitting_selection_or_calibration": False,
            "topology_positive_retention": args.topology_positive_retention,
            "minimum_prototype_samples": args.minimum_prototype_samples,
            "variance_shrinkage": args.variance_shrinkage,
            "variance_floor": args.variance_floor,
            "radial_bins": args.radial_bins,
            "angular_bins": args.angular_bins,
            "projection_bins": args.projection_bins,
            "seed": args.seed,
        },
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "descriptor_order": descriptor_names,
        "prototype_counts": {
            name: int(len(topology_models[index]["centers"]))
            for index, name in enumerate(class_names)
        },
        "known_validation_calibration": {
            name: calibration[index] for index, name in enumerate(class_names)
        },
        "results": results,
        "notes": {
            "unknown_gate": "Dinomaly remains unchanged",
            "topology_role": "reject geometrically unsupported RCR-v2 known-label candidates",
            "unknown_test_role": "final diagnostic evaluation only",
        },
    }
    topology_checkpoint = {
        "method": "known_composition_topology_gate",
        "unknown_class": args.unknown_class,
        "known_classes": class_names,
        "active_known_indices": active_indices,
        "v2_checkpoint": str(v2_path),
        "descriptor_names": descriptor_names,
        "standard_center": torch.from_numpy(standard_center),
        "standard_scale": torch.from_numpy(standard_scale),
        "topology_models": [
            {
                "centers": torch.from_numpy(np.asarray(model["centers"])),
                "variances": torch.from_numpy(np.asarray(model["variances"])),
                "composition_codes": model["composition_codes"],
            }
            for model in topology_models
        ],
        "topology_thresholds": torch.from_numpy(topology_thresholds),
        "configuration": metrics["configuration"],
    }
    torch.save(topology_checkpoint, output_dir / "best_topology_gate.pt")
    with (output_dir / "topology_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\ncomparison")
    for method, method_results in results.items():
        known = method_results["known_test_components"]
        unknown = method_results["unknown_test_known_projection"]
        print(
            f"{method}: known_EM={known['exact_match']:.6f} "
            f"unknown_EM={unknown['exact_match']:.6f} "
            f"unknown_F1={unknown['micro_f1']:.6f} "
            f"FP/sample={unknown['mean_false_positives_per_sample']:.6f} "
            f"pure_false={unknown['pure_unknown_false_known_rate']}"
        )
    print("saved:", output_dir / "best_topology_gate.pt")
    print("saved:", output_dir / "topology_metrics.json")


if __name__ == "__main__":
    main()
