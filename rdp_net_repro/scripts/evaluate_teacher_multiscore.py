#!/usr/bin/env python3
"""Multi-response OOD evaluation directly from the trained FM teacher.

Place this file beside ``train_component_conditioned_teacher.py`` in scripts/.
It trains nothing, writes nothing, and prints one JSON.  Known validation data
is used only to fit/calibrate reference distributions.  AUROC is evaluated on
disjoint known_test and held-out-atomic-defect unknown_test samples.

The reconstruction branch intentionally uses only the image condition.  This
is a Teacher-level WaferDiSC-style baseline, not the final DE-CM Student score.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from train_component_conditioned_teacher import (
    FULL_CLASSES,
    FlowTeacherUNet,
    ImageConditionAdapter,
    forward_rdp_features,
    load_arrays,
    load_frozen_rdp,
    make_loader,
    make_open_set_split,
    normalize_teacher_input,
    set_seed,
    torch_load,
)


MORPHOLOGY_NAMES: Tuple[str, ...] = (
    "defect_ratio",
    "component_count",
    "max_component_area_ratio",
    "max_aspect_ratio",
    "max_straightness",
    "longest_run_ratio",
    "component_line_response",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher multi-response AUROC")
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="checkpoint/component_waferdisc/s_heldout/best_teacher.pt",
    )
    parser.add_argument("--rdp-checkpoint", default=None)
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="S")
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--calibration-n", type=int, default=1000)
    parser.add_argument("--known-eval-n", type=int, default=1000)
    parser.add_argument("--unknown-eval-n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-t", type=int, default=2)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument("--fusion-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def deterministic_subset(indices: Sequence[int], size: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices)
    if size <= 0 or size >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=size, replace=False)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch = len(values)
    numerator = (values * mask).reshape(batch, -1).sum(dim=1)
    denominator = mask.reshape(batch, -1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def defect_dice_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = (reconstruction > 0.75).float()
    truth = (target > 0.75).float()
    intersection = (predicted * truth).flatten(1).sum(dim=1)
    denominator = predicted.flatten(1).sum(dim=1) + truth.flatten(1).sum(dim=1)
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def longest_true_run(vector: np.ndarray) -> int:
    best = current = 0
    for value in vector.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def longest_run_2d(mask: np.ndarray) -> int:
    height, width = mask.shape
    best = 0
    for row in mask:
        best = max(best, longest_true_run(row))
    for column in mask.T:
        best = max(best, longest_true_run(column))
    for offset in range(-height + 1, width):
        best = max(best, longest_true_run(np.diagonal(mask, offset=offset)))
        best = max(
            best,
            longest_true_run(np.diagonal(np.fliplr(mask), offset=offset)),
        )
    return best


def morphology_descriptors(x: torch.Tensor) -> np.ndarray:
    """Seven WBM morphology descriptors from normalized 0/0.5/1 tensors."""
    arrays = x.detach().cpu().numpy()[:, 0]
    rows = []
    structure = np.ones((3, 3), dtype=np.int8)
    for wafer in arrays:
        valid = wafer > 0.25
        defect = wafer > 0.75
        valid_count = max(1, int(valid.sum()))
        labeled, component_count = ndimage.label(defect, structure=structure)
        max_area_ratio = 0.0
        max_aspect_ratio = 0.0
        max_straightness = 0.0
        max_line_response = 0.0

        for component_id in range(1, component_count + 1):
            coordinates = np.argwhere(labeled == component_id)
            area = len(coordinates)
            if area == 0:
                continue
            area_ratio = area / valid_count
            height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
            width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
            aspect = max(height, width) / max(1, min(height, width))
            if area >= 3:
                centered = coordinates.astype(np.float64) - coordinates.mean(axis=0)
                covariance = centered.T @ centered / max(1, area - 1)
                eigenvalues = np.linalg.eigvalsh(covariance)
                straightness = float(
                    (eigenvalues[-1] - eigenvalues[0])
                    / max(eigenvalues[-1] + eigenvalues[0], 1e-8)
                )
            else:
                straightness = 1.0 if area >= 2 else 0.0
            max_area_ratio = max(max_area_ratio, area_ratio)
            max_aspect_ratio = max(max_aspect_ratio, aspect)
            max_straightness = max(max_straightness, straightness)
            max_line_response = max(
                max_line_response,
                straightness * np.log1p(aspect) * np.sqrt(area_ratio),
            )

        longest_run_ratio = longest_run_2d(defect) / max(defect.shape)
        rows.append(
            [
                float(defect.sum() / valid_count),
                float(component_count),
                float(max_area_ratio),
                float(max_aspect_ratio),
                float(max_straightness),
                float(longest_run_ratio),
                float(max_line_response),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


@torch.inference_mode()
def heun_reconstruct(
    teacher: FlowTeacherUNet,
    state: torch.Tensor,
    condition: torch.Tensor,
    start_time: float,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate the instantaneous FM teacher from start_time to t=1."""
    time_sequence = torch.linspace(
        float(start_time), 1.0, steps + 1, device=state.device, dtype=state.dtype
    )
    current = state
    first_velocity = None
    for step in range(steps):
        t_current = torch.full(
            (len(current),),
            float(time_sequence[step].item()),
            device=current.device,
            dtype=current.dtype,
        )
        dt = time_sequence[step + 1] - time_sequence[step]
        velocity_1 = teacher(current, t_current, t_current, condition).float()
        if first_velocity is None:
            first_velocity = velocity_1
        euler_state = current + dt * velocity_1
        t_next = torch.full(
            (len(current),),
            float(time_sequence[step + 1].item()),
            device=current.device,
            dtype=current.dtype,
        )
        velocity_2 = teacher(euler_state, t_next, t_next, condition).float()
        current = current + 0.5 * dt * (velocity_1 + velocity_2)
    assert first_velocity is not None
    return current.clamp(0.0, 1.0), first_velocity


def to_rdp_scale(x_normalized: torch.Tensor, raw_scale: float) -> torch.Tensor:
    return x_normalized * raw_scale


@torch.inference_mode()
def collect_raw_responses(
    loader,
    teacher: FlowTeacherUNet,
    image_adapter: ImageConditionAdapter,
    rdp_model,
    device: torch.device,
    image_size: int,
    raw_scale: float,
    t_values: np.ndarray,
    ode_steps: int,
    amp_enabled: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    scalar_names = (
        "velocity_mse",
        "velocity_die_mse",
        "reconstruction_pixel_mse",
        "reconstruction_dice_loss",
        "reconstruction_pattern_cosine",
    )
    scalar_parts = {name: [] for name in scalar_names}
    feature_parts = []
    morphology_parts = []
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)

    for x_raw, _labels in loader:
        x_raw = x_raw.to(device, non_blocking=True)
        x = normalize_teacher_input(x_raw, image_size)
        _, input_features = forward_rdp_features(
            rdp_model, x_raw.unsqueeze(1).float()
        )
        condition = image_adapter(input_features)
        normalized_input_features = F.normalize(input_features.float(), dim=1)
        feature_parts.append(normalized_input_features.cpu().numpy())
        # Pattern reconstruction compares equal resolution and equal value scale.
        _, input_pattern_features = forward_rdp_features(
            rdp_model, to_rdp_scale(x, raw_scale)
        )
        morphology_parts.append(morphology_descriptors(x))
        batch = len(x)
        valid_die = (x > 0.25).float()
        score_sum = {
            name: torch.zeros(batch, device=device, dtype=torch.float32)
            for name in scalar_names
        }

        for t_value in t_values:
            noise = torch.randn(
                x.shape, device=device, dtype=x.dtype, generator=generator
            )
            t = torch.full(
                (batch,), float(t_value), device=device, dtype=x.dtype
            )
            t_view = t.view(-1, 1, 1, 1)
            noisy_state = (1.0 - t_view) * noise + t_view * x
            target_velocity = x - noise
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                reconstruction, first_velocity = heun_reconstruct(
                    teacher,
                    noisy_state,
                    condition,
                    float(t_value),
                    ode_steps,
                )
            velocity_error = (first_velocity - target_velocity).square()
            pixel_error = (reconstruction - x).square()
            _, reconstruction_features = forward_rdp_features(
                rdp_model,
                to_rdp_scale(reconstruction, raw_scale),
            )
            pattern_cosine = 1.0 - F.cosine_similarity(
                input_pattern_features.float(), reconstruction_features.float(), dim=1
            )
            score_sum["velocity_mse"] += velocity_error.flatten(1).mean(dim=1)
            score_sum["velocity_die_mse"] += masked_mean(
                velocity_error, valid_die
            )
            score_sum["reconstruction_pixel_mse"] += masked_mean(
                pixel_error, valid_die
            )
            score_sum["reconstruction_dice_loss"] += defect_dice_loss(
                reconstruction, x
            )
            score_sum["reconstruction_pattern_cosine"] += pattern_cosine

        for name in scalar_names:
            scalar_parts[name].append(
                (score_sum[name] / float(len(t_values))).cpu().numpy()
            )

    output = {
        name: np.concatenate(parts) for name, parts in scalar_parts.items()
    }
    output["rdp_features"] = np.concatenate(feature_parts, axis=0)
    output["morphology_descriptors"] = np.concatenate(morphology_parts, axis=0)
    return output


def fit_reference(calibration: Dict[str, np.ndarray]) -> Dict[str, object]:
    scalar_names = [
        name
        for name in calibration
        if name not in ("rdp_features", "morphology_descriptors")
    ]
    scalar_reference = {}
    for name in scalar_names:
        values = calibration[name]
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        scalar_reference[name] = {
            "median": float(median),
            "iqr": float(max(q75 - q25, 1e-8)),
        }

    features = calibration["rdp_features"]
    feature_mean = features.mean(axis=0)
    feature_variance = features.var(axis=0) + 1e-5
    morphology = calibration["morphology_descriptors"]
    morphology_median = np.median(morphology, axis=0)
    morphology_q25 = np.quantile(morphology, 0.25, axis=0)
    morphology_q75 = np.quantile(morphology, 0.75, axis=0)
    morphology_iqr = np.maximum(morphology_q75 - morphology_q25, 1e-6)
    return {
        "scalar": scalar_reference,
        "feature_mean": feature_mean,
        "feature_variance": feature_variance,
        "morphology_median": morphology_median,
        "morphology_iqr": morphology_iqr,
    }


def materialize_scores(
    raw: Dict[str, np.ndarray], reference: Dict[str, object]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    raw_scores = {
        name: values
        for name, values in raw.items()
        if name not in ("rdp_features", "morphology_descriptors")
    }
    features = raw["rdp_features"]
    feature_mahalanobis = np.mean(
        (features - reference["feature_mean"]) ** 2
        / reference["feature_variance"],
        axis=1,
    )
    morphology_z = np.abs(
        (raw["morphology_descriptors"] - reference["morphology_median"])
        / reference["morphology_iqr"]
    )
    # Limit one extreme descriptor from dominating the complete morphology score.
    morphology_score = np.mean(np.minimum(morphology_z, 20.0), axis=1)
    raw_scores["rdp_feature_diag_mahalanobis"] = feature_mahalanobis
    raw_scores["morphology_robust_deviation"] = morphology_score

    calibrated = {}
    for name, values in raw_scores.items():
        if name in reference["scalar"]:
            median = reference["scalar"][name]["median"]
            iqr = reference["scalar"][name]["iqr"]
        else:
            # Feature/morphology scores are themselves defined from calibration;
            # use their calibration-score distribution for response-level scaling.
            if name == "rdp_feature_diag_mahalanobis":
                calibration_values = np.mean(
                    (reference["calibration_features"] - reference["feature_mean"]) ** 2
                    / reference["feature_variance"],
                    axis=1,
                )
            else:
                calibration_morphology_z = np.abs(
                    (
                        reference["calibration_morphology"]
                        - reference["morphology_median"]
                    )
                    / reference["morphology_iqr"]
                )
                calibration_values = np.mean(
                    np.minimum(calibration_morphology_z, 20.0), axis=1
                )
            q25, median, q75 = np.quantile(
                calibration_values, [0.25, 0.50, 0.75]
            )
            iqr = max(q75 - q25, 1e-8)
        # One-sided response: only larger-than-known-median evidence is abnormal.
        calibrated[name] = np.maximum((values - median) / iqr, 0.0)
    return raw_scores, calibrated


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    indices = np.flatnonzero(tpr >= 0.95)
    return float(fpr[indices[0]]) if len(indices) else 1.0


def evaluate(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_reject_rate: float,
) -> Dict[str, float]:
    labels = np.concatenate(
        [
            np.zeros(len(known_scores), dtype=np.int64),
            np.ones(len(unknown_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate([known_scores, unknown_scores])
    threshold = float(np.quantile(known_scores, 1.0 - known_reject_rate))
    known_reject = float(np.mean(known_scores > threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr_at_tpr95(labels, scores),
        "threshold": threshold,
        "known_reject_rate": known_reject,
        "unknown_recall_at_known_reject_rate": unknown_recall,
        "balanced_accuracy_at_threshold": float(
            0.5 * ((1.0 - known_reject) + unknown_recall)
        ),
        "known_mean": float(np.mean(known_scores)),
        "unknown_mean": float(np.mean(unknown_scores)),
    }


def softmax_fusion(matrix: np.ndarray, beta: float) -> np.ndarray:
    logits = beta * matrix
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.sum(weights * matrix, axis=1)


def main() -> None:
    args = parse_args()
    if args.num_t < 1 or args.ode_steps < 1:
        raise ValueError("--num-t and --ode-steps must be positive")
    if not 0.0 < args.t_min <= args.t_max < 1.0:
        raise ValueError("Require 0 < t-min <= t-max < 1")
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    checkpoint = torch_load(args.teacher_checkpoint, map_location="cpu")
    configuration = checkpoint["configuration"]
    if configuration["unknown_class"] != args.unknown_class:
        raise ValueError(
            f"Teacher holds out {configuration['unknown_class']}, not {args.unknown_class}"
        )
    unknown_idx = FULL_CLASSES.index(args.unknown_class)
    image_size = int(configuration["image_size"])
    condition_dim = int(configuration["condition_dim"])
    rdp_checkpoint = args.rdp_checkpoint or configuration["rdp_checkpoint"]

    teacher = FlowTeacherUNet(condition_dim).to(device).eval()
    image_adapter = ImageConditionAdapter(512, condition_dim).to(device).eval()
    teacher.load_state_dict(checkpoint["teacher_state_dict"], strict=True)
    image_adapter.load_state_dict(
        checkpoint["image_adapter_state_dict"], strict=True
    )
    rdp_model = load_frozen_rdp(rdp_checkpoint, args.unknown_class, device)

    x, y = load_arrays(args.data)
    raw_scale = 2.0 if float(x.max().item()) > 1.0 else 1.0
    split = make_open_set_split(
        y,
        unknown_idx,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    calibration_indices = deterministic_subset(
        split["known_val"], args.calibration_n, args.seed + 1
    )
    known_indices = deterministic_subset(
        split["known_test"], args.known_eval_n, args.seed + 2
    )
    unknown_indices = deterministic_subset(
        split["unknown_test"], args.unknown_eval_n, args.seed + 3
    )
    loaders = {
        "calibration": make_loader(
            x,
            y,
            calibration_indices,
            args.batch_size,
            False,
            args.num_workers,
            device.type == "cuda",
        ),
        "known_test": make_loader(
            x,
            y,
            known_indices,
            args.batch_size,
            False,
            args.num_workers,
            device.type == "cuda",
        ),
        "unknown_test": make_loader(
            x,
            y,
            unknown_indices,
            args.batch_size,
            False,
            args.num_workers,
            device.type == "cuda",
        ),
    }
    t_values = np.linspace(args.t_min, args.t_max, args.num_t)
    collected = {}
    # The same generator seed gives all equally sized sets the same noise schedule.
    for name, loader in loaders.items():
        print(f"scoring {name}", flush=True)
        collected[name] = collect_raw_responses(
            loader,
            teacher,
            image_adapter,
            rdp_model,
            device,
            image_size,
            raw_scale,
            t_values,
            args.ode_steps,
            amp_enabled,
            seed=args.seed + 30000,
        )

    reference = fit_reference(collected["calibration"])
    reference["calibration_features"] = collected["calibration"]["rdp_features"]
    reference["calibration_morphology"] = collected["calibration"][
        "morphology_descriptors"
    ]
    raw_known, calibrated_known = materialize_scores(
        collected["known_test"], reference
    )
    raw_unknown, calibrated_unknown = materialize_scores(
        collected["unknown_test"], reference
    )

    individual_metrics = {
        name: evaluate(
            raw_known[name], raw_unknown[name], args.known_reject_rate
        )
        for name in raw_known
    }
    fusion_components = (
        "velocity_die_mse",
        "reconstruction_pixel_mse",
        "reconstruction_pattern_cosine",
        "rdp_feature_diag_mahalanobis",
        "morphology_robust_deviation",
    )
    known_matrix = np.column_stack(
        [calibrated_known[name] for name in fusion_components]
    )
    unknown_matrix = np.column_stack(
        [calibrated_unknown[name] for name in fusion_components]
    )
    fusion_scores = {
        "equal_fusion": (
            known_matrix.mean(axis=1),
            unknown_matrix.mean(axis=1),
        ),
        "evidence_softmax_fusion": (
            softmax_fusion(known_matrix, args.fusion_beta),
            softmax_fusion(unknown_matrix, args.fusion_beta),
        ),
    }
    fusion_metrics = {
        name: evaluate(known_score, unknown_score, args.known_reject_rate)
        for name, (known_score, unknown_score) in fusion_scores.items()
    }

    result = {
        "configuration": {
            "unknown_class": args.unknown_class,
            "teacher_checkpoint": args.teacher_checkpoint,
            "rdp_checkpoint": rdp_checkpoint,
            "device": str(device),
            "calibration_samples": len(calibration_indices),
            "known_samples": len(known_indices),
            "unknown_samples": len(unknown_indices),
            "num_t": args.num_t,
            "ode_steps": args.ode_steps,
            "t_values": t_values.tolist(),
            "raw_scale_for_rdp": raw_scale,
            "fusion_components": list(fusion_components),
            "fusion_beta": args.fusion_beta,
        },
        "individual_scores": individual_metrics,
        "known_only_calibrated_fusion": fusion_metrics,
        "morphology_descriptor_order": list(MORPHOLOGY_NAMES),
        "notes": {
            "reconstruction": "iterative Heun integration of the image-condition FM teacher",
            "fusion": "reference statistics use known validation only; no S labels tune weights",
            "limitation": "Teacher-level diagnostic; exact Student trajectory response requires DE-CM Student",
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
