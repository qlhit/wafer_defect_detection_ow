#!/usr/bin/env python3
"""Evaluate whether the trained dual-condition teacher contains OOD signal.

This is a diagnostic gate before DE-CM student training.  It uses known_test
and held-out-atomic-defect unknown_test, trains nothing, writes nothing, and
prints one JSON.  Higher scores are always defined as more unknown-like.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from train_component_conditioned_teacher import (
    FULL_CLASSES,
    CompositionEncoder,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-level unknown-signal AUROC")
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
    parser.add_argument("--known-eval-n", type=int, default=1000)
    parser.add_argument("--unknown-eval-n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-t", type=int, default=4)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--t-min", type=float, default=0.2)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
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


@torch.inference_mode()
def score_loader(
    loader,
    teacher: FlowTeacherUNet,
    image_adapter: ImageConditionAdapter,
    composition_encoder: CompositionEncoder,
    rdp_model,
    active_mask: torch.Tensor,
    device: torch.device,
    image_size: int,
    t_values: np.ndarray,
    noise_repeats: int,
    amp_enabled: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    names = (
        "image_velocity_mse",
        "composition_velocity_mse",
        "composition_minus_image",
        "dual_velocity_mse",
        "dual_velocity_relative",
        "dual_velocity_die_mse",
        "dual_velocity_die_relative",
    )
    outputs = {name: [] for name in names}
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)

    for x_raw, _labels in loader:
        x_raw = x_raw.to(device, non_blocking=True)
        x = normalize_teacher_input(x_raw, image_size)
        logits, rdp_features = forward_rdp_features(
            rdp_model, x_raw.unsqueeze(1).float()
        )
        probabilities = torch.sigmoid(logits) * active_mask
        image_condition = image_adapter(rdp_features)
        composition_condition = composition_encoder(probabilities, active_mask)
        batch = len(x)
        score_sum = {
            name: torch.zeros(batch, device=device, dtype=torch.float32)
            for name in names
        }
        valid_die_mask = (x > 0.25).float()

        for _ in range(noise_repeats):
            noise = torch.randn(
                x.shape, device=device, dtype=x.dtype, generator=generator
            )
            target_velocity = x - noise
            for t_value in t_values:
                t = torch.full(
                    (batch,), float(t_value), device=device, dtype=x.dtype
                )
                t_view = t.view(-1, 1, 1, 1)
                noisy_state = (1.0 - t_view) * noise + t_view * x
                with torch.amp.autocast(
                    device_type=device.type, enabled=amp_enabled
                ):
                    image_velocity = teacher(
                        noisy_state, t, t, image_condition
                    ).float()
                    composition_velocity = teacher(
                        noisy_state, t, t, composition_condition
                    ).float()

                image_squared_error = (image_velocity - target_velocity).square()
                composition_squared_error = (
                    composition_velocity - target_velocity
                ).square()
                dual_squared_error = (
                    image_velocity - composition_velocity
                ).square()
                image_error = image_squared_error.flatten(1).mean(dim=1)
                composition_error = composition_squared_error.flatten(1).mean(dim=1)
                dual_error = dual_squared_error.flatten(1).mean(dim=1)
                base = 0.5 * (image_error + composition_error)
                image_die = masked_mean(image_squared_error, valid_die_mask)
                composition_die = masked_mean(
                    composition_squared_error, valid_die_mask
                )
                dual_die = masked_mean(dual_squared_error, valid_die_mask)
                die_base = 0.5 * (image_die + composition_die)

                score_sum["image_velocity_mse"] += image_error
                score_sum["composition_velocity_mse"] += composition_error
                score_sum["composition_minus_image"] += (
                    composition_error - image_error
                )
                score_sum["dual_velocity_mse"] += dual_error
                score_sum["dual_velocity_relative"] += dual_error / base.clamp_min(1e-8)
                score_sum["dual_velocity_die_mse"] += dual_die
                score_sum["dual_velocity_die_relative"] += (
                    dual_die / die_base.clamp_min(1e-8)
                )

        divisor = float(len(t_values) * noise_repeats)
        for name in names:
            outputs[name].append((score_sum[name] / divisor).cpu().numpy())

    return {name: np.concatenate(parts) for name, parts in outputs.items()}


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    indices = np.flatnonzero(tpr >= 0.95)
    return float(fpr[indices[0]]) if len(indices) else 1.0


def evaluate_metric(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_reject_rate: float,
) -> Dict[str, float]:
    labels = np.concatenate(
        [np.zeros(len(known_scores), dtype=np.int64), np.ones(len(unknown_scores), dtype=np.int64)]
    )
    scores = np.concatenate([known_scores, unknown_scores])
    threshold = float(np.quantile(known_scores, 1.0 - known_reject_rate))
    known_reject = float(np.mean(known_scores > threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr_at_tpr95(labels, scores),
        "known_calibrated_threshold": threshold,
        "known_reject_rate": known_reject,
        "unknown_recall_at_known_reject_rate": unknown_recall,
        "balanced_accuracy_at_threshold": float(
            0.5 * ((1.0 - known_reject) + unknown_recall)
        ),
        "known_mean": float(np.mean(known_scores)),
        "unknown_mean": float(np.mean(unknown_scores)),
        "known_q95": float(np.quantile(known_scores, 0.95)),
        "unknown_q95": float(np.quantile(unknown_scores, 0.95)),
    }


def main() -> None:
    args = parse_args()
    if args.num_t < 1 or args.noise_repeats < 1:
        raise ValueError("--num-t and --noise-repeats must be positive")
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
    component_embed_dim = int(configuration["component_embed_dim"])
    active_mask = torch.tensor(
        configuration["active_mask"], device=device, dtype=torch.float32
    ).view(1, -1)
    rdp_checkpoint = args.rdp_checkpoint or configuration["rdp_checkpoint"]

    teacher = FlowTeacherUNet(condition_dim).to(device).eval()
    image_adapter = ImageConditionAdapter(512, condition_dim).to(device).eval()
    composition_encoder = CompositionEncoder(
        len(FULL_CLASSES), component_embed_dim, condition_dim
    ).to(device).eval()
    teacher.load_state_dict(checkpoint["teacher_state_dict"], strict=True)
    image_adapter.load_state_dict(
        checkpoint["image_adapter_state_dict"], strict=True
    )
    composition_encoder.load_state_dict(
        checkpoint["composition_encoder_state_dict"], strict=True
    )
    rdp_model = load_frozen_rdp(rdp_checkpoint, args.unknown_class, device)

    x, y = load_arrays(args.data)
    split = make_open_set_split(
        y,
        unknown_idx,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    known_indices = deterministic_subset(
        split["known_test"], args.known_eval_n, args.seed + 1
    )
    unknown_indices = deterministic_subset(
        split["unknown_test"], args.unknown_eval_n, args.seed + 2
    )
    known_loader = make_loader(
        x, y, known_indices, args.batch_size, False, args.num_workers, device.type == "cuda"
    )
    unknown_loader = make_loader(
        x, y, unknown_indices, args.batch_size, False, args.num_workers, device.type == "cuda"
    )
    t_values = np.linspace(args.t_min, args.t_max, args.num_t)

    print("scoring known test set", flush=True)
    known_scores = score_loader(
        known_loader,
        teacher,
        image_adapter,
        composition_encoder,
        rdp_model,
        active_mask,
        device,
        image_size,
        t_values,
        args.noise_repeats,
        amp_enabled,
        args.seed + 30000,
    )
    print("scoring unknown test set", flush=True)
    unknown_scores = score_loader(
        unknown_loader,
        teacher,
        image_adapter,
        composition_encoder,
        rdp_model,
        active_mask,
        device,
        image_size,
        t_values,
        args.noise_repeats,
        amp_enabled,
        args.seed + 40000,
    )

    metrics = {
        name: evaluate_metric(
            known_scores[name], unknown_scores[name], args.known_reject_rate
        )
        for name in known_scores
    }
    result = {
        "configuration": {
            "unknown_class": args.unknown_class,
            "teacher_checkpoint": args.teacher_checkpoint,
            "rdp_checkpoint": rdp_checkpoint,
            "device": str(device),
            "known_samples": len(known_indices),
            "unknown_samples": len(unknown_indices),
            "num_t": args.num_t,
            "noise_repeats": args.noise_repeats,
            "t_range": [args.t_min, args.t_max],
            "known_reject_rate_target": args.known_reject_rate,
        },
        "teacher_level_unknown_detection": metrics,
        "notes": {
            "purpose": "diagnostic only; do not select the final paper score using target-S labels",
            "auroc_interpretation": {
                "near_random": "<=0.60",
                "weak_signal": "0.60-0.70",
                "useful_signal": "0.70-0.80",
                "strong_signal": ">=0.80",
            },
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
