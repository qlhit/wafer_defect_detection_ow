#!/usr/bin/env python3
"""Train a polar-coordinate RDP verifier for known-component candidates.

The existing Cartesian RDP + RCR-v2 remains unchanged and proposes known
components.  A copied RDP is fine-tuned on polar-coordinate known wafers and
then verifies those candidates.  In polar coordinates a full edge ring is a
long band over nearly every angle, while an edge-local pattern occupies only a
short angular interval.

Protocol:
  * held-out-class wafers are excluded from training, validation, checkpoint
    selection, and threshold calibration;
  * the polar model is initialized from the same known-only/robust RDP used by
    RCR-v2;
  * known-validation candidates calibrate per-class polar thresholds;
  * held-out data is loaded only for final diagnostic evaluation.

Only two files are written:
  * best_polar_rdp.pt
  * polar_rdp_metrics.json

Place this file in ``rdp_net_repro/scripts/`` and run from the repository root.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_eval_contamination_robust_rdp import (
    AsymmetricLoss,
    forward_features,
    normalized_asl,
)
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
        description="Train/evaluate a polar-coordinate verifier for RDP + RCR-v2"
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
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Default: base_checkpoint recorded in the RCR-v2 checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--angular-roll",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Random circular angular shift during polar-model training",
    )
    parser.add_argument(
        "--positive-retention",
        type=float,
        default=0.995,
        help="Initial retention of true RCR-v2 validation candidates",
    )
    parser.add_argument(
        "--max-known-val-em-drop",
        type=float,
        default=0.005,
        help="Relax thresholds until known-validation EM drops by at most this",
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: checkpoint/rdp_polar_verifier/<class>_heldout",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x[index_tensor], y[index_tensor]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def polar_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return grid_sample coordinates for [radius, angle] output axes."""
    radius = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(
        height, 1
    )
    angle = torch.linspace(
        -math.pi,
        math.pi,
        width + 1,
        device=device,
        dtype=dtype,
    )[:-1].view(1, width)
    source_x = radius * torch.cos(angle)
    source_y = radius * torch.sin(angle)
    return torch.stack((source_x, source_y), dim=-1).unsqueeze(0)


def to_polar(wafer: torch.Tensor) -> torch.Tensor:
    height, width = int(wafer.shape[-2]), int(wafer.shape[-1])
    grid = polar_grid(height, width, wafer.device, wafer.dtype).expand(
        len(wafer), -1, -1, -1
    )
    return F.grid_sample(
        wafer,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )


@torch.inference_mode()
def collect_polar_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    active_indices: Sequence[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probability_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    criterion = AsymmetricLoss()
    for wafer, labels in loader:
        wafer = wafer.unsqueeze(1).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = forward_features(model, to_polar(wafer))
        active_logits = logits[:, active_indices]
        active_targets = labels[:, active_indices]
        loss = normalized_asl(criterion, active_logits, active_targets)
        probability_parts.append(torch.sigmoid(active_logits).cpu().numpy())
        target_parts.append(active_targets.cpu().numpy())
        loss_sum += float(loss.item()) * len(wafer)
        count += len(wafer)
    return (
        np.concatenate(probability_parts).astype(np.float32),
        np.concatenate(target_parts).astype(np.float32),
        loss_sum / max(count, 1),
    )


def polar_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    active_indices: Sequence[int],
    class_names: Sequence[str],
    device: torch.device,
    label_threshold: float,
) -> Dict[str, float]:
    probabilities, targets, loss = collect_polar_probabilities(
        model, loader, active_indices, device
    )
    predictions = probabilities >= label_threshold
    metrics = multilabel_metrics(predictions, targets, class_names)
    return {
        "loss": float(loss),
        "exact_match": float(metrics["exact_match"]),
        "micro_f1": float(metrics["micro_f1"]),
    }


def train_polar_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    active_indices: Sequence[int],
    class_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float], int]:
    criterion = AsymmetricLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
    )

    initial = polar_validation(
        model,
        val_loader,
        active_indices,
        class_names,
        device,
        args.label_threshold,
    )
    best_key = (initial["exact_match"], initial["micro_f1"], -initial["loss"])
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = initial
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "intermediate_training_checkpoint": True,
        },
        checkpoint_path,
    )
    print(
        f"epoch 000 val_loss={initial['loss']:.6f} "
        f"val_EM={initial['exact_match']:.6f} "
        f"val_F1={initial['micro_f1']:.6f} *",
        flush=True,
    )

    patience_count = 0
    epochs_ran = 0
    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        model.train()
        loss_sum = 0.0
        count = 0
        for wafer, labels in train_loader:
            wafer = wafer.unsqueeze(1).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            polar = to_polar(wafer)
            if args.angular_roll:
                shift = int(
                    torch.randint(
                        0, polar.shape[-1], (1,), device=device
                    ).item()
                )
                polar = torch.roll(polar, shifts=shift, dims=-1)
            logits, _ = forward_features(model, polar)
            loss = normalized_asl(
                criterion,
                logits[:, active_indices],
                labels[:, active_indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(wafer)
            count += len(wafer)
        scheduler.step()

        validation = polar_validation(
            model,
            val_loader,
            active_indices,
            class_names,
            device,
            args.label_threshold,
        )
        key = (
            validation["exact_match"],
            validation["micro_f1"],
            -validation["loss"],
        )
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_validation = validation
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "intermediate_training_checkpoint": True,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
        print(
            f"epoch {epoch:03d} train_loss={loss_sum / max(count, 1):.6f} "
            f"val_loss={validation['loss']:.6f} "
            f"val_EM={validation['exact_match']:.6f} "
            f"val_F1={validation['micro_f1']:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"patience={patience_count}/{args.patience}"
            + (" *" if improved else ""),
            flush=True,
        )
        if args.patience > 0 and patience_count >= args.patience:
            print("early stopping", flush=True)
            break
    return best_state, best_epoch, best_validation, epochs_ran


@torch.inference_mode()
def rcr_predictions(
    base_model: torch.nn.Module,
    verifier: SpatialMultiPrototypeVerifier,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    distance_weight: float,
    label_threshold: float,
    support_thresholds: np.ndarray,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    block2, block3, probabilities, targets = extract_multiscale_features(
        base_model,
        x,
        y,
        indices,
        active_indices,
        batch_size,
        device,
        name,
    )
    support, _, _ = collect_scores(
        verifier,
        block2,
        block3,
        batch_size,
        device,
        distance_weight,
    )
    predictions = predict(
        probabilities.numpy(), support, label_threshold, support_thresholds
    )
    result_targets = targets.numpy().astype(np.float32)
    del block2, block3, probabilities, targets, support
    return predictions, result_targets


def calibrate_polar_thresholds(
    baseline: np.ndarray,
    targets: np.ndarray,
    polar_probabilities: np.ndarray,
    class_names: Sequence[str],
    initial_retention: float,
    max_em_drop: float,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    baseline = baseline.astype(bool)
    truth = targets > 0.5
    baseline_em = float(np.mean(np.all(baseline == truth, axis=1)))
    retention_grid = sorted(
        set(
            [
                initial_retention,
                max(initial_retention, 0.9975),
                max(initial_retention, 0.9990),
                max(initial_retention, 0.9995),
                1.0,
            ]
        )
    )
    attempts: List[Dict[str, float]] = []
    selected_thresholds = None
    selected_retention = 1.0
    selected_prediction = baseline

    for retention in retention_grid:
        thresholds = []
        accepted = np.ones_like(baseline, dtype=bool)
        for class_index, class_name in enumerate(class_names):
            pool = baseline[:, class_index] & truth[:, class_index]
            if not pool.any():
                raise ValueError(
                    f"No true RCR-v2 validation candidate for {class_name}"
                )
            threshold = float(
                np.quantile(
                    polar_probabilities[pool, class_index],
                    1.0 - retention,
                )
            )
            thresholds.append(threshold)
            accepted[:, class_index] = (
                polar_probabilities[:, class_index] >= threshold
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
            selected_thresholds = np.asarray(thresholds, dtype=np.float32)
            selected_retention = float(retention)
            selected_prediction = prediction
            break
    if selected_thresholds is None:
        raise RuntimeError("Polar threshold relaxation failed unexpectedly")

    per_class = {}
    for class_index, class_name in enumerate(class_names):
        true_pool = baseline[:, class_index] & truth[:, class_index]
        false_pool = baseline[:, class_index] & ~truth[:, class_index]
        accepted = (
            polar_probabilities[:, class_index] >= selected_thresholds[class_index]
        )
        per_class[class_name] = {
            "threshold": float(selected_thresholds[class_index]),
            "true_candidate_retention": float(
                accepted[true_pool].mean() if true_pool.any() else 0.0
            ),
            "false_candidate_rejection": float(
                1.0 - accepted[false_pool].mean() if false_pool.any() else 0.0
            ),
            "true_candidates": int(true_pool.sum()),
            "false_candidates": int(false_pool.sum()),
            "true_probability_mean": float(
                polar_probabilities[true_pool, class_index].mean()
            ),
        }
    diagnostics = {
        "baseline_known_validation_exact_match": baseline_em,
        "gated_known_validation_exact_match": float(
            np.mean(np.all(selected_prediction == truth, axis=1))
        ),
        "selected_positive_retention": selected_retention,
        "retention_attempts": attempts,
        "per_class": per_class,
    }
    return selected_thresholds, selected_retention, diagnostics


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0.5 < args.positive_retention <= 1.0:
        raise ValueError("--positive-retention must be in (0.5,1.0]")
    if not 0.0 <= args.max_known_val_em_drop < 1.0:
        raise ValueError("--max-known-val-em-drop must be in [0,1)")
    set_seed(args.seed)

    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [index for index in range(len(FULL_CLASSES)) if index != unknown_index]
    class_names = [FULL_CLASSES[index] for index in active_indices]
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
    output_dir = Path(
        args.output_dir
        or f"checkpoint/rdp_polar_verifier/{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_polar_rdp.pt"
    metrics_path = output_dir / "polar_rdp_metrics.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"unknown={args.unknown_class}; known classes={class_names}")
    print(f"loading RCR-v2: {v2_path}")
    print(f"loading Cartesian RDP initialization: {base_path}")

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
    polar_model = copy.deepcopy(base_model).to(device)
    for parameter in polar_model.parameters():
        parameter.requires_grad_(True)

    loaders = {
        name: make_loader(
            x,
            y,
            split[name],
            args.batch_size if name == "known_train" else args.eval_batch_size,
            name == "known_train",
            args.num_workers,
            args.seed + offset,
        )
        for offset, name in enumerate(("known_train", "known_val", "known_test"))
    }
    best_state, best_epoch, best_validation, epochs_ran = train_polar_model(
        polar_model,
        loaders["known_train"],
        loaders["known_val"],
        active_indices,
        class_names,
        device,
        args,
        checkpoint_path,
    )
    polar_model.load_state_dict(best_state, strict=True)
    polar_model.eval()

    verifier = load_v2_model(v2_checkpoint, device)
    rcr_label_threshold = float(v2_checkpoint["label_threshold"])
    distance_weight = float(v2_checkpoint["distance_weight"])
    support_thresholds = v2_checkpoint["support_thresholds"]
    if isinstance(support_thresholds, torch.Tensor):
        support_thresholds = support_thresholds.cpu().numpy()
    support_thresholds = np.asarray(support_thresholds, dtype=np.float32)

    print("calibrating polar verifier on known_val")
    val_rcr, val_targets = rcr_predictions(
        base_model,
        verifier,
        x,
        y,
        split["known_val"],
        active_indices,
        args.eval_batch_size,
        device,
        distance_weight,
        rcr_label_threshold,
        support_thresholds,
        "known_val",
    )
    val_polar, val_polar_targets, _ = collect_polar_probabilities(
        polar_model,
        loaders["known_val"],
        active_indices,
        device,
    )
    if not np.array_equal(val_targets > 0.5, val_polar_targets > 0.5):
        raise RuntimeError("Known-validation target order mismatch")
    polar_thresholds, selected_retention, calibration = calibrate_polar_thresholds(
        val_rcr,
        val_targets,
        val_polar,
        class_names,
        args.positive_retention,
        args.max_known_val_em_drop,
    )

    print("scoring known_test")
    known_rcr, known_targets = rcr_predictions(
        base_model,
        verifier,
        x,
        y,
        split["known_test"],
        active_indices,
        args.eval_batch_size,
        device,
        distance_weight,
        rcr_label_threshold,
        support_thresholds,
        "known_test",
    )
    known_polar, known_polar_targets, _ = collect_polar_probabilities(
        polar_model,
        loaders["known_test"],
        active_indices,
        device,
    )
    if not np.array_equal(known_targets > 0.5, known_polar_targets > 0.5):
        raise RuntimeError("Known-test target order mismatch")

    # The held-out split is not loaded until training and calibration finish.
    print("scoring unknown_test (final diagnostic only)")
    unknown_loader = make_loader(
        x,
        y,
        split["unknown_test"],
        args.eval_batch_size,
        False,
        args.num_workers,
        args.seed + 100,
    )
    unknown_rcr, unknown_targets = rcr_predictions(
        base_model,
        verifier,
        x,
        y,
        split["unknown_test"],
        active_indices,
        args.eval_batch_size,
        device,
        distance_weight,
        rcr_label_threshold,
        support_thresholds,
        "unknown_test",
    )
    unknown_polar, unknown_polar_targets, _ = collect_polar_probabilities(
        polar_model,
        unknown_loader,
        active_indices,
        device,
    )
    if not np.array_equal(unknown_targets > 0.5, unknown_polar_targets > 0.5):
        raise RuntimeError("Unknown-test target order mismatch")

    known_fused = known_rcr & (
        known_polar >= polar_thresholds.reshape(1, -1)
    )
    unknown_fused = unknown_rcr & (
        unknown_polar >= polar_thresholds.reshape(1, -1)
    )
    known_polar_only = known_polar >= args.label_threshold
    unknown_polar_only = unknown_polar >= args.label_threshold
    results = {
        "known_test": {
            "rcr_v2": multilabel_metrics(known_rcr, known_targets, class_names),
            "polar_rdp_only": multilabel_metrics(
                known_polar_only, known_targets, class_names
            ),
            "rcr_v2_plus_polar_verifier": multilabel_metrics(
                known_fused, known_targets, class_names
            ),
        },
        "unknown_test": {
            "rcr_v2": multilabel_metrics(unknown_rcr, unknown_targets, class_names),
            "polar_rdp_only": multilabel_metrics(
                unknown_polar_only, unknown_targets, class_names
            ),
            "rcr_v2_plus_polar_verifier": multilabel_metrics(
                unknown_fused, unknown_targets, class_names
            ),
        },
    }

    configuration = {
        "method": "polar_coordinate_rdp_candidate_verifier",
        "unknown_class": args.unknown_class,
        "unknown_index": unknown_index,
        "known_classes": class_names,
        "data": str(Path(args.data).resolve()),
        "v2_checkpoint": str(v2_path),
        "base_checkpoint": str(base_path),
        "initialization": "copy of frozen Cartesian RDP, then polar fine-tuning",
        "heldout_data_used_for_training_selection_or_calibration": False,
        "polar_interpolation": "nearest",
        "polar_axes": ["radius", "angle"],
        "random_angular_roll": args.angular_roll,
        "epochs": args.epochs,
        "epochs_ran": epochs_ran,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "label_threshold": args.label_threshold,
        "initial_positive_retention": args.positive_retention,
        "selected_positive_retention": selected_retention,
        "max_known_validation_exact_match_drop": args.max_known_val_em_drop,
        "seed": args.seed,
    }
    metrics = {
        "configuration": configuration,
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "training": {
            "best_epoch": best_epoch,
            "best_known_validation": best_validation,
        },
        "known_validation_calibration": calibration,
        "polar_thresholds": {
            class_name: float(polar_thresholds[index])
            for index, class_name in enumerate(class_names)
        },
        "results": results,
        "notes": {
            "unknown_gate": "Dinomaly remains unchanged",
            "polar_role": "verify known-component candidates proposed by Cartesian RDP + RCR-v2",
            "unknown_test_role": "final diagnostic evaluation only",
        },
    }
    checkpoint = {
        "model_state_dict": best_state,
        "active_known_indices": active_indices,
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "full_class_order": list(FULL_CLASSES),
        "polar_thresholds": torch.from_numpy(polar_thresholds),
        "rcr_v2_checkpoint": str(v2_path),
        "base_checkpoint": str(base_path),
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "configuration": configuration,
    }
    torch.save(checkpoint, checkpoint_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\ncomparison")
    for split_name in ("known_test", "unknown_test"):
        print(split_name)
        for method_name, values in results[split_name].items():
            print(
                f"  {method_name}: EM={values['exact_match']:.6f} "
                f"micro_F1={values['micro_f1']:.6f} "
                f"FP/sample={values['mean_false_positives_per_sample']:.6f} "
                f"pure_false={values['pure_unknown_false_known_rate']}"
            )
    before = results["unknown_test"]["rcr_v2"]["per_class"]
    after = results["unknown_test"]["rcr_v2_plus_polar_verifier"]["per_class"]
    print("zero-support false-known classes")
    for class_name in class_names:
        if before[class_name]["positive_samples"] == 0:
            print(
                f"  {class_name}: FPR "
                f"{before[class_name]['false_positive_rate']:.6f} -> "
                f"{after[class_name]['false_positive_rate']:.6f}"
            )
    print("saved:", checkpoint_path)
    print("saved:", metrics_path)


if __name__ == "__main__":
    main()
