#!/usr/bin/env python3
"""Evaluate sector-occlusion evidence on top of frozen RDP + RCR-v2.

This is an evaluation/calibration-only module:

* the frozen RDP and RCR-v2 propose known components;
* every predicted class is tested by removing defect dies in angular sectors;
* a genuine local component should have a concentrated probability/logit drop;
* class-specific evidence scores and thresholds are selected using known
  validation data only;
* the held-out class is used only for the final diagnostic evaluation.

Only one file is written: ``sector_occlusion_metrics.json``.  Place this file
in ``rdp_net_repro/scripts/`` beside the RCR-v2 training script and run it from
the repository root.
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


EVIDENCE_NAMES: Tuple[str, ...] = (
    "probability_drop_gap",
    "probability_max_drop",
    "probability_max_x_concentration",
    "probability_max_x_locality",
    "logit_drop_gap",
    "logit_max_drop",
    "logit_max_x_concentration",
    "logit_max_x_locality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-only calibrated sector-occlusion verifier for RCR-v2"
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--occlusion-batch-size",
        type=int,
        default=512,
        help="Maximum number of sector-occluded wafers in one RDP forward pass",
    )
    parser.add_argument("--sector-count", type=int, default=8)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--positive-retention",
        type=float,
        default=0.995,
        help="Initial per-class retention of true RCR-v2 validation candidates",
    )
    parser.add_argument(
        "--max-known-val-em-drop",
        type=float,
        default=0.005,
        help="Relax evidence thresholds until known-validation EM drops no more",
    )
    parser.add_argument(
        "--minimum-false-candidates",
        type=int,
        default=10,
        help=(
            "Minimum known-validation false candidates required to select a "
            "class-specific evidence definition; otherwise use the default"
        ),
    )
    parser.add_argument(
        "--default-evidence",
        choices=EVIDENCE_NAMES,
        default="probability_drop_gap",
        help="Fallback evidence when known validation has too few false candidates",
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: directory containing the RCR-v2 checkpoint",
    )
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


@torch.inference_mode()
def forward_rdp_logits(model: torch.nn.Module, wafer: torch.Tensor) -> torch.Tensor:
    z = model.conv1(wafer)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    z = model.block_1_1(z)
    z = model.block_1_2(z)
    z = model.block_2_1(z)
    z = model.block_2_2(z)
    z = model.block_3_1(z)
    z = model.block_3_2(z)
    z = model.block_4_1(z)
    z = model.block_4_2(z)
    return model.fc(torch.flatten(model.avgpool(z), 1))


def angular_sector_masks(
    height: int, width: int, sector_count: int, device: torch.device
) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height, device=device).view(height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, width)
    angle = torch.remainder(torch.atan2(yy, xx) + 2.0 * math.pi, 2.0 * math.pi)
    sector_index = torch.floor(angle / (2.0 * math.pi / sector_count)).long()
    return torch.stack(
        [sector_index == index for index in range(sector_count)], dim=0
    )


def summarize_drops(
    base: torch.Tensor, occluded: torch.Tensor, prefix: str
) -> Dict[str, torch.Tensor]:
    """Summarize positive [base - occluded] changes across sectors."""
    drops = (base[:, None, :] - occluded).clamp_min(0.0)
    top = torch.topk(drops, k=min(2, drops.shape[1]), dim=1).values
    largest = top[:, 0]
    second = top[:, 1] if top.shape[1] > 1 else torch.zeros_like(largest)
    total = drops.sum(dim=1)
    concentration = largest / total.clamp_min(1e-8)
    distribution = drops / total[:, None, :].clamp_min(1e-8)
    entropy = -(
        distribution * torch.log(distribution.clamp_min(1e-8))
    ).sum(dim=1) / math.log(max(2, drops.shape[1]))
    locality = (1.0 - entropy).clamp(0.0, 1.0)
    no_drop = total <= 1e-8
    concentration = torch.where(no_drop, torch.zeros_like(concentration), concentration)
    locality = torch.where(no_drop, torch.zeros_like(locality), locality)
    return {
        f"{prefix}_drop_gap": largest - second,
        f"{prefix}_max_drop": largest,
        f"{prefix}_max_x_concentration": largest * concentration,
        f"{prefix}_max_x_locality": largest * locality,
    }


@torch.inference_mode()
def sector_occlusion_evidence(
    model: torch.nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    base_probabilities: torch.Tensor,
    batch_size: int,
    occlusion_batch_size: int,
    sector_count: int,
    device: torch.device,
    normal_value: float,
    defect_value: float,
    name: str,
) -> Dict[str, np.ndarray]:
    """Compute per-class evidence without using labels."""
    height, width = int(x.shape[-2]), int(x.shape[-1])
    sectors = angular_sector_masks(height, width, sector_count, device)
    cutoff = (normal_value + defect_value) / 2.0
    collected: Dict[str, List[np.ndarray]] = {
        evidence_name: [] for evidence_name in EVIDENCE_NAMES
    }
    print(f"sector occlusion: {name} ({len(indices)} samples, {sector_count} sectors)")

    for start in range(0, len(indices), batch_size):
        local_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        wafer = x[local_indices].unsqueeze(1).to(device, non_blocking=True)
        defect = wafer > cutoff
        occluded_parts: List[torch.Tensor] = []
        for sector_index in range(sector_count):
            local = wafer.clone()
            remove = defect & sectors[sector_index].view(1, 1, height, width)
            local[remove] = normal_value
            occluded_parts.append(local)
        all_occluded = torch.cat(occluded_parts, dim=0)

        local_logits: List[torch.Tensor] = []
        for inner_start in range(0, len(all_occluded), occlusion_batch_size):
            logits = forward_rdp_logits(
                model,
                all_occluded[inner_start : inner_start + occlusion_batch_size],
            )
            local_logits.append(logits[:, active_indices])
        occluded_logits = torch.cat(local_logits, dim=0)
        batch_count = len(wafer)
        occluded_logits = occluded_logits.view(
            sector_count, batch_count, len(active_indices)
        ).permute(1, 0, 2)
        occluded_probability = torch.sigmoid(occluded_logits)

        base_probability = base_probabilities[
            start : start + batch_count
        ].to(device, non_blocking=True).clamp(1e-6, 1.0 - 1e-6)
        base_logits = torch.logit(base_probability)
        summaries = {}
        summaries.update(
            summarize_drops(base_probability, occluded_probability, "probability")
        )
        summaries.update(summarize_drops(base_logits, occluded_logits, "logit"))
        for evidence_name in EVIDENCE_NAMES:
            collected[evidence_name].append(
                summaries[evidence_name].cpu().numpy().astype(np.float32)
            )

        del wafer, defect, all_occluded, occluded_logits, occluded_probability

    return {
        evidence_name: np.concatenate(parts, axis=0)
        for evidence_name, parts in collected.items()
    }


@torch.inference_mode()
def score_split(
    base_model: torch.nn.Module,
    verifier: SpatialMultiPrototypeVerifier,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    occlusion_batch_size: int,
    sector_count: int,
    device: torch.device,
    distance_weight: float,
    label_threshold: float,
    support_thresholds: np.ndarray,
    normal_value: float,
    defect_value: float,
    name: str,
) -> Dict[str, object]:
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
    baseline = predict(
        probabilities.numpy(), support, label_threshold, support_thresholds
    )
    evidence = sector_occlusion_evidence(
        base_model,
        x,
        indices,
        active_indices,
        probabilities,
        batch_size,
        occlusion_batch_size,
        sector_count,
        device,
        normal_value,
        defect_value,
        name,
    )
    result = {
        "baseline": baseline,
        "targets": targets.numpy().astype(np.float32),
        "evidence": evidence,
    }
    del block2, block3, probabilities, targets, support
    return result


def threshold_at_retention(
    values: np.ndarray, positive_pool: np.ndarray, retention: float
) -> float:
    if not positive_pool.any():
        raise ValueError("No true positive RCR-v2 candidate for calibration")
    return float(np.quantile(values[positive_pool], 1.0 - retention))


def select_evidence_definitions(
    validation: Dict[str, object],
    class_names: Sequence[str],
    retention: float,
    minimum_false_candidates: int,
    default_evidence: str,
) -> Tuple[List[str], List[Dict[str, object]]]:
    baseline = np.asarray(validation["baseline"], dtype=bool)
    targets = np.asarray(validation["targets"]) > 0.5
    evidence = validation["evidence"]
    selected: List[str] = []
    diagnostics: List[Dict[str, object]] = []

    for class_index, class_name in enumerate(class_names):
        true_pool = baseline[:, class_index] & targets[:, class_index]
        false_pool = baseline[:, class_index] & ~targets[:, class_index]
        if not true_pool.any():
            raise ValueError(f"No true validation candidate for {class_name}")

        candidates: List[Dict[str, object]] = []
        for evidence_name in EVIDENCE_NAMES:
            values = evidence[evidence_name][:, class_index]
            threshold = threshold_at_retention(values, true_pool, retention)
            accepted_false = false_pool & (values >= threshold)
            false_count = int(false_pool.sum())
            rejected = int(false_count - accepted_false.sum())
            false_rejection = rejected / max(false_count, 1)
            if false_pool.any():
                separation = float(
                    np.median(values[true_pool]) - np.median(values[false_pool])
                )
            else:
                separation = 0.0
            candidates.append(
                {
                    "evidence": evidence_name,
                    "threshold_at_initial_retention": threshold,
                    "known_false_rejection": false_rejection,
                    "median_separation": separation,
                }
            )

        if int(false_pool.sum()) >= minimum_false_candidates:
            best = max(
                candidates,
                key=lambda item: (
                    item["known_false_rejection"],
                    item["median_separation"],
                ),
            )
            evidence_name = str(best["evidence"])
            reason = "known_validation_false_candidate_selection"
        else:
            evidence_name = default_evidence
            reason = "default_due_to_few_false_candidates"
        selected.append(evidence_name)
        diagnostics.append(
            {
                "class": class_name,
                "selected_evidence": evidence_name,
                "selection_reason": reason,
                "true_candidates": int(true_pool.sum()),
                "false_candidates": int(false_pool.sum()),
                "candidate_diagnostics": candidates,
            }
        )
    return selected, diagnostics


def calibrate_thresholds(
    validation: Dict[str, object],
    class_names: Sequence[str],
    selected_evidence: Sequence[str],
    initial_retention: float,
    max_known_val_em_drop: float,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    baseline = np.asarray(validation["baseline"], dtype=bool)
    targets = np.asarray(validation["targets"]) > 0.5
    evidence = validation["evidence"]
    baseline_em = float(np.mean(np.all(baseline == targets, axis=1)))
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
        for class_index, evidence_name in enumerate(selected_evidence):
            true_pool = baseline[:, class_index] & targets[:, class_index]
            values = evidence[evidence_name][:, class_index]
            threshold = threshold_at_retention(values, true_pool, retention)
            thresholds.append(threshold)
            accepted[:, class_index] = values >= threshold
        prediction = baseline & accepted
        em = float(np.mean(np.all(prediction == targets, axis=1)))
        attempts.append(
            {
                "positive_retention": float(retention),
                "known_validation_exact_match": em,
                "change_from_baseline": em - baseline_em,
            }
        )
        if em >= baseline_em - max_known_val_em_drop:
            selected_thresholds = np.asarray(thresholds, dtype=np.float32)
            selected_retention = float(retention)
            selected_prediction = prediction
            break

    if selected_thresholds is None:
        raise RuntimeError("Threshold relaxation failed unexpectedly")
    gated_em = float(np.mean(np.all(selected_prediction == targets, axis=1)))
    per_class = {}
    for class_index, class_name in enumerate(class_names):
        true_pool = baseline[:, class_index] & targets[:, class_index]
        false_pool = baseline[:, class_index] & ~targets[:, class_index]
        accepted = (
            validation["evidence"][selected_evidence[class_index]][:, class_index]
            >= selected_thresholds[class_index]
        )
        per_class[class_name] = {
            "evidence": selected_evidence[class_index],
            "threshold": float(selected_thresholds[class_index]),
            "true_candidate_retention": float(
                accepted[true_pool].mean() if true_pool.any() else 0.0
            ),
            "false_candidate_rejection": float(
                1.0 - accepted[false_pool].mean() if false_pool.any() else 0.0
            ),
            "true_candidates": int(true_pool.sum()),
            "false_candidates": int(false_pool.sum()),
        }
    diagnostics = {
        "baseline_known_validation_exact_match": baseline_em,
        "gated_known_validation_exact_match": gated_em,
        "selected_positive_retention": selected_retention,
        "retention_attempts": attempts,
        "per_class": per_class,
    }
    return selected_thresholds, selected_retention, diagnostics


def apply_gate(
    scored: Dict[str, object],
    selected_evidence: Sequence[str],
    thresholds: np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(scored["baseline"], dtype=bool)
    accepted = np.ones_like(baseline, dtype=bool)
    for class_index, evidence_name in enumerate(selected_evidence):
        accepted[:, class_index] = (
            scored["evidence"][evidence_name][:, class_index]
            >= thresholds[class_index]
        )
    return baseline & accepted


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.occlusion_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.sector_count < 3:
        raise ValueError("--sector-count must be >= 3")
    if not 0.5 < args.positive_retention <= 1.0:
        raise ValueError("--positive-retention must be in (0.5,1.0]")
    if not 0.0 <= args.max_known_val_em_drop < 1.0:
        raise ValueError("--max-known-val-em-drop must be in [0,1)")
    if args.minimum_false_candidates < 0:
        raise ValueError("--minimum-false-candidates must be non-negative")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    output_dir = Path(args.output_dir or v2_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "sector_occlusion_metrics.json"
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
    verifier = load_v2_model(v2_checkpoint, device)
    label_threshold = float(v2_checkpoint["label_threshold"])
    distance_weight = float(v2_checkpoint["distance_weight"])
    support_thresholds = v2_checkpoint["support_thresholds"]
    if isinstance(support_thresholds, torch.Tensor):
        support_thresholds = support_thresholds.cpu().numpy()
    support_thresholds = np.asarray(support_thresholds, dtype=np.float32)
    _, normal_value, defect_value = data_values(x)

    scored = {}
    # Calibrate entirely before the held-out split is loaded.
    for name in ("known_val", "known_test"):
        print(f"scoring {name}")
        scored[name] = score_split(
            base_model,
            verifier,
            x,
            y,
            split[name],
            active_indices,
            args.batch_size,
            args.occlusion_batch_size,
            args.sector_count,
            device,
            distance_weight,
            label_threshold,
            support_thresholds,
            normal_value,
            defect_value,
            name,
        )

    selected_evidence, evidence_selection = select_evidence_definitions(
        scored["known_val"],
        class_names,
        args.positive_retention,
        args.minimum_false_candidates,
        args.default_evidence,
    )
    thresholds, selected_retention, calibration = calibrate_thresholds(
        scored["known_val"],
        class_names,
        selected_evidence,
        args.positive_retention,
        args.max_known_val_em_drop,
    )

    print("scoring unknown_test (final diagnostic only)")
    scored["unknown_test"] = score_split(
        base_model,
        verifier,
        x,
        y,
        split["unknown_test"],
        active_indices,
        args.batch_size,
        args.occlusion_batch_size,
        args.sector_count,
        device,
        distance_weight,
        label_threshold,
        support_thresholds,
        normal_value,
        defect_value,
        "unknown_test",
    )

    results = {}
    for name in ("known_test", "unknown_test"):
        baseline = np.asarray(scored[name]["baseline"], dtype=bool)
        gated = apply_gate(scored[name], selected_evidence, thresholds)
        targets = np.asarray(scored[name]["targets"])
        results[name] = {
            "rcr_v2": multilabel_metrics(baseline, targets, class_names),
            "rcr_v2_plus_sector_occlusion": multilabel_metrics(
                gated, targets, class_names
            ),
        }

    metrics = {
        "configuration": {
            "method": "known_only_calibrated_sector_occlusion_rcr",
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_index,
            "known_classes": class_names,
            "data": str(Path(args.data).resolve()),
            "v2_checkpoint": str(v2_path),
            "base_checkpoint": str(base_path),
            "rdp_and_rcr_frozen": True,
            "heldout_data_used_for_fitting_selection_or_calibration": False,
            "sector_count": args.sector_count,
            "initial_positive_retention": args.positive_retention,
            "selected_positive_retention": selected_retention,
            "max_known_validation_exact_match_drop": args.max_known_val_em_drop,
            "default_evidence": args.default_evidence,
            "minimum_false_candidates": args.minimum_false_candidates,
            "seed": args.seed,
        },
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "evidence_selection": {
            class_name: evidence_selection[index]
            for index, class_name in enumerate(class_names)
        },
        "known_validation_calibration": calibration,
        "results": results,
        "notes": {
            "unknown_gate": "Dinomaly remains unchanged",
            "occlusion": "only observed defect dies in one angular sector are reset to normal",
            "threshold_source": "known validation only",
            "unknown_test_role": "final diagnostic evaluation only",
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\ncomparison")
    for split_name in ("known_test", "unknown_test"):
        print(split_name)
        for method, values in results[split_name].items():
            print(
                f"  {method}: EM={values['exact_match']:.6f} "
                f"micro_F1={values['micro_f1']:.6f} "
                f"FP/sample={values['mean_false_positives_per_sample']:.6f} "
                f"pure_false={values['pure_unknown_false_known_rate']}"
            )
    print("dominant ER-like confusion check")
    unknown_before = results["unknown_test"]["rcr_v2"]["per_class"]
    unknown_after = results["unknown_test"][
        "rcr_v2_plus_sector_occlusion"
    ]["per_class"]
    for class_name in class_names:
        if unknown_before[class_name]["positive_samples"] == 0:
            print(
                f"  {class_name}: FPR "
                f"{unknown_before[class_name]['false_positive_rate']:.6f} -> "
                f"{unknown_after[class_name]['false_positive_rate']:.6f}"
            )
    print("saved:", metrics_path)


if __name__ == "__main__":
    main()
