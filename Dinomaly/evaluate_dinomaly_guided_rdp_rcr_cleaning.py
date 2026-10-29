#!/usr/bin/env python3
"""Evaluate Dinomaly-guided unknown-region removal before RDP+RCR-v2.

This is an evaluation-only bridge between two already trained branches:

1. Dinomaly produces a wafer-level unknown score and a pixel anomaly map.
2. High-anomaly *defect dies* are changed to the normal-die value.
3. The cleaned wafer is passed through frozen RDP and frozen RCR-v2 to report
   the remaining known components.

The cleaning strength, image-gate threshold, and model selection use known
validation wafers only.  Held-out wafers are touched only for the final report.
No model, anomaly map, or intermediate feature is saved; the only output is a
single JSON metrics file.

Place this file in the Dinomaly repository root.  The RDP repository remains a
sibling directory and is supplied with --rdp-repo.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_eval_dinomaly_atomic_open_set import (
    FULL_CLASSES,
    MixedWM38DinomalyDataset,
    build_model,
    cal_anomaly_maps,
    get_gaussian_kernel,
    load_arrays,
    make_loader,
    make_open_set_split,
    set_seed,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dinomaly-guided cleaning followed by frozen RDP+RCR-v2"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument(
        "--dinomaly-checkpoint",
        default="checkpoint/dinomaly_atomic_20k/c_heldout/best_model.pt",
    )
    parser.add_argument(
        "--rdp-repo",
        default="../rdp_net_repro",
        help="RDP repository containing scripts/ and src/",
    )
    parser.add_argument(
        "--rdp-checkpoint",
        default=None,
        help="Default: base_checkpoint recorded in the RCR-v2 checkpoint",
    )
    parser.add_argument(
        "--rcr-checkpoint",
        default=(
            "../rdp_net_repro/checkpoint/rdp_reciprocal_v2/"
            "c_heldout/best_spatial_reciprocal_verifier.pt"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rdp-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--known-reject-rate",
        type=float,
        default=0.05,
        help="Known-validation rejection rate used by the Dinomaly image gate",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=None,
        help="Dinomaly image-score fraction; default comes from its checkpoint",
    )
    parser.add_argument(
        "--mask-fractions",
        default="0,0.001,0.0025,0.005,0.01,0.02,0.05,0.10",
        help=(
            "Comma-separated known-validation defect-die fractions considered "
            "for removal. Zero is the no-cleaning baseline."
        ),
    )
    parser.add_argument(
        "--max-known-val-em-drop",
        type=float,
        default=0.005,
        help="Largest known-validation exact-match loss allowed during calibration",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Off by default because this Dinomaly decoder was unstable in FP16",
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument(
        "--output",
        default=(
            "checkpoint/dinomaly_guided_rdp_rcr/"
            "c_heldout/cleaning_metrics.json"
        ),
    )
    return parser.parse_args()


def parse_fractions(text: str) -> List[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("--mask-fractions is empty")
    if values[0] < 0.0 or values[-1] >= 1.0:
        raise ValueError("Every mask fraction must be in [0,1)")
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def import_rdp_modules(rdp_repo: Path):
    scripts = (rdp_repo / "scripts").resolve()
    source = (rdp_repo / "src").resolve()
    if not scripts.is_dir() or not source.is_dir():
        raise FileNotFoundError(
            f"Expected scripts/ and src/ below RDP repository: {rdp_repo}"
        )
    for path in (scripts, source):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    import train_eval_reciprocal_component_rejection as rcr_v1
    import train_eval_reciprocal_component_rejection_v2 as rcr_v2

    return rcr_v1, rcr_v2


def resolve_recorded_path(
    explicit: str | None,
    recorded: str | None,
    rdp_repo: Path,
    description: str,
) -> Path:
    candidate_text = explicit or recorded
    if not candidate_text:
        raise ValueError(f"No {description} path was supplied or recorded")
    candidate = Path(candidate_text).expanduser()
    alternatives = [candidate]
    if not candidate.is_absolute():
        alternatives.extend((rdp_repo / candidate, rdp_repo.parent / candidate))
    for path in alternatives:
        if path.is_file():
            return path.resolve()
    rendered = "\n  ".join(str(path.resolve()) for path in alternatives)
    raise FileNotFoundError(f"Cannot find {description}. Checked:\n  {rendered}")


def load_dinomaly(
    checkpoint_path: Path,
    unknown_class: str,
    device: torch.device,
):
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    configuration = checkpoint.get("configuration", {})
    stored_unknown = configuration.get("unknown_class")
    if stored_unknown is not None and stored_unknown != unknown_class:
        raise ValueError(
            f"Dinomaly checkpoint holds out {stored_unknown}, not {unknown_class}"
        )
    encoder = configuration.get("encoder", "dinov2reg_vit_small_14")
    image_size = int(configuration.get("image_size", 392))
    decoder_depth = int(configuration.get("decoder_depth", 8))
    input_encoding = configuration.get("input_encoding", "gray_repeat")
    build_parameters = inspect.signature(build_model).parameters
    if "input_encoding" in build_parameters:
        model, bottleneck, decoder, _ = build_model(
            encoder,
            decoder_depth,
            device,
            input_encoding=input_encoding,
        )
    else:
        if input_encoding != "gray_repeat":
            raise TypeError(
                "The local build_model() has no input_encoding argument, but "
                f"the checkpoint requires {input_encoding!r}. Update the local "
                "Dinomaly training module before evaluating this checkpoint."
            )
        model, bottleneck, decoder, _ = build_model(
            encoder,
            decoder_depth,
            device,
        )
    bottleneck.load_state_dict(checkpoint["bottleneck_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    adapter = getattr(model, "wafer_adapter", None)
    if adapter is not None:
        if "wafer_adapter_state_dict" not in checkpoint:
            raise KeyError("Adapter Dinomaly checkpoint lacks wafer_adapter_state_dict")
        adapter.load_state_dict(checkpoint["wafer_adapter_state_dict"], strict=True)
    model.eval()
    return model, configuration, image_size, input_encoding


def make_dinomaly_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
    image_size: int,
    input_encoding: str,
):
    """Construct either the old gray-repeat or the newer multi-encoding dataset."""
    dataset_parameters = inspect.signature(
        MixedWM38DinomalyDataset.__init__
    ).parameters
    if "input_encoding" in dataset_parameters:
        return MixedWM38DinomalyDataset(
            x,
            y,
            indices,
            image_size=image_size,
            input_encoding=input_encoding,
        )
    if input_encoding != "gray_repeat":
        raise TypeError(
            "The local MixedWM38DinomalyDataset has no input_encoding argument, "
            f"but the checkpoint requires {input_encoding!r}."
        )
    return MixedWM38DinomalyDataset(
        x,
        y,
        indices,
        image_size=image_size,
    )


def load_rdp_rcr(
    rcr_checkpoint_path: Path,
    explicit_rdp_checkpoint: str | None,
    rdp_repo: Path,
    unknown_class: str,
    device: torch.device,
):
    rcr_v1, rcr_v2 = import_rdp_modules(rdp_repo)
    checkpoint = rcr_v1.torch_load(rcr_checkpoint_path, map_location="cpu")
    if checkpoint.get("unknown_class") != unknown_class:
        raise ValueError(
            "RCR-v2 checkpoint unknown class is "
            f"{checkpoint.get('unknown_class')}, not {unknown_class}"
        )
    rdp_path = resolve_recorded_path(
        explicit_rdp_checkpoint,
        checkpoint.get("base_checkpoint"),
        rdp_repo,
        "RDP checkpoint",
    )
    rdp_model, active_indices, _ = rcr_v1.load_frozen_rdp(
        rdp_path, unknown_class, device
    )
    stored_active = [int(value) for value in checkpoint["active_known_indices"]]
    if stored_active != active_indices:
        raise ValueError(
            f"RCR active indices {stored_active} do not match RDP {active_indices}"
        )
    configuration = checkpoint["model_configuration"]
    verifier = rcr_v2.SpatialMultiPrototypeVerifier(
        class_count=int(configuration["class_count"]),
        block2_shape=configuration["block2_shape"],
        block3_shape=configuration["block3_shape"],
        embedding_dim=int(configuration["embedding_dim"]),
        prototype_count=int(configuration["prototype_count"]),
        prototype_temperature=float(configuration["prototype_temperature"]),
        logit_scale=float(configuration["logit_scale"]),
    ).to(device)
    verifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    verifier.eval()
    for parameter in verifier.parameters():
        parameter.requires_grad = False
    return {
        "rdp_model": rdp_model,
        "verifier": verifier,
        "forward_multiscale": rcr_v2.forward_frozen_multiscale,
        "multilabel_metrics": rcr_v1.multilabel_metrics,
        "active_indices": active_indices,
        "class_names": [FULL_CLASSES[index] for index in active_indices],
        "label_threshold": float(checkpoint.get("label_threshold", 0.5)),
        "distance_weight": float(checkpoint.get("distance_weight", 2.0)),
        "support_thresholds": checkpoint["support_thresholds"].float().cpu(),
        "rdp_checkpoint": rdp_path,
    }


@torch.inference_mode()
def collect_dinomaly_maps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    raw_size: Tuple[int, int],
    top_fraction: float,
    amp: bool,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    map_parts: List[torch.Tensor] = []
    score_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    for images, _labels, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            encoded, decoded = model(images)
            anomaly_map, _ = cal_anomaly_maps(
                encoded, decoded, out_size=image_size
            )
        anomaly_map = gaussian(anomaly_map.float())
        if not torch.isfinite(anomaly_map).all():
            raise FloatingPointError(
                "Dinomaly produced a non-finite map; run with --no-amp"
            )
        flattened = anomaly_map.flatten(1)
        keep = max(1, int(math.ceil(flattened.shape[1] * top_fraction)))
        image_score = torch.topk(flattened, k=keep, dim=1).values.mean(dim=1)
        raw_map = F.interpolate(anomaly_map, size=raw_size, mode="area")[:, 0]
        map_parts.append(raw_map.cpu().to(torch.float16))
        score_parts.append(image_score.cpu().numpy().astype(np.float32))
        index_parts.append(indices.numpy().astype(np.int64))
    return (
        torch.cat(map_parts, dim=0),
        np.concatenate(score_parts),
        np.concatenate(index_parts),
    )


def raw_values(x: torch.Tensor) -> Tuple[float, float]:
    maximum = float(x.max().item())
    if maximum > 1.5:
        return 1.0, 2.0
    return 0.5, 1.0


def cleaning_threshold(
    maps: torch.Tensor,
    wafers: torch.Tensor,
    defect_value: float,
    fraction: float,
    eligible_images: np.ndarray | None,
) -> float:
    if fraction <= 0.0:
        return float(np.finfo(np.float32).max)
    # For categorical 0/1/2 and normalized 0/.5/1 arrays, this is equivalent
    # to selecting only the highest-valued (defect) dies.
    defect = wafers >= (0.75 * defect_value)
    if eligible_images is not None:
        eligible = torch.from_numpy(eligible_images.astype(bool)).view(-1, 1, 1)
        defect = defect & eligible
    values = maps.float()[defect]
    if values.numel() == 0:
        raise ValueError("No eligible known-validation defect dies for calibration")
    return float(np.quantile(values.numpy(), 1.0 - fraction))


def clean_wafers(
    wafers: torch.Tensor,
    maps: torch.Tensor,
    normal_value: float,
    defect_value: float,
    threshold: float,
    eligible_images: np.ndarray | None,
) -> Tuple[torch.Tensor, Dict[str, float | int]]:
    defect = wafers >= (0.75 * defect_value)
    remove = defect & (maps.float() > threshold)
    if eligible_images is not None:
        eligible = torch.from_numpy(eligible_images.astype(bool)).view(-1, 1, 1)
        remove = remove & eligible
    cleaned = wafers.clone()
    cleaned[remove] = float(normal_value)
    defect_count = int(defect.sum().item())
    removed_count = int(remove.sum().item())
    affected = remove.flatten(1).any(dim=1)
    return cleaned, {
        "defect_dies": defect_count,
        "removed_defect_dies": removed_count,
        "removed_defect_die_fraction": removed_count / max(defect_count, 1),
        "affected_samples": int(affected.sum().item()),
        "affected_sample_fraction": float(affected.float().mean().item()),
    }


@torch.inference_mode()
def predict_rdp_rcr(
    wafers: torch.Tensor,
    stack: Dict[str, object],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rdp_model = stack["rdp_model"]
    verifier = stack["verifier"]
    forward_multiscale = stack["forward_multiscale"]
    active_indices = stack["active_indices"]
    label_threshold = stack["label_threshold"]
    distance_weight = stack["distance_weight"]
    thresholds = stack["support_thresholds"].to(device).view(1, -1)
    predictions: List[np.ndarray] = []
    for start in range(0, len(wafers), batch_size):
        local = wafers[start : start + batch_size].unsqueeze(1).to(
            device, non_blocking=True
        )
        logits, block2, block3 = forward_multiscale(rdp_model, local)
        probability = torch.sigmoid(logits[:, active_indices])
        relative, positive_distance, _, _ = verifier(block2, block3)
        support = relative - float(distance_weight) * positive_distance
        prediction = (probability >= float(label_threshold)) & (
            support >= thresholds
        )
        predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions).astype(bool)


def metrics_for(
    predictions: np.ndarray,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    class_names: Sequence[str],
    metric_function,
) -> Dict[str, object]:
    targets = y[torch.as_tensor(indices)][:, active_indices].cpu().numpy()
    return metric_function(predictions, targets, class_names)


def improvement(
    baseline: Dict[str, object], result: Dict[str, object]
) -> Dict[str, float | None]:
    keys = {
        "exact_match": "exact_match",
        "micro_f1": "micro_f1",
        "fp_per_sample": "mean_false_positives_per_sample",
        "fn_per_sample": "mean_false_negatives_per_sample",
        "pure_false_known_rate": "pure_unknown_false_known_rate",
    }
    output: Dict[str, float | None] = {}
    for short, key in keys.items():
        before = baseline.get(key)
        after = result.get(key)
        output[short] = (
            None if before is None or after is None else float(after) - float(before)
        )
    return output


def choose_cleaning_strength(
    mode: str,
    fractions: Iterable[float],
    known_val_wafers: torch.Tensor,
    known_val_maps: torch.Tensor,
    known_val_gate: np.ndarray,
    y: torch.Tensor,
    known_val_indices: np.ndarray,
    stack: Dict[str, object],
    normal_value: float,
    defect_value: float,
    baseline_em: float,
    max_em_drop: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    gate = known_val_gate if mode == "gate_positive" else None
    trials: List[Dict[str, object]] = []
    valid: List[Dict[str, object]] = []
    for fraction in fractions:
        threshold = cleaning_threshold(
            known_val_maps,
            known_val_wafers,
            defect_value,
            fraction,
            gate,
        )
        if fraction == 0.0:
            cleaned = known_val_wafers
            cleaning = {
                "defect_dies": int((known_val_wafers >= 0.75 * defect_value).sum()),
                "removed_defect_dies": 0,
                "removed_defect_die_fraction": 0.0,
                "affected_samples": 0,
                "affected_sample_fraction": 0.0,
            }
        else:
            cleaned, cleaning = clean_wafers(
                known_val_wafers,
                known_val_maps,
                normal_value,
                defect_value,
                threshold,
                gate,
            )
        prediction = predict_rdp_rcr(cleaned, stack, batch_size, device)
        metrics = metrics_for(
            prediction,
            y,
            known_val_indices,
            stack["active_indices"],
            stack["class_names"],
            stack["multilabel_metrics"],
        )
        trial = {
            "requested_mask_fraction": float(fraction),
            "anomaly_map_threshold": threshold,
            "known_validation_exact_match": float(metrics["exact_match"]),
            "known_validation_micro_f1": float(metrics["micro_f1"]),
            "cleaning": cleaning,
            "accepted": bool(metrics["exact_match"] >= baseline_em - max_em_drop),
        }
        trials.append(trial)
        if trial["accepted"]:
            valid.append(trial)
        del prediction
        if cleaned is not known_val_wafers:
            del cleaned
    if not valid:
        raise RuntimeError("Even the no-cleaning candidate failed the EM constraint")
    # The most aggressive known-safe candidate is fixed before unknown_test is read.
    selected = max(valid, key=lambda item: item["requested_mask_fraction"])
    return selected, trials


def evaluate_strategy(
    mode: str,
    selected: Dict[str, object],
    split_wafers: Dict[str, torch.Tensor],
    split_maps: Dict[str, torch.Tensor],
    split_gate: Dict[str, np.ndarray],
    y: torch.Tensor,
    split: Dict[str, np.ndarray],
    stack: Dict[str, object],
    normal_value: float,
    defect_value: float,
    baseline: Dict[str, Dict[str, object]],
    rdp_batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "selected_requested_mask_fraction": selected["requested_mask_fraction"],
        "selected_anomaly_map_threshold": selected["anomaly_map_threshold"],
        "selection_used_unknown_data": False,
        "splits": {},
    }
    threshold = float(selected["anomaly_map_threshold"])
    for name in ("known_test", "unknown_test"):
        gate = split_gate[name] if mode == "gate_positive" else None
        cleaned, cleaning = clean_wafers(
            split_wafers[name],
            split_maps[name],
            normal_value,
            defect_value,
            threshold,
            gate,
        )
        prediction = predict_rdp_rcr(cleaned, stack, rdp_batch_size, device)
        metrics = metrics_for(
            prediction,
            y,
            split[name],
            stack["active_indices"],
            stack["class_names"],
            stack["multilabel_metrics"],
        )
        result["splits"][name] = {
            "cleaning": cleaning,
            "metrics": metrics,
            "change_vs_baseline": improvement(baseline[name], metrics),
        }
        del cleaned, prediction
    return result


def main() -> None:
    args = parse_args()
    fractions = parse_fractions(args.mask_fractions)
    if args.batch_size <= 0 or args.rdp_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if not 0.0 < args.known_reject_rate < 1.0:
        raise ValueError("--known-reject-rate must be in (0,1)")
    if args.max_known_val_em_drop < 0.0:
        raise ValueError("--max-known-val-em-drop must be non-negative")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unknown_index = FULL_CLASSES.index(args.unknown_class)
    dinomaly_path = Path(args.dinomaly_checkpoint).expanduser().resolve()
    rcr_path = Path(args.rcr_checkpoint).expanduser().resolve()
    rdp_repo = Path(args.rdp_repo).expanduser().resolve()
    for path, name in ((dinomaly_path, "Dinomaly"), (rcr_path, "RCR-v2")):
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    print(f"device: {device}")
    print(f"unknown class: {args.unknown_class}")
    print(f"loading Dinomaly: {dinomaly_path}", flush=True)
    dinomaly, dino_configuration, image_size, input_encoding = load_dinomaly(
        dinomaly_path, args.unknown_class, device
    )
    top_fraction = (
        float(args.top_fraction)
        if args.top_fraction is not None
        else float(dino_configuration.get("top_fraction", 0.005))
    )
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0,1]")

    x, y = load_arrays(args.data)
    split = make_open_set_split(
        y,
        unknown_index,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    names = ("known_val", "known_test", "unknown_test")
    print(
        "split: " + ", ".join(f"{name}={len(split[name])}" for name in names),
        flush=True,
    )
    raw_size = (int(x.shape[-2]), int(x.shape[-1]))
    loaders = {
        name: make_loader(
            make_dinomaly_dataset(
                x,
                y,
                split[name],
                image_size=image_size,
                input_encoding=input_encoding,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed + offset,
        )
        for offset, name in enumerate(names, 1)
    }

    split_maps: Dict[str, torch.Tensor] = {}
    split_scores: Dict[str, np.ndarray] = {}
    split_wafers: Dict[str, torch.Tensor] = {}
    for name in names:
        print(f"Dinomaly scoring and mapping: {name}", flush=True)
        maps, scores, indices = collect_dinomaly_maps(
            dinomaly,
            loaders[name],
            device,
            image_size,
            raw_size,
            top_fraction,
            args.amp,
        )
        if not np.array_equal(indices, split[name]):
            raise RuntimeError(f"{name} loader changed sample order")
        split_maps[name] = maps
        split_scores[name] = scores
        split_wafers[name] = x[torch.as_tensor(split[name])].clone()

    # Dinomaly is no longer needed after its maps have been reduced to raw-wafer
    # resolution. Releasing it leaves more GPU memory for batched RDP+RCR passes.
    del dinomaly
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"loading RCR-v2: {rcr_path}", flush=True)
    stack = load_rdp_rcr(
        rcr_path,
        args.rdp_checkpoint,
        rdp_repo,
        args.unknown_class,
        device,
    )

    image_gate_threshold = float(
        np.quantile(split_scores["known_val"], 1.0 - args.known_reject_rate)
    )
    split_gate = {
        name: split_scores[name] > image_gate_threshold for name in names
    }
    print(
        "Dinomaly gate: "
        f"known_test_reject={split_gate['known_test'].mean():.6f}, "
        f"unknown_recall={split_gate['unknown_test'].mean():.6f}",
        flush=True,
    )

    print("RDP+RCR-v2 baseline", flush=True)
    baseline: Dict[str, Dict[str, object]] = {}
    for name in names:
        prediction = predict_rdp_rcr(
            split_wafers[name], stack, args.rdp_batch_size, device
        )
        baseline[name] = metrics_for(
            prediction,
            y,
            split[name],
            stack["active_indices"],
            stack["class_names"],
            stack["multilabel_metrics"],
        )
        del prediction

    normal_value, defect_value = raw_values(x)
    strategies: Dict[str, object] = {}
    calibrations: Dict[str, object] = {}
    for mode in ("gate_positive", "all_samples"):
        print(f"known-only cleaning calibration: {mode}", flush=True)
        selected, trials = choose_cleaning_strength(
            mode,
            fractions,
            split_wafers["known_val"],
            split_maps["known_val"],
            split_gate["known_val"],
            y,
            split["known_val"],
            stack,
            normal_value,
            defect_value,
            float(baseline["known_val"]["exact_match"]),
            args.max_known_val_em_drop,
            args.rdp_batch_size,
            device,
        )
        calibrations[mode] = {
            "selected": selected,
            "candidates": trials,
        }
        print(
            f"selected {mode}: mask_fraction="
            f"{selected['requested_mask_fraction']}, "
            f"known_val_EM={selected['known_validation_exact_match']:.6f}",
            flush=True,
        )
        strategies[mode] = evaluate_strategy(
            mode,
            selected,
            split_wafers,
            split_maps,
            split_gate,
            y,
            split,
            stack,
            normal_value,
            defect_value,
            baseline,
            args.rdp_batch_size,
            device,
        )

    baseline_unknown_per_class = baseline["unknown_test"]["per_class"]
    dominant_false_class = max(
        stack["class_names"],
        key=lambda name: baseline_unknown_per_class[name]["false_positive"],
    )
    report = {
        "configuration": {
            "method": "dinomaly_map_guided_rdp_rcr_v2_cleaning",
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_index,
            "known_classes": stack["class_names"],
            "data": str(Path(args.data).expanduser().resolve()),
            "dinomaly_checkpoint": str(dinomaly_path),
            "rdp_checkpoint": str(stack["rdp_checkpoint"]),
            "rcr_checkpoint": str(rcr_path),
            "dinomaly_input_encoding": input_encoding,
            "dinomaly_image_size": image_size,
            "dinomaly_top_fraction": top_fraction,
            "known_reject_rate": args.known_reject_rate,
            "mask_fraction_candidates": fractions,
            "max_known_validation_exact_match_drop": args.max_known_val_em_drop,
            "normal_die_value": normal_value,
            "defect_die_value": defect_value,
            "seed": args.seed,
        },
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "dinomaly_unknown_gate": {
            "threshold_from_known_validation": image_gate_threshold,
            "known_validation_reject_rate": float(split_gate["known_val"].mean()),
            "known_test_reject_rate": float(split_gate["known_test"].mean()),
            "unknown_test_recall": float(split_gate["unknown_test"].mean()),
        },
        "baseline_rdp_rcr_v2": baseline,
        "known_only_cleaning_calibration": calibrations,
        "cleaning_strategies": strategies,
        "dominant_baseline_false_known_class_on_unknown_test": dominant_false_class,
        "notes": {
            "heldout_used_for_training_selection_or_calibration": False,
            "gate_positive": (
                "Deployment-consistent: clean only wafers accepted by the "
                "Dinomaly unknown gate."
            ),
            "all_samples": (
                "Localization diagnostic: apply the known-calibrated pixel rule "
                "without the wafer gate; do not silently choose it using held-out results."
            ),
            "cleaning_scope": (
                "Only existing defect-valued dies may be changed; background and "
                "normal dies are never synthesized."
            ),
            "output_files": "one metrics JSON; no checkpoint, map, or feature cache",
        },
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\nsummary")
    print(
        "baseline: "
        f"known_EM={baseline['known_test']['exact_match']:.6f} "
        f"unknown_EM={baseline['unknown_test']['exact_match']:.6f} "
        f"unknown_F1={baseline['unknown_test']['micro_f1']:.6f} "
        f"FP/sample={baseline['unknown_test']['mean_false_positives_per_sample']:.6f}"
    )
    for mode, result in strategies.items():
        known_metrics = result["splits"]["known_test"]["metrics"]
        unknown_metrics = result["splits"]["unknown_test"]["metrics"]
        print(
            f"{mode}: known_EM={known_metrics['exact_match']:.6f} "
            f"unknown_EM={unknown_metrics['exact_match']:.6f} "
            f"unknown_F1={unknown_metrics['micro_f1']:.6f} "
            f"FP/sample={unknown_metrics['mean_false_positives_per_sample']:.6f} "
            f"pure_false={unknown_metrics['pure_unknown_false_known_rate']}"
        )
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()
