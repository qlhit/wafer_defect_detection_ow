#!/usr/bin/env python3
"""Full, evaluation-only assessment of a trained Dinomaly unknown gate.

Place this file in the Dinomaly repository root, beside
``train_eval_dinomaly_atomic_open_set.py``.  It never trains or changes the
checkpoint.  The only output file is one JSON metrics report.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_eval_dinomaly_atomic_open_set import (
    FULL_CLASSES,
    MixedWM38DinomalyDataset,
    build_model,
    evaluate_scores,
    get_gaussian_kernel,
    cal_anomaly_maps,
    load_arrays,
    make_loader,
    make_open_set_split,
    set_seed,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Dinomaly atomic unknown gate"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint/dinomaly_atomic/s_heldout/best_model.pt",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Default: data path stored in the checkpoint",
    )
    parser.add_argument(
        "--unknown-class",
        choices=FULL_CLASSES,
        default=None,
        help="Default: held-out class stored in the checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=None,
        help="Default: value stored in the checkpoint (normally 0.01)",
    )
    parser.add_argument(
        "--known-reject-rate",
        type=float,
        default=None,
        help="Default: value stored in the checkpoint (normally 0.05)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Off by default because LinearAttention2 was unstable in FP16",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <checkpoint directory>/full_evaluation_metrics.json",
    )
    return parser.parse_args()


@torch.inference_mode()
def collect_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    top_fraction: float,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    score_parts = []
    index_parts = []

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
        flattened = anomaly_map.flatten(1)
        keep = max(1, int(math.ceil(flattened.shape[1] * top_fraction)))
        scores = torch.topk(flattened, k=keep, dim=1).values.mean(dim=1)

        if not torch.isfinite(scores).all():
            raise FloatingPointError(
                "Non-finite anomaly score detected. Re-run without --amp."
            )
        score_parts.append(scores.cpu().numpy())
        index_parts.append(indices.numpy())

    return np.concatenate(score_parts), np.concatenate(index_parts)


def subset_metrics(
    calibration_scores: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    mask: np.ndarray,
    known_reject_rate: float,
) -> Dict[str, object]:
    selected = unknown_scores[mask]
    result: Dict[str, object] = {"samples": int(mask.sum())}
    if len(selected) == 0:
        result["metrics"] = None
    else:
        result["metrics"] = evaluate_scores(
            calibration_scores,
            known_scores,
            selected,
            known_reject_rate,
        )
    return result


def score_summary(values: np.ndarray) -> Dict[str, float | None]:
    if len(values) == 0:
        return {
            "mean": None,
            "q50": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def exact_known_projection_groups(
    labels: np.ndarray,
    active_indices: Sequence[int],
) -> Iterable[Tuple[str, np.ndarray]]:
    grouped: Dict[Tuple[int, ...], list[int]] = defaultdict(list)
    for position, row in enumerate(labels):
        key = tuple(int(row[index] > 0.5) for index in active_indices)
        grouped[key].append(position)

    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    for key, positions in ordered:
        names = [
            FULL_CLASSES[index]
            for index, present in zip(active_indices, key)
            if present
        ]
        name = "+".join(names) if names else "pure_unknown"
        mask = np.zeros(len(labels), dtype=bool)
        mask[np.asarray(positions, dtype=np.int64)] = True
        yield name, mask


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    configuration = checkpoint.get("configuration", {})

    unknown_class = args.unknown_class or configuration.get("unknown_class")
    if unknown_class not in FULL_CLASSES:
        raise ValueError("Unknown class is missing or invalid")
    stored_unknown = configuration.get("unknown_class")
    if stored_unknown is not None and stored_unknown != unknown_class:
        raise ValueError(
            f"Checkpoint holds out {stored_unknown}, not {unknown_class}"
        )

    data_path = args.data or configuration.get("data")
    if not data_path:
        raise ValueError("Provide --data; the checkpoint has no data path")
    encoder_name = configuration.get("encoder", "dinov2reg_vit_small_14")
    image_size = int(configuration.get("image_size", 392))
    decoder_depth = int(configuration.get("decoder_depth", 8))
    seed = int(configuration.get("seed", 3529))
    top_fraction = (
        float(args.top_fraction)
        if args.top_fraction is not None
        else float(configuration.get("top_fraction", 0.01))
    )
    known_reject_rate = (
        float(args.known_reject_rate)
        if args.known_reject_rate is not None
        else float(configuration.get("known_reject_rate", 0.05))
    )
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0,1]")
    if not 0.0 < known_reject_rate < 1.0:
        raise ValueError("--known-reject-rate must be in (0,1)")

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unknown_index = FULL_CLASSES.index(unknown_class)

    x, y = load_arrays(str(data_path))
    split = make_open_set_split(
        y,
        unknown_index,
        args.known_train_ratio,
        args.known_val_ratio,
        seed,
    )
    evaluation_indices = {
        "known_val": split["known_val"],
        "known_test": split["known_test"],
        "unknown_test": split["unknown_test"],
    }
    loaders = {
        name: make_loader(
            MixedWM38DinomalyDataset(
                x, y, indices, image_size=image_size
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=seed + offset,
        )
        for offset, (name, indices) in enumerate(evaluation_indices.items(), 1)
    }

    print(f"device: {device}")
    print(f"unknown gate: {unknown_class} held out")
    print(
        "full evaluation: "
        + ", ".join(
            f"{name}={len(indices)}"
            for name, indices in evaluation_indices.items()
        ),
        flush=True,
    )
    print(f"loading checkpoint: {checkpoint_path}", flush=True)
    model, bottleneck, decoder, _trainable = build_model(
        encoder_name, decoder_depth, device
    )
    bottleneck.load_state_dict(
        checkpoint["bottleneck_state_dict"], strict=True
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    model.eval()

    collected: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {name}", flush=True)
        collected[name] = collect_scores(
            model,
            loaders[name],
            device,
            image_size,
            top_fraction,
            args.amp,
        )

    calibration_scores, _ = collected["known_val"]
    known_scores, _ = collected["known_test"]
    unknown_scores, unknown_indices = collected["unknown_test"]
    unknown_labels = y[torch.as_tensor(unknown_indices)].cpu().numpy()
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    known_projection_count = unknown_labels[:, active_indices].sum(axis=1)
    pure_mask = known_projection_count == 0
    mixed_mask = known_projection_count > 0

    all_mask = np.ones(len(unknown_scores), dtype=bool)
    detection: Dict[str, object] = {
        "all_unknown": subset_metrics(
            calibration_scores,
            known_scores,
            unknown_scores,
            all_mask,
            known_reject_rate,
        ),
        "pure_unknown_only": subset_metrics(
            calibration_scores,
            known_scores,
            unknown_scores,
            pure_mask,
            known_reject_rate,
        ),
        "mixed_known_plus_unknown": subset_metrics(
            calibration_scores,
            known_scores,
            unknown_scores,
            mixed_mask,
            known_reject_rate,
        ),
    }

    balanced_rng = np.random.default_rng(seed + 99)
    balanced_n = min(len(known_scores), len(unknown_scores))
    balanced_positions = balanced_rng.choice(
        len(unknown_scores), size=balanced_n, replace=False
    )
    balanced_mask = np.zeros(len(unknown_scores), dtype=bool)
    balanced_mask[balanced_positions] = True

    by_component: Dict[str, object] = {}
    for index in active_indices:
        mask = unknown_labels[:, index] > 0.5
        by_component[FULL_CLASSES[index]] = subset_metrics(
            calibration_scores,
            known_scores,
            unknown_scores,
            mask,
            known_reject_rate,
        )

    by_exact_projection = {
        name: subset_metrics(
            calibration_scores,
            known_scores,
            unknown_scores,
            mask,
            known_reject_rate,
        )
        for name, mask in exact_known_projection_groups(
            unknown_labels, active_indices
        )
    }

    result = {
        "configuration": {
            "unknown_class": unknown_class,
            "unknown_index": unknown_index,
            "checkpoint": str(checkpoint_path),
            "data": str(data_path),
            "encoder": encoder_name,
            "image_size": image_size,
            "decoder_depth": decoder_depth,
            "batch_size": args.batch_size,
            "amp": bool(args.amp),
            "top_fraction": top_fraction,
            "known_reject_rate": known_reject_rate,
            "seed": seed,
        },
        "split_counts": {
            name: int(len(indices))
            for name, indices in split.items()
        },
        "unknown_detection": detection,
        "balanced_reference": subset_metrics(
            calibration_scores,
            known_scores,
            unknown_scores,
            balanced_mask,
            known_reject_rate,
        ),
        "by_coexisting_known_component": by_component,
        "by_exact_known_projection": by_exact_projection,
        "score_distributions": {
            "known_validation": score_summary(calibration_scores),
            "known_test": score_summary(known_scores),
            "all_unknown": score_summary(unknown_scores),
            "pure_unknown": score_summary(unknown_scores[pure_mask]),
            "mixed_known_plus_unknown": score_summary(
                unknown_scores[mixed_mask]
            ),
        },
        "notes": {
            "training_performed": False,
            "unknown_labels_used_for_threshold_or_model_selection": False,
            "threshold_source": "full known validation split only",
            "balanced_reference": (
                "Deterministic unknown subset matched to known-test count; "
                "useful for AUPR comparison with the earlier 1:1 evaluation"
            ),
            "subgroup_aupr_warning": (
                "AUPR changes with subgroup prevalence; compare subgroup "
                "AUROC and recall more directly"
            ),
        },
    }

    output_path = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent / "full_evaluation_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    all_metrics = detection["all_unknown"]["metrics"]
    mixed_metrics = detection["mixed_known_plus_unknown"]["metrics"]
    print(
        "all unknown: "
        f"AUROC={all_metrics['auroc']:.6f}, "
        f"recall={all_metrics['unknown_recall_at_threshold']:.6f}"
    )
    print(
        "mixed known+unknown: "
        f"AUROC={mixed_metrics['auroc']:.6f}, "
        f"recall={mixed_metrics['unknown_recall_at_threshold']:.6f}"
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
