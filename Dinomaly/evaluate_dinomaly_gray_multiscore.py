#!/usr/bin/env python3
"""Compare image-level scores for a trained gray-repeat Dinomaly gate.

Place this file in the Dinomaly repository root beside
``train_eval_dinomaly_atomic_open_set.py``. The model is evaluated once per
wafer; all score variants are derived from the same anomaly map. No training,
heatmaps, or per-sample predictions are saved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_eval_dinomaly_atomic_open_set import (
    FULL_CLASSES,
    MixedWM38DinomalyDataset,
    build_model,
    cal_anomaly_maps,
    deterministic_subset,
    evaluate_scores,
    get_gaussian_kernel,
    load_arrays,
    make_loader,
    make_open_set_split,
    set_seed,
    torch_load,
)


TOP_FRACTIONS: Tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gray-repeat Dinomaly anomaly-map scores"
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
        "--calibration-n",
        type=int,
        default=0,
        help="Known-validation count; <=0 uses the full split",
    )
    parser.add_argument(
        "--known-eval-n",
        type=int,
        default=0,
        help="Known-test count; <=0 uses the full split",
    )
    parser.add_argument(
        "--unknown-eval-n",
        type=int,
        default=0,
        help="Unknown-test count; <=0 uses the full split",
    )
    parser.add_argument(
        "--known-reject-rate",
        type=float,
        default=None,
        help="Default: value stored in checkpoint, normally 0.05",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep disabled because FP16 LinearAttention2 can produce NaN",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <checkpoint directory>/multiscore_metrics.json",
    )
    return parser.parse_args()


@torch.inference_mode()
def collect_multiscores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    amp: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    parts: Dict[str, list[np.ndarray]] = {
        "maximum": [],
        "global_mean": [],
        **{f"top_{fraction:g}_mean": [] for fraction in TOP_FRACTIONS},
    }
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

        max_keep = max(
            1,
            int(math.ceil(flattened.shape[1] * max(TOP_FRACTIONS))),
        )
        top_values = torch.topk(
            flattened, k=max_keep, dim=1, largest=True, sorted=True
        ).values
        cumulative = torch.cumsum(top_values, dim=1)

        batch_scores: Dict[str, torch.Tensor] = {
            "maximum": top_values[:, 0],
            "global_mean": flattened.mean(dim=1),
        }
        for fraction in TOP_FRACTIONS:
            keep = max(
                1,
                int(math.ceil(flattened.shape[1] * fraction)),
            )
            batch_scores[f"top_{fraction:g}_mean"] = (
                cumulative[:, keep - 1] / float(keep)
            )

        for name, values in batch_scores.items():
            if not torch.isfinite(values).all():
                raise FloatingPointError(
                    f"Non-finite {name} score; run with --no-amp"
                )
            parts[name].append(values.cpu().numpy())
        index_parts.append(indices.numpy())

    return (
        {name: np.concatenate(values) for name, values in parts.items()},
        np.concatenate(index_parts),
    )


def evaluate_one_score(
    calibration: np.ndarray,
    known: np.ndarray,
    unknown: np.ndarray,
    pure_mask: np.ndarray,
    known_reject_rate: float,
) -> Dict[str, object]:
    mixed_mask = ~pure_mask
    result: Dict[str, object] = {
        "all_unknown": {
            "samples": int(len(unknown)),
            "metrics": evaluate_scores(
                calibration, known, unknown, known_reject_rate
            ),
        }
    }
    if pure_mask.any():
        result["pure_unknown_only"] = {
            "samples": int(pure_mask.sum()),
            "metrics": evaluate_scores(
                calibration,
                known,
                unknown[pure_mask],
                known_reject_rate,
            ),
        }
    if mixed_mask.any():
        result["mixed_known_plus_unknown"] = {
            "samples": int(mixed_mask.sum()),
            "metrics": evaluate_scores(
                calibration,
                known,
                unknown[mixed_mask],
                known_reject_rate,
            ),
        }
    return result


def select_indices(
    split: Dict[str, np.ndarray],
    calibration_n: int,
    known_eval_n: int,
    unknown_eval_n: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    return {
        "known_val": deterministic_subset(
            split["known_val"], calibration_n, seed + 1
        ),
        "known_test": deterministic_subset(
            split["known_test"], known_eval_n, seed + 2
        ),
        "unknown_test": deterministic_subset(
            split["unknown_test"], unknown_eval_n, seed + 3
        ),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    configuration = checkpoint.get("configuration", {})

    input_encoding = configuration.get("input_encoding", "gray_repeat")
    if input_encoding != "gray_repeat":
        raise ValueError(
            f"This evaluator requires gray_repeat, checkpoint uses {input_encoding}"
        )
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
        raise ValueError("Provide --data; checkpoint has no data path")

    encoder_name = configuration.get("encoder", "dinov2reg_vit_small_14")
    image_size = int(configuration.get("image_size", 392))
    decoder_depth = int(configuration.get("decoder_depth", 8))
    seed = int(configuration.get("seed", 3529))
    known_reject_rate = (
        float(args.known_reject_rate)
        if args.known_reject_rate is not None
        else float(configuration.get("known_reject_rate", 0.05))
    )
    if not 0.0 < known_reject_rate < 1.0:
        raise ValueError("known reject rate must be in (0,1)")

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
    selected = select_indices(
        split,
        args.calibration_n,
        args.known_eval_n,
        args.unknown_eval_n,
        seed,
    )
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
        for offset, (name, indices) in enumerate(selected.items(), 1)
    }

    print(f"device: {device}")
    print(f"gray-repeat unknown gate: {unknown_class} held out")
    print(
        "evaluation: "
        + ", ".join(
            f"{name}={len(indices)}" for name, indices in selected.items()
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

    collected: Dict[str, Dict[str, np.ndarray]] = {}
    collected_indices: Dict[str, np.ndarray] = {}
    for split_name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {split_name}", flush=True)
        split_scores, split_indices = collect_multiscores(
            model,
            loaders[split_name],
            device,
            image_size,
            args.amp,
        )
        collected[split_name] = split_scores
        collected_indices[split_name] = split_indices

    unknown_indices = collected_indices["unknown_test"]
    unknown_labels = y[torch.as_tensor(unknown_indices)].cpu().numpy()
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    pure_mask = unknown_labels[:, active_indices].sum(axis=1) == 0

    score_results: Dict[str, object] = {}
    for score_name in collected["known_val"]:
        score_results[score_name] = evaluate_one_score(
            collected["known_val"][score_name],
            collected["known_test"][score_name],
            collected["unknown_test"][score_name],
            pure_mask,
            known_reject_rate,
        )

    diagnostic_ranking = sorted(
        (
            {
                "score": score_name,
                "mixed_auroc": float(
                    score_result["mixed_known_plus_unknown"]["metrics"][
                        "auroc"
                    ]
                ),
                "mixed_unknown_recall": float(
                    score_result["mixed_known_plus_unknown"]["metrics"][
                        "unknown_recall_at_threshold"
                    ]
                ),
                "known_test_reject_rate": float(
                    score_result["mixed_known_plus_unknown"]["metrics"][
                        "known_test_reject_rate"
                    ]
                ),
            }
            for score_name, score_result in score_results.items()
        ),
        key=lambda row: row["mixed_auroc"],
        reverse=True,
    )

    result = {
        "configuration": {
            "unknown_class": unknown_class,
            "unknown_index": unknown_index,
            "checkpoint": str(checkpoint_path),
            "data": str(data_path),
            "input_encoding": input_encoding,
            "encoder": encoder_name,
            "image_size": image_size,
            "batch_size": args.batch_size,
            "amp": bool(args.amp),
            "known_reject_rate": known_reject_rate,
            "top_fractions": list(TOP_FRACTIONS),
            "seed": seed,
        },
        "split_counts": {
            name: int(len(indices)) for name, indices in split.items()
        },
        "used_sample_counts": {
            name: int(len(indices)) for name, indices in selected.items()
        },
        "score_results": score_results,
        "diagnostic_ranking_by_mixed_auroc": diagnostic_ranking,
        "notes": {
            "training_performed": False,
            "model_forward_passes_per_wafer": 1,
            "threshold_source": "known validation only, separately per score",
            "unknown_labels_used_for_training_or_threshold": False,
            "selection_warning": (
                "Ranking uses target held-out labels for diagnosis. Do not "
                "claim the best target-S score as unbiased model selection; "
                "lock the rule across source-validation or all leave-one-out runs."
            ),
        },
    }

    output_path = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent / "multiscore_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("score comparison (mixed known+unknown):")
    for row in diagnostic_ranking:
        print(
            f"  {row['score']:<18} "
            f"AUROC={row['mixed_auroc']:.6f} "
            f"recall={row['mixed_unknown_recall']:.6f} "
            f"known_reject={row['known_test_reject_rate']:.6f}"
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
