#!/usr/bin/env python3
"""Full evaluation for an RDP-conditioned Dinomaly checkpoint.

Place this file beside ``train_eval_rdp_conditioned_dinomaly.py`` in the
Dinomaly repository root. It does not train or modify the checkpoint. It writes
only ``full_evaluation_metrics.json`` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from train_eval_dinomaly_atomic_open_set import (
    FULL_CLASSES,
    MixedWM38DinomalyDataset,
    build_model,
    load_arrays,
    make_loader,
    make_open_set_split,
    set_seed,
    torch_load,
)
from train_eval_rdp_conditioned_dinomaly import (
    RDPConditionedDinomaly,
    collect_scores,
    detection_metrics,
    forward_rdp_logits,
    load_frozen_rdp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full evaluation of RDP-conditioned Dinomaly"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        default=None,
        help="Override the dataset path stored in the checkpoint",
    )
    parser.add_argument(
        "--unknown-class",
        choices=FULL_CLASSES,
        default=None,
        help="Optional consistency check; normally read from checkpoint",
    )
    parser.add_argument("--rdp-repo", default=None)
    parser.add_argument("--rdp-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--known-reject-rate", type=float, default=None)
    parser.add_argument("--rdp-label-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_rdp_predictions(
    rdp_model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
    active_indices: Sequence[int],
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_array = np.asarray(indices, dtype=np.int64)
    probability_parts = []
    for start in range(0, len(index_array), batch_size):
        batch_indices = torch.as_tensor(
            index_array[start : start + batch_size], dtype=torch.long
        )
        raw = x[batch_indices].unsqueeze(1).to(device, non_blocking=True)
        logits = forward_rdp_logits(rdp_model, raw)
        probability_parts.append(
            torch.sigmoid(logits[:, active_indices]).cpu().numpy()
        )
    probabilities = np.concatenate(probability_parts, axis=0)
    predictions = probabilities >= threshold
    targets = (
        y[torch.as_tensor(index_array, dtype=torch.long)][:, active_indices]
        .cpu()
        .numpy()
        > 0.5
    )
    return probabilities, predictions, targets


def multilabel_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    predictions = predictions.astype(bool)
    targets = targets.astype(bool)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    false_positive = predictions & ~targets
    false_negative = ~predictions & targets
    return {
        "samples": int(len(targets)),
        "exact_match": float(np.mean(np.all(predictions == targets, axis=1))),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "hamming_accuracy": float(np.mean(predictions == targets)),
        "mean_true_component_count": float(targets.sum(axis=1).mean()),
        "mean_predicted_component_count": float(
            predictions.sum(axis=1).mean()
        ),
        "mean_false_positives_per_sample": float(
            false_positive.sum(axis=1).mean()
        ),
        "mean_false_negatives_per_sample": float(
            false_negative.sum(axis=1).mean()
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "positive_samples": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    configuration = checkpoint["configuration"]

    unknown_class = str(configuration["unknown_class"])
    if args.unknown_class is not None and args.unknown_class != unknown_class:
        raise ValueError(
            f"Checkpoint holds out {unknown_class}, not {args.unknown_class}"
        )
    unknown_index = FULL_CLASSES.index(unknown_class)
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    stored_active = [
        int(index) for index in configuration["active_known_indices"]
    ]
    if stored_active != active_indices:
        raise ValueError(
            f"Checkpoint active indices {stored_active} != {active_indices}"
        )
    known_classes = [FULL_CLASSES[index] for index in active_indices]

    data_path = args.data or str(configuration["data"])
    rdp_repo = Path(
        args.rdp_repo or str(configuration["rdp_repo"])
    ).resolve()
    rdp_checkpoint = Path(
        args.rdp_checkpoint or str(configuration["rdp_checkpoint"])
    ).resolve()
    top_fraction = float(
        args.top_fraction
        if args.top_fraction is not None
        else configuration["top_fraction"]
    )
    known_reject_rate = float(
        args.known_reject_rate
        if args.known_reject_rate is not None
        else configuration["known_reject_rate"]
    )
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0,1]")

    seed = int(configuration["seed"])
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = load_arrays(data_path)
    split = make_open_set_split(
        y,
        unknown_index,
        0.70,
        0.15,
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
                x,
                y,
                indices,
                image_size=int(configuration["image_size"]),
                # input_encoding="gray_repeat",
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=seed + offset,
        )
        for offset, (name, indices) in enumerate(evaluation_indices.items())
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

    rdp_model, loaded_active = load_frozen_rdp(
        rdp_repo, rdp_checkpoint, unknown_class, device
    )
    if loaded_active != active_indices:
        raise ValueError("RDP and conditioned checkpoint class order differs")
    base, bottleneck, decoder, _ = build_model(
        str(configuration["encoder"]),
        int(configuration["decoder_depth"]),
        device,
    )
    feature_dim = bottleneck[0].fc2.out_features
    model = RDPConditionedDinomaly(
        base,
        condition_dim=len(active_indices),
        condition_hidden_dim=int(configuration["condition_hidden_dim"]),
        feature_dim=feature_dim,
        condition_scale=float(configuration["condition_scale"]),
    ).to(device)
    bottleneck.load_state_dict(
        checkpoint["bottleneck_state_dict"], strict=True
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    model.conditioner.load_state_dict(
        checkpoint["conditioner_state_dict"], strict=True
    )
    model.eval()

    score_results = {}
    score_indices = {}
    rdp_diagnostics = {}
    for name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {name}", flush=True)
        scores, indices, _ = collect_scores(
            model,
            rdp_model,
            loaders[name],
            x,
            active_indices,
            device,
            int(configuration["image_size"]),
            top_fraction,
            args.rdp_label_threshold,
        )
        score_results[name] = scores
        score_indices[name] = indices
        _, predictions, targets = collect_rdp_predictions(
            rdp_model,
            x,
            y,
            indices,
            active_indices,
            args.batch_size,
            args.rdp_label_threshold,
            device,
        )
        rdp_diagnostics[name] = multilabel_metrics(
            predictions, targets, known_classes
        )

    unknown_indices = score_indices["unknown_test"]
    unknown_targets = (
        y[torch.as_tensor(unknown_indices, dtype=torch.long)][:, active_indices]
        .cpu()
        .numpy()
    )
    pure_mask = unknown_targets.sum(axis=1) == 0
    mixed_mask = ~pure_mask
    known_val_scores = score_results["known_val"]
    known_test_scores = score_results["known_test"]
    unknown_scores = score_results["unknown_test"]

    detection = {
        "all_unknown": {
            "samples": int(len(unknown_scores)),
            "metrics": detection_metrics(
                known_val_scores,
                known_test_scores,
                unknown_scores,
                known_reject_rate,
            ),
        }
    }
    if pure_mask.any():
        detection["pure_unknown_only"] = {
            "samples": int(pure_mask.sum()),
            "metrics": detection_metrics(
                known_val_scores,
                known_test_scores,
                unknown_scores[pure_mask],
                known_reject_rate,
            ),
        }
    if mixed_mask.any():
        detection["mixed_known_plus_unknown"] = {
            "samples": int(mixed_mask.sum()),
            "metrics": detection_metrics(
                known_val_scores,
                known_test_scores,
                unknown_scores[mixed_mask],
                known_reject_rate,
            ),
        }

    result = {
        "configuration": {
            "checkpoint": str(checkpoint_path),
            "data": data_path,
            "unknown_class": unknown_class,
            "known_classes": known_classes,
            "rdp_checkpoint": str(rdp_checkpoint),
            "device": str(device),
            "top_fraction": top_fraction,
            "known_reject_rate": known_reject_rate,
            "rdp_label_threshold": args.rdp_label_threshold,
        },
        "full_sample_counts": {
            name: int(len(indices))
            for name, indices in evaluation_indices.items()
        },
        "unknown_detection": detection,
        "rdp_known_component_diagnostics": rdp_diagnostics,
        "notes": {
            "threshold_source": "full known validation set only",
            "checkpoint_selection_used_unknown_labels": False,
            "rdp_exact_match_is_strict": (
                "one missed or extra known component makes the whole sample "
                "incorrect"
            ),
        },
    }
    output_path = Path(
        args.output
        or checkpoint_path.parent / "full_evaluation_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for subset_name in ("all_unknown", "pure_unknown_only", "mixed_known_plus_unknown"):
        if subset_name in detection:
            metrics = detection[subset_name]["metrics"]
            print(
                f"{subset_name}: AUROC={metrics['auroc']:.6f} "
                f"recall={metrics['unknown_recall_at_threshold']:.6f}",
                flush=True,
            )
    unknown_rdp = rdp_diagnostics["unknown_test"]
    print(
        "RDP on unknown_test: "
        f"exact={unknown_rdp['exact_match']:.6f} "
        f"micro_f1={unknown_rdp['micro_f1']:.6f} "
        f"macro_f1={unknown_rdp['macro_f1']:.6f}",
        flush=True,
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
