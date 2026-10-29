#!/usr/bin/env python3
"""Diagnose RDP known-component predictions on held-out-unknown wafers.

This script is deliberately independent of Dinomaly: it loads only MixedWM38
and an RDP leave-one-out checkpoint, evaluates every wafer containing the held-
out class, prints per-class false-positive statistics, and saves one JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


FULL_CLASSES: Tuple[str, ...] = ("C", "D", "EL", "ER", "L", "NF", "S", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RDP-only held-out-unknown projection diagnosis"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument("--rdp-repo", default="../rdp_net_repro")
    parser.add_argument("--rdp-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_arrays(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = np.load(path)
    if "arr_0" not in raw:
        raise ValueError(f"{path} has no arr_0")
    x = torch.from_numpy(raw["arr_0"].astype(np.float32))
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"arr_0 must be [N,H,W], got {tuple(x.shape)}")
    label_key = next(
        (
            key
            for key in ("arr_1", "arr_2")
            if key in raw
            and raw[key].ndim == 2
            and raw[key].shape[1] == len(FULL_CLASSES)
        ),
        None,
    )
    if label_key is None:
        raise ValueError("No [N,8] label array found")
    y = torch.from_numpy(raw[label_key].astype(np.float32))
    return x, (y > 0.5).float()


def load_rdp(
    repo: Path,
    checkpoint_path: Path,
    unknown_class: str,
    device: torch.device,
):
    source = (repo / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"RDP source directory not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from rdpnet.models.rdpnet import RDPNet

    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    stored_unknown = checkpoint.get("unknown_class")
    if stored_unknown is not None and stored_unknown != unknown_class:
        raise ValueError(
            f"RDP checkpoint holds out {stored_unknown}, not {unknown_class}"
        )
    unknown_index = FULL_CLASSES.index(unknown_class)
    expected_active = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    active_indices = [
        int(index)
        for index in checkpoint.get("active_known_indices", expected_active)
    ]
    if active_indices != expected_active:
        raise ValueError(
            f"Expected active indices {expected_active}, got {active_indices}"
        )
    model = RDPNet(
        output_dim=len(FULL_CLASSES), in_channels=1, apply_sigmoid=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, active_indices


def forward_logits(model: torch.nn.Module, wafer: torch.Tensor) -> torch.Tensor:
    x = model.conv1(wafer)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.block_1_1(x)
    x = model.block_1_2(x)
    x = model.block_2_1(x)
    x = model.block_2_2(x)
    x = model.block_3_1(x)
    x = model.block_3_2(x)
    x = model.block_4_1(x)
    x = model.block_4_2(x)
    x = torch.flatten(model.avgpool(x), 1)
    return model.fc(x)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probability_parts = []
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        wafer = x[batch_indices].unsqueeze(1).to(device, non_blocking=True)
        probability_parts.append(
            torch.sigmoid(forward_logits(model, wafer)[:, active_indices])
            .cpu()
            .numpy()
        )
    return np.concatenate(probability_parts, axis=0)


def safe_quantile(values: np.ndarray, q: float):
    if len(values) == 0:
        return None
    return float(np.quantile(values, q))


def diagnostic_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> Dict[str, object]:
    predictions = probabilities >= threshold
    targets = targets.astype(bool)
    tp = np.sum(predictions & targets, axis=0)
    fp = np.sum(predictions & ~targets, axis=0)
    fn = np.sum(~predictions & targets, axis=0)
    tn = np.sum(~predictions & ~targets, axis=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="micro", zero_division=0
    )
    supported = support > 0
    supported_macro_f1 = (
        float(np.mean(f1[supported])) if supported.any() else None
    )

    false_positive_sets = []
    for prediction, target in zip(predictions, targets):
        names = [
            class_names[index]
            for index in range(len(class_names))
            if prediction[index] and not target[index]
        ]
        false_positive_sets.append("+".join(names) if names else "NONE")
    top_false_positive_patterns = [
        {"pattern": pattern, "samples": int(count)}
        for pattern, count in Counter(false_positive_sets).most_common(20)
    ]

    per_class = {}
    for index, name in enumerate(class_names):
        present_values = probabilities[targets[:, index], index]
        absent_values = probabilities[~targets[:, index], index]
        negative_count = int(fp[index] + tn[index])
        per_class[name] = {
            "positive_samples": int(support[index]),
            "negative_samples": negative_count,
            "predicted_positive_samples": int(tp[index] + fp[index]),
            "true_positive": int(tp[index]),
            "false_positive": int(fp[index]),
            "false_negative": int(fn[index]),
            "true_negative": int(tn[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "false_positive_rate": (
                float(fp[index] / negative_count) if negative_count else None
            ),
            "probability_when_present": {
                "mean": (
                    float(present_values.mean())
                    if len(present_values)
                    else None
                ),
                "q05": safe_quantile(present_values, 0.05),
                "q50": safe_quantile(present_values, 0.50),
            },
            "probability_when_absent": {
                "mean": float(absent_values.mean()) if len(absent_values) else None,
                "q50": safe_quantile(absent_values, 0.50),
                "q95": safe_quantile(absent_values, 0.95),
                "q99": safe_quantile(absent_values, 0.99),
            },
        }

    return {
        "samples": int(len(targets)),
        "threshold": float(threshold),
        "exact_match": float(np.mean(np.all(predictions == targets, axis=1))),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "supported_class_macro_f1": supported_macro_f1,
        "supported_classes": [
            class_names[index]
            for index in range(len(class_names))
            if supported[index]
        ],
        "zero_positive_support_classes": [
            class_names[index]
            for index in range(len(class_names))
            if not supported[index]
        ],
        "mean_true_component_count": float(targets.sum(axis=1).mean()),
        "mean_predicted_component_count": float(
            predictions.sum(axis=1).mean()
        ),
        "mean_false_positives_per_sample": float(fp.sum() / len(targets)),
        "mean_false_negatives_per_sample": float(fn.sum() / len(targets)),
        "top_false_positive_patterns": top_false_positive_patterns,
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.label_threshold < 1.0:
        raise ValueError("--label-threshold must be in (0,1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo = Path(args.rdp_repo).resolve()
    checkpoint_path = (
        Path(args.rdp_checkpoint).resolve()
        if args.rdp_checkpoint
        else repo
        / "checkpoint"
        / "rdp_leave_one_out"
        / f"{args.unknown_class.lower()}_heldout"
        / "best_open_set.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RDP checkpoint not found: {checkpoint_path}")

    x, y = load_arrays(args.data)
    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    unknown_indices = np.flatnonzero(
        y[:, unknown_index].cpu().numpy() > 0.5
    )
    model, loaded_active = load_rdp(
        repo, checkpoint_path, args.unknown_class, device
    )
    if loaded_active != active_indices:
        raise ValueError("Unexpected RDP active class order")
    class_names = [FULL_CLASSES[index] for index in active_indices]

    print(f"device: {device}")
    print(
        f"RDP diagnosis: unknown={args.unknown_class}, "
        f"samples={len(unknown_indices)}",
        flush=True,
    )
    probabilities = predict(
        model,
        x,
        unknown_indices,
        active_indices,
        args.batch_size,
        device,
    )
    targets = (
        y[torch.as_tensor(unknown_indices, dtype=torch.long)][:, active_indices]
        .cpu()
        .numpy()
        > 0.5
    )
    metrics = diagnostic_metrics(
        probabilities, targets, class_names, args.label_threshold
    )
    result = {
        "configuration": {
            "data": args.data,
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_index,
            "known_classes": class_names,
            "rdp_checkpoint": str(checkpoint_path),
            "device": str(device),
            "label_threshold": args.label_threshold,
        },
        "unknown_test_known_projection": metrics,
    }
    output_path = Path(
        args.output
        or "checkpoint/rdp_conditioned_dinomaly/"
        f"{args.unknown_class.lower()}_heldout/"
        "rdp_projection_diagnostics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"exact={metrics['exact_match']:.6f} "
        f"micro_f1={metrics['micro_f1']:.6f} "
        f"supported_macro_f1={metrics['supported_class_macro_f1']:.6f}",
        flush=True,
    )
    print("class  support  predicted  FP  FN  FPR", flush=True)
    for name, values in metrics["per_class"].items():
        fpr = values["false_positive_rate"]
        fpr_text = "NA" if fpr is None else f"{fpr:.6f}"
        print(
            f"{name:>5}  {values['positive_samples']:>7}  "
            f"{values['predicted_positive_samples']:>9}  "
            f"{values['false_positive']:>5}  "
            f"{values['false_negative']:>5}  {fpr_text}",
            flush=True,
        )
    print("top false-positive patterns:", flush=True)
    for item in metrics["top_false_positive_patterns"][:10]:
        print(f"  {item['pattern']}: {item['samples']}", flush=True)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
