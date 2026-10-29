#!/usr/bin/env python3
"""Leave-one-defect-out open-set baseline for the RDP-Net repository.

Place this file in the repository's ``scripts/`` directory, then run it from
the repository root.  The original RDP-Net files do not need to be edited.

The script keeps the public RDP-Net head at eight outputs in every scenario.
The held-out output is inactive for that scenario: it is excluded from the
loss and from known-component metrics.

The script:
1. removes every held-out-defect-containing sample from training and validation;
2. trains an 8-output RDP-Net with a scenario-specific active-class mask;
3. builds cosine prototypes from known training embeddings;
4. calibrates an unknown threshold using known validation samples only; and
5. evaluates known-only wafers and Scratch-containing unknown wafers.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# Allow this script to run from scripts/ without installing rdpnet.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdpnet.losses import AsymmetricLoss  # noqa: E402
from rdpnet.models.rdpnet import RDPNet  # noqa: E402


# MixedWM38 C2..C9 label order. Scratch is index 6 (not the last index).
FULL_CLASSES: Tuple[str, ...] = ("C", "D", "EL", "ER", "L", "NF", "S", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an RDP-Net leave-one-defect-out open-set baseline"
    )
    parser.add_argument("--data", default="Wafer_Map_Datasets.npz")
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="S")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--known-reject-rate",
        type=float,
        default=0.05,
        help="Target fraction of known validation wafers rejected as unknown",
    )
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--asl", action="store_true")
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/rdp_<unknown>_heldout_open_set",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arrays(path: str) -> Tuple[Tensor, Tensor]:
    raw = np.load(path)
    x = torch.from_numpy(raw["arr_0"].astype(np.float32))
    y = torch.from_numpy(raw["arr_1"].astype(np.float32))
    if x.ndim != 3:
        raise ValueError(f"arr_0 must have shape (N,H,W), got {tuple(x.shape)}")
    if y.ndim != 2 or y.shape[1] != len(FULL_CLASSES):
        raise ValueError(f"arr_1 must have shape (N,8), got {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError("arr_0 and arr_1 have different sample counts")
    return x, y


def label_codes(y: Tensor) -> np.ndarray:
    y_np = y.to(torch.int64).cpu().numpy()
    return np.asarray(["".join(map(str, row.tolist())) for row in y_np])


def make_open_set_split(
    y_full: Tensor,
    unknown_idx: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("known_train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("known_val_ratio must be positive and train+val must be < 1")

    all_indices = np.arange(len(y_full))
    unknown_mask = y_full[:, unknown_idx].cpu().numpy() > 0.5
    known_indices = all_indices[~unknown_mask]
    unknown_indices = all_indices[unknown_mask]

    keep = [i for i in range(y_full.shape[1]) if i != unknown_idx]
    known_codes = label_codes(y_full[known_indices][:, keep])

    train_idx, remainder_idx = train_test_split(
        known_indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=known_codes,
    )

    # val_ratio is expressed relative to all known samples.
    val_fraction_of_remainder = val_ratio / (1.0 - train_ratio)
    remainder_codes = label_codes(y_full[remainder_idx][:, keep])
    val_idx, known_test_idx = train_test_split(
        remainder_idx,
        train_size=val_fraction_of_remainder,
        random_state=seed + 1,
        shuffle=True,
        stratify=remainder_codes,
    )

    return {
        "known_train": np.asarray(train_idx),
        "known_val": np.asarray(val_idx),
        "known_test": np.asarray(known_test_idx),
        "unknown_test": np.asarray(unknown_indices),
    }


def make_loader(
    x: Tensor,
    y_full: Tensor,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    idx = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(x[idx], y_full[idx])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def forward_features(model: RDPNet, x: Tensor) -> Tuple[Tensor, Tensor]:
    """Run the public RDP-Net backbone and return logits plus 512-D features."""
    x = model.conv1(x)
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
    features = torch.flatten(model.avgpool(x), 1)
    logits = model.fc(features)
    return logits, features


@torch.no_grad()
def predict_loader(
    model: RDPNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    model.eval()
    all_logits: List[Tensor] = []
    all_features: List[Tensor] = []
    all_targets: List[Tensor] = []
    for x, y in loader:
        x = x.unsqueeze(1).to(device, non_blocking=True)
        logits, features = forward_features(model, x)
        all_logits.append(logits.cpu())
        all_features.append(features.cpu())
        all_targets.append(y.cpu())
    return (
        torch.cat(all_logits),
        torch.cat(all_features),
        torch.cat(all_targets),
    )


def exact_match_from_logits(
    logits: Tensor,
    targets: Tensor,
    active_indices: Sequence[int],
    threshold: float,
) -> float:
    pred = (torch.sigmoid(logits[:, active_indices]) >= threshold).to(targets.dtype)
    truth = targets[:, active_indices]
    return float((pred == truth).all(dim=1).float().mean().item())


def train_model(
    model: RDPNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_asl: bool,
    active_indices: Sequence[int],
    label_threshold: float,
) -> Tuple[Dict[str, Tensor], List[Dict[str, float]]]:
    criterion: nn.Module
    criterion = AsymmetricLoss() if use_asl else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_exact = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for x, y in bar:
            x = x.unsqueeze(1).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _ = forward_features(model, x)
            # Keep the physical head at eight outputs, but do not supervise the
            # class designated as unknown in this leave-one-defect-out run.
            active_logits = logits[:, active_indices]
            active_targets = y[:, active_indices]
            loss = criterion(active_logits, active_targets)
            if use_asl:
                # The repository ASL returns a sum; normalize it for stable LR.
                loss = loss / active_targets.numel()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = x.shape[0]
            loss_sum += float(loss.item()) * batch_n
            sample_count += batch_n
            bar.set_postfix(loss=f"{loss.item():.4f}")

        val_logits, _, val_targets = predict_loader(model, val_loader, device)
        val_exact = exact_match_from_logits(
            val_logits,
            val_targets,
            active_indices=active_indices,
            threshold=label_threshold,
        )
        train_loss = loss_sum / max(sample_count, 1)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_exact_match": val_exact}
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_exact_match={val_exact:.6f}"
        )

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_state = copy.deepcopy(model.state_dict())

    return best_state, history


def build_prototypes(
    features: Tensor,
    targets: Tensor,
    active_indices: Sequence[int],
) -> Tuple[Tensor, List[str]]:
    features = F.normalize(features, dim=1)
    codes = label_codes(targets[:, active_indices])
    prototypes: List[Tensor] = []
    prototype_codes: List[str] = []
    for code in sorted(np.unique(codes).tolist()):
        mask = torch.from_numpy(codes == code)
        proto = F.normalize(features[mask].mean(dim=0, keepdim=True), dim=1)
        prototypes.append(proto.squeeze(0))
        prototype_codes.append(code)
    return torch.stack(prototypes), prototype_codes


def prototype_unknown_score(features: Tensor, prototypes: Tensor) -> Tensor:
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    max_similarity = features @ prototypes.T
    return 1.0 - max_similarity.max(dim=1).values


def component_metrics(
    logits: Tensor,
    targets: Tensor,
    active_indices: Sequence[int],
    threshold: float,
) -> Dict[str, float]:
    pred = (
        torch.sigmoid(logits[:, active_indices]) >= threshold
    ).to(torch.int64).numpy()
    truth = targets[:, active_indices].to(torch.int64).numpy()
    return {
        "exact_match": float(np.mean(np.all(pred == truth, axis=1))),
        "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    return float(fpr[valid[0]]) if len(valid) else 1.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(
        args.output_dir
        or f"outputs/rdp_{args.unknown_class.lower()}_heldout_open_set"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    x, y_full = load_arrays(args.data)
    unknown_idx = FULL_CLASSES.index(args.unknown_class)
    known_indices = [i for i in range(len(FULL_CLASSES)) if i != unknown_idx]
    known_classes = [FULL_CLASSES[i] for i in known_indices]
    split = make_open_set_split(
        y_full=y_full,
        unknown_idx=unknown_idx,
        train_ratio=args.known_train_ratio,
        val_ratio=args.known_val_ratio,
        seed=args.seed,
    )
    print(
        "split:",
        {name: int(len(indices)) for name, indices in split.items()},
    )
    print("known classes:", known_classes, "unknown class:", args.unknown_class)

    loaders = {
        name: make_loader(
            x=x,
            y_full=y_full,
            indices=indices,
            batch_size=args.batch_size,
            shuffle=(name == "known_train"),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        for name, indices in split.items()
    }

    model = RDPNet(
        output_dim=len(FULL_CLASSES), in_channels=1, apply_sigmoid=False
    ).to(device)
    best_state, history = train_model(
        model=model,
        train_loader=loaders["known_train"],
        val_loader=loaders["known_val"],
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_asl=args.asl,
        active_indices=known_indices,
        label_threshold=args.label_threshold,
    )
    model.load_state_dict(best_state)

    train_logits, train_features, train_targets = predict_loader(
        model, loaders["known_train"], device
    )
    val_logits, val_features, val_targets = predict_loader(
        model, loaders["known_val"], device
    )
    known_logits, known_features, known_targets = predict_loader(
        model, loaders["known_test"], device
    )
    unknown_logits, unknown_features, unknown_targets = predict_loader(
        model, loaders["unknown_test"], device
    )

    prototypes, prototype_codes = build_prototypes(
        train_features, train_targets, active_indices=known_indices
    )
    val_scores = prototype_unknown_score(val_features, prototypes)
    threshold = float(
        torch.quantile(val_scores, 1.0 - args.known_reject_rate).item()
    )
    known_scores = prototype_unknown_score(known_features, prototypes).numpy()
    unknown_scores = prototype_unknown_score(unknown_features, prototypes).numpy()

    detection_labels = np.concatenate(
        [np.zeros_like(known_scores), np.ones_like(unknown_scores)]
    )
    detection_scores = np.concatenate([known_scores, unknown_scores])
    known_accept_rate = float(np.mean(known_scores <= threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))
    detection = {
        "auroc": float(roc_auc_score(detection_labels, detection_scores)),
        "aupr": float(average_precision_score(detection_labels, detection_scores)),
        "fpr95": fpr_at_tpr95(detection_labels, detection_scores),
        "threshold": threshold,
        "known_accept_rate": known_accept_rate,
        "unknown_recall": unknown_recall,
        "balanced_accuracy_at_threshold": 0.5 * (known_accept_rate + unknown_recall),
    }

    metrics = {
        "configuration": {
            "full_class_order": list(FULL_CLASSES),
            "known_classes": known_classes,
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_idx,
            "label_threshold": args.label_threshold,
            "known_reject_rate": args.known_reject_rate,
            "use_asl": args.asl,
            "seed": args.seed,
        },
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "known_test_components": component_metrics(
            known_logits,
            known_targets,
            active_indices=known_indices,
            threshold=args.label_threshold,
        ),
        "unknown_test_known_components": component_metrics(
            unknown_logits,
            unknown_targets,
            active_indices=known_indices,
            threshold=args.label_threshold,
        ),
        "unknown_detection": detection,
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "output_classes": list(FULL_CLASSES),
        "active_known_indices": known_indices,
        "inactive_unknown_index": unknown_idx,
        "known_classes": known_classes,
        "unknown_class": args.unknown_class,
        "full_class_order": list(FULL_CLASSES),
        "prototypes": prototypes,
        "prototype_codes": prototype_codes,
        "unknown_threshold": threshold,
        "label_threshold": args.label_threshold,
    }
    torch.save(checkpoint, output_dir / "best_open_set.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved:", output_dir / "best_open_set.pt")
    print("saved:", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
