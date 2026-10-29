#!/usr/bin/env python3
"""Dinomaly baseline for atomic leave-one-defect-out MixedWM38.

Place this file in the root of the official Dinomaly repository.  It directly
reads MixedWM38.npz, trains only on wafers not containing the held-out atomic
class, calibrates with known validation data, and evaluates known_test versus
all wafers containing the held-out class.

Outputs only:
  - best_model.pt  (trainable bottleneck/decoder weights; encoder stays frozen)
  - metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from dinov1.utils import trunc_normal_
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import (
    Block as VitBlock,
    LinearAttention2,
    bMlp,
)
from optimizers import StableAdamW
from utils import (
    WarmCosineScheduler,
    cal_anomaly_maps,
    get_gaussian_kernel,
    global_cosine,
    global_cosine_hm_percent,
)


FULL_CLASSES: Tuple[str, ...] = ("C", "D", "EL", "ER", "L", "NF", "S", "R")
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dinomaly atomic open-set baseline on MixedWM38"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="S")
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3529)

    parser.add_argument(
        "--encoder",
        choices=(
            "dinov2reg_vit_small_14",
            "dinov2reg_vit_base_14",
            "dinov2reg_vit_large_14",
        ),
        default="dinov2reg_vit_small_14",
        help="Use small for the first diagnostic; base matches the paper default",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=392,
        help="Full-wafer resize. Must be divisible by the encoder patch size (14)",
    )
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--total-iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--final-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--hard-mining-warmup", type=int, default=1000)
    parser.add_argument("--hard-mining-final-p", type=float, default=0.9)
    parser.add_argument("--hard-mining-easy-gradient", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--calibration-n",
        type=int,
        default=1000,
        help="Known-validation subset size; <=0 uses all",
    )
    parser.add_argument(
        "--known-eval-n", type=int, default=1000, help="<=0 uses all"
    )
    parser.add_argument(
        "--unknown-eval-n", type=int, default=1000, help="<=0 uses all"
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.01,
        help="Mean the highest anomaly-map fraction for image scoring",
    )
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: checkpoint/dinomaly_atomic/<unknown>_heldout",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_arrays(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = np.load(path)
    if "arr_0" not in raw:
        raise ValueError(f"{path} has no arr_0 wafer array")
    x = torch.from_numpy(raw["arr_0"].astype(np.float32))
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"arr_0 must be [N,H,W], got {tuple(x.shape)}")

    label_key = None
    for key in ("arr_1", "arr_2"):
        if (
            key in raw
            and raw[key].ndim == 2
            and raw[key].shape[1] == len(FULL_CLASSES)
        ):
            label_key = key
            break
    if label_key is None:
        shapes = {key: tuple(raw[key].shape) for key in raw.files}
        raise ValueError(f"No [N,8] label array found: {shapes}")
    y = torch.from_numpy(raw[label_key].astype(np.float32))
    if len(x) != len(y):
        raise ValueError("Wafer and label counts differ")
    return x, (y > 0.5).float()


def label_codes(y: torch.Tensor) -> np.ndarray:
    values = y.to(torch.int64).cpu().numpy()
    return np.asarray(["".join(map(str, row.tolist())) for row in values])


def make_open_set_split(
    y: torch.Tensor,
    unknown_idx: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("known-train-ratio must be in (0,1)")
    if not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("known-val-ratio must be positive and train+val < 1")
    indices = np.arange(len(y))
    unknown_mask = y[:, unknown_idx].cpu().numpy() > 0.5
    known_indices = indices[~unknown_mask]
    unknown_indices = indices[unknown_mask]
    active = [index for index in range(len(FULL_CLASSES)) if index != unknown_idx]

    codes = label_codes(y[known_indices][:, active])
    train_indices, remainder = train_test_split(
        known_indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=codes,
    )
    remainder_codes = label_codes(y[remainder][:, active])
    val_fraction = val_ratio / (1.0 - train_ratio)
    val_indices, test_indices = train_test_split(
        remainder,
        train_size=val_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=remainder_codes,
    )
    return {
        "known_train": np.asarray(train_indices),
        "known_val": np.asarray(val_indices),
        "known_test": np.asarray(test_indices),
        "unknown_test": np.asarray(unknown_indices),
    }


def deterministic_subset(
    indices: Sequence[int], size: int, seed: int
) -> np.ndarray:
    indices = np.asarray(indices)
    if size <= 0 or size >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=size, replace=False)


class MixedWM38DinomalyDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        indices: Sequence[int],
        image_size: int,
    ) -> None:
        self.x = x
        self.y = y
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.image_size = int(image_size)
        self.raw_scale = 2.0 if float(x.max().item()) > 1.0 else 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        wafer = (self.x[index] / self.raw_scale).clamp(0.0, 1.0)
        wafer = wafer.unsqueeze(0).unsqueeze(0)
        wafer = F.interpolate(
            wafer,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )[0]
        image = wafer.repeat(3, 1, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image, self.y[index], index


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_model(encoder_name: str, decoder_depth: int, device: torch.device):
    encoder = vit_encoder.load(encoder_name)
    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")

    bottleneck = nn.ModuleList(
        [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)]
    )
    decoder = nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-8),
                attn=LinearAttention2,
            )
            for _ in range(decoder_depth)
        ]
    )
    fuse_encoder = [list(range(0, 4)), list(range(4, 8))]
    fuse_decoder = [list(range(0, 4)), list(range(4, 8))]
    if decoder_depth != 8:
        midpoint = decoder_depth // 2
        fuse_decoder = [
            list(range(0, midpoint)),
            list(range(midpoint, decoder_depth)),
        ]
        if not all(fuse_decoder):
            raise ValueError("decoder-depth must be at least 2")

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_encoder,
        fuse_layer_decoder=fuse_decoder,
    ).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    trainable = nn.ModuleList([bottleneck, decoder])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
    return model, bottleneck, decoder, trainable


@torch.inference_mode()
def known_validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for image, _labels, _indices in loader:
        image = image.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            encoded, decoded = model(image)
            loss = global_cosine(encoded, decoded)
        total += float(loss.item()) * len(image)
        count += len(image)
    return total / max(1, count)


@torch.inference_mode()
def collect_anomaly_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    top_fraction: float,
    amp_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    score_parts, index_parts = [], []
    for image, _labels, indices in loader:
        image = image.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            encoded, decoded = model(image)
            anomaly_map, _ = cal_anomaly_maps(
                encoded, decoded, out_size=image_size
            )
            anomaly_map = gaussian(anomaly_map.float())
        flattened = anomaly_map.flatten(1)
        keep = max(1, int(math.ceil(flattened.shape[1] * top_fraction)))
        scores = torch.topk(flattened, k=keep, dim=1).values.mean(dim=1)
        score_parts.append(scores.cpu().numpy())
        index_parts.append(indices.numpy())
    return np.concatenate(score_parts), np.concatenate(index_parts)


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    candidates = np.flatnonzero(tpr >= 0.95)
    return float(fpr[candidates[0]]) if len(candidates) else 1.0


def evaluate_scores(
    calibration_scores: np.ndarray,
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
    threshold = float(
        np.quantile(calibration_scores, 1.0 - known_reject_rate)
    )
    known_reject = float(np.mean(known_scores > threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr_at_tpr95(labels, scores),
        "threshold_from_known_validation": threshold,
        "known_test_reject_rate": known_reject,
        "unknown_recall_at_threshold": unknown_recall,
        "balanced_accuracy_at_threshold": float(
            0.5 * ((1.0 - known_reject) + unknown_recall)
        ),
        "known_mean": float(np.mean(known_scores)),
        "unknown_mean": float(np.mean(unknown_scores)),
    }


def save_checkpoint(
    path: Path,
    bottleneck: nn.ModuleList,
    decoder: nn.ModuleList,
    configuration: Dict[str, object],
    iteration: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "bottleneck_state_dict": bottleneck.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "configuration": configuration,
            "best_iteration": int(iteration),
            "best_known_validation_loss": float(val_loss),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.image_size % 14 != 0:
        raise ValueError("--image-size must be divisible by patch size 14")
    if args.total_iters < 1 or args.batch_size < 1:
        raise ValueError("total-iters and batch-size must be positive")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top-fraction must be in (0,1]")
    if args.decoder_depth < 2:
        raise ValueError("decoder-depth must be at least 2")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    unknown_idx = FULL_CLASSES.index(args.unknown_class)
    output_dir = Path(
        args.output_dir
        or f"checkpoint/dinomaly_atomic/{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"

    x, y = load_arrays(args.data)
    split = make_open_set_split(
        y,
        unknown_idx,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    selected = {
        "known_train": split["known_train"],
        "known_val": deterministic_subset(
            split["known_val"], args.calibration_n, args.seed + 1
        ),
        "known_test": deterministic_subset(
            split["known_test"], args.known_eval_n, args.seed + 2
        ),
        "unknown_test": deterministic_subset(
            split["unknown_test"], args.unknown_eval_n, args.seed + 3
        ),
    }
    datasets = {
        name: MixedWM38DinomalyDataset(
            x, y, indices, image_size=args.image_size
        )
        for name, indices in selected.items()
    }
    loaders = {
        "known_train": make_loader(
            datasets["known_train"],
            args.batch_size,
            True,
            args.num_workers,
            device.type == "cuda",
            args.seed,
        ),
        "known_val": make_loader(
            datasets["known_val"],
            args.batch_size,
            False,
            args.num_workers,
            device.type == "cuda",
            args.seed + 1,
        ),
        "known_test": make_loader(
            datasets["known_test"],
            args.batch_size,
            False,
            args.num_workers,
            device.type == "cuda",
            args.seed + 2,
        ),
        "unknown_test": make_loader(
            datasets["unknown_test"],
            args.batch_size,
            False,
            args.num_workers,
            device.type == "cuda",
            args.seed + 3,
        ),
    }

    print(f"device: {device}")
    print(f"unknown class: {args.unknown_class} (index {unknown_idx})")
    print(
        "split: "
        + ", ".join(f"{key}={len(value)}" for key, value in selected.items())
    )
    print(f"loading frozen encoder: {args.encoder}", flush=True)
    model, bottleneck, decoder, trainable = build_model(
        args.encoder, args.decoder_depth, device
    )

    configuration: Dict[str, object] = {
        "full_class_order": list(FULL_CLASSES),
        "unknown_class": args.unknown_class,
        "unknown_index": unknown_idx,
        "data": args.data,
        "encoder": args.encoder,
        "image_size": args.image_size,
        "decoder_depth": args.decoder_depth,
        "total_iters": args.total_iters,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_lr": args.final_lr,
        "weight_decay": args.weight_decay,
        "hard_mining_final_p": args.hard_mining_final_p,
        "hard_mining_easy_gradient": args.hard_mining_easy_gradient,
        "top_fraction": args.top_fraction,
        "known_reject_rate": args.known_reject_rate,
        "seed": args.seed,
    }
    optimizer = StableAdamW(
        [{"params": trainable.parameters()}],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=args.lr,
        final_value=args.final_lr,
        total_iters=args.total_iters,
        warmup_iters=min(args.warmup_iters, args.total_iters),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = float("inf")
    best_iteration = 0
    iteration = 0
    running_losses = []
    train_loader = loaders["known_train"]
    while iteration < args.total_iters:
        model.train()
        for image, _labels, _indices in train_loader:
            image = image.to(device, non_blocking=True)
            p = min(
                args.hard_mining_final_p
                * iteration
                / max(1, args.hard_mining_warmup),
                args.hard_mining_final_p,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                encoded, decoded = model(image)
                loss = global_cosine_hm_percent(
                    encoded,
                    decoded,
                    p=p,
                    factor=args.hard_mining_easy_gradient,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            iteration += 1
            running_losses.append(float(loss.item()))

            if iteration % args.log_every == 0:
                print(
                    f"iter {iteration}/{args.total_iters} "
                    f"train_loss={np.mean(running_losses):.6f} p={p:.4f}",
                    flush=True,
                )
                running_losses.clear()

            should_validate = (
                iteration % args.val_every == 0
                or iteration == args.total_iters
            )
            if should_validate:
                val_loss = known_validation_loss(
                    model, loaders["known_val"], device, amp_enabled
                )
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_iteration = iteration
                    save_checkpoint(
                        checkpoint_path,
                        bottleneck,
                        decoder,
                        configuration,
                        iteration,
                        val_loss,
                    )
                print(
                    f"validation iter={iteration} loss={val_loss:.6f}"
                    + (" *" if improved else ""),
                    flush=True,
                )
                model.train()
            if iteration >= args.total_iters:
                break

    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    bottleneck.load_state_dict(checkpoint["bottleneck_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    model.eval()

    collected = {}
    for split_name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {split_name}", flush=True)
        collected[split_name] = collect_anomaly_scores(
            model,
            loaders[split_name],
            device,
            args.image_size,
            args.top_fraction,
            amp_enabled,
        )
    val_scores, _ = collected["known_val"]
    known_scores, _ = collected["known_test"]
    unknown_scores, unknown_original_indices = collected["unknown_test"]

    active = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_idx
    ]
    unknown_is_pure = (
        y[torch.as_tensor(unknown_original_indices)][:, active].sum(dim=1).numpy()
        == 0
    )
    evaluation = {
        "all_unknown": {
            "samples": int(len(unknown_scores)),
            "metrics": evaluate_scores(
                val_scores,
                known_scores,
                unknown_scores,
                args.known_reject_rate,
            ),
        },
    }
    if int(unknown_is_pure.sum()) > 0:
        evaluation["pure_unknown_only"] = {
            "samples": int(unknown_is_pure.sum()),
            "metrics": evaluate_scores(
                val_scores,
                known_scores,
                unknown_scores[unknown_is_pure],
                args.known_reject_rate,
            ),
        }
    if int((~unknown_is_pure).sum()) > 0:
        evaluation["mixed_known_plus_unknown"] = {
            "samples": int((~unknown_is_pure).sum()),
            "metrics": evaluate_scores(
                val_scores,
                known_scores,
                unknown_scores[~unknown_is_pure],
                args.known_reject_rate,
            ),
        }
    result = {
        "configuration": configuration,
        "device": str(device),
        "split_counts": {
            key: int(len(value)) for key, value in split.items()
        },
        "used_sample_counts": {
            key: int(len(value)) for key, value in selected.items()
        },
        "training": {
            "best_iteration": int(best_iteration),
            "best_known_validation_loss": float(best_val_loss),
        },
        "unknown_detection": evaluation,
        "notes": {
            "training_unknown_usage": "none",
            "threshold_source": "known validation only",
            "image_score": "mean of highest anomaly-map fraction",
            "encoder_input": "full wafer resized with nearest interpolation, repeated to RGB, ImageNet normalized",
        },
    }
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {checkpoint_path}")
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()
