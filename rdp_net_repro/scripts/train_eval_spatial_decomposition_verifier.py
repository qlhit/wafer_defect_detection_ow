#!/usr/bin/env python3
"""Spatial known-component decomposition for RDP + RCR-v2.

This script is a minimal segmentation-based experiment for MixedWM38.  It does
not replace Dinomaly and it does not train on the held-out defect class.

Pipeline
--------
1. Keep the existing frozen RDP + RCR-v2 predictions as known-component
   candidates.
2. Build pixel-level supervision from single-known-defect training wafers.
3. Online-compose those atomic wafers into synthetic mixed wafers while
   preserving one independent mask per known class.
4. Train a small multi-label U-Net with independent sigmoid masks.  Real known
   training wafers additionally provide weak image-level composition labels.
5. Describe each predicted class mask by its area, radial profile and
   rotation-invariant angular coverage.
6. Calibrate class-specific support thresholds using known validation only.
7. Retain an RCR-v2 component only when the spatial decomposition supports it.

The held-out split is touched only after training, checkpoint selection,
descriptor fitting and threshold calibration are complete.

Only two files are written:
  * best_spatial_decomposer.pt
  * spatial_decomposition_metrics.json

Place this file in ``rdp_net_repro/scripts/`` and run it from the repository
root.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a spatial decomposition verifier for RDP + RCR-v2"
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
        help="Default: base_checkpoint stored in the RCR-v2 checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--synthetic-samples-per-epoch",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--synthetic-validation-samples",
        type=int,
        default=2048,
    )
    parser.add_argument("--max-synthetic-components", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pixel-bce-weight", type=float, default=1.0)
    parser.add_argument("--pixel-dice-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-presence-weight", type=float, default=0.5)
    parser.add_argument("--real-presence-weight", type=float, default=1.0)
    parser.add_argument("--positive-pixel-weight", type=float, default=4.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--mask-top-fraction", type=float, default=0.01)
    parser.add_argument("--radial-bins", type=int, default=6)
    parser.add_argument("--angular-bins", type=int, default=12)
    parser.add_argument(
        "--shape-distance-weight",
        type=float,
        default=1.0,
        help="support = presence_logit - weight * robust_shape_distance",
    )
    parser.add_argument(
        "--descriptor-reference-max",
        type=int,
        default=12000,
        help="Maximum known-training wafers used to fit mask-shape references",
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
        help="Relax thresholds until validation EM drop is no larger than this",
    )
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: checkpoint/rdp_spatial_decomposition/<class>_heldout",
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_real_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    local = torch.as_tensor(indices, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x[local], y[local]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def random_atomic_transform(
    wafer: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotation = int(torch.randint(0, 4, (1,), generator=generator).item())
    wafer = torch.rot90(wafer, rotation, dims=(-2, -1))
    mask = torch.rot90(mask, rotation, dims=(-2, -1))
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        wafer = torch.flip(wafer, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        wafer = torch.flip(wafer, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    return wafer, mask


class SyntheticAtomicMixtureDataset(Dataset):
    """Online synthetic mixtures with exact overlapping class masks."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_indices: np.ndarray,
        active_indices: Sequence[int],
        samples_per_epoch: int,
        max_components: int,
        valid_threshold: float,
        normal_value: float,
        defect_value: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.active_indices = list(active_indices)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_components = int(max_components)
        self.valid_threshold = float(valid_threshold)
        self.normal_value = float(normal_value)
        self.defect_value = float(defect_value)
        self.seed = int(seed)
        self.epoch = 0

        local_labels = y[torch.as_tensor(train_indices, dtype=torch.long)]
        active_labels = local_labels[:, self.active_indices]
        atomic = active_labels.sum(dim=1) == 1
        self.pools: List[torch.Tensor] = []
        for local_class in range(len(self.active_indices)):
            keep = atomic & (active_labels[:, local_class] > 0.5)
            pool = torch.as_tensor(train_indices, dtype=torch.long)[keep]
            if len(pool) == 0:
                name = FULL_CLASSES[self.active_indices[local_class]]
                raise ValueError(f"No single-defect training wafer for class {name}")
            self.pools.append(pool)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(
        self, item: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 1_000_003 + int(item)
        )
        class_count = len(self.active_indices)
        component_count = int(
            torch.randint(
                1,
                min(self.max_components, class_count) + 1,
                (1,),
                generator=generator,
            ).item()
        )
        selected = torch.randperm(class_count, generator=generator)[
            :component_count
        ]
        first_pool = self.pools[int(selected[0])]
        first_index = int(
            first_pool[
                int(torch.randint(0, len(first_pool), (1,), generator=generator))
            ]
        )
        base = self.x[first_index].clone()
        valid = base > self.valid_threshold
        composed = torch.zeros_like(base)
        composed[valid] = self.normal_value
        masks = torch.zeros(
            class_count,
            base.shape[-2],
            base.shape[-1],
            dtype=torch.float32,
        )
        labels = torch.zeros(class_count, dtype=torch.float32)

        for local_class_tensor in selected:
            local_class = int(local_class_tensor)
            pool = self.pools[local_class]
            source_index = int(
                pool[int(torch.randint(0, len(pool), (1,), generator=generator))]
            )
            source = self.x[source_index].clone()
            source_mask = source > (0.5 * (self.normal_value + self.defect_value))
            source, source_mask = random_atomic_transform(
                source, source_mask.float(), generator
            )
            source_mask = (source_mask > 0.5) & valid
            composed[source_mask] = self.defect_value
            masks[local_class] = torch.maximum(
                masks[local_class], source_mask.float()
            )
            labels[local_class] = 1.0

        scale = max(self.defect_value, 1e-6)
        return (
            (composed / scale).unsqueeze(0),
            masks,
            labels,
            valid.float().unsqueeze(0),
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AtomicMaskNet(nn.Module):
    """Small U-Net with independent masks and an auxiliary presence head."""

    def __init__(self, class_count: int, base_channels: int = 32) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)
        self.bottleneck = ConvBlock(4 * c, 8 * c)
        self.dec3 = ConvBlock(8 * c + 4 * c, 4 * c)
        self.dec2 = ConvBlock(4 * c + 2 * c, 2 * c)
        self.dec1 = ConvBlock(2 * c + c, c)
        self.mask_head = nn.Conv2d(c, class_count, 1)
        self.presence_head = nn.Sequential(
            nn.Linear(8 * c, 4 * c),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(4 * c, class_count),
        )

    @staticmethod
    def up_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat((self.up_like(bottleneck, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up_like(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up_like(d2, e1), e1), dim=1))
        mask_logits = self.mask_head(d1)
        pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        presence_logits = self.presence_head(pooled)
        return mask_logits, presence_logits


def segmentation_loss(
    mask_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    positive_pixel_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = valid.expand_as(mask_logits)
    raw_bce = F.binary_cross_entropy_with_logits(
        mask_logits, masks, reduction="none"
    )
    pixel_weights = 1.0 + (positive_pixel_weight - 1.0) * masks
    pixel_bce = (raw_bce * pixel_weights * valid).sum() / valid.sum().clamp_min(1.0)

    probability = torch.sigmoid(mask_logits) * valid
    intersection = (probability * masks).flatten(2).sum(dim=-1)
    denominator = (
        probability.flatten(2).sum(dim=-1)
        + masks.flatten(2).sum(dim=-1)
    )
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    positive = labels > 0.5
    dice_loss = (
        (1.0 - dice)[positive].mean()
        if positive.any()
        else mask_logits.new_zeros(())
    )
    presence_bce = F.binary_cross_entropy_with_logits(presence_logits, labels)
    return pixel_bce, dice_loss, presence_bce


@torch.inference_mode()
def evaluate_validation(
    model: AtomicMaskNet,
    synthetic_loader: DataLoader,
    real_loader: DataLoader,
    active_indices: Sequence[int],
    defect_value: float,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "total": 0.0,
        "mask_bce": 0.0,
        "mask_dice": 0.0,
        "synthetic_presence": 0.0,
    }
    synthetic_count = 0
    for wafer, masks, labels, valid in synthetic_loader:
        wafer = wafer.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        mask_logits, presence_logits = model(wafer)
        pixel_bce, dice_loss, presence_loss = segmentation_loss(
            mask_logits,
            presence_logits,
            masks,
            labels,
            valid,
            args.positive_pixel_weight,
        )
        loss = (
            args.pixel_bce_weight * pixel_bce
            + args.pixel_dice_weight * dice_loss
            + args.synthetic_presence_weight * presence_loss
        )
        batch_count = len(wafer)
        totals["total"] += float(loss.item()) * batch_count
        totals["mask_bce"] += float(pixel_bce.item()) * batch_count
        totals["mask_dice"] += float(dice_loss.item()) * batch_count
        totals["synthetic_presence"] += float(presence_loss.item()) * batch_count
        synthetic_count += batch_count

    real_total = 0.0
    real_count = 0
    for wafer, full_labels in real_loader:
        wafer = wafer.unsqueeze(1).to(device, non_blocking=True) / defect_value
        labels = full_labels[:, active_indices].to(device, non_blocking=True)
        _, presence_logits = model(wafer)
        loss = F.binary_cross_entropy_with_logits(presence_logits, labels)
        real_total += float(loss.item()) * len(wafer)
        real_count += len(wafer)
    real_presence = real_total / max(real_count, 1)
    result = {
        name: value / max(synthetic_count, 1)
        for name, value in totals.items()
    }
    result["real_presence"] = real_presence
    result["total"] += args.real_presence_weight * real_presence
    return result


def train_model(
    model: AtomicMaskNet,
    synthetic_dataset: SyntheticAtomicMixtureDataset,
    synthetic_loader: DataLoader,
    synthetic_val_loader: DataLoader,
    real_loader: DataLoader,
    val_loader: DataLoader,
    active_indices: Sequence[int],
    defect_value: float,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation: Dict[str, float] = {"total": float("inf")}
    stale = 0
    epochs_ran = 0

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        synthetic_dataset.set_epoch(epoch)
        model.train()
        totals = {"total": 0.0, "bce": 0.0, "dice": 0.0, "real": 0.0}
        count = 0
        real_iterator = iter(real_loader)
        for wafer, masks, labels, valid in synthetic_loader:
            try:
                real_wafer, real_labels_full = next(real_iterator)
            except StopIteration:
                real_iterator = iter(real_loader)
                real_wafer, real_labels_full = next(real_iterator)

            wafer = wafer.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            mask_logits, synthetic_presence = model(wafer)
            pixel_bce, dice_loss, synthetic_presence_loss = segmentation_loss(
                mask_logits,
                synthetic_presence,
                masks,
                labels,
                valid,
                args.positive_pixel_weight,
            )

            real_wafer = (
                real_wafer.unsqueeze(1).to(device, non_blocking=True) / defect_value
            )
            real_labels = real_labels_full[:, active_indices].to(
                device, non_blocking=True
            )
            _, real_presence = model(real_wafer)
            real_presence_loss = F.binary_cross_entropy_with_logits(
                real_presence, real_labels
            )
            loss = (
                args.pixel_bce_weight * pixel_bce
                + args.pixel_dice_weight * dice_loss
                + args.synthetic_presence_weight * synthetic_presence_loss
                + args.real_presence_weight * real_presence_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_count = len(wafer)
            totals["total"] += float(loss.item()) * batch_count
            totals["bce"] += float(pixel_bce.item()) * batch_count
            totals["dice"] += float(dice_loss.item()) * batch_count
            totals["real"] += float(real_presence_loss.item()) * batch_count
            count += batch_count
        scheduler.step()

        validation = evaluate_validation(
            model,
            synthetic_val_loader,
            val_loader,
            active_indices,
            defect_value,
            device,
            args,
        )
        improved = validation["total"] < best_validation["total"] - 1e-6
        if improved:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch:03d} "
            f"total={totals['total'] / max(count, 1):.6f} "
            f"mask_bce={totals['bce'] / max(count, 1):.6f} "
            f"mask_dice={totals['dice'] / max(count, 1):.6f} "
            f"real_presence={totals['real'] / max(count, 1):.6f} "
            f"val={validation['total']:.6f} "
            f"val_mask={validation['mask_bce'] + validation['mask_dice']:.6f} "
            f"val_presence={validation['real_presence']:.6f} "
            f"patience={stale}/{args.patience}"
            + (" *" if improved else ""),
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print("early stopping", flush=True)
            break
    return best_state, best_epoch, best_validation, epochs_ran


def spatial_bin_maps(
    height: int,
    width: int,
    radial_bins: int,
    angular_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    yy = torch.linspace(-1.0, 1.0, height).view(height, 1).expand(height, width)
    xx = torch.linspace(-1.0, 1.0, width).view(1, width).expand(height, width)
    radius = torch.sqrt(xx.square() + yy.square()).clamp(0.0, 0.999999)
    angle = torch.atan2(yy, xx)
    radial = torch.clamp((radius * radial_bins).long(), 0, radial_bins - 1)
    angular = torch.clamp(
        (((angle + math.pi) / (2.0 * math.pi)) * angular_bins).long(),
        0,
        angular_bins - 1,
    )
    return radial, angular


def mask_descriptors(
    probabilities: torch.Tensor,
    valid: torch.Tensor,
    mask_threshold: float,
    top_fraction: float,
    radial_bins: int,
    angular_bins: int,
) -> torch.Tensor:
    """Rotation-invariant descriptor for every sample and known class."""
    batch, class_count, height, width = probabilities.shape
    valid = valid.bool()
    valid_expanded = valid.expand(batch, class_count, height, width)
    probability = probabilities * valid_expanded
    valid_count = valid.flatten(2).sum(dim=-1).clamp_min(1).float()
    soft_area = probability.flatten(2).sum(dim=-1) / valid_count
    hard_area = (
        ((probability >= mask_threshold) & valid_expanded)
        .flatten(2)
        .sum(dim=-1)
        .float()
        / valid_count
    )

    flattened = probability.flatten(2)
    valid_pixels = int(valid.flatten(2).sum(dim=-1).min().item())
    top_count = max(1, int(math.ceil(valid_pixels * top_fraction)))
    top_mean = torch.topk(flattened, k=min(top_count, height * width), dim=-1).values.mean(
        dim=-1
    )

    radial_map, angular_map = spatial_bin_maps(
        height, width, radial_bins, angular_bins
    )
    radial_parts: List[torch.Tensor] = []
    for bin_index in range(radial_bins):
        region = (radial_map == bin_index).view(1, 1, height, width)
        region = region & valid_expanded
        denominator = region.flatten(2).sum(dim=-1).clamp_min(1).float()
        radial_parts.append((probability * region).flatten(2).sum(dim=-1) / denominator)

    angular_parts: List[torch.Tensor] = []
    for bin_index in range(angular_bins):
        region = (angular_map == bin_index).view(1, 1, height, width)
        region = region & valid_expanded
        denominator = region.flatten(2).sum(dim=-1).clamp_min(1).float()
        angular_parts.append(
            (probability * region).flatten(2).sum(dim=-1) / denominator
        )
    angular_profile = torch.stack(angular_parts, dim=-1)
    angular_profile = torch.sort(angular_profile, dim=-1, descending=True).values
    radial_profile = torch.stack(radial_parts, dim=-1)
    return torch.cat(
        (
            soft_area.unsqueeze(-1),
            hard_area.unsqueeze(-1),
            top_mean.unsqueeze(-1),
            radial_profile,
            angular_profile,
        ),
        dim=-1,
    )


@torch.inference_mode()
def collect_decomposition_outputs(
    model: AtomicMaskNet,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    active_indices: Sequence[int],
    batch_size: int,
    defect_value: float,
    valid_threshold: float,
    device: torch.device,
    args: argparse.Namespace,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    presence_parts: List[np.ndarray] = []
    descriptor_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    print(f"spatial decomposition: {name} ({len(indices)})", flush=True)
    for start in range(0, len(indices), batch_size):
        local = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        raw_wafer = x[local].unsqueeze(1)
        wafer = raw_wafer.to(device, non_blocking=True) / defect_value
        mask_logits, presence_logits = model(wafer)
        probabilities = torch.sigmoid(mask_logits).cpu()
        valid = (raw_wafer > valid_threshold).float()
        descriptors = mask_descriptors(
            probabilities,
            valid,
            args.mask_threshold,
            args.mask_top_fraction,
            args.radial_bins,
            args.angular_bins,
        )
        presence_parts.append(presence_logits.cpu().numpy())
        descriptor_parts.append(descriptors.numpy())
        target_parts.append(y[local][:, active_indices].numpy())
    return (
        np.concatenate(presence_parts).astype(np.float32),
        np.concatenate(descriptor_parts).astype(np.float32),
        np.concatenate(target_parts).astype(np.float32),
    )


def fit_robust_shape_reference(
    descriptors: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    class_count = targets.shape[1]
    centers = []
    scales = []
    counts = []
    for class_index in range(class_count):
        positive = targets[:, class_index] > 0.5
        if not positive.any():
            raise ValueError(f"No positive descriptor for local class {class_index}")
        values = descriptors[positive, class_index]
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        scale = np.maximum(1.4826 * mad, 0.02)
        centers.append(center)
        scales.append(scale)
        counts.append(int(positive.sum()))
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
        counts,
    )


def spatial_support(
    presence_logits: np.ndarray,
    descriptors: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    distance_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    z = (descriptors - centers[None, :, :]) / scales[None, :, :]
    # Clip single unstable descriptor dimensions before aggregation.
    distance = np.sqrt(np.mean(np.minimum(z * z, 25.0), axis=-1))
    support = presence_logits - distance_weight * distance
    return support.astype(np.float32), distance.astype(np.float32)


@torch.inference_mode()
def rcr_predictions(
    base_model: nn.Module,
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


def calibrate_spatial_thresholds(
    baseline: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
    class_names: Sequence[str],
    initial_retention: float,
    max_em_drop: float,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    baseline = baseline.astype(bool)
    truth = targets > 0.5
    baseline_em = float(np.mean(np.all(baseline == truth, axis=1)))
    retention_grid = sorted(
        {
            float(initial_retention),
            max(float(initial_retention), 0.9975),
            max(float(initial_retention), 0.9990),
            max(float(initial_retention), 0.9995),
            1.0,
        }
    )
    attempts: List[Dict[str, float]] = []
    selected_thresholds = None
    selected_prediction = baseline
    selected_retention = 1.0

    for retention in retention_grid:
        thresholds = np.empty(targets.shape[1], dtype=np.float32)
        accept = np.ones_like(baseline, dtype=bool)
        for class_index, class_name in enumerate(class_names):
            pool = baseline[:, class_index] & truth[:, class_index]
            if not pool.any():
                raise ValueError(
                    f"No true RCR-v2 validation candidate for {class_name}"
                )
            thresholds[class_index] = float(
                np.quantile(support[pool, class_index], 1.0 - retention)
            )
            accept[:, class_index] = (
                support[:, class_index] >= thresholds[class_index]
            )
        prediction = baseline & accept
        em = float(np.mean(np.all(prediction == truth, axis=1)))
        attempts.append(
            {
                "positive_retention": float(retention),
                "known_validation_exact_match": em,
                "change_from_baseline": em - baseline_em,
            }
        )
        if em >= baseline_em - max_em_drop:
            selected_thresholds = thresholds
            selected_prediction = prediction
            selected_retention = float(retention)
            break
    if selected_thresholds is None:
        raise RuntimeError("Spatial threshold relaxation failed")

    per_class = {}
    for class_index, class_name in enumerate(class_names):
        true_pool = baseline[:, class_index] & truth[:, class_index]
        false_pool = baseline[:, class_index] & ~truth[:, class_index]
        accepted = support[:, class_index] >= selected_thresholds[class_index]
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
        }
    return selected_thresholds, selected_retention, {
        "baseline_known_validation_exact_match": baseline_em,
        "gated_known_validation_exact_match": float(
            np.mean(np.all(selected_prediction == truth, axis=1))
        ),
        "selected_positive_retention": selected_retention,
        "retention_attempts": attempts,
        "per_class": per_class,
    }


def choose_reference_indices(
    indices: np.ndarray,
    y: torch.Tensor,
    active_indices: Sequence[int],
    maximum: int,
    seed: int,
) -> np.ndarray:
    if maximum <= 0 or len(indices) <= maximum:
        return np.asarray(indices)
    generator = np.random.default_rng(seed)
    selected: set[int] = set()
    local = y[torch.as_tensor(indices, dtype=torch.long)][:, active_indices].numpy()
    per_class = max(1, maximum // len(active_indices))
    for class_index in range(len(active_indices)):
        candidates = np.asarray(indices)[local[:, class_index] > 0.5]
        if len(candidates):
            take = min(len(candidates), per_class)
            selected.update(
                int(value)
                for value in generator.choice(candidates, size=take, replace=False)
            )
    if len(selected) < maximum:
        remaining = np.asarray(
            [value for value in indices if int(value) not in selected],
            dtype=np.int64,
        )
        take = min(len(remaining), maximum - len(selected))
        if take:
            selected.update(
                int(value)
                for value in generator.choice(remaining, size=take, replace=False)
            )
    return np.asarray(sorted(selected), dtype=np.int64)


def concise_metrics(metrics: Dict[str, object]) -> str:
    return (
        f"EM={metrics['exact_match']:.6f} "
        f"micro_F1={metrics['micro_f1']:.6f} "
        f"FP/sample={metrics['mean_false_positives_per_sample']:.6f} "
        f"pure_false={metrics['pure_unknown_false_known_rate']}"
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Epochs and batch sizes must be positive")
    if not 1 <= args.max_synthetic_components <= len(FULL_CLASSES) - 1:
        raise ValueError("Invalid --max-synthetic-components")
    if (
        args.synthetic_samples_per_epoch <= 0
        or args.synthetic_validation_samples <= 0
    ):
        raise ValueError("Synthetic sample counts must be positive")
    if not 0.0 < args.mask_top_fraction <= 1.0:
        raise ValueError("--mask-top-fraction must be in (0,1]")
    if args.radial_bins <= 0 or args.angular_bins <= 0:
        raise ValueError("Spatial bin counts must be positive")
    if not 0.5 < args.positive_retention <= 1.0:
        raise ValueError("--positive-retention must be in (0.5,1.0]")
    if not 0.0 <= args.max_known_val_em_drop < 1.0:
        raise ValueError("--max-known-val-em-drop must be in [0,1)")
    set_seed(args.seed)

    unknown_index = FULL_CLASSES.index(args.unknown_class)
    active_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    class_names = [FULL_CLASSES[index] for index in active_indices]
    v2_path = resolve_v2_checkpoint(args.v2_checkpoint, args.unknown_class)
    v2_checkpoint = torch_load(v2_path, map_location="cpu")
    if v2_checkpoint.get("unknown_class") != args.unknown_class:
        raise ValueError(
            f"RCR-v2 holds out {v2_checkpoint.get('unknown_class')}, "
            f"requested {args.unknown_class}"
        )
    stored_active = [
        int(value) for value in v2_checkpoint["active_known_indices"]
    ]
    if stored_active != active_indices:
        raise ValueError(f"Active class mismatch: {stored_active} vs {active_indices}")

    base_path = Path(
        args.base_checkpoint or v2_checkpoint["base_checkpoint"]
    ).resolve()
    output_dir = Path(
        args.output_dir
        or (
            "checkpoint/rdp_spatial_decomposition/"
            f"{args.unknown_class.lower()}_heldout"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_spatial_decomposer.pt"
    metrics_path = output_dir / "spatial_decomposition_metrics.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"unknown={args.unknown_class}; known classes={class_names}")
    print(f"loading RCR-v2: {v2_path}")
    print(f"loading frozen base RDP: {base_path}")

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
        + ", ".join(f"{name}={len(value)}" for name, value in split.items())
    )
    valid_threshold, normal_value, defect_value = data_values(x)
    synthetic_dataset = SyntheticAtomicMixtureDataset(
        x,
        y,
        split["known_train"],
        active_indices,
        args.synthetic_samples_per_epoch,
        args.max_synthetic_components,
        valid_threshold,
        normal_value,
        defect_value,
        args.seed + 1009,
    )
    synthetic_val_dataset = SyntheticAtomicMixtureDataset(
        x,
        y,
        split["known_val"],
        active_indices,
        args.synthetic_validation_samples,
        args.max_synthetic_components,
        valid_threshold,
        normal_value,
        defect_value,
        args.seed + 1013,
    )
    atomic_counts = {
        class_name: int(len(synthetic_dataset.pools[index]))
        for index, class_name in enumerate(class_names)
    }
    print("single-defect source counts:", atomic_counts)
    synthetic_loader = DataLoader(
        synthetic_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    synthetic_val_loader = DataLoader(
        synthetic_val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    real_train_loader = make_real_loader(
        x,
        y,
        split["known_train"],
        args.batch_size,
        True,
        args.num_workers,
        args.seed + 2011,
    )
    val_loader = make_real_loader(
        x,
        y,
        split["known_val"],
        args.eval_batch_size,
        False,
        args.num_workers,
        args.seed + 2017,
    )

    model = AtomicMaskNet(
        class_count=len(active_indices),
        base_channels=args.base_channels,
    ).to(device)
    best_state, best_epoch, best_validation, epochs_ran = train_model(
        model,
        synthetic_dataset,
        synthetic_loader,
        synthetic_val_loader,
        real_train_loader,
        val_loader,
        active_indices,
        defect_value,
        device,
        args,
    )
    model.load_state_dict(best_state, strict=True)
    model.eval()

    # Fit class mask-shape references on known training predictions only.
    reference_indices = choose_reference_indices(
        split["known_train"],
        y,
        active_indices,
        args.descriptor_reference_max,
        args.seed + 3001,
    )
    reference_presence, reference_descriptors, reference_targets = (
        collect_decomposition_outputs(
            model,
            x,
            y,
            reference_indices,
            active_indices,
            args.eval_batch_size,
            defect_value,
            valid_threshold,
            device,
            args,
            "known_train_reference",
        )
    )
    centers, scales, reference_positive_counts = fit_robust_shape_reference(
        reference_descriptors, reference_targets
    )
    del reference_presence, reference_descriptors, reference_targets

    base_model, base_active, _ = load_frozen_rdp(
        base_path, args.unknown_class, device
    )
    if base_active != active_indices:
        raise ValueError(f"Frozen RDP active class mismatch: {base_active}")
    verifier = load_v2_model(v2_checkpoint, device)
    rcr_label_threshold = float(v2_checkpoint["label_threshold"])
    rcr_distance_weight = float(v2_checkpoint["distance_weight"])
    rcr_support_thresholds = v2_checkpoint["support_thresholds"]
    if isinstance(rcr_support_thresholds, torch.Tensor):
        rcr_support_thresholds = rcr_support_thresholds.cpu().numpy()
    rcr_support_thresholds = np.asarray(
        rcr_support_thresholds, dtype=np.float32
    )

    print("calibrating spatial verifier on known_val")
    val_rcr, val_targets = rcr_predictions(
        base_model,
        verifier,
        x,
        y,
        split["known_val"],
        active_indices,
        args.eval_batch_size,
        device,
        rcr_distance_weight,
        rcr_label_threshold,
        rcr_support_thresholds,
        "known_val",
    )
    val_presence, val_descriptors, val_spatial_targets = (
        collect_decomposition_outputs(
            model,
            x,
            y,
            split["known_val"],
            active_indices,
            args.eval_batch_size,
            defect_value,
            valid_threshold,
            device,
            args,
            "known_val",
        )
    )
    if not np.array_equal(val_targets > 0.5, val_spatial_targets > 0.5):
        raise RuntimeError("Known-validation target order mismatch")
    val_support, val_shape_distance = spatial_support(
        val_presence,
        val_descriptors,
        centers,
        scales,
        args.shape_distance_weight,
    )
    spatial_thresholds, selected_retention, calibration = (
        calibrate_spatial_thresholds(
            val_rcr,
            val_targets,
            val_support,
            class_names,
            args.positive_retention,
            args.max_known_val_em_drop,
        )
    )
    del val_presence, val_descriptors, val_shape_distance

    def score_split(
        name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        baseline, targets = rcr_predictions(
            base_model,
            verifier,
            x,
            y,
            split[name],
            active_indices,
            args.eval_batch_size,
            device,
            rcr_distance_weight,
            rcr_label_threshold,
            rcr_support_thresholds,
            name,
        )
        presence, descriptors, spatial_targets = collect_decomposition_outputs(
            model,
            x,
            y,
            split[name],
            active_indices,
            args.eval_batch_size,
            defect_value,
            valid_threshold,
            device,
            args,
            name,
        )
        if not np.array_equal(targets > 0.5, spatial_targets > 0.5):
            raise RuntimeError(f"{name} target order mismatch")
        support, shape_distance = spatial_support(
            presence,
            descriptors,
            centers,
            scales,
            args.shape_distance_weight,
        )
        fused = baseline & (
            support >= spatial_thresholds.reshape(1, -1)
        )
        return baseline, fused, targets, shape_distance

    print("scoring known_test")
    known_baseline, known_fused, known_targets, known_shape_distance = score_split(
        "known_test"
    )

    # Held-out data is used only here, after every model and threshold is fixed.
    print("scoring unknown_test (final diagnostic only)")
    (
        unknown_baseline,
        unknown_fused,
        unknown_targets,
        unknown_shape_distance,
    ) = score_split("unknown_test")

    results = {
        "known_test": {
            "rcr_v2": multilabel_metrics(
                known_baseline, known_targets, class_names
            ),
            "rcr_v2_plus_spatial_decomposition": multilabel_metrics(
                known_fused, known_targets, class_names
            ),
        },
        "unknown_test": {
            "rcr_v2": multilabel_metrics(
                unknown_baseline, unknown_targets, class_names
            ),
            "rcr_v2_plus_spatial_decomposition": multilabel_metrics(
                unknown_fused, unknown_targets, class_names
            ),
        },
    }
    shape_diagnostics = {}
    for local_index, class_name in enumerate(class_names):
        true_known = known_targets[:, local_index] > 0.5
        false_unknown_candidate = (
            unknown_baseline[:, local_index]
            & ~(unknown_targets[:, local_index] > 0.5)
        )
        shape_diagnostics[class_name] = {
            "known_true_shape_distance_mean": (
                float(known_shape_distance[true_known, local_index].mean())
                if true_known.any()
                else None
            ),
            "unknown_false_candidate_shape_distance_mean": (
                float(
                    unknown_shape_distance[
                        false_unknown_candidate, local_index
                    ].mean()
                )
                if false_unknown_candidate.any()
                else None
            ),
            "unknown_false_candidates_before": int(
                false_unknown_candidate.sum()
            ),
            "unknown_false_candidates_after": int(
                (
                    unknown_fused[:, local_index]
                    & ~(unknown_targets[:, local_index] > 0.5)
                ).sum()
            ),
        }

    metrics = {
        "configuration": {
            "method": "synthetic_atomic_multilabel_spatial_decomposition",
            "full_class_order": list(FULL_CLASSES),
            "known_classes": class_names,
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_index,
            "data": str(Path(args.data).resolve()),
            "v2_checkpoint": str(v2_path),
            "base_checkpoint": str(base_path),
            "heldout_data_used_for_training_selection_or_calibration": False,
            "mask_activation": "independent_sigmoid",
            "synthetic_mask_source": (
                "single-known-defect wafers from known_train only"
            ),
            "synthetic_samples_per_epoch": args.synthetic_samples_per_epoch,
            "synthetic_validation_samples": args.synthetic_validation_samples,
            "max_synthetic_components": args.max_synthetic_components,
            "base_channels": args.base_channels,
            "epochs": args.epochs,
            "epochs_ran": epochs_ran,
            "batch_size": args.batch_size,
            "mask_threshold": args.mask_threshold,
            "mask_top_fraction": args.mask_top_fraction,
            "radial_bins": args.radial_bins,
            "angular_bins": args.angular_bins,
            "shape_distance_weight": args.shape_distance_weight,
            "positive_retention": args.positive_retention,
            "max_known_val_em_drop": args.max_known_val_em_drop,
            "seed": args.seed,
        },
        "split_counts": {
            name: int(len(value)) for name, value in split.items()
        },
        "single_defect_source_counts": atomic_counts,
        "training": {
            "best_epoch": best_epoch,
            "best_known_validation": best_validation,
        },
        "shape_reference": {
            "samples": int(len(reference_indices)),
            "positive_samples_per_class": {
                name: reference_positive_counts[index]
                for index, name in enumerate(class_names)
            },
            "descriptor_order": (
                ["soft_area", "hard_area", "top_probability_mean"]
                + [f"radial_bin_{index}" for index in range(args.radial_bins)]
                + [
                    f"sorted_angular_bin_{index}"
                    for index in range(args.angular_bins)
                ]
            ),
        },
        "known_validation_calibration": calibration,
        "results": results,
        "shape_diagnostics": shape_diagnostics,
        "notes": {
            "unknown_gate": "Dinomaly remains the wafer-level unknown detector",
            "spatial_role": (
                "RCR-v2 candidate filtering; it does not name the held-out class"
            ),
            "primary_er_test": (
                "whether ER-held-out EL false-positive rate falls below 1.0"
            ),
            "checkpoint_selection": (
                "known-validation synthetic mask loss plus real presence loss"
            ),
        },
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_configuration": {
            "class_count": len(active_indices),
            "base_channels": args.base_channels,
        },
        "full_class_order": list(FULL_CLASSES),
        "active_known_indices": active_indices,
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "v2_checkpoint": str(v2_path),
        "base_checkpoint": str(base_path),
        "data_values": {
            "valid_threshold": valid_threshold,
            "normal_value": normal_value,
            "defect_value": defect_value,
        },
        "descriptor_configuration": {
            "mask_threshold": args.mask_threshold,
            "mask_top_fraction": args.mask_top_fraction,
            "radial_bins": args.radial_bins,
            "angular_bins": args.angular_bins,
            "shape_distance_weight": args.shape_distance_weight,
        },
        "shape_centers": torch.from_numpy(centers),
        "shape_scales": torch.from_numpy(scales),
        "spatial_support_thresholds": torch.from_numpy(spatial_thresholds),
        "selected_positive_retention": selected_retention,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, checkpoint_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\ncomparison")
    print("known_test")
    for name, values in results["known_test"].items():
        print(f"  {name}: {concise_metrics(values)}")
    print("unknown_test")
    for name, values in results["unknown_test"].items():
        print(f"  {name}: {concise_metrics(values)}")
    if args.unknown_class == "ER" and "EL" in class_names:
        before = results["unknown_test"]["rcr_v2"]["per_class"]["EL"][
            "false_positive_rate"
        ]
        after = results["unknown_test"][
            "rcr_v2_plus_spatial_decomposition"
        ]["per_class"]["EL"]["false_positive_rate"]
        print(f"ER-like confusion: EL FPR {before:.6f} -> {after:.6f}")
    print("saved:", checkpoint_path)
    print("saved:", metrics_path)


if __name__ == "__main__":
    main()
