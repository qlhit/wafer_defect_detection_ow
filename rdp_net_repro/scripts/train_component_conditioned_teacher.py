#!/usr/bin/env python3
"""Train the component-conditioned Flow-Matching teacher for MixedWM38.

Place this file in ``rdp_net_repro/scripts/`` and run it from the repository
root.  RDP-Net is frozen: its 512-D feature is the image condition and its
8-D sigmoid output (with the held-out slot masked) is encoded as the joint
known-component condition.

Only known wafers are used for training/validation.  For example, with
``--unknown-class S``, every wafer containing S is excluded from both sets.

Outputs (and nothing else):
  - best_teacher.pt
  - metrics.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


# Allow this script to run from scripts/ without installing rdpnet.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdpnet.models.rdpnet import RDPNet  # noqa: E402


FULL_CLASSES: Tuple[str, ...] = ("C", "D", "EL", "ER", "L", "NF", "S", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RDP component-conditioned WaferDiSC FM teacher"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument(
        "--rdp-checkpoint",
        default="checkpoint/rdp_leave_one_out/s_heldout/best_open_set.pt",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="S")
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)

    # WaferDiSC teacher settings.
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision (default: true)",
    )

    # Joint known-composition condition.
    parser.add_argument("--component-embed-dim", type=int, default=128)
    parser.add_argument("--condition-dim", type=int, default=512)
    parser.add_argument(
        "--truth-condition-prob",
        type=float,
        default=0.5,
        help="Per-sample probability of using the true known composition; otherwise use frozen RDP soft probabilities",
    )
    parser.add_argument("--image-loss-weight", type=float, default=1.0)
    parser.add_argument("--composition-loss-weight", type=float, default=1.0)
    parser.add_argument("--dual-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--minibatch-ot",
        action="store_true",
        help="Use the old Hungarian minibatch noise pairing. Off by default to match the paper's independent Gaussian pairing.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: checkpoint/component_waferdisc/<unknown>_heldout",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_load(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_arrays(path: str) -> Tuple[Tensor, Tensor]:
    raw = np.load(path)
    if "arr_0" not in raw:
        raise ValueError(f"{path} does not contain arr_0 wafer maps")

    x = torch.from_numpy(raw["arr_0"].astype(np.float32))
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"arr_0 must have shape [N,H,W], got {tuple(x.shape)}")

    # Full MixedWM38 labels normally live in arr_1.  Some prior open-world
    # exports store the 7-D active labels in arr_1 and the full labels in arr_2.
    label_key = None
    for key in ("arr_1", "arr_2"):
        if key in raw and raw[key].ndim == 2 and raw[key].shape[1] == len(FULL_CLASSES):
            label_key = key
            break
    if label_key is None:
        shapes = {k: tuple(raw[k].shape) for k in raw.files}
        raise ValueError(f"No [N,8] full-label array found. NPZ arrays: {shapes}")

    y = torch.from_numpy(raw[label_key].astype(np.float32))
    if len(x) != len(y):
        raise ValueError("Wafer maps and full labels have different sample counts")
    y = (y > 0.5).float()
    return x, y


def label_codes(y: Tensor) -> np.ndarray:
    a = y.to(torch.int64).cpu().numpy()
    return np.asarray(["".join(map(str, row.tolist())) for row in a])


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
    active = [i for i in range(len(FULL_CLASSES)) if i != unknown_idx]

    known_codes = label_codes(y_full[known_indices][:, active])
    train_idx, remainder_idx = train_test_split(
        known_indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=known_codes,
    )
    val_fraction = val_ratio / (1.0 - train_ratio)
    remainder_codes = label_codes(y_full[remainder_idx][:, active])
    val_idx, test_idx = train_test_split(
        remainder_idx,
        train_size=val_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=remainder_codes,
    )
    return {
        "known_train": np.asarray(train_idx),
        "known_val": np.asarray(val_idx),
        "known_test": np.asarray(test_idx),
        "unknown_test": np.asarray(unknown_indices),
    }


def make_loader(
    x: Tensor,
    y: Tensor,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return DataLoader(
        TensorDataset(x[idx], y[idx]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def normalize_teacher_input(x_raw: Tensor, image_size: int) -> Tensor:
    """Map raw 0/1/2 wafers to paper values 0/0.5/1 and resize."""
    x = x_raw.unsqueeze(1).float()
    if float(x.detach().max().item()) > 1.0:
        out = torch.zeros_like(x)
        out[(x > 0.0) & (x <= 1.0)] = 0.5
        out[x > 1.0] = 1.0
        x = out
    else:
        x = x.clamp(0.0, 1.0)
    if image_size > 0 and x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="nearest")
    if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
        raise ValueError(
            f"Teacher input size {tuple(x.shape[-2:])} must be divisible by 4; set --image-size 64"
        )
    return x


def forward_rdp_features(model: RDPNet, x: Tensor) -> Tuple[Tensor, Tensor]:
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


def load_frozen_rdp(
    checkpoint_path: str,
    unknown_class: str,
    device: torch.device,
) -> RDPNet:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("RDP checkpoint must be a dictionary")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("RDP checkpoint contains no model state dictionary")
    state = {k.removeprefix("module."): v for k, v in state.items()}

    saved_order = tuple(checkpoint.get("full_class_order", checkpoint.get("output_classes", FULL_CLASSES)))
    if saved_order != FULL_CLASSES:
        raise ValueError(f"RDP class order {saved_order} does not match {FULL_CLASSES}")
    saved_unknown = checkpoint.get("unknown_class")
    if saved_unknown is not None and saved_unknown != unknown_class:
        raise ValueError(
            f"RDP checkpoint holds out {saved_unknown}, but --unknown-class is {unknown_class}"
        )

    model = RDPNet(output_dim=len(FULL_CLASSES), in_channels=1, apply_sigmoid=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * -scale
        )
        angles = t[:, None] * frequencies[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class WindowConditionalAttention(nn.Module):
    def __init__(self, dim: int, feat_dim: int, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)
        self.condition = nn.Linear(feat_dim, dim, bias=False)

    def forward(self, x: Tensor, feature: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = x_pad.shape[-2:]
        query, key, value = self.to_qkv(x_pad).chunk(3, dim=1)
        condition = self.condition(feature).view(batch, channels, 1, 1)
        query, key = query + condition, key + condition

        def partition(z: Tensor) -> Tensor:
            z = z.view(batch, channels, padded_h // ws, ws, padded_w // ws, ws)
            z = z.permute(0, 2, 4, 3, 5, 1)
            return z.reshape(-1, ws * ws, channels)

        qw, kw, vw = partition(query), partition(key), partition(value)
        attention = torch.bmm(qw, kw.transpose(1, 2)) / math.sqrt(channels)
        output = torch.bmm(attention.softmax(dim=-1), vw)
        output = output.view(
            batch, padded_h // ws, padded_w // ws, ws, ws, channels
        )
        output = output.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, channels, padded_h, padded_w
        )
        return self.to_out(output[:, :, :height, :width])


class PaperBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, feat_dim: int):
        super().__init__()
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.attention = WindowConditionalAttention(out_channels, feat_dim, window_size=8)
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_feature: Tensor, condition: Tensor) -> Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = h + self.time_projection(time_feature)[:, :, None, None]
        h = F.gelu(self.norm2(self.conv2(h)))
        h = h + self.attention(h, condition)
        return h + self.residual(x)


class FlowTeacherUNet(nn.Module):
    """The original WaferDiSC/WigDM-style dual-time conditional U-Net."""

    def __init__(self, condition_dim: int = 512):
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )
        self.target_time = copy.deepcopy(self.time)
        self.input = nn.Conv2d(1, 256, 3, padding=1)
        self.block1 = PaperBlock(256, 256, 1024, condition_dim)
        self.down1 = nn.Conv2d(256, 256, 4, stride=2, padding=1)
        self.block2 = PaperBlock(256, 512, 1024, condition_dim)
        self.down2 = nn.Conv2d(512, 512, 4, stride=2, padding=1)
        self.block3 = PaperBlock(512, 784, 1024, condition_dim)
        self.middle = PaperBlock(784, 784, 1024, condition_dim)
        self.up1 = nn.ConvTranspose2d(784, 784, 4, stride=2, padding=1)
        self.up_block1 = PaperBlock(784, 512, 1024, condition_dim)
        self.up2 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.up_block2 = PaperBlock(512, 256, 1024, condition_dim)
        self.output = nn.Conv2d(256, 1, 1)

    def forward(
        self, x: Tensor, current_time: Tensor, target_time: Tensor, condition: Tensor
    ) -> Tensor:
        time_feature = self.time(current_time) + self.target_time(target_time)
        x = self.input(x)
        x = self.block1(x, time_feature, condition)
        x = self.down1(x)
        x = self.block2(x, time_feature, condition)
        x = self.down2(x)
        x = self.block3(x, time_feature, condition)
        x = self.middle(x, time_feature, condition)
        x = self.up1(x)
        x = self.up_block1(x, time_feature, condition)
        x = self.up2(x)
        x = self.up_block2(x, time_feature, condition)
        return self.output(x)


class ImageConditionAdapter(nn.Module):
    def __init__(self, input_dim: int = 512, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.net(feature)


class CompositionEncoder(nn.Module):
    """Encode a soft 8-D atomic multi-label vector compositionally."""

    def __init__(self, num_classes: int, embed_dim: int, output_dim: int):
        super().__init__()
        self.component_embeddings = nn.Parameter(
            torch.randn(num_classes, embed_dim) * 0.02
        )
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, probabilities: Tensor, active_mask: Tensor) -> Tensor:
        probabilities = probabilities * active_mask
        composition = torch.einsum(
            "bk,kd->bd", probabilities, self.component_embeddings
        )
        component_count = probabilities.sum(dim=1, keepdim=True)
        return self.net(torch.cat([composition, component_count], dim=1))


def choose_composition_condition(
    truth: Tensor,
    predicted_probabilities: Tensor,
    active_mask: Tensor,
    truth_probability: float,
) -> Tensor:
    truth = truth * active_mask
    predicted_probabilities = predicted_probabilities * active_mask
    if truth_probability <= 0.0:
        return predicted_probabilities
    if truth_probability >= 1.0:
        return truth
    use_truth = (
        torch.rand((truth.shape[0], 1), device=truth.device) < truth_probability
    )
    return torch.where(use_truth, truth, predicted_probabilities)


def pair_noise(x: Tensor, minibatch_ot: bool) -> Tensor:
    noise = torch.randn_like(x)
    if not minibatch_ot or len(x) <= 1:
        return noise
    # Optional compatibility with the user's previous teacher implementation.
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        x_flat = x.detach().float().reshape(len(x), -1).cpu().numpy()
        z_flat = noise.detach().float().reshape(len(x), -1).cpu().numpy()
        cost = np.linalg.norm(x_flat[:, None, :] - z_flat[None, :, :], axis=-1)
        _, columns = linear_sum_assignment(cost)
        noise = noise[torch.as_tensor(columns, device=noise.device)]
    return noise


def flow_matching_batch(x: Tensor, minibatch_ot: bool) -> Tuple[Tensor, Tensor, Tensor]:
    batch = len(x)
    time_value = torch.rand((batch,), device=x.device, dtype=x.dtype)
    noise = pair_noise(x, minibatch_ot)
    t_view = time_value.view(-1, 1, 1, 1)
    noisy_state = (1.0 - t_view) * noise + t_view * x
    velocity = x - noise
    return noisy_state, velocity, time_value


def compute_condition_inputs(
    x_raw: Tensor,
    labels: Tensor,
    rdp_model: RDPNet,
    active_mask: Tensor,
    truth_condition_probability: float,
    training: bool,
) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        logits, rdp_features = forward_rdp_features(
            rdp_model, x_raw.unsqueeze(1).float()
        )
        probabilities = torch.sigmoid(logits)
        condition_probabilities = choose_composition_condition(
            labels,
            probabilities,
            active_mask,
            truth_condition_probability if training else 0.0,
        )
    return rdp_features.detach(), condition_probabilities.detach()


@torch.no_grad()
def validate(
    teacher: FlowTeacherUNet,
    image_adapter: ImageConditionAdapter,
    composition_encoder: CompositionEncoder,
    rdp_model: RDPNet,
    loader: DataLoader,
    active_mask: Tensor,
    device: torch.device,
    image_size: int,
    image_weight: float,
    composition_weight: float,
    dual_weight: float,
    amp_enabled: bool,
    seed: int,
) -> Dict[str, float]:
    teacher.eval()
    image_adapter.eval()
    composition_encoder.eval()
    rdp_model.eval()

    # Fixed validation noise/times make early stopping comparable across epochs.
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    sums = {"total": 0.0, "image_fm": 0.0, "composition_fm": 0.0, "dual": 0.0}
    sample_count = 0

    for x_raw, labels in loader:
        x_raw = x_raw.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x = normalize_teacher_input(x_raw, image_size)
        rdp_features, condition_probabilities = compute_condition_inputs(
            x_raw,
            labels,
            rdp_model,
            active_mask,
            truth_condition_probability=0.0,
            training=False,
        )
        image_condition = image_adapter(rdp_features)
        composition_condition = composition_encoder(
            condition_probabilities, active_mask
        )
        batch = len(x)
        t = torch.rand((batch,), device=device, dtype=x.dtype, generator=generator)
        noise = torch.randn(
            x.shape, device=device, dtype=x.dtype, generator=generator
        )
        t_view = t.view(-1, 1, 1, 1)
        noisy_state = (1.0 - t_view) * noise + t_view * x
        target = x - noise

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            image_velocity = teacher(noisy_state, t, t, image_condition)
            composition_velocity = teacher(noisy_state, t, t, composition_condition)
            image_loss = F.mse_loss(image_velocity, target)
            composition_loss = F.mse_loss(composition_velocity, target)
            dual_loss = F.mse_loss(composition_velocity, image_velocity)
            total = (
                image_weight * image_loss
                + composition_weight * composition_loss
                + dual_weight * dual_loss
            )

        sums["total"] += float(total.item()) * batch
        sums["image_fm"] += float(image_loss.item()) * batch
        sums["composition_fm"] += float(composition_loss.item()) * batch
        sums["dual"] += float(dual_loss.item()) * batch
        sample_count += batch

    return {key: value / max(1, sample_count) for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.truth_condition_prob <= 1.0:
        raise ValueError("--truth-condition-prob must be in [0,1]")
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    unknown_idx = FULL_CLASSES.index(args.unknown_class)
    known_classes = [c for i, c in enumerate(FULL_CLASSES) if i != unknown_idx]
    active_mask = torch.ones((1, len(FULL_CLASSES)), device=device)
    active_mask[:, unknown_idx] = 0.0

    output_dir = Path(
        args.output_dir
        or f"checkpoint/component_waferdisc/{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_teacher.pt"
    metrics_path = output_dir / "metrics.json"

    x, y = load_arrays(args.data)
    split = make_open_set_split(
        y,
        unknown_idx,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        x,
        y,
        split["known_train"],
        args.batch_size,
        True,
        args.num_workers,
        pin_memory,
    )
    val_loader = make_loader(
        x,
        y,
        split["known_val"],
        args.batch_size,
        False,
        args.num_workers,
        pin_memory,
    )

    print(f"device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    print(f"unknown: {args.unknown_class} (index {unknown_idx})")
    print(f"known classes: {known_classes}")
    print(
        "split: "
        + ", ".join(f"{name}={len(indices)}" for name, indices in split.items())
    )
    print(f"teacher image size: {args.image_size}x{args.image_size}")

    rdp_model = load_frozen_rdp(args.rdp_checkpoint, args.unknown_class, device)
    teacher = FlowTeacherUNet(condition_dim=args.condition_dim).to(device)
    image_adapter = ImageConditionAdapter(512, args.condition_dim).to(device)
    composition_encoder = CompositionEncoder(
        len(FULL_CLASSES), args.component_embed_dim, args.condition_dim
    ).to(device)

    trainable_parameters = (
        list(teacher.parameters())
        + list(image_adapter.parameters())
        + list(composition_encoder.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    accumulation_steps = max(
        1, int(math.ceil(args.effective_batch_size / args.batch_size))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    best_validation = float("inf")
    best_epoch = 0
    best_breakdown: Dict[str, float] = {}
    patience = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        teacher.train()
        image_adapter.train()
        composition_encoder.train()
        optimizer.zero_grad(set_to_none=True)
        running = {"total": 0.0, "image_fm": 0.0, "composition_fm": 0.0, "dual": 0.0}
        seen = 0

        for step, (x_raw, labels) in enumerate(train_loader, start=1):
            x_raw = x_raw.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x_teacher = normalize_teacher_input(x_raw, args.image_size)
            rdp_features, condition_probabilities = compute_condition_inputs(
                x_raw,
                labels,
                rdp_model,
                active_mask,
                args.truth_condition_prob,
                training=True,
            )
            noisy_state, target_velocity, t = flow_matching_batch(
                x_teacher, args.minibatch_ot
            )

            # Backpropagate the image branch first and detach its prediction.
            # This keeps peak memory close to the original one-condition teacher.
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                image_condition = image_adapter(rdp_features)
                image_velocity = teacher(noisy_state, t, t, image_condition)
                image_loss = F.mse_loss(image_velocity, target_velocity)
                image_objective = args.image_loss_weight * image_loss
            image_velocity_target = image_velocity.detach()
            scaler.scale(image_objective / accumulation_steps).backward()

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                composition_condition = composition_encoder(
                    condition_probabilities, active_mask
                )
                composition_velocity = teacher(
                    noisy_state, t, t, composition_condition
                )
                composition_loss = F.mse_loss(
                    composition_velocity, target_velocity
                )
                dual_loss = F.mse_loss(
                    composition_velocity, image_velocity_target
                )
                composition_objective = (
                    args.composition_loss_weight * composition_loss
                    + args.dual_loss_weight * dual_loss
                )
            scaler.scale(composition_objective / accumulation_steps).backward()

            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch = len(x_raw)
            total_value = float((image_objective + composition_objective).item())
            running["total"] += total_value * batch
            running["image_fm"] += float(image_loss.item()) * batch
            running["composition_fm"] += float(composition_loss.item()) * batch
            running["dual"] += float(dual_loss.item()) * batch
            seen += batch

        train_metrics = {
            key: value / max(1, seen) for key, value in running.items()
        }
        validation_metrics = validate(
            teacher,
            image_adapter,
            composition_encoder,
            rdp_model,
            val_loader,
            active_mask,
            device,
            args.image_size,
            args.image_loss_weight,
            args.composition_loss_weight,
            args.dual_loss_weight,
            amp_enabled,
            seed=args.seed + 10000,
        )

        improved = validation_metrics["total"] < best_validation - args.min_delta
        if improved:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            best_breakdown = dict(validation_metrics)
            patience = 0
            checkpoint = {
                "teacher_state_dict": teacher.state_dict(),
                "image_adapter_state_dict": image_adapter.state_dict(),
                "composition_encoder_state_dict": composition_encoder.state_dict(),
                "configuration": {
                    "full_class_order": list(FULL_CLASSES),
                    "known_classes": known_classes,
                    "unknown_class": args.unknown_class,
                    "unknown_index": unknown_idx,
                    "active_mask": active_mask.detach().cpu().view(-1).tolist(),
                    "rdp_checkpoint": args.rdp_checkpoint,
                    "image_size": args.image_size,
                    "condition_dim": args.condition_dim,
                    "component_embed_dim": args.component_embed_dim,
                    "truth_condition_prob": args.truth_condition_prob,
                    "image_loss_weight": args.image_loss_weight,
                    "composition_loss_weight": args.composition_loss_weight,
                    "dual_loss_weight": args.dual_loss_weight,
                    "minibatch_ot": args.minibatch_ot,
                    "seed": args.seed,
                },
                "best_epoch": best_epoch,
                "best_validation": best_breakdown,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            patience += 1

        marker = " *" if improved else ""
        print(
            f"epoch {epoch:03d} | "
            f"train={train_metrics['total']:.6f} "
            f"val={validation_metrics['total']:.6f} "
            f"val_img={validation_metrics['image_fm']:.6f} "
            f"val_comp={validation_metrics['composition_fm']:.6f} "
            f"val_dual={validation_metrics['dual']:.6f} "
            f"patience={patience}/{args.early_stop_patience}{marker}"
        )

        if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
            break

    metrics = {
        "configuration": {
            "full_class_order": list(FULL_CLASSES),
            "known_classes": known_classes,
            "unknown_class": args.unknown_class,
            "unknown_index": unknown_idx,
            "rdp_checkpoint": args.rdp_checkpoint,
            "seed": args.seed,
            "image_size": args.image_size,
            "truth_condition_prob": args.truth_condition_prob,
            "image_loss_weight": args.image_loss_weight,
            "composition_loss_weight": args.composition_loss_weight,
            "dual_loss_weight": args.dual_loss_weight,
            "minibatch_ot": args.minibatch_ot,
        },
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "training": {
            "best_epoch": best_epoch,
            "best_validation": best_breakdown,
            "elapsed_seconds": round(time.time() - started, 2),
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {checkpoint_path}")
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()
