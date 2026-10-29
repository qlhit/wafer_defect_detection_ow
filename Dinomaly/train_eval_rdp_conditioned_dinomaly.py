#!/usr/bin/env python3
"""RDP-composition-conditioned Dinomaly for atomic open-set wafer detection.

Place this file in the Dinomaly repository root. The RDP repository is expected
to be a sibling directory by default (``../rdp_net_repro``). RDP and DINOv2
stay frozen; only the Dinomaly bottleneck/decoder and a small composition FiLM
module are trained.

Only two files are written:
  - best_model.pt
  - metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from optimizers import StableAdamW
from train_eval_dinomaly_atomic_open_set import (
    FULL_CLASSES,
    MixedWM38DinomalyDataset,
    build_model,
    cal_anomaly_maps,
    deterministic_subset,
    get_gaussian_kernel,
    load_arrays,
    make_loader,
    make_open_set_split,
    set_seed,
    torch_load,
)
from utils import WarmCosineScheduler, global_cosine_hm_percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate RDP-conditioned Dinomaly on MixedWM38"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default="C")
    parser.add_argument("--rdp-repo", default="../rdp_net_repro")
    parser.add_argument(
        "--rdp-checkpoint",
        default=None,
        help=(
            "Default: <rdp-repo>/checkpoint/rdp_leave_one_out/"
            "<unknown>_heldout/best_open_set.pt"
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=(
            "dinov2reg_vit_small_14",
            "dinov2reg_vit_base_14",
            "dinov2reg_vit_large_14",
        ),
        default="dinov2reg_vit_small_14",
    )
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--condition-hidden-dim", type=int, default=128)
    parser.add_argument("--condition-scale", type=float, default=0.10)
    parser.add_argument("--truth-condition-prob", type=float, default=0.50)
    parser.add_argument("--condition-margin", type=float, default=0.02)
    parser.add_argument("--condition-loss-weight", type=float, default=0.50)

    parser.add_argument("--total-iters", type=int, default=20000)
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
        "--patience",
        type=int,
        default=0,
        help="Known-validation checks without improvement; <=0 disables",
    )

    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--calibration-n", type=int, default=1000)
    parser.add_argument("--known-eval-n", type=int, default=1000)
    parser.add_argument("--unknown-eval-n", type=int, default=1000)
    parser.add_argument("--top-fraction", type=float, default=0.005)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument("--rdp-label-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Default: checkpoint/rdp_conditioned_dinomaly/"
            "<unknown>_heldout"
        ),
    )
    return parser.parse_args()


def add_rdp_source(rdp_repo: Path) -> None:
    source = (rdp_repo / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"RDP source directory not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def load_frozen_rdp(
    rdp_repo: Path,
    checkpoint_path: Path,
    unknown_class: str,
    device: torch.device,
):
    add_rdp_source(rdp_repo)
    from rdpnet.models.rdpnet import RDPNet

    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    stored_unknown = checkpoint.get("unknown_class")
    if stored_unknown is not None and stored_unknown != unknown_class:
        raise ValueError(
            f"RDP checkpoint holds out {stored_unknown}, not {unknown_class}"
        )
    stored_order = tuple(checkpoint.get("full_class_order", FULL_CLASSES))
    if stored_order != FULL_CLASSES:
        raise ValueError(
            f"RDP class order {stored_order} differs from {FULL_CLASSES}"
        )
    unknown_index = FULL_CLASSES.index(unknown_class)
    active_indices = [
        int(index)
        for index in checkpoint.get(
            "active_known_indices",
            [index for index in range(len(FULL_CLASSES)) if index != unknown_index],
        )
    ]
    expected_indices = [
        index for index in range(len(FULL_CLASSES)) if index != unknown_index
    ]
    if active_indices != expected_indices:
        raise ValueError(
            "RDP active indices must follow the canonical seven-class order: "
            f"expected {expected_indices}, got {active_indices}"
        )

    model = RDPNet(
        output_dim=len(FULL_CLASSES), in_channels=1, apply_sigmoid=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, active_indices


def forward_rdp_logits(model: nn.Module, wafer: torch.Tensor) -> torch.Tensor:
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


class CompositionFiLM(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int,
        feature_dim: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, tokens: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.net(condition).chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return tokens * (1.0 + self.scale * gamma) + self.scale * beta


class RDPConditionedDinomaly(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        condition_dim: int,
        condition_hidden_dim: int,
        feature_dim: int,
        condition_scale: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.conditioner = CompositionFiLM(
            condition_dim,
            condition_hidden_dim,
            feature_dim,
            condition_scale,
        )

    @torch.no_grad()
    def encode(
        self, image: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        x = self.base.encoder.prepare_tokens(image)
        encoder_tokens: List[torch.Tensor] = []
        for index, block in enumerate(self.base.encoder.blocks):
            if index > self.base.target_layers[-1]:
                break
            x = block(x)
            if index in self.base.target_layers:
                encoder_tokens.append(x)

        register_count = self.base.encoder.num_register_tokens
        side = int(
            math.sqrt(encoder_tokens[0].shape[1] - 1 - register_count)
        )
        fused_tokens = self.base.fuse_feature(encoder_tokens)
        encoder_groups = [
            self.base.fuse_feature(
                [encoder_tokens[index] for index in indices]
            )
            for indices in self.base.fuse_layer_encoder
        ]
        patch_start = 1 + register_count
        encoder_maps = [
            tokens[:, patch_start:, :]
            .permute(0, 2, 1)
            .reshape(image.shape[0], -1, side, side)
            .contiguous()
            for tokens in encoder_groups
        ]
        return encoder_maps, fused_tokens, side

    def decode(
        self,
        fused_tokens: torch.Tensor,
        side: int,
        condition: torch.Tensor,
    ) -> List[torch.Tensor]:
        x = self.conditioner(fused_tokens, condition)
        for block in self.base.bottleneck:
            x = block(x)

        decoder_tokens: List[torch.Tensor] = []
        for block in self.base.decoder:
            x = block(x, attn_mask=None)
            decoder_tokens.append(x)
        decoder_tokens = decoder_tokens[::-1]
        decoder_groups = [
            self.base.fuse_feature(
                [decoder_tokens[index] for index in indices]
            )
            for indices in self.base.fuse_layer_decoder
        ]
        patch_start = 1 + self.base.encoder.num_register_tokens
        return [
            tokens[:, patch_start:, :]
            .permute(0, 2, 1)
            .reshape(tokens.shape[0], -1, side, side)
            .contiguous()
            for tokens in decoder_groups
        ]

    def forward(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        encoded, fused, side = self.encode(image)
        decoded = self.decode(fused, side, condition)
        return encoded, decoded


def per_sample_reconstruction_error(
    encoded: Sequence[torch.Tensor], decoded: Sequence[torch.Tensor]
) -> torch.Tensor:
    errors = []
    for target, prediction in zip(encoded, decoded):
        cosine = F.cosine_similarity(target.detach(), prediction, dim=1)
        errors.append((1.0 - cosine).flatten(1).mean(dim=1))
    return torch.stack(errors, dim=1).mean(dim=1)


def make_wrong_condition(condition: torch.Tensor) -> torch.Tensor:
    batch_size = condition.shape[0]
    if batch_size < 2:
        return 1.0 - condition
    wrong = condition.roll(1, dims=0)
    equal = torch.isclose(wrong, condition, atol=1e-6).all(dim=1)
    for shift in range(2, batch_size):
        if not equal.any():
            break
        candidate = condition.roll(shift, dims=0)
        different = ~torch.isclose(
            candidate, condition, atol=1e-6
        ).all(dim=1)
        replace = equal & different
        wrong[replace] = candidate[replace]
        equal = torch.isclose(wrong, condition, atol=1e-6).all(dim=1)
    if equal.any():
        wrong[equal] = 1.0 - condition[equal]
    return wrong


def save_rng_state(device: torch.device):
    cpu_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    return cpu_state, cuda_state


def restore_rng_state(state, device: torch.device) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def raw_wafer_batch(
    full_x: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return full_x[indices.to(torch.long)].unsqueeze(1).to(
        device, non_blocking=True
    )


@torch.no_grad()
def rdp_condition(
    rdp_model: nn.Module,
    raw_wafer: torch.Tensor,
    active_indices: Sequence[int],
) -> torch.Tensor:
    logits = forward_rdp_logits(rdp_model, raw_wafer)
    return torch.sigmoid(logits[:, active_indices])


def mixed_training_condition(
    rdp_soft: torch.Tensor,
    truth_full: torch.Tensor,
    active_indices: Sequence[int],
    truth_probability: float,
) -> torch.Tensor:
    truth = truth_full[:, active_indices]
    if truth_probability <= 0:
        return rdp_soft
    if truth_probability >= 1:
        return truth
    use_truth = (
        torch.rand(len(truth), 1, device=truth.device) < truth_probability
    )
    return torch.where(use_truth, truth, rdp_soft)


@torch.no_grad()
def validate(
    model: RDPConditionedDinomaly,
    rdp_model: nn.Module,
    loader,
    full_x: torch.Tensor,
    active_indices: Sequence[int],
    device: torch.device,
    margin: float,
    margin_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_sum = rec_sum = margin_sum = gap_sum = 0.0
    count = 0
    for image, _labels, indices in loader:
        image = image.to(device, non_blocking=True)
        raw = raw_wafer_batch(full_x, indices, device)
        condition = rdp_condition(rdp_model, raw, active_indices)
        wrong = make_wrong_condition(condition)
        encoded, fused, side = model.encode(image)
        decoded = model.decode(fused, side, condition)
        wrong_decoded = model.decode(fused, side, wrong)
        correct_error = per_sample_reconstruction_error(encoded, decoded)
        wrong_error = per_sample_reconstruction_error(encoded, wrong_decoded)
        margin_loss = F.relu(margin + correct_error - wrong_error)
        total = correct_error + margin_weight * margin_loss
        batch_size = len(image)
        total_sum += float(total.sum().item())
        rec_sum += float(correct_error.sum().item())
        margin_sum += float(margin_loss.sum().item())
        gap_sum += float((wrong_error - correct_error).sum().item())
        count += batch_size
    denominator = max(1, count)
    return {
        "total": total_sum / denominator,
        "reconstruction": rec_sum / denominator,
        "condition_margin": margin_sum / denominator,
        "wrong_minus_correct_gap": gap_sum / denominator,
    }


@torch.no_grad()
def collect_scores(
    model: RDPConditionedDinomaly,
    rdp_model: nn.Module,
    loader,
    full_x: torch.Tensor,
    active_indices: Sequence[int],
    device: torch.device,
    image_size: int,
    top_fraction: float,
    rdp_label_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    score_parts = []
    index_parts = []
    exact_sum = 0.0
    count = 0
    for image, labels, indices in loader:
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        raw = raw_wafer_batch(full_x, indices, device)
        condition = rdp_condition(rdp_model, raw, active_indices)
        encoded, decoded = model(image, condition)
        anomaly_map, _ = cal_anomaly_maps(
            encoded, decoded, out_size=image_size
        )
        anomaly_map = gaussian(anomaly_map.float())
        flattened = anomaly_map.flatten(1)
        keep = max(1, int(math.ceil(flattened.shape[1] * top_fraction)))
        scores = torch.topk(flattened, k=keep, dim=1).values.mean(dim=1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite anomaly score")
        prediction = condition >= rdp_label_threshold
        truth = labels[:, active_indices] > 0.5
        exact_sum += float((prediction == truth).all(dim=1).sum().item())
        count += len(image)
        score_parts.append(scores.cpu().numpy())
        index_parts.append(indices.numpy())
    return (
        np.concatenate(score_parts),
        np.concatenate(index_parts),
        exact_sum / max(1, count),
    )


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    candidates = np.flatnonzero(tpr >= 0.95)
    return float(fpr[candidates[0]]) if len(candidates) else 1.0


def detection_metrics(
    calibration: np.ndarray,
    known: np.ndarray,
    unknown: np.ndarray,
    known_reject_rate: float,
) -> Dict[str, float]:
    labels = np.concatenate(
        [
            np.zeros(len(known), dtype=np.int64),
            np.ones(len(unknown), dtype=np.int64),
        ]
    )
    scores = np.concatenate([known, unknown])
    threshold = float(np.quantile(calibration, 1.0 - known_reject_rate))
    known_reject = float(np.mean(known > threshold))
    unknown_recall = float(np.mean(unknown > threshold))
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
        "known_mean": float(np.mean(known)),
        "unknown_mean": float(np.mean(unknown)),
    }


def save_checkpoint(
    path: Path,
    bottleneck: nn.ModuleList,
    decoder: nn.ModuleList,
    conditioner: nn.Module,
    configuration: Dict[str, object],
    iteration: int,
    validation: Dict[str, float],
) -> None:
    torch.save(
        {
            "bottleneck_state_dict": bottleneck.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "conditioner_state_dict": conditioner.state_dict(),
            "configuration": configuration,
            "best_iteration": int(iteration),
            "best_validation": validation,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.image_size % 14 != 0:
        raise ValueError("--image-size must be divisible by 14")
    if args.decoder_depth < 2:
        raise ValueError("--decoder-depth must be at least 2")
    if not 0.0 <= args.truth_condition_prob <= 1.0:
        raise ValueError("--truth-condition-prob must be in [0,1]")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0,1]")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rdp_repo = Path(args.rdp_repo).resolve()
    rdp_checkpoint = (
        Path(args.rdp_checkpoint).resolve()
        if args.rdp_checkpoint
        else rdp_repo
        / "checkpoint"
        / "rdp_leave_one_out"
        / f"{args.unknown_class.lower()}_heldout"
        / "best_open_set.pt"
    )
    if not rdp_checkpoint.is_file():
        raise FileNotFoundError(f"RDP checkpoint not found: {rdp_checkpoint}")

    output_dir = Path(
        args.output_dir
        or "checkpoint/rdp_conditioned_dinomaly/"
        f"{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"

    x, y = load_arrays(args.data)
    unknown_index = FULL_CLASSES.index(args.unknown_class)
    split = make_open_set_split(
        y,
        unknown_index,
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
    loaders = {
        name: make_loader(
            MixedWM38DinomalyDataset(
                x, y, indices, image_size=args.image_size
            ),
            batch_size=args.batch_size,
            shuffle=name == "known_train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed + offset,
        )
        for offset, (name, indices) in enumerate(selected.items())
    }

    print(f"device: {device}")
    print(
        f"unknown class: {args.unknown_class}; "
        f"RDP checkpoint: {rdp_checkpoint}",
        flush=True,
    )
    print(
        "split: "
        + ", ".join(f"{name}={len(indices)}" for name, indices in selected.items()),
        flush=True,
    )
    rdp_model, active_indices = load_frozen_rdp(
        rdp_repo, rdp_checkpoint, args.unknown_class, device
    )
    base, bottleneck, decoder, _ = build_model(
        args.encoder, args.decoder_depth, device
    )
    feature_dim = bottleneck[0].fc2.out_features
    model = RDPConditionedDinomaly(
        base,
        condition_dim=len(active_indices),
        condition_hidden_dim=args.condition_hidden_dim,
        feature_dim=feature_dim,
        condition_scale=args.condition_scale,
    ).to(device)
    trainable = nn.ModuleList([bottleneck, decoder, model.conditioner])

    configuration: Dict[str, object] = {
        "full_class_order": list(FULL_CLASSES),
        "unknown_class": args.unknown_class,
        "unknown_index": unknown_index,
        "active_known_indices": active_indices,
        "known_classes": [FULL_CLASSES[index] for index in active_indices],
        "data": args.data,
        "rdp_repo": str(rdp_repo),
        "rdp_checkpoint": str(rdp_checkpoint),
        "encoder": args.encoder,
        "image_size": args.image_size,
        "input_encoding": "gray_repeat",
        "decoder_depth": args.decoder_depth,
        "condition_hidden_dim": args.condition_hidden_dim,
        "condition_scale": args.condition_scale,
        "truth_condition_prob": args.truth_condition_prob,
        "condition_margin": args.condition_margin,
        "condition_loss_weight": args.condition_loss_weight,
        "total_iters": args.total_iters,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_lr": args.final_lr,
        "weight_decay": args.weight_decay,
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

    best_iteration = 0
    best_validation = {
        "total": float("inf"),
        "reconstruction": float("inf"),
        "condition_margin": float("inf"),
        "wrong_minus_correct_gap": float("-inf"),
    }
    patience_count = 0
    iteration = 0
    running_total: List[float] = []
    running_rec: List[float] = []
    running_margin: List[float] = []
    running_gap: List[float] = []

    while iteration < args.total_iters:
        model.train()
        for image, labels, indices in loaders["known_train"]:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            raw = raw_wafer_batch(x, indices, device)
            soft_condition = rdp_condition(rdp_model, raw, active_indices)
            condition = mixed_training_condition(
                soft_condition,
                labels,
                active_indices,
                args.truth_condition_prob,
            )
            wrong_condition = make_wrong_condition(condition)
            p = min(
                args.hard_mining_final_p
                * iteration
                / max(1, args.hard_mining_warmup),
                args.hard_mining_final_p,
            )

            optimizer.zero_grad(set_to_none=True)
            encoded, fused, side = model.encode(image)
            rng_state = save_rng_state(device)
            decoded = model.decode(fused, side, condition)
            restore_rng_state(rng_state, device)
            wrong_decoded = model.decode(fused, side, wrong_condition)

            reconstruction = global_cosine_hm_percent(
                encoded,
                decoded,
                p=p,
                factor=args.hard_mining_easy_gradient,
            )
            correct_error = per_sample_reconstruction_error(encoded, decoded)
            wrong_error = per_sample_reconstruction_error(
                encoded, wrong_decoded
            )
            margin_loss = F.relu(
                args.condition_margin + correct_error - wrong_error
            ).mean()
            loss = reconstruction + args.condition_loss_weight * margin_loss
            loss.backward()

            if iteration == 0:
                grad_sq = 0.0
                for parameter in model.conditioner.parameters():
                    if parameter.grad is not None:
                        grad_sq += float(
                            parameter.grad.detach().float().square().sum().item()
                        )
                grad_norm = math.sqrt(grad_sq)
                if not math.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError("Composition conditioner received no gradient")
                print(
                    f"conditioner first-step grad_norm={grad_norm:.6e}",
                    flush=True,
                )

            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()

            iteration += 1
            running_total.append(float(loss.item()))
            running_rec.append(float(reconstruction.item()))
            running_margin.append(float(margin_loss.item()))
            running_gap.append(
                float((wrong_error - correct_error).mean().item())
            )
            if iteration % args.log_every == 0:
                print(
                    f"iter {iteration}/{args.total_iters} "
                    f"total={np.mean(running_total):.6f} "
                    f"rec={np.mean(running_rec):.6f} "
                    f"margin={np.mean(running_margin):.6f} "
                    f"gap={np.mean(running_gap):.6f} p={p:.4f}",
                    flush=True,
                )
                running_total.clear()
                running_rec.clear()
                running_margin.clear()
                running_gap.clear()

            if iteration % args.val_every == 0 or iteration == args.total_iters:
                validation = validate(
                    model,
                    rdp_model,
                    loaders["known_val"],
                    x,
                    active_indices,
                    device,
                    args.condition_margin,
                    args.condition_loss_weight,
                )
                improved = validation["total"] < best_validation["total"]
                if improved:
                    best_iteration = iteration
                    best_validation = validation
                    patience_count = 0
                    save_checkpoint(
                        checkpoint_path,
                        bottleneck,
                        decoder,
                        model.conditioner,
                        configuration,
                        iteration,
                        validation,
                    )
                else:
                    patience_count += 1
                print(
                    f"validation iter={iteration} "
                    f"total={validation['total']:.6f} "
                    f"rec={validation['reconstruction']:.6f} "
                    f"margin={validation['condition_margin']:.6f} "
                    f"gap={validation['wrong_minus_correct_gap']:.6f} "
                    f"patience={patience_count}/{args.patience}"
                    + (" *" if improved else ""),
                    flush=True,
                )
                model.train()
                if args.patience > 0 and patience_count >= args.patience:
                    print("early stopping", flush=True)
                    break
            if iteration >= args.total_iters:
                break
        if (
            iteration >= args.total_iters
            or (args.patience > 0 and patience_count >= args.patience)
        ):
            break

    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    bottleneck.load_state_dict(checkpoint["bottleneck_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    model.conditioner.load_state_dict(
        checkpoint["conditioner_state_dict"], strict=True
    )
    model.eval()

    collected = {}
    rdp_exact = {}
    for split_name in ("known_val", "known_test", "unknown_test"):
        print(f"scoring {split_name}", flush=True)
        scores, indices, exact = collect_scores(
            model,
            rdp_model,
            loaders[split_name],
            x,
            active_indices,
            device,
            args.image_size,
            args.top_fraction,
            args.rdp_label_threshold,
        )
        collected[split_name] = (scores, indices)
        rdp_exact[split_name] = exact

    calibration_scores, _ = collected["known_val"]
    known_scores, _ = collected["known_test"]
    unknown_scores, unknown_indices = collected["unknown_test"]
    unknown_labels = y[torch.as_tensor(unknown_indices)].cpu().numpy()
    pure_mask = unknown_labels[:, active_indices].sum(axis=1) == 0
    mixed_mask = ~pure_mask

    detection = {
        "all_unknown": {
            "samples": int(len(unknown_scores)),
            "metrics": detection_metrics(
                calibration_scores,
                known_scores,
                unknown_scores,
                args.known_reject_rate,
            ),
        }
    }
    if pure_mask.any():
        detection["pure_unknown_only"] = {
            "samples": int(pure_mask.sum()),
            "metrics": detection_metrics(
                calibration_scores,
                known_scores,
                unknown_scores[pure_mask],
                args.known_reject_rate,
            ),
        }
    if mixed_mask.any():
        detection["mixed_known_plus_unknown"] = {
            "samples": int(mixed_mask.sum()),
            "metrics": detection_metrics(
                calibration_scores,
                known_scores,
                unknown_scores[mixed_mask],
                args.known_reject_rate,
            ),
        }

    result = {
        "configuration": configuration,
        "device": str(device),
        "split_counts": {
            name: int(len(indices)) for name, indices in split.items()
        },
        "used_sample_counts": {
            name: int(len(indices)) for name, indices in selected.items()
        },
        "training": {
            "best_iteration": int(best_iteration),
            "best_validation": best_validation,
        },
        "rdp_known_projection_exact_match": rdp_exact,
        "unknown_detection": detection,
        "notes": {
            "unknown_used_for_training_or_checkpoint_selection": False,
            "rdp_and_dino_encoder_frozen": True,
            "evaluation_condition": "RDP soft known-component probabilities",
            "condition_enforcement": (
                "correct-condition reconstruction plus shuffled-condition "
                "margin loss"
            ),
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
