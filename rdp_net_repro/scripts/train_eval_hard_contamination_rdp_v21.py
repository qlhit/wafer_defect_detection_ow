#!/usr/bin/env python3
"""V2.1: positive-distilled hard-contamination RDP-Net fine-tuning.

This program keeps the held-out class completely unused.  It starts from the
corresponding known-only RDP checkpoint, freezes a copy as a teacher, and
fine-tunes a student with:

  * clean asymmetric classification loss;
  * hard pseudo-contamination classification loss;
  * contaminated/clean probability consistency;
  * positive-label-only clean teacher distillation;
  * rejection of the strongest absent-class response.

For every pseudo sample, several generic irregular masks are generated and the
frozen teacher selects the candidate that most strongly activates an absent
known class.  No held-out image or label is used for generation or selection.

Only two files are written: ``best_model.pt`` and ``metrics.json``.
The V1 script ``train_eval_contamination_robust_rdp.py`` must be in the same
directory so its dataset, RDP and metric utilities can be reused exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

import train_eval_contamination_robust_rdp as v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard-contamination fine-tuning of leave-one-out RDP-Net"
    )
    parser.add_argument(
        "--data",
        default="/home/qlh/project/wafer_dataset/mixedwm38/MixedWM38.npz",
    )
    parser.add_argument("--unknown-class", choices=v1.FULL_CLASSES, default="C")
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help=(
            "Known-only leave-one-out checkpoint. Default: "
            "checkpoint/rdp_leave_one_out/<class>_heldout/best_open_set.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--known-train-ratio", type=float, default=0.70)
    parser.add_argument("--known-val-ratio", type=float, default=0.15)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3529)

    parser.add_argument("--pseudo-batch-fraction", type=float, default=0.50)
    parser.add_argument("--pseudo-min-die-fraction", type=float, default=0.02)
    parser.add_argument("--pseudo-max-die-fraction", type=float, default=0.12)
    parser.add_argument(
        "--hard-candidates",
        type=int,
        default=4,
        help="Number of generic pseudo masks from which the frozen teacher selects.",
    )
    parser.add_argument("--contamination-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--hard-reject-loss-weight", type=float, default=0.50)
    parser.add_argument("--distillation-loss-weight", type=float, default=0.25)

    parser.add_argument("--selection-clean-weight", type=float, default=0.65)
    parser.add_argument("--selection-pure-false-penalty", type=float, default=0.10)
    parser.add_argument("--selection-pure-soft-penalty", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: checkpoint/rdp_hard_contamination_v21/<class>_heldout",
    )
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_student_and_teacher(
    checkpoint_path: Path,
    unknown_class: str,
    active_indices: Sequence[int],
    device: torch.device,
) -> Tuple[v1.RDPNet, v1.RDPNet]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base RDP checkpoint not found: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path)
    stored_unknown = checkpoint.get("unknown_class")
    if stored_unknown is not None and stored_unknown != unknown_class:
        raise ValueError(
            f"Checkpoint holds out {stored_unknown}, not requested {unknown_class}"
        )
    stored_active = checkpoint.get("active_known_indices")
    if stored_active is not None:
        stored_active = [int(index) for index in stored_active]
        if stored_active != list(active_indices):
            raise ValueError(
                f"Checkpoint active indices {stored_active}, expected {list(active_indices)}"
            )
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("Checkpoint contains neither model_state_dict nor state_dict")

    student = v1.RDPNet(
        output_dim=len(v1.FULL_CLASSES), in_channels=1, apply_sigmoid=False
    ).to(device)
    student.load_state_dict(state, strict=True)
    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return student, teacher


@torch.inference_mode()
def select_hard_candidates(
    teacher: v1.RDPNet,
    wafer: torch.Tensor,
    active_targets: torch.Tensor,
    active_indices: Sequence[int],
    candidate_count: int,
    generator: torch.Generator,
    min_fraction: float,
    max_fraction: float,
    valid_threshold: float,
    normal_value: float,
    defect_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Select pseudo inputs that maximize teacher false-known activation."""
    contaminated_candidates = []
    pure_candidates = []
    contaminated_scores = []
    pure_scores = []
    absent = active_targets < 0.5

    for _ in range(candidate_count):
        contaminated, pure = v1.synthesize_pseudo_unknown(
            wafer,
            min_fraction,
            max_fraction,
            generator,
            valid_threshold,
            normal_value,
            defect_value,
        )
        candidate_inputs = torch.cat([contaminated, pure], dim=0)
        candidate_logits, _ = v1.forward_features(teacher, candidate_inputs)
        count = len(wafer)
        contaminated_probability = torch.sigmoid(
            candidate_logits[:count, active_indices]
        )
        pure_probability = torch.sigmoid(candidate_logits[count:, active_indices])

        masked = contaminated_probability.masked_fill(~absent, -1.0)
        contaminated_score = masked.max(dim=1).values
        has_absent = absent.any(dim=1)
        contaminated_score = torch.where(
            has_absent,
            contaminated_score,
            contaminated_probability.max(dim=1).values,
        )
        contaminated_candidates.append(contaminated)
        pure_candidates.append(pure)
        contaminated_scores.append(contaminated_score)
        pure_scores.append(pure_probability.max(dim=1).values)

    contaminated_stack = torch.stack(contaminated_candidates, dim=0)
    pure_stack = torch.stack(pure_candidates, dim=0)
    contaminated_score_stack = torch.stack(contaminated_scores, dim=0)
    pure_score_stack = torch.stack(pure_scores, dim=0)
    best_contaminated = contaminated_score_stack.argmax(dim=0)
    best_pure = pure_score_stack.argmax(dim=0)
    batch_indices = torch.arange(len(wafer), device=wafer.device)
    selected_contaminated = contaminated_stack[best_contaminated, batch_indices]
    selected_pure = pure_stack[best_pure, batch_indices]
    diagnostics = {
        "teacher_hard_contam_score": float(
            contaminated_score_stack.max(dim=0).values.mean().item()
        ),
        "teacher_hard_pure_score": float(
            pure_score_stack.max(dim=0).values.mean().item()
        ),
    }
    return selected_contaminated, selected_pure, diagnostics


def strongest_absent_rejection(
    contaminated_logits: torch.Tensor,
    pure_logits: torch.Tensor,
    active_targets: torch.Tensor,
) -> torch.Tensor:
    """Penalize the single strongest invalid known response per sample."""
    absent = active_targets < 0.5
    masked = contaminated_logits.masked_fill(~absent, -1e4)
    contaminated_max = masked.max(dim=1).values
    has_absent = absent.any(dim=1)
    contaminated_term = F.softplus(contaminated_max[has_absent]).mean()
    if not bool(has_absent.any()):
        contaminated_term = contaminated_logits.sum() * 0.0
    pure_term = F.softplus(pure_logits.max(dim=1).values).mean()
    # Division keeps this term on a scale comparable with normalized ASL.
    return 0.5 * (contaminated_term + pure_term) / contaminated_logits.shape[1]


@torch.inference_mode()
def validation_metrics_v2(
    student: v1.RDPNet,
    teacher: v1.RDPNet,
    loader,
    active_indices: Sequence[int],
    device: torch.device,
    args: argparse.Namespace,
    valid_threshold: float,
    normal_value: float,
    defect_value: float,
) -> Dict[str, float]:
    student.eval()
    teacher.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 10001)
    clean_correct = 0
    contaminated_correct = 0
    pure_false_known = 0
    contaminated_false_positive = 0
    pure_max_probability_sum = 0.0
    count = 0

    for wafer, labels in loader:
        wafer = wafer.unsqueeze(1).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        truth = labels[:, active_indices] > 0.5
        contaminated, pure, _ = select_hard_candidates(
            teacher,
            wafer,
            truth.float(),
            active_indices,
            args.hard_candidates,
            generator,
            args.pseudo_min_die_fraction,
            args.pseudo_max_die_fraction,
            valid_threshold,
            normal_value,
            defect_value,
        )
        inputs = torch.cat([wafer, contaminated, pure], dim=0)
        logits, _ = v1.forward_features(student, inputs)
        batch_count = len(wafer)
        clean_probability = torch.sigmoid(logits[:batch_count, active_indices])
        contaminated_probability = torch.sigmoid(
            logits[batch_count : 2 * batch_count, active_indices]
        )
        pure_probability = torch.sigmoid(logits[2 * batch_count :, active_indices])
        clean_prediction = clean_probability >= args.label_threshold
        contaminated_prediction = contaminated_probability >= args.label_threshold
        pure_prediction = pure_probability >= args.label_threshold

        clean_correct += int((clean_prediction == truth).all(dim=1).sum().item())
        contaminated_correct += int(
            (contaminated_prediction == truth).all(dim=1).sum().item()
        )
        pure_false_known += int(pure_prediction.any(dim=1).sum().item())
        contaminated_false_positive += int(
            (contaminated_prediction & ~truth).sum().item()
        )
        pure_max_probability_sum += float(
            pure_probability.max(dim=1).values.sum().item()
        )
        count += batch_count

    clean_exact = clean_correct / max(1, count)
    contaminated_exact = contaminated_correct / max(1, count)
    pure_false_rate = pure_false_known / max(1, count)
    pure_max_probability = pure_max_probability_sum / max(1, count)
    contaminated_fp_per_sample = contaminated_false_positive / max(1, count)
    contaminated_weight = 1.0 - args.selection_clean_weight
    score = (
        args.selection_clean_weight * clean_exact
        + contaminated_weight * contaminated_exact
        - args.selection_pure_false_penalty * pure_false_rate
        - args.selection_pure_soft_penalty * pure_max_probability
    )
    return {
        "clean_exact_match": clean_exact,
        "hard_contaminated_exact_match": contaminated_exact,
        "pure_pseudo_false_known_rate": pure_false_rate,
        "pure_pseudo_mean_max_known_probability": pure_max_probability,
        "hard_contaminated_false_positives_per_sample": contaminated_fp_per_sample,
        "selection_score": score,
    }


def save_recoverable(
    path: Path,
    state: Dict[str, torch.Tensor],
    active_indices: Sequence[int],
    unknown_index: int,
    class_names: Sequence[str],
    unknown_class: str,
    best_epoch: int,
    validation: Dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": state,
            "active_known_indices": list(active_indices),
            "inactive_unknown_index": unknown_index,
            "known_classes": list(class_names),
            "unknown_class": unknown_class,
            "full_class_order": list(v1.FULL_CLASSES),
            "label_threshold": 0.5,
            "best_epoch": best_epoch,
            "best_validation": validation,
            "intermediate_training_checkpoint": True,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.pseudo_batch_fraction <= 1.0:
        raise ValueError("--pseudo-batch-fraction must be in (0,1]")
    if not 0.0 < args.pseudo_min_die_fraction <= args.pseudo_max_die_fraction < 1.0:
        raise ValueError("Invalid pseudo die-fraction range")
    if args.hard_candidates < 1:
        raise ValueError("--hard-candidates must be >= 1")
    if not 0.0 <= args.selection_clean_weight <= 1.0:
        raise ValueError("--selection-clean-weight must be in [0,1]")

    v1.set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    x, y = v1.load_arrays(args.data)
    valid_threshold, normal_value, defect_value = v1.data_values(x)
    unknown_index = v1.FULL_CLASSES.index(args.unknown_class)
    active_indices = [
        index for index in range(len(v1.FULL_CLASSES)) if index != unknown_index
    ]
    class_names = [v1.FULL_CLASSES[index] for index in active_indices]
    split = v1.make_split(
        y,
        unknown_index,
        args.known_train_ratio,
        args.known_val_ratio,
        args.seed,
    )
    loaders = {
        name: v1.make_loader(
            x,
            y,
            indices,
            args.batch_size,
            shuffle=name == "known_train",
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            seed=args.seed + offset,
        )
        for offset, (name, indices) in enumerate(split.items())
    }

    base_checkpoint = Path(
        args.base_checkpoint
        or f"checkpoint/rdp_leave_one_out/{args.unknown_class.lower()}_heldout/best_open_set.pt"
    ).resolve()
    output_dir = Path(
        args.output_dir
        or f"checkpoint/rdp_hard_contamination_v21/{args.unknown_class.lower()}_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"

    student, teacher = load_student_and_teacher(
        base_checkpoint, args.unknown_class, active_indices, device
    )
    criterion = v1.AsymmetricLoss()
    optimizer = torch.optim.Adam(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
    )
    train_generator = torch.Generator(device=device).manual_seed(args.seed + 5001)

    print(f"device: {device}")
    print(f"base checkpoint: {base_checkpoint}")
    print(
        f"unknown={args.unknown_class}; active known={class_names}; "
        + ", ".join(f"{name}={len(indices)}" for name, indices in split.items()),
        flush=True,
    )

    # Epoch 0 is the unmodified original RDP. It remains a valid fallback if
    # fine-tuning cannot improve the known-only validation objective.
    initial_validation = validation_metrics_v2(
        student,
        teacher,
        loaders["known_val"],
        active_indices,
        device,
        args,
        valid_threshold,
        normal_value,
        defect_value,
    )
    best_score = initial_validation["selection_score"]
    best_epoch = 0
    best_validation = initial_validation
    best_state = copy.deepcopy(student.state_dict())
    save_recoverable(
        checkpoint_path,
        best_state,
        active_indices,
        unknown_index,
        class_names,
        args.unknown_class,
        best_epoch,
        best_validation,
    )
    print(
        f"epoch 000 | val_clean={initial_validation['clean_exact_match']:.6f} "
        f"val_hard={initial_validation['hard_contaminated_exact_match']:.6f} "
        f"val_pure_false={initial_validation['pure_pseudo_false_known_rate']:.6f} "
        f"val_pure_max={initial_validation['pure_pseudo_mean_max_known_probability']:.6f} "
        f"score={initial_validation['selection_score']:.6f} *",
        flush=True,
    )

    patience_count = 0
    epochs_ran = 0
    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        student.train()
        totals = {
            "total": 0.0,
            "clean": 0.0,
            "contam": 0.0,
            "cons": 0.0,
            "hard": 0.0,
            "kd": 0.0,
            "teacher_contam": 0.0,
            "teacher_pure": 0.0,
        }
        sample_count = 0

        for wafer, labels in loaders["known_train"]:
            wafer = wafer.unsqueeze(1).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pseudo_count = max(
                1,
                min(
                    len(wafer),
                    int(round(len(wafer) * args.pseudo_batch_fraction)),
                ),
            )
            active_targets = labels[:, active_indices]
            pseudo_targets = active_targets[:pseudo_count]

            with torch.inference_mode():
                teacher_clean_logits, _ = v1.forward_features(teacher, wafer)
                contaminated, pure, hard_diagnostics = select_hard_candidates(
                    teacher,
                    wafer[:pseudo_count],
                    pseudo_targets,
                    active_indices,
                    args.hard_candidates,
                    train_generator,
                    args.pseudo_min_die_fraction,
                    args.pseudo_max_die_fraction,
                    valid_threshold,
                    normal_value,
                    defect_value,
                )

            # Candidate selection runs in inference mode. Clone its outputs
            # into ordinary tensors before a trainable convolution needs to
            # save them for weight-gradient computation.
            teacher_clean_logits = teacher_clean_logits.clone()
            contaminated = contaminated.clone()
            pure = pure.clone()

            clean_logits, _ = v1.forward_features(student, wafer)
            pseudo_logits, _ = v1.forward_features(
                student, torch.cat([contaminated, pure], dim=0)
            )
            contaminated_logits = pseudo_logits[:pseudo_count, active_indices]
            pure_logits = pseudo_logits[pseudo_count:, active_indices]
            clean_active = clean_logits[:, active_indices]
            teacher_clean_active = teacher_clean_logits[:, active_indices]

            clean_loss = v1.normalized_asl(criterion, clean_active, active_targets)
            contamination_loss = v1.normalized_asl(
                criterion, contaminated_logits, pseudo_targets
            )
            consistency_loss = F.mse_loss(
                torch.sigmoid(contaminated_logits),
                torch.sigmoid(teacher_clean_active[:pseudo_count]),
            )
            # Preserve teacher confidence only for components that really are
            # present. Negative logits are deliberately not distilled: ASL
            # still supervises known negatives, while the student is free to
            # move those decision boundaries to reject pseudo-unknown input.
            positive_mask = active_targets > 0.5
            if bool(positive_mask.any()):
                distillation_loss = F.smooth_l1_loss(
                    clean_active[positive_mask],
                    teacher_clean_active[positive_mask],
                )
            else:
                distillation_loss = clean_active.sum() * 0.0
            hard_reject_loss = strongest_absent_rejection(
                contaminated_logits, pure_logits, pseudo_targets
            )
            loss = (
                clean_loss
                + args.contamination_loss_weight * contamination_loss
                + args.consistency_loss_weight * consistency_loss
                + args.hard_reject_loss_weight * hard_reject_loss
                + args.distillation_loss_weight * distillation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_count = len(wafer)
            sample_count += batch_count
            for key, value in (
                ("total", float(loss.item())),
                ("clean", float(clean_loss.item())),
                ("contam", float(contamination_loss.item())),
                ("cons", float(consistency_loss.item())),
                ("hard", float(hard_reject_loss.item())),
                ("kd", float(distillation_loss.item())),
                ("teacher_contam", hard_diagnostics["teacher_hard_contam_score"]),
                ("teacher_pure", hard_diagnostics["teacher_hard_pure_score"]),
            ):
                totals[key] += value * batch_count

        scheduler.step()
        validation = validation_metrics_v2(
            student,
            teacher,
            loaders["known_val"],
            active_indices,
            device,
            args,
            valid_threshold,
            normal_value,
            defect_value,
        )
        improved = validation["selection_score"] > best_score
        if improved:
            best_score = validation["selection_score"]
            best_epoch = epoch
            best_validation = validation
            best_state = copy.deepcopy(student.state_dict())
            patience_count = 0
            save_recoverable(
                checkpoint_path,
                best_state,
                active_indices,
                unknown_index,
                class_names,
                args.unknown_class,
                best_epoch,
                best_validation,
            )
        else:
            patience_count += 1

        means = {key: value / max(1, sample_count) for key, value in totals.items()}
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} total={means['total']:.6f} "
            f"clean={means['clean']:.6f} contam={means['contam']:.6f} "
            f"cons={means['cons']:.6f} hard={means['hard']:.6f} "
            f"kd={means['kd']:.6f} teacher_hard={means['teacher_contam']:.4f}/"
            f"{means['teacher_pure']:.4f} lr={current_lr:.2e} | "
            f"val_clean={validation['clean_exact_match']:.6f} "
            f"val_hard={validation['hard_contaminated_exact_match']:.6f} "
            f"val_pure_false={validation['pure_pseudo_false_known_rate']:.6f} "
            f"val_pure_max={validation['pure_pseudo_mean_max_known_probability']:.6f} "
            f"score={validation['selection_score']:.6f} "
            f"patience={patience_count}/{args.patience}"
            + (" *" if improved else ""),
            flush=True,
        )
        if args.patience > 0 and patience_count >= args.patience:
            print("early stopping", flush=True)
            break

    student.load_state_dict(best_state, strict=True)
    collected = {}
    for name in ("known_train", "known_val", "known_test", "unknown_test"):
        print(f"scoring {name}", flush=True)
        collected[name] = v1.collect_predictions(student, loaders[name], device)
    _, train_features, train_targets = collected["known_train"]
    _, val_features, _ = collected["known_val"]
    known_logits, known_features, known_targets = collected["known_test"]
    unknown_logits, unknown_features, unknown_targets = collected["unknown_test"]

    prototypes, prototype_codes = v1.build_prototypes(
        train_features, train_targets, active_indices
    )
    val_scores = v1.prototype_score(val_features, prototypes)
    known_scores = v1.prototype_score(known_features, prototypes)
    unknown_scores = v1.prototype_score(unknown_features, prototypes)
    unknown_threshold = float(
        np.quantile(val_scores, 1.0 - args.known_reject_rate)
    )
    detection_labels = np.concatenate(
        [
            np.zeros(len(known_scores), dtype=np.int64),
            np.ones(len(unknown_scores), dtype=np.int64),
        ]
    )
    detection_scores = np.concatenate([known_scores, unknown_scores])
    known_reject = float(np.mean(known_scores > unknown_threshold))
    unknown_recall = float(np.mean(unknown_scores > unknown_threshold))

    known_metrics = v1.multilabel_metrics(
        known_logits,
        known_targets,
        active_indices,
        class_names,
        args.label_threshold,
    )
    unknown_metrics = v1.multilabel_metrics(
        unknown_logits,
        unknown_targets,
        active_indices,
        class_names,
        args.label_threshold,
    )
    configuration = {
        "method": "hard_contamination_v2_1_positive_distillation",
        "full_class_order": list(v1.FULL_CLASSES),
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "unknown_index": unknown_index,
        "base_checkpoint": str(base_checkpoint),
        "initialization": "known-only RDP plus frozen teacher",
        "distillation_scope": "positive known labels only",
        "heldout_data_used_for_training_or_selection": False,
        "label_threshold": args.label_threshold,
        "known_reject_rate": args.known_reject_rate,
        "seed": args.seed,
        "epochs": args.epochs,
        "epochs_ran": epochs_ran,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "hard_candidates": args.hard_candidates,
        "pseudo_batch_fraction": args.pseudo_batch_fraction,
        "pseudo_min_die_fraction": args.pseudo_min_die_fraction,
        "pseudo_max_die_fraction": args.pseudo_max_die_fraction,
        "contamination_loss_weight": args.contamination_loss_weight,
        "consistency_loss_weight": args.consistency_loss_weight,
        "hard_reject_loss_weight": args.hard_reject_loss_weight,
        "distillation_loss_weight": args.distillation_loss_weight,
        "selection_clean_weight": args.selection_clean_weight,
        "selection_pure_false_penalty": args.selection_pure_false_penalty,
        "selection_pure_soft_penalty": args.selection_pure_soft_penalty,
        "pseudo_unknown_source": (
            "known-training wafers plus teacher-selected online irregular masks; "
            "held-out data unused"
        ),
    }
    result = {
        "configuration": configuration,
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "training": {
            "best_epoch": best_epoch,
            "best_validation": best_validation,
        },
        "known_test_components": known_metrics,
        "unknown_test_known_projection": unknown_metrics,
        "unknown_detection": {
            "auroc": float(roc_auc_score(detection_labels, detection_scores)),
            "aupr": float(
                average_precision_score(detection_labels, detection_scores)
            ),
            "fpr95": v1.fpr_at_tpr95(detection_labels, detection_scores),
            "threshold": unknown_threshold,
            "known_reject_rate": known_reject,
            "unknown_recall": unknown_recall,
            "balanced_accuracy_at_threshold": float(
                0.5 * ((1.0 - known_reject) + unknown_recall)
            ),
            "role": "diagnostic only; final unknown gate remains Dinomaly",
        },
    }
    checkpoint_payload = {
        "model_state_dict": best_state,
        "output_classes": list(v1.FULL_CLASSES),
        "active_known_indices": active_indices,
        "inactive_unknown_index": unknown_index,
        "known_classes": class_names,
        "unknown_class": args.unknown_class,
        "full_class_order": list(v1.FULL_CLASSES),
        "prototypes": prototypes,
        "prototype_codes": prototype_codes,
        "unknown_threshold": unknown_threshold,
        "label_threshold": args.label_threshold,
        "training_configuration": configuration,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {checkpoint_path}")
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()
