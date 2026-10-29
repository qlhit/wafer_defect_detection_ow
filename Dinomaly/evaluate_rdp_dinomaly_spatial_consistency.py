#!/usr/bin/env python3
"""Diagnose spatial consistency between RDP class evidence and Dinomaly anomaly maps.

This program does not train or modify either model. It calibrates the Dinomaly
image/pixel thresholds using known validation wafers only, creates Grad-CAMs at
RDP ``block_3_2`` for thresholded candidate labels, and tests whether false
known labels place more CAM mass inside the Dinomaly unknown region than true
known labels. Only one JSON file is written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

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
from train_eval_rdp_conditioned_dinomaly import (
    RDPConditionedDinomaly,
    load_frozen_rdp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RDP Grad-CAM versus Dinomaly anomaly-map diagnosis"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--unknown-class", choices=FULL_CLASSES, default=None)
    parser.add_argument("--rdp-repo", default=None)
    parser.add_argument("--rdp-checkpoint", default=None)
    parser.add_argument(
        "--rdp-threshold-json",
        default=None,
        help="Default: rdp_threshold_calibration.json beside checkpoint",
    )
    parser.add_argument(
        "--rdp-threshold-scheme",
        choices=("fixed_0_5", "known_val_max_f1", "known_val_fpr_control"),
        default="known_val_max_f1",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--calibration-n", type=int, default=1000)
    parser.add_argument("--unknown-eval-n", type=int, default=1000)
    parser.add_argument("--known-reject-rate", type=float, default=0.05)
    parser.add_argument(
        "--known-pixel-tail-rate",
        type=float,
        default=0.01,
        help="Known-validation valid-die fraction above the pixel threshold",
    )
    parser.add_argument(
        "--cam-top-fraction",
        type=float,
        default=0.20,
        help="Fraction of valid die retained in the binary CAM region",
    )
    parser.add_argument("--seed", type=int, default=3529)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_thresholds(
    path: Path,
    scheme: str,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, str]:
    if not path.is_file():
        if scheme != "fixed_0_5":
            raise FileNotFoundError(
                f"RDP threshold calibration not found: {path}"
            )
        return np.full(len(class_names), 0.5, dtype=np.float32), "fixed_0_5"
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold_dict = payload["threshold_schemes"][scheme]["thresholds"]
    return (
        np.asarray([threshold_dict[name] for name in class_names], dtype=np.float32),
        scheme,
    )


def raw_batch(
    x: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return x[indices.to(torch.long)].unsqueeze(1).to(device, non_blocking=True)


def rdp_forward_at_block3(
    model: torch.nn.Module, wafer: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # The early frozen RDP path needs no graph. The detached block_3_2 output
    # becomes the Grad-CAM target, so only block4/fc are differentiated.
    with torch.no_grad():
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
    feature = x.detach().requires_grad_(True)
    x = model.block_4_1(feature)
    x = model.block_4_2(x)
    x = torch.flatten(model.avgpool(x), 1)
    logits = model.fc(x)
    return logits, feature


def gradcams_for_active_classes(
    logits: torch.Tensor,
    feature: torch.Tensor,
    active_indices: Sequence[int],
    needed_local_classes: Sequence[int],
    output_size: Tuple[int, int],
) -> Dict[int, torch.Tensor]:
    cams: Dict[int, torch.Tensor] = {}
    needed = sorted(set(int(index) for index in needed_local_classes))
    for position, local_index in enumerate(needed):
        global_index = int(active_indices[local_index])
        gradient = torch.autograd.grad(
            logits[:, global_index].sum(),
            feature,
            retain_graph=position < len(needed) - 1,
            create_graph=False,
        )[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feature).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        cams[local_index] = cam[:, 0].detach()
    return cams


@torch.inference_mode()
def dinomaly_maps(
    model: RDPConditionedDinomaly,
    image: torch.Tensor,
    condition: torch.Tensor,
    image_size: int,
    gaussian: torch.nn.Module,
    raw_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded, decoded = model(image, condition)
    anomaly_map, _ = cal_anomaly_maps(encoded, decoded, out_size=image_size)
    anomaly_map = gaussian(anomaly_map.float())
    raw_map = F.interpolate(
        anomaly_map,
        size=raw_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return anomaly_map[:, 0], raw_map


def wafer_valid_mask(raw: torch.Tensor) -> torch.Tensor:
    # MixedWM38 is normally encoded as 0/1/2. This also supports normalized
    # 0/0.5/1 arrays without changing the wafer-valid decision.
    maximum = float(raw.detach().max().item())
    threshold = 0.25 if maximum <= 1.0 else 0.5
    return raw[:, 0] > threshold


def image_scores(anomaly_map: torch.Tensor, top_fraction: float) -> torch.Tensor:
    flattened = anomaly_map.flatten(1)
    keep = max(1, int(math.ceil(flattened.shape[1] * top_fraction)))
    return torch.topk(flattened, k=keep, dim=1).values.mean(dim=1)


def top_mask(
    values: torch.Tensor,
    valid: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    output = torch.zeros_like(valid, dtype=torch.bool)
    for batch_index in range(len(values)):
        positions = torch.nonzero(valid[batch_index].flatten(), as_tuple=False)[:, 0]
        if len(positions) == 0:
            continue
        keep = max(1, int(math.ceil(len(positions) * fraction)))
        flat_values = values[batch_index].flatten()[positions]
        chosen = positions[torch.topk(flat_values, k=keep).indices]
        output[batch_index].view(-1)[chosen] = True
    return output


def spatial_record(
    cam: torch.Tensor,
    unknown_mask: torch.Tensor,
    valid: torch.Tensor,
    cam_top_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = F.relu(cam) * valid.float()
    cam_mass = cam.flatten(1).sum(dim=1)
    mass_overlap = (
        (cam * unknown_mask.float()).flatten(1).sum(dim=1)
        / (cam_mass + 1e-8)
    )
    cam_binary = top_mask(cam, valid, cam_top_fraction)
    binary_overlap = (
        (cam_binary & unknown_mask).flatten(1).sum(dim=1).float()
        / (cam_binary.flatten(1).sum(dim=1).float() + 1e-8)
    )
    unknown_fraction = (
        unknown_mask.flatten(1).sum(dim=1).float()
        / (valid.flatten(1).sum(dim=1).float() + 1e-8)
    )
    enrichment = mass_overlap / (unknown_fraction + 1e-8)
    return (
        mass_overlap.detach().cpu().numpy(),
        binary_overlap.detach().cpu().numpy(),
        enrichment.detach().cpu().numpy(),
    )


def safe_mean(values: np.ndarray):
    return float(values.mean()) if len(values) else None


def summarize_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {"candidate_labels": 0}
    truth = np.asarray([bool(item["is_true_label"]) for item in records])
    false_label = ~truth
    result: Dict[str, object] = {
        "candidate_labels": int(len(records)),
        "true_candidate_labels": int(truth.sum()),
        "false_candidate_labels": int(false_label.sum()),
    }
    for score_name in (
        "cam_mass_in_unknown_region",
        "cam_top_region_overlap",
        "cam_unknown_enrichment",
    ):
        scores = np.asarray([float(item[score_name]) for item in records])
        score_result: Dict[str, object] = {
            "true_label_mean": safe_mean(scores[truth]),
            "false_label_mean": safe_mean(scores[false_label]),
        }
        if truth.any() and false_label.any():
            labels = false_label.astype(np.int64)
            score_result["false_label_auroc"] = float(
                roc_auc_score(labels, scores)
            )
            score_result["false_label_aupr"] = float(
                average_precision_score(labels, scores)
            )
            # Diagnostic/oracle threshold only: it uses target labels and must
            # not be deployed or reported as a target-independent threshold.
            threshold = float(np.quantile(scores[truth], 0.95))
            score_result["oracle_threshold_retaining_95pct_true"] = threshold
            score_result["true_label_retention"] = float(
                np.mean(scores[truth] <= threshold)
            )
            score_result["false_label_rejection"] = float(
                np.mean(scores[false_label] > threshold)
            )
        result[score_name] = score_result

    per_class: Dict[str, object] = {}
    for class_name in sorted(set(str(item["class_name"]) for item in records)):
        selected = [item for item in records if item["class_name"] == class_name]
        selected_truth = np.asarray(
            [bool(item["is_true_label"]) for item in selected]
        )
        scores = np.asarray(
            [float(item["cam_mass_in_unknown_region"]) for item in selected]
        )
        per_class[class_name] = {
            "candidate_labels": int(len(selected)),
            "true_labels": int(selected_truth.sum()),
            "false_labels": int((~selected_truth).sum()),
            "true_label_mean_overlap": safe_mean(scores[selected_truth]),
            "false_label_mean_overlap": safe_mean(scores[~selected_truth]),
        }
    result["per_class"] = per_class
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.known_reject_rate < 1.0:
        raise ValueError("--known-reject-rate must be in (0,1)")
    if not 0.0 < args.known_pixel_tail_rate < 1.0:
        raise ValueError("--known-pixel-tail-rate must be in (0,1)")
    if not 0.0 < args.cam_top_fraction <= 1.0:
        raise ValueError("--cam-top-fraction must be in (0,1]")

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
    class_names = [FULL_CLASSES[index] for index in active_indices]
    if [int(index) for index in configuration["active_known_indices"]] != active_indices:
        raise ValueError("Conditioned checkpoint class order is inconsistent")

    data_path = args.data or str(configuration["data"])
    rdp_repo = Path(args.rdp_repo or str(configuration["rdp_repo"])).resolve()
    rdp_checkpoint = Path(
        args.rdp_checkpoint or str(configuration["rdp_checkpoint"])
    ).resolve()
    threshold_json = Path(
        args.rdp_threshold_json
        or checkpoint_path.parent / "rdp_threshold_calibration.json"
    ).resolve()
    thresholds, threshold_scheme = load_thresholds(
        threshold_json, args.rdp_threshold_scheme, class_names
    )

    seed = int(configuration["seed"])
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = load_arrays(data_path)
    split = make_open_set_split(y, unknown_index, 0.70, 0.15, seed)
    selected = {
        "known_val": deterministic_subset(
            split["known_val"], args.calibration_n, seed + 1
        ),
        "unknown_test": deterministic_subset(
            split["unknown_test"], args.unknown_eval_n, seed + 3
        ),
    }
    loaders = {
        name: make_loader(
            MixedWM38DinomalyDataset(
                x, y, indices, image_size=int(configuration["image_size"])
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=seed + offset,
        )
        for offset, (name, indices) in enumerate(selected.items())
    }

    print(f"device: {device}")
    print(
        f"spatial diagnosis: unknown={unknown_class}, "
        f"known_val={len(selected['known_val'])}, "
        f"unknown_test={len(selected['unknown_test'])}",
        flush=True,
    )
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
    bottleneck.load_state_dict(checkpoint["bottleneck_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    model.conditioner.load_state_dict(
        checkpoint["conditioner_state_dict"], strict=True
    )
    model.eval()
    rdp_model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    image_top_fraction = float(configuration["top_fraction"])
    image_size = int(configuration["image_size"])
    known_pixel_values: List[np.ndarray] = []
    known_image_scores: List[np.ndarray] = []
    print("calibrating Dinomaly thresholds on known_val", flush=True)
    for image, _labels, indices in loaders["known_val"]:
        image = image.to(device, non_blocking=True)
        raw = raw_batch(x, indices, device)
        with torch.no_grad():
            logits, _feature = rdp_forward_at_block3(rdp_model, raw)
            condition = torch.sigmoid(logits[:, active_indices])
            anomaly_full, anomaly = dinomaly_maps(
                model,
                image,
                condition,
                image_size,
                gaussian,
                raw.shape[-2:],
            )
        valid = wafer_valid_mask(raw)
        known_pixel_values.append(anomaly[valid].detach().cpu().numpy())
        known_image_scores.append(
            image_scores(anomaly_full, image_top_fraction).detach().cpu().numpy()
        )
    pixel_values = np.concatenate(known_pixel_values)
    calibration_scores = np.concatenate(known_image_scores)
    pixel_threshold = float(
        np.quantile(pixel_values, 1.0 - args.known_pixel_tail_rate)
    )
    image_threshold = float(
        np.quantile(calibration_scores, 1.0 - args.known_reject_rate)
    )
    del known_pixel_values, known_image_scores, pixel_values

    all_records: List[Dict[str, object]] = []
    gated_records: List[Dict[str, object]] = []
    unknown_gate_decisions: List[np.ndarray] = []
    unknown_region_fractions: List[np.ndarray] = []
    native_feature_shape = None
    print("scoring unknown_test and creating RDP Grad-CAMs", flush=True)
    threshold_tensor = torch.as_tensor(thresholds, device=device).view(1, -1)
    for image, labels, indices in loaders["unknown_test"]:
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        raw = raw_batch(x, indices, device)
        logits, feature = rdp_forward_at_block3(rdp_model, raw)
        if native_feature_shape is None:
            native_feature_shape = list(feature.shape[1:])
            print(f"RDP block_3_2 feature shape: {native_feature_shape}", flush=True)
        probability = torch.sigmoid(logits[:, active_indices]).detach()
        predicted = probability >= threshold_tensor
        needed_classes = torch.nonzero(
            predicted.any(dim=0), as_tuple=False
        ).flatten().tolist()
        cams = gradcams_for_active_classes(
            logits,
            feature,
            active_indices,
            needed_classes,
            raw.shape[-2:],
        )
        with torch.no_grad():
            anomaly_full, anomaly = dinomaly_maps(
                model,
                image,
                probability,
                image_size,
                gaussian,
                raw.shape[-2:],
            )
        valid = wafer_valid_mask(raw)
        unknown_mask = (anomaly > pixel_threshold) & valid
        gate = image_scores(anomaly_full, image_top_fraction) > image_threshold
        unknown_gate_decisions.append(gate.detach().cpu().numpy())
        unknown_region_fractions.append(
            (
                unknown_mask.flatten(1).sum(dim=1).float()
                / (valid.flatten(1).sum(dim=1).float() + 1e-8)
            )
            .detach()
            .cpu()
            .numpy()
        )

        truth = labels[:, active_indices] > 0.5
        for local_index in needed_classes:
            candidate_mask = predicted[:, local_index]
            mass, binary, enrichment = spatial_record(
                cams[local_index],
                unknown_mask,
                valid,
                args.cam_top_fraction,
            )
            candidate_positions = torch.nonzero(
                candidate_mask, as_tuple=False
            ).flatten().tolist()
            for batch_index in candidate_positions:
                record: Dict[str, object] = {
                    "class_name": class_names[local_index],
                    "is_true_label": bool(truth[batch_index, local_index].item()),
                    "unknown_gate_positive": bool(gate[batch_index].item()),
                    "cam_mass_in_unknown_region": float(mass[batch_index]),
                    "cam_top_region_overlap": float(binary[batch_index]),
                    "cam_unknown_enrichment": float(enrichment[batch_index]),
                }
                all_records.append(record)
                if record["unknown_gate_positive"]:
                    gated_records.append(record)

    gate_values = np.concatenate(unknown_gate_decisions)
    region_values = np.concatenate(unknown_region_fractions)
    result = {
        "configuration": {
            "checkpoint": str(checkpoint_path),
            "data": data_path,
            "unknown_class": unknown_class,
            "known_classes": class_names,
            "rdp_checkpoint": str(rdp_checkpoint),
            "rdp_threshold_json": str(threshold_json),
            "rdp_threshold_scheme": threshold_scheme,
            "rdp_thresholds": {
                name: float(thresholds[index])
                for index, name in enumerate(class_names)
            },
            "device": str(device),
            "calibration_samples": int(len(selected["known_val"])),
            "unknown_samples": int(len(selected["unknown_test"])),
            "known_reject_rate": args.known_reject_rate,
            "known_pixel_tail_rate": args.known_pixel_tail_rate,
            "cam_top_fraction": args.cam_top_fraction,
            "rdp_gradcam_layer": "block_3_2",
            "rdp_gradcam_native_feature_shape": native_feature_shape,
        },
        "known_only_calibration": {
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
        },
        "unknown_gate": {
            "recall_on_held_out_unknown": float(gate_values.mean()),
            "positive_samples": int(gate_values.sum()),
            "samples": int(len(gate_values)),
        },
        "unknown_region": {
            "mean_valid_die_fraction": float(region_values.mean()),
            "q50_valid_die_fraction": float(np.quantile(region_values, 0.50)),
            "q95_valid_die_fraction": float(np.quantile(region_values, 0.95)),
        },
        "spatial_consistency_all_unknown_samples": summarize_records(all_records),
        "spatial_consistency_unknown_gate_positive_only": summarize_records(
            gated_records
        ),
        "notes": {
            "purpose": "diagnostic only; no label is removed by this program",
            "threshold_source": "Dinomaly image and pixel thresholds use known validation only",
            "false_label_auroc": "uses held-out target labels only to measure feasibility",
            "oracle_overlap_threshold": "target-label diagnostic; not a deployable threshold",
        },
    }
    output_path = Path(
        args.output
        or checkpoint_path.parent / "spatial_consistency_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"unknown gate recall={result['unknown_gate']['recall_on_held_out_unknown']:.6f}",
        flush=True,
    )
    for name, summary in (
        ("all", result["spatial_consistency_all_unknown_samples"]),
        ("gate_positive", result["spatial_consistency_unknown_gate_positive_only"]),
    ):
        primary = summary.get("cam_mass_in_unknown_region", {})
        print(
            f"{name}: candidates={summary.get('candidate_labels', 0)} "
            f"true_mean={primary.get('true_label_mean')} "
            f"false_mean={primary.get('false_label_mean')} "
            f"false_AUROC={primary.get('false_label_auroc')} "
            f"false_rejection_at_95pct_true_retention={primary.get('false_label_rejection')}",
            flush=True,
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
