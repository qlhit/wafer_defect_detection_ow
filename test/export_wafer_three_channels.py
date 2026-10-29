#!/usr/bin/env python3
"""Export one original wafer map and its three repeated grayscale channels.

This script visualizes the gray-repeat input used before a three-channel
pretrained encoder such as DINOv2. The original image retains its source
resolution, while the three exported channel images use the requested encoder
input size and should be identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image


PREFERRED_IMAGE_KEYS = (
    "images",
    "image",
    "wafer_maps",
    "wafer_map",
    "waferMap",
    "x",
    "X",
    "data",
    "arr_0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one wafer map from an NPZ file, export its original image, "
            "and export the three identical grayscale channels used by "
            "gray-repeat encoding."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to the NPZ file.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based sample index in the selected image array. Default: 0.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="NPZ key containing wafer maps. If omitted, it is detected automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wafer_three_channels"),
        help="Directory for the three PNG files.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=392,
        help=(
            "Output height and width. Nearest-neighbor resizing is used. "
            "Set to 0 to retain the original resolution. Default: 392."
        ),
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert grayscale only for visualization.",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print every NPZ key and shape, then exit.",
    )
    return parser.parse_args()


def describe_npz(npz: np.lib.npyio.NpzFile) -> None:
    print("NPZ arrays:")
    for key in npz.files:
        array = npz[key]
        print(f"  {key}: shape={array.shape}, dtype={array.dtype}")


def image_candidate(array: np.ndarray) -> bool:
    """Return whether an array can reasonably contain wafer images."""
    if not np.issubdtype(array.dtype, np.number):
        return False
    if array.ndim == 2:
        return min(array.shape) > 8
    if array.ndim == 3:
        return min(array.shape[-2:]) > 8
    if array.ndim == 4:
        return (
            array.shape[1] in (1, 3)
            or array.shape[-1] in (1, 3)
        ) and max(array.shape[-2:]) > 8
    return False


def candidate_size(array: np.ndarray) -> int:
    """Estimate spatial size so labels are not mistaken for images."""
    if array.ndim == 2:
        return int(np.prod(array.shape))
    if array.ndim == 3:
        return int(np.prod(array.shape[-2:]))
    if array.ndim == 4:
        if array.shape[1] in (1, 3):
            return int(np.prod(array.shape[-2:]))
        return int(np.prod(array.shape[1:3]))
    return 0


def select_image_array(
    npz: np.lib.npyio.NpzFile, requested_key: str | None
) -> Tuple[str, np.ndarray]:
    if requested_key is not None:
        if requested_key not in npz.files:
            available = ", ".join(npz.files)
            raise KeyError(
                f"Image key {requested_key!r} was not found. Available keys: {available}"
            )
        array = np.asarray(npz[requested_key])
        if not image_candidate(array):
            raise ValueError(
                f"Array {requested_key!r} with shape {array.shape} does not look "
                "like a wafer-image array."
            )
        return requested_key, array

    candidates: Dict[str, np.ndarray] = {
        key: np.asarray(npz[key])
        for key in npz.files
        if image_candidate(np.asarray(npz[key]))
    }
    if not candidates:
        raise ValueError(
            "No wafer-image array was detected. Run with --list-keys and then "
            "pass the correct key using --image-key."
        )

    for preferred_key in PREFERRED_IMAGE_KEYS:
        if preferred_key in candidates:
            return preferred_key, candidates[preferred_key]

    selected_key = max(candidates, key=lambda key: candidate_size(candidates[key]))
    return selected_key, candidates[selected_key]


def extract_single_wafer(array: np.ndarray, index: int) -> np.ndarray:
    """Extract one HxW wafer from common NPZ image layouts."""
    if array.ndim == 2:
        if index != 0:
            raise IndexError("The selected array contains one image, so --index must be 0.")
        wafer = array
    else:
        sample_count = array.shape[0]
        if not 0 <= index < sample_count:
            raise IndexError(
                f"--index {index} is outside the valid range 0 to {sample_count - 1}."
            )
        wafer = array[index]

    wafer = np.asarray(wafer)
    if wafer.ndim == 3 and wafer.shape[0] in (1, 3):
        wafer = wafer[0]
    elif wafer.ndim == 3 and wafer.shape[-1] in (1, 3):
        wafer = wafer[..., 0]

    if wafer.ndim != 2:
        raise ValueError(
            f"One sample must reduce to HxW, but its final shape is {wafer.shape}."
        )
    return wafer.astype(np.float32, copy=False)


def normalize_for_display(wafer: np.ndarray, invert: bool) -> np.ndarray:
    """Convert the raw wafer values to uint8 without altering spatial structure."""
    finite = np.isfinite(wafer)
    if not finite.any():
        raise ValueError("The selected wafer contains no finite pixel values.")

    valid = wafer[finite]
    minimum = float(valid.min())
    maximum = float(valid.max())

    normalized = np.zeros_like(wafer, dtype=np.float32)
    if maximum > minimum:
        normalized[finite] = (wafer[finite] - minimum) / (maximum - minimum)
    elif maximum != 0:
        normalized[finite] = 1.0

    if invert:
        normalized = 1.0 - normalized

    normalized[~finite] = 0.0
    return np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def resize_nearest(image: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        return image
    pil_image = Image.fromarray(image, mode="L")
    pil_image = pil_image.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    return np.asarray(pil_image)


def save_outputs(
    original: np.ndarray,
    channel: np.ndarray,
    output_dir: Path,
    index: int,
) -> Iterable[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / f"wafer_{index:05d}_original.png"
    Image.fromarray(original, mode="L").save(original_path)
    saved_paths = [original_path]
    for channel_index in range(1, 4):
        output_path = output_dir / f"wafer_{index:05d}_channel_{channel_index}.png"
        Image.fromarray(channel, mode="L").save(output_path)
        saved_paths.append(output_path)
    return saved_paths


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"NPZ file not found: {args.data}")

    with np.load(args.data, allow_pickle=False) as npz:
        if args.list_keys:
            describe_npz(npz)
            return
        image_key, image_array = select_image_array(npz, args.image_key)
        wafer = extract_single_wafer(image_array, args.index)

    original = normalize_for_display(wafer, args.invert)
    channel = resize_nearest(original, args.image_size)
    saved_paths = list(
        save_outputs(original, channel, args.output_dir, args.index)
    )

    stacked = np.stack([channel, channel, channel], axis=0)
    print(f"image key: {image_key}")
    print(f"sample index: {args.index}")
    print(f"raw wafer shape: {wafer.shape}")
    print(f"saved original shape: {original.shape}")
    print(f"saved channel shape: {channel.shape}")
    print(f"three channels identical: {bool(np.all(stacked[0] == stacked[1]) and np.all(stacked[1] == stacked[2]))}")
    for path in saved_paths:
        print(f"saved: {path.resolve()}")


if __name__ == "__main__":
    main()