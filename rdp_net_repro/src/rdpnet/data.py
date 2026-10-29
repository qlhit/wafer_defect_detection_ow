import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_npz_dataset(npz_path: str) -> TensorDataset:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    raw = np.load(npz_path)
    data = raw["arr_0"].astype(np.float32)
    labels = raw["arr_1"].astype(np.float32)

    x = torch.from_numpy(data)
    y = torch.from_numpy(labels)
    return TensorDataset(x, y)


def make_dataloaders(
    npz_path: str,
    batch_size: int = 128,
    val_split: float = 0.2,
    seed: int = 3529,
) -> Tuple[DataLoader, DataLoader]:
    dataset = load_npz_dataset(npz_path)
    data_len = len(dataset)
    val_len = int(data_len * val_split)
    train_len = data_len - val_len

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

