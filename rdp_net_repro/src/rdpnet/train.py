import csv
import os
from typing import Optional, Dict, Any

import torch
from torch import nn, optim
from tqdm import tqdm

from .data import make_dataloaders
from .eval import evaluate
from .losses import AsymmetricLoss
from .models.rdpnet import RDPNet
from .utils import set_seed


def train(
    npz_path: str = "Wafer_Map_Datasets.npz",
    output_dim: int = 8,
    in_channels: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 100,
    val_split: float = 0.2,
    device: Optional[torch.device] = None,
    use_asl: bool = False,
    ckpt_dir: str = "checkpoint",
    resume: Optional[str] = None,
    seed: int = 3529,
    train_log_csv: Optional[str] = "train_data.csv",
) -> Dict[str, Any]:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(npz_path, batch_size=batch_size, val_split=val_split, seed=seed)

    model = RDPNet(output_dim=output_dim, in_channels=in_channels, apply_sigmoid=not use_asl).to(device)

    if use_asl:
        criterion = AsymmetricLoss()
    else:
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if resume and os.path.isfile(resume):
        model.load_state_dict(torch.load(resume, map_location=device))

    os.makedirs(ckpt_dir, exist_ok=True)

    best_acc = 0.0
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, leave=False)
        for x, y in train_bar:
            x = x.unsqueeze(1).to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_description(f"train epoch[{epoch+1}/{epochs}] loss:{loss:.3f}")

        val_acc, _ = evaluate(model, val_loader, device=device, return_details=False, from_logits=use_asl)

        if train_log_csv:
            with open(train_log_csv, mode="a", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss.detach().item()])

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch{epoch}.mdl"))
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.mdl"))

    return {"best_acc": best_acc, "best_epoch": best_epoch, "ckpt_dir": ckpt_dir}
