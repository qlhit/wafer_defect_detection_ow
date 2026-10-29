#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch

# Allow running scripts without installing the package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdpnet.data import make_dataloaders
from rdpnet.eval import evaluate
from rdpnet.models.rdpnet import RDPNet


def main():
    p = argparse.ArgumentParser(description="Evaluate RDPNet on Wafer dataset")
    p.add_argument("--data", default="Wafer_Map_Datasets.npz", help="Path to dataset .npz")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint .mdl")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--logits", action="store_true", help="Checkpoint expects sigmoid at eval time (trained with ASL)")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, val_loader = make_dataloaders(args.data, batch_size=args.batch_size, val_split=0.2)

    model = RDPNet(output_dim=8, in_channels=1, apply_sigmoid=not args.logits).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    acc, details = evaluate(
        model,
        val_loader,
        device=device,
        threshold=args.threshold,
        return_details=True,
        from_logits=args.logits,
    )
    print(f"Val acc (mean over C1..C38): {acc:.6f}")
    if details:
        print(f"Precision mean: {details['precision_mean']:.3f}")
        print(f"Recall mean:    {details['recall_mean']:.3f}")
        print(f"F1 mean:        {details['f1_mean']:.3f}")
        # Avoid printing full confusion matrix for large output in CLI
        cm = details["confusion_matrix"]
        print("CM shape:", np.array(cm).shape)


if __name__ == "__main__":
    main()
