import sys
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .utils import label2name


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    return_details: bool = False,
    from_logits: bool = False,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    model.eval()

    correct = {f"C{i}": 0 for i in range(39)}
    data_num = {f"C{i}": 0 for i in range(39)}

    cm = np.zeros((39, 39), dtype=np.int64)

    with torch.no_grad():
        val_bar = tqdm(loader, file=sys.stdout, leave=False)
        for x, y in val_bar:
            # input data expected shape: (B,H,W) -> add channel
            x = x.unsqueeze(dim=1).to(device)
            y = y.to(device)

            outputs = model(x)
            if from_logits:
                outputs = torch.sigmoid(outputs)
            pred = (outputs >= threshold).int()

            for i in range(x.shape[0]):
                true_label = y[i].tolist()
                pred_label = pred[i].tolist()

                true_name = label2name([int(v) for v in true_label])
                pred_name = label2name([int(v) for v in pred_label])

                data_num[true_name] += 1
                if torch.equal(pred[i].int(), y[i].int()):
                    correct[true_name] += 1

                try:
                    t_idx = int(true_name[1:])
                    p_idx = int(pred_name[1:])
                except ValueError:
                    t_idx = 0
                    p_idx = 0
                cm[t_idx, p_idx] += 1

    # per-class metrics for C1..C38
    acc_sum = 0.0
    per_class_acc = {}
    precisions = []
    recalls = []
    f1s = []
    for i in range(1, 39):
        name = f"C{i}"
        if data_num[name] > 0:
            acc = correct[name] / data_num[name]
        else:
            acc = 0.0
        per_class_acc[name] = acc
        acc_sum += acc

        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        # tn not needed for p/r/f1 here
        pre = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * pre * rec / (pre + rec)) if (pre + rec) > 0 else 0.0
        precisions.append(pre)
        recalls.append(rec)
        f1s.append(f1)

    overall = acc_sum / 38.0

    if not return_details:
        return overall, None

    details = {
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "precision_mean": float(np.mean(np.array(precisions))) if len(precisions) else 0.0,
        "recall_mean": float(np.mean(np.array(recalls))) if len(recalls) else 0.0,
        "f1_mean": float(np.mean(np.array(f1s))) if len(f1s) else 0.0,
    }
    return overall, details
