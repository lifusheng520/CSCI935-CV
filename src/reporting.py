"""Metrics and visual diagnostics; no GUI dependency."""
import csv
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .data import CLASSES, MEAN, STD


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.inference_mode()
def evaluate(model, loader, device, output):
    model.eval()
    rows, samples = [], []
    for batch in loader:
        out = model(batch["image"].to(device), batch["mask"].to(device), batch["quality"].to(device))
        probs, weights = out["log_prob"].exp().cpu().numpy(), out["weights"].cpu().numpy()
        for i, path in enumerate(batch["path"]):
            pred, target = int(probs[i].argmax()), int(batch["label"][i])
            rows.append(dict(path=path, domain=batch["domain"][i], target=target, prediction=pred,
                             target_name=CLASSES[target], prediction_name=CLASSES[pred],
                             confidence=float(probs[i, pred]), quality=float(batch["quality"][i]),
                             mask_valid=bool(out["valid"][i]),
                             **{f"weight_{b}": float(weights[i, j]) for j, b in enumerate(("global", "local", "mask"))},
                             **{f"p_{name}": float(probs[i, j]) for j, name in enumerate(CLASSES)}))
            # Representative first case of each domain/class, independent of prediction correctness.
            key = (batch["domain"][i], target)
            if key not in [s[0] for s in samples]:
                rgb = np.clip((batch["image"][i].numpy().transpose(1, 2, 0) * STD + MEAN), 0, 1)
                samples.append((key, rgb, batch["mask"][i, 0].numpy(),
                                out["selection"][i, 0].cpu().numpy(), rows[-1]))
    metrics = {}
    for domain in ("Mixed", "White", "Field"):
        selected = [r for r in rows if domain == "Mixed" or r["domain"] == domain]
        if not selected:
            metrics[domain] = {"count": 0}
            continue
        y, p = [r["target"] for r in selected], [r["prediction"] for r in selected]
        cm = confusion_matrix(y, p, labels=list(range(5)))
        metrics[domain] = dict(count=len(y), accuracy=accuracy_score(y, p),
                              macro_f1=f1_score(y, p, labels=list(range(5)), average="macro", zero_division=0),
                              mask_acceptance=float(np.mean([r["mask_valid"] for r in selected])),
                              confusion_matrix=cm.tolist(),
                              report=classification_report(y, p, labels=list(range(5)), target_names=CLASSES,
                                                           output_dict=True, zero_division=0))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted", ylabel="True", title=f"{domain} test set (n={len(y)})")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(output / f"confusion_{domain.lower()}.png", dpi=150)
        plt.close(fig)
    save_json(output / "metrics.json", metrics)
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw_samples(samples, output / "attention_masks.png")
    return metrics


def draw_samples(samples, path):
    fig, axes = plt.subplots(len(samples), 3, figsize=(11, 3 * len(samples)), squeeze=False)
    for axes_row, (key, rgb, mask, attention, row) in zip(axes, samples):
        axes_row[0].imshow(rgb)
        axes_row[0].set_title(f'{key[0]} / True: {row["target_name"]}\nPred: {row["prediction_name"]} ({row["confidence"]:.2f})')
        axes_row[1].imshow(rgb)
        axes_row[1].imshow(np.ma.masked_where(mask < 0.5, mask), cmap="Greens", alpha=0.5, vmin=0, vmax=1)
        axes_row[1].set_title(f'Mask q={row["quality"]:.2f}, accepted={row["mask_valid"]}')
        axes_row[2].imshow(rgb)
        heat = cv2.resize(attention, (rgb.shape[1], rgb.shape[0]))
        axes_row[2].imshow(heat, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        axes_row[2].set_title('Soft selection (not lesion ground truth)\nG/L/M=' +
                              '/'.join(f'{row[f"weight_{b}"]:.2f}' for b in ("global", "local", "mask")))
        for ax in axes_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def training_curve(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["train_loss"] for r in history], label="train objective")
    axes[0].plot([r["val_nll"] for r in history], label="validation NLL")
    axes[1].plot([r["val_accuracy"] for r in history], label="validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch (zero-based)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
