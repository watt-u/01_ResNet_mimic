#!/usr/bin/env python3
"""Evaluate a trained STEMI / NSTEMI / Control classifier on the test fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ECGDataset, make_split
from model import build_model
from utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path, help="training output dir")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--checkpoint", default="best_checkpoint.pt", help="checkpoint name")
    return p.parse_args()


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_logits = []
    all_paths = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_labels.append(y.numpy())
            all_logits.append(logits.cpu().numpy())

    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    return labels, logits, np.array(all_paths)


def compute_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    if class_names is None:
        class_names = ["control", "nstemi", "stemi"]

    preds = logits.argmax(axis=1)
    num_classes = logits.shape[1]

    metrics = {
        "n": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ),
    }

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        y_true = (labels == i).astype(int)
        y_score = logits[:, i]
        y_pred = (preds == i).astype(int)

        if len(np.unique(y_true)) < 2:
            auroc = np.nan
            auprc = np.nan
        else:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)

        per_class[name] = {
            "support": int(y_true.sum()),
            "precision": float(
                precision_score_single(y_true, y_pred)
            ),
            "recall": float(
                recall_score_single(y_true, y_pred)
            ),
            "f1": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "auroc": float(auroc),
            "auprc": float(auprc),
        }

    metrics["per_class"] = per_class
    metrics["macro_auroc"] = float(
        np.nanmean([v["auroc"] for v in per_class.values()])
    )
    metrics["macro_auprc"] = float(
        np.nanmean([v["auprc"] for v in per_class.values()])
    )

    return metrics


def precision_score_single(y_true, y_pred):
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score_single(y_true, y_pred):
    tp = (y_true * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, name in enumerate(class_names):
        y_true = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true, logits[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, name in enumerate(class_names):
        y_true = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, logits[:, i])
        pr_auc = average_precision_score(y_true, logits[:, i])
        ax.plot(recall, precision, label=f"{name} (AUPRC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    device = get_device()

    cohort_csv = Path(cfg["cohort_csv"])
    _, val_df, test_df = make_split(cohort_csv, cfg["folds"])
    eval_df = val_df if args.split == "val" else test_df

    eval_ds = ECGDataset(
        eval_df,
        zarr_root=cfg["zarr_root"],
        cfg=cfg,
        split=args.split,
        transform=None,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg.get("pin_memory", True),
    )

    model = build_model(cfg).to(device)
    ckpt_path = run_dir / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    labels, logits, _ = collect_predictions(model, loader, device)
    class_names = ["control", "nstemi", "stemi"]
    metrics = compute_metrics(labels, logits, class_names)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    preds = logits.argmax(axis=1)
    plot_confusion_matrix(labels, preds, class_names, fig_dir / f"cm_{args.split}.png")
    plot_roc_curves(labels, logits, class_names, fig_dir / f"roc_{args.split}.png")
    plot_pr_curves(labels, logits, class_names, fig_dir / f"pr_{args.split}.png")

    # Save detailed predictions
    eval_df = eval_df.copy()
    eval_df["pred"] = preds
    for i, name in enumerate(class_names):
        eval_df[f"score_{name}"] = logits[:, i]
    eval_df.to_csv(run_dir / f"predictions_{args.split}.csv", index=False)

    # Save metrics
    with open(run_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== {args.split.upper()} METRICS ===")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"Macro AUROC:  {metrics['macro_auroc']:.4f}")
    print(f"Macro AUPRC:  {metrics['macro_auprc']:.4f}")
    print("\nPer-class:")
    for name, vals in metrics["per_class"].items():
        print(
            f"  {name:10s} n={vals['support']:5d}  "
            f"P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f} "
            f"AUROC={vals['auroc']:.3f} AUPRC={vals['auprc']:.3f}"
        )


if __name__ == "__main__":
    main()
