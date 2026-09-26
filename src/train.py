#!/usr/bin/env python3
"""Supervised training script for STEMI / NSTEMI / Control classification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Augmentation, ECGDataset, compute_class_weights, make_split
from model import build_model
from utils import CSVLogger, count_parameters, get_device, load_config, save_config, set_seed


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    p.add_argument("--cohort", default=None, type=Path, help="override cohort CSV")
    p.add_argument("--label-source", default=None, help="for logging only")
    p.add_argument("--run-name", required=True, help="run name for output subdir")
    p.add_argument("--resume", default=None, type=Path, help="checkpoint to resume")
    p.add_argument(
        "--force",
        action="store_true",
        help="allow writing into an existing output directory",
    )
    return p.parse_args()


def compute_epoch_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    loss: float,
    num_classes: int = 3,
) -> dict:
    preds = logits.argmax(axis=1)
    acc = (preds == labels).mean()

    # Per-class AUROC using one-vs-rest
    aurocs = {}
    for c in range(num_classes):
        y_true = (labels == c).astype(int)
        y_score = logits[:, c]
        if len(np.unique(y_true)) < 2:
            aurocs[f"class_{c}_auroc"] = np.nan
        else:
            aurocs[f"class_{c}_auroc"] = roc_auc_score(y_true, y_score)

    macro_auroc = np.nanmean(list(aurocs.values()))

    return {
        "loss": loss,
        "accuracy": float(acc),
        "macro_auroc": float(macro_auroc),
        **{k: float(v) for k, v in aurocs.items()},
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    optimizer=None,
    desc: str = "epoch",
) -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_labels = []
    all_logits = []
    total_loss = 0.0
    n_samples = 0

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            all_labels.append(y.cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    avg_loss = total_loss / n_samples
    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    return compute_epoch_metrics(labels, logits, avg_loss, num_classes=3)


def build_dataloaders(cfg: dict, cohort_csv: Path):
    train_df, val_df, test_df = make_split(cohort_csv, cfg["folds"])

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print("Train class distribution:")
    print(train_df["label"].value_counts().to_string())

    zarr_root = cfg["zarr_root"]
    preload = cfg.get("preload", False)
    train_aug = Augmentation(cfg)
    train_ds = ECGDataset(
        train_df,
        zarr_root=zarr_root,
        cfg=cfg,
        split="train",
        transform=train_aug,
        preload=preload,
    )
    val_ds = ECGDataset(
        val_df,
        zarr_root=zarr_root,
        cfg=cfg,
        split="val",
        transform=None,
        preload=preload,
    )
    test_ds = ECGDataset(
        test_df,
        zarr_root=zarr_root,
        cfg=cfg,
        split="test",
        transform=None,
        preload=preload,
    )

    # Preloading puts the whole split in RAM; workers would duplicate that cache,
    # so run the data loader in the main process when preloading is enabled.
    loader_kwargs: dict = {
        "batch_size": cfg["batch_size"],
        "pin_memory": cfg.get("pin_memory", True),
    }
    if preload:
        loader_kwargs["num_workers"] = 0
    else:
        num_workers = cfg["num_workers"]
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = num_workers > 0
        prefetch_factor = cfg.get("prefetch_factor")
        if prefetch_factor is not None and num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def build_optimizer(model: nn.Module, cfg: dict):
    training_cfg = cfg["training"]
    lr = training_cfg["lr"]
    wd = training_cfg["weight_decay"]
    opt_name = training_cfg["optimizer"].lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    scheduler_name = training_cfg.get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_cfg["epochs"]
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3
        )
    elif scheduler_name in ("none", ""):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return optimizer, scheduler


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    cohort_csv = Path(args.cohort) if args.cohort else Path(cfg["cohort_csv"])
    if not cohort_csv.exists():
        raise FileNotFoundError(
            f"Cohort CSV not found: {cohort_csv}. Run build_cohort.py first."
        )

    # Output directory
    run_name = args.run_name
    output_dir = Path(cfg["output_dir"]) / run_name
    if output_dir.exists() and not args.resume and not args.force:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Use --resume to continue training or --force to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    # Save metadata
    meta = {
        "run_name": run_name,
        "label_source": args.label_source or cfg["label_source"],
        "cohort_csv": str(cohort_csv),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger = CSVLogger(output_dir / "metrics.csv")

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg, cohort_csv)

    # Warm-up / timing sanity check for the data pipeline
    t0 = time.monotonic()
    x_batch, y_batch = next(iter(train_loader))
    print(
        f"First training batch loaded in {time.monotonic() - t0:.2f}s "
        f"(shape {tuple(x_batch.shape)})"
    )

    model = build_model(cfg).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Class weights
    train_labels = pd.read_csv(cohort_csv, usecols=["fold", "label_idx"])
    train_labels = train_labels[train_labels["fold"].isin(cfg["folds"]["train"])][
        "label_idx"
    ]
    class_weights = compute_class_weights(
        train_labels,
        mode=cfg["training"].get("class_weight", "inverse_sqrt"),
        device=str(device),
    )
    print(f"Class weights: {class_weights.cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
    )

    optimizer, scheduler = build_optimizer(model, cfg)
    start_epoch = 0
    best_metric = -np.inf
    patience_counter = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_metric = checkpoint.get("best_metric", -np.inf)

    patience = cfg["training"]["early_stopping_patience"]
    metric_name = cfg["training"]["early_stopping_metric"]
    mode = cfg["training"]["early_stopping_mode"]

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")

        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, desc="train"
        )
        print(f"  train loss={train_metrics['loss']:.4f} "
              f"acc={train_metrics['accuracy']:.4f} "
              f"macro_auroc={train_metrics['macro_auroc']:.4f}")

        val_metrics = run_epoch(
            model, val_loader, criterion, device, optimizer=None, desc="val"
        )
        print(f"  val   loss={val_metrics['loss']:.4f} "
              f"acc={val_metrics['accuracy']:.4f} "
              f"macro_auroc={val_metrics['macro_auroc']:.4f}")

        logger.log_metrics(epoch, train_metrics, split="train")
        logger.log_metrics(epoch, val_metrics, split="val")

        current_metric = val_metrics[metric_name]
        improved = current_metric > best_metric if mode == "max" else current_metric < best_metric

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_metric)
            else:
                scheduler.step()

        if improved:
            best_metric = current_metric
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_metric": best_metric,
                "cfg": cfg,
            }
            ckpt_path = output_dir / "best_checkpoint.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  saved best checkpoint (metric={metric_name}={current_metric:.4f})")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{patience})")

        # Always save latest
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_metric": best_metric,
            },
            output_dir / "latest_checkpoint.pt",
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    logger.close()
    print(f"\nTraining complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
