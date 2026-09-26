#!/usr/bin/env python3
"""Shared utilities for the STEMI/NSTEMI/Control experiment."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

def _find_project_root() -> Path:
    """Find the repository root locally, with a safe fallback for containers."""
    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        if (parent / "experiments" / "01_ResNet_mimic").is_dir():
            return parent

    # Modal mounts this experiment as /experiment/src, without the repository's
    # surrounding directory hierarchy.
    return source_path.parent.parent


PROJECT_ROOT = _find_project_root()


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against project root
    for key in ["zarr_root", "labels_csv", "cohort_csv", "output_dir"]:
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(PROJECT_ROOT / cfg[key])
    return cfg


def save_config(cfg: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior can hurt performance; leave these optional
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def class_distribution(labels: np.ndarray | pd.Series) -> dict:
    if isinstance(labels, np.ndarray):
        labels = pd.Series(labels)
    return labels.value_counts().sort_index().to_dict()
