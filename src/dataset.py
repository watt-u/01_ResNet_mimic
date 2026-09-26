#!/usr/bin/env python3
"""PyTorch Dataset for MIMIC-IV-ECG zarr stores + cohort CSV."""

from __future__ import annotations

from pathlib import Path

import time

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm


class Augmentation:
    """On-the-fly augmentation for ECG signals (lead-first tensor)."""

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.cfg = cfg.get("augmentation", {})
        self.rng = rng or np.random.default_rng()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (n_leads, seq_len)
        if self.cfg.get("scale_range"):
            lo, hi = self.cfg["scale_range"]
            scale = self.rng.uniform(lo, hi)
            x = x * scale

        if self.cfg.get("shift_max_samples"):
            max_shift = int(self.cfg["shift_max_samples"])
            shift = self.rng.integers(-max_shift, max_shift + 1)
            if shift > 0:
                x = torch.nn.functional.pad(x, (shift, 0), mode="constant", value=0.0)[
                    :, : x.shape[1]
                ]
            elif shift < 0:
                x = torch.nn.functional.pad(x, (0, -shift), mode="constant", value=0.0)[
                    :, -shift :
                ]

        if self.cfg.get("noise_std"):
            std = float(self.cfg["noise_std"])
            x = x + torch.randn_like(x) * std

        if self.cfg.get("lead_dropout_prob"):
            p = float(self.cfg["lead_dropout_prob"])
            mask = torch.rand(x.shape[0]) > p
            if mask.any():
                x = x * mask.unsqueeze(1).float()

        if self.cfg.get("time_mask_prob"):
            p = float(self.cfg["time_mask_prob"])
            max_len = int(self.cfg.get("time_mask_max_len", 50))
            if self.rng.random() < p:
                length = self.rng.integers(1, max_len + 1)
                start = self.rng.integers(0, x.shape[1] - length + 1)
                x[:, start : start + length] = 0.0

        return x


class ECGDataset(Dataset):
    """Load ECG waveforms from per-prefix zarr stores.

    The cohort DataFrame must contain at least:
        - prefix: zarr store prefix, e.g. 'p1000'
        - zarr_idx: integer index of the record inside the zarr store
        - label_idx: integer class label
        - fold: fold number for split selection
    """

    def __init__(
        self,
        cohort: pd.DataFrame,
        zarr_root: str | Path,
        cfg: dict | None = None,
        split: str = "train",
        transform: Augmentation | None = None,
        preload: bool = False,
    ):
        self.cfg = cfg or {}
        self.zarr_root = Path(zarr_root)
        self.split = split
        self.transform = transform
        self.preload = preload

        self.cohort = cohort.reset_index(drop=True)
        self._zarr_cache: dict[str, zarr.Group] = {}

        self.clip_mV = self.cfg.get("clip_mV", 3.0)
        self.normalize = self.cfg.get("normalize", "zscore_per_record")

        self._preload_arrays: dict[str, np.ndarray] = {}
        self._preload_idx_map: dict[str, dict[int, int]] = {}

        if self.preload:
            self._preload_data()

    def __len__(self) -> int:
        return len(self.cohort)

    def _open_zarr(self, prefix: str) -> zarr.Group:
        if prefix not in self._zarr_cache:
            zarr_path = self.zarr_root / f"{prefix}.zarr"
            self._zarr_cache[prefix] = zarr.open_group(str(zarr_path), mode="r")
        return self._zarr_cache[prefix]

    def _preload_data(self) -> None:
        """Load all cohort records into RAM, reading each prefix sequentially."""
        print(
            f"Preloading {self.split} split into RAM ({len(self.cohort):,} records) ..."
        )
        t0 = time.monotonic()
        for prefix, group in tqdm(
            self.cohort.groupby("prefix"),
            total=self.cohort["prefix"].nunique(),
            desc=f"preload {self.split}",
            leave=False,
        ):
            zarr_idx_list = group["zarr_idx"].astype(int).tolist()
            z = self._open_zarr(prefix)
            full = np.asarray(z["signals"][:])  # (N, seq_len, n_leads)
            subset = full[zarr_idx_list]  # (k, seq_len, n_leads)
            idx_map = {orig: new for new, orig in enumerate(zarr_idx_list)}
            self._preload_arrays[prefix] = subset
            self._preload_idx_map[prefix] = idx_map
        self._zarr_cache.clear()
        elapsed = time.monotonic() - t0
        total_bytes = sum(a.nbytes for a in self._preload_arrays.values())
        print(
            f"Preloaded {self.split}: {len(self.cohort):,} records, "
            f"{total_bytes / 1e9:.2f} GB, in {elapsed:.1f}s"
        )

    def _normalize_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a (n_leads, seq_len) tensor."""
        if self.normalize == "none":
            return x
        if self.normalize == "zscore_per_record":
            # Per-lead z-score
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            return (x - mean) / std
        if self.normalize == "zscore_global":
            # Placeholder: global stats would be loaded from a file
            # For now fall back to per-record
            mean = x.mean()
            std = x.std()
            std = std if std > 0 else 1.0
            return (x - mean) / std
        raise ValueError(f"Unknown normalize: {self.normalize}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.cohort.iloc[idx]
        prefix = str(row["prefix"])
        zarr_idx = int(row["zarr_idx"])
        label = int(row["label_idx"])

        if self._preload_arrays:
            local_idx = self._preload_idx_map[prefix][zarr_idx]
            signal = self._preload_arrays[prefix][local_idx]
        else:
            z = self._open_zarr(prefix)
            signal = z["signals"][zarr_idx]  # (seq_len, n_leads)
        signal = torch.from_numpy(np.asarray(signal)).float()
        signal = signal.permute(1, 0)  # (n_leads, seq_len)

        # Clip extreme amplitudes
        if self.clip_mV is not None:
            signal = torch.clamp(signal, -self.clip_mV, self.clip_mV)

        signal = self._normalize_signal(signal)

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, torch.tensor(label, dtype=torch.long)


def make_split(
    cohort_csv: str | Path,
    folds: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test DataFrames based on fold column."""
    df = pd.read_csv(cohort_csv, dtype={"subject_id": str})
    train_df = df[df["fold"].isin(folds["train"])].copy()
    val_df = df[df["fold"].isin(folds["val"])].copy()
    test_df = df[df["fold"].isin(folds["test"])].copy()
    return train_df, val_df, test_df


def compute_class_weights(
    labels: pd.Series,
    mode: str = "inverse_sqrt",
    device: str = "cpu",
) -> torch.Tensor:
    """Compute class weights for imbalanced classification.

    mode:
        none:         uniform weights
        inverse:      1 / count
        inverse_sqrt: 1 / sqrt(count)
    """
    if mode == "none" or mode is None:
        return torch.ones(len(labels.unique()), dtype=torch.float32, device=device)

    counts = labels.value_counts().sort_index()
    if mode == "inverse":
        weights = 1.0 / counts.values
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts.values)
    else:
        raise ValueError(f"Unknown class_weight mode: {mode}")

    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)
