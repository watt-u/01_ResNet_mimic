#!/usr/bin/env python3
"""Build the STEMI / NSTEMI / Control cohort CSV from Ext-ICD labels.

Usage:
    python build_cohort.py
    python build_cohort.py --label-source all_diag_all
    python build_cohort.py --config config.yaml --output outputs/cohort.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def add_zarr_index(df: pd.DataFrame, zarr_root: Path) -> pd.DataFrame:
    """Map each record's rel_path to its index inside the per-prefix zarr store.

    Adds a 'zarr_idx' column. Records whose path cannot be found in the zarr
    store are dropped (e.g. corrupted or skipped during zarr conversion).
    """
    try:
        import zarr
    except ImportError as e:
        raise ImportError("zarr is required to build the zarr index") from e

    print(f"Building zarr index from {zarr_root} ...")
    prefix_to_idx: dict[str, dict[str, int]] = {}
    prefixes = sorted(df["prefix"].unique())
    for prefix in prefixes:
        zarr_path = zarr_root / f"{prefix}.zarr"
        if not zarr_path.exists():
            print(f"  WARNING: {zarr_path} not found, skipping prefix {prefix}")
            continue
        try:
            root = zarr.open_group(str(zarr_path), mode="r")
            paths = np.asarray(root["path"])
            prefix_to_idx[prefix] = {str(p): i for i, p in enumerate(paths)}
        except Exception as e:
            print(f"  WARNING: could not read {zarr_path}: {e}")

    def lookup(row):
        idx_map = prefix_to_idx.get(row["prefix"])
        if idx_map is None:
            return -1
        return idx_map.get(row["rel_path"], -1)

    df["zarr_idx"] = df.apply(lookup, axis=1)
    missing = df["zarr_idx"] == -1
    n_missing = int(missing.sum())
    if n_missing:
        print(f"  {n_missing:,} records not found in zarr stores (dropped)")
        df = df[~missing].copy()
    print(f"  indexed {len(df):,} records")
    return df


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compile_patterns(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns))


def extract_rel_path(file_name: str) -> str:
    """Strip the leading dataset name from file_name to match zarr path entries."""
    # file_name looks like:
    # mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1000/...
    parts = Path(file_name).parts
    if parts[0].endswith(".0") or "mimic-iv-ecg" in parts[0].lower():
        parts = parts[1:]
    if parts[0] == "files":
        parts = parts[1:]
    return "/".join(parts)


def assign_label(code_str: str, cfg: dict) -> tuple[str | None, str | None]:
    """Return (label, reason) for one record.

    label is one of 'stemi', 'nstemi', 'control', or None (excluded).
    reason explains the exclusion when label is None.
    """
    code_str = "" if pd.isna(code_str) else str(code_str)

    stemi_pat = compile_patterns(cfg["classes"]["stemi"]["patterns"])
    nstemi_pat = compile_patterns(cfg["classes"]["nstemi"]["patterns"])
    acute_mi_pat = compile_patterns(cfg["acute_mi_exclusion"])

    if stemi_pat.search(code_str):
        return "stemi", None
    if nstemi_pat.search(code_str):
        return "nstemi", None
    if acute_mi_pat.search(code_str):
        return None, "other_acute_mi"
    return "control", None


def build_cohort(
    labels_csv: Path,
    cfg: dict,
    label_source: str | None = None,
    zarr_root: Path | None = None,
) -> pd.DataFrame:
    label_source = label_source or cfg["label_source"]
    filters = cfg["filters"]

    print(f"Loading labels from {labels_csv} ...")
    usecols = [
        "file_name",
        "study_id",
        "subject_id",
        "ecg_time",
        label_source,
        "ecg_taken_in_ed",
        "ecg_no_within_stay",
        "fold",
        "age",
        "gender",
    ]
    df = pd.read_csv(labels_csv, usecols=usecols, dtype={"subject_id": str})
    print(f"  raw rows: {len(df):,}")

    # Apply filters
    if filters.get("ecg_taken_in_ed") is not None:
        df = df[df["ecg_taken_in_ed"] == filters["ecg_taken_in_ed"]]
    if filters.get("ecg_no_within_stay") is not None:
        df = df[df["ecg_no_within_stay"] == filters["ecg_no_within_stay"]]
    print(f"  after ED first-ECG filter: {len(df):,}")

    # Assign labels
    labels_reasons = df[label_source].apply(lambda s: assign_label(s, cfg))
    df["label"] = [lr[0] for lr in labels_reasons]
    df["exclude_reason"] = [lr[1] for lr in labels_reasons]

    excluded = df["label"].isna()
    n_excluded = int(excluded.sum())
    if n_excluded:
        print(f"  excluded (other acute MI): {n_excluded:,}")
        print(
            df.loc[excluded, "exclude_reason"].value_counts().to_string()
        )
    df = df[~excluded].copy()

    # Build relative path and prefix for zarr lookup
    df["rel_path"] = df["file_name"].apply(extract_rel_path)
    df["prefix"] = df["rel_path"].apply(lambda p: p.split("/")[0])

    # Build zarr index if the zarr root is available
    zarr_root_path = zarr_root or Path(cfg["zarr_root"])
    if not zarr_root_path.is_absolute():
        zarr_root_path = PROJECT_ROOT / zarr_root_path
    if zarr_root_path.exists():
        df = add_zarr_index(df, zarr_root_path)
    else:
        print(f"  WARNING: zarr root {zarr_root_path} not found; skipping zarr index")
        df["zarr_idx"] = -1

    # Keep only columns needed for training/evaluation
    keep_cols = [
        "study_id",
        "subject_id",
        "ecg_time",
        "rel_path",
        "prefix",
        "zarr_idx",
        "label",
        "fold",
        "age",
        "gender",
        label_source,
    ]
    df = df[keep_cols].reset_index(drop=True)

    # Label to integer
    label_map = {"control": 0, "nstemi": 1, "stemi": 2}
    df["label_idx"] = df["label"].map(label_map)

    print("\nCohort summary:")
    print(df["label"].value_counts().to_string())
    print(f"\nTotal usable records: {len(df):,}")
    print(f"Unique patients: {df['subject_id'].nunique():,}")

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        type=Path,
        help="path to config.yaml",
    )
    p.add_argument(
        "--label-source",
        default=None,
        help="override label source (e.g., ed_diag_hosp, all_diag_all)",
    )
    p.add_argument(
        "--output",
        default=None,
        type=Path,
        help="output cohort CSV path (default: from config)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    labels_csv = PROJECT_ROOT / cfg["labels_csv"]
    output_csv = (
        PROJECT_ROOT / cfg["cohort_csv"]
        if args.output is None
        else args.output
    )

    df = build_cohort(
        labels_csv=labels_csv,
        cfg=cfg,
        label_source=args.label_source,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved cohort to {output_csv}")


if __name__ == "__main__":
    main()
