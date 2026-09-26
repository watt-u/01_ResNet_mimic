#!/usr/bin/env python3
"""Modal training entrypoint for the STEMI/NSTEMI/Control experiment.

Expects:
  - Modal volume `mimic-iv-ecg-100hz-v2` containing per-prefix zarr stores.
  - Cohort CSV built locally by `src/build_cohort.py`.

The cohort CSV, source code, and config are mounted into the container at
/experiment. Outputs are written to /outputs, which is backed by a Modal volume
(default: `ecg-stemi-outputs`) so checkpoints and logs persist after the run.

Usage:
    modal run experiments/01_ResNet_mimic/src/modal_train.py
    modal run ... --label-source all_diag_all --run-name all_diag_all_run1
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ------------------------------------------------------------------------------
# Paths from the host (where `modal run` is invoked)
# ------------------------------------------------------------------------------
HOST_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
HOST_SRC_DIR = HOST_EXPERIMENT_DIR / "src"
HOST_CONFIG = HOST_EXPERIMENT_DIR / "config.yaml"
HOST_COHORT = HOST_EXPERIMENT_DIR / "outputs" / "cohort.csv"

# ------------------------------------------------------------------------------
# Modal image
# ------------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "numpy",
        "scipy",
        "pandas",
        "zarr",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "tensorboard",
    )
    .add_local_dir(HOST_SRC_DIR, remote_path="/experiment/src")
    .add_local_file(HOST_CONFIG, remote_path="/experiment/config.yaml")
)

app = modal.App("01-resnet-mimic-train")

zarr_volume = modal.Volume.from_name("mimic-iv-ecg-100hz-v2")
output_volume = modal.Volume.from_name("ecg-stemi-outputs", create_if_missing=True)

# The cohort CSV is passed as bytes at runtime so it does not need to exist while
# the Modal app is being imported.


# ------------------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={
        "/data/zarr_100hz": zarr_volume,
        "/outputs": output_volume,
    },
    gpu="a10",  # or "h100", "a100" if available
    timeout=86400,
    memory=32768,
)
def train(
    cohort_bytes: bytes,
    run_name: str,
    label_source: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> str:
    import subprocess

    # Write cohort CSV into the container
    cohort_path = Path("/experiment/outputs/cohort.csv")
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_path.write_bytes(cohort_bytes)

    # Make source importable
    sys.path.insert(0, "/experiment")

    # Override config for Modal paths
    import yaml

    config_path = Path("/experiment/config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["zarr_root"] = "/data/zarr_100hz"
    cfg["cohort_csv"] = str(cohort_path)
    cfg["output_dir"] = "/outputs"
    cfg["preload"] = True  # keep cohort in RAM; avoids slow random reads from the volume

    if epochs is not None:
        cfg["training"]["epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if lr is not None:
        cfg["training"]["lr"] = lr

    modal_config_path = Path("/experiment/config_modal.yaml")
    with open(modal_config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # Run training via subprocess so sys.path and cwd are clean
    cmd = [
        "python",
        "/experiment/src/train.py",
        "--config", str(modal_config_path),
        "--cohort", str(cohort_path),
    ]
    cmd += ["--run-name", run_name]
    if label_source:
        cmd += ["--label-source", label_source]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    output_volume.commit()
    return f"/outputs/{run_name}"


# ------------------------------------------------------------------------------
# Local entrypoint
# ------------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    run_name: str,
    label_source: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> None:
    if not HOST_COHORT.exists():
        print(
            f"ERROR: cohort CSV not found at {HOST_COHORT}.\n"
            "Build it locally first with:\n"
            "  python experiments/01_ResNet_mimic/src/build_cohort.py"
        )
        sys.exit(1)

    cohort_bytes = HOST_COHORT.read_bytes()
    print(f"Uploading cohort ({len(cohort_bytes) / 1e6:.1f} MB) ...")

    result_path = train.remote(
        cohort_bytes=cohort_bytes,
        label_source=label_source,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    print(f"Training complete. Outputs at {result_path}")
    print(
        "Download with:\n"
        f"  modal volume get ecg-stemi-outputs {Path(result_path).name} "
        "local_outputs/"
    )
