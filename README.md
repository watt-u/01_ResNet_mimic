# STEMI / NSTEMI / Control ECG Classification

Supervised 1-D ResNet classifier for 12-lead ECGs using MIMIC-IV-ECG and the
MIMIC-IV-ECG-Ext-ICD diagnostic labels.

## Cohort

- **Waveforms:** MIMIC-IV-ECG downsampled to 100 Hz, stored as per-prefix zarr.
- **Labels:** `ed_diag_hosp` (primary) or `all_diag_all` (sensitivity).
- **Filters:** first ECG of each ED stay (`ecg_taken_in_ed == True`,
  `ecg_no_within_stay == 0`).
- **Classes:**
  - `stemi`: I21.0*, I21.1*, I21.2*, I21.3
  - `nstemi`: I21.4
  - `control`: no I21.* or I22.* codes

Records with other acute MI codes (`I21.9`, `I21.A*`, `I21.B`, `I22.*`) are
excluded entirely.

## Quick start

### 1. Install dependencies

Create a project virtual environment (recommended):

```bash
cd experiments/01_ResNet_mimic
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Build the cohort CSV

```bash
python src/build_cohort.py
```

This writes `outputs/cohort.csv` and builds the zarr index so the dataloader
can locate each record quickly.

### 3. Train locally

```bash
python src/train.py --run-name baseline
```

To use the sensitivity label source:

```bash
python src/train.py --label-source all_diag_all --run-name all_diag_all_baseline
```

Checkpoints, a per-epoch CSV metrics file, TensorBoard logs, and config are
saved under `outputs/<run_name>/`. `--run-name` is required. If the directory
already exists, use `--resume` to continue training or `--force` to overwrite.

### 4. Evaluate

```bash
python src/evaluate.py --run-dir outputs/<run_name> --split test
```

This produces `metrics_test.json`, `predictions_test.csv`, and plots in
`outputs/<run_name>/figures/`.

## Train on Modal

The zarr data is expected to be in the Modal volume `mimic-iv-ecg-100hz-v2`.

```bash
modal run src/modal_train.py --run-name baseline
modal run src/modal_train.py --label-source all_diag_all --run-name all_diag_all_run1
```

Outputs are written to the Modal volume `ecg-stemi-outputs`. Download with:

```bash
modal volume get ecg-stemi-outputs <run_name> local_outputs/
```

## Config

Edit `config.yaml` to change:

- Data paths
- Cohort filters and class definitions
- `preload`: load all cohort records into RAM at startup (enabled automatically
  on Modal; useful when zarr access is the bottleneck)
- Model architecture
- Training hyperparameters (LR, batch size, class weights, etc.)
- Augmentation

Command-line overrides are available for common options in the Modal entrypoint;
local training reads `config.yaml` directly.

## Repo layout

```text
experiments/01_ResNet_mimic/
├── config.yaml           # experiment configuration
├── requirements.txt      # Python dependencies
├── README.md             # this file
├── src/
│   ├── build_cohort.py   # build labeled cohort CSV
│   ├── dataset.py        # PyTorch Dataset + augmentations
│   ├── model.py          # ResNet-1D classifier
│   ├── train.py          # supervised training loop
│   ├── evaluate.py       # test-set evaluation & plots
│   ├── modal_train.py    # Modal cloud entrypoint
│   └── utils.py          # shared helpers
└── outputs/
    ├── cohort.csv
    └── <run_name>/
        ├── config.yaml
        ├── best_checkpoint.pt
        ├── latest_checkpoint.pt
        ├── metrics.csv     # per-epoch train/val metrics
        ├── logs/           # TensorBoard event files
        └── figures/        # ROC/PR/confusion plots
```

## Notes

- The dataset is heavily imbalanced (~153 k control, ~3.6 k NSTEMI, ~1 k STEMI).
  The default loss uses inverse-square-root class weighting.
- The provided `fold` column is patient-disjoint; we use folds 0–17 for train,
  18 for validation, and 19 for test.
