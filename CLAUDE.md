# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML pipeline for predicting diabetes incidence in gallstone (cholelithiasis) patients using NHIC health checkup data. Runs a 7-step pipeline (dummy data → Table 1 → preprocessing → GridSearchCV training → evaluation+SHAP → performance table → comparison figures) for two outcome targets (`outA`: any diabetes, `out2`: type 2 diabetes) across 6 models.

## Commands

```bash
# Full pipeline (both targets, all models)
python run_all.py

# Quick test (small param grid, fewer bootstraps)
python run_all.py --small-grid --n-bootstrap 50

# Specific target/models
python run_all.py --targets "outA" --models "xgboost lightgbm" --skip-dummy

# Shell wrapper alternative
bash scripts/run_all.sh --small-grid

# Individual steps (run from code/ or use absolute paths)
python code/make_dummy.py
python code/preprocessing.py --target outA --add-missing-indicator
python code/train_gridsearch.py --data-dir data/processed/outA --small-grid
```

No test suite exists. Validate by running the pipeline with `--small-grid --n-bootstrap 50`.

## Architecture

**Entry point**: `run_all.py` orchestrates all steps, iterating over targets and models.

**Pipeline flow** (per target):
1. `make_dummy.py` → generates synthetic CSV (10k samples)
2. `create_table1.py` → baseline characteristics via tableone → Excel
3. `preprocessing.py` → `DiabetesPreprocessor` class: impute missing values, scale numerics, add missing indicators for features with >5% missingness, stratified split (70/10/20)
4. `train_gridsearch.py` → `ModelTrainer` class: GridSearchCV (5-fold, AUROC scoring) for 6 models → saves `.pkl` + `_meta.json`
5. `evaluate.py` → metrics (AUROC, AUPRC, Sens, Spec, etc.) + SHAP analysis (TreeExplainer for tree models, KernelExplainer for ANN) → per-model figures + `metrics.json`
6. `create_performance_table.py` → bootstrap 95% CI, Youden Index threshold → `model_performance.xlsx`
7. `create_comparison_figures.py` → multi-model overlay plots (ROC, PR, calibration, SHAP) → PNG/TIFF/PDF

**Output structure**: `data/processed/{target}/`, `models/{target}/`, `results/{target}/{model}/`, `results/{target}/tables/`, `results/{target}/comparison/`

## Key Constants & Configuration

- **RANDOM_STATE**: `1004` everywhere
- **Feature lists**: `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` in `preprocessing.py`; `FEATURE_RENAME` dict maps internal names to display names
- **Param grids**: `PARAM_GRIDS` (full) and `PARAM_GRIDS_SMALL` (fast test) in `train_gridsearch.py`
- **Missing indicator threshold**: 5% (adds `{var}_missing` binary columns, excluded from SHAP plots)

## Critical Compatibility Constraints

This runs on Python 3.8 in an air-gapped environment with pinned old packages:

- **xgboost 0.80**: Use `seed` not `random_state`, `silent=True` not `verbosity`. No `use_label_encoder`.
- **shap 0.32**: Requires numpy compat patches (`np.int = int`, `np.float = float`). TreeExplainer falls back to KernelExplainer. ANN limited to 500 samples.
- **tableone 0.5.13**: Uses `inspect` module to detect available parameters (`overall`, `htest_name`) before calling, since API changed across versions.
- **All models serialize as `.pkl`** — no `.json`/`.cbm` format.

## Environment

```bash
conda activate nhiss  # Python 3.8.20
pip install -r requirements_project.txt  # Core dependencies only
```

`requirements.txt` contains the full environment snapshot. `requirements_project.txt` has only the packages needed for this project.
