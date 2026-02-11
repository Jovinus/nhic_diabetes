# Diabetes Prediction Pipeline

Machine learning pipeline for predicting diabetes incidence in gallstone patients.

## Environment

- **Python**: 3.8.16 (air-gapped network)
- **Key packages**: numpy 1.24.4, pandas 1.4.4, scikit-learn 1.2.2, xgboost 0.80, lightgbm 3.3.5, shap 0.32.0, matplotlib 3.6.3, tableone 0.7.12, statsmodels 0.14.0

See `requirements.txt` for full list with pinned versions.

## Project Structure

```
nhic_diabetes/
├── code/
│   ├── make_dummy.py                  # Generate dummy data
│   ├── preprocessing.py               # Data preprocessing (with Missing Indicator)
│   ├── train_gridsearch.py            # GridSearchCV model training
│   ├── evaluate.py                    # Model evaluation & SHAP analysis
│   ├── create_table1.py               # Baseline characteristics table
│   ├── create_performance_table.py    # Performance comparison (Bootstrap CI)
│   └── create_comparison_figures.py   # Model comparison figures
├── scripts/
│   ├── run_all.sh                     # Run entire pipeline (simple)
│   ├── run_pipeline.sh                # Run pipeline (advanced options)
│   ├── download_packages.sh           # Download packages for offline
│   ├── install_offline.sh             # Install packages (creates new venv)
│   └── install_to_existing_env.sh     # Install to existing environment
├── notebooks/
│   └── run_pipeline.ipynb             # Jupyter notebook pipeline
├── run_all.py                         # Python entry point for pipeline
├── data/
│   ├── dummy_diabetes_data.csv
│   └── processed/{target}/            # Preprocessed numpy arrays
├── models/{target}/                   # Trained models (.pkl)
├── results/{target}/
│   ├── {model_name}/                  # Per-model evaluation + SHAP
│   ├── comparison/                    # Model comparison figures
│   └── tables/                        # Table 1, Performance table
├── packages/                          # Offline installation packages
└── requirements.txt
```

## Quick Start

### Option A: Shell Script

```bash
# Full pipeline (both targets, all models)
bash scripts/run_all.sh

# Quick test
bash scripts/run_all.sh --small-grid --n-bootstrap 100

# Specific target / models
bash scripts/run_all.sh --targets "outA" --models "decision_tree random_forest lightgbm"

# Skip dummy data (use real data)
bash scripts/run_all.sh --skip-dummy
```

### Option B: Python Script

```bash
python run_all.py --small-grid --n-bootstrap 100
python run_all.py --targets "outA out2" --models "decision_tree random_forest xgboost lightgbm ann"
```

### Option C: Jupyter Notebook

```bash
cd notebooks
jupyter notebook run_pipeline.ipynb
```

## Pipeline Steps

For each target (`outA`, `out2`):

| Step | Script | Output |
|------|--------|--------|
| 1. Dummy Data | `make_dummy.py` | `data/dummy_diabetes_data.csv` |
| 2. Table 1 | `create_table1.py` | `results/{target}/tables/table1_*.xlsx` |
| 3. Preprocessing | `preprocessing.py` | `data/processed/{target}/` |
| 4. Training | `train_gridsearch.py` | `models/{target}/*_best_model.pkl` |
| 5. Evaluation + SHAP | `evaluate.py` | `results/{target}/{model}/` |
| 6. Performance Table | `create_performance_table.py` | `results/{target}/tables/model_performance.xlsx` |
| 7. Comparison Figures | `create_comparison_figures.py` | `results/{target}/comparison/` |

## Models

| Model | Key | Notes |
|-------|-----|-------|
| Decision Tree | `decision_tree` | sklearn DecisionTreeClassifier |
| Random Forest | `random_forest` | sklearn RandomForestClassifier |
| XGBoost | `xgboost` | xgboost 0.80 (`seed`, `silent` API) |
| LightGBM | `lightgbm` | lightgbm 3.3.5 |
| ANN (MLP) | `ann` | sklearn MLPClassifier |

All models are saved as `.pkl` (pickle) for cross-version compatibility.

## Output Files

### Per-Model Results

```
results/{target}/{model_name}/
├── roc_curve.png
├── pr_curve.png
├── calibration_curve.png
├── confusion_matrix.png
├── feature_importance.png
├── shap_summary.png          # Beeswarm plot (1:1 ratio)
├── shap_bar.png              # Bar plot (1:1 ratio)
└── metrics.json
```

### Comparison Figures

```
results/{target}/comparison/
├── comparison_roc.png
├── comparison_pr.png
├── comparison_calibration.png
├── comparison_combined.{png,tiff,pdf}    # ROC+PR+Cal combined
└── comparison_shap.{png,tiff,pdf}        # SHAP beeswarm (2x3 subplot)
```

### Tables

```
results/{target}/tables/
├── table1_train_test.xlsx      # Train vs Test baseline
├── table1_by_{target}.xlsx     # By outcome baseline
└── model_performance.xlsx      # All metrics with 95% Bootstrap CI
```

### Performance Table Format

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | PPV | NPV | F1 Score | Threshold |
|-------|-------|-------|----------|-------------|-------------|-----|-----|----------|-----------|
| XGBoost | 0.82 (0.80-0.84) | 0.61 (0.57-0.65) | ... | ... | ... | ... | ... | ... | 0.32 |

- Threshold determined by **Youden Index** (maximizes Sensitivity + Specificity - 1)
- 95% CI from **1000 bootstrap** iterations

## Key Features

### Package Compatibility (Python 3.8 + Old Packages)

- **xgboost 0.80**: Uses `seed`/`silent` instead of `random_state`/`verbosity`
- **shap 0.32**: numpy compatibility patch (`np.int`, `np.float`, etc.), `TreeExplainer` with `KernelExplainer` fallback
- **tableone 0.7.12**: Dynamic parameter detection (`overall`, `htest_name` checked via `inspect`)
- All models saved as `.pkl` (no `.json`/`.cbm` format dependencies)

### SHAP Analysis

- **Tree models** (DT, RF, XGBoost, LightGBM): `TreeExplainer` first, falls back to `KernelExplainer` if unsupported
- **ANN (MLP)**: `KernelExplainer` with auto-limited samples (max 500) for feasible runtime
- **Stratified sampling**: 2000 samples (tree) / 500 samples (kernel), preserving outcome ratio
- Missing indicator features excluded from SHAP plots by default
- 1:1 aspect ratio for all SHAP plots

### Multi-Target Support

- `outA`: Primary outcome
- `out2`: Secondary outcome
- Fully separate directories for data, models, and results

### Missing Indicator

- Variables with >5% missing rate get `{var}_missing` binary indicator
- Included in training, excluded from SHAP visualizations

## Offline Installation

```bash
# 1. On online machine: download packages
bash scripts/download_packages.sh

# 2. Copy entire project to offline machine

# 3a. Install to new venv
bash scripts/install_offline.sh

# 3b. Or install to existing environment
conda activate your_env
bash scripts/install_to_existing_env.sh
```

## Configuration

### Random State

All random seeds are set to `1004` for reproducibility.

### Parameter Grids

Edit `PARAM_GRIDS` / `PARAM_GRIDS_SMALL` in `code/train_gridsearch.py`.

### Feature Lists

Edit `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in `code/preprocessing.py`.

## License

This project is for research purposes.
