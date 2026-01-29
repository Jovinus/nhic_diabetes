# Diabetes Prediction Pipeline

Machine learning pipeline for predicting diabetes incidence in gallstone patients.

## 📁 Project Structure

```
nhic_diabetes/
├── code/
│   ├── make_dummy.py                  # Generate dummy data
│   ├── preprocessing.py               # Data preprocessing (with Missing Indicator)
│   ├── train_gridsearch.py            # GridSearchCV-based model training
│   ├── evaluate.py                    # Model evaluation & SHAP analysis
│   ├── create_table1.py               # Baseline characteristics table
│   ├── create_performance_table.py    # Performance comparison table (Bootstrap CI)
│   └── create_comparison_figures.py   # Model comparison figures for paper
├── scripts/
│   ├── download_packages.sh           # Download packages for offline
│   ├── install_offline.sh             # Install packages in offline (creates venv)
│   ├── install_to_existing_env.sh     # Install packages to existing environment
│   └── run_pipeline.sh                # Run entire pipeline
├── notebooks/
│   └── run_pipeline.ipynb             # Jupyter notebook version of pipeline
├── data/
│   ├── dummy_diabetes_data.csv        # Raw data
│   └── processed/
│       ├── outA/                      # Preprocessed data for outA target
│       └── out2/                      # Preprocessed data for out2 target
├── models/
│   ├── outA/                          # Trained models for outA target
│   └── out2/                          # Trained models for out2 target
├── results/
│   ├── outA/
│   │   ├── {model_name}/              # Individual model results
│   │   ├── comparison/                # Model comparison figures
│   │   └── tables/                    # Table 1, Performance table
│   └── out2/
│       ├── {model_name}/
│       ├── comparison/
│       └── tables/
├── packages/                          # Offline installation packages
└── requirements.txt                   # Python dependencies
```

## 🚀 Quick Start

### Run Pipeline (Both Targets)

```bash
# Full pipeline for both outA and out2 targets
./scripts/run_pipeline.sh

# Quick test (small parameter grid, fewer bootstraps)
./scripts/run_pipeline.sh --small-grid --n-bootstrap 100

# Skip dummy data generation (use real data)
./scripts/run_pipeline.sh --skip-dummy

# Run specific target only
./scripts/run_pipeline.sh --targets "outA"

# Run specific models only
./scripts/run_pipeline.sh --models "decision_tree random_forest xgboost"
```

### Jupyter Notebook

For step-by-step execution:
```bash
cd notebooks
jupyter notebook run_pipeline.ipynb
```

## 📖 Detailed Usage

### 1. Environment Setup

#### Online Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

#### Offline Environment (Air-gapped Network)

```bash
# 1. Download packages on online machine (Python 3.8)
./scripts/download_packages.sh

# 2. Copy entire project folder to offline machine

# 3. Option A: Create new venv and install
./scripts/install_offline.sh

# 3. Option B: Install to existing conda/venv environment
conda activate your_env  # or source your_venv/bin/activate
./scripts/install_to_existing_env.sh
```

### 2. Run Pipeline Options

```bash
./scripts/run_pipeline.sh [OPTIONS]

Options:
  --skip-dummy        Skip dummy data generation
  --small-grid        Use reduced parameter grid (faster)
  --n-bootstrap N     Bootstrap iterations (default: 1000)
  --targets "T1 T2"   Target variables (default: "outA out2")
  --models "M1 M2"    Models to train (default: all 5 models)
```

**Supported Models:**
| Model | Name | Save Format |
|-------|------|-------------|
| Decision Tree | `decision_tree` | `.pkl` |
| Random Forest | `random_forest` | `.pkl` |
| XGBoost | `xgboost` | `.json` |
| CatBoost | `catboost` | `.cbm` |
| ANN (MLP) | `ann` | `.pkl` |

### 3. Individual Scripts

#### Data Preprocessing

```bash
cd code

python preprocessing.py \
    --data ../data/your_data.csv \
    --output ../data/processed/outA \
    --target outA \
    --add-missing-indicator \
    --missing-threshold 0.05
```

#### Baseline Characteristics (Table 1)

```bash
python create_table1.py \
    --data ../data/your_data.csv \
    --output ../results/outA/tables \
    --target outA
```

#### Model Training

```bash
python train_gridsearch.py \
    --data-dir ../data/processed/outA \
    --output ../models/outA \
    --models decision_tree random_forest xgboost catboost ann \
    --cv 5 \
    --scoring roc_auc \
    --small-grid  # optional: for quick test
```

#### Model Evaluation & SHAP

```bash
python evaluate.py \
    --model ../models/outA/xgboost_best_model.json \
    --data-dir ../data/processed/outA \
    --output ../results/outA \
    --shap
```

#### Performance Comparison Table (Bootstrap CI)

```bash
python create_performance_table.py \
    --models-dir ../models/outA \
    --data-dir ../data/processed/outA \
    --output ../results/outA/tables/model_performance.xlsx \
    --n-bootstrap 1000
```

#### Model Comparison Figures

```bash
python create_comparison_figures.py \
    --models-dir ../models/outA \
    --data-dir ../data/processed/outA \
    --output ../results/outA/comparison
```

## 📊 Output Files

### Per Target Directory Structure

```
results/{target}/
├── {model_name}/
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── calibration_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png           # SHAP Beeswarm Plot
│   ├── shap_bar.png               # SHAP Bar Plot
│   └── metrics.json
├── comparison/
│   ├── comparison_roc.png         # ROC curves (all models)
│   ├── comparison_pr.png          # PR curves (all models)
│   ├── comparison_calibration.png # Calibration curves
│   ├── comparison_combined.png    # Combined figure (paper-ready)
│   ├── comparison_combined.tiff   # High-resolution TIFF
│   ├── comparison_combined.pdf    # PDF version
│   ├── comparison_shap.png        # SHAP comparison (2x3 subplot)
│   ├── comparison_shap.tiff
│   └── comparison_shap.pdf
└── tables/
    ├── table1_train_test.xlsx     # Train vs Test comparison
    ├── table1_by_{target}.xlsx    # By outcome comparison
    └── model_performance.xlsx     # Performance with Bootstrap CI
```

### Performance Table Format

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | PPV | NPV | Threshold |
|-------|-------|-------|----------|-------------|-------------|-----|-----|-----------|
| XGBoost | 0.823 (0.801-0.844) | 0.612 (0.571-0.652) | ... | ... | ... | ... | ... | 0.324 |

- **AUROC, AUPRC**: Probability-based metrics
- **Others**: Calculated at Youden Index optimal threshold
- **95% CI**: Based on 1000 bootstrap iterations

## 📊 Key Features

### Multi-Target Support
- `outA`: Primary outcome (any diabetes)
- `out2`: Secondary outcome (type 2 diabetes)
- Results stored separately for each target

### Missing Indicator
- Variables with >5% missing rate get `{var}_missing` indicator feature
- Excluded from SHAP visualizations by default

### Youden Index Threshold
- Optimal classification cutoff: maximizes (Sensitivity + Specificity - 1)
- Displayed on ROC curves

### Bootstrap Confidence Intervals
- 1000 iterations for 95% CI
- Included in performance comparison tables

### SHAP Analysis
- Uses native SHAP library plotting functions
- Default SHAP color schemes
- Missing indicators excluded from visualizations

## 🔧 Configuration

### Random State
All random seeds are set to `1004` for reproducibility.

### Parameter Grid Customization

Edit `PARAM_GRIDS` in `train_gridsearch.py`:

```python
PARAM_GRIDS = {
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        ...
    },
    ...
}
```

### Feature Configuration

Edit top variables in `preprocessing.py`:

```python
NUMERIC_FEATURES = ['age', 'BMI', 'SBP', ...]
CATEGORICAL_FEATURES = ['gender', 'smoking', ...]
```

## 📋 Requirements

- **Python**: 3.8.16
- **Key packages**:
  - numpy==1.24.4
  - pandas==2.0.3
  - scikit-learn==1.3.2
  - xgboost==2.0.3
  - catboost==1.2.2
  - shap==0.44.1
  - tableone==0.7.12
  - matplotlib==3.7.5

See `requirements.txt` for full list.

## 📝 Example Output

### Model Evaluation Results

```
📊 xgboost Evaluation Results
==================================================
  AUROC (C-statistic): 0.8234
  AUPRC:               0.6125
  Threshold (Youden):  0.3241
  Youden Index:        0.5123
--------------------------------------------------
  Accuracy:            0.7512
  Sensitivity (Recall):0.7243
  Specificity:         0.7632
  PPV (Precision):     0.5891
  NPV:                 0.8521
  F1 Score:            0.6492
  Brier Score:         0.1523
```

### Pipeline Completion

```
📁 Results Location:

  [Data]
    - Raw: data/dummy_diabetes_data.csv
    - Processed (outA): data/processed/outA/
    - Processed (out2): data/processed/out2/

  [Models]
    - outA: models/outA/*_best_model.*
    - out2: models/out2/*_best_model.*

  [Results]
    - outA: results/outA/{model_name}/, results/outA/comparison/
    - out2: results/out2/{model_name}/, results/out2/comparison/

  [Tables]
    - Table 1: results/{target}/tables/table1_*.xlsx
    - Performance: results/{target}/tables/model_performance.xlsx
```

## 📄 License

This project is for research purposes.
