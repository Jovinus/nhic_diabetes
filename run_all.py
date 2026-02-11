#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전체 파이프라인 실행 스크립트 (Python)
- 더미 데이터 생성 (옵션)
- 각 타겟(outA, out2)에 대해:
    - Table 1 생성
    - 전처리
    - 모델 학습 (GridSearchCV)
    - 모델 평가 + SHAP
    - 성능 비교 테이블
    - 비교 Figure

사용법:
    python run_all.py
    python run_all.py --skip-dummy --small-grid --n-bootstrap 100
    python run_all.py --targets outA --models "decision_tree random_forest"
"""

import os
import sys
import argparse
import warnings
import glob
import numpy as np
warnings.filterwarnings('ignore')

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, 'code')
sys.path.insert(0, CODE_DIR)


def run_pipeline(
    targets=None,
    models=None,
    skip_dummy=False,
    small_grid=False,
    n_bootstrap=1000,
    cv_folds=5,
    scoring='roc_auc',
    missing_threshold=0.05,
    data_path=None
):
    """전체 파이프라인 실행"""
    
    if targets is None:
        targets = ['outA', 'out2']
    if models is None:
        models = ['decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'ann']
    if data_path is None:
        data_path = os.path.join(SCRIPT_DIR, 'data', 'dummy_diabetes_data.csv')
    
    print("=" * 70)
    print("  Diabetes Prediction Pipeline (Multi-Target)")
    print("=" * 70)
    print(f"  Targets: {targets}")
    print(f"  Models: {models}")
    print(f"  Small grid: {small_grid}")
    print(f"  Bootstrap: {n_bootstrap}")
    print(f"  Data: {data_path}")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Dummy data
    # =========================================================================
    if not skip_dummy:
        print("\n[Step 1] Generating dummy data...")
        from make_dummy import generate_dummy_data
        df = generate_dummy_data(n_samples=10000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"  Saved: {data_path} ({len(df)} samples)")
    else:
        print("\n[Step 1] Skipping dummy data generation")
    
    # =========================================================================
    # For each target
    # =========================================================================
    from create_table1 import create_all_tables
    from preprocessing import preprocess_and_save
    from train_gridsearch import ModelTrainer
    from evaluate import evaluate_model, run_shap_analysis
    from create_performance_table import create_performance_table
    from create_comparison_figures import create_comparison_figures
    
    for target in targets:
        print(f"\n{'#'*70}")
        print(f"  TARGET: {target}")
        print(f"{'#'*70}")
        
        # Paths
        processed_dir = os.path.join(SCRIPT_DIR, 'data', 'processed', target)
        models_dir = os.path.join(SCRIPT_DIR, 'models', target)
        results_dir = os.path.join(SCRIPT_DIR, 'results', target)
        tables_dir = os.path.join(results_dir, 'tables')
        comparison_dir = os.path.join(results_dir, 'comparison')
        
        for d in [processed_dir, models_dir, results_dir, tables_dir, comparison_dir]:
            os.makedirs(d, exist_ok=True)
        
        # =====================================================================
        # Step 2: Table 1
        # =====================================================================
        print(f"\n--- [{target}] Step 2: Baseline Characteristics ---")
        try:
            create_all_tables(
                data_path=data_path,
                output_dir=tables_dir,
                target_col=target
            )
        except Exception as e:
            print(f"  Error: {e}")
        
        # =====================================================================
        # Step 3: Preprocessing
        # =====================================================================
        print(f"\n--- [{target}] Step 3: Preprocessing ---")
        try:
            preprocess_and_save(
                data_path=data_path,
                output_dir=processed_dir,
                target_col=target,
                add_missing_indicator=True,
                missing_threshold=missing_threshold
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        # Load data
        X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
        
        with open(os.path.join(processed_dir, 'feature_names.txt'), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        # =====================================================================
        # Step 4: Model Training
        # =====================================================================
        print(f"\n--- [{target}] Step 4: Model Training ---")
        
        trainer = ModelTrainer(
            cv=cv_folds, scoring=scoring, use_small_grid=small_grid
        )
        
        for model_name in models:
            try:
                trainer.train_model(model_name, X_train, y_train, verbose=1)
                result = trainer.evaluate_model(model_name, X_test, y_test)
                print(f"  {model_name}: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}")
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
        
        trainer.save_all_models(models_dir, feature_names)
        
        # =====================================================================
        # Step 5: Evaluation + SHAP
        # =====================================================================
        print(f"\n--- [{target}] Step 5: Evaluation & SHAP ---")
        
        # 학습 순서대로 모델 파일 목록 구성 (glob은 filesystem order라 순서 보장 안됨)
        model_files = []
        for m in models:
            mf = os.path.join(models_dir, f'{m}_best_model.pkl')
            if os.path.exists(mf):
                model_files.append(mf)
        
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('_best_model.pkl', '')
            try:
                evaluate_model(
                    model_path=model_file,
                    data_dir=processed_dir,
                    output_dir=results_dir,
                    model_name=model_name
                )
                run_shap_analysis(
                    model_path=model_file,
                    data_dir=processed_dir,
                    output_dir=results_dir,
                    model_name=model_name
                )
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
        
        # =====================================================================
        # Step 6: Performance Table
        # =====================================================================
        print(f"\n--- [{target}] Step 6: Performance Table ---")
        try:
            create_performance_table(
                model_paths=model_files,
                data_dir=processed_dir,
                n_bootstrap=n_bootstrap,
                output_path=os.path.join(tables_dir, 'model_performance.xlsx')
            )
        except Exception as e:
            print(f"  Error: {e}")
        
        # =====================================================================
        # Step 7: Comparison Figures
        # =====================================================================
        print(f"\n--- [{target}] Step 7: Comparison Figures ---")
        try:
            create_comparison_figures(
                models_dir=models_dir,
                data_dir=processed_dir,
                output_dir=comparison_dir
            )
        except Exception as e:
            print(f"  Error: {e}")
        
        print(f"\n  Target {target} complete!")
    
    # =========================================================================
    # Done
    # =========================================================================
    print(f"\n{'='*70}")
    print("  Pipeline Complete!")
    print(f"{'='*70}")
    
    for target in targets:
        print(f"\n  [{target}]")
        print(f"    Data:       data/processed/{target}/")
        print(f"    Models:     models/{target}/")
        print(f"    Results:    results/{target}/")
        print(f"    Tables:     results/{target}/tables/")
        print(f"    Comparison: results/{target}/comparison/")


def main():
    parser = argparse.ArgumentParser(description='Diabetes Prediction Pipeline')
    parser.add_argument('--skip-dummy', action='store_true',
                        help='Skip dummy data generation')
    parser.add_argument('--small-grid', action='store_true',
                        help='Use small parameter grid (quick test)')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations')
    parser.add_argument('--targets', type=str, default='outA out2',
                        help='Target variables (space-separated)')
    parser.add_argument('--models', type=str, 
                        default='decision_tree random_forest xgboost lightgbm ann',
                        help='Models to train (space-separated)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data CSV file')
    parser.add_argument('--cv', type=int, default=5,
                        help='CV folds')
    
    args = parser.parse_args()
    
    run_pipeline(
        targets=args.targets.split(),
        models=args.models.split(),
        skip_dummy=args.skip_dummy,
        small_grid=args.small_grid,
        n_bootstrap=args.n_bootstrap,
        cv_folds=args.cv,
        data_path=args.data
    )


if __name__ == '__main__':
    main()
