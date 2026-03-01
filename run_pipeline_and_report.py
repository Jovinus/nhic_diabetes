#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전체 파이프라인 + PDF 보고서 생성 + export 반출 래퍼 스크립트

실행 예시:
    # 빠른 검증 (1000 샘플)
    python run_pipeline_and_report.py --n-samples 1000 --small-grid --n-bootstrap 50

    # 전체 실행
    python run_pipeline_and_report.py

    # 실제 데이터 사용
    python run_pipeline_and_report.py --data data/real_data.csv --skip-dummy

출력 구조:
    results/          분석 결과 (모델별, 비교 figure 등)
    report/           보고서 (PDF 한글/영문)
    export/           반출용 (타겟_카테고리_파일명 형식으로 flat copy)
"""

import os
import sys
import shutil
import glob
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
REPORT_DIR = os.path.join(SCRIPT_DIR, 'report')
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'export')

TARGETS = ['outA', 'out2']


def run_full_pipeline(args):
    """Step 1: run_all.py 파이프라인 실행."""
    print("=" * 70)
    print("  [1/3] ML Pipeline")
    print("=" * 70)

    from run_all import run_pipeline

    run_pipeline(
        targets=args.targets.split(),
        models=args.models.split(),
        skip_dummy=args.skip_dummy,
        small_grid=args.small_grid,
        n_bootstrap=args.n_bootstrap,
        n_samples=args.n_samples,
        cv_folds=args.cv,
        data_path=args.data,
        use_gpu=args.use_gpu,
    )


def run_pdf_report():
    """Step 2: PDF 보고서 생성 (한글 + 영문)."""
    print("\n" + "=" * 70)
    print("  [2/3] PDF Report Generation")
    print("=" * 70)

    from create_pdf_report import create_pdf_report
    create_pdf_report()


def export_all(targets):
    """Step 3: report/ 및 export/ 디렉토리에 결과 반출."""
    print("\n" + "=" * 70)
    print("  [3/3] Export Results")
    print("=" * 70)

    for d in [REPORT_DIR, EXPORT_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── PDF 보고서 → report/ 및 export/ ──
    copied = 0
    for lang in ['ko', 'en']:
        fname = 'analysis_report_{}.pdf'.format(lang)
        src = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(REPORT_DIR, fname))
            shutil.copy2(src, os.path.join(EXPORT_DIR, fname))
            copied += 1

    print("  PDF reports -> report/, export/  ({} files)".format(copied))

    # ── 각 타겟별 결과 → export/ (flat copy: {target}_{category}_{filename}) ──
    exported = 0
    for target in targets:
        # Tables (xlsx)
        tables_dir = os.path.join(RESULTS_DIR, target, 'tables')
        if os.path.isdir(tables_dir):
            for f in os.listdir(tables_dir):
                if f.endswith('.xlsx'):
                    src = os.path.join(tables_dir, f)
                    dst = os.path.join(EXPORT_DIR,
                                       '{}_tables_{}'.format(target, f))
                    shutil.copy2(src, dst)
                    exported += 1

        # Comparison figures (png, tiff, pdf)
        comp_dir = os.path.join(RESULTS_DIR, target, 'comparison')
        if os.path.isdir(comp_dir):
            for f in os.listdir(comp_dir):
                src = os.path.join(comp_dir, f)
                dst = os.path.join(EXPORT_DIR,
                                   '{}_comparison_{}'.format(target, f))
                shutil.copy2(src, dst)
                exported += 1

        # Per-model results (metrics.json, figures)
        model_dirs = ['logistic_regression', 'decision_tree', 'random_forest',
                      'xgboost', 'lightgbm', 'ann']
        for mdl in model_dirs:
            mdl_dir = os.path.join(RESULTS_DIR, target, mdl)
            if not os.path.isdir(mdl_dir):
                continue
            for f in os.listdir(mdl_dir):
                src = os.path.join(mdl_dir, f)
                if os.path.isfile(src):
                    dst = os.path.join(EXPORT_DIR,
                                       '{}_{}_{}'.format(target, mdl, f))
                    shutil.copy2(src, dst)
                    exported += 1

    # summary_report_draft.md
    md_src = os.path.join(RESULTS_DIR, 'summary_report_draft.md')
    if os.path.exists(md_src):
        shutil.copy2(md_src, os.path.join(EXPORT_DIR, 'summary_report_draft.md'))
        exported += 1

    print("  Results -> export/  ({} files)".format(exported))

    # ── 요약 ──
    print("\n  Output directories:")
    print("    report/ : PDF 보고서 (한글/영문)")
    print("    export/ : 전체 반출 파일 (flat)")
    print("    results/: 원본 분석 결과 (계층 구조)")


def main():
    parser = argparse.ArgumentParser(
        description='Full Pipeline + Report + Export')
    parser.add_argument('--skip-dummy', action='store_true',
                        help='Skip dummy data generation')
    parser.add_argument('--small-grid', action='store_true',
                        help='Use small parameter grid (quick test)')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of dummy data samples')
    parser.add_argument('--targets', type=str, default='outA out2',
                        help='Target variables (space-separated)')
    parser.add_argument('--models', type=str,
                        default='logistic_regression decision_tree '
                                'random_forest xgboost lightgbm ann',
                        help='Models to train (space-separated)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data CSV file')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable GPU for XGBoost')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Skip ML pipeline, only generate reports')

    args = parser.parse_args()

    # Step 1: ML pipeline
    if not args.skip_pipeline:
        run_full_pipeline(args)
    else:
        print("  Skipping ML pipeline (--skip-pipeline)")

    # Step 2: PDF report
    run_pdf_report()

    # Step 3: Export
    export_all(args.targets.split())

    print("\n" + "=" * 70)
    print("  All Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
