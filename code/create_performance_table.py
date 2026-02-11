"""
논문용 모델 성능 비교 테이블 생성 스크립트
- Youden Index 기반 최적 threshold
- Bootstrap 95% CI
- AUROC, AUPRC, Accuracy, Sensitivity, Specificity, PPV, NPV
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


def find_optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Youden Index를 사용하여 최적 threshold 찾기
    
    Youden Index = Sensitivity + Specificity - 1 = TPR - FPR
    
    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률
        
    Returns:
        (optimal_threshold, youden_index)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Youden Index 계산
    youden = tpr - fpr
    
    # 최대 Youden Index의 인덱스
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden[optimal_idx]
    
    return optimal_threshold, optimal_youden


def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    특정 threshold에서 분류 지표 계산
    
    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률
        threshold: 분류 임계값
        
    Returns:
        지표 딕셔너리
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'threshold': threshold
    }
    
    return metrics


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    use_youden: bool = True,
    random_state: int = 1004
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap을 사용한 성능 지표 및 신뢰구간 계산
    
    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률
        n_bootstrap: bootstrap 반복 횟수
        ci_level: 신뢰구간 수준 (기본 95%)
        use_youden: Youden index 기반 threshold 사용 여부
        random_state: 랜덤 시드
        
    Returns:
        각 지표별 point estimate, lower CI, upper CI
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # 원본 데이터에서 최적 threshold 계산 (Youden Index)
    if use_youden:
        optimal_threshold, _ = find_optimal_threshold_youden(y_true, y_prob)
    else:
        optimal_threshold = 0.5
    
    # 원본 지표 계산
    original_metrics = calculate_metrics_at_threshold(y_true, y_prob, optimal_threshold)
    
    # Bootstrap 샘플링
    bootstrap_results = {metric: [] for metric in original_metrics.keys()}
    
    for i in range(n_bootstrap):
        # 복원 추출
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # 클래스가 하나만 있으면 건너뛰기
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        # 각 bootstrap 샘플에서도 Youden threshold 재계산 (더 정확한 CI)
        if use_youden:
            boot_threshold, _ = find_optimal_threshold_youden(y_true_boot, y_prob_boot)
        else:
            boot_threshold = optimal_threshold
        
        boot_metrics = calculate_metrics_at_threshold(y_true_boot, y_prob_boot, boot_threshold)
        
        for metric, value in boot_metrics.items():
            bootstrap_results[metric].append(value)
    
    # CI 계산
    alpha = 1 - ci_level
    results = {}
    
    for metric in original_metrics.keys():
        values = np.array(bootstrap_results[metric])
        if len(values) > 0:
            lower = np.percentile(values, alpha / 2 * 100)
            upper = np.percentile(values, (1 - alpha / 2) * 100)
        else:
            lower = upper = original_metrics[metric]
        
        results[metric] = {
            'point': original_metrics[metric],
            'lower': lower,
            'upper': upper
        }
    
    return results


def format_metric_with_ci(point: float, lower: float, upper: float, decimals: int = 3) -> str:
    """지표를 CI와 함께 포맷팅"""
    return f"{point:.{decimals}f} ({lower:.{decimals}f}-{upper:.{decimals}f})"


def load_model(model_path: str) -> Any:
    """모델 로드 (모든 모델 pkl로 통일)"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"⚠️ 모델 로드 실패: {os.path.basename(model_path)} - {e}")
        return None


def evaluate_single_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 1000,
    model_name: str = None
) -> Dict[str, Dict[str, float]]:
    """
    단일 모델 평가 (Bootstrap CI 포함)
    
    Args:
        model_path: 모델 파일 경로
        X_test: 테스트 특성
        y_test: 테스트 타겟
        n_bootstrap: Bootstrap 반복 횟수
        model_name: 모델 이름 (없으면 파일명에서 추출)
        
    Returns:
        Bootstrap CI 포함 성능 지표
    """
    # 모델 이름 추출
    if model_name is None:
        model_name = os.path.basename(model_path)
        model_name = model_name.replace('_best_model', '').replace('_model', '')
        model_name = model_name.replace('.pkl', '').replace('.json', '').replace('.cbm', '').replace('.txt', '')
    
    # 모델 로드 및 예측
    model = load_model(model_path)
    
    if model is None:
        print(f"   ⚠️ {model_name}: 모델 로드 실패 - 건너뜀")
        return None
    
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
    
    # Bootstrap 평가
    print(f"   🔄 {model_name}: Bootstrap CI 계산 중 (n={n_bootstrap})...")
    results = bootstrap_metrics(y_test, y_prob, n_bootstrap=n_bootstrap, use_youden=True)
    
    return results


def create_performance_table(
    model_paths: List[str],
    data_dir: str = '../data/processed',
    n_bootstrap: int = 1000,
    output_path: str = None,
    model_names: List[str] = None
) -> pd.DataFrame:
    """
    여러 모델의 성능 비교 테이블 생성
    
    Args:
        model_paths: 모델 파일 경로 리스트
        data_dir: 전처리된 데이터 디렉토리
        n_bootstrap: Bootstrap 반복 횟수
        output_path: 저장 경로
        model_names: 모델 이름 리스트 (없으면 파일명에서 추출)
        
    Returns:
        성능 비교 테이블 DataFrame
    """
    # 데이터 로드
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print("=" * 70)
    print("📊 논문용 모델 성능 비교 테이블 생성")
    print("=" * 70)
    print(f"테스트 데이터: {X_test.shape[0]} 샘플")
    print(f"Bootstrap 반복: {n_bootstrap}")
    print(f"Threshold: Youden Index 기준")
    print()
    
    # 각 모델 평가
    all_results = {}
    
    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            print(f"   ⚠️ 모델 파일 없음: {model_path}")
            continue
        
        name = model_names[i] if model_names and i < len(model_names) else None
        if name is None:
            name = os.path.basename(model_path)
            name = name.replace('_best_model', '').replace('_model', '')
            name = name.replace('.pkl', '').replace('.json', '').replace('.cbm', '').replace('.txt', '')
        
        results = evaluate_single_model(model_path, X_test, y_test, n_bootstrap, name)
        if results is not None:
            all_results[name] = results
    
    # 테이블 생성
    metrics_order = ['auroc', 'auprc', 'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'threshold']
    metrics_labels = {
        'auroc': 'AUROC',
        'auprc': 'AUPRC',
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'ppv': 'PPV',
        'npv': 'NPV',
        'f1': 'F1 Score',
        'threshold': 'Optimal Threshold'
    }
    
    # 모델 표시 이름 매핑
    display_names = {
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'ann': 'MLP',
    }
    
    # DataFrame 구성
    rows = []
    for model_name, results in all_results.items():
        row = {'Model': display_names.get(model_name, model_name)}
        for metric in metrics_order:
            if metric in results:
                m = results[metric]
                if metric == 'threshold':
                    row[metrics_labels[metric]] = f"{m['point']:.3f}"
                else:
                    row[metrics_labels[metric]] = format_metric_with_ci(m['point'], m['lower'], m['upper'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 출력
    print("\n" + "=" * 70)
    print("📋 성능 비교 테이블 (95% Bootstrap CI)")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.xlsx'):
            # Excel 저장 시 포맷팅
            df.to_excel(output_path, index=False)
        elif output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.tex'):
            # LaTeX 테이블
            latex_str = df.to_latex(index=False, escape=False)
            with open(output_path, 'w') as f:
                f.write(latex_str)
        else:
            df.to_csv(output_path + '.csv', index=False)
        
        print(f"\n✅ 테이블 저장: {output_path}")
    
    return df


def create_detailed_results(
    model_paths: List[str],
    data_dir: str = '../data/processed',
    n_bootstrap: int = 1000,
    output_dir: str = '../results/tables'
) -> Dict[str, pd.DataFrame]:
    """
    상세 결과 테이블 생성 (Excel 여러 시트)
    
    Args:
        model_paths: 모델 파일 경로 리스트
        data_dir: 전처리된 데이터 디렉토리
        n_bootstrap: Bootstrap 반복 횟수
        output_dir: 출력 디렉토리
        
    Returns:
        결과 딕셔너리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 메인 테이블
    main_table = create_performance_table(
        model_paths, data_dir, n_bootstrap,
        output_path=os.path.join(output_dir, 'model_performance_comparison.xlsx')
    )
    
    # 추가: Point estimates만 있는 간단한 테이블
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    simple_rows = []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
        
        model = load_model(model_path)
        model_name = os.path.basename(model_path).replace('_best_model', '').replace('_model', '')
        model_name = model_name.replace('.pkl', '').replace('.json', '').replace('.cbm', '').replace('.txt', '')
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        
        threshold, youden = find_optimal_threshold_youden(y_test, y_prob)
        metrics = calculate_metrics_at_threshold(y_test, y_prob, threshold)
        metrics['model'] = model_name
        metrics['youden_index'] = youden
        simple_rows.append(metrics)
    
    simple_df = pd.DataFrame(simple_rows)
    cols = ['model', 'auroc', 'auprc', 'accuracy', 'sensitivity', 'specificity', 
            'ppv', 'npv', 'f1', 'threshold', 'youden_index']
    simple_df = simple_df[[c for c in cols if c in simple_df.columns]]
    
    simple_path = os.path.join(output_dir, 'model_performance_simple.csv')
    simple_df.to_csv(simple_path, index=False)
    print(f"\n✅ 간단한 테이블 저장: {simple_path}")
    
    return {'main': main_table, 'simple': simple_df}


def main():
    """메인 실행 함수"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='논문용 모델 성능 비교 테이블 생성')
    parser.add_argument('--models-dir', type=str, default='../models',
                        help='모델 파일 디렉토리')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='전처리된 데이터 디렉토리')
    parser.add_argument('--output', type=str, default='../results/tables/model_performance.xlsx',
                        help='출력 파일 경로')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Bootstrap 반복 횟수')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='평가할 모델 파일 리스트 (없으면 models-dir에서 자동 검색)')
    
    args = parser.parse_args()
    
    # 모델 파일 찾기
    if args.models:
        model_paths = args.models
    else:
        # 자동 검색
        patterns = ['*.json', '*.pkl', '*.cbm']
        model_paths = []
        for pattern in patterns:
            model_paths.extend(glob.glob(os.path.join(args.models_dir, f'*best_model{pattern}')))
            model_paths.extend(glob.glob(os.path.join(args.models_dir, f'*_model{pattern}')))
        
        # 중복 제거 및 meta 파일 제외
        model_paths = list(set(model_paths))
        model_paths = [p for p in model_paths if 'meta' not in p]
    
    if not model_paths:
        print("❌ 평가할 모델 파일이 없습니다.")
        print(f"   검색 경로: {args.models_dir}")
        return
    
    print(f"📂 발견된 모델 파일: {len(model_paths)}개")
    for p in model_paths:
        print(f"   - {os.path.basename(p)}")
    
    # 테이블 생성
    create_performance_table(
        model_paths=model_paths,
        data_dir=args.data_dir,
        n_bootstrap=args.n_bootstrap,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
