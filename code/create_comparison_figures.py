"""
모델 비교 시각화 스크립트
- ROC Curve 비교
- PR Curve 비교
- Calibration Curve 비교
- SHAP Beeswarm 비교
- 논문용 Figure 생성
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import argparse
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_curve, precision_recall_curve, 
    roc_auc_score, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# numpy 호환성 패치 (shap 0.32 + numpy>=1.24)
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'object'):
    np.object = object

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP 미설치 - SHAP 비교 Figure 생성 불가")

# Missing Indicator 접미사
MISSING_INDICATOR_SUFFIX = '_missing'

# 폰트 설정
def set_font():
    """시스템에 맞는 폰트 설정"""
    import platform
    system = platform.system()
    
    if system == 'Darwin':
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'AppleGothic'
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        # Linux - fallback to DejaVu Sans
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

set_font()

# 모델 표시 이름
MODEL_DISPLAY_NAMES = {
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'catboost': 'CatBoost',
    'lightgbm': 'LightGBM',
    'ann': 'MLP',
    'logistic': 'Logistic Regression'
}

# 모델별 색상
MODEL_COLORS = {
    'decision_tree': '#1f77b4',
    'random_forest': '#ff7f0e',
    'xgboost': '#2ca02c',
    'catboost': '#d62728',
    'lightgbm': '#9467bd',
    'ann': '#8c564b',
    'logistic': '#e377c2'
}

# 모델별 라인 스타일
MODEL_LINESTYLES = {
    'decision_tree': '-',
    'random_forest': '-',
    'xgboost': '-',
    'catboost': '-',
    'lightgbm': '-',
    'ann': '--',
    'logistic': '--'
}


def _save_figure(fig, save_path, dpi=500, bbox_inches='tight', pad_inches=0.1):
    """Figure를 png, tiff, pdf 3종으로 저장"""
    base, _ = os.path.splitext(save_path)
    for fmt in ['png', 'tiff', 'pdf']:
        out = f"{base}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, format=fmt)
    print(f"✅ Figure 저장: {base}.{{png,tiff,pdf}}")


def load_model(model_path: str) -> Any:
    """모델 로드 (모든 모델 pkl로 통일)"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"⚠️ 모델 로드 실패: {os.path.basename(model_path)} - {e}")
        return None


def find_models(models_dir: str) -> Dict[str, str]:
    """모델 디렉토리에서 모든 모델 찾기"""
    models = {}
    
    patterns = [
        ('decision_tree', 'decision_tree_best_model.pkl'),
        ('random_forest', 'random_forest_best_model.pkl'),
        ('xgboost', 'xgboost_best_model.pkl'),
        ('lightgbm', 'lightgbm_best_model.pkl'),
        ('ann', 'ann_best_model.pkl'),
        ('logistic', 'logistic_best_model.pkl')
    ]
    
    for model_name, filename in patterns:
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            models[model_name] = model_path
    
    return models


def get_predictions(model: Any, X: np.ndarray) -> np.ndarray:
    """모델 예측 확률 반환"""
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            return y_prob[:, 1]
        return y_prob
    elif hasattr(model, 'predict'):
        return model.predict(X)
    else:
        raise ValueError("모델이 predict_proba 또는 predict 메서드를 지원하지 않습니다.")


def plot_roc_comparison(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    ROC Curve 비교 시각화
    
    Args:
        models_data: {model_name: (fpr, tpr, auc)}
        save_path: 저장 경로
        figsize: Figure 크기
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (fpr, tpr, auc_score) in models_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        linestyle = MODEL_LINESTYLES.get(model_name, '-')
        
        ax.plot(fpr, tpr, 
                label=f'{display_name} (AUC = {auc_score:.3f})',
                color=color, 
                linestyle=linestyle,
                linewidth=2)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity (False Positive Rate)')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_pr_comparison(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    baseline: float = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Precision-Recall Curve 비교 시각화
    
    Args:
        models_data: {model_name: (recall, precision, ap)}
        baseline: 베이스라인 (양성 클래스 비율)
        save_path: 저장 경로
        figsize: Figure 크기
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (recall, precision, ap_score) in models_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        linestyle = MODEL_LINESTYLES.get(model_name, '-')
        
        ax.plot(recall, precision,
                label=f'{display_name} (AP = {ap_score:.3f})',
                color=color,
                linestyle=linestyle,
                linewidth=2)
    
    # Baseline
    if baseline is not None:
        ax.axhline(y=baseline, color='gray', linestyle='--', 
                   linewidth=1, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title('Precision-Recall Curve Comparison')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_calibration_comparison(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Calibration Curve 비교 시각화
    
    Args:
        models_data: {model_name: (prob_true, prob_pred, brier)}
        save_path: 저장 경로
        figsize: Figure 크기
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (prob_true, prob_pred, brier) in models_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        
        ax.plot(prob_pred, prob_true,
                marker='o',
                label=f'{display_name} (Brier = {brier:.3f})',
                color=color,
                linewidth=2,
                markersize=6)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect Calibration')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve Comparison')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_combined_comparison(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    cal_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    baseline: float = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 5)
) -> plt.Figure:
    """
    ROC, PR, Calibration을 하나의 Figure로 결합
    
    Args:
        roc_data: ROC curve 데이터
        pr_data: PR curve 데이터
        cal_data: Calibration curve 데이터
        baseline: PR curve 베이스라인
        save_path: 저장 경로
        figsize: Figure 크기
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # (A) ROC Curve
    ax = axes[0]
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        linestyle = MODEL_LINESTYLES.get(model_name, '-')
        
        ax.plot(fpr, tpr,
                label=f'{display_name} ({auc_score:.3f})',
                color=color,
                linestyle=linestyle,
                linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('(A) ROC Curve')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # (B) PR Curve
    ax = axes[1]
    for model_name, (recall, precision, ap_score) in pr_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        linestyle = MODEL_LINESTYLES.get(model_name, '-')
        
        ax.plot(recall, precision,
                label=f'{display_name} ({ap_score:.3f})',
                color=color,
                linestyle=linestyle,
                linewidth=2)
    
    if baseline is not None:
        ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('(B) Precision-Recall Curve')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # (C) Calibration Curve
    ax = axes[2]
    for model_name, (prob_true, prob_pred, brier) in cal_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        
        ax.plot(prob_pred, prob_true,
                marker='o',
                label=f'{display_name} ({brier:.3f})',
                color=color,
                linewidth=2,
                markersize=5)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('(C) Calibration Curve')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def _detect_model_type(model: Any) -> str:
    """모델 타입 자동 감지"""
    model_class_name = type(model).__name__.lower()
    
    tree_models = ['xgbclassifier', 'lgbmclassifier', 'randomforestclassifier', 
                   'decisiontreeclassifier', 'catboostclassifier', 'booster']
    
    for tree_model in tree_models:
        if tree_model in model_class_name:
            return 'tree'
    
    if 'mlp' in model_class_name or 'neural' in model_class_name:
        return 'kernel'
    
    return 'kernel'


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    y: np.ndarray = None,
    max_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    SHAP 값 계산
    
    Args:
        model: 학습된 모델
        X: 입력 데이터
        feature_names: 특성 이름 리스트
        y: 타겟 변수 (stratified sampling용, None이면 랜덤 샘플링)
        max_samples: 최대 샘플 수 (기본: 1000)
    
    Returns:
        (shap_values, X_sample, feature_names_filtered)
    """
    if not HAS_SHAP:
        raise ImportError("SHAP가 설치되어 있지 않습니다.")
    
    # Stratified sampling (outcome 비율 유지)
    if len(X) > max_samples:
        if y is not None:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=1004)
            indices, _ = next(sss.split(X, y))
            X_sample = X[indices]
            print(f"      Stratified sampling: {len(X)} -> {len(X_sample)} (ratio: {y[indices].mean():.3f})")
        else:
            np.random.seed(1004)
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            print(f"      Random sampling: {len(X)} -> {len(X_sample)}")
    else:
        X_sample = X
    
    # KernelExplainer는 느리므로 샘플 수 제한
    model_type = _detect_model_type(model)
    kernel_max = 500
    if model_type == 'kernel' and len(X_sample) > kernel_max:
        np.random.seed(1004)
        k_indices = np.random.choice(len(X_sample), kernel_max, replace=False)
        X_sample = X_sample[k_indices]
        print(f"      KernelExplainer용 추가 샘플링: -> {len(X_sample)} samples")
    
    # Missing indicator 제외
    non_missing_mask = [not name.endswith(MISSING_INDICATOR_SUFFIX) for name in feature_names]
    non_missing_indices = [i for i, mask in enumerate(non_missing_mask) if mask]
    feature_names_filtered = [feature_names[i] for i in non_missing_indices]
    X_filtered = X_sample[:, non_missing_indices]
    
    # Explainer 선택 (shap 0.32 호환)
    def _make_kernel_explainer(bg_data):
        def predict_fn(x):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(x)[:, 1]
            return model.predict(x)
        return shap.KernelExplainer(predict_fn, bg_data)
    
    try:
        if model_type == 'tree':
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                print(f"      TreeExplainer 사용")
            except Exception as te:
                print(f"      TreeExplainer failed, using KernelExplainer: {te}")
                bg_size = min(50, len(X_sample))
                bg_indices = np.random.choice(len(X_sample), bg_size, replace=False)
                explainer = _make_kernel_explainer(X_sample[bg_indices])
                shap_values = explainer.shap_values(X_sample, nsamples=100)
        else:
            print(f"      KernelExplainer 사용 ({len(X_sample)} samples)...")
            bg_size = min(50, len(X_sample))
            bg_indices = np.random.choice(len(X_sample), bg_size, replace=False)
            explainer = _make_kernel_explainer(X_sample[bg_indices])
            shap_values = explainer.shap_values(X_sample, nsamples=200)
    except Exception as e:
        print(f"      ⚠️ SHAP 계산 오류: {e}")
        return None, None, None
    
    # SHAP values 처리 (2D로 변환)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    # Missing indicator 제외
    shap_values_filtered = shap_values[:, non_missing_indices]
    
    return shap_values_filtered, X_filtered, feature_names_filtered


def plot_shap_comparison(
    models_shap_data: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    save_path: str = None,
    top_n: int = 15,
    figsize: Tuple[int, int] = None
) -> plt.Figure:
    """
    여러 모델의 SHAP Beeswarm Plot을 subplot으로 비교 (SHAP 기본 시각화 사용)
    
    Args:
        models_shap_data: {model_name: (shap_values, X, feature_names)}
        save_path: 저장 경로
        top_n: 상위 N개 feature만 표시
        figsize: Figure 크기 (None이면 자동 계산)
    """
    if not HAS_SHAP:
        print("⚠️ SHAP가 설치되어 있지 않습니다.")
        return None
    
    n_models = len(models_shap_data)
    if n_models == 0:
        print("⚠️ SHAP 데이터가 없습니다.")
        return None
    
    # (2, 3) 고정 레이아웃, 각 subplot 1:1 비율 (3:2 전체 비율)
    n_rows = 2
    n_cols = 3
    if figsize is None:
        figsize = (24, 16)  # 3:2 비율 → 각 셀 (8, 8) ≈ 1:1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # 알파벳 레이블
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for idx, (model_name, (shap_values, X, feature_names)) in enumerate(models_shap_data.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        # summary_plot 호출 전 기존 axes 기록
        existing_axes = set(fig.get_axes())
        
        # 현재 axes를 활성화
        plt.sca(ax)
        
        # SHAP 기본 summary_plot 사용
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            max_display=top_n,
            show=False,
            plot_size=None  # subplot 크기 사용
        )
        
        # shap이 자동 생성한 colorbar axes 제거 (나중에 공유 colorbar 추가)
        new_axes = set(fig.get_axes()) - existing_axes - {ax}
        for cb_ax in new_axes:
            cb_ax.remove()
        
        # 제목 추가
        ax.set_title(f'({labels[idx]}) {display_name}', fontsize=12, fontweight='bold')
        # 1:1 비율 적용
        try:
            ax.set_box_aspect(1)
        except AttributeError:
            pass  # matplotlib < 3.6 호환
        
    # 빈 subplot 숨기기
    for idx in range(len(models_shap_data), len(axes)):
        axes[idx].set_visible(False)
    
    # 공유 Feature Value colorbar 추가
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # SHAP 기본 colormap 가져오기
    try:
        from shap.plots.colors import red_blue as shap_cmap
    except ImportError:
        try:
            from shap import plots
            shap_cmap = plots.colors.red_blue
        except (ImportError, AttributeError):
            # SHAP 0.32 호환: 기본 blue-red colormap 생성
            shap_cmap = plt.cm.get_cmap('bwr')
    
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=shap_cmap, norm=norm)
    sm.set_array([])
    
    plt.tight_layout(rect=[0, 0, 0.92, 1], pad=1.5)
    
    # 오른쪽 여백에 colorbar 배치
    cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=11)
    cbar.set_label('Feature Value', fontsize=13, labelpad=8)
    cbar.ax.tick_params(length=0)
    
    if save_path:
        _save_figure(fig, save_path, pad_inches=0.3)
    
    return fig


def create_comparison_figures(
    models_dir: str,
    data_dir: str,
    output_dir: str
) -> None:
    """
    모든 비교 Figure 생성
    
    Args:
        models_dir: 모델 디렉토리
        data_dir: 데이터 디렉토리
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Feature names 로드
    feature_names_path = os.path.join(data_dir, 'feature_names.txt')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    print(f"\n📊 모델 비교 Figure 생성")
    print(f"   테스트 데이터: {X_test.shape[0]}개 샘플")
    print(f"   양성 클래스 비율: {y_test.mean():.3f}")
    
    # 모델 찾기
    models = find_models(models_dir)
    print(f"   발견된 모델: {len(models)}개")
    
    if len(models) == 0:
        print("⚠️ 모델을 찾을 수 없습니다.")
        return
    
    # 각 모델에 대해 데이터 수집
    roc_data = {}
    pr_data = {}
    cal_data = {}
    shap_data = {}
    loaded_models = {}
    
    for model_name, model_path in models.items():
        print(f"\n   📈 {MODEL_DISPLAY_NAMES.get(model_name, model_name)} 처리 중...")
        
        try:
            model = load_model(model_path)
            if model is None:
                print(f"      ⚠️ 모델 로드 실패 - 건너뜀")
                continue
            loaded_models[model_name] = model
            y_prob = get_predictions(model, X_test)
            
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            roc_data[model_name] = (fpr, tpr, auc_score)
            
            # PR
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap_score = average_precision_score(y_test, y_prob)
            pr_data[model_name] = (recall, precision, ap_score)
            
            # Calibration
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
            brier = brier_score_loss(y_test, y_prob)
            cal_data[model_name] = (prob_true, prob_pred, brier)
            
            print(f"      AUROC: {auc_score:.3f}, AUPRC: {ap_score:.3f}, Brier: {brier:.3f}")
            
        except Exception as e:
            print(f"      ⚠️ 오류: {e}")
            continue
    
    # Figure 생성
    baseline = y_test.mean()
    
    # 개별 Figure
    print(f"\n📊 개별 Figure 생성...")
    plot_roc_comparison(roc_data, 
                        save_path=os.path.join(output_dir, 'comparison_roc.png'))
    plot_pr_comparison(pr_data, baseline=baseline,
                       save_path=os.path.join(output_dir, 'comparison_pr.png'))
    plot_calibration_comparison(cal_data,
                                save_path=os.path.join(output_dir, 'comparison_calibration.png'))
    
    # 결합 Figure (논문용) - png, tiff, pdf 3종 자동 저장
    print(f"\n📊 논문용 결합 Figure 생성...")
    plot_combined_comparison(roc_data, pr_data, cal_data, baseline=baseline,
                             save_path=os.path.join(output_dir, 'comparison_combined.png'))
    
    # SHAP Comparison Figure 생성
    if HAS_SHAP:
        print(f"\n📊 SHAP Comparison Figure 생성 중...")
        
        for model_name, model in loaded_models.items():
            print(f"   📈 {MODEL_DISPLAY_NAMES.get(model_name, model_name)} SHAP 계산 중...")
            try:
                result = compute_shap_values(model, X_test, feature_names, y=y_test, max_samples=1000)
                if result[0] is not None:
                    shap_data[model_name] = result
            except Exception as e:
                print(f"      ⚠️ SHAP 오류: {e}")
        
        if len(shap_data) > 0:
            # SHAP Comparison Figure - png, tiff, pdf 3종 자동 저장
            plot_shap_comparison(shap_data,
                                save_path=os.path.join(output_dir, 'comparison_shap.png'))
    else:
        print("\n⚠️ SHAP가 설치되어 있지 않아 SHAP Comparison Figure를 건너뜁니다.")
    
    print(f"\n✅ 모든 비교 Figure 생성 완료!")
    print(f"   저장 위치: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='모델 비교 시각화')
    parser.add_argument('--models-dir', type=str, default='../models',
                        help='모델 디렉토리')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='전처리된 데이터 디렉토리')
    parser.add_argument('--output', type=str, default='../results/comparison',
                        help='출력 디렉토리')
    
    args = parser.parse_args()
    
    create_comparison_figures(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
