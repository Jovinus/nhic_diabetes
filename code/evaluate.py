"""
모델 평가 및 해석 스크립트
- AUROC, AUPRC, 정확도, 민감도, 특이도 등
- SHAP 분석
- Calibration 분석
- Feature Importance
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Missing Indicator 접미사 (preprocessing.py와 동일)
MISSING_INDICATOR_SUFFIX = '_missing'


def convert_to_serializable(obj):
    """numpy/pandas 타입을 JSON 직렬화 가능한 Python 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# 평가 지표
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# 시각화
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (macOS, Windows, Linux 대응)
def set_korean_font():
    """한글 폰트 설정"""
    import platform
    import subprocess
    system = platform.system()
    
    font_set = False
    
    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'AppleGothic'
            font_set = True
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
        font_set = True
    else:  # Linux
        # NanumGothic 폰트 경로 확인
        nanum_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/nanum/NanumGothic.ttf',
            os.path.expanduser('~/.fonts/NanumGothic.ttf')
        ]
        for font_path in nanum_paths:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'NanumGothic'
                font_set = True
                break
        
        # DejaVu Sans로 fallback (한글 미지원, 하지만 에러 방지)
        if not font_set:
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            fallback_fonts = ['DejaVu Sans', 'Liberation Sans', 'FreeSans', 'sans-serif']
            for font in fallback_fonts:
                if font in available_fonts or font == 'sans-serif':
                    plt.rcParams['font.family'] = font
                    font_set = True
                    break
    
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# numpy 호환성 패치 (shap 0.32 + numpy>=1.24)
# shap 0.32가 내부적으로 np.int, np.float, np.bool 사용
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
    print("⚠️ SHAP 미설치 - SHAP 분석 불가")


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


def _save_figure(fig, save_path, dpi=500, bbox_inches='tight', pad_inches=0.1):
    """Figure를 png, tiff, pdf 3종으로 저장"""
    import os
    base, _ = os.path.splitext(save_path)
    for fmt in ['png', 'tiff', 'pdf']:
        out = f"{base}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, format=fmt)
    print(f"✅ Figure 저장: {base}.{{png,tiff,pdf}}")


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str] = None,
        model_name: str = 'model'
    ):
        """
        Args:
            model: 학습된 모델
            feature_names: 특성 이름 리스트
            model_name: 모델 이름
        """
        # 모델 표시 이름 매핑
        display_names = {
            'logistic_regression': 'LR',
            'decision_tree': 'DT',
            'random_forest': 'RF',
            'xgboost': 'XGB',
            'lightgbm': 'LGBM',
            'ann': 'MLP',
        }
        self.model = model
        self.feature_names = feature_names
        self.model_name = display_names.get(model_name, model_name)
        self.results = {}
        self.optimal_threshold = None
        self.youden_index = None
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """예측 확률 반환"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def find_optimal_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Youden Index 기반 최적 threshold 찾기
        
        Args:
            X: 특성 배열
            y: 타겟 배열
            
        Returns:
            최적 threshold
        """
        y_prob = self.predict_proba(X)
        self.optimal_threshold, self.youden_index = find_optimal_threshold_youden(y, y_prob)
        return self.optimal_threshold
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = None,
        use_youden: bool = True
    ) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            X: 특성 배열
            y: 타겟 배열
            threshold: 분류 임계값 (None이면 Youden Index 사용)
            use_youden: threshold가 None일 때 Youden Index 사용 여부
            
        Returns:
            평가 지표 딕셔너리
        """
        # 예측
        y_prob = self.predict_proba(X)
        
        # Threshold 결정
        if threshold is None:
            if use_youden:
                threshold = self.find_optimal_threshold(X, y)
            else:
                threshold = 0.5
        
        y_pred = (y_prob >= threshold).astype(int)
        
        # 평가 지표 계산
        results = {
            'auroc': roc_auc_score(y, y_prob),
            'auprc': average_precision_score(y, y_prob),
            'threshold': threshold,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),  # Sensitivity
            'f1': f1_score(y, y_pred, zero_division=0),
            'brier_score': brier_score_loss(y, y_prob)
        }
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        results['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        if self.youden_index is not None:
            results['youden_index'] = self.youden_index
        
        # 저장
        self.results = results
        
        return results
    
    def print_results(self) -> None:
        """결과 출력"""
        print(f"\n📊 {self.model_name} 평가 결과")
        print("=" * 50)
        print(f"  AUROC (C-statistic): {self.results['auroc']:.4f}")
        print(f"  AUPRC:               {self.results['auprc']:.4f}")
        if 'threshold' in self.results:
            print(f"  Threshold (Youden):  {self.results['threshold']:.4f}")
        if 'youden_index' in self.results:
            print(f"  Youden Index:        {self.results['youden_index']:.4f}")
        print("-" * 50)
        print(f"  Accuracy:            {self.results['accuracy']:.4f}")
        print(f"  Sensitivity (Recall):{self.results.get('sensitivity', self.results.get('recall', 0)):.4f}")
        print(f"  Specificity:         {self.results['specificity']:.4f}")
        print(f"  PPV (Precision):     {self.results['ppv']:.4f}")
        print(f"  NPV:                 {self.results['npv']:.4f}")
        print(f"  F1 Score:            {self.results['f1']:.4f}")
        print(f"  Brier Score:         {self.results['brier_score']:.4f}")
    
    def plot_roc_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: str = None,
        show_optimal_threshold: bool = True
    ) -> plt.Figure:
        """
        ROC Curve 시각화
        
        Args:
            X: 특성 배열
            y: 타겟 배열
            save_path: 저장 경로
            show_optimal_threshold: Youden Index 기반 최적점 표시 여부
        """
        y_prob = self.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        auroc = roc_auc_score(y, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, 'b-', lw=2, label=f'{self.model_name} (AUROC = {auroc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUROC = 0.500)')
        ax.fill_between(fpr, tpr, alpha=0.3)
        
        # 최적 threshold 지점 표시 (Youden Index)
        if show_optimal_threshold:
            youden = tpr - fpr
            optimal_idx = np.argmax(youden)
            optimal_fpr = fpr[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            optimal_threshold = thresholds[optimal_idx]
            
            ax.scatter([optimal_fpr], [optimal_tpr], marker='o', s=100, c='red', 
                      zorder=5, label=f'Optimal (Youden)\nThreshold = {optimal_threshold:.3f}')
            ax.annotate(f'Sens={optimal_tpr:.2f}\nSpec={1-optimal_fpr:.2f}',
                       xy=(optimal_fpr, optimal_tpr), xytext=(optimal_fpr + 0.1, optimal_tpr - 0.1),
                       fontsize=10, ha='left',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            _save_figure(fig, save_path)
        
        return fig
    
    def plot_pr_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: str = None
    ) -> plt.Figure:
        """Precision-Recall Curve 시각화"""
        y_prob = self.predict_proba(X)
        precision, recall, thresholds = precision_recall_curve(y, y_prob)
        auprc = average_precision_score(y, y_prob)
        
        # 기준선 (no skill)
        baseline = y.mean()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(recall, precision, 'b-', lw=2, label=f'{self.model_name} (AUPRC = {auprc:.3f})')
        ax.axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline (AUPRC = {baseline:.3f})')
        ax.fill_between(recall, precision, alpha=0.3)
        
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax.set_ylabel('Precision (PPV)', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            _save_figure(fig, save_path)
        
        return fig
    
    def plot_calibration_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        save_path: str = None
    ) -> plt.Figure:
        """Calibration Curve 시각화"""
        y_prob = self.predict_proba(X)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calibration plot
        ax1 = axes[0]
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly Calibrated')
        ax1.plot(mean_predicted_value, fraction_of_positives, 'b-o', lw=2, 
                label=f'{self.model_name}')
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curve', fontsize=14)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2 = axes[1]
        ax2.hist(y_prob, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Predicted Probabilities', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            _save_figure(fig, save_path)
        
        return fig
    
    def plot_confusion_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        save_path: str = None
    ) -> plt.Figure:
        """Confusion Matrix 시각화"""
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               ylabel='True Label',
               xlabel='Predicted Label',
               title=f'Confusion Matrix (threshold={threshold})')
        
        # 값 표시
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            _save_figure(fig, save_path)
        
        return fig
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Feature Importance 반환"""
        importance = None
        
        # CatBoost 모델 처리
        if hasattr(self.model, 'get_feature_importance'):
            try:
                importance = self.model.get_feature_importance()
            except Exception:
                pass
        
        # 일반적인 feature_importances_ 속성
        if importance is None and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        
        # 선형 모델의 coef_ 속성
        if importance is None and hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        
        if importance is None:
            print("⚠️ 해당 모델은 feature importance를 지원하지 않습니다.")
            return None
        
        # numpy array로 변환
        importance = np.array(importance, dtype=float)
        
        # None 또는 NaN 값을 0으로 대체
        importance = np.nan_to_num(importance, nan=0.0)
        
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: str = None
    ) -> plt.Figure:
        """Feature Importance 시각화"""
        df = self.get_feature_importance()
        if df is None:
            return None
        
        df_top = df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(df_top))
        ax.barh(y_pos, df_top['importance'].values, align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_top['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance (Top {top_n})', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            _save_figure(fig, save_path)
        
        return fig


class SHAPAnalyzer:
    """SHAP 분석 클래스"""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str] = None,
        model_type: str = 'tree',
        exclude_missing_indicator: bool = True
    ):
        """
        Args:
            model: 학습된 모델
            feature_names: 특성 이름
            model_type: 모델 타입 ('tree', 'linear', 'kernel')
            exclude_missing_indicator: SHAP 시각화에서 missing indicator 제외 여부
        """
        if not HAS_SHAP:
            raise ImportError("SHAP가 설치되어 있지 않습니다.")
        
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.exclude_missing_indicator = exclude_missing_indicator
        self.explainer = None
        self.shap_values = None
        self.expected_value = None  # base value 저장
    
    def _get_non_missing_indices(self) -> List[int]:
        """Missing indicator가 아닌 특성의 인덱스 반환"""
        if self.feature_names is None:
            return list(range(self.shap_values.shape[1]))
        
        return [
            i for i, name in enumerate(self.feature_names)
            if not name.endswith(MISSING_INDICATOR_SUFFIX)
        ]
    
    def _filter_missing_indicators(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Missing indicator 특성 제외
        
        Args:
            shap_values: SHAP values
            X: 특성 데이터
            feature_names: 특성 이름
            
        Returns:
            (filtered_shap_values, filtered_X, filtered_feature_names)
        """
        if not self.exclude_missing_indicator or feature_names is None:
            return shap_values, X, feature_names
        
        indices = self._get_non_missing_indices()
        
        if len(indices) == len(feature_names):
            # Missing indicator가 없음
            return shap_values, X, feature_names
        
        filtered_shap = shap_values[:, indices]
        filtered_X = X[:, indices]
        filtered_names = [feature_names[i] for i in indices]
        
        return filtered_shap, filtered_X, filtered_names
    
    def _extract_shap_values_for_positive_class(self, shap_values: Any) -> np.ndarray:
        """
        Binary classification에서 positive class의 SHAP values 추출
        
        다양한 형태의 SHAP output 처리:
        - list of arrays: [shap_class_0, shap_class_1] -> shap_class_1 선택
        - 3D array: (n_samples, n_features, n_classes) -> [:, :, 1] 선택
        - 2D array: (n_samples, n_features) -> 그대로 사용
        """
        # Case 1: List of arrays (older SHAP versions, some models)
        if isinstance(shap_values, list):
            print(f"   SHAP values 형태: list (길이={len(shap_values)})")
            if len(shap_values) == 2:
                return shap_values[1]  # positive class
            else:
                return shap_values[0]
        
        # Case 2: numpy array
        if isinstance(shap_values, np.ndarray):
            print(f"   SHAP values 형태: ndarray, shape={shap_values.shape}")
            
            # 3D array: (n_samples, n_features, n_classes)
            if shap_values.ndim == 3:
                if shap_values.shape[2] == 2:
                    return shap_values[:, :, 1]  # positive class
                else:
                    return shap_values[:, :, 0]
            
            # 2D array: (n_samples, n_features) - 이미 올바른 형태
            elif shap_values.ndim == 2:
                return shap_values
            
            # 1D array: 단일 샘플
            elif shap_values.ndim == 1:
                return shap_values.reshape(1, -1)
        
        # Case 3: shap.Explanation object (newer SHAP versions)
        if hasattr(shap_values, 'values'):
            return self._extract_shap_values_for_positive_class(shap_values.values)
        
        raise ValueError(f"알 수 없는 SHAP values 형태: {type(shap_values)}")
    
    def _extract_expected_value(self, expected_value: Any) -> float:
        """
        expected_value (base value) 추출
        
        다양한 형태 처리:
        - scalar: 그대로 사용
        - array: positive class 선택
        - list: positive class 선택
        """
        if expected_value is None:
            return 0.0
        
        # numpy array
        if isinstance(expected_value, np.ndarray):
            if expected_value.ndim == 0:
                return float(expected_value)
            elif len(expected_value) == 2:
                return float(expected_value[1])  # positive class
            else:
                return float(expected_value[0])
        
        # list
        if isinstance(expected_value, list):
            if len(expected_value) == 2:
                return float(expected_value[1])  # positive class
            else:
                return float(expected_value[0])
        
        # scalar
        return float(expected_value)
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        background_data: np.ndarray = None,
        max_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SHAP 값 계산
        
        Args:
            X: 설명할 데이터
            y: 타겟 변수 (stratified sampling용, None이면 랜덤 샘플링)
            background_data: 배경 데이터 (kernel SHAP용)
            max_samples: 최대 샘플 수 (기본: 1000)
            
        Returns:
            (SHAP 값 배열, 샘플 데이터)
        """
        print("\n🔍 SHAP 값 계산 중...")
        
        # Stratified sampling (outcome 비율 유지)
        if len(X) > max_samples:
            if y is not None:
                # Stratified sampling
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=1004)
                indices, _ = next(sss.split(X, y))
                X_sample = X[indices]
                print(f"   Stratified sampling: {len(X)} -> {len(X_sample)} samples")
                if y is not None:
                    original_ratio = y.mean()
                    sampled_ratio = y[indices].mean()
                    print(f"   Outcome ratio: original={original_ratio:.3f}, sampled={sampled_ratio:.3f}")
            else:
                # Random sampling (fallback)
                np.random.seed(1004)
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
                print(f"   Random sampling: {len(X)} -> {len(X_sample)} samples")
        else:
            X_sample = X.copy()
        
        # KernelExplainer는 느리므로 샘플 수 제한
        kernel_max = 500
        if self.model_type == 'kernel' and len(X_sample) > kernel_max:
            np.random.seed(1004)
            k_indices = np.random.choice(len(X_sample), kernel_max, replace=False)
            X_sample = X_sample[k_indices]
            print(f"   KernelExplainer용 추가 샘플링: -> {len(X_sample)} samples")
        
        print(f"   샘플 수: {len(X_sample)}")
        print(f"   모델 타입: {self.model_type}")
        
        # Explainer 생성 및 SHAP 값 계산
        # shap 0.32 호환: TreeExplainer 시도 후 실패 시 KernelExplainer 폴백
        
        def _make_kernel_explainer(bg_data):
            """KernelExplainer 생성 헬퍼"""
            def predict_proba_positive(x):
                proba = self.model.predict_proba(x)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return proba[:, 1]
                return proba
            return shap.KernelExplainer(predict_proba_positive, bg_data)
        
        try:
            if self.model_type == 'tree':
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    raw_shap_values = self.explainer.shap_values(X_sample)
                    print("   TreeExplainer 사용")
                except Exception as te:
                    print(f"   ⚠️ TreeExplainer 실패: {te}")
                    print("   KernelExplainer로 폴백...")
                    bg_size = min(50, len(X_sample))
                    bg_indices = np.random.choice(len(X_sample), bg_size, replace=False)
                    self.explainer = _make_kernel_explainer(X_sample[bg_indices])
                    raw_shap_values = self.explainer.shap_values(X_sample, nsamples=100)
                
            elif self.model_type == 'linear':
                try:
                    self.explainer = shap.LinearExplainer(self.model, X_sample)
                    raw_shap_values = self.explainer.shap_values(X_sample)
                except Exception as le:
                    print(f"   ⚠️ LinearExplainer 실패: {le}")
                    bg_size = min(50, len(X_sample))
                    bg_indices = np.random.choice(len(X_sample), bg_size, replace=False)
                    self.explainer = _make_kernel_explainer(X_sample[bg_indices])
                    raw_shap_values = self.explainer.shap_values(X_sample, nsamples=100)
                
            else:  # kernel (ANN, 기타 모델)
                print("   KernelExplainer 사용...")
                if background_data is None:
                    bg_size = min(50, len(X_sample))
                    bg_indices = np.random.choice(len(X_sample), bg_size, replace=False)
                    background_data = X_sample[bg_indices]
                
                self.explainer = _make_kernel_explainer(background_data)
                raw_shap_values = self.explainer.shap_values(X_sample, nsamples=200)
                
        except Exception as e:
            print(f"   ⚠️ {self.model_type} explainer 실패, KernelExplainer로 폴백: {e}")
            bg_size = min(50, len(X_sample))
            bg_indices = np.random.choice(len(X_sample), bg_size, replace=False)
            self.explainer = _make_kernel_explainer(X_sample[bg_indices])
            raw_shap_values = self.explainer.shap_values(X_sample, nsamples=100)
        
        # SHAP values를 2D 배열로 변환 (positive class)
        self.shap_values = self._extract_shap_values_for_positive_class(raw_shap_values)
        
        # Expected value 추출
        if hasattr(self.explainer, 'expected_value'):
            self.expected_value = self._extract_expected_value(self.explainer.expected_value)
        else:
            self.expected_value = 0.0
        
        print(f"✅ SHAP 값 계산 완료:")
        print(f"   shape: {self.shap_values.shape}")
        print(f"   expected_value (base): {self.expected_value:.4f}")
        
        return self.shap_values, X_sample
    
    def plot_summary(
        self,
        X: np.ndarray,
        save_path: str = None,
        max_display: int = 20
    ) -> plt.Figure:
        """
        SHAP Summary Plot (Beeswarm Plot)
        
        Note: exclude_missing_indicator=True인 경우 Missing indicator 특성 제외
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Missing indicator 제외 처리
        shap_vals, X_plot, feat_names = self._filter_missing_indicators(
            self.shap_values, X, self.feature_names
        )
        
        if self.exclude_missing_indicator and feat_names != self.feature_names:
            print(f"   📊 Missing indicator 제외: {len(self.feature_names)} -> {len(feat_names)} 특성")
        
        fig = plt.figure(figsize=(10, 10))
        shap.summary_plot(
            shap_vals, X_plot,
            feature_names=feat_names,
            max_display=max_display,
            show=False
        )
        
        # colorbar 크기 조정 - Feature Value bar가 잘 보이도록
        for cb_ax in fig.get_axes():
            # colorbar axes는 보통 매우 좁은 width를 가짐
            pos = cb_ax.get_position()
            if pos.width < 0.05 and pos.width < pos.height * 0.3:
                # colorbar axes로 판단 → 너비를 키우고 위치 조정
                cb_ax.set_position([pos.x0 + 0.02, pos.y0, 0.02, pos.height])
                cb_ax.tick_params(labelsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.92, 1], pad=1.0)
        
        if save_path:
            _save_figure(fig, save_path, pad_inches=0.3)
        
        return fig
    
    def plot_bar(
        self,
        save_path: str = None,
        max_display: int = 20
    ) -> plt.Figure:
        """
        SHAP Bar Plot (Mean absolute SHAP values)
        
        Note: exclude_missing_indicator=True인 경우 Missing indicator 특성 제외
        """
        if self.shap_values is None:
            raise ValueError("먼저 compute_shap_values()를 실행하세요.")
        
        # Missing indicator 제외 처리
        dummy_X = np.zeros_like(self.shap_values)
        shap_vals, _, feat_names = self._filter_missing_indicators(
            self.shap_values, dummy_X, self.feature_names
        )
        
        fig = plt.figure(figsize=(10, 10))
        
        # SHAP 기본 bar plot 사용
        shap.summary_plot(
            shap_vals,
            feature_names=feat_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        
        # 1:1 비율 맞추기
        fig.set_size_inches(10, 10)
        plt.tight_layout(pad=1.0)
        
        if save_path:
            _save_figure(fig, save_path, bbox_inches=None, pad_inches=0.3)
        
        return fig
    
    # Note: waterfall and dependence plots removed for shap 0.32 compatibility


def evaluate_model(
    model_path: str,
    data_dir: str = '../data/processed',
    output_dir: str = '../results',
    model_name: str = None
) -> Dict:
    """
    모델 평가 실행
    
    Args:
        model_path: 모델 파일 경로
        data_dir: 전처리된 데이터 디렉토리
        output_dir: 결과 저장 디렉토리
        model_name: 모델 이름
        
    Returns:
        평가 결과
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 이름 추출
    if model_name is None:
        model_name = os.path.basename(model_path).replace('_best_model', '').replace('_model', '')
        model_name = model_name.replace('.pkl', '').replace('.json', '').replace('.txt', '').replace('.cbm', '')
    
    print("=" * 60)
    print(f"모델 평가: {model_name}")
    print("=" * 60)
    
    # 모델 로드 (모든 모델 pkl로 통일)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 데이터 로드
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    print(f"📂 테스트 데이터 로드: {X_test.shape}")
    
    # 평가
    evaluator = ModelEvaluator(model, feature_names, model_name)
    results = evaluator.evaluate(X_test, y_test)
    evaluator.print_results()
    
    # 시각화
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    evaluator.plot_roc_curve(X_test, y_test, 
                            save_path=os.path.join(model_output_dir, 'roc_curve.png'))
    evaluator.plot_pr_curve(X_test, y_test,
                           save_path=os.path.join(model_output_dir, 'pr_curve.png'))
    evaluator.plot_calibration_curve(X_test, y_test,
                                    save_path=os.path.join(model_output_dir, 'calibration_curve.png'))
    evaluator.plot_confusion_matrix(X_test, y_test,
                                   save_path=os.path.join(model_output_dir, 'confusion_matrix.png'))
    evaluator.plot_feature_importance(save_path=os.path.join(model_output_dir, 'feature_importance.png'))
    
    # 결과 저장 (numpy 타입을 Python 타입으로 변환)
    with open(os.path.join(model_output_dir, 'metrics.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    return results


def _detect_model_type(model: Any) -> str:
    """
    모델 타입 자동 감지
    
    Returns:
        'tree', 'linear', 'kernel' 중 하나
    """
    model_class_name = type(model).__name__.lower()
    
    # Tree-based models
    tree_models = [
        'xgbclassifier', 'xgbregressor', 'xgboost',
        'lgbmclassifier', 'lgbmregressor', 'lightgbm', 'booster',
        'catboostclassifier', 'catboostregressor', 'catboost',
        'randomforestclassifier', 'randomforestregressor',
        'gradientboostingclassifier', 'gradientboostingregressor',
        'decisiontreeclassifier', 'decisiontreeregressor',
        'extratreesclassifier', 'extratreesregressor'
    ]
    
    # Linear models
    linear_models = [
        'logisticregression', 'linearregression',
        'ridge', 'lasso', 'elasticnet',
        'sgdclassifier', 'sgdregressor'
    ]
    
    for tree_model in tree_models:
        if tree_model in model_class_name:
            return 'tree'
    
    for linear_model in linear_models:
        if linear_model in model_class_name:
            return 'linear'
    
    # Default to kernel (for ANN, SVM, etc.)
    return 'kernel'


def run_shap_analysis(
    model_path: str,
    data_dir: str = '../data/processed',
    output_dir: str = '../results',
    model_name: str = None,
    model_type: str = None,  # None이면 자동 감지
    exclude_missing_indicator: bool = True
) -> None:
    """
    SHAP 분석 실행
    
    Args:
        model_path: 모델 파일 경로
        data_dir: 전처리된 데이터 디렉토리
        output_dir: 결과 저장 디렉토리
        model_name: 모델 이름
        model_type: 모델 타입 ('tree', 'linear', 'kernel'), None이면 자동 감지
        exclude_missing_indicator: SHAP 시각화에서 missing indicator 제외 여부
    """
    if not HAS_SHAP:
        print("⚠️ SHAP가 설치되어 있지 않습니다.")
        return
    
    # 모델 이름 추출
    if model_name is None:
        model_name = os.path.basename(model_path).replace('_best_model', '').replace('_model', '')
        model_name = model_name.replace('.pkl', '').replace('.json', '').replace('.txt', '').replace('.cbm', '')
    
    print("\n" + "=" * 60)
    print(f"SHAP 분석: {model_name}")
    print("=" * 60)
    
    # 모델 로드 (모든 모델 pkl로 통일)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 모델 타입 자동 감지
    if model_type is None:
        model_type = _detect_model_type(model)
        print(f"📊 모델 타입 자동 감지: {model_type}")
    
    # 데이터 로드
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    # SHAP 분석
    print(f"📊 Missing indicator 제외: {exclude_missing_indicator}")
    analyzer = SHAPAnalyzer(
        model, feature_names, model_type,
        exclude_missing_indicator=exclude_missing_indicator
    )
    shap_values, X_sample = analyzer.compute_shap_values(X_test, y=y_test, max_samples=1000)
    
    # 시각화
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    analyzer.plot_summary(X_sample, save_path=os.path.join(model_output_dir, 'shap_summary.png'))
    analyzer.plot_bar(save_path=os.path.join(model_output_dir, 'shap_bar.png'))
    
    print(f"\n✅ SHAP 분석 완료: {model_output_dir}/")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='당뇨병 예측 모델 평가')
    parser.add_argument('--model', type=str, required=True,
                        help='모델 파일 경로')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='전처리된 데이터 디렉토리')
    parser.add_argument('--output', type=str, default='../results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--shap', action='store_true',
                        help='SHAP 분석 실행')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['tree', 'linear', 'kernel'],
                        help='SHAP explainer 타입 (기본: 자동 감지)')
    parser.add_argument('--include-missing-indicator', action='store_true',
                        help='SHAP 시각화에 missing indicator 특성 포함 (기본: 제외)')
    
    args = parser.parse_args()
    
    # 모델 평가
    results = evaluate_model(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output
    )
    
    # SHAP 분석
    if args.shap:
        run_shap_analysis(
            model_path=args.model,
            data_dir=args.data_dir,
            output_dir=args.output,
            model_type=args.model_type,
            exclude_missing_indicator=not args.include_missing_indicator
        )
    
    print("\n✅ 평가 완료!")


if __name__ == '__main__':
    main()
