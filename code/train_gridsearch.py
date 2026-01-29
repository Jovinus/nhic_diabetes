"""
GridSearchCV를 사용한 모델 학습 스크립트
- Decision Tree, Random Forest, XGBoost, CatBoost, ANN(MLP)
- Train/Test 분할 후 Train으로 GridSearchCV, Test로 최종 평가
- 최적 모델 저장 (SHAP 분석용)
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost 미설치")

# CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️ CatBoost 미설치")

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 하이퍼파라미터 그리드 정의
# =============================================================================

PARAM_GRIDS = {
    'decision_tree': {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    },
    
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'class_weight': ['balanced', None]
    },
    
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    },
    
    'catboost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7]
    },
    
    'ann': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    }
}

# 빠른 테스트용 축소 그리드
PARAM_GRIDS_SMALL = {
    'decision_tree': {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
        'criterion': ['gini'],
        'class_weight': ['balanced']
    },
    
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    },
    
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 3]
    },
    
    'catboost': {
        'iterations': [100, 200],
        'depth': [6, 8],
        'learning_rate': [0.05, 0.1],
        'l2_leaf_reg': [3, 5]
    },
    
    'ann': {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu'],
        'alpha': [0.001],
        'learning_rate_init': [0.001],
        'max_iter': [300]
    }
}


class ModelTrainer:
    """GridSearchCV를 사용한 모델 학습 클래스"""
    
    def __init__(
        self,
        random_state: int = 1004,
        cv: int = 5,
        scoring: str = 'roc_auc',
        n_jobs: int = -1,
        use_small_grid: bool = False
    ):
        """
        Args:
            random_state: 랜덤 시드
            cv: Cross-validation fold 수
            scoring: 평가 지표
            n_jobs: 병렬 처리 수
            use_small_grid: 축소된 파라미터 그리드 사용 여부
        """
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.param_grids = PARAM_GRIDS_SMALL if use_small_grid else PARAM_GRIDS
        
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        self.test_results = {}
        
    def _get_base_model(self, model_name: str) -> Any:
        """기본 모델 인스턴스 반환"""
        if model_name == 'decision_tree':
            return DecisionTreeClassifier(random_state=self.random_state)
        
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        
        elif model_name == 'xgboost':
            if not HAS_XGB:
                raise ImportError("XGBoost가 설치되어 있지 않습니다.")
            return xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        
        elif model_name == 'catboost':
            if not HAS_CATBOOST:
                raise ImportError("CatBoost가 설치되어 있지 않습니다.")
            return CatBoostClassifier(
                random_state=self.random_state,
                verbose=0,
                thread_count=self.n_jobs if self.n_jobs > 0 else -1
            )
        
        elif model_name == 'ann':
            return MLPClassifier(
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: int = 1
    ) -> Tuple[Any, Dict]:
        """
        단일 모델 GridSearchCV 학습
        
        Args:
            model_name: 모델 이름
            X_train: 훈련 특성
            y_train: 훈련 타겟
            verbose: 출력 상세도
            
        Returns:
            (best_model, cv_results)
        """
        print(f"\n{'='*60}")
        print(f"🎯 {model_name.upper()} 학습 시작")
        print(f"{'='*60}")
        
        # 모델과 파라미터 그리드
        base_model = self._get_base_model(model_name)
        param_grid = self.param_grids[model_name]
        
        # 파라미터 조합 수 계산
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        print(f"📊 파라미터 조합 수: {n_combinations}")
        print(f"📊 총 fit 횟수: {n_combinations * self.cv}")
        
        # GridSearchCV
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv_splitter,
            n_jobs=self.n_jobs,
            verbose=verbose,
            refit=True,
            return_train_score=True
        )
        
        print(f"\n🔍 GridSearchCV 진행 중...")
        start_time = datetime.now()
        
        grid_search.fit(X_train, y_train)
        
        elapsed = datetime.now() - start_time
        print(f"⏱️  소요 시간: {elapsed}")
        
        # 결과 저장
        self.models[model_name] = grid_search.best_estimator_
        self.best_params[model_name] = grid_search.best_params_
        self.cv_results[model_name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': grid_search.cv_results_['mean_train_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']]
            }
        }
        
        print(f"\n✅ 최적 파라미터: {grid_search.best_params_}")
        print(f"✅ 최적 CV 점수 ({self.scoring}): {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, self.cv_results[model_name]
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        테스트 데이터로 모델 평가
        
        Args:
            model_name: 모델 이름
            X_test: 테스트 특성
            y_test: 테스트 타겟
            
        Returns:
            평가 지표 딕셔너리
        """
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 학습되지 않았습니다.")
        
        model = self.models[model_name]
        
        # 예측
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 평가 지표
        results = {
            'auroc': roc_auc_score(y_test, y_prob),
            'auprc': average_precision_score(y_test, y_prob),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        self.test_results[model_name] = results
        
        return results
    
    def save_model(
        self,
        model_name: str,
        output_dir: str,
        feature_names: List[str] = None
    ) -> str:
        """
        학습된 모델 저장
        
        Args:
            model_name: 모델 이름
            output_dir: 저장 디렉토리
            feature_names: 특성 이름 리스트
            
        Returns:
            저장된 파일 경로
        """
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 학습되지 않았습니다.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        model = self.models[model_name]
        
        # 모델 타입에 따른 저장
        if model_name == 'xgboost':
            model_path = os.path.join(output_dir, f'{model_name}_best_model.json')
            model.save_model(model_path)
        elif model_name == 'catboost':
            model_path = os.path.join(output_dir, f'{model_name}_best_model.cbm')
            model.save_model(model_path)
        else:
            model_path = os.path.join(output_dir, f'{model_name}_best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"✅ 모델 저장: {model_path}")
        
        # 메타데이터 저장
        meta = {
            'model_name': model_name,
            'best_params': self.best_params.get(model_name, {}),
            'cv_score': self.cv_results.get(model_name, {}).get('best_score'),
            'test_results': self.test_results.get(model_name, {}),
            'feature_names': feature_names,
            'train_date': datetime.now().isoformat(),
            'scoring': self.scoring,
            'cv_folds': self.cv
        }
        
        meta_path = os.path.join(output_dir, f'{model_name}_best_model_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"✅ 메타데이터 저장: {meta_path}")
        
        return model_path
    
    def save_all_models(
        self,
        output_dir: str,
        feature_names: List[str] = None
    ) -> Dict[str, str]:
        """모든 학습된 모델 저장"""
        paths = {}
        for model_name in self.models:
            paths[model_name] = self.save_model(model_name, output_dir, feature_names)
        return paths
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        """
        저장된 모델 로드
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            로드된 모델
        """
        if model_path.endswith('.json'):
            # XGBoost
            model = xgb.XGBClassifier()
            model.load_model(model_path)
        elif model_path.endswith('.cbm'):
            # CatBoost
            model = CatBoostClassifier()
            model.load_model(model_path)
        elif model_path.endswith('.pkl'):
            # Pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {model_path}")
        
        print(f"✅ 모델 로드: {model_path}")
        return model


def load_data(
    data_path: str,
    target_col: str = 'outA',
    test_size: float = 0.2,
    random_state: int = 1004
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    데이터 로드 및 Train/Test 분할
    
    Args:
        data_path: 전처리된 데이터 디렉토리 또는 CSV 파일 경로
        target_col: 타겟 변수
        test_size: 테스트 세트 비율
        random_state: 랜덤 시드
        
    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    # 전처리된 numpy 파일 로드
    if os.path.isdir(data_path):
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        # Validation 데이터가 있으면 train에 합치기
        X_val_path = os.path.join(data_path, 'X_val.npy')
        if os.path.exists(X_val_path):
            X_val = np.load(X_val_path)
            y_val = np.load(os.path.join(data_path, 'y_val.npy'))
            X_train = np.vstack([X_train, X_val])
            y_train = np.concatenate([y_train, y_val])
        
        # 특성 이름 로드
        feature_names_path = os.path.join(data_path, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = f.read().strip().split('\n')
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    else:
        raise ValueError(f"데이터 경로가 존재하지 않습니다: {data_path}")
    
    print(f"📂 데이터 로드 완료:")
    print(f"   - 훈련: {X_train.shape}, 양성: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"   - 테스트: {X_test.shape}, 양성: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, feature_names


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: str = '../models',
    model_list: List[str] = None,
    use_small_grid: bool = False,
    cv: int = 5,
    scoring: str = 'roc_auc',
    n_jobs: int = -1
) -> Tuple[ModelTrainer, pd.DataFrame]:
    """
    여러 모델 학습 및 비교
    
    Args:
        X_train, y_train: 훈련 데이터
        X_test, y_test: 테스트 데이터
        feature_names: 특성 이름
        output_dir: 모델 저장 디렉토리
        model_list: 학습할 모델 리스트
        use_small_grid: 축소 그리드 사용 여부
        cv: CV fold 수
        scoring: 평가 지표
        n_jobs: 병렬 처리 수
        
    Returns:
        (trainer, results_df)
    """
    # 기본 모델 리스트
    if model_list is None:
        model_list = ['decision_tree', 'random_forest']
        if HAS_XGB:
            model_list.append('xgboost')
        if HAS_CATBOOST:
            model_list.append('catboost')
        model_list.append('ann')
    
    print("=" * 60)
    print("🚀 GridSearchCV 모델 학습 시작")
    print("=" * 60)
    print(f"📊 학습 모델: {model_list}")
    print(f"📊 CV Folds: {cv}")
    print(f"📊 평가 지표: {scoring}")
    print(f"📊 파라미터 그리드: {'축소' if use_small_grid else '전체'}")
    
    # Trainer 생성
    trainer = ModelTrainer(
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        use_small_grid=use_small_grid
    )
    
    # 결과 저장
    results = []
    
    for model_name in model_list:
        try:
            # 학습
            trainer.train_model(model_name, X_train, y_train, verbose=1)
            
            # 평가
            test_metrics = trainer.evaluate_model(model_name, X_test, y_test)
            
            # 모델 저장
            trainer.save_model(model_name, output_dir, feature_names)
            
            # 결과 기록
            results.append({
                'model': model_name,
                'cv_score': trainer.cv_results[model_name]['best_score'],
                **test_metrics
            })
            
            print(f"\n📊 {model_name.upper()} 테스트 결과:")
            print(f"   AUROC: {test_metrics['auroc']:.4f}")
            print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Sensitivity: {test_metrics['recall']:.4f}")
            print(f"   Specificity: {test_metrics['specificity']:.4f}")
            
        except Exception as e:
            print(f"\n❌ {model_name} 학습 실패: {e}")
            continue
    
    # 결과 DataFrame
    results_df = pd.DataFrame(results)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 모델 비교 결과")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 최고 모델
    if len(results_df) > 0:
        best_model = results_df.loc[results_df['auroc'].idxmax(), 'model']
        best_auroc = results_df['auroc'].max()
        print(f"\n🏆 최고 모델: {best_model} (Test AUROC: {best_auroc:.4f})")
    
    # 결과 저장
    results_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ 결과 저장: {results_path}")
    
    return trainer, results_df


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GridSearchCV를 사용한 당뇨병 예측 모델 학습')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='전처리된 데이터 디렉토리')
    parser.add_argument('--output', type=str, default='../models',
                        help='모델 저장 디렉토리')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['decision_tree', 'random_forest', 'xgboost', 'catboost', 'ann'],
                        help='학습할 모델 리스트')
    parser.add_argument('--small-grid', action='store_true',
                        help='축소된 파라미터 그리드 사용 (빠른 테스트용)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Cross-validation fold 수')
    parser.add_argument('--scoring', type=str, default='roc_auc',
                        help='평가 지표')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='병렬 처리 수 (-1: 모든 코어)')
    
    args = parser.parse_args()
    
    # 데이터 로드
    X_train, X_test, y_train, y_test, feature_names = load_data(args.data_dir)
    
    # 모델 학습
    trainer, results_df = train_all_models(
        X_train, y_train, X_test, y_test, feature_names,
        output_dir=args.output,
        model_list=args.models,
        use_small_grid=args.small_grid,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs
    )
    
    print("\n✅ 모든 학습 완료!")
    print(f"\n저장된 모델 위치: {args.output}/")
    print("SHAP 분석 시 모델 로드 예시:")
    print("  model = ModelTrainer.load_model('models/xgboost_best_model.json')")


if __name__ == '__main__':
    main()
