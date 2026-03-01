"""
GridSearchCV를 사용한 모델 학습 스크립트
- Decision Tree, Random Forest, XGBoost, LightGBM, ANN(MLP)
- Train/Test 분할 후 Train으로 GridSearchCV, Test로 최종 평가
- 최적 모델 저장 (SHAP 분석용)

호환: Python 3.8, xgboost==0.80, scikit-learn==1.2.2
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# XGBoost (0.80)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost 미설치")

# LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM 미설치")

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 하이퍼파라미터 그리드 정의
# =============================================================================

PARAM_GRIDS = {
    'decision_tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'class_weight': ['balanced', None]
    },
    
    'random_forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
        'class_weight': ['balanced', None]
    },
    
    'xgboost': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 3]
    },
    
    'lightgbm': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 3]
    },
    
    'ann': {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    },

    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }
}

# 빠른 테스트용 축소 그리드
PARAM_GRIDS_SMALL = {
    'decision_tree': {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
        'class_weight': ['balanced']
    },
    
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
        'class_weight': ['balanced']
    },
    
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1]
    },
    
    'lightgbm': {
        'n_estimators': [100, 200],
        'max_depth': [5, -1],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1]
    },
    
    'ann': {
        'hidden_layer_sizes': [(128,), (64, 32)],
        'activation': ['relu'],
        'alpha': [0.001],
        'learning_rate_init': [0.001],
        'max_iter': [500]
    },

    'logistic_regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
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
        use_small_grid: bool = False,
        use_gpu: bool = False
    ):
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.param_grids = PARAM_GRIDS_SMALL if use_small_grid else PARAM_GRIDS
        self.use_gpu = use_gpu

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
            # xgboost 0.80 호환 - use_label_encoder, verbosity 등 없음
            xgb_params = dict(
                seed=self.random_state,
                nthread=self.n_jobs,
                silent=True,
                objective='binary:logistic'
            )
            if self.use_gpu:
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['gpu_id'] = 0
                print("  XGBoost: GPU 모드 활성화 (tree_method='gpu_hist')")
            return xgb.XGBClassifier(**xgb_params)

        elif model_name == 'lightgbm':
            if not HAS_LGB:
                raise ImportError("LightGBM이 설치되어 있지 않습니다.")
            if self.use_gpu:
                print("  LightGBM: GPU 빌드 미포함, CPU로 실행합니다.")
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
        
        elif model_name == 'ann':
            return MLPClassifier(
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )

        elif model_name == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state)

        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: int = 1
    ) -> Tuple[Any, Dict]:
        """단일 모델 GridSearchCV 학습"""
        print(f"\n{'='*60}")
        print(f"🎯 {model_name.upper()} 학습 시작")
        print(f"{'='*60}")
        
        base_model = self._get_base_model(model_name)
        param_grid = self.param_grids[model_name]
        
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        print(f"📊 파라미터 조합 수: {n_combinations}")
        print(f"📊 총 fit 횟수: {n_combinations * self.cv}")
        
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
            'best_score': float(grid_search.best_score_),
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
        """테스트 데이터로 모델 평가"""
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 학습되지 않았습니다.")
        
        model = self.models[model_name]
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results = {
            'auroc': roc_auc_score(y_test, y_prob),
            'auprc': average_precision_score(y_test, y_prob),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
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
        """학습된 모델 저장 (모든 모델을 pkl로 저장)"""
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 학습되지 않았습니다.")
        
        os.makedirs(output_dir, exist_ok=True)
        model = self.models[model_name]
        
        # 모든 모델을 pkl로 저장 (xgboost 0.80 호환)
        model_path = os.path.join(output_dir, '{}_best_model.pkl'.format(model_name))
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ 모델 저장: {model_path}")
        
        # 메타데이터 저장
        meta = {
            'model_name': model_name,
            'best_params': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                           for k, v in self.best_params.get(model_name, {}).items()},
            'cv_score': self.cv_results.get(model_name, {}).get('best_score'),
            'test_results': {k: float(v) for k, v in self.test_results.get(model_name, {}).items()},
            'feature_names': feature_names,
            'train_date': datetime.now().isoformat(),
            'scoring': self.scoring,
            'cv_folds': self.cv
        }
        
        meta_path = os.path.join(output_dir, '{}_best_model_meta.json'.format(model_name))
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
        """저장된 모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ 모델 로드: {model_path}")
            return model
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {model_path} - {e}")
            return None


def load_data(
    data_path: str,
    target_col: str = 'outA',
    test_size: float = 0.2,
    random_state: int = 1004
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데이터 로드 및 Train/Test 분할"""
    if os.path.isdir(data_path):
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        X_val_path = os.path.join(data_path, 'X_val.npy')
        if os.path.exists(X_val_path):
            X_val = np.load(X_val_path)
            y_val = np.load(os.path.join(data_path, 'y_val.npy'))
            X_train = np.vstack([X_train, X_val])
            y_train = np.concatenate([y_train, y_val])
        
        feature_names_path = os.path.join(data_path, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = f.read().strip().split('\n')
        else:
            feature_names = ['feature_{}'.format(i) for i in range(X_train.shape[1])]
    else:
        raise ValueError(f"데이터 경로가 존재하지 않습니다: {data_path}")
    
    print(f"📂 데이터 로드 완료:")
    print(f"   - 훈련: {X_train.shape}, 양성: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"   - 테스트: {X_test.shape}, 양성: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, feature_names


def train_all_models(
    X_train, y_train, X_test, y_test, feature_names,
    output_dir='../models', model_list=None,
    use_small_grid=False, cv=5, scoring='roc_auc', n_jobs=-1
):
    """여러 모델 학습 및 비교"""
    if model_list is None:
        model_list = ['decision_tree', 'random_forest']
        if HAS_XGB:
            model_list.append('xgboost')
        if HAS_LGB:
            model_list.append('lightgbm')
        model_list.append('ann')
        model_list.append('logistic_regression')
    
    print("=" * 60)
    print("🚀 GridSearchCV 모델 학습 시작")
    print("=" * 60)
    print(f"📊 학습 모델: {model_list}")
    print(f"📊 CV Folds: {cv}")
    print(f"📊 평가 지표: {scoring}")
    print(f"📊 파라미터 그리드: {'축소' if use_small_grid else '전체'}")
    
    trainer = ModelTrainer(
        cv=cv, scoring=scoring, n_jobs=n_jobs, use_small_grid=use_small_grid
    )
    
    results = []
    
    for model_name in model_list:
        try:
            trainer.train_model(model_name, X_train, y_train, verbose=1)
            test_metrics = trainer.evaluate_model(model_name, X_test, y_test)
            trainer.save_model(model_name, output_dir, feature_names)
            
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
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("📊 모델 비교 결과")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    if len(results_df) > 0:
        best_model = results_df.loc[results_df['auroc'].idxmax(), 'model']
        best_auroc = results_df['auroc'].max()
        print(f"\n🏆 최고 모델: {best_model} (Test AUROC: {best_auroc:.4f})")
    
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
                        default=['decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'ann', 'logistic_regression'],
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
    
    X_train, X_test, y_train, y_test, feature_names = load_data(args.data_dir)
    
    trainer, results_df = train_all_models(
        X_train, y_train, X_test, y_test, feature_names,
        output_dir=args.output, model_list=args.models,
        use_small_grid=args.small_grid, cv=args.cv,
        scoring=args.scoring, n_jobs=args.n_jobs
    )
    
    print("\n✅ 모든 학습 완료!")
    print(f"\n저장된 모델 위치: {args.output}/")


if __name__ == '__main__':
    main()
