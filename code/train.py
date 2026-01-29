"""
모델 학습 스크립트
- Logistic Regression, XGBoost, LightGBM 지원
- 하이퍼파라미터 튜닝 (Optuna)
- Cross Validation
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# 모델
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 폐쇄망에서 XGBoost, LightGBM이 설치되어 있다면 사용
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost 미설치 - XGBoost 모델 사용 불가")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM 미설치 - LightGBM 모델 사용 불가")

# 하이퍼파라미터 튜닝
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️ Optuna 미설치 - 하이퍼파라미터 튜닝 불가")


class DiabetesTrainer:
    """당뇨병 예측 모델 학습 클래스"""
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        random_state: int = 1004,
        n_jobs: int = -1
    ):
        """
        Args:
            model_type: 모델 타입 ('logistic', 'rf', 'gbdt', 'xgboost', 'lightgbm')
            random_state: 랜덤 시드
            n_jobs: 병렬 처리 수
        """
        self.model_type = model_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.best_params = None
        self.cv_scores = None
        
    def get_default_model(self, params: Dict = None) -> Any:
        """기본 모델 반환"""
        params = params or {}
        
        if self.model_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=self.n_jobs,
                **params
            )
        
        elif self.model_type == 'rf':
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **params
            )
        
        elif self.model_type == 'gbdt':
            return GradientBoostingClassifier(
                random_state=self.random_state,
                **params
            )
        
        elif self.model_type == 'xgboost':
            if not HAS_XGB:
                raise ImportError("XGBoost가 설치되어 있지 않습니다.")
            return xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                use_label_encoder=False,
                eval_metric='logloss',
                **params
            )
        
        elif self.model_type == 'lightgbm':
            if not HAS_LGB:
                raise ImportError("LightGBM이 설치되어 있지 않습니다.")
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                **params
            )
        
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
    
    def get_param_space(self, trial) -> Dict:
        """Optuna용 하이퍼파라미터 탐색 공간"""
        
        if self.model_type == 'logistic':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'saga'
            }
        
        elif self.model_type == 'rf':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        
        elif self.model_type == 'gbdt':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            }
        
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
        
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }
        
        return {}
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv: int = 5,
        scoring: str = 'roc_auc',
        timeout: int = None
    ) -> Dict:
        """
        Optuna를 사용한 하이퍼파라미터 최적화
        
        Args:
            X: 특성 배열
            y: 타겟 배열
            n_trials: 시도 횟수
            cv: Cross-validation fold 수
            scoring: 평가 지표
            timeout: 최대 실행 시간 (초)
            
        Returns:
            최적 하이퍼파라미터
        """
        if not HAS_OPTUNA:
            print("⚠️ Optuna가 설치되어 있지 않아 기본 파라미터를 사용합니다.")
            return {}
        
        def objective(trial):
            params = self.get_param_space(trial)
            model = self.get_default_model(params)
            
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=self.n_jobs)
            
            return scores.mean()
        
        print(f"\n🔍 하이퍼파라미터 최적화 시작 ({self.model_type})")
        print(f"   시도 횟수: {n_trials}, CV: {cv}-fold, 지표: {scoring}")
        
        # Optuna study 생성
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        # verbosity 설정
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"\n✅ 최적화 완료!")
        print(f"   최고 점수: {study.best_value:.4f}")
        print(f"   최적 파라미터: {self.best_params}")
        
        return self.best_params
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        params: Dict = None,
        early_stopping_rounds: int = 50
    ) -> Any:
        """
        모델 학습
        
        Args:
            X_train: 훈련 특성
            y_train: 훈련 타겟
            X_val: 검증 특성 (early stopping용)
            y_val: 검증 타겟
            params: 하이퍼파라미터 (None이면 기본값 또는 최적화된 값 사용)
            early_stopping_rounds: Early stopping 라운드 수
            
        Returns:
            학습된 모델
        """
        # 파라미터 설정
        if params is None:
            params = self.best_params or {}
        
        print(f"\n🚀 모델 학습 시작 ({self.model_type})")
        
        # 모델 생성
        self.model = self.get_default_model(params)
        
        # 학습
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            # Early stopping 사용
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif self.model_type == 'lightgbm':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
        else:
            self.model.fit(X_train, y_train)
        
        print("✅ 학습 완료!")
        
        return self.model
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict:
        """
        Cross-validation 수행
        
        Args:
            X: 특성 배열
            y: 타겟 배열
            cv: fold 수
            scoring: 평가 지표
            
        Returns:
            CV 결과
        """
        print(f"\n📊 Cross-Validation ({cv}-fold, {scoring})")
        
        params = self.best_params or {}
        model = self.get_default_model(params)
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=self.n_jobs)
        
        self.cv_scores = {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"   점수: {scores.round(4)}")
        print(f"   평균: {scores.mean():.4f} (±{scores.std():.4f})")
        
        return self.cv_scores
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'xgboost' and HAS_XGB:
            self.model.save_model(filepath.replace('.pkl', '.json'))
            print(f"✅ XGBoost 모델 저장: {filepath.replace('.pkl', '.json')}")
        elif self.model_type == 'lightgbm' and HAS_LGB:
            self.model.booster_.save_model(filepath.replace('.pkl', '.txt'))
            print(f"✅ LightGBM 모델 저장: {filepath.replace('.pkl', '.txt')}")
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ 모델 저장: {filepath}")
        
        # 메타데이터 저장
        meta = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'train_date': datetime.now().isoformat()
        }
        meta_path = filepath.replace('.pkl', '_meta.json').replace('.json', '_meta.json').replace('.txt', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✅ 메타데이터 저장: {meta_path}")
    
    def load_model(self, filepath: str) -> Any:
        """모델 로드"""
        if filepath.endswith('.json') and HAS_XGB:
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)
        elif filepath.endswith('.txt') and HAS_LGB:
            self.model = lgb.Booster(model_file=filepath)
        else:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        
        print(f"✅ 모델 로드: {filepath}")
        return self.model


def load_processed_data(data_dir: str = '../data/processed') -> Tuple:
    """전처리된 데이터 로드"""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    X_val_path = os.path.join(data_dir, 'X_val.npy')
    if os.path.exists(X_val_path):
        X_val = np.load(X_val_path)
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    else:
        X_val, y_val = None, None
    
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # 특성 이름 로드
    with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    print(f"📂 데이터 로드 완료:")
    print(f"   - 훈련: {X_train.shape}")
    if X_val is not None:
        print(f"   - 검증: {X_val.shape}")
    print(f"   - 테스트: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str = '../models',
    optimize: bool = True,
    n_trials: int = 30
) -> Dict[str, DiabetesTrainer]:
    """
    여러 모델 학습
    
    Args:
        X_train, y_train: 훈련 데이터
        X_val, y_val: 검증 데이터
        output_dir: 모델 저장 디렉토리
        optimize: 하이퍼파라미터 최적화 여부
        n_trials: Optuna 시도 횟수
        
    Returns:
        학습된 모델들
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 사용 가능한 모델 목록
    model_types = ['logistic', 'rf', 'gbdt']
    if HAS_XGB:
        model_types.append('xgboost')
    if HAS_LGB:
        model_types.append('lightgbm')
    
    trainers = {}
    results = {}
    
    for model_type in model_types:
        print("\n" + "=" * 60)
        print(f"🎯 {model_type.upper()} 학습")
        print("=" * 60)
        
        trainer = DiabetesTrainer(model_type=model_type)
        
        # 하이퍼파라미터 최적화
        if optimize and HAS_OPTUNA:
            trainer.optimize_hyperparameters(
                X_train, y_train,
                n_trials=n_trials,
                cv=5
            )
        
        # 학습
        trainer.train(X_train, y_train, X_val, y_val)
        
        # Cross-validation
        cv_results = trainer.cross_validate(X_train, y_train)
        
        # 저장
        model_path = os.path.join(output_dir, f'{model_type}_model.pkl')
        trainer.save_model(model_path)
        
        trainers[model_type] = trainer
        results[model_type] = cv_results['mean']
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 모델 비교 (CV AUROC)")
    print("=" * 60)
    for model_type, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model_type:12s}: {score:.4f}")
    
    return trainers


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='당뇨병 예측 모델 학습')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='전처리된 데이터 디렉토리')
    parser.add_argument('--output', type=str, default='../models',
                        help='모델 저장 디렉토리')
    parser.add_argument('--model', type=str, default='all',
                        choices=['logistic', 'rf', 'gbdt', 'xgboost', 'lightgbm', 'all'],
                        help='학습할 모델')
    parser.add_argument('--no-optimize', action='store_true',
                        help='하이퍼파라미터 최적화 비활성화')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Optuna 시도 횟수')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("당뇨병 예측 모델 학습")
    print("=" * 60)
    
    # 데이터 로드
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_processed_data(args.data_dir)
    
    if args.model == 'all':
        # 모든 모델 학습
        trainers = train_all_models(
            X_train, y_train, X_val, y_val,
            output_dir=args.output,
            optimize=not args.no_optimize,
            n_trials=args.n_trials
        )
    else:
        # 단일 모델 학습
        trainer = DiabetesTrainer(model_type=args.model)
        
        if not args.no_optimize and HAS_OPTUNA:
            trainer.optimize_hyperparameters(X_train, y_train, n_trials=args.n_trials)
        
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.cross_validate(X_train, y_train)
        
        os.makedirs(args.output, exist_ok=True)
        trainer.save_model(os.path.join(args.output, f'{args.model}_model.pkl'))
    
    print("\n✅ 학습 완료!")


if __name__ == '__main__':
    main()
