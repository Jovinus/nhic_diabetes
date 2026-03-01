"""
전처리 스크립트
- 결측치 처리 (Missing Indicator 옵션)
- 특성 선택 및 변환
- 훈련/검증/테스트 분할
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


# 특성 변수 정의 (증례기록지 기준)
NUMERIC_FEATURES = [
    'age', 'BMI', 'SBP', 'DBP', 'FBS', 'TOT_CHOL', 'WAIST',
    'TG', 'HDL_CHOL', 'LDL_CHOL', 'Creatinine'
]

CATEGORICAL_FEATURES = [
    'diag', 'act', 'gender', 'smoking', 'drink', 'training', 'proteinUria',
    'co_HLD', 'co_HTN', 'co_fattyLiver', 'co_Impaird', 'metS'
]

# 타겟 변수
TARGET_VARS = {
    'outA': '당뇨병 발생',
    'out2': '2형 당뇨병 발생'
}

# Feature 이름 매핑 (내부 변수명 → 증례기록지 표시명)
FEATURE_RENAME = {
    'age': 'Age',
    'BMI': 'BMI',
    'SBP': 'SBP',
    'DBP': 'DBP',
    'FBS': 'Glucose',
    'TOT_CHOL': 'Total cholesterol',
    'WAIST': 'Waist',
    'TG': 'Triglyceride',
    'HDL_CHOL': 'HDL cholesterol',
    'LDL_CHOL': 'LDL cholesterol',
    'Creatinine': 'Creatinine',
    'diag': 'Cholelithiasis',
    'act': 'Cholecystectomy',
    'gender': 'Sex',
    'smoking': 'Smoking',
    'drink': 'Alcohol',
    'training': 'Training',
    'proteinUria': 'Proteinuria',
    'co_HLD': 'Dyslipidemia',
    'co_HTN': 'Hypertension',
    'co_fattyLiver': 'Fatty liver',
    'co_Impaird': 'Impaired fasting glucose',
    'metS': 'Metabolic syndrome',
}

# Missing Indicator 접미사
MISSING_INDICATOR_SUFFIX = '_missing'


class DiabetesPreprocessor:
    """당뇨병 예측 데이터 전처리 클래스"""
    
    def __init__(
        self,
        numeric_features: List[str] = None,
        categorical_features: List[str] = None,
        target_col: str = 'outA',
        impute_strategy: str = 'median',
        scale_numeric: bool = True,
        add_missing_indicator: bool = False,
        missing_threshold: float = 0.01
    ):
        """
        Args:
            numeric_features: 연속형 특성 리스트
            categorical_features: 범주형 특성 리스트
            target_col: 타겟 변수 컬럼명
            impute_strategy: 결측치 대체 전략 ('mean', 'median', 'most_frequent')
            scale_numeric: 연속형 변수 정규화 여부
            add_missing_indicator: 결측치 지시자 특성 추가 여부
            missing_threshold: 이 비율 이상 결측이 있는 변수만 indicator 추가 (0.01 = 1%)
        """
        self.numeric_features = numeric_features or NUMERIC_FEATURES.copy()
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES.copy()
        self.target_col = target_col
        self.impute_strategy = impute_strategy
        self.scale_numeric = scale_numeric
        self.add_missing_indicator = add_missing_indicator
        self.missing_threshold = missing_threshold
        
        # 전처리 객체들
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.label_encoders = {}
        
        # Missing indicator 관련
        self.missing_indicator_features = []  # 실제로 indicator가 추가된 변수 목록
        
        # 컬럼 순서 저장
        self.feature_names = None
        self.feature_names_without_missing = None  # Missing indicator 제외 버전
        
    def fit(self, df: pd.DataFrame) -> 'DiabetesPreprocessor':
        """
        전처리기 학습
        
        Args:
            df: 학습 데이터프레임
            
        Returns:
            self
        """
        # Missing indicator 대상 결정 (결측 비율이 threshold 이상인 변수)
        if self.add_missing_indicator:
            all_features = self.numeric_features + self.categorical_features
            for feat in all_features:
                if feat in df.columns:
                    missing_rate = df[feat].isna().mean()
                    if missing_rate >= self.missing_threshold:
                        self.missing_indicator_features.append(feat)
            
            if self.missing_indicator_features:
                print(f"📊 Missing Indicator 추가 대상: {self.missing_indicator_features}")
                print(f"   (결측률 {self.missing_threshold*100:.1f}% 이상)")
        
        # 연속형 변수 imputer
        self.numeric_imputer = SimpleImputer(strategy=self.impute_strategy)
        numeric_data = df[self.numeric_features].values
        self.numeric_imputer.fit(numeric_data)
        
        # 범주형 변수 imputer (최빈값)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        categorical_data = df[self.categorical_features].values
        self.categorical_imputer.fit(categorical_data)
        
        # 정규화
        if self.scale_numeric:
            self.scaler = StandardScaler()
            imputed_numeric = self.numeric_imputer.transform(numeric_data)
            self.scaler.fit(imputed_numeric)
        
        # 특성 이름 저장 (증례기록지 표시명으로 변환)
        base_feature_names = [
            FEATURE_RENAME.get(f, f) for f in self.numeric_features + self.categorical_features
        ]
        self.feature_names_without_missing = base_feature_names.copy()

        # Missing indicator 특성 이름 추가
        if self.add_missing_indicator and self.missing_indicator_features:
            missing_indicator_names = [
                f"{FEATURE_RENAME.get(feat, feat)}{MISSING_INDICATOR_SUFFIX}"
                for feat in self.missing_indicator_features
            ]
            self.feature_names = base_feature_names + missing_indicator_names
        else:
            self.feature_names = base_feature_names
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 변환
        
        Args:
            df: 변환할 데이터프레임
            
        Returns:
            (X, y) 튜플
        """
        # Missing indicator 생성 (imputation 전에!)
        missing_indicators = None
        if self.add_missing_indicator and self.missing_indicator_features:
            missing_indicators = np.zeros((len(df), len(self.missing_indicator_features)))
            for i, feat in enumerate(self.missing_indicator_features):
                if feat in df.columns:
                    missing_indicators[:, i] = df[feat].isna().astype(int)
        
        # 연속형 변수 처리
        numeric_data = df[self.numeric_features].values
        numeric_imputed = self.numeric_imputer.transform(numeric_data)
        
        if self.scale_numeric and self.scaler is not None:
            numeric_transformed = self.scaler.transform(numeric_imputed)
        else:
            numeric_transformed = numeric_imputed
        
        # 범주형 변수 처리
        categorical_data = df[self.categorical_features].values
        categorical_imputed = self.categorical_imputer.transform(categorical_data)
        
        # 결합
        if missing_indicators is not None:
            X = np.hstack([numeric_transformed, categorical_imputed, missing_indicators])
        else:
            X = np.hstack([numeric_transformed, categorical_imputed])
        
        # 타겟 변수
        if self.target_col in df.columns:
            y = df[self.target_col].values
        else:
            y = None
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """fit과 transform을 한번에 수행"""
        self.fit(df)
        return self.transform(df)
    
    def save(self, filepath: str) -> None:
        """전처리기 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ 전처리기 저장: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DiabetesPreprocessor':
        """전처리기 로드"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_feature_names(self, include_missing_indicator: bool = True) -> List[str]:
        """
        특성 이름 반환
        
        Args:
            include_missing_indicator: Missing indicator 특성 포함 여부
            
        Returns:
            특성 이름 리스트
        """
        if include_missing_indicator:
            return self.feature_names
        else:
            return self.feature_names_without_missing
    
    def get_missing_indicator_mask(self) -> List[bool]:
        """
        Missing indicator 특성인지 여부 마스크 반환
        
        Returns:
            각 특성이 missing indicator인지 여부 (True/False 리스트)
        """
        return [
            feat.endswith(MISSING_INDICATOR_SUFFIX) 
            for feat in self.feature_names
        ]
    
    def get_non_missing_indicator_indices(self) -> List[int]:
        """
        Missing indicator가 아닌 특성의 인덱스 반환
        
        Returns:
            인덱스 리스트
        """
        return [
            i for i, feat in enumerate(self.feature_names)
            if not feat.endswith(MISSING_INDICATOR_SUFFIX)
        ]


def load_and_split_data(
    data_path: str,
    target_col: str = 'outA',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 1004
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터 로드 및 분할
    
    Args:
        data_path: 데이터 경로
        target_col: 타겟 변수
        test_size: 테스트 세트 비율
        val_size: 검증 세트 비율 (훈련 세트에서 분할)
        random_state: 랜덤 시드
        
    Returns:
        (train_df, val_df, test_df)
    """
    # 데이터 로드
    df = pd.read_csv(data_path)
    print(f"📂 데이터 로드 완료: {len(df)} 샘플")
    
    # 훈련/테스트 분할
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[target_col]
    )
    
    # 훈련/검증 분할
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df[target_col]
        )
    else:
        train_df = train_val_df
        val_df = None
    
    print(f"✅ 데이터 분할 완료:")
    print(f"   - 훈련: {len(train_df)} 샘플")
    if val_df is not None:
        print(f"   - 검증: {len(val_df)} 샘플")
    print(f"   - 테스트: {len(test_df)} 샘플")
    
    return train_df, val_df, test_df


def preprocess_and_save(
    data_path: str,
    output_dir: str = '../data/processed',
    target_col: str = 'outA',
    test_size: float = 0.2,
    val_size: float = 0.1,
    scale_numeric: bool = True,
    add_missing_indicator: bool = False,
    missing_threshold: float = 0.01
):
    """
    전처리 수행 및 저장

    Args:
        data_path: 원본 데이터 경로
        output_dir: 출력 디렉토리
        target_col: 타겟 변수
        test_size: 테스트 세트 비율
        val_size: 검증 세트 비율
        scale_numeric: 정규화 여부
        add_missing_indicator: Missing indicator 추가 여부
        missing_threshold: Missing indicator 추가 기준 결측률 (기본 1%)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("데이터 전처리 시작")
    print("=" * 60)
    
    # 데이터 로드 및 분할
    train_df, val_df, test_df = load_and_split_data(
        data_path, target_col, test_size, val_size
    )
    
    # 전처리기 초기화 및 학습 (훈련 데이터로만)
    preprocessor = DiabetesPreprocessor(
        target_col=target_col,
        scale_numeric=scale_numeric,
        add_missing_indicator=add_missing_indicator,
        missing_threshold=missing_threshold
    )
    
    print("\n📊 사용 특성:")
    print(f"   - 연속형: {preprocessor.numeric_features}")
    print(f"   - 범주형: {preprocessor.categorical_features}")
    if add_missing_indicator:
        print(f"   - Missing Indicator: 활성화 (threshold: {missing_threshold*100:.1f}%)")
    
    # 훈련 데이터 처리
    X_train, y_train = preprocessor.fit_transform(train_df)
    print(f"\n✅ 훈련 데이터 전처리 완료: {X_train.shape}")
    
    # 검증 데이터 처리
    if val_df is not None:
        X_val, y_val = preprocessor.transform(val_df)
        print(f"✅ 검증 데이터 전처리 완료: {X_val.shape}")
    else:
        X_val, y_val = None, None
    
    # 테스트 데이터 처리
    X_test, y_test = preprocessor.transform(test_df)
    print(f"✅ 테스트 데이터 전처리 완료: {X_test.shape}")
    
    # 저장
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    
    if X_val is not None:
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # 전처리기 저장
    preprocessor.save(os.path.join(output_dir, 'preprocessor.pkl'))
    
    # 특성 이름 저장 (전체)
    feature_names = preprocessor.get_feature_names(include_missing_indicator=True)
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Missing indicator 제외 특성 이름도 저장
    feature_names_no_missing = preprocessor.get_feature_names(include_missing_indicator=False)
    with open(os.path.join(output_dir, 'feature_names_no_missing.txt'), 'w') as f:
        f.write('\n'.join(feature_names_no_missing))
    
    print(f"\n✅ 모든 전처리 결과 저장 완료: {output_dir}/")
    
    # 요약 정보
    print("\n" + "=" * 60)
    print("전처리 요약")
    print("=" * 60)
    print(f"타겟 변수: {target_col}")
    print(f"총 특성 수: {len(feature_names)}")
    if preprocessor.missing_indicator_features:
        print(f"  - 원본 특성: {len(feature_names_no_missing)}")
        print(f"  - Missing Indicator: {len(preprocessor.missing_indicator_features)}")
        print(f"    ({', '.join(preprocessor.missing_indicator_features)})")
    print(f"훈련 세트 클래스 분포: {np.bincount(y_train.astype(int))}")
    print(f"테스트 세트 클래스 분포: {np.bincount(y_test.astype(int))}")
    
    return preprocessor


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='당뇨병 예측 데이터 전처리')
    parser.add_argument('--data', type=str, default='../data/dummy_diabetes_data.csv',
                        help='입력 데이터 경로')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='출력 디렉토리')
    parser.add_argument('--target', type=str, default='outA',
                        choices=['outA', 'out2'],
                        help='타겟 변수')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='테스트 세트 비율')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='검증 세트 비율')
    parser.add_argument('--no-scale', action='store_true',
                        help='정규화 비활성화')
    parser.add_argument('--add-missing-indicator', action='store_true',
                        help='Missing indicator 특성 추가')
    parser.add_argument('--missing-threshold', type=float, default=0.05,
                        help='Missing indicator 추가 기준 결측률 (기본: 0.05 = 5%%)')
    
    args = parser.parse_args()
    
    preprocess_and_save(
        data_path=args.data,
        output_dir=args.output,
        target_col=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
        scale_numeric=not args.no_scale,
        add_missing_indicator=args.add_missing_indicator,
        missing_threshold=args.missing_threshold
    )


if __name__ == '__main__':
    main()
