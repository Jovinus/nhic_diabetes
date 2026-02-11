"""
Baseline Characteristics 테이블 생성 스크립트
- tableone 패키지를 사용하여 Table 1 생성
- Train vs Test 비교
- Outcome별 비교
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from tableone import TableOne
    HAS_TABLEONE = True
except ImportError:
    HAS_TABLEONE = False
    print("⚠️ tableone 패키지가 설치되어 있지 않습니다.")
    print("   pip install tableone")


# 변수 정의
CONTINUOUS_VARS = [
    'age', 'BMI', 'SBP', 'DBP', 'FBS', 'TOT_CHOL', 'WAIST',
    'TG', 'HDL_CHOL', 'Creatinine', 'LDL_CHOL'
]

CATEGORICAL_VARS = [
    'gender', 'smoking', 'drink', 'training', 'proteinUria',
    'co_HLD', 'co_HTN', 'co_fattyLiver', 'co_Impaird', 'BMIG', 'metS', 'group'
]

# Variable labels (English)
VARIABLE_LABELS = {
    'age': 'Age (years)',
    'gender': 'Sex',
    'BMI': 'BMI (kg/m²)',
    'SBP': 'Systolic BP (mmHg)',
    'DBP': 'Diastolic BP (mmHg)',
    'FBS': 'Fasting glucose (mg/dL)',
    'TOT_CHOL': 'Total cholesterol (mg/dL)',
    'TG': 'Triglyceride (mg/dL)',
    'HDL_CHOL': 'HDL cholesterol (mg/dL)',
    'LDL_CHOL': 'LDL cholesterol (mg/dL)',
    'WAIST': 'Waist circumference (cm)',
    'Creatinine': 'Creatinine (mg/dL)',
    'smoking': 'Smoking status',
    'drink': 'Alcohol (≥2/week)',
    'training': 'Exercise (≥3/week)',
    'proteinUria': 'Proteinuria',
    'co_HLD': 'Hyperlipidemia',
    'co_HTN': 'Hypertension',
    'co_fattyLiver': 'Fatty liver',
    'co_Impaird': 'Impaired glucose tolerance',
    'BMIG': 'BMI group',
    'metS': 'Metabolic syndrome',
    'group': 'Diagnosis group',
    'outA': 'Diabetes incidence',
    'out2': 'T2DM incidence'
}

# Category labels (English)
CATEGORY_LABELS = {
    'gender': {0: 'Male', 1: 'Female'},
    'smoking': {0: 'Never', 1: 'Former', 2: 'Current'},
    'drink': {0: 'No', 1: 'Yes'},
    'training': {0: 'No', 1: 'Yes'},
    'proteinUria': {0: 'Normal', 1: 'Trace/+1', 2: '≥+2'},
    'co_HLD': {0: 'No', 1: 'Yes'},
    'co_HTN': {0: 'No', 1: 'Yes'},
    'co_fattyLiver': {0: 'No', 1: 'Yes'},
    'co_Impaird': {0: 'No', 1: 'Yes'},
    'BMIG': {0: 'Normal (<25)', 1: 'Overweight (25-30)', 2: 'Obese (≥30)'},
    'metS': {0: 'No', 1: 'Yes'},
    'group': {1: 'GS+/Op-', 2: 'GS+/Op+', 3: 'GS-'},
    'outA': {0: 'No', 1: 'Yes'},
    'out2': {0: 'No', 1: 'Yes'}
}


def apply_category_labels(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """범주형 변수에 레이블 적용"""
    df = df.copy()
    columns = columns or list(CATEGORY_LABELS.keys())
    
    for col in columns:
        if col in df.columns and col in CATEGORY_LABELS:
            df[col] = df[col].map(CATEGORY_LABELS[col]).fillna(df[col])
    
    return df


def create_train_test_table(
    df: pd.DataFrame,
    split_col: str = 'split',
    continuous_vars: List[str] = None,
    categorical_vars: List[str] = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Train vs Test 비교 테이블 생성
    
    Args:
        df: 데이터프레임 (split_col 컬럼 포함)
        split_col: Train/Test 구분 컬럼
        continuous_vars: 연속형 변수 리스트
        categorical_vars: 범주형 변수 리스트
        output_path: 저장 경로 (xlsx 또는 csv)
        
    Returns:
        TableOne 결과 데이터프레임
    """
    if not HAS_TABLEONE:
        raise ImportError("tableone 패키지가 필요합니다.")
    
    continuous_vars = continuous_vars or [v for v in CONTINUOUS_VARS if v in df.columns]
    categorical_vars = categorical_vars or [v for v in CATEGORICAL_VARS if v in df.columns]
    
    # 존재하는 변수만 필터링
    continuous_vars = [v for v in continuous_vars if v in df.columns]
    categorical_vars = [v for v in categorical_vars if v in df.columns]
    
    all_vars = continuous_vars + categorical_vars
    
    # 범주형 레이블 적용
    df_labeled = apply_category_labels(df, categorical_vars)
    
    # tableone에서 'Test'가 예약어이므로 레이블 변경
    if split_col in df_labeled.columns:
        df_labeled[split_col] = df_labeled[split_col].replace({
            'Train': 'Training',
            'Test': 'Validation',
            'train': 'Training',
            'test': 'Validation'
        })
    
    # TableOne 생성 (하위 버전 호환 - overall 파라미터 유무 확인)
    import inspect
    tableone_params = inspect.signature(TableOne.__init__).parameters
    
    tableone_kwargs = dict(
        data=df_labeled,
        columns=all_vars,
        categorical=categorical_vars,
        groupby=split_col,
        pval=True,
        rename=VARIABLE_LABELS,
        missing=True,
    )
    if 'overall' in tableone_params:
        tableone_kwargs['overall'] = True
    if 'htest_name' in tableone_params:
        tableone_kwargs['htest_name'] = True
    
    table = TableOne(**tableone_kwargs
    )
    
    print("\n" + "=" * 70)
    print("📊 Train vs Test 비교 테이블")
    print("=" * 70)
    print(table.tabulate(tablefmt="grid"))
    
    # 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith('.xlsx'):
            table.to_excel(output_path)
        elif output_path.endswith('.csv'):
            table.to_csv(output_path)
        else:
            table.to_csv(output_path + '.csv')
        print(f"\n✅ 테이블 저장: {output_path}")
    
    return table.tableone


def create_outcome_table(
    df: pd.DataFrame,
    outcome_col: str = 'outA',
    continuous_vars: List[str] = None,
    categorical_vars: List[str] = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Outcome별 Baseline Characteristics 테이블 생성
    
    Args:
        df: 데이터프레임
        outcome_col: 아웃컴 컬럼명
        continuous_vars: 연속형 변수 리스트
        categorical_vars: 범주형 변수 리스트
        output_path: 저장 경로
        
    Returns:
        TableOne 결과 데이터프레임
    """
    if not HAS_TABLEONE:
        raise ImportError("tableone 패키지가 필요합니다.")
    
    continuous_vars = continuous_vars or [v for v in CONTINUOUS_VARS if v in df.columns]
    categorical_vars = categorical_vars or [v for v in CATEGORICAL_VARS if v in df.columns]
    
    # Outcome 컬럼은 categorical에서 제외
    if outcome_col in categorical_vars:
        categorical_vars = [v for v in categorical_vars if v != outcome_col]
    
    # 존재하는 변수만 필터링
    continuous_vars = [v for v in continuous_vars if v in df.columns]
    categorical_vars = [v for v in categorical_vars if v in df.columns]
    
    all_vars = continuous_vars + categorical_vars
    
    # 범주형 레이블 적용
    df_labeled = apply_category_labels(df, categorical_vars + [outcome_col])
    
    # TableOne 생성 (하위 버전 호환)
    import inspect
    tableone_params = inspect.signature(TableOne.__init__).parameters
    
    tableone_kwargs = dict(
        data=df_labeled,
        columns=all_vars,
        categorical=categorical_vars,
        groupby=outcome_col,
        pval=True,
        rename=VARIABLE_LABELS,
        missing=True,
    )
    if 'overall' in tableone_params:
        tableone_kwargs['overall'] = True
    if 'htest_name' in tableone_params:
        tableone_kwargs['htest_name'] = True
    
    table = TableOne(**tableone_kwargs)
    
    outcome_name = VARIABLE_LABELS.get(outcome_col, outcome_col)
    print("\n" + "=" * 70)
    print(f"📊 Baseline Characteristics by {outcome_name}")
    print("=" * 70)
    print(table.tabulate(tablefmt="grid"))
    
    # 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith('.xlsx'):
            table.to_excel(output_path)
        elif output_path.endswith('.csv'):
            table.to_csv(output_path)
        else:
            table.to_csv(output_path + '.csv')
        print(f"\n✅ 테이블 저장: {output_path}")
    
    return table.tableone


def create_all_tables(
    data_path: str,
    output_dir: str = '../results/tables',
    target_col: str = 'outA',
    test_size: float = 0.2,
    random_state: int = 1004
) -> Dict[str, pd.DataFrame]:
    """
    모든 Baseline Characteristics 테이블 생성
    
    Args:
        data_path: 데이터 파일 경로
        output_dir: 출력 디렉토리
        target_col: 아웃컴 변수
        test_size: 테스트 세트 비율
        random_state: 랜덤 시드
        
    Returns:
        테이블 딕셔너리
    """
    from sklearn.model_selection import train_test_split
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    print("📂 데이터 로드 중...")
    df = pd.read_csv(data_path)
    print(f"   총 {len(df)} 샘플")
    
    # Train/Test 분할
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'Train'
    test_df['split'] = 'Test'
    
    df_with_split = pd.concat([train_df, test_df], ignore_index=True)
    
    tables = {}
    
    # 1. Train vs Test 비교 테이블
    print("\n" + "=" * 70)
    print("1️⃣ Train vs Test 비교 테이블 생성")
    print("=" * 70)
    tables['train_test'] = create_train_test_table(
        df_with_split,
        split_col='split',
        output_path=os.path.join(output_dir, 'table1_train_test.xlsx')
    )
    
    # 2. Outcome별 Baseline Characteristics (전체 데이터)
    print("\n" + "=" * 70)
    print(f"2️⃣ {target_col}별 Baseline Characteristics 테이블 생성 (전체 데이터)")
    print("=" * 70)
    tables['outcome'] = create_outcome_table(
        df,
        outcome_col=target_col,
        output_path=os.path.join(output_dir, f'table1_by_{target_col}.xlsx')
    )
    
    print("\n" + "=" * 70)
    print("✅ 모든 테이블 생성 완료!")
    print("=" * 70)
    print(f"\n저장 위치: {output_dir}/")
    print("  - table1_train_test.xlsx: Train vs Test 비교")
    print(f"  - table1_by_{target_col}.xlsx: {target_col}별 비교")
    
    return tables


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline Characteristics 테이블 생성')
    parser.add_argument('--data', type=str, default='../data/dummy_diabetes_data.csv',
                        help='입력 데이터 경로')
    parser.add_argument('--output', type=str, default='../results/tables',
                        help='출력 디렉토리')
    parser.add_argument('--target', type=str, default='outA',
                        choices=['outA', 'out2'],
                        help='아웃컴 변수')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='테스트 세트 비율')
    
    args = parser.parse_args()
    
    if not HAS_TABLEONE:
        print("❌ tableone 패키지가 설치되어 있지 않습니다.")
        print("   pip install tableone")
        return
    
    create_all_tables(
        data_path=args.data,
        output_dir=args.output,
        target_col=args.target,
        test_size=args.test_size
    )


if __name__ == '__main__':
    main()
