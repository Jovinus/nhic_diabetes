"""
Crude/Adjusted Odds Ratio 분석 스크립트
- statsmodels 기반 로지스틱 회귀분석
- Crude OR: 각 변수별 단변량 분석
- Adjusted OR: 3단계 다변량 모델 (diag, act에 대해)
- 결과를 Excel 테이블로 출력
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

from preprocessing import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_RENAME
)


# =============================================================================
# 분석 변수 정의
# =============================================================================

# 주요 노출 변수
EXPOSURE_VARS = ['diag', 'act']

# Adjusted 모델 보정 변수 (단계적)
ADJUST_MODEL1 = ['age', 'gender']
ADJUST_MODEL2 = ADJUST_MODEL1 + ['BMI', 'smoking', 'drink', 'training']
ADJUST_MODEL3 = ADJUST_MODEL2 + [
    'SBP', 'DBP', 'FBS', 'TOT_CHOL', 'TG', 'HDL_CHOL', 'Creatinine',
    'co_HTN', 'co_HLD', 'co_fattyLiver', 'co_Impaird'
]

# Crude 분석 대상 변수 (모든 독립변수)
CRUDE_VARS = [
    'diag', 'act', 'age', 'gender', 'BMI', 'smoking', 'drink', 'training',
    'SBP', 'DBP', 'FBS', 'TOT_CHOL', 'WAIST', 'TG', 'HDL_CHOL',
    'LDL_CHOL', 'Creatinine', 'proteinUria',
    'co_HLD', 'co_HTN', 'co_fattyLiver', 'co_Impaird', 'metS'
]


def format_or(or_val, ci_lower, ci_upper, p_val):
    """OR (95% CI) 포맷팅"""
    or_str = "{:.2f} ({:.2f}-{:.2f})".format(or_val, ci_lower, ci_upper)
    if p_val < 0.001:
        p_str = "<0.001"
    else:
        p_str = "{:.3f}".format(p_val)
    return or_str, p_str


def run_logistic_single(df, outcome_col, predictor, covariates=None):
    """
    단일 로지스틱 회귀 실행 (crude 또는 adjusted)

    Args:
        df: 데이터프레임
        outcome_col: 종속변수 컬럼명
        predictor: 주요 독립변수 컬럼명
        covariates: 보정변수 리스트 (None이면 crude)

    Returns:
        dict with OR, CI, p-value for the predictor, or None on failure
    """
    if covariates is None:
        X_cols = [predictor]
    else:
        X_cols = [predictor] + [c for c in covariates if c != predictor]

    # 분석에 필요한 컬럼만 선택 후 결측치 제거 (complete case)
    cols_needed = X_cols + [outcome_col]
    df_sub = df[cols_needed].dropna()

    if len(df_sub) < 20:
        return None

    y = df_sub[outcome_col].values
    X = df_sub[X_cols].values
    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0, maxiter=100)

        # predictor의 인덱스는 1 (상수항이 0)
        idx = 1
        coef = result.params[idx]
        ci = result.conf_int()[idx]
        p_val = result.pvalues[idx]

        or_val = np.exp(coef)
        ci_lower = np.exp(ci[0])
        ci_upper = np.exp(ci[1])

        return {
            'OR': or_val,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_val,
            'n': len(df_sub),
            'converged': result.mle_retvals.get('converged', True)
        }
    except Exception as e:
        print("  Warning: {} regression failed for {}: {}".format(
            'Adjusted' if covariates else 'Crude', predictor, e))
        return None


def run_crude_analysis(df, outcome_col, variables=None):
    """
    Crude OR 분석 (각 변수별 단변량 로지스틱 회귀)

    Args:
        df: 데이터프레임
        outcome_col: 종속변수
        variables: 분석할 변수 리스트

    Returns:
        DataFrame with crude OR results
    """
    if variables is None:
        variables = CRUDE_VARS

    variables = [v for v in variables if v in df.columns]

    results = []
    for var in variables:
        res = run_logistic_single(df, outcome_col, var, covariates=None)
        if res is not None:
            or_str, p_str = format_or(res['OR'], res['CI_lower'], res['CI_upper'], res['p_value'])
            results.append({
                'Variable': FEATURE_RENAME.get(var, var),
                'Variable_raw': var,
                'Crude OR (95% CI)': or_str,
                'p-value': p_str,
                'p_numeric': res['p_value'],
                'N': res['n']
            })
        else:
            results.append({
                'Variable': FEATURE_RENAME.get(var, var),
                'Variable_raw': var,
                'Crude OR (95% CI)': 'N/A',
                'p-value': 'N/A',
                'p_numeric': np.nan,
                'N': 0
            })

    return pd.DataFrame(results)


def run_adjusted_analysis(df, outcome_col, exposure_vars=None):
    """
    Adjusted OR 분석 (3단계 모델로 exposure 변수의 보정된 OR 산출)

    Args:
        df: 데이터프레임
        outcome_col: 종속변수
        exposure_vars: 주요 노출 변수 리스트

    Returns:
        DataFrame with adjusted OR results
    """
    if exposure_vars is None:
        exposure_vars = EXPOSURE_VARS

    exposure_vars = [v for v in exposure_vars if v in df.columns]

    models = {
        'Model 1 (Age, Sex)': ADJUST_MODEL1,
        'Model 2 (+Lifestyle)': ADJUST_MODEL2,
        'Model 3 (+Clinical)': ADJUST_MODEL3,
    }

    results = []
    for exp_var in exposure_vars:
        row = {
            'Variable': FEATURE_RENAME.get(exp_var, exp_var),
            'Variable_raw': exp_var,
        }

        # Crude
        res_crude = run_logistic_single(df, outcome_col, exp_var, covariates=None)
        if res_crude is not None:
            or_str, p_str = format_or(
                res_crude['OR'], res_crude['CI_lower'], res_crude['CI_upper'], res_crude['p_value'])
            row['Crude OR (95% CI)'] = or_str
            row['Crude p-value'] = p_str
        else:
            row['Crude OR (95% CI)'] = 'N/A'
            row['Crude p-value'] = 'N/A'

        # Adjusted models
        for model_name, covariates in models.items():
            covariates_available = [c for c in covariates if c in df.columns]
            res = run_logistic_single(df, outcome_col, exp_var, covariates=covariates_available)
            if res is not None:
                or_str, p_str = format_or(
                    res['OR'], res['CI_lower'], res['CI_upper'], res['p_value'])
                row['{} OR (95% CI)'.format(model_name)] = or_str
                row['{} p-value'.format(model_name)] = p_str
                row['{} N'.format(model_name)] = res['n']
            else:
                row['{} OR (95% CI)'.format(model_name)] = 'N/A'
                row['{} p-value'.format(model_name)] = 'N/A'
                row['{} N'.format(model_name)] = 0

        results.append(row)

    return pd.DataFrame(results)


def run_or_analysis(data_path, output_dir, target_col='outA'):
    """
    전체 OR 분석 실행 및 Excel 저장

    Args:
        data_path: 데이터 CSV 경로
        output_dir: 결과 저장 디렉토리
        target_col: 종속변수 ('outA' 또는 'out2')
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Crude/Adjusted OR Analysis")
    print("  Outcome: {}".format(target_col))
    print("=" * 60)

    # 데이터 로드
    df = pd.read_csv(data_path)
    print("  Data loaded: {} samples".format(len(df)))

    # Target 분포
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        print("  {} distribution: 0={}, 1={}".format(
            target_col, counts.get(0, 0), counts.get(1, 0)))
    else:
        raise ValueError("Target column '{}' not found in data".format(target_col))

    # 1. Crude OR 분석
    print("\n--- Crude OR Analysis ---")
    crude_df = run_crude_analysis(df, target_col)
    print("  {} variables analyzed".format(len(crude_df)))

    # 유의한 변수 출력
    sig = crude_df[crude_df['p_numeric'] < 0.05]
    if len(sig) > 0:
        print("  Significant variables (p<0.05):")
        for _, r in sig.iterrows():
            print("    {}: OR={}, p={}".format(r['Variable'], r['Crude OR (95% CI)'], r['p-value']))

    # 2. Adjusted OR 분석
    print("\n--- Adjusted OR Analysis ---")
    adjusted_df = run_adjusted_analysis(df, target_col)
    print("  {} exposure variables analyzed".format(len(adjusted_df)))

    for _, r in adjusted_df.iterrows():
        print("\n  {}:".format(r['Variable']))
        print("    Crude: OR={}".format(r.get('Crude OR (95% CI)', 'N/A')))
        for col in r.index:
            if 'Model' in col and 'OR' in col:
                print("    {}: {}".format(col.split(' OR')[0], r[col]))

    # 3. Excel 저장
    output_path = os.path.join(output_dir, 'or_analysis_{}.xlsx'.format(target_col))

    # crude_df에서 내부용 컬럼 제거
    crude_export = crude_df.drop(columns=['Variable_raw', 'p_numeric'], errors='ignore')
    adjusted_export = adjusted_df.drop(columns=['Variable_raw'], errors='ignore')

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        crude_export.to_excel(writer, sheet_name='Crude OR', index=False)
        adjusted_export.to_excel(writer, sheet_name='Adjusted OR', index=False)

    print("\n  Saved: {}".format(output_path))

    return crude_df, adjusted_df


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Crude/Adjusted OR Analysis')
    parser.add_argument('--data', type=str, default='../data/dummy_diabetes_data.csv',
                        help='Input data CSV path')
    parser.add_argument('--output', type=str, default='../results/tables',
                        help='Output directory')
    parser.add_argument('--target', type=str, default='outA',
                        choices=['outA', 'out2'],
                        help='Outcome variable')

    args = parser.parse_args()

    run_or_analysis(
        data_path=args.data,
        output_dir=args.output,
        target_col=args.target
    )


if __name__ == '__main__':
    main()
