#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF Analysis Report Generator (Korean + English)
- matplotlib PdfPages 기반 다중 페이지 PDF
- NanumGothic 한글 폰트 사용
- 가로(landscape) A4 레이아웃
- 한글/영문 2개 PDF 동시 생성
"""

import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'export')

# ── Korean font setup ──
_font_dir = os.path.expanduser('~/.local/share/fonts')
for _fn in ['NanumGothic-Regular.ttf', 'NanumGothic-Bold.ttf']:
    _fp = os.path.join(_font_dir, _fn)
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
FONT_KO = 'NanumGothic'
FONT_EN = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ── Constants ──
TARGETS = ['outA', 'out2']
MODEL_ORDER = [
    'logistic_regression', 'decision_tree', 'random_forest',
    'xgboost', 'lightgbm', 'ann'
]
MODEL_DISPLAY = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'ann': 'ANN (MLP)',
}

PAGE_W, PAGE_H = 11.69, 8.27  # Landscape A4

# Colors
C_HEADER = '#2F5496'
C_HEADER_LT = '#D6E4F0'
C_ALT = '#F2F7FC'
C_BEST = '#E2EFDA'
C_WARN = '#FFFDE7'


# ═══════════════════════════════════════════════════════════════════════
# i18n Text Dictionary
# ═══════════════════════════════════════════════════════════════════════

T = {
    # ── Title page ──
    'title_main': {
        'ko': '담석증 및 담낭절제술이\n당뇨병 발생에 미치는 영향 분석',
        'en': 'Association of Cholelithiasis\nand Cholecystectomy\nwith Diabetes Incidence',
    },
    'title_sub': {
        'ko': '머신러닝 예측 모델 개발 및 위험인자 분석 보고서',
        'en': 'ML Prediction Model Development & Risk Factor Analysis Report',
    },
    'title_info': {
        'ko': (
            '데이터: 국민건강보험공단 건강검진 코호트\n'
            '          (파이프라인 검증용 합성 데이터, N=10,000)\n\n'
            '연구 가설: 담석증 진단 여부 및 담낭절제술 시행 여부가\n'
            '          당뇨병 발생에 영향을 미치는가?\n\n'
            'Outcome: outA (당뇨병 전체)  /  out2 (2형 당뇨병)\n\n'
            '모델: LR, Decision Tree, Random Forest,\n'
            '       XGBoost, LightGBM, ANN (MLP)\n\n'
            '분석일: {date}'
        ),
        'en': (
            'Data: NHIC Health Checkup Cohort\n'
            '       (Synthetic data, N=10,000, for pipeline validation)\n\n'
            'Hypothesis: Whether cholelithiasis diagnosis and/or\n'
            '       cholecystectomy affect diabetes incidence\n\n'
            'Outcomes: outA (Any Diabetes) / out2 (Type 2 Diabetes)\n\n'
            'Models: LR, Decision Tree, Random Forest,\n'
            '       XGBoost, LightGBM, ANN (MLP)\n\n'
            'Date: {date}'
        ),
    },

    # ── Overview ──
    'sec_overview': {'ko': '1. 연구 개요', 'en': '1. Study Overview'},
    'research_q': {
        'ko': (
            "연구 질문 (Research Question)\n\n"
            "담석증(Cholelithiasis) 진단 및 담낭절제술(Cholecystectomy) 시행이\n"
            "당뇨병 발생 위험에 영향을 미치는가?\n\n"
            " - 노출 변수: 담석증 진단(diag), 담낭절제술(act)\n"
            " - 결과 변수: 당뇨병(outA), 2형 당뇨병(out2)\n"
            " - 연구 대상: 담석증 환자 + 비환자 모두 포함 (건강검진 코호트)"
        ),
        'en': (
            "Research Question\n\n"
            "Do cholelithiasis (gallstone) diagnosis and cholecystectomy\n"
            "(gallbladder removal) affect the risk of developing diabetes?\n\n"
            " - Exposure: Cholelithiasis (diag), Cholecystectomy (act)\n"
            " - Outcome: Any diabetes (outA), Type 2 diabetes (out2)\n"
            " - Includes BOTH patients with and without cholelithiasis"
        ),
    },
    'data_overview': {'ko': '데이터 개요', 'en': 'Data Overview'},
    'data_rows': {
        'ko': [['전체 대상자', '10,000명 (담석증 유/무 포함)'],
               ['훈련 세트 (70%)', '7,000명'],
               ['검증 세트 (10%)', '1,000명'],
               ['테스트 세트 (20%)', '2,000명'],
               ['입력 변수', '23개 + Missing indicator 2개']],
        'en': [['Total Subjects', '10,000 (with & without cholelithiasis)'],
               ['Training Set (70%)', '7,000'],
               ['Validation Set (10%)', '1,000'],
               ['Test Set (20%)', '2,000'],
               ['Input Features', '23 + 2 missing indicators']],
    },
    'data_headers': {'ko': ['항목', '값'], 'en': ['Item', 'Value']},
    'outcome_title': {'ko': 'Outcome 분포', 'en': 'Outcome Distribution'},
    'outcome_headers': {
        'ko': ['Outcome', '양성(n)', '음성(n)', '발생률'],
        'en': ['Outcome', 'Positive', 'Negative', 'Incidence'],
    },
    'outcome_rows': {
        'ko': [['outA (당뇨병)', '3,139', '6,861', '31.4%'],
               ['out2 (2형 당뇨병)', '2,771', '7,229', '27.7%']],
        'en': [['outA (Any Diabetes)', '3,139', '6,861', '31.4%'],
               ['out2 (Type 2 Diabetes)', '2,771', '7,229', '27.7%']],
    },
    'methods_title': {'ko': '분석 방법', 'en': 'Methods'},
    'methods': {
        'ko': [
            ('전처리',
             '연속형: 중앙값 대체 + StandardScaler  |  범주형: 최빈값 대체\n'
             'Missing Indicator: 결측률 5% 이상 변수에 추가 (LDL, Proteinuria)'),
            ('OR 분석',
             'Crude OR (단변량) + Adjusted OR (3단계 보정 모델, diag/act)'),
            ('모델 학습',
             'GridSearchCV (5-fold CV, AUROC)  |  6개 모델'),
            ('평가',
             'AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, F1\n'
             'Bootstrap 95% CI  |  SHAP 해석'),
        ],
        'en': [
            ('Preprocessing',
             'Continuous: median imputation + StandardScaler  |  Categorical: mode\n'
             'Missing Indicator: added for features with >5% missingness'),
            ('OR Analysis',
             'Crude OR (univariate) + Adjusted OR (3-step models for diag/act)'),
            ('Training',
             'GridSearchCV (5-fold CV, AUROC scoring)  |  6 models'),
            ('Evaluation',
             'AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, F1\n'
             'Bootstrap 95% CI  |  SHAP interpretation'),
        ],
    },

    # ── Variables ──
    'sec_variables': {'ko': '2. 변수 정의', 'en': '2. Variable Definitions'},
    'var_numeric_title': {
        'ko': '연속형 입력 변수 (11개)',
        'en': 'Numeric Input Variables (11)',
    },
    'var_cat_title': {
        'ko': '범주형 입력 변수 (12개) + 결과 변수 (2개)',
        'en': 'Categorical Input Variables (12) + Outcomes (2)',
    },
    'var_headers': {
        'ko': ['변수명', '표시명', '타입', '설명'],
        'en': ['Variable', 'Display Name', 'Type', 'Description'],
    },
    'var_numeric': {
        'ko': [['age', 'Age (years)', '연속형', '연령'],
               ['BMI', 'BMI (kg/m²)', '연속형', '체질량지수'],
               ['SBP', 'SBP (mmHg)', '연속형', '수축기 혈압'],
               ['DBP', 'DBP (mmHg)', '연속형', '이완기 혈압'],
               ['FBS', 'Glucose (mg/dL)', '연속형', '공복혈당'],
               ['TOT_CHOL', 'Total cholesterol (mg/dL)', '연속형', '총 콜레스테롤'],
               ['WAIST', 'Waist (cm)', '연속형', '허리둘레'],
               ['TG', 'Triglyceride (mg/dL)', '연속형', '중성지방'],
               ['HDL_CHOL', 'HDL cholesterol (mg/dL)', '연속형', 'HDL 콜레스테롤'],
               ['LDL_CHOL', 'LDL cholesterol (mg/dL)', '연속형', 'LDL 콜레스테롤 (결측 ~30%)'],
               ['Creatinine', 'Creatinine (mg/dL)', '연속형', '크레아티닌']],
        'en': [['age', 'Age (years)', 'Continuous', 'Age at examination'],
               ['BMI', 'BMI (kg/m²)', 'Continuous', 'Body mass index'],
               ['SBP', 'SBP (mmHg)', 'Continuous', 'Systolic blood pressure'],
               ['DBP', 'DBP (mmHg)', 'Continuous', 'Diastolic blood pressure'],
               ['FBS', 'Glucose (mg/dL)', 'Continuous', 'Fasting blood sugar'],
               ['TOT_CHOL', 'Total cholesterol (mg/dL)', 'Continuous', 'Total cholesterol'],
               ['WAIST', 'Waist (cm)', 'Continuous', 'Waist circumference'],
               ['TG', 'Triglyceride (mg/dL)', 'Continuous', 'Triglyceride'],
               ['HDL_CHOL', 'HDL cholesterol (mg/dL)', 'Continuous', 'HDL cholesterol'],
               ['LDL_CHOL', 'LDL cholesterol (mg/dL)', 'Continuous', 'LDL cholesterol (~30% missing)'],
               ['Creatinine', 'Creatinine (mg/dL)', 'Continuous', 'Serum creatinine']],
    },
    'var_cat': {
        'ko': [['diag', 'Cholelithiasis (담석증)', '이진 (0/1)', '노출변수: 담석증 진단 유무'],
               ['act', 'Cholecystectomy (담낭절제술)', '이진 (0/1)', '노출변수: 담낭절제술 시행 유무'],
               ['gender', 'Sex', '이진 (0/1)', '성별 (0=남, 1=여)'],
               ['smoking', 'Smoking', '범주 (0/1/2)', '흡연 (0=비흡연, 1=과거, 2=현재)'],
               ['drink', 'Alcohol', '이진 (0/1)', '주 2일 이상 음주'],
               ['training', 'Training', '이진 (0/1)', '주 3일 이상 운동'],
               ['proteinUria', 'Proteinuria', '범주 (0/1/2)', '단백뇨 (0=정상, 1=미량, 2=2+ 이상)'],
               ['co_HLD', 'Dyslipidemia', '이진 (0/1)', '이상지질혈증 동반'],
               ['co_HTN', 'Hypertension', '이진 (0/1)', '고혈압 동반'],
               ['co_fattyLiver', 'Fatty liver', '이진 (0/1)', '지방간 동반'],
               ['co_Impaird', 'Impaired fasting glucose', '이진 (0/1)', '공복혈당장애 동반'],
               ['metS', 'Metabolic syndrome', '이진 (0/1)', '대사증후군'],
               ['outA', 'Diabetes incidence', '이진 (0/1)', '결과변수: 당뇨병 발생 (primary)'],
               ['out2', 'Type 2 DM incidence', '이진 (0/1)', '결과변수: 2형 당뇨병 발생 (secondary)']],
        'en': [['diag', 'Cholelithiasis', 'Binary (0/1)', 'EXPOSURE: Gallstone diagnosis'],
               ['act', 'Cholecystectomy', 'Binary (0/1)', 'EXPOSURE: Gallbladder removal'],
               ['gender', 'Sex', 'Binary (0/1)', '0=Male, 1=Female'],
               ['smoking', 'Smoking', 'Cat (0/1/2)', '0=Never, 1=Former, 2=Current'],
               ['drink', 'Alcohol', 'Binary (0/1)', 'Alcohol >= 2 days/week'],
               ['training', 'Training', 'Binary (0/1)', 'Exercise >= 3 days/week'],
               ['proteinUria', 'Proteinuria', 'Cat (0/1/2)', '0=Normal, 1=Trace, 2>=+2'],
               ['co_HLD', 'Dyslipidemia', 'Binary (0/1)', 'Comorbid dyslipidemia'],
               ['co_HTN', 'Hypertension', 'Binary (0/1)', 'Comorbid hypertension'],
               ['co_fattyLiver', 'Fatty liver', 'Binary (0/1)', 'Comorbid fatty liver'],
               ['co_Impaird', 'IFG', 'Binary (0/1)', 'Impaired fasting glucose'],
               ['metS', 'Metabolic syndrome', 'Binary (0/1)', 'Metabolic syndrome'],
               ['outA', 'Diabetes incidence', 'Binary (0/1)', 'OUTCOME: any diabetes (primary)'],
               ['out2', 'T2DM incidence', 'Binary (0/1)', 'OUTCOME: type 2 diabetes (secondary)']],
    },

    # ── Crude OR ──
    'sec_crude': {'ko': '3. Crude OR 분석', 'en': '3. Crude OR Analysis'},
    'crude_sub': {
        'ko': '각 변수별 단변량 로지스틱 회귀분석 결과',
        'en': 'Univariate logistic regression for each variable',
    },
    'crude_table_title': {
        'ko': 'Crude OR — 전체 변수 (N={n})',
        'en': 'Crude OR — All Variables (N={n})',
    },
    'crude_headers': {
        'ko': ['변수', 'outA OR (95% CI)', 'p-value', 'out2 OR (95% CI)', 'p-value'],
        'en': ['Variable', 'outA OR (95% CI)', 'p-value', 'out2 OR (95% CI)', 'p-value'],
    },
    'crude_note': {
        'ko': '* 초록색 = 주요 노출변수 (담석증, 담낭절제술)',
        'en': '* Green = primary exposure variables (cholelithiasis, cholecystectomy)',
    },

    # ── Adjusted OR ──
    'sec_adjusted': {'ko': '4. Adjusted OR 분석', 'en': '4. Adjusted OR Analysis'},
    'adj_sub': {
        'ko': '노출변수(담석증, 담낭절제술)에 대한 단계적 보정 모델',
        'en': 'Stepwise adjustment models for exposure variables (diag, act)',
    },
    'adj_desc': {
        'ko': (
            "보정 모델 구성\n"
            "  Model 1: Age, Sex\n"
            "  Model 2: + BMI, Smoking, Alcohol, Training\n"
            "  Model 3: + SBP, DBP, Glucose, Total cholesterol, TG, HDL,\n"
            "           Creatinine, HTN, Dyslipidemia, Fatty liver, IFG"
        ),
        'en': (
            "Adjustment Models\n"
            "  Model 1: Age, Sex\n"
            "  Model 2: + BMI, Smoking, Alcohol, Training\n"
            "  Model 3: + SBP, DBP, Glucose, Total cholesterol, TG, HDL,\n"
            "           Creatinine, HTN, Dyslipidemia, Fatty liver, IFG"
        ),
    },
    'adj_chol_title': {
        'ko': '담석증 (Cholelithiasis)',
        'en': 'Cholelithiasis (Gallstone)',
    },
    'adj_chole_title': {
        'ko': '담낭절제술 (Cholecystectomy)',
        'en': 'Cholecystectomy (Gallbladder Removal)',
    },
    'adj_headers': {
        'ko': ['보정 모델', 'outA OR (95% CI)', 'out2 OR (95% CI)'],
        'en': ['Model', 'outA OR (95% CI)', 'out2 OR (95% CI)'],
    },
    'adj_interp': {
        'ko': (
            "해석\n\n"
            "담석증(Cholelithiasis)은 당뇨병 발생의 독립적 위험인자\n"
            "  - 완전 보정(Model 3): outA OR=1.23 (p<0.001), out2 OR=1.17 (p<0.001)\n"
            "  - 보정 변수 추가에도 OR이 유지/증가 → 교란변수에 의한 결과가 아님\n\n"
            "담낭절제술(Cholecystectomy)은 당뇨병 발생과 유의한 연관 없음\n"
            "  - 모든 모델에서 OR ≈ 0.96-0.97, 모두 p > 0.05"
        ),
        'en': (
            "Interpretation\n\n"
            "Cholelithiasis is an independent risk factor for diabetes.\n"
            "  - Fully adjusted (Model 3): outA OR=1.23, out2 OR=1.17 (both p<0.001)\n"
            "  - OR maintained/increased after adjustment -> not confounding-driven\n\n"
            "Cholecystectomy shows NO significant association with diabetes.\n"
            "  - All models: OR ~ 0.96-0.97, all p > 0.05"
        ),
    },

    # ── Performance ──
    'sec_perf': {
        'ko': '5. 모델 성능 — {target}',
        'en': '5. Model Performance — {target}',
    },
    'target_label': {
        'ko': {'outA': '당뇨병 발생 (Any Diabetes)',
               'out2': '2형 당뇨병 발생 (Type 2 Diabetes)'},
        'en': {'outA': 'Diabetes Incidence (Any Type)',
               'out2': 'Type 2 Diabetes Incidence'},
    },
    'perf_table_title': {
        'ko': '테스트 세트 성능 (Bootstrap 95% CI)',
        'en': 'Test Set Performance (Bootstrap 95% CI)',
    },
    'auroc_title': {'ko': 'AUROC 요약', 'en': 'AUROC Summary'},
    'auroc_headers': {'ko': ['모델', 'Test AUROC'], 'en': ['Model', 'Test AUROC']},
    'perf_note': {
        'ko': ('비고\n'
               '- 초록색 행 = AUROC 최고 모델\n'
               '- Bootstrap: n=50 resamples\n'
               '- 임계값: Youden Index 기준\n'
               '- 합성(더미) 데이터 기반 결과\n'
               '  (파이프라인 검증 목적)'),
        'en': ('Notes\n'
               '- Green row = best AUROC model\n'
               '- Bootstrap: n=50 resamples\n'
               '- Threshold: Youden Index\n'
               '- Synthetic (dummy) data results\n'
               '  (pipeline validation only)'),
    },

    # ── Figures ──
    'fig_roc': {'ko': '6. ROC Curve 비교', 'en': '6. ROC Curve Comparison'},
    'fig_pr_cal': {'ko': '7. PR & Calibration 비교', 'en': '7. PR & Calibration'},
    'fig_shap': {'ko': '8. SHAP Feature Importance 비교', 'en': '8. SHAP Feature Importance'},
    'fig_missing': {'ko': '그림 없음: {}', 'en': 'Figure not available: {}'},

    # ── Findings ──
    'sec_findings': {'ko': '주요 발견 및 한계', 'en': 'Key Findings & Limitations'},
    'findings': {
        'ko': (
            "주요 발견 (Key Findings)\n\n"
            "1. 담석증(Cholelithiasis)은 당뇨병 발생의 독립적 위험인자\n"
            "   - outA: 완전 보정(Model 3) OR = 1.23 (1.10-1.37), p<0.001\n"
            "   - out2: 완전 보정(Model 3) OR = 1.17 (1.05-1.31), p<0.001\n"
            "   - 보정 변수 추가 시에도 OR이 유지/증가\n\n"
            "2. 담낭절제술(Cholecystectomy)은 유의한 연관성 없음\n"
            "   - 모든 모델에서 OR ≈ 0.96-0.97 (p > 0.05)\n\n"
            "3. 가장 강력한 위험인자 (Crude OR)\n"
            "   - 공복혈당장애(IFG): OR ≈ 1.56-1.58\n"
            "   - 대사증후군: OR ≈ 1.33-1.36\n\n"
            "4. 머신러닝 모델 성능\n"
            "   - AUROC 범위: 0.53-0.58 (합성 데이터 한계)\n"
            "   - Logistic Regression이 동등하거나 최고 성능"
        ),
        'en': (
            "Key Findings\n\n"
            "1. Cholelithiasis is an independent risk factor for diabetes\n"
            "   - outA: Model 3 OR = 1.23 (1.10-1.37), p<0.001\n"
            "   - out2: Model 3 OR = 1.17 (1.05-1.31), p<0.001\n"
            "   - OR maintained/increased after full adjustment\n\n"
            "2. Cholecystectomy: no significant association\n"
            "   - OR ~ 0.96-0.97 across all models (p > 0.05)\n\n"
            "3. Strongest risk factors (Crude OR)\n"
            "   - Impaired fasting glucose: OR ~ 1.56-1.58\n"
            "   - Metabolic syndrome: OR ~ 1.33-1.36\n\n"
            "4. ML model performance\n"
            "   - AUROC range: 0.53-0.58 (modest; synthetic data)\n"
            "   - LR comparable or best among all models"
        ),
    },
    'limitations': {
        'ko': (
            "한계 및 참고사항\n\n"
            "1. 현재 결과는 합성(더미) 데이터 기반\n"
            "   - 파이프라인 검증 목적이며, 임상 해석은 실제 데이터 필요\n\n"
            "2. 실제 NHIC 데이터 적용 시 예상 차이\n"
            "   - 당뇨병 발생률 하락 (현재 31.4% → 실제 5-15% 예상)\n"
            "   - 변수 간 실제 상관관계 반영 시 모델 성능 개선 기대\n"
            "   - 결측치 패턴 상이 가능\n\n"
            "3. 기술적 사항\n"
            "   - Bootstrap: n=50 (논문 제출 시 n=1000 권장)\n"
            "   - SHAP: TreeExplainer (트리모델) / KernelExplainer (ANN)\n"
            "   - Random state: 1004 (전 단계)"
        ),
        'en': (
            "Limitations & Notes\n\n"
            "1. Results are based on synthetic (dummy) data\n"
            "   - Pipeline validation only; real data needed for interpretation\n\n"
            "2. Expected differences with real NHIC data\n"
            "   - Lower incidence (31.4% -> ~5-15%)\n"
            "   - Better model performance expected\n"
            "   - Different missing patterns\n\n"
            "3. Technical notes\n"
            "   - Bootstrap: n=50 (use n=1000 for publication)\n"
            "   - SHAP: TreeExplainer / KernelExplainer (ANN)\n"
            "   - Random state: 1004 (all steps)"
        ),
    },

    # ── Footer ──
    'footer_page': {'ko': '{pg} 페이지', 'en': 'Page {pg}'},
    'footer_date': {'ko': '생성일: {d}', 'en': 'Generated: {d}'},
}


def t(key, lang, **kwargs):
    """Look up a translated string."""
    val = T[key][lang]
    if kwargs and isinstance(val, str):
        return val.format(**kwargs)
    return val


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _font(lang):
    return FONT_KO if lang == 'ko' else FONT_EN


def _new():
    return plt.figure(figsize=(PAGE_W, PAGE_H))


def _footer(fig, pg, lang):
    ff = _font(lang)
    fig.text(0.97, 0.02, t('footer_page', lang, pg=pg),
             ha='right', fontsize=7.5, color='#999999', fontfamily=ff)
    fig.text(0.03, 0.02,
             t('footer_date', lang, d=datetime.now().strftime('%Y-%m-%d')),
             ha='left', fontsize=7.5, color='#999999', fontfamily=ff)


def _title(fig, text, lang, y=0.93):
    fig.text(0.05, y, text, fontsize=19, fontweight='bold',
             color=C_HEADER, fontfamily=_font(lang))


def _subtitle(fig, text, lang, y=0.88):
    fig.text(0.05, y, text, fontsize=11, color='#666666',
             style='italic', fontfamily=_font(lang))


def _table(ax, headers, rows, col_widths=None, fontsize=9,
           highlight=None, row_h=None, lang='ko'):
    ax.axis('off')
    n_rows = len(rows)
    n_cols = len(headers)
    ff = _font(lang)

    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellLoc='center', loc='center', colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    if row_h is None:
        row_h = 1.0 / (n_rows + 1.5)

    for j in range(n_cols):
        c = tbl[0, j]
        c.set_facecolor(C_HEADER)
        c.set_text_props(color='white', fontweight='bold',
                         fontsize=fontsize, fontfamily=ff)
        c.set_edgecolor('white')
        c.set_height(row_h * 1.25)

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            c = tbl[i, j]
            c.set_edgecolor('#CCCCCC')
            c.set_height(row_h)
            c.set_text_props(fontfamily=ff, fontsize=fontsize)
            if highlight and (i - 1) in highlight:
                c.set_facecolor(C_BEST)
            elif i % 2 == 0:
                c.set_facecolor(C_ALT)
            else:
                c.set_facecolor('white')
            if j == 0:
                c.set_text_props(ha='left', fontfamily=ff, fontsize=fontsize)
    return tbl


# ── Data loaders ──

def _load_metrics(target, model):
    p = os.path.join(RESULTS_DIR, target, model, 'metrics.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _load_or(target):
    p = os.path.join(RESULTS_DIR, target, 'tables',
                     'or_analysis_{}.xlsx'.format(target))
    if not os.path.exists(p):
        return None, None
    return (pd.read_excel(p, sheet_name='Crude OR'),
            pd.read_excel(p, sheet_name='Adjusted OR'))


def _load_perf(target):
    p = os.path.join(RESULTS_DIR, target, 'tables', 'model_performance.xlsx')
    return pd.read_excel(p) if os.path.exists(p) else None


def _load_fig(target, name):
    p = os.path.join(RESULTS_DIR, target, 'comparison',
                     '{}.png'.format(name))
    return plt.imread(p) if os.path.exists(p) else None


# ═══════════════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════════════

def pg_title(pdf, pg, lang):
    fig = _new()
    ff = _font(lang)
    fig.text(0.50, 0.62, t('title_main', lang),
             ha='center', va='center', fontsize=26, fontweight='bold',
             color=C_HEADER, linespacing=1.7, fontfamily=ff)
    fig.text(0.50, 0.46, t('title_sub', lang),
             ha='center', va='center', fontsize=15, color='#555555',
             fontfamily=ff)
    lax = fig.add_axes([0.25, 0.41, 0.50, 0.002])
    lax.axhline(y=0, color=C_HEADER, linewidth=2); lax.axis('off')
    fig.text(0.50, 0.34,
             t('title_info', lang, date=datetime.now().strftime('%Y-%m-%d')),
             ha='center', va='top', fontsize=11, color='#666666',
             linespacing=1.6, fontfamily=ff)
    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_overview(pdf, pg, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t('sec_overview', lang), lang)

    fig.text(0.07, 0.87, t('research_q', lang), fontsize=10.5, va='top',
             linespacing=1.5, fontfamily=ff,
             bbox=dict(boxstyle='round,pad=0.6', facecolor=C_WARN,
                       edgecolor='#FFC107', alpha=0.8))

    ax1 = fig.add_axes([0.05, 0.35, 0.42, 0.26])
    ax1.set_title(t('data_overview', lang), fontsize=12, fontweight='bold',
                  color=C_HEADER, loc='left', pad=6, fontfamily=ff)
    _table(ax1, t('data_headers', lang), t('data_rows', lang),
           col_widths=[0.35, 0.65], fontsize=10, lang=lang)

    ax2 = fig.add_axes([0.55, 0.35, 0.42, 0.18])
    ax2.set_title(t('outcome_title', lang), fontsize=12, fontweight='bold',
                  color=C_HEADER, loc='left', pad=6, fontfamily=ff)
    _table(ax2, t('outcome_headers', lang), t('outcome_rows', lang),
           col_widths=[0.35, 0.2, 0.2, 0.25], fontsize=10, lang=lang)

    fig.text(0.05, 0.29, t('methods_title', lang), fontsize=13,
             fontweight='bold', color=C_HEADER, fontfamily=ff)
    y = 0.25
    for label, desc in t('methods', lang):
        fig.text(0.08, y, label, fontsize=10.5, fontweight='bold',
                 color='#333333', fontfamily=ff)
        fig.text(0.20, y, desc, fontsize=9.5, color='#555555',
                 va='top', linespacing=1.4, fontfamily=ff)
        y -= 0.055

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_variables(pdf, pg, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t('sec_variables', lang), lang)

    ax1 = fig.add_axes([0.03, 0.52, 0.94, 0.38])
    ax1.set_title(t('var_numeric_title', lang), fontsize=12,
                  fontweight='bold', color=C_HEADER, loc='left', pad=6,
                  fontfamily=ff)
    _table(ax1, t('var_headers', lang), t('var_numeric', lang),
           col_widths=[0.15, 0.30, 0.10, 0.45], fontsize=9, lang=lang)

    ax2 = fig.add_axes([0.03, 0.03, 0.94, 0.46])
    ax2.set_title(t('var_cat_title', lang), fontsize=12, fontweight='bold',
                  color=C_HEADER, loc='left', pad=6, fontfamily=ff)
    _table(ax2, t('var_headers', lang), t('var_cat', lang),
           col_widths=[0.15, 0.30, 0.12, 0.43], fontsize=8.5,
           highlight={0, 1, 12, 13}, lang=lang)

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_crude_or(pdf, pg, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t('sec_crude', lang), lang)
    _subtitle(fig, t('crude_sub', lang), lang)

    crude_a, _ = _load_or('outA')
    crude_2, _ = _load_or('out2')
    if crude_a is None or crude_2 is None:
        fig.text(0.5, 0.5, 'OR data not available', ha='center',
                 fontsize=14, color='red')
        _footer(fig, pg, lang); pdf.savefig(fig); plt.close(fig); return

    all_vars = crude_a['Variable'].tolist()
    rows = []
    exposure_idx = set()
    for i, var in enumerate(all_vars):
        r2 = crude_2[crude_2['Variable'] == var]
        or_a = str(crude_a.iloc[i]['Crude OR (95% CI)'])
        p_a = str(crude_a.iloc[i]['p-value'])
        or_2 = str(r2['Crude OR (95% CI)'].values[0]) if len(r2) > 0 else '-'
        p_2 = str(r2['p-value'].values[0]) if len(r2) > 0 else '-'
        rows.append([var, or_a, p_a, or_2, p_2])
        if var in ('Cholelithiasis', 'Cholecystectomy'):
            exposure_idx.add(i)

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.80])
    ax.set_title(t('crude_table_title', lang, n=len(rows)),
                 fontsize=12, fontweight='bold', color=C_HEADER,
                 loc='left', pad=6, fontfamily=ff)
    _table(ax, t('crude_headers', lang), rows,
           col_widths=[0.26, 0.22, 0.10, 0.22, 0.10],
           fontsize=8.5, highlight=exposure_idx, lang=lang)
    fig.text(0.05, 0.02, t('crude_note', lang),
             fontsize=8, color='#888888', style='italic', fontfamily=ff)

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_adjusted_or(pdf, pg, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t('sec_adjusted', lang), lang)
    _subtitle(fig, t('adj_sub', lang), lang)

    fig.text(0.07, 0.83, t('adj_desc', lang), fontsize=9.5, va='top',
             linespacing=1.4, fontfamily=ff,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=C_HEADER_LT,
                       edgecolor=C_HEADER, alpha=0.3))

    _, adj_a = _load_or('outA')
    _, adj_2 = _load_or('out2')

    def _get(df, var, col):
        r = df[df['Variable'] == var]
        if len(r) > 0 and col in r.columns:
            return str(r[col].values[0])
        return 'N/A'

    if adj_a is not None and adj_2 is not None:
        chol_rows = []
        chole_rows = []
        for lbl, col in [('Crude', 'Crude OR (95% CI)'),
                         ('Model 1 (Age, Sex)', 'Model 1 (Age, Sex) OR (95% CI)'),
                         ('Model 2 (+Lifestyle)', 'Model 2 (+Lifestyle) OR (95% CI)'),
                         ('Model 3 (+Clinical)', 'Model 3 (+Clinical) OR (95% CI)')]:
            chol_rows.append([lbl, _get(adj_a, 'Cholelithiasis', col),
                              _get(adj_2, 'Cholelithiasis', col)])
            chole_rows.append([lbl, _get(adj_a, 'Cholecystectomy', col),
                               _get(adj_2, 'Cholecystectomy', col)])
    else:
        chol_rows = [['Crude', '1.18 (1.07-1.29)', '1.14 (1.03-1.25)'],
                     ['Model 1', '1.17 (1.07-1.29)', '1.13 (1.03-1.25)'],
                     ['Model 2', '1.18 (1.08-1.30)', '1.14 (1.03-1.25)'],
                     ['Model 3', '1.23 (1.10-1.37)', '1.17 (1.05-1.31)']]
        chole_rows = [['Crude', '0.97 (0.89-1.06)', '0.96 (0.88-1.05)'],
                      ['Model 1', '0.97 (0.89-1.06)', '0.96 (0.87-1.05)'],
                      ['Model 2', '0.96 (0.88-1.05)', '0.97 (0.88-1.06)'],
                      ['Model 3', '0.96 (0.87-1.06)', '0.97 (0.87-1.07)']]

    ax1 = fig.add_axes([0.04, 0.35, 0.44, 0.28])
    ax1.set_title(t('adj_chol_title', lang), fontsize=12, fontweight='bold',
                  color=C_HEADER, loc='left', pad=6, fontfamily=ff)
    _table(ax1, t('adj_headers', lang), chol_rows,
           col_widths=[0.38, 0.31, 0.31], fontsize=10,
           highlight={3}, lang=lang)

    ax2 = fig.add_axes([0.53, 0.35, 0.44, 0.28])
    ax2.set_title(t('adj_chole_title', lang), fontsize=12, fontweight='bold',
                  color=C_HEADER, loc='left', pad=6, fontfamily=ff)
    _table(ax2, t('adj_headers', lang), chole_rows,
           col_widths=[0.38, 0.31, 0.31], fontsize=10, lang=lang)

    fig.text(0.07, 0.28, t('adj_interp', lang), fontsize=10, va='top',
             linespacing=1.5, fontfamily=ff,
             bbox=dict(boxstyle='round,pad=0.6', facecolor=C_WARN,
                       edgecolor='#FFC107', alpha=0.8))

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_performance(pdf, pg, target, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t('sec_perf', lang, target=target), lang)
    _subtitle(fig, t('target_label', lang)[target], lang)

    perf_df = _load_perf(target)

    if perf_df is not None:
        ax1 = fig.add_axes([0.03, 0.42, 0.94, 0.42])
        ax1.set_title(t('perf_table_title', lang), fontsize=12,
                      fontweight='bold', color=C_HEADER, loc='left', pad=6,
                      fontfamily=ff)
        cols = ['Model', 'AUROC', 'AUPRC', 'Accuracy', 'Sensitivity',
                'Specificity', 'PPV', 'NPV', 'F1 Score']
        avail = [c for c in cols if c in perf_df.columns]
        data = [[str(row[c]) for c in avail] for _, row in perf_df.iterrows()]

        best_idx = None
        if 'AUROC' in perf_df.columns:
            for i, val in enumerate(perf_df['AUROC']):
                try:
                    num = float(str(val).split(' ')[0])
                    if best_idx is None or num > float(
                            str(perf_df['AUROC'].iloc[best_idx]).split(' ')[0]):
                        best_idx = i
                except (ValueError, IndexError):
                    pass

        n = len(avail)
        cw = [0.08] + [round(0.92 / (n - 1), 3)] * (n - 1)
        _table(ax1, avail, data, col_widths=cw, fontsize=8,
               highlight={best_idx} if best_idx is not None else None,
               lang=lang)
    else:
        ax1 = fig.add_axes([0.03, 0.42, 0.94, 0.42])
        data = []; best_idx = None; best_val = 0
        for i, mdl in enumerate(MODEL_ORDER):
            m = _load_metrics(target, mdl)
            if m:
                data.append([MODEL_DISPLAY[mdl]] +
                            ['{:.4f}'.format(m[k]) for k in
                             ['auroc','auprc','accuracy','sensitivity',
                              'specificity','ppv','npv','f1']])
                if m['auroc'] > best_val:
                    best_val = m['auroc']; best_idx = i
        _table(ax1,
               ['Model','AUROC','AUPRC','Acc','Sens','Spec','PPV','NPV','F1'],
               data, fontsize=8.5,
               highlight={best_idx} if best_idx is not None else None,
               lang=lang)

    ax2 = fig.add_axes([0.04, 0.06, 0.40, 0.28])
    ax2.set_title(t('auroc_title', lang), fontsize=12, fontweight='bold',
                  color=C_HEADER, loc='left', pad=6, fontfamily=ff)
    cv_rows = []
    for mdl in MODEL_ORDER:
        m = _load_metrics(target, mdl)
        if m:
            cv_rows.append([MODEL_DISPLAY[mdl], '{:.4f}'.format(m['auroc'])])
    if cv_rows:
        _table(ax2, t('auroc_headers', lang), cv_rows,
               col_widths=[0.62, 0.38], fontsize=10, lang=lang)

    fig.text(0.52, 0.31, t('perf_note', lang), fontsize=10, va='top',
             color='#555555', fontfamily=ff,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=C_HEADER_LT,
                       edgecolor=C_HEADER, alpha=0.3))

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_figure(pdf, pg, target, fname, title_key, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t(title_key, lang) + ' \u2014 ' + target, lang)
    _subtitle(fig, t('target_label', lang)[target], lang)

    img = _load_fig(target, fname)
    if img is not None:
        img_h, img_w = img.shape[:2]
        img_ratio = img_w / img_h
        avail_w, avail_h = 0.90, 0.78
        avail_ratio = (avail_w * PAGE_W) / (avail_h * PAGE_H)
        if img_ratio > avail_ratio:
            ax_w = avail_w
            ax_h = avail_w * PAGE_W / (img_ratio * PAGE_H)
        else:
            ax_h = avail_h
            ax_w = avail_h * PAGE_H * img_ratio / PAGE_W
        ax = fig.add_axes([(1.0 - ax_w) / 2, 0.05, ax_w, ax_h])
        ax.imshow(img); ax.axis('off')
    else:
        fig.text(0.5, 0.5, t('fig_missing', lang).format(fname),
                 ha='center', fontsize=14, color='red', fontfamily=ff)

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_two_figures(pdf, pg, target, fn1, fn2, t1, t2, title_key, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t(title_key, lang) + ' \u2014 ' + target, lang)
    _subtitle(fig, t('target_label', lang)[target], lang)

    for i, (fn, tt) in enumerate([(fn1, t1), (fn2, t2)]):
        img = _load_fig(target, fn)
        x0 = 0.02 + i * 0.50
        if img is not None:
            img_h, img_w = img.shape[:2]
            ratio = img_w / img_h
            ax_w = 0.47
            ax_h = ax_w * PAGE_W / (ratio * PAGE_H)
            if ax_h > 0.78:
                ax_h = 0.78
                ax_w = ax_h * PAGE_H * ratio / PAGE_W
            ax = fig.add_axes([x0, 0.05, ax_w, ax_h])
            ax.imshow(img); ax.axis('off')
            ax.set_title(tt, fontsize=11, fontweight='bold',
                         color='#333333', fontfamily=ff)
        else:
            ax = fig.add_axes([x0, 0.05, 0.47, 0.78])
            ax.text(0.5, 0.5, t('fig_missing', lang).format(fn),
                    ha='center', transform=ax.transAxes, color='red',
                    fontfamily=ff)
            ax.axis('off')

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


def pg_findings(pdf, pg, lang):
    fig = _new(); ff = _font(lang)
    _title(fig, t('sec_findings', lang), lang)

    fig.text(0.07, 0.85, t('findings', lang), fontsize=11, va='top',
             linespacing=1.5, fontfamily=ff,
             bbox=dict(boxstyle='round,pad=0.7', facecolor=C_BEST,
                       edgecolor='#4CAF50', alpha=0.5))

    fig.text(0.07, 0.37, t('limitations', lang), fontsize=11, va='top',
             linespacing=1.5, fontfamily=ff,
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFF3E0',
                       edgecolor='#FF9800', alpha=0.5))

    _footer(fig, pg, lang)
    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _generate_one(output_path, lang):
    """Generate a single PDF for the given language."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    label = {'ko': '한글', 'en': 'English'}[lang]
    print("\n  --- {} ({}) ---".format(label, os.path.basename(output_path)))

    plt.rcParams['font.family'] = _font(lang)

    with PdfPages(output_path) as pdf:
        pg = 1
        pg_title(pdf, pg, lang); pg += 1
        pg_overview(pdf, pg, lang); pg += 1
        pg_variables(pdf, pg, lang); pg += 1
        pg_crude_or(pdf, pg, lang); pg += 1
        pg_adjusted_or(pdf, pg, lang); pg += 1
        for tgt in TARGETS:
            pg_performance(pdf, pg, tgt, lang); pg += 1
        for tgt in TARGETS:
            pg_figure(pdf, pg, tgt, 'comparison_roc', 'fig_roc', lang)
            pg += 1
        for tgt in TARGETS:
            pg_two_figures(pdf, pg, tgt,
                           'comparison_pr', 'comparison_calibration',
                           'Precision-Recall', 'Calibration',
                           'fig_pr_cal', lang)
            pg += 1
        for tgt in TARGETS:
            pg_figure(pdf, pg, tgt, 'comparison_shap', 'fig_shap', lang)
            pg += 1
        pg_findings(pdf, pg, lang)

    print("    {} pages -> {}".format(pg, output_path))
    return pg


def create_pdf_report():
    """Generate both Korean and English PDF reports."""
    print("=" * 60)
    print("  PDF 분석 보고서 생성 (한글 + English)")
    print("=" * 60)

    os.makedirs(EXPORT_DIR, exist_ok=True)

    for lang, fname in [('ko', 'analysis_report_ko.pdf'),
                        ('en', 'analysis_report_en.pdf')]:
        out = os.path.join(RESULTS_DIR, fname)
        _generate_one(out, lang)
        exp = os.path.join(EXPORT_DIR, fname)
        shutil.copy2(out, exp)
        print("    -> export: {}".format(exp))

    print("\n  Done!")


if __name__ == '__main__':
    create_pdf_report()
