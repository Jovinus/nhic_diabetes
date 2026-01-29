"""
더미 데이터 생성 스크립트
- 담석증 환자 당뇨병 발생 예측 연구를 위한 더미 데이터
- 결측치 포함 (LDL_CHOL 등은 결측 비율 높음)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 재현 가능성을 위한 시드 설정
np.random.seed(1004)

def generate_dummy_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    변수 설명에 맞는 더미 데이터 생성
    
    Args:
        n_samples: 생성할 샘플 수
    
    Returns:
        pd.DataFrame: 생성된 더미 데이터
    """
    
    # 개인식별 정보
    data = {
        'INDI_DSCM_NO': [f'ID_{str(i).zfill(6)}' for i in range(n_samples)],
        'yy': np.random.choice([2010, 2011, 2012, 2013, 2014, 2015], n_samples),
        'gender': np.random.choice([0, 1], n_samples),  # 0: 남성, 1: 여성
        'age': np.random.randint(30, 80, n_samples),
    }
    
    # 날짜 관련 변수
    base_date = datetime(2010, 1, 1)
    index_dates = [base_date + timedelta(days=int(np.random.randint(0, 3650))) for _ in range(n_samples)]
    data['indexDate'] = index_dates
    
    # 사망일 - 약 5%만 사망
    dth_dates = []
    for idx in index_dates:
        if np.random.random() < 0.05:
            dth_dates.append(idx + timedelta(days=int(np.random.randint(30, 3000))))
        else:
            dth_dates.append(pd.NaT)
    data['dthDate'] = dth_dates
    
    # 담낭제거수술일 - 약 40%가 수술
    act_dates = []
    for idx in index_dates:
        if np.random.random() < 0.40:
            act_dates.append(idx + timedelta(days=int(np.random.randint(1, 365))))
        else:
            act_dates.append(pd.NaT)
    data['actDate'] = act_dates
    
    # 검진일
    gj_dates = []
    for idx in index_dates:
        offset = np.random.randint(-180, 180)  # indexDate 전후 6개월 내
        gj_dates.append(idx + timedelta(days=offset))
    data['gjDate'] = gj_dates
    
    # 건강검진 지표 (연속형 변수, 일부 결측치 포함)
    # BMI: 정규분포 18-35
    bmi = np.random.normal(25, 4, n_samples)
    bmi = np.clip(bmi, 15, 45)
    bmi[np.random.choice(n_samples, int(n_samples * 0.03), replace=False)] = np.nan  # 3% 결측
    data['BMI'] = bmi
    
    # SBP: 정규분포 90-180
    sbp = np.random.normal(125, 18, n_samples)
    sbp = np.clip(sbp, 80, 200)
    sbp[np.random.choice(n_samples, int(n_samples * 0.02), replace=False)] = np.nan
    data['SBP'] = sbp
    
    # DBP: 정규분포 60-100
    dbp = np.random.normal(80, 12, n_samples)
    dbp = np.clip(dbp, 50, 120)
    dbp[np.random.choice(n_samples, int(n_samples * 0.02), replace=False)] = np.nan
    data['DBP'] = dbp
    
    # 단백뇨 (1-6)
    g1e_urn_prot = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    g1e_urn_prot = g1e_urn_prot.astype(float)
    g1e_urn_prot[np.random.choice(n_samples, int(n_samples * 0.05), replace=False)] = np.nan
    data['G1E_URN_PROT'] = g1e_urn_prot
    
    # FBS (혈당): 70-200
    fbs = np.random.normal(100, 25, n_samples)
    fbs = np.clip(fbs, 60, 300)
    fbs[np.random.choice(n_samples, int(n_samples * 0.03), replace=False)] = np.nan
    data['FBS'] = fbs
    
    # TOT_CHOL: 총 콜레스테롤 130-280
    tot_chol = np.random.normal(200, 40, n_samples)
    tot_chol = np.clip(tot_chol, 100, 350)
    tot_chol[np.random.choice(n_samples, int(n_samples * 0.03), replace=False)] = np.nan
    data['TOT_CHOL'] = tot_chol
    
    # WAIST: 허리둘레 60-120
    waist = np.random.normal(85, 12, n_samples)
    waist = np.clip(waist, 55, 130)
    waist[np.random.choice(n_samples, int(n_samples * 0.04), replace=False)] = np.nan
    data['WAIST'] = waist
    
    # TG: 중성지방 (right-skewed)
    tg = np.random.lognormal(4.8, 0.5, n_samples)
    tg = np.clip(tg, 30, 800)
    tg[np.random.choice(n_samples, int(n_samples * 0.04), replace=False)] = np.nan
    data['TG'] = tg
    
    # HDL_CHOL: HDL 콜레스테롤 30-100
    hdl = np.random.normal(55, 15, n_samples)
    hdl = np.clip(hdl, 20, 120)
    hdl[np.random.choice(n_samples, int(n_samples * 0.04), replace=False)] = np.nan
    data['HDL_CHOL'] = hdl
    
    # Creatinine: 0.5-1.5
    creatinine = np.random.normal(1.0, 0.3, n_samples)
    creatinine = np.clip(creatinine, 0.3, 3.0)
    creatinine[np.random.choice(n_samples, int(n_samples * 0.03), replace=False)] = np.nan
    data['Creatinine'] = creatinine
    
    # LDL_CHOL: 결측치가 많음 (약 30%)
    ldl = np.random.normal(120, 35, n_samples)
    ldl = np.clip(ldl, 40, 250)
    ldl[np.random.choice(n_samples, int(n_samples * 0.30), replace=False)] = np.nan
    data['LDL_CHOL'] = ldl
    
    # 생활습관 변수
    # 주당 격렬한 운동일수 (0-7)
    data['Q_PA_VD'] = np.random.choice(range(8), n_samples, p=[0.4, 0.2, 0.15, 0.1, 0.06, 0.04, 0.03, 0.02])
    
    # 주당 중간정도 운동일수 (0-7)
    data['Q_PA_MD'] = np.random.choice(range(8), n_samples, p=[0.3, 0.2, 0.18, 0.12, 0.08, 0.05, 0.04, 0.03])
    
    # 흡연상태 (0: 비흡연, 1: 끊음, 2: 현재흡연)
    data['smoking'] = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.2, 0.3])
    
    # 주당 음주일수 (0-7)
    data['drinkDay'] = np.random.choice(range(8), n_samples, p=[0.35, 0.15, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02])
    
    # 주2일 이상 음주 (0: no, 1: yes)
    data['drink'] = (np.array(data['drinkDay']) >= 2).astype(int)
    
    # 주3일 이상 운동 (0: no, 1: yes)
    total_exercise = np.array(data['Q_PA_VD']) + np.array(data['Q_PA_MD'])
    data['training'] = (total_exercise >= 3).astype(int)
    
    # 단백뇨 유무 (0: 정상, 1: trace/+1, 2: +2~)
    proteinuria = np.zeros(n_samples)
    prot_raw = np.array(data['G1E_URN_PROT'])
    proteinuria[np.isin(prot_raw, [2, 3])] = 1
    proteinuria[np.isin(prot_raw, [4, 5, 6])] = 2
    proteinuria[np.isnan(prot_raw)] = np.nan
    data['proteinUria'] = proteinuria
    
    # diff: indexDate~gjDate 사이 기간 (일)
    data['diff'] = [(gj - idx).days if pd.notna(gj) else np.nan 
                    for gj, idx in zip(data['gjDate'], data['indexDate'])]
    
    # 진단 관련 변수
    # diag: 담석증 진단유무 (0: no, 1: yes) - 코호트 특성상 대부분 yes
    data['diag'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # act: 담낭제거수술여부 (0: no, 1: yes)
    data['act'] = [0 if pd.isna(d) else 1 for d in data['actDate']]
    
    # 동반질환
    data['co_HLD'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 고지혈증
    data['co_HTN'] = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])  # 고혈압
    data['co_fattyLiver'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 지방간
    data['co_Impaird'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # impaired glucose
    
    # BMIG: BMI 그룹 (0: BMI<25, 1: 25<=BMI<30, 2: BMI>=30)
    bmig = np.zeros(n_samples)
    bmi_arr = np.array(data['BMI'])
    bmig[(bmi_arr >= 25) & (bmi_arr < 30)] = 1
    bmig[bmi_arr >= 30] = 2
    bmig[np.isnan(bmi_arr)] = np.nan
    data['BMIG'] = bmig
    
    # metS: metabolic syndrome 유무
    # 간단히 BMI>=25, TG>=150, HDL<40(남)/50(여), SBP>=130 or DBP>=85 중 3개 이상
    met_count = np.zeros(n_samples)
    met_count += (bmi_arr >= 25).astype(int)
    met_count += (np.array(data['TG']) >= 150).astype(int)
    hdl_arr = np.array(data['HDL_CHOL'])
    gender_arr = np.array(data['gender'])
    met_count += ((gender_arr == 0) & (hdl_arr < 40) | (gender_arr == 1) & (hdl_arr < 50)).astype(int)
    met_count += ((np.array(data['SBP']) >= 130) | (np.array(data['DBP']) >= 85)).astype(int)
    met_count += (np.array(data['FBS']) >= 100).astype(int)
    data['metS'] = (met_count >= 3).astype(int)
    
    # iid: 개인식별숫자 (의미없음)
    data['iid'] = range(n_samples)
    
    # group: 진단Group
    # 1: 담석증yes & 담석제거no, 2: 담석증yes & 담석제거yes, 3: 담석증no
    group = np.zeros(n_samples, dtype=int)
    diag_arr = np.array(data['diag'])
    act_arr = np.array(data['act'])
    group[(diag_arr == 1) & (act_arr == 0)] = 1
    group[(diag_arr == 1) & (act_arr == 1)] = 2
    group[diag_arr == 0] = 3
    data['group'] = group
    
    # 결과 변수 (당뇨병 발생)
    # outA: 당뇨병 발생 유무 (0: no, 1: yes)
    # 위험인자 기반으로 확률적으로 생성
    base_prob = 0.15  # 기본 발생률
    risk_score = np.zeros(n_samples)
    risk_score += 0.02 * (np.array(data['age']) - 40) / 10  # 연령 증가
    risk_score += 0.05 * (bmi_arr > 25)  # 비만
    risk_score += 0.05 * (np.array(data['FBS']) > 100)  # 공복혈당 이상
    risk_score += 0.03 * (diag_arr == 1)  # 담석증
    risk_score += 0.03 * np.array(data['co_HLD'])  # 고지혈증
    risk_score += 0.03 * np.array(data['co_HTN'])  # 고혈압
    risk_score += 0.04 * np.array(data['co_fattyLiver'])  # 지방간
    risk_score += 0.10 * np.array(data['co_Impaird'])  # impaired glucose
    risk_score += 0.03 * np.array(data['metS'])  # 대사증후군
    
    prob = np.clip(base_prob + risk_score, 0, 0.8)
    data['outA'] = (np.random.random(n_samples) < prob).astype(int)
    
    # out2: 2형 당뇨병 발생 (outA와 유사하게)
    data['out2'] = (np.random.random(n_samples) < prob * 0.9).astype(int)
    
    # treatDateA: 당뇨병 진단일
    treat_dates_a = []
    for i, (idx, out) in enumerate(zip(data['indexDate'], data['outA'])):
        if out == 1:
            treat_dates_a.append(idx + timedelta(days=int(np.random.randint(30, 2500))))
        else:
            treat_dates_a.append(datetime(2019, 12, 31))  # outA=0이면 고정
    data['treatDateA'] = treat_dates_a
    
    # treatDate2: 2형 당뇨병 진단일
    treat_dates_2 = []
    for i, (idx, out) in enumerate(zip(data['indexDate'], data['out2'])):
        if out == 1:
            treat_dates_2.append(idx + timedelta(days=int(np.random.randint(30, 2500))))
        else:
            treat_dates_2.append(datetime(2019, 12, 31))  # out2=0이면 고정
    data['treatDate2'] = treat_dates_2
    
    # fuYA: 추적관찰기간 (년)
    data['fuYA'] = [(t - idx).days / 365.25 for t, idx in zip(data['treatDateA'], data['indexDate'])]
    
    # fuY2: 추적관찰기간 (년)
    data['fuY2'] = [(t - idx).days / 365.25 for t, idx in zip(data['treatDate2'], data['indexDate'])]
    
    return pd.DataFrame(data)


def main():
    print("=" * 60)
    print("더미 데이터 생성 시작")
    print("=" * 60)
    
    # 데이터 생성
    df = generate_dummy_data(n_samples=5000)
    
    # 데이터 저장
    output_path = '../data/dummy_diabetes_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 데이터 저장 완료: {output_path}")
    
    # 데이터 요약 출력
    print("\n" + "=" * 60)
    print("데이터 요약")
    print("=" * 60)
    print(f"총 샘플 수: {len(df)}")
    print(f"변수 수: {len(df.columns)}")
    
    print("\n📊 결측치 현황:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        '결측수': missing[missing > 0],
        '결측률(%)': missing_pct[missing > 0]
    })
    print(missing_df.to_string())
    
    print("\n📊 결과 변수 분포:")
    print(f"outA (당뇨병 발생): {df['outA'].value_counts().to_dict()}")
    print(f"out2 (2형 당뇨병 발생): {df['out2'].value_counts().to_dict()}")
    
    print("\n📊 그룹 분포:")
    print(df['group'].value_counts().to_dict())
    
    print("\n📊 연속형 변수 기술통계:")
    numeric_cols = ['age', 'BMI', 'SBP', 'DBP', 'FBS', 'TOT_CHOL', 'WAIST', 
                    'TG', 'HDL_CHOL', 'Creatinine', 'LDL_CHOL']
    print(df[numeric_cols].describe().round(2).to_string())


if __name__ == '__main__':
    main()
