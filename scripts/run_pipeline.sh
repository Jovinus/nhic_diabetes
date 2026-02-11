#!/bin/bash
# =============================================================================
# 전체 파이프라인 실행 스크립트 (다중 타겟 지원)
# =============================================================================
# 
# 1. 더미 데이터 생성
# 2. 각 타겟(outA, out2)에 대해:
#    - Baseline Characteristics (Table 1) 생성
#    - 전처리 (Missing Indicator 포함)
#    - 모델 학습 (GridSearchCV)
#    - 모델 평가 및 SHAP 분석
#    - 논문용 성능 테이블 생성
#    - 모델 비교 Figure 생성
#
# 사용법:
#   ./scripts/run_pipeline.sh [options]
#
# 옵션:
#   --skip-dummy     더미 데이터 생성 건너뛰기
#   --small-grid     축소 파라미터 그리드 사용 (빠른 테스트용)
#   --n-bootstrap N  Bootstrap 반복 횟수 (기본: 1000)
#   --targets T      타겟 지정 (기본: "outA out2", 예: --targets "outA")
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 기본값
SKIP_DUMMY=false
SMALL_GRID=""
N_BOOTSTRAP=1000
TARGETS="outA out2"
MODELS="decision_tree random_forest xgboost lightgbm ann"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dummy)
            SKIP_DUMMY=true
            shift
            ;;
        --small-grid)
            SMALL_GRID="--small-grid"
            shift
            ;;
        --n-bootstrap)
            N_BOOTSTRAP="$2"
            shift 2
            ;;
        --targets)
            TARGETS="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}🏥 당뇨병 예측 모델 파이프라인 (다중 타겟)${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e ""
echo -e "${BLUE}설정:${NC}"
echo -e "  - 더미 데이터 생성: $([ "$SKIP_DUMMY" = true ] && echo '건너뛰기' || echo '실행')"
echo -e "  - 파라미터 그리드: $([ -n "$SMALL_GRID" ] && echo '축소' || echo '전체')"
echo -e "  - Bootstrap 반복: ${N_BOOTSTRAP}"
echo -e "  - 타겟 변수: ${TARGETS}"
echo -e "  - 모델: ${MODELS}"
echo -e ""

# 가상환경 확인 및 활성화
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo -e "${GREEN}✓ 가상환경 활성화${NC}"
else
    echo -e "${YELLOW}⚠️ 가상환경이 없습니다. 시스템 Python 사용${NC}"
fi

cd "$CODE_DIR"

# =============================================================================
# Step 1: 더미 데이터 생성
# =============================================================================
if [ "$SKIP_DUMMY" = false ]; then
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[Step 1] 더미 데이터 생성${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    python make_dummy.py
else
    echo -e "\n${YELLOW}[Step 1] 더미 데이터 생성 - 건너뛰기${NC}"
fi

# =============================================================================
# 각 타겟에 대해 파이프라인 실행
# =============================================================================
for TARGET in $TARGETS; do
    echo -e "\n${MAGENTA}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║  🎯 타겟: ${TARGET}                                          ║${NC}"
    echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════╝${NC}"
    
    # 타겟별 디렉토리 설정
    PROCESSED_DIR="../data/processed/${TARGET}"
    MODELS_DIR="../models/${TARGET}"
    RESULTS_DIR="../results/${TARGET}"
    TABLES_DIR="../results/${TARGET}/tables"
    COMPARISON_DIR="../results/${TARGET}/comparison"
    
    # 디렉토리 생성
    mkdir -p "$PROCESSED_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$TABLES_DIR" "$COMPARISON_DIR"
    
    # =========================================================================
    # Step 2: Baseline Characteristics (Table 1)
    # =========================================================================
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[${TARGET}] Step 2: Baseline Characteristics 테이블 생성${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    python create_table1.py \
        --data ../data/dummy_diabetes_data.csv \
        --output "$TABLES_DIR" \
        --target "$TARGET"
    
    # =========================================================================
    # Step 3: 전처리 (Missing Indicator 포함)
    # =========================================================================
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[${TARGET}] Step 3: 데이터 전처리 (Missing Indicator 포함)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    python preprocessing.py \
        --data ../data/dummy_diabetes_data.csv \
        --output "$PROCESSED_DIR" \
        --target "$TARGET" \
        --add-missing-indicator \
        --missing-threshold 0.05
    
    # =========================================================================
    # Step 4: 모델 학습 (GridSearchCV)
    # =========================================================================
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[${TARGET}] Step 4: 모델 학습 (GridSearchCV)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    python train_gridsearch.py \
        --data-dir "$PROCESSED_DIR" \
        --output "$MODELS_DIR" \
        --models $MODELS \
        --cv 5 \
        --scoring roc_auc \
        $SMALL_GRID
    
    # =========================================================================
    # Step 5: 모델 평가 및 SHAP 분석
    # =========================================================================
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[${TARGET}] Step 5: 모델 평가 및 SHAP 분석${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # 학습 순서대로 모델 평가 (glob은 filesystem order라 순서 보장 안됨)
    for model_name in $MODELS; do
        model_file="${MODELS_DIR}/${model_name}_best_model.pkl"
        if [ -f "$model_file" ]; then
            echo -e "\n${BLUE}📊 ${model_name} 평가 중...${NC}"
            python evaluate.py \
                --model "$model_file" \
                --data-dir "$PROCESSED_DIR" \
                --output "$RESULTS_DIR" \
                --shap
        fi
    done
    
    # =========================================================================
    # Step 6: 논문용 성능 비교 테이블
    # =========================================================================
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[${TARGET}] Step 6: 논문용 성능 비교 테이블 생성 (Bootstrap CI)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    python create_performance_table.py \
        --models-dir "$MODELS_DIR" \
        --data-dir "$PROCESSED_DIR" \
        --output "${TABLES_DIR}/model_performance.xlsx" \
        --n-bootstrap $N_BOOTSTRAP
    
    # =========================================================================
    # Step 7: 모델 비교 Figure 생성 (논문용)
    # =========================================================================
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[${TARGET}] Step 7: 모델 비교 Figure 생성 (논문용)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    python create_comparison_figures.py \
        --models-dir "$MODELS_DIR" \
        --data-dir "$PROCESSED_DIR" \
        --output "$COMPARISON_DIR"
    
    echo -e "\n${GREEN}✅ 타겟 ${TARGET} 완료!${NC}"
done

# =============================================================================
# 완료
# =============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}✅ 전체 파이프라인 완료!${NC}"
echo -e "${GREEN}============================================================${NC}"

echo -e "\n${BLUE}📁 결과물 위치:${NC}"
echo -e ""
echo -e "  ${YELLOW}데이터:${NC}"
echo -e "    - 원본 데이터: data/dummy_diabetes_data.csv"
for TARGET in $TARGETS; do
echo -e "    - 전처리 데이터 (${TARGET}): data/processed/${TARGET}/"
done
echo -e ""
echo -e "  ${YELLOW}모델:${NC}"
for TARGET in $TARGETS; do
echo -e "    - 학습된 모델 (${TARGET}): models/${TARGET}/*_best_model.*"
done
echo -e ""
echo -e "  ${YELLOW}결과:${NC}"
for TARGET in $TARGETS; do
echo -e "    - 개별 평가 (${TARGET}): results/${TARGET}/{model_name}/"
echo -e "    - 모델 비교 (${TARGET}): results/${TARGET}/comparison/"
done
echo -e ""
echo -e "  ${YELLOW}테이블:${NC}"
for TARGET in $TARGETS; do
echo -e "    - Table 1 (${TARGET}): results/${TARGET}/tables/table1_*.xlsx"
echo -e "    - 성능 비교 (${TARGET}): results/${TARGET}/tables/model_performance.xlsx"
done
echo -e ""
echo -e "${GREEN}완료!${NC}"
