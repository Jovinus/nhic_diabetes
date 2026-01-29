#!/bin/bash
# =============================================================================
# 기존 가상환경에 오프라인 패키지 설치 스크립트
# =============================================================================
# 
# 사용법:
#   1. 설치하고자 하는 가상환경을 먼저 활성화
#      예: source /path/to/your/venv/bin/activate
#          또는 conda activate your_env
#   2. 이 스크립트 실행: ./scripts/install_to_existing_env.sh
#
# 주의: 새로운 가상환경을 만들지 않고, 현재 활성화된 환경에 설치합니다.
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PACKAGES_DIR="$PROJECT_DIR/packages"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}기존 가상환경에 오프라인 패키지 설치${NC}"
echo -e "${GREEN}============================================================${NC}"

# 가상환경 활성화 확인
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
    echo -e ""
    echo -e "${RED}경고: 가상환경이 활성화되어 있지 않습니다.${NC}"
    echo -e ""
    echo -e "먼저 가상환경을 활성화하세요:"
    echo -e "  ${YELLOW}source /path/to/venv/bin/activate${NC}"
    echo -e "  또는"
    echo -e "  ${YELLOW}conda activate your_env${NC}"
    echo -e ""
    read -p "그래도 시스템 Python에 설치하시겠습니까? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo -e "설치를 취소합니다."
        exit 1
    fi
fi

# Python 정보 출력
PYTHON_PATH=$(which python)
PYTHON_VERSION=$(python --version 2>&1)
PIP_PATH=$(which pip)

echo -e ""
echo -e "${BLUE}현재 환경 정보:${NC}"
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "  - 가상환경 (venv): $VIRTUAL_ENV"
elif [ -n "$CONDA_PREFIX" ]; then
    echo -e "  - 가상환경 (conda): $CONDA_PREFIX"
else
    echo -e "  - 가상환경: ${RED}없음 (시스템 Python)${NC}"
fi
echo -e "  - Python 경로: $PYTHON_PATH"
echo -e "  - Python 버전: $PYTHON_VERSION"
echo -e "  - pip 경로: $PIP_PATH"

# 패키지 디렉토리 확인
echo -e ""
if [ ! -d "$PACKAGES_DIR" ]; then
    echo -e "${RED}에러: packages/ 디렉토리가 없습니다.${NC}"
    echo -e "먼저 인터넷 환경에서 download_packages.sh를 실행하여"
    echo -e "패키지를 다운로드하세요."
    exit 1
fi

PACKAGE_COUNT=$(ls -1 "$PACKAGES_DIR"/*.whl 2>/dev/null | wc -l)
if [ "$PACKAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}에러: packages/ 디렉토리에 .whl 파일이 없습니다.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 패키지 디렉토리 확인: ${PACKAGE_COUNT}개 패키지${NC}"

# 설치 확인
echo -e ""
echo -e "${YELLOW}다음 환경에 패키지를 설치합니다:${NC}"
echo -e "  $PYTHON_PATH"
echo -e ""
read -p "계속하시겠습니까? (Y/n): " confirm
if [ "$confirm" == "n" ] || [ "$confirm" == "N" ]; then
    echo -e "설치를 취소합니다."
    exit 0
fi

# pip 업그레이드 시도
echo -e ""
echo -e "${YELLOW}pip 업그레이드 시도...${NC}"
if ls "$PACKAGES_DIR"/pip-*.whl 1> /dev/null 2>&1; then
    pip install --no-index --find-links="$PACKAGES_DIR" --upgrade pip 2>/dev/null || true
    echo -e "${GREEN}✓ pip 업그레이드 완료${NC}"
else
    echo -e "  pip 패키지 없음 - 건너뜀"
fi

# setuptools, wheel 설치
echo -e ""
echo -e "${YELLOW}setuptools, wheel 설치...${NC}"
if ls "$PACKAGES_DIR"/setuptools-*.whl 1> /dev/null 2>&1; then
    pip install --no-index --find-links="$PACKAGES_DIR" setuptools wheel 2>/dev/null || true
    echo -e "${GREEN}✓ setuptools, wheel 설치 완료${NC}"
else
    echo -e "  setuptools 패키지 없음 - 건너뜀"
fi

# 패키지 설치
echo -e ""
echo -e "${YELLOW}requirements.txt 패키지 설치 중...${NC}"
echo -e ""

pip install \
    --no-index \
    --find-links="$PACKAGES_DIR" \
    -r "$REQUIREMENTS"

# 설치 확인
echo -e ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}✅ 설치 완료!${NC}"
echo -e "${GREEN}============================================================${NC}"

echo -e ""
echo -e "${BLUE}설치된 주요 패키지:${NC}"
pip show numpy pandas scikit-learn xgboost catboost shap 2>/dev/null | grep -E "^(Name|Version):" | paste - - || pip list | head -20

echo -e ""
echo -e "${YELLOW}사용 방법:${NC}"
echo -e ""
echo -e "  ${GREEN}1. 더미 데이터 생성 (선택):${NC}"
echo -e "     cd code && python make_dummy.py"
echo -e ""
echo -e "  ${GREEN}2. 전처리 실행:${NC}"
echo -e "     cd code && python preprocessing.py --data ../data/your_data.csv"
echo -e ""
echo -e "  ${GREEN}3. 모델 학습:${NC}"
echo -e "     cd code && python train_gridsearch.py --small-grid"
echo -e ""
echo -e "  ${GREEN}4. 모델 평가:${NC}"
echo -e "     cd code && python evaluate.py --model ../models/xgboost_best_model.json --shap"
echo -e ""
echo -e "  ${GREEN}또는 전체 파이프라인:${NC}"
echo -e "     ./scripts/run_pipeline.sh --skip-dummy --small-grid"
echo -e ""
echo -e "${GREEN}설치가 완료되었습니다!${NC}"
