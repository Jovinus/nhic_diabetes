#!/bin/bash
# =============================================================================
# 폐쇄망 환경에서 오프라인 패키지 설치 스크립트
# =============================================================================
# 
# 대상 환경: Python 3.8.16 (폐쇄망)
#
# 사용법:
#   1. 인터넷이 되는 환경에서 download_packages.sh 실행
#   2. packages/ 폴더와 함께 프로젝트를 폐쇄망으로 복사
#   3. 이 스크립트 실행: ./scripts/install_offline.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PACKAGES_DIR="$PROJECT_DIR/packages"
VENV_DIR="$PROJECT_DIR/venv"

# 필요한 Python 버전
REQUIRED_PYTHON_VERSION="3.8"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}폐쇄망 오프라인 패키지 설치${NC}"
echo -e "${GREEN}============================================================${NC}"

# Python 버전 확인
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f1,2)

echo -e ""
echo -e "${BLUE}환경 정보:${NC}"
echo -e "  - Python 버전: $PYTHON_VERSION"
echo -e "  - 필요 버전: Python ${REQUIRED_PYTHON_VERSION}.x"

if [ "$PYTHON_MAJOR_MINOR" != "$REQUIRED_PYTHON_VERSION" ]; then
    echo -e ""
    echo -e "${RED}에러: Python 버전이 맞지 않습니다.${NC}"
    echo -e "  현재: Python $PYTHON_VERSION"
    echo -e "  필요: Python ${REQUIRED_PYTHON_VERSION}.x"
    echo -e ""
    echo -e "올바른 Python 버전을 사용해주세요."
    exit 1
fi

echo -e "${GREEN}✓ Python 버전 확인 완료${NC}"

# 패키지 디렉토리 확인
if [ ! -d "$PACKAGES_DIR" ]; then
    echo -e ""
    echo -e "${RED}에러: packages/ 디렉토리가 없습니다.${NC}"
    echo -e "먼저 인터넷 환경에서 download_packages.sh를 실행하여"
    echo -e "패키지를 다운로드한 후 폐쇄망으로 복사하세요."
    exit 1
fi

PACKAGE_COUNT=$(ls -1 "$PACKAGES_DIR"/*.whl 2>/dev/null | wc -l)
if [ "$PACKAGE_COUNT" -eq 0 ]; then
    echo -e ""
    echo -e "${RED}에러: packages/ 디렉토리에 패키지가 없습니다.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 패키지 디렉토리 확인 완료 (${PACKAGE_COUNT}개 패키지)${NC}"

# 가상환경 생성
echo -e ""
echo -e "${YELLOW}가상환경 생성 중...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "  기존 가상환경 삭제..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
echo -e "${GREEN}✓ 가상환경 생성 완료: $VENV_DIR${NC}"

# 가상환경 활성화
source "$VENV_DIR/bin/activate"

# pip 업그레이드 (오프라인)
echo -e ""
echo -e "${YELLOW}pip 업그레이드 시도...${NC}"
if ls "$PACKAGES_DIR"/pip-*.whl 1> /dev/null 2>&1; then
    pip install --no-index --find-links="$PACKAGES_DIR" --upgrade pip 2>/dev/null || true
fi

# setuptools, wheel 설치
if ls "$PACKAGES_DIR"/setuptools-*.whl 1> /dev/null 2>&1; then
    pip install --no-index --find-links="$PACKAGES_DIR" setuptools wheel 2>/dev/null || true
fi

# 모든 패키지 설치
echo -e ""
echo -e "${YELLOW}패키지 설치 중...${NC}"
pip install \
    --no-index \
    --find-links="$PACKAGES_DIR" \
    -r "$PROJECT_DIR/requirements.txt"

# 설치 확인
echo -e ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}설치 완료!${NC}"
echo -e "${GREEN}============================================================${NC}"

echo -e ""
echo -e "${BLUE}설치된 주요 패키지:${NC}"
pip show numpy pandas scikit-learn xgboost lightgbm shap optuna matplotlib 2>/dev/null | grep -E "^(Name|Version):" | paste - - | column -t || pip list

echo -e ""
echo -e "${YELLOW}사용 방법:${NC}"
echo -e ""
echo -e "  ${GREEN}1. 가상환경 활성화:${NC}"
echo -e "     source venv/bin/activate"
echo -e ""
echo -e "  ${GREEN}2. 더미 데이터 생성 (선택):${NC}"
echo -e "     cd code && python make_dummy.py"
echo -e ""
echo -e "  ${GREEN}3. 전처리 실행:${NC}"
echo -e "     cd code && python preprocessing.py --data ../data/dummy_diabetes_data.csv"
echo -e ""
echo -e "  ${GREEN}4. 모델 학습:${NC}"
echo -e "     cd code && python train.py --model xgboost"
echo -e ""
echo -e "  ${GREEN}5. 모델 평가:${NC}"
echo -e "     cd code && python evaluate.py --model ../models/xgboost_model.json --shap"
echo -e ""
echo -e "  ${GREEN}또는 전체 파이프라인 실행:${NC}"
echo -e "     ./scripts/run_pipeline.sh"

echo -e ""
echo -e "${GREEN}설치가 완료되었습니다!${NC}"
