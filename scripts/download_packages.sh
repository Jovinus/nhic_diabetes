#!/bin/bash
# =============================================================================
# 폐쇄망 환경을 위한 오프라인 패키지 다운로드 스크립트
# =============================================================================
# 
# 대상 환경: Python 3.8.16 (폐쇄망)
#
# 사용법:
#   1. 인터넷이 되는 환경에서 이 스크립트 실행
#   2. 생성된 packages/ 폴더를 폐쇄망으로 반입
#   3. 폐쇄망에서 install_offline.sh 실행
#
# 옵션:
#   --platform <linux|macos|windows>  대상 플랫폼 지정 (기본: 현재 시스템)
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PACKAGES_DIR="$PROJECT_DIR/packages"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

# 대상 Python 버전 (폐쇄망 환경)
TARGET_PYTHON_VERSION="3.8"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 플랫폼 감지 함수
detect_platform() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        MINGW*|MSYS*|CYGWIN*)    echo "windows";;
        *)          echo "linux";;
    esac
}

# 인자 파싱
PLATFORM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            exit 1
            ;;
    esac
done

# 플랫폼 설정
if [ -z "$PLATFORM" ]; then
    PLATFORM=$(detect_platform)
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}폐쇄망용 오프라인 패키지 다운로드${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e ""
echo -e "${BLUE}대상 환경 정보:${NC}"
echo -e "  - Python 버전: ${TARGET_PYTHON_VERSION}"
echo -e "  - 플랫폼: ${PLATFORM}"
echo -e ""

# 현재 Python 버전 확인
CURRENT_PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo -e "${YELLOW}현재 Python 버전: $CURRENT_PYTHON_VERSION${NC}"

if [ "$CURRENT_PYTHON_VERSION" != "$TARGET_PYTHON_VERSION" ]; then
    echo -e "${YELLOW}⚠️  경고: 현재 Python 버전($CURRENT_PYTHON_VERSION)이 대상 버전($TARGET_PYTHON_VERSION)과 다릅니다.${NC}"
    echo -e "${YELLOW}   대상 버전(Python $TARGET_PYTHON_VERSION)에 맞는 패키지를 다운로드합니다.${NC}"
fi

# 패키지 디렉토리 생성
mkdir -p "$PACKAGES_DIR"

# 기존 패키지 삭제
echo -e "\n${YELLOW}기존 패키지 정리 중...${NC}"
rm -f "$PACKAGES_DIR"/*.whl "$PACKAGES_DIR"/*.tar.gz 2>/dev/null || true

# 플랫폼별 설정
case "$PLATFORM" in
    linux)
        PLATFORM_TAG="manylinux"
        PIP_PLATFORM="manylinux2014_x86_64"
        ;;
    macos)
        PLATFORM_TAG="macosx"
        PIP_PLATFORM="macosx_10_9_x86_64"
        ;;
    windows)
        PLATFORM_TAG="win"
        PIP_PLATFORM="win_amd64"
        ;;
esac

# 패키지 다운로드
echo -e "\n${YELLOW}패키지 다운로드 중...${NC}"
echo -e "  대상: Python ${TARGET_PYTHON_VERSION}, 플랫폼: ${PIP_PLATFORM}"

# pip download 실행 - Python 버전 및 플랫폼 지정
echo -e "\n${YELLOW}[1/3] Binary wheel 다운로드...${NC}"
pip download \
    -r "$REQUIREMENTS" \
    -d "$PACKAGES_DIR" \
    --python-version "${TARGET_PYTHON_VERSION}" \
    --platform "${PIP_PLATFORM}" \
    --implementation cp \
    --abi cp38 \
    --only-binary=:all: \
    --no-cache-dir \
    2>&1 | tee "$PACKAGES_DIR/download.log" || true

# manylinux2010도 시도 (일부 구버전 패키지용)
echo -e "\n${YELLOW}[1-2/3] manylinux2010 wheel 다운로드 시도...${NC}"
pip download \
    -r "$REQUIREMENTS" \
    -d "$PACKAGES_DIR" \
    --python-version "${TARGET_PYTHON_VERSION}" \
    --platform "manylinux2010_x86_64" \
    --implementation cp \
    --abi cp38 \
    --only-binary=:all: \
    --no-cache-dir \
    2>&1 | tee -a "$PACKAGES_DIR/download.log" || true

# any 플랫폼 (pure python 패키지)
echo -e "\n${YELLOW}[1-3/3] Pure Python wheel 다운로드...${NC}"
pip download \
    -r "$REQUIREMENTS" \
    -d "$PACKAGES_DIR" \
    --python-version "${TARGET_PYTHON_VERSION}" \
    --platform "any" \
    --implementation cp \
    --abi none \
    --only-binary=:all: \
    --no-cache-dir \
    2>&1 | tee -a "$PACKAGES_DIR/download.log" || true

# source 패키지가 필요한 경우를 위한 폴백 다운로드
echo -e "\n${YELLOW}[2/3] 소스 패키지 다운로드 (binary가 없는 경우)...${NC}"
pip download \
    -r "$REQUIREMENTS" \
    -d "$PACKAGES_DIR" \
    --no-binary=:none: \
    --no-cache-dir \
    2>&1 | tee -a "$PACKAGES_DIR/download.log" || true

# pip 자체도 다운로드 (오프라인 환경에서 pip 업그레이드용)
echo -e "\n${YELLOW}[3/3] pip, setuptools, wheel 다운로드...${NC}"
pip download \
    pip setuptools wheel \
    -d "$PACKAGES_DIR" \
    --only-binary=:all: \
    --no-cache-dir || true

# 다운로드된 패키지 확인
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}다운로드 완료!${NC}"
echo -e "${GREEN}============================================================${NC}"

WHEEL_COUNT=$(ls -1 "$PACKAGES_DIR"/*.whl 2>/dev/null | wc -l)
TAR_COUNT=$(ls -1 "$PACKAGES_DIR"/*.tar.gz 2>/dev/null | wc -l)

echo -e ""
echo -e "다운로드된 파일:"
echo -e "  - Wheel 파일: ${WHEEL_COUNT}개"
echo -e "  - Source 파일: ${TAR_COUNT}개"
echo -e "  - 위치: $PACKAGES_DIR"

# 용량 확인
TOTAL_SIZE=$(du -sh "$PACKAGES_DIR" | cut -f1)
echo -e "  - 총 용량: $TOTAL_SIZE"

# 다운로드된 주요 패키지 목록
echo -e "\n${BLUE}주요 패키지 버전:${NC}"
ls "$PACKAGES_DIR"/*.whl 2>/dev/null | xargs -I {} basename {} | grep -E "^(numpy|pandas|scikit_learn|scipy|xgboost|lightgbm|shap|optuna|matplotlib)" | head -20 || true

echo -e "\n${YELLOW}다음 단계:${NC}"
echo -e "1. ${PACKAGES_DIR}/ 폴더를 폐쇄망 환경으로 복사"
echo -e "2. 프로젝트 전체를 폐쇄망으로 이동"
echo -e "3. 폐쇄망에서 scripts/install_offline.sh 실행"

echo -e "\n${GREEN}완료!${NC}"
