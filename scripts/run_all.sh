#!/bin/bash
# ================================================================
# 전체 파이프라인 실행 (간편 스크립트)
# 
# 사용법:
#   bash scripts/run_all.sh                    # 기본 실행
#   bash scripts/run_all.sh --small-grid       # 빠른 테스트
#   bash scripts/run_all.sh --skip-dummy       # 더미 데이터 건너뜀
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "  Running Full Pipeline"
echo "  Project: $PROJECT_DIR"
echo "=================================================="

cd "$PROJECT_DIR"

# Python 실행 (run_all.py에 모든 인자 전달)
python run_all.py "$@"

echo ""
echo "Done!"
