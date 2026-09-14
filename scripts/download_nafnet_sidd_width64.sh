#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${ROOT_DIR}/nafnet/weights"
OUT_PATH="${WEIGHTS_DIR}/nafnet_sidd_width64.pth"
URL="https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p "${WEIGHTS_DIR}"

if [[ -f "${OUT_PATH}" ]]; then
  echo "[download_nafnet_sidd_width64] File already exists: ${OUT_PATH}"
  exit 0
fi

if ! "${PYTHON_BIN}" -c "import gdown" >/dev/null 2>&1; then
  echo "[download_nafnet_sidd_width64] Installing gdown..."
  "${PYTHON_BIN}" -m pip install gdown
fi

echo "[download_nafnet_sidd_width64] Downloading checkpoint..."
if command -v gdown >/dev/null 2>&1; then
  gdown --fuzzy "${URL}" -O "${OUT_PATH}"
else
  "${PYTHON_BIN}" -m gdown.cli --fuzzy "${URL}" -O "${OUT_PATH}"
fi

echo "[download_nafnet_sidd_width64] Saved to: ${OUT_PATH}"
