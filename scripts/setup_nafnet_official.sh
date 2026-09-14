#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_DIR="${ROOT_DIR}/nafnet/official"
REPO_URL="https://github.com/megvii-research/NAFNet.git"

if [[ -f "${OFFICIAL_DIR}/basicsr/models/archs/NAFNet_arch.py" ]]; then
  echo "[setup_nafnet_official] Official NAFNet code already exists: ${OFFICIAL_DIR}"
  exit 0
fi

if [[ -d "${OFFICIAL_DIR}" ]]; then
  echo "[setup_nafnet_official] ${OFFICIAL_DIR} exists but does not look like the official NAFNet checkout."
  echo "[setup_nafnet_official] Move it aside or remove it, then run this script again."
  exit 1
fi

mkdir -p "${ROOT_DIR}/nafnet"

echo "[setup_nafnet_official] Cloning official NAFNet implementation..."
git clone --depth 1 "${REPO_URL}" "${OFFICIAL_DIR}"
echo "[setup_nafnet_official] Saved to: ${OFFICIAL_DIR}"
