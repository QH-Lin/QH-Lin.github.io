#!/usr/bin/env bash

set -euo pipefail

readonly HUGO_VERSION="0.164.0"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TOOLS_DIR="${PROJECT_ROOT}/.tools"
readonly HUGO_ENV="${TOOLS_DIR}/hugo-env"
readonly HUGO_LINK="${TOOLS_DIR}/hugo"

if [[ -x "${HUGO_LINK}" ]]; then
  "${HUGO_LINK}" version
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required for the project-local Hugo installation." >&2
  echo "Install Hugo Extended manually, or install Conda and rerun this script." >&2
  exit 1
fi

mkdir -p "${TOOLS_DIR}"
conda create \
  --prefix "${HUGO_ENV}" \
  --channel conda-forge \
  "hugo=${HUGO_VERSION}" \
  --yes

ln -s "hugo-env/bin/hugo" "${HUGO_LINK}"
"${HUGO_LINK}" version
