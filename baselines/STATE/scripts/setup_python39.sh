#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-py39}"
CELL_LOAD_DIR="${ROOT_DIR}/.compat/cell-load"
CELL_LOAD_TAG="v0.10.4"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

"${UV_BIN}" python install "${PYTHON_VERSION}"

if [[ ! -d "${CELL_LOAD_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${CELL_LOAD_DIR}")"
    git clone --branch "${CELL_LOAD_TAG}" --depth 1 \
        https://github.com/ArcInstitute/cell-load.git "${CELL_LOAD_DIR}"
fi

if git -C "${CELL_LOAD_DIR}" apply --check \
    "${ROOT_DIR}/compat/cell-load-py39.patch" >/dev/null 2>&1; then
    git -C "${CELL_LOAD_DIR}" apply "${ROOT_DIR}/compat/cell-load-py39.patch"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    -e "${CELL_LOAD_DIR}" \
    -e "${ROOT_DIR}"

"${VENV_DIR}/bin/python" -m compileall -q \
    "${ROOT_DIR}/src" \
    "${CELL_LOAD_DIR}/src"

"${VENV_DIR}/bin/python" -c \
    "import state, cell_load; print('STATE Python 3.9 environment is ready')"

echo "Activate with: source ${VENV_DIR}/bin/activate"
