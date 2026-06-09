#!/usr/bin/env bash
# install.sh — Setup ambiente sulla VPS per questa release

set -euo pipefail

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${RELEASE_DIR}/.venv"

echo "📁 Release dir: ${RELEASE_DIR}"
echo "🐍 Venv dir:    ${VENV_DIR}"

# Crea venv
echo "🔧 Creazione venv..."
python3 -m venv "${VENV_DIR}"

# Installa dipendenze
echo "📦 Installazione dipendenze..."
if [[ -f "${RELEASE_DIR}/requirements.lock" ]]; then
    "${VENV_DIR}/bin/pip" install --quiet -r "${RELEASE_DIR}/requirements.lock"
else
    "${VENV_DIR}/bin/pip" install --quiet "${RELEASE_DIR}"
fi

# Installa CLI iq
echo "🔧 Installazione CLI iq..."
"${VENV_DIR}/bin/pip" install --quiet -e "${RELEASE_DIR}"

# Aggiungi lib/ al sys.path via file .pth
echo "🔧 Registro lib/ nel sys.path del venv..."
PY_VER=$("${VENV_DIR}/bin/python3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "${RELEASE_DIR}/lib" > "${VENV_DIR}/lib/python${PY_VER}/site-packages/investia_libs.pth"
echo "  ✅ lib/ aggiunto a sys.path (investia_libs.pth)"

# Crea .envrc per direnv
cat > "${RELEASE_DIR}/.envrc" << ENVRC_EOF
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:$PATH"
export IQ_INPUTS_DIR="${RELEASE_DIR}/inputs"
export IQ_OUTPUTS_DIR="${RELEASE_DIR}/outputs"
export IQ_CACHE_DIR="${RELEASE_DIR}/cache"
ENVRC_EOF

echo ""
echo "✅ Installazione completata."
echo "   Attiva con: source ${VENV_DIR}/bin/activate"
echo "   Testa con:  ${VENV_DIR}/bin/iq --help"
