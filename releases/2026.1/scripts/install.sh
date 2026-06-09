#!/usr/bin/env bash
# install.sh — Setup ambiente sulla VPS per questa release
#
# Uso (dalla directory della release):
#   ./scripts/install.sh
#
# Oppure con path assoluto:
#   /path/to/releases/2026.1/scripts/install.sh

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
    "${VENV_DIR}/bin/pip" install --quiet -e "${RELEASE_DIR}[runtime]"
fi

# Installa CLI iq
echo "🔧 Installazione CLI iq..."
"${VENV_DIR}/bin/pip" install --quiet -e "${RELEASE_DIR}"

# Crea .envrc per direnv (opzionale)
cat > "${RELEASE_DIR}/.envrc" << ENVRC_EOF
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:$PATH"
export IQ_INPUTS_DIR="${RELEASE_DIR}/inputs"
export IQ_OUTPUTS_DIR="${RELEASE_DIR}/outputs"
export IQ_CACHE_DIR="${RELEASE_DIR}/cache"
ENVRC_EOF

echo ""
echo "✅ Installazione completata."
echo ""
echo "   Per attivare manualmente:"
echo "   source ${VENV_DIR}/bin/activate"
echo ""
echo "   Per testare:"
echo "   ${VENV_DIR}/bin/iq --help"
