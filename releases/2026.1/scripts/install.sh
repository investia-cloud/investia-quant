#!/usr/bin/env bash
set -euo pipefail

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${RELEASE_DIR}/.venv"

echo "📁 Release dir: ${RELEASE_DIR}"
echo "🐍 Venv dir:    ${VENV_DIR}"

echo "🔧 Creazione venv..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip

echo "📦 Installazione dipendenze..."
"${VENV_DIR}/bin/pip" install \
    numpy pandas scipy scikit-learn matplotlib seaborn plotly \
    vectorbt yfinance statsmodels reportlab Pillow joblib tqdm \
    tqdm-joblib PyPortfolioOpt tabulate psutil pytz requests click

PY_VER=$("${VENV_DIR}/bin/python3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "🔧 Registro lib/ nel sys.path..."
echo "${RELEASE_DIR}/lib" > "${VENV_DIR}/lib/python${PY_VER}/site-packages/investia_libs.pth"

echo "🔧 Registro investia_quant/ nel sys.path..."
echo "${RELEASE_DIR}" > "${VENV_DIR}/lib/python${PY_VER}/site-packages/investia_quant.pth"

echo "🔧 Crea wrapper CLI iq..."
cat > "${VENV_DIR}/bin/iq" << WRAPPER_EOF
#!/bin/bash
export IQ_INPUTS_DIR="${RELEASE_DIR}/inputs"
export IQ_OUTPUTS_DIR="${RELEASE_DIR}/outputs"
export IQ_CACHE_DIR="${RELEASE_DIR}/cache"
exec "${VENV_DIR}/bin/python3" -c "from investia_quant.cli import app; app()" "\$@"
WRAPPER_EOF
chmod +x "${VENV_DIR}/bin/iq"

cat > "${RELEASE_DIR}/.envrc" << ENVRC_EOF
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:$PATH"
export IQ_INPUTS_DIR="${RELEASE_DIR}/inputs"
export IQ_OUTPUTS_DIR="${RELEASE_DIR}/outputs"
export IQ_CACHE_DIR="${RELEASE_DIR}/cache"
ENVRC_EOF

echo ""
echo "✅ Installazione completata."
echo "   Testa con: ${VENV_DIR}/bin/iq --help"
