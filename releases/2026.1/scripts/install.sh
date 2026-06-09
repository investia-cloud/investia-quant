#!/usr/bin/env bash
# install.sh — Setup ambiente sulla VPS per questa release

set -euo pipefail

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${RELEASE_DIR}/.venv"

echo "📁 Release dir: ${RELEASE_DIR}"
echo "🐍 Venv dir:    ${VENV_DIR}"

# 1) Crea venv
echo "🔧 Creazione venv..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip

# 2) Installa dipendenze senza versioni pinnate (compatibile con qualsiasi Python 3.x)
echo "📦 Installazione dipendenze..."
"${VENV_DIR}/bin/pip" install \
    numpy pandas scipy scikit-learn matplotlib seaborn plotly \
    vectorbt yfinance statsmodels reportlab Pillow joblib tqdm \
    tqdm-joblib PyPortfolioOpt tabulate psutil pytz requests click

# 3) Registra lib/ nel sys.path via .pth
PY_VER=$("${VENV_DIR}/bin/python3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🔧 Registro lib/ nel sys.path (python${PY_VER})..."
echo "${RELEASE_DIR}/lib" > "${VENV_DIR}/lib/python${PY_VER}/site-packages/investia_libs.pth"
echo "  ✅ lib/ → investia_libs.pth"

# 4) Registra investia_quant/ nel sys.path via .pth
echo "🔧 Registro investia_quant/ nel sys.path..."
echo "${RELEASE_DIR}" > "${VENV_DIR}/lib/python${PY_VER}/site-packages/investia_quant.pth"
echo "  ✅ investia_quant/ → investia_quant.pth"

# 5) Crea wrapper CLI iq
echo "🔧 Crea wrapper CLI iq..."
cat > "${VENV_DIR}/bin/iq" << WRAPPER_EOF
#!/bin/bash
export IQ_INPUTS_DIR="${RELEASE_DIR}/inputs"
export IQ_OUTPUTS_DIR="${RELEASE_DIR}/outputs"
export IQ_CACHE_DIR="${RELEASE_DIR}/cache"
exec "${VENV_DIR}/bin/python3" -c "from investia_quant.cli import app; app()" "\$@"
WRAPPER_EOF
chmod +x "${VENV_DIR}/bin/iq"
echo "  ✅ wrapper iq creato"

# 6) Crea .envrc per direnv
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
