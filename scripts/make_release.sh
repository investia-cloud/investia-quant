#!/usr/bin/env bash
# =============================================================================
# make_release.sh — Crea una release versionata di investia-quant
#
# Uso:
#   ./scripts/make_release.sh [VERSION]
#
# Esempi:
#   ./scripts/make_release.sh          # versione auto: 2026.1, 2026.2, ...
#   ./scripts/make_release.sh 2026.1   # versione esplicita
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0) Verifica root progetto
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "❌ Esegui lo script dalla root del progetto (investia-quant/)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1) Versione
# ---------------------------------------------------------------------------
YEAR=$(date +%Y)

if [[ $# -ge 1 ]]; then
    VERSION="$1"
else
    N=1
    while [[ -d "releases/${YEAR}.${N}" ]]; do
        N=$((N + 1))
    done
    VERSION="${YEAR}.${N}"
fi

RELEASE_DIR="releases/${VERSION}"

if [[ -d "$RELEASE_DIR" ]]; then
    echo "❌ Release $VERSION esiste già: $RELEASE_DIR" >&2
    exit 1
fi

echo "🚀 Creazione release ${VERSION} in ${RELEASE_DIR}..."

# ---------------------------------------------------------------------------
# 2) Verifica working tree pulito (ignora releases/ non tracciata)
# ---------------------------------------------------------------------------
if [[ -n "$(git status --porcelain 2>/dev/null | grep -v '^?? releases/')" ]]; then
    echo "⚠️  Working tree non pulito. Committa o stasha prima di fare release."
    git status --short
    exit 1
fi

# ---------------------------------------------------------------------------
# 3) Crea struttura directory
# ---------------------------------------------------------------------------
mkdir -p "${RELEASE_DIR}/lib"
mkdir -p "${RELEASE_DIR}/investia_quant"
mkdir -p "${RELEASE_DIR}/inputs"
mkdir -p "${RELEASE_DIR}/outputs"
mkdir -p "${RELEASE_DIR}/logs"
mkdir -p "${RELEASE_DIR}/cache"
mkdir -p "${RELEASE_DIR}/scripts"

# ---------------------------------------------------------------------------
# 4) Copia codice
# ---------------------------------------------------------------------------
echo "  📦 Copia libs_py/ → lib/"
cp notebooks/libs_py/*.py "${RELEASE_DIR}/lib/"

echo "  📦 Copia investia_quant/ → investia_quant/"
cp investia_quant/*.py "${RELEASE_DIR}/investia_quant/"

echo "  📦 Copia pyproject.toml + requirements.lock"
cp pyproject.toml "${RELEASE_DIR}/"
if [[ -f requirements.lock ]]; then
    cp requirements.lock "${RELEASE_DIR}/"
else
    echo "  ⚠️  requirements.lock non trovato"
fi

# ---------------------------------------------------------------------------
# 5) Copia dati runtime
# ---------------------------------------------------------------------------
echo "  📦 Copia inputs/WFO_*_RUN_RESULTS/"
cp -r inputs/WFO_T_RUN_RESULTS "${RELEASE_DIR}/inputs/"
cp -r inputs/WFO_R_RUN_RESULTS "${RELEASE_DIR}/inputs/"

echo "  📦 Copia cache/"
cp -r cache/. "${RELEASE_DIR}/cache/"

echo "  📦 Copia scripts/portfolios.conf"
cp scripts/portfolios.conf "${RELEASE_DIR}/scripts/"

# ---------------------------------------------------------------------------
# 6) Genera install.sh
# ---------------------------------------------------------------------------
echo "  📝 Genera scripts/install.sh"
cat > "${RELEASE_DIR}/scripts/install.sh" << 'INSTALL_EOF'
#!/usr/bin/env bash
# install.sh — Setup ambiente sulla VPS per questa release
#
# Uso: ./scripts/install.sh

set -euo pipefail

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${RELEASE_DIR}/.venv"

echo "📁 Release dir: ${RELEASE_DIR}"
echo "🐍 Venv dir:    ${VENV_DIR}"

# 1) Crea venv
echo "🔧 Creazione venv..."
python3 -m venv "${VENV_DIR}"

# 2) Installa dipendenze da requirements.lock (no git clone)
echo "📦 Installazione dipendenze..."
if [[ -f "${RELEASE_DIR}/requirements.lock" ]]; then
    "${VENV_DIR}/bin/pip" install --quiet --no-deps -r "${RELEASE_DIR}/requirements.lock" || \
    "${VENV_DIR}/bin/pip" install --quiet -r "${RELEASE_DIR}/requirements.lock"
else
    echo "⚠️  requirements.lock non trovato — installo da pyproject.toml (solo dipendenze)"
    "${VENV_DIR}/bin/pip" install --quiet \
        numpy pandas scipy scikit-learn matplotlib seaborn plotly \
        vectorbt yfinance statsmodels reportlab Pillow joblib tqdm \
        tqdm-joblib PyPortfolioOpt tabulate psutil pytz requests click
fi

# 3) Registra lib/ nel sys.path via .pth
echo "🔧 Registro lib/ nel sys.path..."
PY_VER=$("${VENV_DIR}/bin/python3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "${RELEASE_DIR}/lib" > "${VENV_DIR}/lib/python${PY_VER}/site-packages/investia_libs.pth"
echo "  ✅ lib/ → investia_libs.pth"

# 4) Registra investia_quant/ nel sys.path via .pth
echo "🔧 Registro investia_quant/ nel sys.path..."
echo "${RELEASE_DIR}" > "${VENV_DIR}/lib/python${PY_VER}/site-packages/investia_quant.pth"
echo "  ✅ investia_quant/ → investia_quant.pth"

# 5) Crea script wrapper iq nel venv
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
INSTALL_EOF
chmod +x "${RELEASE_DIR}/scripts/install.sh"

# ---------------------------------------------------------------------------
# 7) Genera crontab.txt
# ---------------------------------------------------------------------------
echo "  📝 Genera scripts/crontab.txt"
cat > "${RELEASE_DIR}/scripts/crontab.txt" << CRON_EOF
# investia-quant ${VERSION} — crontab entries
# Adatta RELEASE_DIR al path reale sulla VPS
# RELEASE_DIR=/home/luca/investia-quant/releases/current
# IQ=\${RELEASE_DIR}/.venv/bin/iq

# R-portfolio: primo lunedì del mese alle 07:00
0 7 1-7 * 1 \${RELEASE_DIR}/.venv/bin/iq run --ptf-all-r >> \${RELEASE_DIR}/logs/r_run.log 2>&1

# K-portfolio: ogni giorno feriale alle 18:30
30 18 * * 1-5 \${RELEASE_DIR}/.venv/bin/iq run --ptf-all-k >> \${RELEASE_DIR}/logs/k_run.log 2>&1
CRON_EOF

# ---------------------------------------------------------------------------
# 8) Genera README.md
# ---------------------------------------------------------------------------
echo "  📝 Genera README.md"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE=$(date +%Y-%m-%d)
cat > "${RELEASE_DIR}/README.md" << README_EOF
# investia-quant — Release ${VERSION}

**Data**: ${DATE}
**Commit**: ${COMMIT}
**Branch**: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

## Deploy su VPS

\`\`\`bash
./scripts/deploy.sh ${VERSION} <vps_host> <install_dir>
# Es: ./scripts/deploy.sh ${VERSION} tslab.investia.cloud /home/luca
\`\`\`

## Rollback

\`\`\`bash
ssh <vps_host> "ln -sfn <install_dir>/investia-quant/releases/<versione_precedente> <install_dir>/investia-quant/releases/current"
\`\`\`

## Struttura

\`\`\`
${VERSION}/
├── lib/                  librerie runtime (.py)
├── investia_quant/       CLI iq (cli.py)
├── inputs/               dati WFO
├── cache/                cache ticker/ISIN
├── outputs/              output report (vuota)
├── logs/                 log cron (vuota)
├── scripts/
│   ├── portfolios.conf
│   ├── install.sh        setup venv + .pth + wrapper iq
│   └── crontab.txt
└── .venv/                venv (creato da install.sh, non in git)
\`\`\`
README_EOF

# ---------------------------------------------------------------------------
# 9) Aggiorna symlink current
# ---------------------------------------------------------------------------
echo "  🔗 Aggiorno symlink releases/current → ${VERSION}"
ln -sfn "${VERSION}" "releases/current"

# ---------------------------------------------------------------------------
# 10) Riepilogo
# ---------------------------------------------------------------------------
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo ""
echo "✅ Release ${VERSION} creata."
echo "   Directory: ${RELEASE_DIR}"
echo "   Symlink:   releases/current → ${VERSION}"
echo "   Commit:    ${COMMIT}"
echo ""
echo "   Prossimi passi:"
echo "   1. git add releases/${VERSION}/ releases/current && git commit -m 'release: ${VERSION}'"
echo "   2. ./scripts/deploy.sh ${VERSION} <vps_host> <install_dir>"
