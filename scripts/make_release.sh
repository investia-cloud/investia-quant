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
#
# Struttura prodotta:
#   releases/2026.1/
#   ├── lib/                  librerie runtime (libs_py/)
#   ├── investia_quant/       CLI iq
#   ├── pyproject.toml
#   ├── requirements.lock
#   ├── inputs/
#   │   ├── WFO_T_RUN_RESULTS/
#   │   └── WFO_R_RUN_RESULTS/
#   ├── cache/
#   ├── outputs/              (vuota)
#   ├── scripts/
#   │   ├── portfolios.conf
#   │   ├── install.sh
#   │   └── crontab.txt
#   └── README.md
#   releases/current -> releases/2026.1/  (symlink aggiornato)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0) Verifica che siamo nella root del progetto
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
    # Auto-incrementa: cerca releases/YEAR.N esistenti
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
# 2) Verifica working tree pulito
# ---------------------------------------------------------------------------
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
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
mkdir -p "${RELEASE_DIR}/cache"
mkdir -p "${RELEASE_DIR}/scripts"

# ---------------------------------------------------------------------------
# 4) Copia codice
# ---------------------------------------------------------------------------
echo "  📦 Copia libs_py/ → lib/"
cp notebooks/libs_py/*.py "${RELEASE_DIR}/lib/"

echo "  📦 Copia investia_quant/ → investia_quant/"
cp -r investia_quant/*.py "${RELEASE_DIR}/investia_quant/"

echo "  📦 Copia pyproject.toml + requirements.lock"
cp pyproject.toml "${RELEASE_DIR}/"
if [[ -f requirements.lock ]]; then
    cp requirements.lock "${RELEASE_DIR}/"
else
    echo "  ⚠️  requirements.lock non trovato — verrà generato da install.sh"
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
INSTALL_EOF
chmod +x "${RELEASE_DIR}/scripts/install.sh"

# ---------------------------------------------------------------------------
# 7) Genera crontab.txt
# ---------------------------------------------------------------------------
echo "  📝 Genera scripts/crontab.txt"
cat > "${RELEASE_DIR}/scripts/crontab.txt" << CRON_EOF
# investia-quant ${VERSION} — crontab entries
# Incolla queste righe con: crontab -e
#
# Variabili ambiente (adatta i path alla VPS)
# RELEASE_DIR=/home/luca/investia-quant/releases/current
# IQ=/home/luca/investia-quant/releases/current/.venv/bin/iq

# R-portfolio: primo lunedì del mese alle 07:00
0 7 1-7 * 1 IQ_INPUTS_DIR=\$RELEASE_DIR/inputs IQ_OUTPUTS_DIR=\$RELEASE_DIR/outputs IQ_CACHE_DIR=\$RELEASE_DIR/cache \$IQ run --ptf-all-r >> \$RELEASE_DIR/logs/r_run.log 2>&1

# K-portfolio: ogni giorno feriale alle 18:30
30 18 * * 1-5 IQ_INPUTS_DIR=\$RELEASE_DIR/inputs IQ_OUTPUTS_DIR=\$RELEASE_DIR/outputs IQ_CACHE_DIR=\$RELEASE_DIR/cache \$IQ run --ptf-all-k >> \$RELEASE_DIR/logs/k_run.log 2>&1
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
# 1. Copia la release sulla VPS
rsync -av releases/${VERSION}/ tslab.investia.cloud:~/investia-quant/releases/${VERSION}/

# 2. Sulla VPS: installa l'ambiente
ssh tslab.investia.cloud "~/investia-quant/releases/${VERSION}/scripts/install.sh"

# 3. Sulla VPS: aggiorna il symlink current
ssh tslab.investia.cloud "ln -sfn ~/investia-quant/releases/${VERSION} ~/investia-quant/releases/current"

# 4. Verifica
ssh tslab.investia.cloud "~/investia-quant/releases/current/.venv/bin/iq --help"
\`\`\`

## Rollback

\`\`\`bash
# Torna alla release precedente
ssh tslab.investia.cloud "ln -sfn ~/investia-quant/releases/2026.0 ~/investia-quant/releases/current"
\`\`\`

## Struttura

\`\`\`
${VERSION}/
├── lib/                  librerie runtime
├── investia_quant/       CLI iq
├── inputs/               dati WFO + config portafogli
├── cache/                cache ticker/ISIN
├── outputs/              output report (vuota all'installazione)
├── scripts/
│   ├── portfolios.conf   configurazione portafogli
│   ├── install.sh        setup venv
│   └── crontab.txt       entries cron
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
echo ""
echo "✅ Release ${VERSION} creata."
echo ""
echo "   Directory: ${RELEASE_DIR}"
echo "   Symlink:   releases/current → ${VERSION}"
echo "   Commit:    ${COMMIT}"
echo ""
echo "   Prossimi passi:"
echo "   1. git add releases/${VERSION}/ && git commit -m 'release: ${VERSION}'"
echo "   2. rsync -av releases/${VERSION}/ tslab.investia.cloud:~/investia-quant/releases/${VERSION}/"
echo "   3. ssh tslab.investia.cloud '~/investia-quant/releases/${VERSION}/scripts/install.sh'"
echo "   4. ssh tslab.investia.cloud 'ln -sfn ~/investia-quant/releases/${VERSION} ~/investia-quant/releases/current'"
