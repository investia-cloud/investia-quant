#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy release versionata su VPS
#
# Uso:
#   ./scripts/deploy.sh [VERSION] [VPS_HOST] [INSTALL_DIR]
#
# Esempi:
#   ./scripts/deploy.sh                                        # current, tslab.investia.cloud, ~
#   ./scripts/deploy.sh 2026.1                                 # versione esplicita
#   ./scripts/deploy.sh 2026.1 tslab.investia.cloud            # host esplicito
#   ./scripts/deploy.sh 2026.1 tslab.investia.cloud /opt/iq    # dir installazione custom
#
# Flusso:
#   1. Verifica release locale
#   2. Crea struttura directory sulla VPS
#   3. rsync release → VPS
#   4. Esegue install.sh sulla VPS (crea venv + installa dipendenze)
#   5. Aggiorna symlink current sulla VPS
#   6. Verifica: iq --help
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
VERSION="${1:-}"
VPS_HOST="${2:-tslab.investia.cloud}"
INSTALL_DIR="${3:-~}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Risolvi versione: se non specificata usa current
if [[ -z "$VERSION" ]]; then
    if [[ -L "releases/current" ]]; then
        VERSION=$(readlink releases/current)
    else
        echo "❌ Nessuna versione specificata e releases/current non esiste." >&2
        exit 1
    fi
fi

RELEASE_DIR="releases/${VERSION}"

# ---------------------------------------------------------------------------
# Verifica release locale
# ---------------------------------------------------------------------------
if [[ ! -d "$RELEASE_DIR" ]]; then
    echo "❌ Release non trovata: ${RELEASE_DIR}" >&2
    exit 1
fi

echo "🚀 Deploy release ${VERSION} su ${VPS_HOST}:${INSTALL_DIR}"
echo "   Release locale: ${ROOT}/${RELEASE_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Crea struttura directory sulla VPS
# ---------------------------------------------------------------------------
echo "📁 Creo directory sulla VPS..."
ssh "${VPS_HOST}" "mkdir -p ${INSTALL_DIR}/investia-quant/releases/${VERSION}"

# ---------------------------------------------------------------------------
# 2. rsync release → VPS (escludi .venv se presente)
# ---------------------------------------------------------------------------
echo "📦 Trasferisco release (rsync)..."
rsync -av --progress \
    --exclude='.venv/' \
    --exclude='outputs/*' \
    "${RELEASE_DIR}/" \
    "${VPS_HOST}:${INSTALL_DIR}/investia-quant/releases/${VERSION}/"

# ---------------------------------------------------------------------------
# 3. Esegui install.sh sulla VPS
# ---------------------------------------------------------------------------
echo ""
echo "🔧 Eseguo install.sh sulla VPS..."
ssh "${VPS_HOST}" "bash ${INSTALL_DIR}/investia-quant/releases/${VERSION}/scripts/install.sh"

# ---------------------------------------------------------------------------
# 4. Aggiorna symlink current sulla VPS
# ---------------------------------------------------------------------------
echo ""
echo "🔗 Aggiorno symlink current sulla VPS..."
ssh "${VPS_HOST}" "ln -sfn ${INSTALL_DIR}/investia-quant/releases/${VERSION} ${INSTALL_DIR}/investia-quant/releases/current"

# ---------------------------------------------------------------------------
# 5. Verifica
# ---------------------------------------------------------------------------
echo ""
echo "✅ Verifica installazione..."
ssh "${VPS_HOST}" "${INSTALL_DIR}/investia-quant/releases/current/.venv/bin/iq --help"

# ---------------------------------------------------------------------------
# Riepilogo
# ---------------------------------------------------------------------------
echo ""
echo "✅ Deploy ${VERSION} completato su ${VPS_HOST}."
echo ""
echo "   VPS release dir: ${INSTALL_DIR}/investia-quant/releases/${VERSION}"
echo "   VPS symlink:     ${INSTALL_DIR}/investia-quant/releases/current → ${VERSION}"
echo ""
echo "   Per rollback:"
echo "   ssh ${VPS_HOST} \"ln -sfn <versione_precedente> ${INSTALL_DIR}/investia-quant/releases/current\""
