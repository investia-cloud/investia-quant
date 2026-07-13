#!/usr/bin/env bash
# ============================================================================
# session_start.sh — Inizio sessione di lavoro investia-quant
#
# Uso:
#   bash docs/session_start.sh              # usa il branch corrente
#   bash docs/session_start.sh <nome-branch> # switcha esplicitamente a un branch
#
# Cosa fa:
#   1. git fetch origin
#   2. git checkout <branch>  (solo se specificato)
#   3. git pull origin <branch-corrente> --ff-only  (si ferma se non e' un
#      fast-forward pulito, invece di creare merge/conflitti a sorpresa)
#   4. verifica che il filtro nbstripout sia configurato (lo installa se manca)
#   5. mostra stato e ultimi commit, per partire con un quadro chiaro
#
# Se il pull --ff-only fallisce: NON continuare da soli. Vuol dire che questo
# branch ha divergenze reali tra locale e remoto — va capito il motivo prima
# di procedere (vedi convenzione: mai git add . / mai forzare senza guardare).
# ============================================================================
set -euo pipefail
REPO_DIR="${INVESTIA_QUANT_DIR:-$HOME/investia-quant}"
cd "$REPO_DIR"

BRANCH="${1:-}"

echo "=== 1/5 — Fetch origin ==="
git fetch origin
echo ""

if [[ -n "$BRANCH" ]]; then
    echo "=== 2/5 — Checkout $BRANCH ==="
    git checkout "$BRANCH"
else
    BRANCH="$(git branch --show-current)"
    echo "=== 2/5 — Nessun branch specificato, resto su quello corrente: $BRANCH ==="
fi
echo ""

echo "=== 3/5 — Pull (fast-forward only) su $BRANCH ==="
if ! git pull origin "$BRANCH" --ff-only; then
    echo ""
    echo "!!! ATTENZIONE: pull --ff-only fallito."
    echo "!!! Il branch locale e quello remoto sono divergenti, oppure ci sono"
    echo "!!! modifiche locali non committate che bloccano l'aggiornamento."
    echo "!!! NON procedere alla cieca — controlla 'git status' e 'git log"
    echo "!!! --oneline --graph -10' prima di continuare."
    exit 1
fi
echo ""

echo "=== 4/5 — Verifica filtro nbstripout ==="
if [[ -z "$(git config --get filter.nbstripout.clean || true)" ]]; then
    echo "Filtro nbstripout non configurato su questa macchina — lo installo."
    pip install nbstripout --break-system-packages --quiet || true
    nbstripout --install
else
    echo "OK, filtro nbstripout gia' configurato."
fi
echo ""

echo "=== 5/5 — Stato attuale ==="
git status
echo ""
echo "--- Ultimi 5 commit ---"
git log --oneline -5
echo ""
echo "=== Sessione pronta su branch '$BRANCH'. Buon lavoro. ==="
