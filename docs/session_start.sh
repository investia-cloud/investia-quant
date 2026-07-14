#!/usr/bin/env bash
# ============================================================================
# session_start.sh — Inizio sessione di lavoro investia-quant (v2)
#
# Uso:
#   bash docs/session_start.sh              # usa il branch corrente
#   bash docs/session_start.sh <nome-branch> # switcha esplicitamente a un branch
#
# Cosa fa:
#   1. git fetch origin
#   2. git checkout <branch>  (solo se specificato) — gestisce in automatico
#      il filtro nbstripout non idempotente se blocca il checkout
#   3. git pull origin <branch-corrente> --ff-only  (si ferma se non e' un
#      fast-forward pulito, invece di creare merge/conflitti a sorpresa)
#   4. verifica che il filtro nbstripout sia configurato (lo installa se manca)
#   5. mostra stato, ultimi 5 commit, e QUALUNQUE branch non ancora
#      mergiato in main (visibilita' immediata di lavoro sospeso)
#
# Se il pull --ff-only fallisce: NON continuare da soli. Vuol dire che questo
# branch ha divergenze reali tra locale e remoto — va capito il motivo prima
# di procedere (vedi convenzione: mai git add . / mai forzare senza guardare).
# ============================================================================
set -euo pipefail
REPO_DIR="${INVESTIA_QUANT_DIR:-$HOME/investia-quant}"
cd "$REPO_DIR"

NBSTRIPOUT_CLEAN="$(git config --get filter.nbstripout.clean || true)"
_disable_nbstripout() {
    if [[ -n "$NBSTRIPOUT_CLEAN" ]]; then
        git config --local filter.nbstripout.clean cat
    fi
}
_restore_nbstripout() {
    if [[ -n "$NBSTRIPOUT_CLEAN" ]]; then
        git config --local filter.nbstripout.clean "$NBSTRIPOUT_CLEAN"
    fi
}

BRANCH="${1:-}"

echo "=== 1/5 — Fetch origin ==="
git fetch origin
echo ""

if [[ -n "$BRANCH" ]]; then
    echo "=== 2/5 — Checkout $BRANCH ==="
    if ! git checkout "$BRANCH" 2>/tmp/session_start_checkout_err.log; then
        if grep -q "would be overwritten by checkout" /tmp/session_start_checkout_err.log; then
            echo "Checkout bloccato da modifiche locali spurie (probabile nbstripout"
            echo "non idempotente su un notebook) — ritento disabilitando il filtro..."
            _disable_nbstripout
            git checkout "$BRANCH"
            _restore_nbstripout
        else
            cat /tmp/session_start_checkout_err.log
            exit 1
        fi
    fi
else
    BRANCH="$(git branch --show-current)"
    echo "=== 2/5 — Nessun branch specificato, resto su quello corrente: $BRANCH ==="
fi
echo ""

echo "=== 3/5 — Pull (fast-forward only) su $BRANCH ==="
PULL_OK=1
git pull origin "$BRANCH" --ff-only 2>/tmp/session_start_pull_err.log || PULL_OK=0
if [[ "$PULL_OK" == "0" ]]; then
    if grep -q "would be overwritten by merge" /tmp/session_start_pull_err.log; then
        echo "Pull bloccato da modifiche locali spurie (probabile nbstripout"
        echo "non idempotente) — ritento disabilitando il filtro..."
        _disable_nbstripout
        PULL_OK=1
        git pull origin "$BRANCH" --ff-only || PULL_OK=0
        _restore_nbstripout
    fi
fi
if [[ "$PULL_OK" == "0" ]]; then
    echo ""
    cat /tmp/session_start_pull_err.log
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
echo "--- Branch non ancora mergiati in main ---"
UNMERGED="$(git branch -a --no-merged main 2>/dev/null || true)"
if [[ -z "$UNMERGED" ]]; then
    echo "Nessuno. Tutto il lavoro esistente e' consolidato in main."
else
    echo "$UNMERGED"
    echo "^^^ Questi branch hanno commit non ancora in main — verifica se e'"
    echo "    lavoro sospeso da recuperare o solo un branch da eliminare."
fi
echo ""
echo "=== Sessione pronta su branch '$BRANCH'. Buon lavoro. ==="
