#!/usr/bin/env bash
# ============================================================================
# session_end.sh — Fine sessione di lavoro investia-quant (v2)
#
# Uso:
#   bash docs/session_end.sh "messaggio di commit"
#
# Cosa fa:
#   1. mostra git status
#   2. verifica sintassi Python di r_functions.py e u_functions.py, se
#      modificati (blocca il commit se la sintassi e' rotta)
#   3. git add -u (SOLO file gia' tracciati — mai file nuovi in automatico)
#   4. commit con il messaggio passato come argomento
#   5. push sul branch corrente
#   6. SE il branch corrente NON e' main: merge automatico in main
#      (checkout main, pull --ff-only, merge --no-ff, verifica sintassi,
#      push), gestendo in automatico il filtro nbstripout non idempotente
#      che altrimenti blocca il merge su notebook "sporchi"
#   7. torna sul branch di lavoro originale
#
# Se il merge fallisce per un conflitto REALE (non nbstripout), lo script
# si ferma su main con il merge a meta' e ti dice esattamente cosa fare.
# ============================================================================
set -euo pipefail
REPO_DIR="${INVESTIA_QUANT_DIR:-$HOME/investia-quant}"
MSG="${1:-}"
if [[ -z "$MSG" ]]; then
    echo "Uso: bash docs/session_end.sh \"messaggio di commit\""
    exit 1
fi
cd "$REPO_DIR"

ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "=== 1/7 — Stato attuale (rivedi prima di continuare) ==="
git status
echo ""

echo "=== 2/7 — Verifica sintassi Python (se toccati) ==="
for f in notebooks/libs_py/r_functions.py notebooks/libs_py/u_functions.py; do
    if ! git diff --quiet -- "$f" 2>/dev/null || ! git diff --cached --quiet -- "$f" 2>/dev/null; then
        echo "Verifico sintassi: $f"
        python3 -c "import ast; ast.parse(open('$f').read())" \
            && echo "  OK" \
            || { echo "  ERRORE DI SINTASSI in $f — commit interrotto."; exit 1; }
    fi
done
echo ""

echo "=== 3/7 — Staging modifiche a file gia' tracciati (git add -u) ==="
git add -u
echo "Nota: eventuali file NUOVI non vengono aggiunti automaticamente."
echo "Se ne hai, aggiungili a mano con 'git add <path>' e rilancia."
echo ""

HAS_COMMIT=0
if git diff --cached --quiet; then
    echo "Nessuna modifica in staging sul branch di lavoro — niente da committare qui."
else
    echo "=== 4/7 — Commit ==="
    git commit -m "$MSG"
    HAS_COMMIT=1
    echo ""
fi

if [[ "$ORIGINAL_BRANCH" == "main" ]]; then
    if [[ "$HAS_COMMIT" == "1" ]]; then
        echo "=== 5/7 — Push main ==="
        git push origin main
    fi
    echo ""
    echo "=== Gia' su main, nessun merge necessario. Sessione chiusa. ==="
    exit 0
fi

echo "=== 5/7 — Push branch '$ORIGINAL_BRANCH' ==="
git push origin "$ORIGINAL_BRANCH"
echo ""

echo "=== 6/7 — Merge automatico di '$ORIGINAL_BRANCH' in main ==="
NBSTRIPOUT_CLEAN="$(git config --get filter.nbstripout.clean || true)"
git checkout main
git pull origin main --ff-only

if [[ -n "$NBSTRIPOUT_CLEAN" ]]; then
    git config --local filter.nbstripout.clean cat
fi

MERGE_OK=1
git merge "origin/$ORIGINAL_BRANCH" --no-ff \
    -m "merge: $ORIGINAL_BRANCH in main — $MSG" || MERGE_OK=0

if [[ -n "$NBSTRIPOUT_CLEAN" ]]; then
    git config --local filter.nbstripout.clean "$NBSTRIPOUT_CLEAN"
fi

if [[ "$MERGE_OK" == "0" ]]; then
    echo ""
    echo "!!! MERGE FALLITO (probabile conflitto reale, non solo nbstripout)."
    echo "!!! Sei su main con il merge a meta'. Risolvi manualmente:"
    echo "!!!   git status              (vedi i file in conflitto)"
    echo "!!!   <risolvi i conflitti>"
    echo "!!!   git add <file risolti>"
    echo "!!!   git commit"
    echo "!!!   git push origin main"
    exit 1
fi

echo ""
echo "=== 7/7 — Verifica sintassi post-merge e push main ==="
for f in notebooks/libs_py/r_functions.py notebooks/libs_py/u_functions.py investia_quant/cli.py; do
    if [[ -f "$f" ]]; then
        python3 -c "import ast; ast.parse(open('$f').read())" \
            && echo "  OK: $f" \
            || { echo "  ERRORE DI SINTASSI POST-MERGE in $f — push interrotto."; \
                 echo "  Risolvi manualmente su main prima di pushare."; exit 1; }
    fi
done
git push origin main

echo ""
echo "=== Torno sul branch di lavoro '$ORIGINAL_BRANCH' ==="
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "=== Sessione chiusa: '$ORIGINAL_BRANCH' committato, pushato, E mergiato in main. ==="
