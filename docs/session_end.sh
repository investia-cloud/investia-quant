#!/usr/bin/env bash
# ============================================================================
# session_end.sh — Fine sessione di lavoro investia-quant
#
# Uso:
#   bash docs/session_end.sh "messaggio di commit"
#
# Cosa fa:
#   1. mostra git status (per rivedere cosa sta per essere committato)
#   2. verifica sintassi Python di r_functions.py e u_functions.py, se
#      modificati (blocca il commit se la sintassi e' rotta)
#   3. git add -u   (SOLO file gia' tracciati e modificati/cancellati —
#      MAI aggiunge file nuovi in automatico, coerente con la convenzione
#      "mai git add ."; i file nuovi vanno aggiunti a mano con git add
#      <path> prima di lanciare questo script, cosi' resta una scelta
#      esplicita)
#   4. commit con il messaggio passato come argomento
#   5. push sul branch corrente
#
# Se non hai nulla da committare, lo script lo segnala e si ferma senza
# fare nulla (nessun commit vuoto).
# ============================================================================
set -euo pipefail

REPO_DIR="${INVESTIA_QUANT_DIR:-$HOME/investia-quant}"
MSG="${1:-}"

if [[ -z "$MSG" ]]; then
    echo "Uso: bash docs/session_end.sh \"messaggio di commit\""
    exit 1
fi

cd "$REPO_DIR"

echo "=== 1/5 — Stato attuale (rivedi prima di continuare) ==="
git status

echo ""
echo "=== 2/5 — Verifica sintassi Python (se toccati) ==="
for f in notebooks/libs_py/r_functions.py notebooks/libs_py/u_functions.py; do
    if ! git diff --quiet -- "$f" 2>/dev/null || ! git diff --cached --quiet -- "$f" 2>/dev/null; then
        echo "Verifico sintassi: $f"
        python3 -c "import ast; ast.parse(open('$f').read())" \
            && echo "  OK" \
            || { echo "  ERRORE DI SINTASSI in $f — commit interrotto."; exit 1; }
    fi
done

echo ""
echo "=== 3/5 — Staging modifiche a file gia' tracciati (git add -u) ==="
git add -u
echo "Nota: eventuali file NUOVI non vengono aggiunti automaticamente."
echo "Se ne hai, aggiungili a mano con 'git add <path>' e rilancia."

echo ""
if git diff --cached --quiet; then
    echo "Nessuna modifica in staging — niente da committare. Esco senza fare nulla."
    exit 0
fi

echo "=== 4/5 — Commit ==="
git commit -m "$MSG"

echo ""
echo "=== 5/5 — Push ==="
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push origin "$CURRENT_BRANCH"

echo ""
echo "=== Sessione chiusa su branch '$CURRENT_BRANCH'. Lavoro salvato su origin. ==="
