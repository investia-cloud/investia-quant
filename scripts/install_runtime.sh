#!/bin/bash

# ==============================
# Configurazione
# ==============================

REMOTE_USER="luca"
REMOTE_HOST="tslab.investia.cloud"
REMOTE_BASE_DIR="/opt/TSlab"      # <-- modifica se necessario
SSH_PORT=22                       # <-- modifica se necessario

# Directory locali da sincronizzare
DIRS_TO_SYNC=(
    "inputs"
    "notebooks/libs"
    "notebooks/runtime"
    "Portal"
    "scripts"
)

# ==============================
# Parsing opzioni
# ==============================

DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Opzione sconosciuta: $1"
      exit 1
      ;;
  esac
done

# Flag rsync dry-run
RSYNC_DRY=""
if [ "$DRY_RUN" = true ]; then
  echo "Modalità DRY-RUN attiva (nessuna modifica verrà effettuata)"
  RSYNC_DRY="--dry-run -v"
fi

# ==============================
# Controlli preliminari
# ==============================

if [ ! -d "inputs" ]; then
    echo "Errore: eseguire lo script dalla root del progetto TSlab."
    exit 1
fi

echo "Deploy runtime TSlab verso ${REMOTE_USER}@${REMOTE_HOST}"
echo "Directory remota: ${REMOTE_BASE_DIR}"
echo "---------------------------------------------"

# ==============================
# Creazione directory remota base
# ==============================

ssh -p ${SSH_PORT} ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_BASE_DIR}"

# ==============================
# Sync directory principali
# ==============================

for DIR in "${DIRS_TO_SYNC[@]}"; do
    echo "Sincronizzo ${DIR}..."

    rsync -az --delete --mkpath ${RSYNC_DRY}  \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.ipynb_checkpoints' \
        -e "ssh -p ${SSH_PORT}" \
        "${DIR}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/${DIR}/"
done

# ==============================
# Creazione directory outputs e cache (vuote)
# ==============================

for EMPTY_DIR in outputs cache; do
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Creerei directory ${EMPTY_DIR} e la svuoterei"
    else
        ssh -p ${SSH_PORT} ${REMOTE_USER}@${REMOTE_HOST} \
            "mkdir -p ${REMOTE_BASE_DIR}/${EMPTY_DIR} && rm -rf ${REMOTE_BASE_DIR}/${EMPTY_DIR}/*"
    fi
done

echo "---------------------------------------------"
echo "Deploy completato correttamente."