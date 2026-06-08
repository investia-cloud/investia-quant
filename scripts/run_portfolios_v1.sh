#!/usr/bin/env bash
# run_portfolios.sh
#
# Esegue portfolio via Papermill scegliendo automaticamente il notebook runner
# in base al "type" associato a ciascun portfolio nella lista:
#   portfolio_name:type
# dove type:
#   R -> Rotational  (NOTEBOOK_ROTATIONAL)
#   T -> Trading     (NOTEBOOK_TRADING)
#
# Supporta parametri Papermill arbitrari ripetendo: -p <name> <value>
# Supporta:
#   --portfolio "name" / --ptf "name" -> tradotto in -p portfolio "name" (single-run; tipo dedotto dalla lista)
#   --report              -> aggiunge -p send_report True
#   --noreport            -> aggiunge -p send_report False
#   --performance         -> aggiunge -p send_performance True
#   --noperformance       -> aggiunge -p send_performance False
#   --mail EMAIL          -> aggiunge -p recipient_email EMAIL
#
# Supporta filtri:
#   --only-rotational      (solo type=R)
#   --only-trading         (solo type=T)
#
# NOTA: NON passa più sender_email/recipient_email via -p se non tramite --mail o -p esplicito.
# NOTA: Esegue "cd $RUNTIME_DIR" solo prima dell'esecuzione reale (non in dry-run).
#
set -o errexit
set -o pipefail

# TSlab_HOME=/home/luca/TSlab_project
# ==============================
# Calcolo TSlab_HOME dinamico
# ==============================

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Se lo script è in TSlab/scripts/...
TSlab_HOME="$(dirname "$SCRIPT_DIR")"


VENV_HOME=/home/luca/PEnv/Investia-3

EMAIL_ME="lf27963@gmail.com"
EMAIL_TUTTI="lf27963@gmail.com,customercare.ec@gmail.com"

PAPERMILL_CMD="${PAPERMILL_CMD:-papermill}"

# --- Runtime base directory ---
# NOTEBOOK_DIR="notebooks"
RUNTIME_DIR="notebooks/runtime"

# --- Notebook runner (relativi a RUNTIME_DIR) ---
NOTEBOOK_ROTATIONAL="R_Run_Portfolio.ipynb"
NOTEBOOK_TRADING="K_Run_Portfolio.ipynb"

# --- Output --- (relativa a RUNTIME_DIR)
OUTDIR="../../outputs"

# --- Opzioni ---
ENV_FILE=""
DRY_RUN=0

# --- Filtri esecuzione ---
RUN_ROTATIONAL=0
RUN_TRADING=0

# --- Single portfolio (optional) ---
SINGLE_PORTFOLIO=""

# --- Mail custom (optional) ---
MAIL_CUSTOM=""

# --- Flag: shortcut params ---
FLAG_REPORT=0
FLAG_PERFORMANCE=0
FLAG_NOREPORT=0
FLAG_NOPERFORMANCE=0

# --- Lista portfolio: "name:type" con type=R|T ---
PORTFOLIOS=(
  "portfolio_us_trading_2026:T"
  "portfolio_euro_trading_2026:T"
  "portfolio_alpha_world:R"
  "portfolio_alpha_euro:R"
  "portfolio_alpha_sect:R"
  "portfolio_germany_plan:R"
  "portfolio_alpha_sp100:R"
  "portfolio_alpha_nasdaq100:R"
)

# --- Parametri extra per papermill (accumulo di -p name value) ---
EXTRA_PM_PARAMS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-p name value]...

Options:
  --outdir PATH                 Directory output (default: $OUTDIR)
  --env-file PATH               Source file .env (KEY=VALUE)
  --dry-run|-n                  Stampa i comandi senza eseguirli

  --notebook-dir PATH           Directory dei notebook (default: $RUNTIME_DIR)
  --notebook-rotational NAME    Runner rotational (default: $NOTEBOOK_ROTATIONAL) [relativo a RUNTIME_DIR]
  --notebook-trading NAME       Runner trading    (default: $NOTEBOOK_TRADING)    [relativo a RUNTIME_DIR]

  --rotational
                                Esegue portfolio type=R
  --trading
                                Esegue portfolio type=T
    
   --all                        Esegue portfolio type=R e type=T (default)
                              

  --portfolio,--ptf "name"      Esegue un solo portfolio (equivale a -p portfolio "name").
                                Il tipo (R/T) viene dedotto dalla lista PORTFOLIOS.
                                Supporta shortcuts:
                                  us_trading, euro_trading, world, euro, sect,
                                  germany/germania, sp100, nasdaq100

  --report                      Aggiunge: -p send_report True
  --performance                 Aggiunge: -p send_performance True
                                (se non specificati → entrambi False)

  --mail EMAIL                  Aggiunge: -p recipient_email EMAIL
                                Shortcut supportati:
                                  --mail me
                                  --mail tutti

  -p name value                 Parametro papermill (ripetibile, qualsiasi nome)

Examples:
  $(basename "$0") --ptf world --report --mail me
  $(basename "$0") --only-trading --performance --mail tutti
  $(basename "$0") --ptf euro_trading --mail desk@azienda.com \\
    -p start_date 2024-01-01
EOF
  exit 1
}


FLAG_REPORT=0
FLAG_PERFORMANCE=0


# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) OUTDIR="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --dry-run|-n) DRY_RUN=1; shift ;;

    --notebook-dir) RUNTIME_DIR="$2"; shift 2 ;;
    --notebook-rotational) NOTEBOOK_ROTATIONAL="$2"; shift 2 ;;
    --notebook-trading) NOTEBOOK_TRADING="$2"; shift 2 ;;

    --all)
      RUN_ROTATIONAL=1
      RUN_TRADING=1
      shift
      ;;

    --rotational)
      # RUN_TRADING=0
      RUN_ROTATIONAL=1
      shift
      ;;

    --trading)
      # RUN_ROTATIONAL=0
      RUN_TRADING=1
      shift
      ;;

    --portfolio|--ptf)
      SINGLE_PORTFOLIO="$2"

      # --- Shortcuts per portfolio ---
      case "$SINGLE_PORTFOLIO" in
        us_trading)   SINGLE_PORTFOLIO="portfolio_us_trading_2026" ;;
        euro_trading) SINGLE_PORTFOLIO="portfolio_euro_trading_2026" ;;
        world)        SINGLE_PORTFOLIO="portfolio_alpha_world" ;;
        euro)         SINGLE_PORTFOLIO="portfolio_alpha_euro" ;;
        sect)         SINGLE_PORTFOLIO="portfolio_alpha_sect" ;;
        germany|germania) SINGLE_PORTFOLIO="portfolio_germany_plan" ;;
        sp100)        SINGLE_PORTFOLIO="portfolio_alpha_sp100" ;;
        nasdaq100)    SINGLE_PORTFOLIO="portfolio_alpha_nasdaq100" ;;
      esac

      shift 2
      ;;

    --mail)
      case "$2" in
        me)
          MAIL_CUSTOM="$EMAIL_ME"
          ;;
        tutti)
          MAIL_CUSTOM="$EMAIL_TUTTI"
          ;;
        *)
          MAIL_CUSTOM="$2"
          ;;
      esac
      shift 2
      ;;

    --report)
      FLAG_REPORT=1
      shift
      ;;

    --performance)
      FLAG_PERFORMANCE=1
      shift
      ;;
      
    -p)
      EXTRA_PM_PARAMS+=("-p" "$2" "$3")
      shift 3
      ;;

    -h|--help) usage ;;
    *)
      echo "Opzione sconosciuta: $1" >&2
      usage
      ;;
  esac
done

cd "$TSlab_HOME"

# --- Load env file (optional) ---
if [[ -n "$ENV_FILE" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  else
    echo "[WARN] env-file non trovato: $ENV_FILE" >&2
  fi
fi

# Attiva l’ambiente virtuale
# shellcheck disable=SC1091
source "$VENV_HOME/bin/activate"

# --- Basic checks ---
if ! command -v "$PAPERMILL_CMD" >/dev/null 2>&1; then
  echo "[ERROR] papermill non trovato (cmd: $PAPERMILL_CMD)" >&2
  exit 2
fi

if [[ "$RUN_ROTATIONAL" -eq 1 && ! -f "$RUNTIME_DIR/$NOTEBOOK_ROTATIONAL" ]]; then
  echo "[ERROR] notebook rotational non trovato: $RUNTIME_DIR/$NOTEBOOK_ROTATIONAL" >&2
  exit 3
fi

if [[ "$RUN_TRADING" -eq 1 && ! -f "$RUNTIME_DIR/$NOTEBOOK_TRADING" ]]; then
  echo "[ERROR] notebook trading non trovato: $RUNTIME_DIR/$NOTEBOOK_TRADING" >&2
  exit 4
fi

# mkdir -p "$OUTDIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }


# --- Applica shortcut flags come parametri papermill ---
if [[ "$FLAG_REPORT" -eq 1 ]]; then
  EXTRA_PM_PARAMS+=("-p" "send_report" "True")
else
  EXTRA_PM_PARAMS+=("-p" "send_report" "False")
fi

if [[ "$FLAG_PERFORMANCE" -eq 1 ]]; then
  EXTRA_PM_PARAMS+=("-p" "send_performance" "True")
else
  EXTRA_PM_PARAMS+=("-p" "send_performance" "False")
fi

# --- Mail shortcut: --mail email -> -p recipient_email email ---
if [[ -n "$MAIL_CUSTOM" ]]; then
  EXTRA_PM_PARAMS+=("-p" "recipient_email" "$MAIL_CUSTOM")
fi

echo "[INFO] start - outdir=$OUTDIR dry_run=$DRY_RUN"
echo "[INFO] RUNTIME_DIR=$RUNTIME_DIR"
echo "[INFO] notebooks: rotational=$NOTEBOOK_ROTATIONAL trading=$NOTEBOOK_TRADING"
echo "[INFO] extra -p params: ${EXTRA_PM_PARAMS[*]}"
if [[ -n "$SINGLE_PORTFOLIO" ]]; then
  echo "[INFO] single portfolio: $SINGLE_PORTFOLIO"
fi

# --- Helper: trova type (R/T) dato un portfolio name nella lista PORTFOLIOS ---
get_portfolio_type() {
  local name="$1"
  local item n t
  for item in "${PORTFOLIOS[@]}"; do
    n="${item%%:*}"
    t="${item##*:}"
    if [[ "$n" == "$name" ]]; then
      echo "$t"
      return 0
    fi
  done
  echo ""
  return 0
}

run_one() {
  local notebook_name="$1"   # nome file .ipynb (relativo a RUNTIME_DIR)
  local portfolio_name="$2"
  local kind="$3"            # "rotational" | "trading"

  # local out_file="../${OUTDIR}/$(basename "${notebook_name%.ipynb}")__${portfolio_name}__$(timestamp).ipynb"
  local out_file="${OUTDIR}/$(basename "${notebook_name%.ipynb}")__${portfolio_name}__$(timestamp).ipynb"

  local cmd=(
    "$PAPERMILL_CMD"
    "$notebook_name"
    "$out_file"
    -p portfolio "$portfolio_name"
    "${EXTRA_PM_PARAMS[@]}"
  )

  echo "----------------------------------------------------------------"
  echo "[INFO] kind      = $kind"
  echo "[INFO] notebook  = $RUNTIME_DIR/$notebook_name"
  echo "[INFO] portfolio = $portfolio_name"
  echo "[INFO] output    = $out_file"
  echo "[INFO] CMD       = (cd $RUNTIME_DIR && ${cmd[*]})"
  echo "----------------------------------------------------------------"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  (
    cd "$RUNTIME_DIR" || exit 1
    "${cmd[@]}"
  ) || {
    echo "[ERROR] papermill failed ($kind) for $portfolio_name" >&2
    return 1
  }

  echo "[OK] completed ($kind) $portfolio_name -> $out_file"
  return 0
}

# --- Single-run mode ---
if [[ -n "$SINGLE_PORTFOLIO" ]]; then
  ptype="$(get_portfolio_type "$SINGLE_PORTFOLIO")"
  if [[ "$ptype" == "R" ]]; then
      run_one "$NOTEBOOK_ROTATIONAL" "$SINGLE_PORTFOLIO" "rotational"
  elif [[ "$ptype" == "T" ]]; then
      run_one "$NOTEBOOK_TRADING" "$SINGLE_PORTFOLIO" "trading"
  fi
  echo "[INFO] finished (single-run)"
  exit 0
fi

# --- Batch mode: scorre PORTFOLIOS "name:type" ---
for item in "${PORTFOLIOS[@]}"; do
  name="${item%%:*}"
  typ="${item##*:}"

  if [[ "$typ" == "R" ]]; then
    [[ "$RUN_ROTATIONAL" -eq 1 ]] || continue
    run_one "$NOTEBOOK_ROTATIONAL" "$name" "rotational" || true
  elif [[ "$typ" == "T" ]]; then
    [[ "$RUN_TRADING" -eq 1 ]] || continue
    run_one "$NOTEBOOK_TRADING" "$name" "trading" || true
  else
    echo "[WARN] type sconosciuto per '$item' (atteso R o T): skip" >&2
  fi
done

echo "[INFO] finished"
