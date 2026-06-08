#!/usr/bin/env bash
# run_portfolios.sh
#
# Esegue portfolio via Papermill scegliendo automaticamente il notebook runner
# in base al "type" associato a ciascun portfolio.
#
# La lista portfolio e i destinatari sono definiti nel file esterno portfolios.conf
# (accanto allo script). Formato righe:
#   portfolio_name:type:email1,email2,...
# dove:
#   type = R (Rotational -> NOTEBOOK_ROTATIONAL) | T (Trading -> NOTEBOOK_TRADING)
#   il terzo campo (email) e' opzionale e usato da --mail customers
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
#   --rotational           (solo type=R)
#   --trading              (solo type=T)
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
EMAIL_MANAGERS="lf27963@gmail.com,customercare.ec@gmail.com"

PAPERMILL_CMD="${PAPERMILL_CMD:-papermill}"

# --- Runtime base directory ---
# NOTEBOOK_DIR="notebooks"
RUNTIME_DIR="notebooks/runtime"

# --- File esterno: lista portfolio + destinatari ---
# Formato righe: portfolio_name:type:email1,email2,...
# type = R|T ; terzo campo (email) opzionale.
# Righe vuote o che iniziano con # vengono ignorate.
PORTFOLIOS_FILE="${SCRIPT_DIR}/portfolios.conf"

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
# Valore grezzo passato a --mail. Risolto in run_one (per supportare liste per-portfolio).
MAIL_CUSTOM=""

# --- Flag: shortcut params ---
FLAG_REPORT=0
FLAG_PERFORMANCE=0
FLAG_NOREPORT=0
FLAG_NOPERFORMANCE=0

# --- Lista portfolio: "name:type" con type=R|T ---
# Popolata da PORTFOLIOS_FILE (vedi caricamento piu' sotto).
PORTFOLIOS=()

# --- Mappa destinatari "customers" per-portfolio ---
# Popolata da PORTFOLIOS_FILE (terzo campo).
declare -A RECIPIENTS_CUSTOMERS=()

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

  --rotational                  Esegue portfolio type=R
  --trading                     Esegue portfolio type=T
  --all                         Esegue portfolio type=R e type=T (default)

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
                                  --mail me        -> $EMAIL_ME
                                  --mail managers     -> $EMAIL_MANAGERS
                                  --mail customers -> lista per-portfolio dal terzo
                                                      campo di $PORTFOLIOS_FILE
                                                      (fallback su $EMAIL_ME se assente)

  -p name value                 Parametro papermill (ripetibile, qualsiasi nome)

Examples:
  $(basename "$0") --ptf world --report --mail me
  $(basename "$0") --trading --performance --mail managers
  $(basename "$0") --ptf euro_trading --mail customers
  $(basename "$0") --report --mail customers --trading
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
      # NON risolviamo subito: il valore grezzo viene risolto in run_one,
      # cosi' "customers" puo' espandersi in liste diverse per ogni portfolio.
      MAIL_CUSTOM="$2"
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

# --- Load portfolios file (richiesto) ---
# Formato righe: portfolio_name:type:email1,email2,...
# Popola PORTFOLIOS (name:type) e RECIPIENTS_CUSTOMERS (name -> emails).
if [[ ! -f "$PORTFOLIOS_FILE" ]]; then
  echo "[ERROR] portfolios-file non trovato: $PORTFOLIOS_FILE" >&2
  exit 5
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  # salta righe vuote o commenti
  [[ -z "${line// }" ]] && continue
  [[ "${line#\#}" != "$line" ]] && continue

  # split sui ':' -> name : type : mails
  pf_name="${line%%:*}"
  rest="${line#*:}"
  pf_type="${rest%%:*}"
  if [[ "$rest" == *:* ]]; then
    pf_mails="${rest#*:}"
  else
    pf_mails=""
  fi

  # trim spazi
  pf_name="${pf_name#"${pf_name%%[![:space:]]*}"}"
  pf_name="${pf_name%"${pf_name##*[![:space:]]}"}"
  pf_type="${pf_type#"${pf_type%%[![:space:]]*}"}"
  pf_type="${pf_type%"${pf_type##*[![:space:]]}"}"
  pf_mails="${pf_mails#"${pf_mails%%[![:space:]]*}"}"
  pf_mails="${pf_mails%"${pf_mails##*[![:space:]]}"}"

  if [[ -z "$pf_name" || -z "$pf_type" ]]; then
    echo "[WARN] riga portfolios.conf malformata (name/type mancante): '$line' -> skip" >&2
    continue
  fi
  if [[ "$pf_type" != "R" && "$pf_type" != "T" ]]; then
    echo "[WARN] type sconosciuto per '$pf_name' (atteso R o T): '$pf_type' -> skip" >&2
    continue
  fi

  PORTFOLIOS+=("${pf_name}:${pf_type}")
  if [[ -n "$pf_mails" ]]; then
    RECIPIENTS_CUSTOMERS["$pf_name"]="$pf_mails"
  fi
done < "$PORTFOLIOS_FILE"

if [[ "${#PORTFOLIOS[@]}" -eq 0 ]]; then
  echo "[ERROR] nessun portfolio valido in $PORTFOLIOS_FILE" >&2
  exit 6
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

# NOTA: recipient_email NON viene aggiunto qui a EXTRA_PM_PARAMS.
# Viene risolto e aggiunto per-portfolio dentro run_one (vedi resolve_recipient).

echo "[INFO] start - outdir=$OUTDIR dry_run=$DRY_RUN"
echo "[INFO] RUNTIME_DIR=$RUNTIME_DIR"
echo "[INFO] portfolios-file=$PORTFOLIOS_FILE (${#PORTFOLIOS[@]} portfolio)"
echo "[INFO] notebooks: rotational=$NOTEBOOK_ROTATIONAL trading=$NOTEBOOK_TRADING"
echo "[INFO] extra -p params: ${EXTRA_PM_PARAMS[*]}"
if [[ -n "$MAIL_CUSTOM" ]]; then
  echo "[INFO] mail (raw): $MAIL_CUSTOM"
fi
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

# --- Helper: risolve il destinatario per un dato portfolio ---
# Input:  $1 = portfolio_name
# Output: stampa la mail risolta (vuoto se nessuna mail richiesta)
# Regole:
#   ""         -> nessuna mail (stdout vuoto)
#   me         -> $EMAIL_ME
#   managers      -> $EMAIL_MANAGERS
#   customers  -> RECIPIENTS_CUSTOMERS[portfolio]  (fallback su $EMAIL_ME se assente)
#   altro      -> usato cosi' com'e' (email diretta)
resolve_recipient() {
  local portfolio_name="$1"
  case "$MAIL_CUSTOM" in
    "")
      echo ""
      ;;
    me)
      echo "$EMAIL_ME"
      ;;
    managers)
      echo "$EMAIL_MANAGERS"
      ;;
    customers)
      local m="${RECIPIENTS_CUSTOMERS[$portfolio_name]:-}"
      if [[ -z "$m" ]]; then
        echo "[WARN] nessuna lista 'customers' per '$portfolio_name': fallback su $EMAIL_ME" >&2
        echo "$EMAIL_ME"
      else
        echo "$m"
      fi
      ;;
    *)
      echo "$MAIL_CUSTOM"
      ;;
  esac
}

run_one() {
  local notebook_name="$1"   # nome file .ipynb (relativo a RUNTIME_DIR)
  local portfolio_name="$2"
  local kind="$3"            # "rotational" | "trading"

  # local out_file="../${OUTDIR}/$(basename "${notebook_name%.ipynb}")__${portfolio_name}__$(timestamp).ipynb"
  local out_file="${OUTDIR}/$(basename "${notebook_name%.ipynb}")__${portfolio_name}__$(timestamp).ipynb"

  # --- Risoluzione destinatario per-portfolio ---
  local recipient
  recipient="$(resolve_recipient "$portfolio_name")"

  # Parametri -p specifici di questo run (mail inclusa se presente)
  local per_run_params=()
  if [[ -n "$recipient" ]]; then
    per_run_params+=("-p" "recipient_email" "$recipient")
  fi

  local cmd=(
    "$PAPERMILL_CMD"
    "$notebook_name"
    "$out_file"
    -p portfolio "$portfolio_name"
    "${EXTRA_PM_PARAMS[@]}"
    "${per_run_params[@]}"
  )

  echo "----------------------------------------------------------------"
  echo "[INFO] kind      = $kind"
  echo "[INFO] notebook  = $RUNTIME_DIR/$notebook_name"
  echo "[INFO] portfolio = $portfolio_name"
  echo "[INFO] output    = $out_file"
  if [[ -n "$recipient" ]]; then
    echo "[INFO] recipient = $recipient"
  else
    echo "[INFO] recipient = (nessuno)"
  fi
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
  else
      echo "[ERROR] portfolio sconosciuto: '$SINGLE_PORTFOLIO' (non in $PORTFOLIOS_FILE)" >&2
      exit 7
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
