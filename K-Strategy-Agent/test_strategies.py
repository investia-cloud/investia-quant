"""
test_strategies.py
──────────────────
Script standalone che testa tutte le K-Strategies presenti in strategies.ipynb
sui ticker dei portafogli portfolio_us_trading_2026 e portfolio_euro_trading_2026,
usando wfo_strategy_panel con override=False per saltare combinazioni già testate.

Uso:
    python test_strategies.py               # esegue subito
    python test_strategies.py --dry-run     # mostra cosa verrebbe lanciato senza eseguire
    python test_strategies.py --schedule    # esegue subito + schedula ogni giorno alle 09:00
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import nbformat
import schedule
import time

# ─────────────────────────────────────────────
# PATH BASE
# ─────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
LIBS_DIR       = PROJECT_ROOT / "notebooks" / "libs"
STRATEGIES_NB  = Path(__file__).parent / "strategies.ipynb"
WFO_RESULTS_DIR    = str(PROJECT_ROOT / "outputs" / "WFO_T_DEV_RESULTS")
EXCEL_RESULTS_FILE = PROJECT_ROOT / "outputs" / "WFO_T_DEV_RESULTS" / "test_results_history.xlsx"

DAILY_RUN_TIME = "09:00"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# MODULO 1 – ESECUZIONE NOTEBOOK
# ═══════════════════════════════════════════════════════════════

def _exec_notebook(nb_path: Path, ns: dict) -> None:
    """
    Esegue tutte le celle code di un notebook nel namespace ns.
    Gestisce %run <path.ipynb> ricorsivamente.
    Ignora le altre magic IPython.
    """
    nb_path = nb_path.resolve()
    log.debug(f"  exec_notebook: {nb_path.name}")

    nb = nbformat.read(str(nb_path), as_version=4)

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        src = cell.source

        # Gestisci %run <qualcosa.ipynb>
        def _handle_run(m):
            rel = m.group(1).strip().strip("'\"")
            target = (nb_path.parent / rel).resolve()
            if target.suffix == ".ipynb" and target.exists():
                _exec_notebook(target, ns)
            return ""  # rimuovi la riga %run dall'exec successivo

        src = re.sub(r"^%run\s+(.+)$", _handle_run, src, flags=re.MULTILINE)

        # Rimuovi altre magic IPython
        src = re.sub(r"^\s*%[^\n]*", "", src, flags=re.MULTILINE)

        # Salta celle vuote dopo la pulizia
        if not src.strip():
            continue

        try:
            exec(compile(src, str(nb_path), "exec"), ns)
        except Exception as e:
            log.debug(f"  [skip] {nb_path.name}: {e}")


def build_namespace(strategies_nb: Path = STRATEGIES_NB) -> dict:
    """
    Carica nell'ordine del _bootstrap_dev tutti i notebook necessari
    e restituisce il namespace condiviso con tutte le funzioni disponibili.

    Parameters
    ----------
    strategies_nb : Path
        Notebook delle strategie da caricare (default: STRATEGIES_NB).
    """
    ns: dict = {"__builtins__": __builtins__}

    # Stessa sequenza di _bootstrap_dev.ipynb (senza r_functions, mc_functions, ecc.)
    notebooks = [
        LIBS_DIR / "k_functions.ipynb",
        LIBS_DIR / "k_tickers.ipynb",
        LIBS_DIR / "k_portfolios.ipynb",
        LIBS_DIR / "u_functions.ipynb",
    ]

    for nb_path in notebooks:
        if not nb_path.exists():
            log.warning(f"Notebook non trovato, skip: {nb_path}")
            continue
        log.info(f"Carico: {nb_path.name}")
        _exec_notebook(nb_path, ns)

    # Carica anche le strategie generate dall'agente
    if strategies_nb.exists():
        log.info(f"Carico: {strategies_nb.name}")
        _exec_notebook(strategies_nb, ns)
    else:
        log.warning(f"Notebook strategie non trovato: {strategies_nb}")

    return ns


# ═══════════════════════════════════════════════════════════════
# MODULO 2 – ESTRAZIONE TICKERS E STRATEGIE
# ═══════════════════════════════════════════════════════════════

def get_test_tickers(ns: dict) -> list[str]:
    """
    Estrae i ticker unici da portfolio_us_trading_2026 e
    portfolio_euro_trading_2026, scartando i duplicati.
    """
    tickers: list[str] = []
    seen: set[str] = set()

    for portfolio_name in ("portfolio_us_trading_2026", "portfolio_euro_trading_2026"):
        portfolio = ns.get(portfolio_name)
        if portfolio is None:
            log.warning(f"Portfolio non trovato nel namespace: {portfolio_name}")
            continue
        for ts in portfolio.get("trading_systems", []):
            sym = ts.get("symbol")
            if sym and sym not in seen:
                tickers.append(sym)
                seen.add(sym)

    log.info(f"Ticker da testare: {len(tickers)}  →  {tickers}")
    return tickers


def get_strategy_names(ns: dict) -> list[str]:
    """
    Restituisce i nomi di tutte le strategie disponibili nel namespace,
    individuate dalla presenza di strategy_<name>_param_ranges.
    """
    names = [
        key.replace("_param_ranges", "").replace("strategy_", "")
        for key in ns
        if key.startswith("strategy_") and key.endswith("_param_ranges")
    ]
    log.info(f"Strategie trovate: {len(names)}  →  {names}")
    return names


# ═══════════════════════════════════════════════════════════════
# MODULO 2b – CHECK STATICI DI PERFORMANCE
# ═══════════════════════════════════════════════════════════════

# Pattern noti per essere lenti, con relativo messaggio diagnostico.
# Ogni entry è (compiled_re, msg) oppure (callable(src)->bool, msg) per check complessi.
_PERF_CHECKS: list[tuple] = [
    (
        re.compile(r"\.apply\s*\(.*?raw\s*=\s*False", re.DOTALL),
        "rolling.apply(raw=False): crea un pandas.Series per ogni finestra — usa raw=True con lambda numpy",
    ),
    (
        # rolling.apply presente ma raw= completamente assente nella cella
        lambda src: bool(re.search(r"\.rolling\b", src) and re.search(r"\.apply\s*\(", src))
                    and "raw=" not in src,
        "rolling.apply senza raw=True: aggiungi raw=True per evitare overhead pandas",
    ),
    (
        re.compile(r"pd\.Series\s*\(x\)"),
        "pd.Series(x) dentro rolling lambda: costruisce un oggetto pandas per ogni finestra — usa operazioni numpy dirette su x",
    ),
    (
        re.compile(r"pd\.concat\s*\(", re.DOTALL),
        "pd.concat nelle funzioni indicatore: usa np.maximum(np.maximum(a, b), c) per il True Range — evita DataFrame temporanei",
    ),
    (
        re.compile(r"(?<!\bsafe_)\b\w+\s*/\s*\w+_sum\b"),
        "possibile divisione per zero su _sum: usa il pattern safe-denominator (np.where) per evitare RuntimeWarning e NaN",
    ),
    (
        re.compile(r"for\s+\w+\s+in\s+range\s*\(.*\)\s*:.*\.iloc\[", re.DOTALL),
        "loop Python con .iloc: converte gli array in numpy (.values) e usa indici interi — speedup tipico 50-100x",
    ),
]

_MAX_PARAM_COMBINATIONS = 1500   # soglia: ~10000 step WFO con 7 finestre


def check_strategy_performance(strategy_name: str, source: str) -> list[str]:
    """
    Analisi statica del sorgente di una strategia alla ricerca di pattern
    noti per degradare le performance nel WFO.
    Restituisce una lista di warning (stringhe). Lista vuota = nessun problema.
    """
    warnings_found = []
    for check, msg in _PERF_CHECKS:
        if callable(check):
            triggered = check(source)
        else:
            triggered = bool(check.search(source))
        if triggered:
            warnings_found.append(msg)
    return warnings_found


def check_param_grid_size(strategy_name: str, source: str) -> list[str]:
    """
    Conta le combinazioni del param grid della strategia e segnala se supera
    _MAX_PARAM_COMBINATIONS. Restituisce lista di warning (vuota = OK).
    """
    import ast
    from math import prod
    ranges = re.findall(r"'[^']+_range'\s*:\s*(range\([^)]+\))", source)
    if not ranges:
        return []
    try:
        counts = [len(list(eval(r))) for r in ranges]
        total = prod(counts)
    except Exception:
        return []
    if total > _MAX_PARAM_COMBINATIONS:
        detail = " × ".join(str(c) for c in counts)
        return [
            f"griglia parametri troppo grande: {detail} = {total} combinazioni "
            f"(~{total * 7} step WFO, soglia: {_MAX_PARAM_COMBINATIONS}) — "
            f"aumenta il passo dei range per ridurre a ≤{_MAX_PARAM_COMBINATIONS} combinazioni"
        ]
    return []


def check_all_strategies_performance(strategies_nb_path: Path) -> dict[str, list[str]]:
    """
    Esegue check_strategy_performance su ogni cella del notebook strategies.ipynb
    che contiene una strategia (riconosciuta da 'def strategy_').
    Restituisce {strategy_name: [warning, ...]}. Solo strategie con warning.
    """
    import json
    issues: dict[str, list[str]] = {}
    if not strategies_nb_path.exists():
        return issues
    with open(strategies_nb_path) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        m = re.search(r"def strategy_([a-z0-9_]+)\s*\(", src)
        if not m:
            continue
        name = m.group(1)
        found = check_strategy_performance(name, src) + check_param_grid_size(name, src)
        if found:
            issues[name] = found
    return issues


# ═══════════════════════════════════════════════════════════════
# MODULO 3 – TEST
# ═══════════════════════════════════════════════════════════════

def run_strategy_tests(
    dry_run         : bool            = False,
    only_tickers    : list[str] | None = None,
    only_strategies : list[str] | None = None,
    override        : bool            = False,
    scenario        : str             = "B",
    strategies_nb   : Path            = STRATEGIES_NB,
    _ns             : dict | None     = None,
) -> "pd.DataFrame | None":
    log.info("═══ Avvio test strategie ═══")
    log.info(f"  WFO results dir : {WFO_RESULTS_DIR}")
    log.info(f"  Dry-run         : {dry_run}")
    log.info(f"  Strategie NB    : {strategies_nb}")

    # 1. Costruisci namespace (riutilizza se passato dalla pipeline)
    if _ns is not None:
        ns = _ns
    else:
        log.info("Caricamento namespace...")
        ns = build_namespace(strategies_nb=strategies_nb)

    # 1b. Check statici di performance (solo al primo caricamento, non nelle run interne della pipeline)
    if _ns is None:
        perf_issues = check_all_strategies_performance(strategies_nb)
    else:
        perf_issues = {}
    if perf_issues:
        log.warning("⚠️  Check performance: %d strategia/e con pattern lenti:", len(perf_issues))
        for strat, msgs in perf_issues.items():
            for msg in msgs:
                log.warning("   [%s] %s", strat, msg)
    else:
        log.info("✅ Check performance: nessun pattern lento rilevato.")

    # 2. Tickers e strategie
    tickers    = only_tickers    if only_tickers    else get_test_tickers(ns)
    strategies = only_strategies if only_strategies else get_strategy_names(ns)

    if only_tickers:
        log.info(f"Filtro ticker applicato: {tickers}")
    if only_strategies:
        available = get_strategy_names(ns)
        validated = []
        for name in only_strategies:
            if name in available:
                validated.append(name)
            else:
                # prova match parziale (prefisso)
                matches = [a for a in available if a.startswith(name)]
                if matches:
                    log.info(f"  '{name}' → match parziale: {matches}")
                    validated.extend(matches)
                else:
                    log.error(
                        f"  Strategia '{name}' non trovata nel namespace.\n"
                        f"  Strategie disponibili: {available}"
                    )
        strategies = validated
        if not strategies:
            log.error("Nessuna strategia valida dopo il filtro. Interruzione.")
            return
        log.info(f"Filtro strategie applicato: {strategies}")

    if not tickers:
        log.error("Nessun ticker trovato. Interruzione.")
        return None
    if not strategies:
        log.error("Nessuna strategia trovata. Interruzione.")
        return None

    # 3. Chiama wfo_strategy_panel
    wfo_fn = ns.get("wfo_strategy_panel")
    if wfo_fn is None:
        log.error("wfo_strategy_panel non trovata nel namespace. Interruzione.")
        return None

    Path(WFO_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    log.info(f"Lancio wfo_strategy_panel: {len(tickers)} ticker × {len(strategies)} strategie")

    wfo_kwargs = dict(
        start_date      = "2015-01-01",
        end_date        = datetime.now().strftime("%Y-%m-%d"),
        ratio           = "4:1",
        init_cash       = 100_000.0,
        fees            = 0.001,
        slippage        = 0.0,
        show_progress   = True,
        save_results    = True,
        wfo_results_dir = WFO_RESULTS_DIR,
        override        = override,
        namespace       = ns,
        dry_run         = dry_run,
        scenario        = scenario,
    )

    frames = []
    for ticker in tickers:
        for strategy in strategies:
            log.info("Running %s@%s", ticker, strategy)
            df_i, _, _ = wfo_fn(tickers=[ticker], strategies=[strategy], **wfo_kwargs)
            if df_i is not None and not df_i.empty:
                frames.append(df_i)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    _print_summary(df, tickers, strategies)
    _save_results_excel(df, scenario)
    log.info("═══ Test strategie completato ═══")
    return df


def _save_results_excel(df: "pd.DataFrame", scenario: str) -> None:
    """
    Appende i risultati del run corrente al file Excel storico.
    Se il file non esiste lo crea. Ogni riga include run_date e scenario.
    """
    if df is None or df.empty:
        return

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        log.warning("  openpyxl non installato — skip salvataggio Excel (pip install openpyxl)")
        return

    out = df.copy()
    out.insert(0, "scenario",  scenario)
    out.insert(0, "run_date",  datetime.now().strftime("%Y-%m-%d %H:%M"))

    excel_path = EXCEL_RESULTS_FILE
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if excel_path.exists():
            existing = pd.read_excel(excel_path, sheet_name="Results")
            combined = pd.concat([existing, out], ignore_index=True)
        else:
            combined = out

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="Results", index=False)

        log.info("  Risultati Excel aggiornati: %s  (%d righe totali)", excel_path, len(combined))
    except Exception as e:
        log.warning("  Errore salvataggio Excel: %s", e)


def _print_summary(df, tickers: list, strategies: list) -> None:
    """Stampa un riepilogo leggibile dei risultati WFO."""
    total   = len(tickers) * len(strategies)
    ran     = len(df) if df is not None else 0
    failed  = total - ran

    print()
    print("=" * 70)
    print("  RIEPILOGO TEST STRATEGIE")
    print("=" * 70)
    print(f"  Combinazioni testate : {total}  ({len(tickers)} ticker × {len(strategies)} strategie)")
    print(f"  Precheck superato    : {ran}")
    print(f"  Scartate (precheck)  : {failed}")
    print()

    if df is not None and not df.empty:
        metric_cols = ["Ticker", "Strategy", "Sharpe", "CAR %", "Total Return %", "MaxDD %"]
        show_cols   = [c for c in metric_cols if c in df.columns]
        print("  Strategie che hanno completato WFO:")
        print()

        # formattazione manuale per allineamento
        col_w = {"Ticker": 8, "Strategy": 30, "Sharpe": 8,
                 "CAR %": 8, "Total Return %": 14, "MaxDD %": 9}
        header = "".join(c.ljust(col_w.get(c, 12)) for c in show_cols)
        print("  " + header)
        print("  " + "-" * len(header))
        for _, row in df[show_cols].iterrows():
            line = ""
            for c in show_cols:
                v = row[c]
                if isinstance(v, float):
                    cell = f"{v:.2f}"
                else:
                    cell = str(v) if v is not None else "N/A"
                line += cell.ljust(col_w.get(c, 12))
            print("  " + line)
        print()
    else:
        print("  Nessuna strategia ha superato il precheck.")
        print()
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
# MODULO 4 – PIPELINE B → E
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    dry_run         : bool            = False,
    only_tickers    : list[str] | None = None,
    only_strategies : list[str] | None = None,
    override        : bool            = False,
    strategies_nb   : Path            = STRATEGIES_NB,
) -> None:
    """
    Pipeline di selezione in due stadi:

      Stage 1 — Scenario B (Gate1 + Gate2 ≥40%)
          Precheck veloce su tutti i candidati.
          Solo le combinazioni (ticker, strategia) che superano B
          vengono promosse allo stage successivo.

      Stage 2 — Scenario E (Gate1 + Gate2 ≥50% + Gate3)
          Valutazione completa con tutti i gate attivi.
          I sopravvissuti sono i candidati per il deploy in produzione.
    """
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  PIPELINE SELEZIONE PRODUZIONE  (B → E)             ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # ── Stage 1: Scenario B ──────────────────────────────────────
    log.info("── Stage 1: Scenario B — filtro base ──────────────────")
    log.info("Caricamento namespace...")
    ns = build_namespace(strategies_nb=strategies_nb)

    perf_issues = check_all_strategies_performance(strategies_nb)
    if perf_issues:
        log.warning("⚠️  Check performance: %d strategia/e con pattern lenti:", len(perf_issues))
        for strat, msgs in perf_issues.items():
            for msg in msgs:
                log.warning("   [%s] %s", strat, msg)
    else:
        log.info("✅ Check performance: nessun pattern lento rilevato.")

    df_b = run_strategy_tests(
        dry_run         = dry_run,
        only_tickers    = only_tickers,
        only_strategies = only_strategies,
        override        = override,
        scenario        = "B",
        strategies_nb   = strategies_nb,
        _ns             = ns,
    )

    if df_b is None or df_b.empty:
        log.warning("Nessun candidato ha superato lo Scenario B. Pipeline terminata.")
        return

    # Sopravvissuti al Stage 1
    surv_tickers    = sorted(df_b["Ticker"].unique().tolist())   if "Ticker"   in df_b.columns else (only_tickers or [])
    surv_strategies = sorted(df_b["Strategy"].unique().tolist()) if "Strategy" in df_b.columns else (only_strategies or [])

    log.info("── Stage 1 completato ──────────────────────────────────")
    log.info("  Candidati promossi : %d  (%d ticker × %d strategie)",
             len(df_b), len(surv_tickers), len(surv_strategies))
    log.info("  Ticker    : %s", surv_tickers)
    log.info("  Strategie : %s", surv_strategies)

    # ── Stage 2: Scenario E ──────────────────────────────────────
    log.info("── Stage 2: Scenario E — selezione produzione ─────────")

    df_e = run_strategy_tests(
        dry_run         = dry_run,
        only_tickers    = surv_tickers,
        only_strategies = surv_strategies,
        override        = True,   # sempre override: ricalcola con gates più stringenti
        scenario        = "E",
        strategies_nb   = strategies_nb,
        _ns             = ns,
    )

    # ── Riepilogo finale pipeline ────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  RIEPILOGO PIPELINE B → E                           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Candidati dopo Stage 1 (Scenario B) : {len(df_b)}")
    n_e = len(df_e) if df_e is not None and not df_e.empty else 0
    print(f"  Candidati dopo Stage 2 (Scenario E) : {n_e}")
    print()
    if n_e > 0:
        print("  ✅ Trading system candidati per il deploy:")
        metric_cols = ["Ticker", "Strategy", "Sharpe", "CAR %", "Total Return %", "MaxDD %"]
        show_cols   = [c for c in metric_cols if c in df_e.columns]
        col_w = {"Ticker": 8, "Strategy": 30, "Sharpe": 8,
                 "CAR %": 8, "Total Return %": 14, "MaxDD %": 9}
        header = "".join(c.ljust(col_w.get(c, 12)) for c in show_cols)
        print("  " + header)
        print("  " + "─" * len(header))
        for _, row in df_e[show_cols].iterrows():
            line = "".join(
                (f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c] or "N/A"))
                .ljust(col_w.get(c, 12))
                for c in show_cols
            )
            print("  " + line)
    else:
        print("  ❌ Nessun trading system ha superato tutti i gate di produzione.")
    print()


# ═══════════════════════════════════════════════════════════════

_HELP_DESCRIPTION = """\
test_strategies.py — Testa le K-Strategies su Walk-Forward Optimization (WFO)

Carica strategies.ipynb e le librerie del progetto, esegue un precheck di
robustezza parametrica e poi la WFO completa per ogni combinazione
ticker × strategia. I risultati vengono salvati in:
  outputs/WFO_T_DEV_RESULTS/<strategia>/

SCENARI PRECHECK (--scenario)
──────────────────────────────
  A  Esplorazione    Gate1 ON                          Permissivo, primo screening
  B  Qualità base    Gate1 ON + Gate2 (≥40% robusti)  Default — strategie agente
  C  Qualità media   Gate1 ON + Gate2 (≥50% robusti)  Test sistematici bilanciati
  D  Notebook        Gate1 ON + Gate3 (recommend_wfo)  Giudizio composito
  E  Produzione      Gate1 ON + Gate2 (≥50%) + Gate3   Tutti i gate, solo deploy

DESCRIZIONE DEI GATE
──────────────────────────────
  Gate1  Best-param vs Buy & Hold (obbligatorio in tutti gli scenari)
         Il parametro migliore trovato in-sample deve produrre un rendimento
         totale superiore al Buy & Hold sullo stesso periodo. Filtra le
         strategie che non battono il mercato neanche nelle condizioni ottimali.

  Gate2  Robustezza parametrica (percentuale di param set che batte B&H)
         Misura quanti parametri su N campionati (precheck stress) battono il
         Buy & Hold. Una strategia robusta funziona su un'ampia zona del
         parametro-spazio, non solo sul picco. Soglia: ≥40%% (scenario B) o
         ≥50%% (scenari C ed E). Filtra le strategie overfit su un singolo punto.

  Gate3  Giudizio composito recommend_wfo
         Verdetto sintetico prodotto dall'analisi di overfitting: considera
         la forma della superficie IS vs OOS, la consistenza tra finestre WFO,
         e la presenza di plateau stabili nel parametro-spazio. Più selettivo
         dei soli gate numerici, ma anche più soggettivo. Usato in produzione
         (scenari D ed E) per un ulteriore filtro qualitativo.

PIPELINE DI SELEZIONE (--pipeline)
──────────────────────────────
  Implementa il flusso logico consigliato per identificare i trading system
  candidati al deploy, in due stadi sequenziali:

    Stage 1 — Scenario B  (filtro base, veloce)
      Eseguito su tutti i candidati. Scarta le strategie che non superano
      Gate1 + Gate2≥40%%. Chi non passa qui non può passare lo Stage 2.

    Stage 2 — Scenario E  (selezione produzione, completo)
      Eseguito solo sui sopravvissuti del Stage 1. Applica tutti e tre i gate.
      L'output è la lista definitiva dei trading system pronti per il deploy.

  GARANZIA DI MONOTONIA
  La pipeline è logicamente coerente: Scenario E è strettamente più
  selettivo di Scenario B (richiede Gate2≥50%% invece di ≥40%%, più Gate3).
  Pertanto: pass(E) ⊆ pass(B) ⊆ pass(A).
  Una strategia che non supera B non può superare E — non ci sono eccezioni.

  Il namespace viene caricato una sola volta e riutilizzato nei due stadi,
  quindi --pipeline non ha overhead aggiuntivo rispetto a due run separate.

CHECK STATICI DI PERFORMANCE
──────────────────────────────
  All'avvio viene eseguita un'analisi statica su strategies.ipynb che segnala
  pattern noti per essere lenti nel WFO (rolling.apply raw=False, loop con .iloc,
  pd.concat per il True Range, divisioni per zero, ecc.).

ESEMPI
──────────────────────────────
  # Esecuzione standard su tutti i ticker e strategie
  python test_strategies.py

  # Solo due ticker, scenario di qualità media
  python test_strategies.py --tickers NVDA AAPL --scenario C

  # Ritesta tutto da zero (ignora risultati già salvati)
  python test_strategies.py --override

  # Testa una singola strategia su un ticker con scenario deploy
  python test_strategies.py --tickers INTC --strategies cms_keltner_srpr --scenario E

  # Mostra cosa verrebbe eseguito senza lanciare il WFO
  python test_strategies.py --dry-run --tickers MSFT

  # Esegui subito e poi schedula ogni giorno alle {daily_run_time}
  python test_strategies.py --schedule

  # Pipeline B → E: filtra con Scenario B poi seleziona candidati deploy con Scenario E
  python test_strategies.py --pipeline
  python test_strategies.py --pipeline --tickers NVDA INTC MSFT
  python test_strategies.py --pipeline --override   # ripartenza da zero
""".format(daily_run_time=DAILY_RUN_TIME)


def main():
    parser = argparse.ArgumentParser(
        description=_HELP_DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra ticker e strategie che verrebbero testati, senza eseguire il WFO",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help=f"Esegui subito, poi ripeti ogni giorno alle {DAILY_RUN_TIME}",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Limita il test a questi ticker  (es. --tickers NVDA AAPL MSFT)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        metavar="STRATEGY",
        help="Limita il test a queste strategie  (es. --strategies cms_keltner_srpr dbma_matrix)\n"
             "Supporta prefissi parziali: --strategies cms corrisponde a cms_keltner_srpr",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Ricalcola e sovrascrive i risultati WFO già salvati su disco\n"
             "Senza questo flag le combinazioni già testate vengono saltate",
    )
    parser.add_argument(
        "--scenario",
        default="B",
        metavar="SCENARIO",
        help=(
            "Scenario precheck da usare (default: B):\n"
            "  A  Esplorazione  — Gate1 only, passa quasi tutto\n"
            "  B  Qualità base  — Gate1 + Gate2 ≥40%%  (default)\n"
            "  C  Qualità media — Gate1 + Gate2 ≥50%%\n"
            "  D  Notebook      — Gate1 + Gate3 (recommend_wfo)\n"
            "  E  Produzione    — Gate1 + Gate2 ≥50%% + Gate3  (deploy)\n"
            "Ignorato se --pipeline è attivo (la pipeline usa B poi E automaticamente)"
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "Esegui la pipeline di selezione in due stadi:\n"
            "  Stage 1: Scenario B — filtro base veloce su tutti i candidati\n"
            "  Stage 2: Scenario E — valutazione completa solo sui sopravvissuti\n"
            "Produce la lista dei trading system candidati per il deploy"
        ),
    )
    parser.add_argument(
        "--strategies-nb",
        metavar="PATH",
        default=None,
        help=(
            f"Notebook delle strategie da caricare\n"
            f"(default: {STRATEGIES_NB})\n"
            f"Es.: --strategies-nb ../K-Strategy-Agent/strategies_v2.ipynb"
        ),
    )
    args = parser.parse_args()

    strategies_nb = Path(args.strategies_nb) if args.strategies_nb else STRATEGIES_NB

    if args.pipeline:
        run_pipeline(
            dry_run         = args.dry_run,
            only_tickers    = args.tickers,
            only_strategies = args.strategies,
            override        = args.override,
            strategies_nb   = strategies_nb,
        )
    else:
        # Esegui singolo scenario
        run_strategy_tests(
            dry_run            = args.dry_run,
            only_tickers       = args.tickers,
            only_strategies    = args.strategies,
            override           = args.override,
            scenario           = args.scenario,
            strategies_nb      = strategies_nb,
        )

    # Opzionale: schedula (solo in modalità singolo scenario)
    if args.schedule and not args.pipeline:
        log.info(f"Schedulazione giornaliera alle {DAILY_RUN_TIME}")
        schedule.every().day.at(DAILY_RUN_TIME).do(
            run_strategy_tests,
            dry_run         = False,
            only_tickers    = args.tickers,
            only_strategies = args.strategies,
        )
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == "__main__":
    main()
