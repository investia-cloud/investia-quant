"""
k_functions.py — K-Portfolio / K-Strategy Functions
Refactored from notebooks/libs/k_functions.ipynb
"""

# =================================
# New K Functions
# =================================

import itertools

import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import matplotlib.pyplot as plt
import os
from itertools import product
from typing import Union, List, Dict, Tuple, Any, Iterable, Optional
import seaborn as sns
import json
from k_strategies import *  # strategie richieste da _resolve_strategy via globals()

# Flag globale per attivare/disattivare l’uso della engine grid
USE_ENGINE_GRID: bool = False

# Engine grid globale: può essere sovrascritta prima di chiamare la WFO
ENGINE_PARAM_GRID: Dict[str, Iterable] = {
    "sl_pct"             : [0.10, 0.15],   # stop-loss 10% o 15%
    "tp_pct"             : [0.20, 0.30],   # take-profit 20% o 30%
    "time_exit"          : [5, 10],        # chiusura dopo 5 o 10 barre
    "risk_per_trade_pct" : [0.01, 0.02],   # 1% o 2% del capitale per trade
}

# try: tqdm, else no-op
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

from IPython.display import display
from datetime import datetime, timezone
import fnmatch
from sklearn.linear_model import LinearRegression
from typing import Optional, Iterable, Dict, Tuple, List, Union, Callable, Any


# --- Parallel helpers (optional) ---
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

try:
    from tqdm_joblib import tqdm_joblib
    _HAVE_TQDM_JOBLIB = True
except Exception:
    _HAVE_TQDM_JOBLIB = False

# --- CPU info helpers (optional) ---
try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False



# ==========================
# Utilità generiche framework
# ==========================


def infer_wfo_base_years(
    df: pd.DataFrame,
    wfo_start_year: Optional[int] = None,
    wfo_end_year: Optional[int] = None,
) -> Tuple[int, int]:
    """Ritorna gli **anni base** (start, end) per la WFO, senza applicare il train offset.

    Semantica dell'autore:
    - start/end sono i limiti logici passati dall'utente (o derivati dai dati).
    - Il primo anno di test sarà start + train_years.
    - Se end è oltre l’ultimo anno disponibile nei dati, lo manteniamo comunque:
      la WFO mostrerà quell’anno in tabella, anche se poi non ci saranno barre.
    """
    first_year = int(df.index.min().year)
    last_year = int(df.index.max().year)

    base_start = int(wfo_start_year) if wfo_start_year is not None else first_year
    base_end = int(wfo_end_year) if wfo_end_year is not None else last_year

    # clamp solo su start per non andare prima dei dati
    base_start = max(base_start, first_year)
    # nessun clamp su end

    return base_start, base_end


def param_product(param_ranges: Dict[str, Iterable]) -> Iterable[Tuple]:
    """Prodotto cartesiano deterministico dei range di parametri."""
    keys = list(param_ranges.keys())
    ranges = [list(param_ranges[k]) for k in keys]
    for combo in itertools.product(*ranges):
        yield combo


def total_param_combinations(param_ranges: Dict[str, Iterable]) -> int:
    """Numero totale di combinazioni nella griglia di parametri."""
    total = 1
    for rng in param_ranges.values():
        total *= len(list(rng))
    return total



def compute_exposure_percentage(data: pd.DataFrame, entries: pd.Series, exits: pd.Series) -> float:
    """Calcola l'esposizione come % di tempo in posizione, come nel codice originale."""
    in_position = False
    exposure_days = 0
    total_days = len(data)
    entry_day = 0

    for i in range(1, len(data)):
        if entries.iloc[i] and not in_position:
            in_position = True
            entry_day = i
        elif exits.iloc[i] and in_position:
            in_position = False
            exit_day = i
            exposure_days += exit_day - entry_day

    if in_position:
        exposure_days += total_days - entry_day

    return (exposure_days / total_days) * 100.0


def compute_portfolio_stats_with_adjusted(
    portfolio: vbt.Portfolio,
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    risk_free_rate: float = 0.02,
) -> pd.Series:

    
    """Metriche base + CAR aggiustate (Sharpe-style, Calmar-style) + esposizione."""
    car = portfolio.annualized_return()
    vol = portfolio.annualized_volatility()
    max_dd = abs(portfolio.max_drawdown())

    adj_sharpe = (car - risk_free_rate) / vol if vol != 0 else np.nan
    adj_calmar = car / max_dd if max_dd != 0 else np.nan

    exposure_pct = compute_exposure_percentage(data, entries, exits)

    stats = portfolio.stats()
    stats['CAR'] = f"{car:.2%}"
    stats['Adjusted CAR (Sharpe-style)'] = f"{adj_sharpe:.2%}"
    stats['Adjusted CAR (Calmar-style)'] = f"{adj_calmar:.2%}"
    stats['Market Time Exposure'] = f"{exposure_pct:.2f}%"
    return stats



def save_trading_wfo_summary(strategy, symbol, ratio, portfolio, grid, summary_df, wfo_results_dir, verbose=True,
                     bh_portfolio=None, bh_metrics=None):
    """
    Salva:
      - WFO results (DataFrame) in pickle
      - Portfolio (vectorbt) via .save()
      - (opz.) Buy&Hold portfolio e metriche associate

    Ritorna: (portfolio_result_file, wfo_result_file, bh_portfolio_file | None, bh_metrics_file | None)
    """
    info_string = f'{strategy}_{symbol}_{ratio}'
    base_dir = f"{wfo_results_dir}/{strategy}"
    portfolio_result_file = f"{base_dir}/portfolio_{info_string}_results.pkl"
    # wfo_result_file       = f"{base_dir}/portfolio_{info_string}_wfo_results.pkl"
    wfo_result_file       = f"{base_dir}/portfolio_{info_string}_wfo_results.csv"

    bh_portfolio_file     = f"{base_dir}/portfolio_{info_string}_bh_results.pkl"
    bh_metrics_file       = f"{base_dir}/portfolio_{info_string}_bh_metrics.pkl"

    if isinstance(portfolio, str) and portfolio == "":
        print(f"WFO for ticker {symbol} failed. Please, check this ticker or the strategy!")
        return None

    os.makedirs(base_dir, exist_ok=True)

    # Save WFO results
    # summary_df.to_pickle(wfo_result_file)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "strategy":strategy,
        "symbol": symbol,
        "ratio": ratio,
        "param_grid": grid,
    }

    # Luca     
    with open(wfo_result_file, "w", encoding="utf-8") as f:
        f.write("# === WFO METADATA ===\n")
        for k, v in meta.items():
            if isinstance(v, dict):
                # Serializzazione Python => preserva True/False (e None)
                f.write(f"# {k} = {repr(v)}\n")
            else:
                f.write(f"# {k} = {v}\n")
        f.write("# === WFO RESULTS ===\n")
        summary_df.to_csv(f)
   
    # summary_df.to_csv(wfo_result_file)

    # Save Portfolio results
    portfolio.save(portfolio_result_file)

    # Salvataggi B&H opzionali
    if bh_portfolio is not None:
        try:
            bh_portfolio.save(bh_portfolio_file)
        except Exception as e:
            print(f"[WARN] Impossibile salvare bh_portfolio: {e}")
            bh_portfolio_file = None
    if bh_metrics is not None:
        try:
            # salva in pickle in modo trasparente (dict/Series/DataFrame)
            import pandas as pd
            if hasattr(bh_metrics, "to_pickle"):
                bh_metrics.to_pickle(bh_metrics_file)
            else:
                pd.to_pickle(bh_metrics, bh_metrics_file)
        except Exception as e:
            print(f"[WARN] Impossibile salvare bh_metrics: {e}")
            bh_metrics_file = None

    if verbose:
        print(
                "File salvati con successo:\n"
                f"- WFO results : {wfo_result_file}\n"
                f"- Portfolio   : {portfolio_result_file}\n"
                f"- BH Portfolio: {bh_portfolio_file if bh_portfolio is not None else 'n/a'}\n"
                f"- BH Metrics  : {bh_metrics_file if bh_metrics is not None else 'n/a'}"
            )

    return portfolio_result_file, wfo_result_file, (bh_portfolio_file if bh_portfolio is not None else None), (bh_metrics_file if bh_metrics is not None else None)

# =========================================
# Massive RUNS
# =========================================

# =====================
# WFO Strategy Panel 
# =====================




def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _extract_metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    aliases = {
        "Total Return %": ["Total Return [%]", "Total Return %", "Total Return"],
        "Benchmark Return %": ["Benchmark Return [%]", "Benchmark Return %"],
        "CAR %": ["CAR", "CAGR", "CAR %", "CAGR %"],
        "Sharpe": ["Sharpe Ratio", "Sharpe"],
        "Calmar": ["Calmar Ratio", "Calmar"],
        "MaxDD %": ["Max Drawdown [%]", "Max Drawdown %", "MaxDD %"],
        "Trades": ["Total Trades", "Trades"],
        "Win Rate %": ["Win Rate [%]", "Win Rate %"],
        "PF": ["Profit Factor", "PF"],
        "Sortino": ["Sortino Ratio", "Sortino"],
        "Exposure %": ["Market Time Exposure", "Exposure %"],
    }
    out: Dict[str, float] = {}
    for k, names in aliases.items():
        for n in names:
            if n in df.index:
                v = df.loc[n]
                v = v.iloc[0] if hasattr(v, "iloc") else v
                out[k] = _safe_float(v)
                break
    return out


    
def wfo_strategy_panel(
    tickers: "Iterable[str]",
    strategies: "Optional[Iterable[str]]" = None,
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    ratio: str = "4:1",
    risk_free_rate: float = 0.02,
    show_progress: bool = True,
    init_cash: float = 100_000.0,
    fees: float = 0.001,
    slippage: float = 0.002,
    price_col: str = "Open",
    selection_metric: "Union[str, Callable]" = "total_return",
    warmup_years: int = 1,
    verbose: bool = False,
    save_results: bool = False,
    wfo_results_dir: "Optional[str]" = None,
    namespace: "Optional[Dict[str, Any]]" = None,
    override: bool = False,

    # ==========================================================
    # Catalog (opzionale) - screening "a priori" dei task
    # ==========================================================
    use_catalog: bool = False,
    catalog_top_k_per_ticker: int = 10,
    catalog_price_group: str = "Price",
    return_catalog: bool = False,
    return_joblist: bool = False,

    # ==========================================================
    # Scenario precheck — sovrascrive i parametri precheck_*
    # ==========================================================
    # "A" exploration  : Gate1 ON, Gate2 OFF, Gate3 OFF  (permissivo)
    # "B" quality_base : Gate1 ON, Gate2 ON  40%,  Gate3 OFF
    # "C" quality      : Gate1 ON, Gate2 ON  50%,  Gate3 OFF  (default consigliato)
    # "D" notebook     : Gate1 ON, Gate2 OFF, Gate3 ON
    # "E" production   : Gate1 ON, Gate2 ON  50%,  Gate3 ON   (selettivo)
    # None             : usa i valori passati ai singoli parametri precheck_*
    scenario: "Optional[str]" = None,

    # ==========================================================
    # Precheck robusto - screening evidence-based
    # ==========================================================
    precheck_mode: str = "stress",          # "none" | "stress" | "stress+core"
    precheck_stress_n_samples: int = 800,
    precheck_stress_method: str = "lhs",    # "random" | "lhs"
    precheck_seed: int | None = 42,

    # gate policy
    precheck_use_equity_gate: bool = True,
    precheck_min_beat_bh_pct: float = 0.55,

    # NUOVO gate hard: best overfit IS deve battere B&H
    precheck_require_best_is_beat_bh: bool = True,
    precheck_min_best_is_excess: float = 0.0,

    # filtro opzionale additivo
    precheck_require_recommend_wfo: bool = False,

    # robust thresholds (coerenti con overfitting_optimization)
    precheck_min_top_cluster_pct: float = 0.05,
    precheck_top_cluster_ratio: float = 0.80,
    precheck_min_plateau_neighbors: int = 8,
    precheck_min_plateau_above_pct: float = 0.20,
    precheck_max_outlier_robust_z: float = 6.0,

    # cache precheck su disco
    cache_precheck: bool = True,

    # dry-run
    dry_run: bool = False,
    dry_run_max_rows: int = 200,

    # debug
    catalog_debug: bool = False,

) -> "Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]":
    """
    Esegue un pannello massivo di Walk-Forward Optimization (WFO) su una lista di
    ticker e strategie con pipeline:

      1. Catalog opzionale
      2. Precheck
      3. WFO completa

    LOGICA PRECHECK
    ---------------
    - precheck_mode="none"
        Nessun precheck

    - precheck_mode="stress"
        Una sola call a overfitting_optimization_NEW(..., grid_mode="stress")
        e da quel report estrae:
            * best result vs B&H   -> gate hard
            * beat_bh_pct          -> gate robustezza
            * recommend_wfo        -> gate opzionale

    - precheck_mode="stress+core"
        Prima stress, poi core solo se stress passa.
        Quindi qui le call possono essere due, ma solo per scelta esplicita della modalità.

    SIGNIFICATO CORRETTO DI precheck_require_recommend_wfo
    ------------------------------------------------------
    Non è più alternativo al gate vs B&H.
    È solo un filtro additivo:
        - False: ignora recommend_wfo
        - True : richiede anche recommend_wfo=True

    SCENARI PRECHECK (parametro scenario)
    --------------------------------------
    Il parametro scenario sovrascrive automaticamente i parametri precheck_*.
    Se scenario=None (default) valgono i valori passati ai singoli parametri.

    scenario="A"  Esplorazione
        Gate1 ON  (best param batte B&H)
        Gate2 OFF
        Gate3 OFF
        → Permissivo. Passa quasi tutto. Utile per primo screening.

    scenario="B"  Qualità base  ← consigliato per strategie generate dall'agente
        Gate1 ON  (best param batte B&H)
        Gate2 ON  (beat_bh_pct >= 40%)
        Gate3 OFF
        → Richiede un minimo di robustezza parametrica.

    scenario="C"  Qualità media
        Gate1 ON  (best param batte B&H)
        Gate2 ON  (beat_bh_pct >= 50%)
        Gate3 OFF
        → Bilanciato. Default consigliato per test sistematici.

    scenario="D"  Configurazione notebook
        Gate1 ON  (best param batte B&H)
        Gate2 OFF
        Gate3 ON  (recommend_wfo=True)
        → Selettivo sul giudizio composito, non sulla robustezza diretta.

    scenario="E"  Produzione
        Gate1 ON  (best param batte B&H)
        Gate2 ON  (beat_bh_pct >= 50%)
        Gate3 ON  (recommend_wfo=True)
        → Tutti i gate. Solo per strategie candidate al deploy.
    """

    try:
        pd.set_option("future.no_silent_downcasting", True)
    except Exception:
        pass

    if namespace is None:
        namespace = globals()

    tickers = list(tickers)

    # ------------------------------------------------------------------
    # Helpers filesystem / caching
    # ------------------------------------------------------------------
    def _cdebug(*args, **kwargs):
        if catalog_debug:
            print(*args, **kwargs)

    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def _find_existing_result_paths(symbol: str, strat: str, ratio_: str, results_dir: str | None):
        if not results_dir:
            return []
        base = os.path.join(results_dir, strat)
        info = f"{strat}_{symbol}_{ratio_}"
        files = [
            f"portfolio_{info}_wfo_results.csv",
            f"portfolio_{info}_results.pkl",
            f"portfolio_{info}_bh_results.pkl",
            f"portfolio_{info}_bh_metrics.pkl",
        ]
        return [os.path.join(base, f) for f in files if os.path.isfile(os.path.join(base, f))]

    def _precheck_path(symbol: str, strat: str, ratio_: str, results_dir: str | None):
        if not results_dir:
            return None
        base = os.path.join(results_dir, strat)
        return os.path.join(base, f"portfolio_{strat}_{symbol}_{ratio_}_precheck.pkl")

    def _has_precheck(symbol: str, strat: str, ratio_: str, results_dir: str | None) -> bool:
        p = _precheck_path(symbol, strat, ratio_, results_dir)
        return bool(p) and os.path.isfile(p)

    def _read_precheck(symbol: str, strat: str, ratio_: str, results_dir: str | None) -> dict | None:
        import pickle
        p = _precheck_path(symbol, strat, ratio_, results_dir)
        if not p or not os.path.isfile(p):
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _write_precheck(symbol: str, strat: str, ratio_: str, results_dir: str, payload: dict):
        import pickle
        base = os.path.join(results_dir, strat)
        _ensure_dir(base)
        p = _precheck_path(symbol, strat, ratio_, results_dir)
        with open(p, "wb") as f:
            pickle.dump(payload, f)

    # ------------------------------------------------------------------
    # Strategy / Optimization resolution
    # ------------------------------------------------------------------
    def _resolve_strategy_fn(name: str):
        fn = None
        if isinstance(namespace, dict):
            fn = namespace.get(f"strategy_{name}")
        if fn is None:
            fn = globals().get(f"strategy_{name}")
        if fn is None:
            raise Exception(f"Funzione strategy_{name} non trovata")
        return fn

    def _resolve_param_ranges(name: str) -> dict:
        pr = None
        if isinstance(namespace, dict):
            pr = namespace.get(f"strategy_{name}_param_ranges")
        if pr is None:
            pr = globals().get(f"strategy_{name}_param_ranges")
        return pr or {}

    def _resolve_overfitting_optimization():
        fn = None
        if isinstance(namespace, dict):
            fn = namespace.get("overfitting_optimization")
        if fn is None:
            fn = globals().get("overfitting_optimization")
        if fn is None:
            raise NameError("overfitting_optimization() non trovata (necessaria per precheck).")
        return fn

    def _resolve_run_strategy():
        fn = None
        if isinstance(namespace, dict):
            fn = namespace.get("run_strategy")
        if fn is None:
            fn = globals().get("run_strategy")
        if fn is None:
            raise NameError("run_strategy() non trovata.")
        return fn

    # ------------------------------------------------------------------
    # Buy & Hold benchmark
    # ------------------------------------------------------------------
    def _bh_total_return(close: pd.Series) -> float:
        pf = vbt.Portfolio.from_holding(
            close=close,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq="1D",
        )
        return float(pf.total_return())

    # ------------------------------------------------------------------
    # Helper parsing report overfitting
    # ------------------------------------------------------------------
    def _safe_float(x):
        try:
            x = float(x)
            return x if np.isfinite(x) else None
        except Exception:
            return None

    def _deep_get(d, path, default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _normalize_fraction(v):
        fv = _safe_float(v)
        if fv is None:
            return None
        # Se qualcuno salva 42 invece di 0.42, normalizza
        if abs(fv) > 1.0 and abs(fv) <= 1000:
            # per i total return è ambiguo; qui NON normalizzo automaticamente
            # salvo il valore così com'è e lascio la logica esplicita sopra
            return fv
        return fv

    def _extract_recommend_wfo(report: dict | None) -> bool:
        if not isinstance(report, dict):
            return False
        dec = report.get("decision") or {}
        return bool(dec.get("recommend_wfo", False))

    def _extract_beat_bh_pct(report: dict | None) -> float | None:
        if not isinstance(report, dict):
            return None

        candidates = [
            _deep_get(report, ["equity_sweep", "beat_bh_pct_by_total_return"]),
            _deep_get(report, ["equity_sweep", "beat_bh_pct"]),
            report.get("beat_bh_pct_by_total_return"),
            report.get("beat_bh_pct"),
        ]

        for v in candidates:
            fv = _safe_float(v)
            if fv is not None:
                if fv > 1.0 and fv <= 100.0:
                    fv = fv / 100.0
                return fv
        return None

    def _extract_best_total_return(report: dict | None) -> float | None:
        """
        Estrae il best total return dal report di overfitting.
        Cerca più chiavi compatibili.
        """
        if not isinstance(report, dict):
            return None

        candidate_paths = [
            ["best_result", "total_return"],
            ["best_result", "best_total_return"],
            ["best_candidate", "total_return"],
            ["best_candidate", "best_total_return"],
            ["summary", "best_total_return"],
            ["optimization_summary", "best_total_return"],
            ["equity_sweep", "best_total_return"],
            ["equity_sweep", "max_total_return"],
            ["decision", "best_total_return"],
            ["decision", "max_total_return"],
        ]

        for path in candidate_paths:
            v = _deep_get(report, path)
            fv = _safe_float(v)
            if fv is not None:
                return fv

        candidate_keys = [
            "best_total_return",
            "max_total_return",
            "total_return",
        ]
        for k in candidate_keys:
            fv = _safe_float(report.get(k))
            if fv is not None:
                return fv

        return None

    # ------------------------------------------------------------------
    # Resolve strategies list
    # ------------------------------------------------------------------
    if strategies is None:
        list_strats = None
        if isinstance(namespace, dict):
            list_strats = namespace.get("list_strategies")
        if list_strats is None:
            list_strats = globals().get("list_strategies")
        if list_strats is None:
            raise ValueError("list_strategies() non trovata")
        strategies = list(list_strats())
        if not strategies:
            raise ValueError("list_strategies() non ha restituito alcuna strategia.")
        print(f"[AUTO] strategies=None -> uso {len(strategies)} strategie da list_strategies().")
    else:
        strategies = list(strategies)

    # ------------------------------------------------------------------
    # Catalog (opzionale) -> build tasks
    # ------------------------------------------------------------------
    catalog_out = None
    joblist_out = None

    def _normalize_joblist(joblist):
        import numpy as np

        if joblist is None:
            return pd.DataFrame(columns=["Ticker", "Strategy"])

        if isinstance(joblist, list):
            if len(joblist) == 0:
                return pd.DataFrame(columns=["Ticker", "Strategy"])
            if isinstance(joblist[0], tuple) and len(joblist[0]) >= 2:
                return pd.DataFrame(joblist, columns=["Ticker", "Strategy"])
            if isinstance(joblist[0], dict):
                joblist = pd.DataFrame(joblist)
            else:
                raise TypeError("joblist_out è una lista con formato non supportato.")

        if isinstance(joblist, dict):
            joblist = pd.DataFrame(joblist)

        if not isinstance(joblist, pd.DataFrame):
            raise TypeError(f"joblist_out inatteso: {type(joblist)}")

        dfj = joblist.copy()

        if dfj.empty and len(dfj.columns) == 0:
            return pd.DataFrame(columns=["Ticker", "Strategy"])

        if getattr(dfj.index, "nlevels", 1) >= 1:
            idx_names = [n for n in dfj.index.names if n is not None]
            if idx_names:
                dfj = dfj.reset_index()

        cols_lower = {str(c).lower(): c for c in dfj.columns}
        rename = {}

        for alias in ["ticker", "symbol", "asset", "security"]:
            if alias in cols_lower:
                rename[cols_lower[alias]] = "Ticker"
                break

        for alias in ["strategy", "strategyname", "strategy_name", "name", "model", "rule"]:
            if alias in cols_lower:
                rename[cols_lower[alias]] = "Strategy"
                break

        if rename:
            dfj = dfj.rename(columns=rename)

        if "Ticker" in dfj.columns and "Strategy" not in dfj.columns:
            other_cols = [c for c in dfj.columns if c != "Ticker"]
            if len(other_cols) > 0:
                wide = dfj.set_index("Ticker")[other_cols]
                stacked = wide.stack(dropna=False).reset_index()
                stacked.columns = ["Ticker", "Strategy", "Value"]

                val_num = pd.to_numeric(stacked["Value"], errors="coerce")
                active = (
                    stacked["Value"].astype(object).isin([True, 1, 1.0])
                    | (val_num.fillna(0) != 0)
                )
                dfj = stacked.loc[active, ["Ticker", "Strategy"]].drop_duplicates()

        if "Ticker" in dfj.columns and "Strategy" in dfj.columns:
            dfj = dfj[["Ticker", "Strategy"]].dropna()
            dfj["Ticker"] = dfj["Ticker"].astype(str)
            dfj["Strategy"] = dfj["Strategy"].astype(str)
            return dfj.reset_index(drop=True)

        if dfj.empty:
            return pd.DataFrame(columns=["Ticker", "Strategy"])

        raise ValueError(
            "joblist_out non è normalizzabile a colonne ['Ticker','Strategy'].\n"
            f"Colonne disponibili: {list(dfj.columns)}"
        )

    def _safe_preview(obj, max_chars: int = 800) -> str:
        try:
            txt = repr(obj)
        except Exception as e:
            txt = f"<repr failed: {e}>"
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " ... [truncated]"
        return txt

    def _print_df_info(name: str, df: pd.DataFrame | None, max_rows: int = 5):
        if not catalog_debug:
            return

        print(f"[CATALOG DEBUG] {name} type: {type(df)}")

        if df is None:
            print(f"[CATALOG DEBUG] {name} is None")
            return

        if not isinstance(df, pd.DataFrame):
            print(f"[CATALOG DEBUG] {name} repr: {repr(df)[:500]}")
            return

        print(f"[CATALOG DEBUG] {name} shape: {df.shape}")
        print(f"[CATALOG DEBUG] {name} columns: {list(df.columns)}")

        if not df.empty:
            try:
                print(df.head(max_rows).to_string(index=False))
            except Exception:
                pass

    if use_catalog:
        build_catalog = None
        build_joblist = None
        get_clean_data = None

        if isinstance(namespace, dict):
            build_catalog = namespace.get("build_strategy_catalog_from_globals")
            build_joblist = namespace.get("build_wfo_joblist_from_clean_data")
            get_clean_data = namespace.get("get_clean_financial_data")

        build_catalog = build_catalog or globals().get("build_strategy_catalog_from_globals")
        build_joblist = build_joblist or globals().get("build_wfo_joblist_from_clean_data")
        get_clean_data = get_clean_data or globals().get("get_clean_financial_data")

        if build_catalog is None or build_joblist is None or get_clean_data is None:
            raise NameError(
                "use_catalog=True richiede:\n"
                "- build_strategy_catalog_from_globals\n"
                "- build_wfo_joblist_from_clean_data\n"
                "- get_clean_financial_data\n"
                "oppure passale nel namespace."
            )

        catalog_out = build_catalog(namespace, use_cache=True, require_param_grid=False)
        _cdebug("[CATALOG DEBUG] --------------------------------------------------")
        _cdebug("[CATALOG DEBUG] STEP 1/4 - build_catalog")
        _cdebug(f"[CATALOG DEBUG] requested tickers n={len(tickers)}")
        _cdebug(f"[CATALOG DEBUG] requested strategies n={len(strategies) if strategies is not None else 'None'}")
        _cdebug(f"[CATALOG DEBUG] catalog_top_k_per_ticker={catalog_top_k_per_ticker}")
        _cdebug(f"[CATALOG DEBUG] catalog_price_group={catalog_price_group!r}")
        _print_df_info("catalog_out", catalog_out, max_rows=5)

        tickers_data = get_clean_data(tickers, start_date, end_date)

        _cdebug("[CATALOG DEBUG] --------------------------------------------------")
        _cdebug("[CATALOG DEBUG] STEP 2/4 - get_clean_financial_data")
        _cdebug(f"[CATALOG DEBUG] tickers_data type: {type(tickers_data)}")

        if tickers_data is None:
            _cdebug("[CATALOG DEBUG] tickers_data is None")
        else:
            try:
                _cdebug(f"[CATALOG DEBUG] tickers_data shape: {tickers_data.shape}")
            except Exception:
                _cdebug("[CATALOG DEBUG] tickers_data shape unavailable")

            if hasattr(tickers_data, "columns"):
                _cdebug(f"[CATALOG DEBUG] tickers_data columns type: {type(tickers_data.columns)}")
                if isinstance(tickers_data.columns, pd.MultiIndex):
                    _cdebug(f"[CATALOG DEBUG] tickers_data MultiIndex names: {tickers_data.columns.names}")
                    _cdebug(f"[CATALOG DEBUG] tickers_data first columns: {list(tickers_data.columns[:10])}")
                else:
                    _cdebug(f"[CATALOG DEBUG] tickers_data columns: {list(tickers_data.columns[:10])}")

            if hasattr(tickers_data, "head"):
                try:
                    _cdebug(tickers_data.head(3).to_string())
                except Exception:
                    _cdebug(f"[CATALOG DEBUG] tickers_data head repr: {_safe_preview(tickers_data.head(3))}")

        if tickers_data is not None and not isinstance(tickers_data.columns, pd.MultiIndex):
            if len(tickers) == 1:
                _t = str(tickers[0])
                tickers_data = tickers_data.copy()
                tickers_data.columns = pd.MultiIndex.from_product(
                    [[catalog_price_group], [_t], list(tickers_data.columns)],
                    names=["Group", "Ticker", "Field"],
                )
                _cdebug("[CATALOG DEBUG] tickers_data converted to MultiIndex (single ticker case)")
                _cdebug(f"[CATALOG DEBUG] tickers_data MultiIndex names: {tickers_data.columns.names}")
                _cdebug(f"[CATALOG DEBUG] tickers_data first columns: {list(tickers_data.columns[:10])}")
            else:
                raise ValueError(
                    "tickers_data non ha colonne MultiIndex ma tickers contiene più di un simbolo."
                )

        _cdebug("[CATALOG DEBUG] --------------------------------------------------")
        _cdebug("[CATALOG DEBUG] STEP 3/4 - build_wfo_joblist_from_clean_data")

        joblist_raw = build_joblist(
            tickers=tickers,
            tickers_data=tickers_data,
            catalog=catalog_out,
            top_k_per_ticker=catalog_top_k_per_ticker,
            price_group=catalog_price_group,
            use_cache=True,
            show_progress=show_progress,
            allow_unknown_profile=True,
        )

        _cdebug(f"[CATALOG DEBUG] raw joblist type: {type(joblist_raw)}")

        if isinstance(joblist_raw, pd.DataFrame):
            _cdebug(f"[CATALOG DEBUG] raw joblist shape: {joblist_raw.shape}")
            _cdebug(f"[CATALOG DEBUG] raw joblist columns: {list(joblist_raw.columns)}")
            if joblist_raw.empty:
                _cdebug("[CATALOG DEBUG] raw joblist is empty")
            else:
                try:
                    _cdebug(joblist_raw.head(10).to_string(index=False))
                except Exception:
                    _cdebug(f"[CATALOG DEBUG] raw joblist head repr: {_safe_preview(joblist_raw.head(10))}")
        elif isinstance(joblist_raw, list):
            _cdebug(f"[CATALOG DEBUG] raw joblist len: {len(joblist_raw)}")
            _cdebug(f"[CATALOG DEBUG] raw joblist first items: {_safe_preview(joblist_raw[:10])}")
        elif isinstance(joblist_raw, dict):
            _cdebug(f"[CATALOG DEBUG] raw joblist keys: {list(joblist_raw.keys())[:20]}")
            _cdebug(f"[CATALOG DEBUG] raw joblist repr: {_safe_preview(joblist_raw)}")
        else:
            _cdebug(f"[CATALOG DEBUG] raw joblist repr: {_safe_preview(joblist_raw)}")

        _cdebug("[CATALOG DEBUG] --------------------------------------------------")
        _cdebug("[CATALOG DEBUG] STEP 4/4 - normalize joblist")

        try:
            joblist_out = _normalize_joblist(joblist_raw)
        except Exception as e:
            _cdebug(f"[CATALOG DEBUG] _normalize_joblist FAILED: {e}")
            raise

        _print_df_info("joblist_out (normalized)", joblist_out, max_rows=10)

        if not joblist_out.empty:
            try:
                uniq_tickers = sorted(joblist_out["Ticker"].astype(str).unique().tolist())
                uniq_strats = sorted(joblist_out["Strategy"].astype(str).unique().tolist())
                _cdebug(f"[CATALOG DEBUG] normalized unique tickers ({len(uniq_tickers)}): {uniq_tickers[:30]}")
                _cdebug(f"[CATALOG DEBUG] normalized unique strategies ({len(uniq_strats)}): {uniq_strats[:50]}")
            except Exception as e:
                _cdebug(f"[CATALOG DEBUG] failed unique extraction on normalized joblist: {e}")

        if strategies:
            _cdebug("[CATALOG DEBUG] --------------------------------------------------")
            _cdebug("[CATALOG DEBUG] STEP 5/5 - explicit strategies filter")
            _cdebug(f"[CATALOG DEBUG] requested strategies ({len(strategies)}): {list(strategies)[:50]}")

            if not joblist_out.empty:
                available_strats = sorted(joblist_out["Strategy"].astype(str).unique().tolist())
                _cdebug(f"[CATALOG DEBUG] available joblist strategies before filter ({len(available_strats)}): {available_strats[:50]}")

            before_n = len(joblist_out)
            joblist_out = joblist_out[joblist_out["Strategy"].isin(set(strategies))].copy()
            after_n = len(joblist_out)

            _cdebug(f"[CATALOG DEBUG] strategy filter rows: before={before_n}, after={after_n}")

            if before_n > 0 and after_n == 0:
                _cdebug("[CATALOG DEBUG] strategy filter removed everything")
            elif after_n > 0:
                try:
                    _cdebug(joblist_out.head(10).to_string(index=False))
                except Exception:
                    _cdebug(f"[CATALOG DEBUG] filtered joblist head repr: {_safe_preview(joblist_out.head(10))}")

        tasks = list(zip(joblist_out["Ticker"].tolist(), joblist_out["Strategy"].tolist()))

        naive_total = len(tickers) * len(strategies) if strategies is not None else None

        _cdebug("[CATALOG DEBUG] --------------------------------------------------")
        if naive_total is not None:
            print(f"[CATALOG] tasks naive: {naive_total:,} | tasks selezionati: {len(tasks):,}")
        else:
            print(f"[CATALOG] tasks naive: n/a (strategies=None) | tasks selezionati: {len(tasks):,}")
    else:
        tasks = [(t, s) for t in tickers for s in strategies]

    if not tasks:
        print("[INFO] Nessun task da eseguire (tasks vuoto).")
        df_empty = pd.DataFrame()
        results_empty: Dict[Tuple[str, str], Dict[str, Any]] = {}
        extra: Dict[str, Any] = {}
        if return_catalog:
            extra["catalog"] = catalog_out
        if return_joblist:
            extra["joblist"] = joblist_out
        extra["tasks_preview"] = pd.DataFrame(columns=["Ticker", "Strategy"])
        return df_empty, results_empty, extra

    # ------------------------------------------------------------------
    # DRY RUN
    # ------------------------------------------------------------------
    if dry_run:
        print("\n[DRY-RUN] Nessuna WFO verrà eseguita. Elenco attività previste:")
        print(f" - Tickers: {len(tickers)}")
        print(f" - Strategie candidate: {len(set([s for _, s in tasks]))}")
        print(f" - Tasks totali: {len(tasks)}")
        preview = pd.DataFrame(tasks, columns=["Ticker", "Strategy"])
        if dry_run_max_rows is not None and preview.shape[0] > int(dry_run_max_rows):
            display(preview.head(int(dry_run_max_rows)))
            print(f"[DRY-RUN] Mostrate le prime {int(dry_run_max_rows)} righe su {preview.shape[0]}.")
        else:
            display(preview)

        extra: Dict[str, Any] = {"tasks_preview": preview}
        if return_catalog:
            extra["catalog"] = catalog_out
        if return_joblist:
            extra["joblist"] = joblist_out

        return pd.DataFrame(), {}, extra

    # ------------------------------------------------------------------
    # PRECHECK GATE
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Scenario: sovrascrive i parametri precheck_* se specificato
    # ------------------------------------------------------------------
    _SCENARIOS = {
        "A": dict(
            precheck_mode                  = "stress",
            precheck_require_best_is_beat_bh = True,
            precheck_min_best_is_excess    = 0.0,
            precheck_use_equity_gate       = False,
            precheck_min_beat_bh_pct       = 0.40,
            precheck_require_recommend_wfo = False,
        ),
        "B": dict(
            precheck_mode                  = "stress",
            precheck_require_best_is_beat_bh = True,
            precheck_min_best_is_excess    = 0.0,
            precheck_use_equity_gate       = True,
            precheck_min_beat_bh_pct       = 0.40,
            precheck_require_recommend_wfo = False,
        ),
        "C": dict(
            precheck_mode                  = "stress",
            precheck_require_best_is_beat_bh = True,
            precheck_min_best_is_excess    = 0.0,
            precheck_use_equity_gate       = True,
            precheck_min_beat_bh_pct       = 0.50,
            precheck_require_recommend_wfo = False,
        ),
        "D": dict(
            precheck_mode                  = "stress",
            precheck_require_best_is_beat_bh = True,
            precheck_min_best_is_excess    = 0.0,
            precheck_use_equity_gate       = False,
            precheck_min_beat_bh_pct       = 0.50,
            precheck_require_recommend_wfo = True,
        ),
        "E": dict(
            precheck_mode                  = "stress",
            precheck_require_best_is_beat_bh = True,
            precheck_min_best_is_excess    = 0.0,
            precheck_use_equity_gate       = True,
            precheck_min_beat_bh_pct       = 0.50,
            precheck_require_recommend_wfo = True,
        ),
    }
    if scenario is not None:
        _sc = str(scenario).upper()
        if _sc not in _SCENARIOS:
            raise ValueError(
                f"scenario={scenario!r} non valido. "
                f"Valori ammessi: {list(_SCENARIOS)} oppure None."
            )
        _sc_params = _SCENARIOS[_sc]
        precheck_mode                   = _sc_params["precheck_mode"]
        precheck_require_best_is_beat_bh = _sc_params["precheck_require_best_is_beat_bh"]
        precheck_min_best_is_excess     = _sc_params["precheck_min_best_is_excess"]
        precheck_use_equity_gate        = _sc_params["precheck_use_equity_gate"]
        precheck_min_beat_bh_pct        = _sc_params["precheck_min_beat_bh_pct"]
        precheck_require_recommend_wfo  = _sc_params["precheck_require_recommend_wfo"]

    precheck_mode = (precheck_mode or "none").lower()
    if precheck_mode not in ("none", "stress", "stress+core"):
        precheck_mode = "none"

    of_opt = _resolve_overfitting_optimization()
    run_strategy_fn = _resolve_run_strategy()

    price_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    def _run_overfit(symbol: str, strat: str, pr: dict, grid_mode: str):
        kwargs = dict(
            symbol=symbol,
            strategy=strat,
            start_date=start_date,
            end_date=end_date,
            param_ranges=pr,
            price_col=price_col,
            freq="1D",
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            plot_results=False,
            smart_plot=False,
            analyze_results=True,
            return_analysis=True,
            equity_sweep=True,
            equity_plot_benchmark=True,
            equity_plot_all=False,
            wfo_use_equity_sweep_gate=bool(precheck_use_equity_gate),
            wfo_min_beat_bh_pct=float(precheck_min_beat_bh_pct),
            wfo_min_top_cluster_pct=float(precheck_min_top_cluster_pct),
            wfo_top_cluster_ratio=float(precheck_top_cluster_ratio),
            wfo_min_plateau_neighbors=int(precheck_min_plateau_neighbors),
            wfo_min_plateau_above_pct=float(precheck_min_plateau_above_pct),
            wfo_max_outlier_robust_z=float(precheck_max_outlier_robust_z),
            grid_mode=grid_mode,
        )

        if grid_mode == "stress":
            kwargs["stress_n_samples"] = int(precheck_stress_n_samples)
            kwargs["stress_method"] = str(precheck_stress_method)
            kwargs["stress_seed"] = precheck_seed

        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            return of_opt(**kwargs)

    def _evaluate_report(report: dict, bh_tr: float) -> dict:
        best_tr = _extract_best_total_return(report)
        beat_bh_pct = _extract_beat_bh_pct(report)
        recommend_wfo = _extract_recommend_wfo(report)

        # Gate hard: best IS deve battere B&H
        if precheck_require_best_is_beat_bh:
            min_required = float(bh_tr) + float(precheck_min_best_is_excess)
            pass_best_is = (
                best_tr is not None and
                np.isfinite(best_tr) and
                best_tr > min_required
            )
        else:
            min_required = None
            pass_best_is = True

        # Gate robustezza sullo stress/core report
        if precheck_use_equity_gate:
            pass_equity_gate = (
                beat_bh_pct is not None and
                np.isfinite(beat_bh_pct) and
                beat_bh_pct >= float(precheck_min_beat_bh_pct)
            )
        else:
            pass_equity_gate = True

        # Gate opzionale additivo
        pass_recommend = (not precheck_require_recommend_wfo) or bool(recommend_wfo)

        return {
            "best_total_return": best_tr,
            "beat_bh_pct": beat_bh_pct,
            "recommend_wfo": bool(recommend_wfo),
            "pass_best_is": bool(pass_best_is),
            "pass_equity_gate": bool(pass_equity_gate),
            "pass_recommend": bool(pass_recommend),
            "min_required_best_is_total_return": min_required,
        }

    def _precheck_gate(symbol: str, strat: str) -> Tuple[bool, dict]:
        """
        Ritorna: (pass_gate, payload)

        stress:
            una sola call overfitting (grid_mode='stress')

        stress+core:
            stress -> se passa -> core
        """
        can_cache = bool(cache_precheck and save_results and wfo_results_dir)

        if can_cache and (not override) and _has_precheck(symbol, strat, ratio, wfo_results_dir):
            cached = _read_precheck(symbol, strat, ratio, wfo_results_dir)
            if isinstance(cached, dict) and "pass_gate" in cached:
                return bool(cached["pass_gate"]), cached

        key = (symbol, start_date, end_date)
        if key not in price_cache:
            price_cache[key] = load_ohlcv(symbol, start_date, end_date)
        df_daily = price_cache[key]

        if df_daily is None or df_daily.empty:
            payload_err: Dict[str, Any] = {
                "symbol": symbol, "strategy": strat, "ratio": ratio,
                "start_date": start_date, "end_date": end_date,
                "mode": precheck_mode,
                "pass_gate": False, "reason": "download_failed",
            }
            return False, payload_err

        bh_tr = _bh_total_return(df_daily[price_col])
        pr = _resolve_param_ranges(strat)

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "strategy": strat,
            "ratio": ratio,
            "start_date": start_date,
            "end_date": end_date,
            "price_col": price_col,
            "bh_total_return": float(bh_tr),
            "mode": precheck_mode,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "precheck_require_best_is_beat_bh": bool(precheck_require_best_is_beat_bh),
            "precheck_min_best_is_excess": float(precheck_min_best_is_excess),
            "precheck_use_equity_gate": bool(precheck_use_equity_gate),
            "precheck_min_beat_bh_pct": float(precheck_min_beat_bh_pct),
            "precheck_require_recommend_wfo": bool(precheck_require_recommend_wfo),
        }

        if precheck_mode == "none":
            payload["pass_gate"] = True
            payload["reason"] = "precheck_mode=none"
            if can_cache:
                _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
            return True, payload

        # --------------------------------------------------------------
        # MODE = STRESS
        # --------------------------------------------------------------
        if precheck_mode == "stress":
            try:
                _, _, _, _, analysis_s = _run_overfit(symbol, strat, pr, grid_mode="stress")
                stress_report = analysis_s or {}
                eval_s = _evaluate_report(stress_report, bh_tr)

                payload["stress_analysis"] = stress_report
                payload["stress_best_total_return"] = eval_s["best_total_return"]
                payload["beat_bh_pct"] = eval_s["beat_bh_pct"]
                payload["stress_recommend_wfo"] = eval_s["recommend_wfo"]
                payload["stress_min_required_best_is_total_return"] = eval_s["min_required_best_is_total_return"]

                if not eval_s["pass_best_is"]:
                    payload["pass_gate"] = False
                    payload["reason"] = "best_is_not_beating_bh"
                elif not eval_s["pass_equity_gate"]:
                    payload["pass_gate"] = False
                    payload["reason"] = "stress_beat_bh_pct_fail"
                elif not eval_s["pass_recommend"]:
                    payload["pass_gate"] = False
                    payload["reason"] = "stress_recommend_wfo_false"
                else:
                    payload["pass_gate"] = True
                    payload["reason"] = "stress_pass"

                if can_cache:
                    _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
                return bool(payload["pass_gate"]), payload

            except Exception as e:
                payload["pass_gate"] = False
                payload["reason"] = f"stress_error: {e}"
                if can_cache:
                    _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
                return False, payload

        # --------------------------------------------------------------
        # MODE = STRESS + CORE
        # --------------------------------------------------------------
        try:
            _, _, _, _, analysis_s = _run_overfit(symbol, strat, pr, grid_mode="stress")
            stress_report = analysis_s or {}
            eval_s = _evaluate_report(stress_report, bh_tr)

            payload["stress_analysis"] = stress_report
            payload["stress_best_total_return"] = eval_s["best_total_return"]
            payload["beat_bh_pct"] = eval_s["beat_bh_pct"]
            payload["stress_recommend_wfo"] = eval_s["recommend_wfo"]
            payload["stress_min_required_best_is_total_return"] = eval_s["min_required_best_is_total_return"]

            if not eval_s["pass_best_is"]:
                payload["pass_gate"] = False
                payload["reason"] = "best_is_not_beating_bh"
                if can_cache:
                    _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
                return False, payload

            if not eval_s["pass_equity_gate"]:
                payload["pass_gate"] = False
                payload["reason"] = "stress_beat_bh_pct_fail"
                if can_cache:
                    _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
                return False, payload

            if not eval_s["pass_recommend"]:
                payload["pass_gate"] = False
                payload["reason"] = "stress_recommend_wfo_false"
                if can_cache:
                    _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
                return False, payload

        except Exception as e:
            payload["pass_gate"] = False
            payload["reason"] = f"stress_error: {e}"
            if can_cache:
                _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
            return False, payload

        # Se arrivo qui, stress è passato; ora faccio core solo perché l'utente ha scelto stress+core
        try:
            _, _, _, _, analysis_c = _run_overfit(symbol, strat, pr, grid_mode="core")
            core_report = analysis_c or {}
            eval_c = _evaluate_report(core_report, bh_tr)

            payload["core_analysis"] = core_report
            payload["core_best_total_return"] = eval_c["best_total_return"]
            payload["core_recommend_wfo"] = eval_c["recommend_wfo"]
            payload["core_min_required_best_is_total_return"] = eval_c["min_required_best_is_total_return"]

            # Nel core NON riapplico il gate beat_bh_pct come criterio principale,
            # perché la robustezza l'ho già richiesta nello stress.
            # Riapplico però:
            #   - best IS > B&H
            #   - recommend_wfo se richiesto
            if not eval_c["pass_best_is"]:
                payload["pass_gate"] = False
                payload["reason"] = "core_best_is_not_beating_bh"
            elif precheck_require_recommend_wfo and not eval_c["pass_recommend"]:
                payload["pass_gate"] = False
                payload["reason"] = "core_recommend_wfo_false"
            else:
                payload["pass_gate"] = True
                payload["reason"] = "stress_core_pass"

            if can_cache:
                _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
            return bool(payload["pass_gate"]), payload

        except Exception as e:
            payload["pass_gate"] = False
            payload["reason"] = f"core_error: {e}"
            if can_cache:
                _write_precheck(symbol, strat, ratio, wfo_results_dir, payload)
            return False, payload

    # ------------------------------------------------------------------
    # RUN PANEL (precheck -> WFO)
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    extra: Dict[str, Any] = {}

    pbar = tqdm(
        total=len(tasks),
        disable=not show_progress,
        desc="WFO panel",
        leave=True,
        position=0,
    )

    try:
        for symbol, strat in tasks:
            if show_progress:
                pbar.set_postfix_str(f"{symbol} · {strat}", refresh=True)

            try:
                # (1) SKIP se risultati WFO esistono e override=False
                if (not override) and save_results and wfo_results_dir:
                    if _find_existing_result_paths(symbol, strat, ratio, wfo_results_dir):
                        if verbose:
                            print(f"[SKIP] {strat}@{symbol}: risultati già presenti")
                        continue

                # (2) PRECHECK gate
                pass_gate, pre_payload = _precheck_gate(symbol, strat)

                if verbose:
                    _rsn  = pre_payload.get("reason")
                    if _rsn == "download_failed":
                        print(f"[PRECHECK] {strat}@{symbol}: download dati fallito — skip")
                    else:
                        _bh   = pre_payload.get("bh_total_return")
                        _best = pre_payload.get("stress_best_total_return")
                        _beat = pre_payload.get("beat_bh_pct")
                        _rec  = pre_payload.get("stress_recommend_wfo")
                        _mode = pre_payload.get("mode")

                        _bh_str   = f"{_bh*100:.1f}%"   if _bh   is not None else "N/A"
                        _best_str = f"{_best*100:.1f}%"  if _best is not None else "N/A"
                        _beat_str = f"{_beat*100:.1f}%"  if _beat is not None else "N/A"

                        print(f"[PRECHECK] {strat}@{symbol}  (mode={_mode})")
                        print(f"  Struttura (overfitting_opt): recommend={_rec}")
                        if precheck_require_best_is_beat_bh:
                            _g1label = "OK" if (_best is not None and _best > (_bh or 0)) else "FALLITO"
                            print(f"  Gate 1 – best IS batte B&H : {_g1label}  (best={_best_str}, B&H={_bh_str})")
                        if precheck_use_equity_gate:
                            _g2label = "OK" if (_beat is not None and _beat >= precheck_min_beat_bh_pct) else "FALLITO"
                            print(f"  Gate 2 – % param batte B&H : {_g2label}  ({_beat_str} >= {precheck_min_beat_bh_pct*100:.0f}% richiesto)")
                        if precheck_require_recommend_wfo:
                            _g3label = "OK" if _rec else "FALLITO"
                            print(f"  Gate 3 – recommend_wfo      : {_g3label}")
                        print(f"  → {'PASSATO' if pass_gate else 'SCARTATA'}")

                if not pass_gate:
                    continue

                # (3) RUN WFO
                if verbose:
                    print(f"[RUN] {strat}@{symbol}")

                portfolio, metrics, bh_portfolio, bh_metrics, wfo_res = run_strategy_fn(
                    symbol=symbol,
                    strategy_name=strat,
                    param_ranges={},  # vuoto -> default grid della strategia in run_strategy
                    start_date=start_date,
                    end_date=end_date,
                    ratio=ratio,
                    risk_free_rate=risk_free_rate,
                    show_progress=show_progress,
                    init_cash=init_cash,
                    fees=fees,
                    slippage=slippage,
                    price_col=price_col,
                    selection_metric=selection_metric,
                    warmup_years=warmup_years,
                    verbose=verbose,
                    save_results=save_results,
                    wfo_results_dir=wfo_results_dir,
                )

                results[(symbol, strat)] = {
                    "portfolio": portfolio,
                    "metrics": metrics,
                    "bh_portfolio": bh_portfolio,
                    "bh_metrics": bh_metrics,
                    "wfo_results": wfo_res,
                    "precheck": pre_payload,
                }

                met = _extract_metrics_from_df(metrics)
                bhm = _extract_metrics_from_df(bh_metrics)

                row = {
                    "Ticker": symbol,
                    "Strategy": strat,
                    "Start": start_date,
                    "End": end_date,
                    "Ratio": ratio,
                    "Warmup_Years": warmup_years,
                    **met,
                }

                for k, v in bhm.items():
                    row[f"BM_{k}"] = v

                row["Precheck_Mode"] = pre_payload.get("mode")
                row["Precheck_Reason"] = pre_payload.get("reason")
                row["Precheck_BH_TR"] = pre_payload.get("bh_total_return")
                row["Precheck_Stress_BestTR"] = pre_payload.get("stress_best_total_return")
                row["Precheck_Core_BestTR"] = pre_payload.get("core_best_total_return")
                row["Precheck_BeatBH%"] = pre_payload.get("beat_bh_pct")
                row["Precheck_StressRecommendWFO"] = pre_payload.get("stress_recommend_wfo")
                row["Precheck_CoreRecommendWFO"] = pre_payload.get("core_recommend_wfo")

                rows.append(row)

            except Exception as e:
                print(e)

            finally:
                if show_progress:
                    pbar.update(1)
                    pbar.refresh()

    finally:
        if show_progress:
            pbar.refresh()

    df = pd.DataFrame(rows)

    # ordinamento: Sharpe -> CAR -> Total Return -> MaxDD (invertito)
    sort = [("Sharpe", False), ("CAR %", False), ("Total Return %", False), ("MaxDD %", True)]
    by = [c for c, asc in sort if c in df.columns]
    asc = [asc for c, asc in sort if c in df.columns]
    if by:
        df = df.sort_values(by=by, ascending=asc, na_position="last").reset_index(drop=True)

    if return_catalog:
        extra["catalog"] = catalog_out
    if return_joblist:
        extra["joblist"] = joblist_out

    return df, results, extra
    
def wfo_strategy_panner(
    strategies=None,          # "cog_qqe", ["cog_qqe", ...], "*" o None -> autodetect
    tickers=None,             # "AVGO", ["AVGO","NVDA"], "*" o None -> autodetect
    ratios=None,              # "4:1", ["2:1","4:1"], "*" o None -> autodetect
    wfo_results_dir: str = "./wfo_results",
    order_by=None,
    top_k: int | None = None,
    print_summary: bool = True,
    export_csv_path: str | None = None,
    include_best_params: bool = False,
    require_bh: bool = True,          # se True, scarta righe senza B&H e raccoglie i path non conformi
    # --- parametro unico di selezione (unifica winners_only e top_per_ticker) ---
    selection: str | None = None      # None | "winners" | "best_return" | "best_dd"
):
    """
    Report massivo con confronto contro B&H.

    Selezione:
      - selection=None          -> nessun filtro (mostra tutte le strategie)
      - selection="winners"     -> mantiene righe con (Return vs B&H > 0) e (DD vs B&H < 0)
      - selection="best_return" -> mantiene righe con (Return vs B&H > 0)
      - selection="best_dd"     -> mantiene righe con (DD vs B&H < 0)

    Nota: se selection != None, PRIMA esclude le righe con Return vs B&H == 0.00 o DD vs B&H == 0.00
          (arrotondamento a 2 decimali per catturare anche -0.00).

    Ritorna:
      - se (selection is None) e print_summary=True:
            (summary_df, df_wfo_all, noncompliant_paths, styler)
        se print_summary=False:
            (summary_df, df_wfo_all, noncompliant_paths)

      - altrimenti (quando selection in {"winners","best_return","best_dd"}):
        se print_summary=True:
            (summary_df, df_wfo_all, noncompliant_paths, winners_list, styler)
        se print_summary=False:
            (summary_df, df_wfo_all, noncompliant_paths, winners_list)
    """
    import os, re, glob, numpy as np, pandas as pd

    # -------- helpers di base --------
    def _as_list(x):
        if x is None: return []
        if isinstance(x, str): return [x]
        try: return list(x)
        except TypeError: return [x]

    def _norm_wildcards(seq):
        seq = _as_list(seq)
        return seq if seq else ["*"]

    FILE_RE = re.compile(r"^portfolio_(?P<strategy>.+)_(?P<symbol>.+)_(?P<ratio>[^_]+)_results\.pkl$")

    def _discover_all(wdir):
        s_set, t_set, r_set = set(), set(), set()
        for strat_dir in glob.glob(os.path.join(wdir, "*")):
            if not os.path.isdir(strat_dir):
                continue
            for p in glob.glob(os.path.join(strat_dir, "portfolio_*_results.pkl")):
                m = FILE_RE.search(os.path.basename(p))
                if not m:
                    continue
                s_set.add(m.group("strategy"))
                t_set.add(m.group("symbol"))
                r_set.add(m.group("ratio"))
        return sorted(s_set), sorted(t_set), sorted(r_set)

    def _expand_from_fs(wdir, patterns_s, patterns_t, patterns_r):
        import fnmatch
        triples = set()
        for strat_dir in glob.glob(os.path.join(wdir, "*")):
            if not os.path.isdir(strat_dir):
                continue
            for p in glob.glob(os.path.join(strat_dir, "portfolio_*_results.pkl")):
                base = os.path.basename(p)
                m = FILE_RE.search(base)
                if not m:
                    continue
                s, t, r = m.group("strategy"), m.group("symbol"), m.group("ratio")
                if any(fnmatch.fnmatch(s, pat) for pat in patterns_s) and \
                   any(fnmatch.fnmatch(t, pat) for pat in patterns_t) and \
                   any(fnmatch.fnmatch(r, pat) for pat in patterns_r):
                    triples.add((s, t, r))
        return sorted(triples)

    # --- raccoglie file esistenti per un prefix base ---
    def _existing_files_for_base(base: str):
        cand = [
            base + "_results.pkl",
            base + "_wfo_results.pkl",
            base + "_bh_results.pkl",
            base + "_bh_metrics.pkl",
        ]
        return [os.path.abspath(p) for p in cand if os.path.exists(p)]

    # -------- best_params formatting --------
    def _format_params_cell_multiline_indented(dfw: pd.DataFrame) -> str:
        if not isinstance(dfw, pd.DataFrame) or "Year" not in dfw.columns or "Best_Params" not in dfw.columns:
            return "n/a"
        indent = " " * 12
        lines = []
        for _, row in dfw.sort_values("Year").iterrows():
            yr = int(row["Year"]) if pd.notna(row["Year"]) else "n/a"
            bp = row["Best_Params"]
            params_dict = None
            if isinstance(bp, dict):
                params_dict = bp
            else:
                try:
                    import ast
                    d = ast.literal_eval(str(bp))
                    if isinstance(d, dict):
                        params_dict = d
                except Exception:
                    pass
            if params_dict is None:
                lines.append(f"{yr}: {str(bp)}"); continue
            items = list(params_dict.items())
            if not items:
                lines.append(f"{yr}: -"); continue
            first_k, first_v = items[0]
            lines.append(f"{yr}: {first_k}={first_v}")
            for k, v in items[1:]:
                lines.append(f"{indent}{k}={v}")
        return "\n".join(lines) if lines else "n/a"

    def _get_stat(stats_obj, keys):
        for k in keys:
            try:
                if hasattr(stats_obj, "get"):
                    v = stats_obj.get(k)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        return float(v)
                if isinstance(stats_obj, pd.DataFrame):
                    if k in stats_obj.index:
                        v = stats_obj.loc[k]
                        v = v.iloc[0] if hasattr(v, "iloc") else v
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            return float(v)
                    if k in stats_obj.columns:
                        v = stats_obj[k].iloc[0]
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            return float(v)
                if isinstance(stats_obj, dict) and k in stats_obj:
                    v = stats_obj[k]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        return float(v)
            except Exception:
                continue
        return np.nan

    def _load_vbt_portfolio_stats(pfile):
        out = {"Total Return [%]": np.nan, "Max Drawdown [%]": np.nan, "Sharpe": np.nan, "Total Trades": np.nan}
        if not os.path.exists(pfile):
            return out
        try:
            import vectorbt as vbt
            pf = vbt.Portfolio.load(pfile)
            try: st = pf.stats()
            except Exception: st = {}
            out["Total Return [%]"]     = _get_stat(st, ["Total Return [%]", "Return [%]"])
            out["Max Drawdown [%]"]     = _get_stat(st, ["Max Drawdown [%]", "MaxDD [%]", "MaxDD"])
            out["Sharpe"]               = _get_stat(st, ["Sharpe Ratio", "Sharpe"])
            out["Total Trades"]         = _get_stat(st, ["Total Trades"])
            if np.isnan(out["Total Return [%]"]):
                try:
                    val = pf.value()
                    out["Total Return [%]"] = (float(val.iloc[-1]) / float(val.iloc[0]) - 1.0) * 100.0
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def _load_bh_metrics_from_pickle(bh_metrics_file):
        out = {"B&H Return [%]": np.nan, "B&H Max Drawdown [%]": np.nan, "B&H Sharpe": np.nan}
        if not os.path.exists(bh_metrics_file):
            return out
        try:
            import pandas as pd
            obj = pd.read_pickle(bh_metrics_file)
            out["B&H Return [%]"]        = _get_stat(obj, ["Total Return [%]", "Return [%]", "B&H Return [%]"])
            out["B&H Max Drawdown [%]"]  = _get_stat(obj, ["Max Drawdown [%]", "MaxDD [%]", "MaxDD", "B&H Max Drawdown [%]"])
            out["B&H Sharpe"]            = _get_stat(obj, ["Sharpe Ratio", "Sharpe", "B&H Sharpe"])
        except Exception:
            pass
        return out

    def _load_bh_portfolio_stats(bh_portfolio_file):
        out = {"B&H Return [%]": np.nan, "B&H Max Drawdown [%]": np.nan, "B&H Sharpe": np.nan}
        if not os.path.exists(bh_portfolio_file):
            return out
        try:
            import vectorbt as vbt
            pf = vbt.Portfolio.load(bh_portfolio_file)
            try: st = pf.stats()
            except Exception: st = {}
            out["B&H Return [%]"]        = _get_stat(st, ["Total Return [%]", "Return [%]"])
            out["B&H Max Drawdown [%]"]  = _get_stat(st, ["Max Drawdown [%]", "MaxDD [%]", "MaxDD"])
            out["B&H Sharpe"]            = _get_stat(st, ["Sharpe Ratio", "Sharpe"])
            if np.isnan(out["B&H Return [%]"]):
                try:
                    val = pf.value()
                    out["B&H Return [%]"] = (float(val.iloc[-1]) / float(val.iloc[0]) - 1.0) * 100.0
                except Exception:
                    pass
        except Exception:
            pass
        return out

    # -------- autodetect universi & filtri wildcard --------
    patterns_s = _norm_wildcards(strategies)
    patterns_t = _norm_wildcards(tickers)
    patterns_r = _norm_wildcards(ratios)

    _, _, _ = _discover_all(wfo_results_dir)  # rilevazione potenzialmente utile
    if patterns_s == ["*"] and patterns_t == ["*"] and patterns_r == ["*"]:
        triples = _expand_from_fs(wfo_results_dir, ["*"], ["*"], ["*"])
    else:
        triples = _expand_from_fs(wfo_results_dir, patterns_s, patterns_t, patterns_r)

    if not triples:
        raise FileNotFoundError(
            f"Nessun risultato trovato in '{wfo_results_dir}' con i pattern: "
            f"strategies={patterns_s}, tickers={patterns_t}, ratios={patterns_r}"
        )

    # -------- parsing file & building rows --------
    rows, wfo_all, noncompliant_paths = [], [], []
    for strategy, symbol, r_str in triples:
        base_dir = f"{wfo_results_dir}/{strategy}"
        base = f"{base_dir}/portfolio_{strategy}_{symbol}_{r_str}"
        p_file            = base + "_results.pkl"
        wfo_file          = base + "_wfo_results.pkl"
        bh_portfolio_file = base + "_bh_results.pkl"
        bh_metrics_file   = base + "_bh_metrics.pkl"

        # controllo B&H: se richiedo B&H e non ci sono file → mark non conformi e salta
        if require_bh and (not os.path.exists(bh_metrics_file) and not os.path.exists(bh_portfolio_file)):
            noncompliant_paths.extend(_existing_files_for_base(base))
            continue

        # Stats portfolio
        stats = _load_vbt_portfolio_stats(p_file)

        # B&H metrics
        bh = _load_bh_metrics_from_pickle(bh_metrics_file)
        if np.isnan(bh["B&H Return [%]"]) and os.path.exists(bh_portfolio_file):
            bh = _load_bh_portfolio_stats(bh_portfolio_file)

        # se metriche B&H tutte NaN → non conforme
        if require_bh and all(np.isnan(bh[k]) for k in ["B&H Return [%]", "B&H Max Drawdown [%]", "B&H Sharpe"]):
            noncompliant_paths.extend(_existing_files_for_base(base))
            continue

        # Best Params (solo se richiesti)
        best_params = None
        if include_best_params and os.path.exists(wfo_file):
            try:
                dfw = pd.read_pickle(wfo_file)
                if isinstance(dfw, pd.DataFrame):
                    best_params = _format_params_cell_multiline_indented(dfw)
                    tmp = dfw.copy()
                    tmp["Strategy"], tmp["Ticker"], tmp["Ratio"] = strategy, symbol, r_str
                    wfo_all.append(tmp)
            except Exception:
                pass

        # Calcoli differenze vs B&H
        tr     = stats["Total Return [%]"]
        dd     = stats["Max Drawdown [%]"]
        sharpe = stats["Sharpe"]
        trades = stats["Total Trades"]

        bh_ret = bh["B&H Return [%]"]
        bh_dd  = bh["B&H Max Drawdown [%]"]
        bh_sh  = bh["B&H Sharpe"]

        # --- confronta i MODULI dei drawdown: <0 = DD migliore del B&H
        def _dd_mag(x):
            try: return abs(float(x))
            except Exception: return np.nan

        dd_mag    = _dd_mag(dd)
        bh_dd_mag = _dd_mag(bh_dd)

        excess_vs_bh  = tr - bh_ret if (pd.notna(tr) and pd.notna(bh_ret)) else np.nan
        dd_diff_vs_bh = (dd_mag - bh_dd_mag) if (pd.notna(dd_mag) and pd.notna(bh_dd_mag)) else np.nan

        row = {
            "Strategy": strategy,
            "Ticker": symbol,
            "Ratio": r_str,
            "Total Return [%]": tr,
            "Max DD [%]": dd,
            "Sharpe": sharpe,
            "Total Trades": trades,
            "B&H Return [%]": bh_ret,
            "B&H Max DD [%]": bh_dd,
            "B&H Sharpe": bh_sh,
            "Return vs B&H [%]": excess_vs_bh,
            "DD vs B&H [%]": dd_diff_vs_bh,
        }
        if include_best_params:
            row["Best_Params"] = (best_params if best_params is not None else "n/a")

        rows.append(row)

    if require_bh and not rows:
        print("Nessuna combinazione valida dopo il filtro 'require_bh'. ")
        print(f"Non conformi: {noncompliant_paths}")
        need_selection = selection is not None
        if need_selection:
            return ([], [], noncompliant_paths, [], []) if print_summary else ([], [], noncompliant_paths, [])
        return ([], [], noncompliant_paths, []) if print_summary else ([], [], noncompliant_paths)

    summary_df = pd.DataFrame(rows)

    # --- flag 'Winning' (informativo, non usato per filtrare) ---
    def _to_num(s): return pd.to_numeric(s, errors="coerce")
    if not summary_df.empty:
        summary_df["Winning"] = (_to_num(summary_df["Return vs B&H [%]"]) > 0) & \
                                (_to_num(summary_df["DD vs B&H [%]"]) < 0)
    else:
        summary_df["Winning"] = pd.Series(dtype=bool)

    # -------- ordinamento base (prima dei filtri opzionali) --------
    def _normalize_order_by(order_by, defaults):
        if not order_by:
            return defaults
        out = []
        for item in order_by:
            if isinstance(item, tuple):
                col, dir_or_bool = item
                asc = (dir_or_bool if isinstance(dir_or_bool, bool) else str(dir_or_bool).lower() == "asc")
            else:
                part = str(item).split("|")
                col = part[0].strip()
                asc = (len(part) > 1 and part[1].strip().lower() == "asc")
            out.append((col, asc))
        return out

    defaults = [
        ("Return vs B&H [%]", False),
        ("DD vs B&H [%]", True),
        ("Total Return [%]", False),
        ("Max DD [%]", True),
        ("Sharpe", False),
    ]
    order_pairs = _normalize_order_by(order_by, defaults)
    sort_cols = [c for c, _ in order_pairs if c in summary_df.columns]
    sort_ascs = [a for c, a in order_pairs if c in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(by=sort_cols, ascending=sort_ascs, na_position="last").reset_index(drop=True)

    # -------- SELEZIONE SEMPLICE UNIFICATA --------
    winners_list = []
    if selection is not None:
        if selection not in ("winners", "best_return", "best_dd"):
            raise ValueError("selection deve essere None, 'winners', 'best_return' oppure 'best_dd'.")

        rvb = _to_num(summary_df["Return vs B&H [%]"])
        ddv = _to_num(summary_df["DD vs B&H [%]"])

        # 1) Escludi righe con 0.00 o -0.00 in QUALSIASI delle due colonne
        rvb_r = rvb.round(2)
        ddv_r = ddv.round(2)
        nz_mask = (rvb_r != 0) & (ddv_r != 0)
        summary_df = summary_df[nz_mask].copy().reset_index(drop=True)

        # Serie aggiornate dopo il filtro 0.00/-0.00
        rvb = _to_num(summary_df["Return vs B&H [%]"])
        ddv = _to_num(summary_df["DD vs B&H [%]"])

        # 2) Applica la selezione richiesta
        if selection == "winners":
            mask = (rvb > 0) & (ddv < 0)
        elif selection == "best_return":
            mask = (rvb > 0)
        elif selection == "best_dd":
            mask = (ddv < 0)

        summary_df = summary_df[mask].copy().reset_index(drop=True)
        winners_list = list(summary_df[["Strategy", "Ticker", "Ratio"]].itertuples(index=False, name=None))

    # -------- stampa con stile --------
    printable_df = summary_df.head(top_k) if (top_k is not None and top_k > 0) else summary_df
    styler = None

    if print_summary:
        try:
            from IPython.display import display
            def color_excess(v):
                try: v = float(v)
                except Exception: return ""
                return "background-color: #d1fadf; color: #054f31; font-weight: 600;" if v > 0 else \
                       "background-color: #fee4e2; color: #7a271a; font-weight: 600;" if v < 0 else ""
            def color_dd_diff(v):
                try: v = float(v)
                except Exception: return ""
                if np.isnan(v): return ""
                return "background-color: #d1fadf; color: #054f31; font-weight: 600;" if v < 0 else \
                       "background-color: #fee4e2; color: #7a271a; font-weight: 600;" if v > 0 else \
                       "background-color: #f2f4f7; color: #344054; font-weight: 600;"
            num_cols = printable_df.select_dtypes(include="number").columns
            fmt_map = {c: "{:.2f}" for c in num_cols}
            if "Total Trades" in fmt_map: fmt_map["Total Trades"] = "{:.0f}"
            styler = (printable_df.style
                      .format(fmt_map)
                      .applymap(color_excess, subset=["Return vs B&H [%]"])
                      .applymap(color_dd_diff, subset=["DD vs B&H [%]"]))
            print("\n=== SUMMARY (Return vs B&H, DD diff vs B&H, Sharpe, Trades) ===")
            if selection is not None:
                print(f"(selection: {selection}; righe: {len(summary_df)})")
            try:
                my_display(styler)  # se definito nell'ambiente
            except Exception:
                display(styler)
        except Exception:
            print("\n=== SUMMARY (Return vs B&H, DD diff vs B&H, Sharpe, Trades) ===")
            if selection is not None:
                print(f"(selection: {selection}; righe: {len(summary_df)})")
            print(printable_df.to_string(index=False))

    # -------- export opzionale --------
    df_wfo_all = pd.DataFrame()
    if include_best_params and wfo_all:
        df_wfo_all = pd.concat(wfo_all, ignore_index=True)
    if export_csv_path:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        summary_df.to_csv(export_csv_path, index=False)
        if not df_wfo_all.empty:
            df_wfo_all.to_csv(export_csv_path.replace(".csv", "_wfo.csv"), index=False)
        noncompliant_paths = sorted(set(noncompliant_paths))
        if require_bh and noncompliant_paths:
            with open(export_csv_path.replace(".csv", "_noncompliant.txt"), "w") as f:
                f.write("\n".join(noncompliant_paths))
        print(f"\nFile esportati in: {export_csv_path}")

    # dedup + sort anche nel return
    noncompliant_paths = sorted(set(noncompliant_paths))
    
    return (summary_df, df_wfo_all, noncompliant_paths, winners_list, styler)

def strategy_registry(
    patterns: Union[str, Iterable[str]] = "*",
    *,
    namespace: Optional[Dict[str, Any]] = None,
    require_param_ranges: bool = True,
    case_sensitive: bool = False,
    sort: bool = True,
) -> List[str]:
    """
    Ritorna una lista di NOME-STRATEGIA (senza prefisso 'strategy_') che matcha i pattern.
    Convenzioni richieste:
      - funzione:    strategy_<nome>
      - param grid:  strategy_<nome>_param_ranges

    Parametri
    ---------
    patterns : str | Iterable[str]
        Wildcard stile shell. Esempi: "supertrend*", "*breakout*", ["supertrend*","*ema*"].
        Il match avviene sul NOME-STRATEGIA (cioè senza 'strategy_').
    namespace : dict, opz.
        Dizionario in cui cercare (default: globals()).
    require_param_ranges : bool
        Se True (default) include solo strategie che hanno ANCHE la griglia parametri.
    case_sensitive : bool
        Match case-sensitive (False di default).
    sort : bool
        Ordina alfabeticamente l’output (default True).

    Ritorna
    -------
    List[str] : ad es. ["supertrend_basic", "supertrend_adaptive_vol"]
    """
    if namespace is None:
        namespace = globals()

    # normalizza patterns a lista
    if isinstance(patterns, (str, bytes)):
        patterns = [patterns]

    # helper match su short name
    def _match_short(short: str) -> bool:
        target = short if case_sensitive else short.lower()
        for pat in patterns:
            pat_ = pat if case_sensitive else pat.lower()
            if fnmatch.fnmatch(target, pat_):
                return True
        return False

    strategies: List[str] = []
    # trova tutte le funzioni "strategy_*"
    for k, v in namespace.items():
        if not k.startswith("strategy_") or k.endswith("_param_ranges"):
            continue
        if not callable(v):
            continue
        short = k[len("strategy_"):]  # nome strategia senza prefisso
        if not _match_short(short):
            continue

        # se richiesto, verifica presenza griglia
        if require_param_ranges:
            grid_key = f"strategy_{short}_param_ranges"
            if grid_key not in namespace or not isinstance(namespace[grid_key], dict):
                continue

        strategies.append(short)

    # unici + ordine
    strategies = list(dict.fromkeys(strategies))  # preserva ordine di scoperta
    if sort:
        strategies.sort()
    return strategies

# Alias comodo
def list_strategies(pattern: str = "*", **kwargs) -> List[str]:
    """Alias minimale per strategy_registry(patterns=pattern, **kwargs)."""
    return strategy_registry(patterns=pattern, **kwargs)

# # 1) Tutte le strategie che contengono "supertrend"
# strategies = strategy_registry("supertrend*")
# # -> ["supertrend_adaptive_vol", "supertrend_basic", "supertrend_breakout_trail", ...]

# # 2) Più pattern (OR logico)
# strategies = strategy_registry(["*supertrend*", "*breakout*"])

# # 3) Permettere strategie senza griglia (non consigliato per il panel)
# strategies = strategy_registry("*", require_param_ranges=False)

def load_ts_wfo_summary(file_path: str) -> pd.DataFrame:
    """
    Carica il summary WFO salvato da save_rotational_wfo_summary().

    - ignora l'header commentato grazie a comment="#"
    - imposta l'indice su 'Year'
    """
    df = pd.read_csv(
        file_path,
        index_col="Year",
        comment="#",
    )
    return df

def load_ts(
    symbol: str,
    strategy: str,
    wfo_results_dir: str = "WFO_RESULTS",
    ratio: str = "4:1",
    vbt_plot_width: int = 900,
    *,
    show_result: bool = True,
    create_structure: bool = False,
    only_wfo_results: bool = False,
    legacy: bool = False,
) -> Tuple[Optional[vbt.Portfolio], Optional[vbt.Portfolio], Optional[pd.DataFrame]]:
    """
    Carica e (opzionalmente) visualizza i risultati WFO per (symbol, strategy).

    Parametri
    ---------
    show_result : bool, default True
        - True  -> comportamento completo (print, display, stats, plot)
        - False -> solo load silenzioso, nessuna stampa o grafico
    """
    from pathlib import Path
    import shutil
    
    portfolio, portfolio_bh, wfo_results = {},{},{}

    base_dir = f"{wfo_results_dir}/{strategy}"
    portfolio_result_file = f"{base_dir}/portfolio_{strategy}_{symbol}_{ratio}_results.pkl"
    portfolio_bh_result_file = f"{base_dir}/portfolio_{strategy}_{symbol}_{ratio}_bh_results.pkl"
    wfo_result_file_pkl = f"{base_dir}/portfolio_{strategy}_{symbol}_{ratio}_wfo_results.pkl"
    wfo_result_file = f"{base_dir}/portfolio_{strategy}_{symbol}_{ratio}_wfo_results.csv"


    # Retro compatibilita luca
    if legacy:
        if os.path.exists(wfo_result_file_pkl):
            wfo_results_pkl = pd.read_pickle(wfo_result_file_pkl).set_index("Year")
            cols_to_drop = [c for c in wfo_results_pkl.columns if c.startswith("Best_")]
            wfo_results_clean = wfo_results_pkl.drop(columns=cols_to_drop)
            
            # display(wfo_results_clean)
            # print(f"Retrocompatibilta: scrivo il file {wfo_result_file}")   
            wfo_results_clean.to_csv(wfo_result_file)

    if not (os.path.exists(portfolio_result_file) and os.path.exists(wfo_result_file)):
        print(f"{symbol}: risultati portfolio {portfolio_result_file} e/o elaborazione WFO {wfo_result_file} inesistenti o incompleti. Ignoro")
        return None, None, None

    if create_structure:
        # print(f"cp {wfo_result_file} WFO_TS_RESULTS")
        dest_dir = Path(f"WFO_TS_RESULTS/{strategy}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(portfolio_result_file, dest_dir)
        shutil.copy(portfolio_bh_result_file, dest_dir)
        shutil.copy(wfo_result_file, dest_dir)

    # --- Load WFO Results ---
    try:
        wfo_results = load_ts_wfo_summary(wfo_result_file)
        # display(wfo_results)
            
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento di wfo_results: {e}") from e
    

    if only_wfo_results:
        return None, None, wfo_results
        
    # --- Load portafogli ---
    try:
        portfolio = vbt.Portfolio.load(portfolio_result_file)
        portfolio_bh = vbt.Portfolio.load(portfolio_bh_result_file)
    except Exception as e:
        print(f"Errore nel caricamento del portfolio: {e}")
        return None, None, None

    if show_result:
        # ============================================================
        # MODALITÀ VERBOSA (comportamento originale)
        # ============================================================
    
        print(f"Risultati elaborazione WFO per ticker {BOLD}{symbol}{RESET} strategia {BOLD}{strategy}{RESET}")
    
        if wfo_results is not None:
            with pd.option_context('display.max_colwidth', None):
                display(wfo_results)
    
        summary = None
        try:
            summary = print_summary(portfolio)
        except NameError:
            try:
                print("print_summary non disponibile: stampo stats sintetiche.")
                print(portfolio.stats())
            except Exception as e:
                print(f"Impossibile stampare stats portfolio: {e}")
        except Exception as e:
            print(f"Errore in print_summary: {e}")
    
        try:
            print(f"\n{BOLD}Strategy stats:{RESET}")
            print(portfolio.stats())
            print("")
            print(f"\n{BOLD}Buy&Hold stats:{RESET}")
            print(portfolio_bh.stats())
        except Exception as e:
            print(f"Errore in portfolio.stats(): {e}")
    
        try:
            fig = portfolio.plot(
                width=vbt_plot_width,
                subplots=[
                    'cum_returns',
                    'orders',
                    'trade_pnl',
                    'drawdowns',
                    'underwater',
                    'gross_exposure',
                    'trades'
                ]
            )
            fig.show()
        except Exception as e:
            print(f"Errore nel plotting del portfolio: {e}")
    
    return portfolio, portfolio_bh, wfo_results




# ===================================
# Core: Walk-Forward Optimization (WFO)
# ===================================

def _evaluate_combo_performance(
    train_data: pd.DataFrame,
    strategy_func,
    keys: List[str],
    combo: Tuple,
    init_cash: float,
    fees: float,
    slippage: float,
    price_col : str,
) -> float:
    """Calcola la performance (total_return) per una singola combinazione."""
    params_dict = {k: v for k, v in zip(keys, combo)}
    entries, exits = strategy_func(train_data.copy(), params_dict, year=None)
    portfolio = backtest_from_signals(
        train_data, entries, exits,
        init_cash=init_cash, fees=fees, slippage=slippage, price_col =price_col 
    )
    return float(portfolio.total_return())

def recovery_factor(equity: pd.Series) -> float:
    """
    Calcola il Recovery Factor = Total Return / Max Drawdown
    equity: serie del capitale cumulato della strategia
    """
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1).min()
    rf = total_return / abs(dd) if dd != 0 else np.nan
    return rf




def walk_forward_optimization(
    df: pd.DataFrame,
    param_ranges: Dict[str, Iterable],
    strategy_func,                     # firma: strategy_func(data, params_dict, year=None) -> (entries, exits)
    train_years: int = 4,
    wfo_start_year: Optional[int] = None,
    wfo_end_year: Optional[int] = None,
    show_progress: bool = True,
    init_cash: float = 100_000.0,
    fees: float = 0.001,
    slippage: float = 0.002,
    price_col: str = 'Open',
    n_jobs: Optional[int] = None,      # None/1 => sequenziale; >1 => parallelo se joblib disponibile
    parallel_backend: str = 'loky',
    batch_size: int = 1,               # aggiornamenti frequenti della barra
    pre_dispatch: str = "2*n_jobs",    # mantiene la coda alimentata senza saturare
    selection_metric: Union[str, Callable] = "total_return",
    engine_param_grid: Optional[Dict[str, Iterable]] = None,  # override locale della ENGINE_PARAM_GRID globale
) -> pd.DataFrame:
    """
    Esegue un'ottimizzazione Walk-Forward (WFO) ANNUALE.
    Per ogni anno di test:
      - ottimizza i parametri della strategia sui dati di TRAIN
      - opzionalmente ottimizza anche i parametri di engine (SL/TP/time_exit/risk)

    Output: DataFrame con una riga per anno e colonne:
      - 'Year'
      - 'Best_Params'          (dict parametri strategia)
      - 'Best_Engine_Params'   (dict parametri engine o None)
      - colonne "flattened" dei parametri strategia (una per chiave di param_ranges)
      - colonne "flattened" dei parametri engine, con prefisso 'eng_' (se engine grid attiva)
    """

    # --- 1) Anni base WFO ---
    base_start, base_end = infer_wfo_base_years(
        df, wfo_start_year=wfo_start_year, wfo_end_year=wfo_end_year
    )
    test_start = base_start + train_years
    test_end   = base_end
    if test_start > test_end:
        raise ValueError(
            f"Intervallo insufficiente per WFO: base_start={base_start}, base_end={base_end}, train_years={train_years}"
        )

    # --- 2) Griglia strategia ---
    keys        = list(param_ranges.keys())
    combos_list = list(param_product(param_ranges))
    n_combos    = len(combos_list)
    n_years     = (test_end - test_start + 1)
    total_cases = n_combos * n_years

    # --- 3) Parallelizzazione ---
    use_parallel = (
        (n_jobs is not None and int(n_jobs) > 1)
        and _HAVE_JOBLIB
    )
    n_jobs_eff = int(n_jobs) if use_parallel else 1

    # --- 4) Engine grid effettiva ---
    if engine_param_grid is None:
        engine_grid_eff = ENGINE_PARAM_GRID
    else:
        engine_grid_eff = engine_param_grid

    # --- 5) Helper di scoring ---
    def _score(pf) -> float:
        # callable custom
        if callable(selection_metric):
            try:
                s = float(selection_metric(pf))
                return s if np.isfinite(s) else -np.inf
            except Exception:
                return -np.inf

        # total return
        if selection_metric == "total_return":
            try:
                s = float(pf.total_return())
                return s if np.isfinite(s) else -np.inf
            except Exception:
                return -np.inf

        # recovery factor = total_return / |max_drawdown|
        if selection_metric == "recovery_factor":
            try:
                tr = float(pf.total_return())
                try:
                    mdd = float(pf.max_drawdown())
                except Exception:
                    eq = None
                    for attr in ("equity", "equity_curve", "value", "portfolio_value", "balance"):
                        if hasattr(pf, attr):
                            eq = getattr(pf, attr)
                            break
                    if isinstance(eq, pd.Series):
                        roll_max = eq.cummax()
                        dd_series = (eq / roll_max - 1.0)
                        mdd = float(dd_series.min())
                    else:
                        return -np.inf
                denom = abs(mdd) if mdd != 0 else np.nan
                rf = tr / denom if np.isfinite(denom) else np.nan
                return float(rf) if np.isfinite(rf) else -np.inf
            except Exception:
                return -np.inf

        # default: total_return
        try:
            s = float(pf.total_return())
            return s if np.isfinite(s) else -np.inf
        except Exception:
            return -np.inf

    # --- 6) Helper per valutare UNA combo di parametri strategici ---
    def _eval_combo(train_data: pd.DataFrame, combo: Tuple) -> Tuple[float, Optional[Dict]]:
        """
        Restituisce:
          (score_migliore, best_engine_params_dict_o_None)
        """
        params_dict = {k: v for k, v in zip(keys, combo)}
        entries, exits = strategy_func(train_data.copy(), params_dict, year=None)

        # Caso base: engine-grid disattivata o non definita
        if (not USE_ENGINE_GRID) or (engine_grid_eff is None) or (len(engine_grid_eff) == 0):
            pf = backtest_from_signals(
                train_data,
                entries,
                exits,
                init_cash=init_cash,
                fees=fees,
                slippage=slippage,
                price_col=price_col,
                sl_pct=None,
                tp_pct=None,
                time_exit=None,
                risk_per_trade_pct=None,
            )
            return _score(pf), None

        # Caso avanzato: engine-grid attiva
        ekeys        = list(engine_grid_eff.keys())
        evalues_list = list(engine_grid_eff.values())

        best_metric = -np.inf
        best_engine_params: Optional[Dict] = None

        for evalues in product(*evalues_list):
            eng = dict(zip(ekeys, evalues))

            pf = backtest_from_signals(
                train_data,
                entries,
                exits,
                init_cash=init_cash,
                fees=fees,
                slippage=slippage,
                price_col=price_col,
                sl_pct=eng.get("sl_pct"),
                tp_pct=eng.get("tp_pct"),
                time_exit=eng.get("time_exit"),
                risk_per_trade_pct=eng.get("risk_per_trade_pct"),
            )

            metric = _score(pf)
            if metric > best_metric:
                best_metric = metric
                best_engine_params = eng

        return best_metric, best_engine_params

    # --- 7) Loop WFO sugli anni di test ---
    results: List[Dict] = []
    pbar = tqdm(total=total_cases, disable=not show_progress, desc="WFO grid cases")

    for test_year in range(test_start, test_end + 1):
        # finestra train
        train_start = test_year - train_years
        train_end   = test_year - 1
        train_data  = df[(df.index.year >= train_start) & (df.index.year <= train_end)].copy()

        best_engine_params_for_year: Optional[Dict] = None

        if use_parallel:
            if _HAVE_TQDM_JOBLIB:
                with tqdm_joblib(pbar):
                    perfs = Parallel(
                        n_jobs=n_jobs_eff,
                        backend=parallel_backend,
                        batch_size=batch_size,
                        pre_dispatch=pre_dispatch
                    )(
                        delayed(_eval_combo)(train_data, combo) for combo in combos_list
                    )
            else:
                perfs = Parallel(
                    n_jobs=n_jobs_eff,
                    backend=parallel_backend,
                    batch_size=batch_size,
                    pre_dispatch=pre_dispatch
                )(
                    delayed(_eval_combo)(train_data, combo) for combo in combos_list
                )
                pbar.update(n_combos)

            # perfs: lista di tuple (score, best_engine_params)
            scores = np.array([p[0] for p in perfs], dtype=float)
            best_idx = int(np.nanargmax(scores))
            best_params_tuple = combos_list[best_idx]
            best_engine_params_for_year = perfs[best_idx][1]

        else:
            best_perf = -np.inf
            best_params_tuple = None
            best_engine_params_for_year = None

            for combo in combos_list:
                perf, eng_best = _eval_combo(train_data, combo)
                if perf > best_perf:
                    best_perf = perf
                    best_params_tuple = combo
                    best_engine_params_for_year = eng_best
                pbar.update(1)

        # --- 8) Costruisci la riga risultato ---
        best_params = {k: v for k, v in zip(keys, best_params_tuple)}

        row: Dict[str, object] = {
            'Year'              : test_year,
            # luca
            # 'Best_Params'       : best_params,
            # 'Best_Engine_Params': best_engine_params_for_year
        }

        # Espandi i parametri strategia in colonne dedicate
        for pk, pv in best_params.items():
            row[pk] = pv

        # Espandi i parametri engine (se presenti) con prefisso 'eng_'
        if best_engine_params_for_year is not None:
            for ek, ev in best_engine_params_for_year.items():
                row[f"eng_{ek}"] = ev

        results.append(row)

    pbar.close()
    # return pd.DataFrame(results)
    return pd.DataFrame(results).set_index("Year")



# ============================
# RISOLUZIONE STRATEGIA E RATIO
# ============================

def _parse_ratio(ratio):
    """Accetta forme tipo "4:1", (4,1) o dict {"train":4,"test":1}. Ritorna (train_years, test_years)."""
    if isinstance(ratio, str):
        parts = ratio.replace(" ", "").split(":")
        if len(parts) != 2:
            raise ValueError("ratio string must be like '4:1'")
        return int(parts[0]), int(parts[1])
    if isinstance(ratio, (tuple, list)) and len(ratio) == 2:
        return int(ratio[0]), int(ratio[1])
    if isinstance(ratio, dict):
        return int(ratio.get("train")), int(ratio.get("test", 1))
    raise ValueError("Unsupported ratio format")

def _resolve_strategy(strategy_name: str):
    """Dato un nome, cerca `strategy_<name>` e `strategy_<name>_param_ranges` nel namespace globale."""
    func_name = f"strategy_{strategy_name}"
    grid_name = f"strategy_{strategy_name}_param_ranges"
    g = globals()
    if func_name not in g:
        raise ValueError(f"Funzione {func_name} non trovata")
    if grid_name not in g:
        raise ValueError(f"Griglia parametri {grid_name} non trovata")
    return g[func_name], g[grid_name]  # default grid

def get_strategy_name_from_func(func) -> str:
    # unwrap (decoratori con functools.wraps)
    real = getattr(func, "__wrapped__", func)
    name = getattr(real, "__name__", None)
    
    # caso standard: la funzione si chiama "strategy_<name>"
    if name and name.startswith("strategy_"):
        return name.split("strategy_", 1)[1]
    
    # fallback: cerca un alias nel globals() che punti allo stesso oggetto
    for k, v in globals().items():
        if k.startswith("strategy_") and v is func:
            return k.split("strategy_", 1)[1]
    
    raise ValueError("La funzione non sembra essere una strategy_* registrata.")



# ============================
# FUNZIONE GENERALE DI ESECUZIONE
# ============================

def _debug_vbt_from_signals_config(
    *,
    price_col: str,
    freq,
    fees,
    slippage,
    direction,
    accumulate,
    verbose: bool = True
):
    """
    Stampa diagnostica completa della configurazione effettiva usata da
    vbt.Portfolio.from_signals nel framework.

    Serve a:
    - verificare versione vectorbt
    - verificare parametri espliciti passati
    - ispezionare i default impliciti via vbt.settings
    - evitare divergenze silenziose tra backtest
    """
    if not verbose:
        return

    import vectorbt as vbt
    import inspect

    print("\n[vbt.from_signals CONFIG]")
    print(f"  vectorbt version : {vbt.__version__}")
    print(f"  price_col        : {price_col}")
    print(f"  freq             : {freq}")
    print(f"  fees             : {fees}")
    print(f"  slippage         : {slippage}")
    print(f"  direction        : {direction}")
    print(f"  accumulate       : {accumulate}")

    # Settings globali di vectorbt (sezione portfolio)
    try:
        portfolio_settings = vbt.settings.get("portfolio", {})
        print("  vbt.settings['portfolio'] :")
        if isinstance(portfolio_settings, dict) and portfolio_settings:
            for k, v in portfolio_settings.items():
                print(f"    {k}: {v}")
        else:
            print("    <empty or not set>")
    except Exception as e:
        print("  vbt.settings['portfolio'] : <unable to read>")
        print(f"    error: {e}")

    print("[/vbt.from_signals CONFIG]\n")

def backtest_from_signals(
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 100_000.0,
    fees: float = 0.001,
    slippage: float = 0.002,
    price_col: str = "Open",
    freq: str = "1D",
    accumulate: bool = False,           # ATTENZIONE!
    direction: str = 'longonly',         # ATTENZIONE!
    debug: bool = False,
    # --- parametri di engine opzionali (per WFO / USE_ENGINE_GRID) ---
    sl_pct: float | None = None,              # stop-loss percentuale (es. 0.10 = -10%)
    tp_pct: float | None = None,              # take-profit percentuale (es. 0.20 = +20%)
    time_exit: int | None = None,             # chiusura dopo N barre se non è scattato altro exit
    risk_per_trade_pct: float | None = None,  # % del capitale allocata per trade (size)
) -> vbt.Portfolio:
    """
    Backtest finale con vectorbt usando i segnali così come arrivano (già shiftati a monte).
    Questa versione evita l'uso di fillna/ffill/bfill per eliminare il FutureWarning di pandas.
    """

    if debug:
        _debug_vbt_from_signals_config(
            price_col=price_col,
            freq=freq,
            direction=direction,
            accumulate=accumulate,
            fees=fees,
            slippage=slippage,
            verbose=verbose
        )

    if price_col not in data.columns:
        raise ValueError(f"Colonna '{price_col}' non trovata nei dati: {list(data.columns)}")

    # --- entries / exits come array booleani NumPy, con NaN -> False ---
    # pandas >= 1.5: to_numpy supporta na_value
    try:
        entries_np = entries.to_numpy(dtype=bool, na_value=False)
        exits_np   = exits.to_numpy(dtype=bool, na_value=False)
    except TypeError:
        # fallback per versioni più vecchie di pandas
        entries_np = entries.fillna(False).astype(bool).to_numpy()
        exits_np   = exits.fillna(False).astype(bool).to_numpy()

    # --- time-exit via NumPy: OR logico con exit dopo N barre dall'entry ---
    if time_exit is not None and int(time_exit) > 0:
        t = int(time_exit)
        # array booleano per le uscite temporali
        exits_time_np = np.zeros_like(entries_np, dtype=bool)
        if t < len(entries_np):
            # per gli indici t..end, l'exit_time si attiva se c'era un'entry t barre prima
            exits_time_np[t:] = entries_np[:-t]
        # combina: exit "logica" OR exit per time-stop
        exits_np = np.logical_or(exits_np, exits_time_np)

    # --- kwargs per Portfolio.from_signals ---
    pf_kwargs = dict(
        close=data[price_col],
        entries=entries_np,
        exits=exits_np,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
        # accumulate=False,           # ATTENZIONE!
        # direction='longonly'        # ATTENZIONE!
    )

    # --- stop-loss / take-profit percentuali (scalar) ---
    if sl_pct is not None:
        pf_kwargs["sl_stop"] = float(sl_pct)
    if tp_pct is not None:
        pf_kwargs["tp_stop"] = float(tp_pct)

    # --- dimensione posizione basata su risk_per_trade_pct, senza fillna ---
    if risk_per_trade_pct is not None and risk_per_trade_pct > 0:
        cash_per_trade = init_cash * float(risk_per_trade_pct)
        price_arr = data[price_col].to_numpy(dtype=float)
        # size = cash_per_trade / prezzo, con gestione esplicita di nan/inf
        size_arr = cash_per_trade / price_arr
        mask_bad = ~np.isfinite(size_arr)
        size_arr[mask_bad] = 0.0
        pf_kwargs["size"] = size_arr

    portfolio = vbt.Portfolio.from_signals(**pf_kwargs)
    return portfolio

def apply_wfo_and_build_signals(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    strategy_func,
) -> Tuple[pd.Series, pd.Series]:
    """
    Applica i best params per anno e costruisce i segnali finali.
    - Calcola indicatori su [year-1, year]
    - Passa year=year alla strategia (la strategia applica il proprio filtro + eventuale shift)
    - Maschera per sicurezza solo all'anno 'year' e unisce nei vettori combinati
    """
    combined_entries = pd.Series(False, index=df.index)
    combined_exits   = pd.Series(False, index=df.index)

    # param_cols = [c for c in summary_df.columns if c.endswith("_range")]
    param_cols = [c for c in summary_df.columns]

    for year, row in summary_df.iterrows():
        year = int(year)
        # print(f"[anno: {year}")
    # for _, row in summary_df.iterrows():
    #     year = int(row["Year"])
        # params = row["Best_Params"]  # atteso dict dalla WFO unificata
        # luca
        # params "stile rotazionale": flat da colonne, niente Best_Params
        params = {c: row[c] for c in param_cols}
    
        # opzionale: cast coerente (molti param sono int)
        for k, v in params.items():
            if pd.notna(v):
                try:
                    params[k] = int(v)
                except Exception:
                    pass

        if params is None:
            continue

        # finestra di due anni per stabilizzare gli indicatori
        two_years = df[(df.index.year >= year - 1) & (df.index.year <= year)].copy()

        # la strategia filtra + shifta internamente sull'anno indicato
        entries_y, exits_y = strategy_func(two_years, params, year=year)

        # maschera all'anno corrente e unisci
        mask_year = (entries_y.index.year == year)
        entries_y = entries_y[mask_year].astype(bool).fillna(False)
        exits_y   = exits_y[mask_year].astype(bool).fillna(False)

        combined_entries.loc[entries_y.index] = entries_y
        combined_exits.loc[exits_y.index]     = exits_y

    return combined_entries, combined_exits
    
def _apply_strategy_augment(
    df: pd.DataFrame,
    strategy_name: str,
    strategy_func,
    start_date: str,
    end_date: str,
    loader,                 # es. load_ohlcv(symbol, start, end) -> DataFrame
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Se nel modulo della strategia esiste una funzione:
        ind_<strategy_name>_augment_df(df, start, end, loader) -> df_enhanced
    la esegue e ritorna il df arricchito. Altrimenti ritorna df invariato.
    """
    try:
        import importlib
        mod = importlib.import_module(strategy_func.__module__)
        hook_name = f"ind_{strategy_name}_augment_df"
        hook_fn = getattr(mod, hook_name, None)
        if callable(hook_fn):
            if verbose:
                print(f"[augment] Invoco hook '{hook_name}' del modulo '{strategy_func.__module__}'")
            return hook_fn(df, start_date, end_date, loader)
    except Exception as e:
        if verbose:
            print(f"[WARN] augment hook fallito: {e}")
    return df

def run_strategy(
    symbol: str,
    strategy_name: str,
    start_date: str,
    end_date: str,
    ratio: str = "4:1",
    wfo_start_year: Optional[int] = None,
    wfo_end_year: Optional[int] = None,
    risk_free_rate: float = 0.02,
    show_progress: bool = True,
    test_start_year: Optional[int] = None,
    test_end_year: Optional[int] = None,
    init_cash: float = 100_000.0,
    fees: float = 0.001,
    slippage: float = 0.002,
    param_ranges: Optional[dict] = None,
    price_col: str = "Open",
    warmup_years: int = 1,
    selection_metric: Union[str, Callable] = "total_return",
    verbose: bool = False,
    n_jobs: Optional[int] = None,
    parallel_backend: str = "loky",
    save_results: bool = True,
    wfo_results_dir: str = "./wfo_results",
    engine_param_grid: Optional[Dict[str, Iterable]] = None,
    summary_df: Optional[pd.DataFrame] = None,  # <<< NEW
    auto_adjust: bool = True,
    run_time: bool = False,
):
    """Esegue WFO + backtest finale.

    Ritorna:
      (portfolio, metrics_df, bh_portfolio, bh_metrics_df, summary_df)

    Note:
    - Se `summary_df` è None: calcola i risultati WFO come in precedenza.
    - Se `summary_df` è fornito: lo usa direttamente e salta la procedura di ottimizzazione WFO.
    """

    bh_portfolio = None
    bh_metrics   = None

    # 1) Carica dati (ATTENIONE USA ROW_DATA in WFO, RUN-TIME e calcolo performance - Totale Price , Adjusted nel calcolo performance - Totale Return)
    df = load_ohlcv(symbol, start=start_date, end=end_date, auto_adjust=auto_adjust)

    # 2) Strategia e griglia (NON sovrascrivere param_ranges dell'utente)
    strategy_func, default_grid = _resolve_strategy(strategy_name)
    grid = param_ranges if param_ranges else default_grid

    if verbose:
        print("[WFO] Uso griglia strategia:", "override (utente)" if param_ranges else "default (strategia)")
        n_combos = 1
        for v in grid.values():
            n_combos *= (len(v) if isinstance(v, (list, tuple, range)) else len(list(v)))
        print(f"[WFO] Chiavi strategia: {list(grid.keys())} | Combinazioni: {n_combos}")

        # Info su engine grid
        if USE_ENGINE_GRID:
            eff_engine_grid = engine_param_grid if engine_param_grid is not None else ENGINE_PARAM_GRID
            if eff_engine_grid is None or len(eff_engine_grid) == 0:
                print("[WFO] USE_ENGINE_GRID=True ma ENGINE_PARAM_GRID è vuota / None → nessuna ottimizzazione engine.")
            else:
                n_engine_combos = 1
                for v in eff_engine_grid.values():
                    n_engine_combos *= (len(v) if isinstance(v, (list, tuple, range)) else len(list(v)))
                print(f"[WFO] Engine grid attiva: chiavi={list(eff_engine_grid.keys())} | "
                      f"combinazioni={n_engine_combos}")
        else:
            print("[WFO] Engine grid disattivata (USE_ENGINE_GRID=False)")

    # 2b) Arricchimento specifico della strategia (se definito)
    df = _apply_strategy_augment(df, strategy_name, strategy_func, start_date, end_date, load_ohlcv, verbose)

    # 3) Ratio (anni)
    train_years, test_years = _parse_ratio(ratio)

    # Header
    if verbose:
        print(f"\nStrategy: {strategy_name} | Ticker: {symbol} | Ratio (train:test): {train_years}:{test_years}")

    # 4) Anni base WFO (solo se devo calcolare WFO internamente)
    if summary_df is None:
        desired_base_start = pd.to_datetime(start_date).year + int(warmup_years)
        desired_base_end   = pd.to_datetime(end_date).year
        base_start, base_end = infer_wfo_base_years(
            df,
            wfo_start_year=wfo_start_year if wfo_start_year is not None else desired_base_start,
            wfo_end_year=wfo_end_year if wfo_end_year is not None else desired_base_end,
        )
    else:
        # Usa direttamente summary_df: ricava base_start/base_end da Year
        if not isinstance(summary_df, pd.DataFrame):
            raise TypeError("summary_df deve essere un pd.DataFrame oppure None.")
        # if "Year" not in summary_df.columns:
        #     raise ValueError("summary_df deve contenere la colonna 'Year'.")

        # years = pd.to_numeric(summary_df["Year"], errors="coerce").dropna().astype(int)
        years = (
            pd.to_numeric(summary_df.index, errors="coerce")
            .dropna()
            .astype(int)
        )
        if years.empty:
            raise ValueError("summary_df['Year'] non contiene anni validi.")

        base_start = int(years.min())
        base_end   = int(years.max())

        if verbose:
            print(f"[WFO] summary_df fornito dall'esterno → base years dedotti: {base_start}..{base_end}")

    # 5) WFO (seq/parallelo in base a n_jobs) - SOLO SE summary_df è None
    if summary_df is None:
        if n_jobs is None:
            if _HAVE_PSUTIL:
                cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
            else:
                cores = os.cpu_count() or 1
            n_jobs_eff = int(cores)
        else:
            n_jobs_eff = int(n_jobs)

        if verbose and _HAVE_JOBLIB:
            print(f"Using: {BOLD}{n_jobs_eff}{RESET} cores")

        summary_df = walk_forward_optimization(
            df=df,
            param_ranges=grid,
            strategy_func=strategy_func,
            train_years=train_years,
            wfo_start_year=base_start,
            wfo_end_year=base_end,
            show_progress=show_progress,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            price_col=price_col,
            selection_metric=selection_metric,
            n_jobs=n_jobs_eff,
            parallel_backend=parallel_backend,
            engine_param_grid=engine_param_grid,
        )
    else:
        # summary_df già disponibile: non calcolare nulla
        n_jobs_eff = None  # solo per chiarezza/diagnostica

    # 5b) WFO wide (Year + colonne parametriche + eventuali engine)
    keys = list(grid.keys())

    def _as_dict(p):
        if isinstance(p, dict):
            return p
        return {k: v for k, v in zip(keys, p)}

    wfo_display = summary_df.copy()

    # Flatten esplicito dei parametri strategia (robusto)
    for k in keys:
        if "Best_Params" in wfo_display.columns:
            wfo_display[k] = wfo_display["Best_Params"].apply(lambda p: _as_dict(p).get(k))

    base_cols = ["Year"] + keys
    eng_cols = [c for c in wfo_display.columns if c.startswith("eng_")]

    # Se alcune colonne non esistono (es. Best_Params mancante), evita KeyError e mostra ciò che c'è
    cols_to_show = [c for c in (base_cols + eng_cols) if c in wfo_display.columns]
    if len(cols_to_show) == 0:
        cols_to_show = list(wfo_display.columns)

    wfo_display = wfo_display[cols_to_show]

    if verbose:
        print("\nWalk-Forward Optimization Results:")
        display(wfo_display)

    # 6) Segnali combinati sul periodo di test
    eff_test_start = test_start_year if test_start_year is not None else (base_start + train_years)
    eff_test_end   = test_end_year if test_end_year is not None else base_end

    if verbose:
        print(
            f"\nPeriodo base: {base_start}..{base_end} (warmup={warmup_years}) → "
            f"Test: {eff_test_start}..{eff_test_end} (train={train_years})"
        )

    combined_entries, combined_exits = apply_wfo_and_build_signals(
        df=df,
        summary_df=summary_df,
        strategy_func=strategy_func,
    )

    # Filtro al periodo di test
    df_test = df[(df.index.year >= eff_test_start) & (df.index.year <= eff_test_end)].copy()
    entries_test = combined_entries.loc[df_test.index]
    exits_test   = combined_exits.loc[df_test.index]

    # 7) Backtest finale
    portfolio = backtest_from_signals(
        data=df_test,
        entries=entries_test,
        exits=exits_test,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        price_col=price_col,
    )
    
    if run_time: return portfolio, None, None, None, None

    # luca
    # 8) Metriche strategia
    stats = compute_portfolio_stats_with_adjusted(
        portfolio=portfolio,
        data=df_test,
        entries=entries_test,
        exits=exits_test,
        risk_free_rate=risk_free_rate,
    )
    metrics_df = pd.DataFrame(stats)
    metrics_df.columns = ['Value']

    if verbose:
        print(
            f"\n📈 Final Portfolio Stats ([{eff_test_start}-{eff_test_end}]) "
            f"stock={symbol} strategy={strategy_name} ratio={train_years}:{test_years}:"
        )
        display(metrics_df)

    # 9) Buy & Hold (benchmark)
    bh_portfolio = vbt.Portfolio.from_holding(df_test[price_col], init_cash=init_cash, freq="1D")
    bh_metrics_df = pd.DataFrame(bh_portfolio.stats())
    if verbose:
        print("\nBuy & Hold (Holding) Metrics:")
        display(bh_metrics_df)

        # 10) Grafico equity
        fig = portfolio.plot(
            width=vbt_plot_width,
            subplots=[
                'cum_returns',
                'orders',
                'trade_pnl',
                'drawdowns',
                'underwater',
                'gross_exposure',
                'trades'
            ]
        )
        fig.show()

    # --- salvataggi condizionali ---
    if save_results:
        save_trading_wfo_summary(
            strategy=strategy_name,
            symbol=symbol,
            ratio=ratio,
            grid=grid,
            portfolio=portfolio,
            summary_df=summary_df,
            wfo_results_dir=wfo_results_dir,
            verbose=verbose,
            bh_portfolio=bh_portfolio,
            bh_metrics=bh_metrics
        )

    return portfolio, metrics_df, bh_portfolio, bh_metrics_df, summary_df


    
###############################################################################
# Performance: funzioni per il calcolo della performance
###############################################################################


def sample_param_combinations(
    param_ranges: dict,
    n_samples: int = 1200,
    method: str = "random",
    seed: int | None = 42
):
    """
    Campiona combinazioni di parametri da param_ranges (che contiene oggetti range/list/tuple).
    Output: lista di tuple (stesso ordine delle chiavi) e lista param_keys.

    Metodi:
      - "random": campionamento uniforme su ciascun parametro (con replacement)
      - "lhs": Latin Hypercube Sampling (discretizzato) per copertura migliore (senza dipendenze esterne)
    """
    import numpy as np

    if not isinstance(param_ranges, dict) or len(param_ranges) == 0:
        return [], []

    param_keys = list(param_ranges.keys())
    value_lists = [list(param_ranges[k]) for k in param_keys]

    # se qualche lista è vuota, non possiamo campionare
    if any(len(v) == 0 for v in value_lists):
        return [], param_keys

    rng = np.random.default_rng(seed)

    method = (method or "random").lower()
    if method not in ("random", "lhs"):
        method = "random"

    n_params = len(param_keys)

    if method == "random":
        samples = []
        for _ in range(int(n_samples)):
            combo = tuple(value_lists[j][rng.integers(0, len(value_lists[j]))] for j in range(n_params))
            samples.append(combo)
        return samples, param_keys

    # --- "lhs" (Latin Hypercube Sampling discretizzato) ---
    # Costruiamo n_samples intervalli per ogni dimensione e ne permutiamo l'ordine.
    # Poi mappiamo ciascun punto nell'intervallo al relativo indice discreto nella lista valori.
    n = int(n_samples)
    # permutazioni indipendenti per dimensione
    perms = [rng.permutation(n) for _ in range(n_params)]

    samples = []
    for i in range(n):
        combo = []
        for j in range(n_params):
            # posizione LHS in [0,1): (perm[i] + u) / n
            u = rng.random()
            x = (perms[j][i] + u) / n
            idx = int(np.floor(x * len(value_lists[j])))
            if idx >= len(value_lists[j]):
                idx = len(value_lists[j]) - 1
            combo.append(value_lists[j][idx])
        samples.append(tuple(combo))

    return samples, param_keys

def overfitting_optimization(
    symbol,
    strategy,
    start_date,
    end_date: str = None,
    param_ranges: dict | None = None,
    price_col: str = "Open",
    freq: str = "1D",
    init_cash: float = 100_000.0,
    fees: float = 0.001,
    slippage: float = 0.002,
    plot_results: bool = True,
    # --- smart plotting per N parametri ---
    smart_plot: bool = True,
    smart_top_k_params: int = 4,
    smart_max_pairs: int = 6,
    smart_aggfunc: str = "median",        # "median" | "mean"
    smart_ratio_thr: float = 0.80,        # plateau threshold: vicini >= thr * best
    smart_local_fix: bool = True,
    smart_global_fallback: bool = True,
    top_n_table: int = 10,
    debug: bool = False,
    # --- NEW: analisi risultati + raccomandazione WFO ---
    analyze_results: bool = True,
    # policy thresholds (default ragionevoli, modificabili)
    wfo_min_top_cluster_pct: float = 0.05,
    wfo_top_cluster_ratio: float = 0.80,
    wfo_min_plateau_neighbors: int = 8,
    wfo_min_plateau_above_pct: float = 0.20,
    wfo_max_outlier_robust_z: float = 6.0,
    return_analysis: bool = False,
    # ------------------------------------------------------------------
    # NEW: integrazione approccio "plot all curves + media vs B&H"
    # ------------------------------------------------------------------
    equity_sweep: bool = True,                 # abilita il calcolo delle curve equity per ogni combo
    equity_stat: str = "mean",                 # "mean" | "median" (curva aggregata principale)
    equity_quantiles: tuple[float, float] = (0.10, 0.90),  # bande quantili
    equity_plot_all: bool = True,              # plotta (campionate) le curve di tutte le combo
    equity_plot_max_curves: int = 250,         # massimo curve da plottare (per non distruggere il browser)
    equity_plot_alpha: float = 0.08,              # trasparenza curve (plotly line opacity)
    equity_plot_benchmark: bool = True,           # include B&H nel plot aggregato
    equity_beat_bh_metric: str = "Total Return",  # metrica per "beat B&H": "Total Return" (default)
    wfo_use_equity_sweep_gate: bool = True,
    wfo_min_beat_bh_pct: float = 0.55,
    #
    # --- NEW: grid control ---
    #
    grid_mode: str = "core",        # "core" | "stress"
    stress_n_samples: int = 1200,
    stress_method: str = "lhs",     # "random" | "lhs"
    stress_seed: int | None = 42,
):
    """
    Overfitting optimization (IS) + diagnosi robustezza (cluster/plateau/outlier)
    + sweep di equity curves (plot tutte le curve, curva media/mediana, bande quantili, confronto B&H).
    """

    # -----------------------------
    # Defaults / imports
    # -----------------------------
    if end_date is None:
        end_date = now().strftime("%Y-%m-%d")

    if param_ranges is None:
        param_ranges = {}

    import itertools
    import numpy as np
    import pandas as pd

    # Fix FutureWarning downcasting (persistente nel progetto)
    try:
        pd.set_option("future.no_silent_downcasting", True)
    except Exception:
        pass

    from tqdm.auto import tqdm

    import plotly.express as px
    import plotly.graph_objects as go

    # -----------------------------
    # Helpers
    # -----------------------------
    def _rank_params_by_spearman(df: pd.DataFrame, keys: list, metric_col: str):
        ranks = []
        y = df[metric_col]
        y_rank = y.rank(method="average")
        for k in keys:
            rho = df[k].rank(method="average").corr(y_rank)
            ranks.append((k, float(abs(rho)) if pd.notna(rho) else 0.0))
        return sorted(ranks, key=lambda x: x[1], reverse=True)

    def _plot_pair_heatmaps(
        df: pd.DataFrame,
        keys: list,
        best: dict,
        metric_col: str,
        top_k_params: int,
        max_pairs: int,
        aggfunc: str,
        local_fix: bool,
        global_fallback: bool
    ):
        ranked = _rank_params_by_spearman(df, keys, metric_col=metric_col)
        top_params = [k for k, _ in ranked[:max(2, top_k_params)]]
        pairs = list(itertools.combinations(top_params, 2))[:max_pairs]

        print("\n[SMART] Parametri più informativi (|Spearman rho|):")
        for k, s in ranked[:min(len(ranked), top_k_params)]:
            print(f"  - {k}: {s:.3f}")

        for a, b in pairs:
            df_view = df

            if local_fix:
                mask = pd.Series(True, index=df.index)
                for k in keys:
                    if k not in (a, b):
                        mask &= (df[k] == best[k])
                df_loc = df[mask].copy()

                if df_loc.empty and global_fallback:
                    df_view = df.copy()
                else:
                    df_view = df_loc

            agg = "median" if aggfunc.lower() != "mean" else "mean"
            pv = df_view.pivot_table(index=a, columns=b, values=metric_col, aggfunc=agg)

            fig = px.imshow(
                pv,
                labels=dict(x=b, y=a, color=metric_col),
                aspect="equal",
                color_continuous_scale="Viridis",
                title=f"[SMART] Heatmap {metric_col} — {a} vs {b}"
                      + (" (local fix to best)" if local_fix else " (global)")
            )

            fig.add_trace(
                go.Scatter(
                    x=[best[b]],
                    y=[best[a]],
                    mode="markers",
                    marker=dict(color="red", size=14, symbol="circle-open-dot",
                                line=dict(width=2, color="white")),
                    name="Best"
                )
            )

            fig.update_layout(width=850, height=700)
            fig.show()

    def _plateau_score(df: pd.DataFrame, keys: list, best: dict, ranges: dict,
                       metric_col: str, ratio_thr: float):
        value_lists = {k: list(ranges[k]) for k in keys}

        idx_map = {}
        for k in keys:
            vals = value_lists[k]
            try:
                idx_map[k] = vals.index(best[k])
            except ValueError:
                idx_map[k] = None

        neighbors = []
        for deltas in itertools.product([-1, 0, 1], repeat=len(keys)):
            if all(d == 0 for d in deltas):
                continue
            cand = {}
            ok = True
            for i, k in enumerate(keys):
                base_idx = idx_map[k]
                if base_idx is None:
                    ok = False
                    break
                new_idx = base_idx + deltas[i]
                if new_idx < 0 or new_idx >= len(value_lists[k]):
                    ok = False
                    break
                cand[k] = value_lists[k][new_idx]
            if ok:
                neighbors.append(cand)

        seen = set()
        unique = []
        for d in neighbors:
            t = tuple(d[k] for k in keys)
            if t not in seen:
                seen.add(t)
                unique.append(d)

        m_best = (df[keys] == pd.Series(best)).all(axis=1)
        if not m_best.any():
            return {"neighbors": 0, "note": "Best non trovato in results_df."}

        best_val = float(df.loc[m_best, metric_col].iloc[0])

        neigh_vals = []
        for d in unique:
            m = (df[keys] == pd.Series(d)).all(axis=1)
            if m.any():
                neigh_vals.append(float(df.loc[m, metric_col].iloc[0]))

        if len(neigh_vals) == 0:
            return {"neighbors": 0, "best": best_val, "note": "Nessun vicino trovato."}

        neigh_vals = np.array(neigh_vals, dtype=float)
        ratio = neigh_vals / best_val

        return {
            "neighbors": int(len(neigh_vals)),
            "best": best_val,
            "neighbors_median": float(np.median(neigh_vals)),
            "neighbors_iqr": float(np.percentile(neigh_vals, 75) - np.percentile(neigh_vals, 25)),
            "neighbors_above_thr_pct": float((ratio >= ratio_thr).mean()),
            "ratio_thr": float(ratio_thr)
        }

    def _robust_stats(x: np.ndarray):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4826 * mad if mad > 0 else np.nan
        return med, mad, scale

    def _analyze_results(
        results_df: "pd.DataFrame",
        param_keys: list,
        best_params: dict,
        param_ranges: dict,
        metric_col: str = "Total Return",
        equity_report: dict | None = None
    ):
        """
        Produce un report + raccomandazione WFO.
        Integra opzionalmente anche un segnale equity sweep vs B&H.
        """
        import numpy as np
        import pandas as pd
    
        x = results_df[metric_col].astype(float).values
        best_val = float(results_df.iloc[0][metric_col])  # results_df è già ordinato desc
        n_total = int(len(results_df))
    
        # --- Distribuzione globale / percentili ---
        p95 = float(np.percentile(x, 95))
        p99 = float(np.percentile(x, 99))
        p999 = float(np.percentile(x, 99.9)) if n_total >= 1000 else float(np.percentile(x, 99))
    
        med, mad, scale = _robust_stats(x)
        if np.isfinite(scale) and scale > 0:
            robust_z = float((best_val - med) / scale)
        else:
            robust_z = np.nan
    
        # --- Cluster di quasi-best (densità soluzioni alte) ---
        cluster_thr = wfo_top_cluster_ratio * best_val
        cluster_count = int((x >= cluster_thr).sum())
        cluster_pct = float(cluster_count / n_total) if n_total > 0 else 0.0
    
        # --- Plateau locale (vicini) ---
        plateau = _plateau_score(
            df=results_df,
            keys=param_keys,
            best=best_params,
            ranges=param_ranges,
            metric_col=metric_col,
            ratio_thr=smart_ratio_thr
        )
    
        # --- Heuristica decisionale base (3 segnali) ---
        cond_cluster = cluster_pct >= wfo_min_top_cluster_pct
    
        neigh = int(plateau.get("neighbors", 0) or 0)
        above_pct = float(plateau.get("neighbors_above_thr_pct", 0.0) or 0.0)
        cond_plateau = (neigh >= wfo_min_plateau_neighbors) and (above_pct >= wfo_min_plateau_above_pct)
    
        cond_outlier_ok = True
        if np.isfinite(robust_z):
            cond_outlier_ok = (robust_z <= wfo_max_outlier_robust_z)
    
        # --- NEW: Equity sweep gate (opzionale) come 4° segnale ---
        cond_equity_ok = None
        equity_reason = None
        if wfo_use_equity_sweep_gate and equity_report is not None:
            beat_pct = equity_report.get("beat_bh_pct_by_total_return", None)
            bh_tr = equity_report.get("bh_total_return", None)
    
            if isinstance(beat_pct, (int, float)) and np.isfinite(beat_pct):
                cond_equity_ok = (float(beat_pct) >= float(wfo_min_beat_bh_pct))
                equity_reason = (
                    f"Equity sweep {'OK' if cond_equity_ok else 'DEBOLE'}: "
                    f"{float(beat_pct):.2%} combinazioni battono B&H"
                    + (f" (B&H TR={float(bh_tr):.2%})"
                       if isinstance(bh_tr, (int, float)) and np.isfinite(bh_tr) else "")
                    + f" | soglia={float(wfo_min_beat_bh_pct):.0%}"
                )
            else:
                cond_equity_ok = False
                equity_reason = "Equity sweep: impossibile valutare (beat_bh_pct non disponibile)"
    
        # --- Decisione finale ---
        signals = [cond_cluster, cond_plateau, cond_outlier_ok]
        if wfo_use_equity_sweep_gate and equity_report is not None:
            signals.append(bool(cond_equity_ok))
    
        positives = sum(bool(s) for s in signals)
    
        # criterio: >=2/3 (originale) oppure >=3/4 se equity aggiunto
        required = 3 if (wfo_use_equity_sweep_gate and equity_report is not None) else 2
        recommend_wfo = positives >= required
    
        reasons = []
        reasons.append(
            f"Cluster globale {'OK' if cond_cluster else 'DEBOLE'}: {cluster_pct:.2%} combinazioni >= {wfo_top_cluster_ratio:.0%} del best"
        )
    
        if neigh < wfo_min_plateau_neighbors:
            reasons.append(f"Plateau locale non valutabile bene: solo {neigh} vicini disponibili (min {wfo_min_plateau_neighbors})")
        else:
            reasons.append(
                f"Plateau locale {'OK' if cond_plateau else 'DEBOLE'}: {above_pct:.2%} vicini >= {smart_ratio_thr:.0%} del best"
            )
    
        if np.isfinite(robust_z):
            reasons.append(
                f"Outlier check {'OK' if cond_outlier_ok else 'NEGATIVO'}: robust_z={robust_z:.2f} "
                f"({'<=' if cond_outlier_ok else '>'} {wfo_max_outlier_robust_z})"
            )
        else:
            reasons.append("Outlier check: robust_z non calcolabile (MAD≈0), ignoro questa regola")
    
        if equity_reason is not None:
            reasons.append(equity_reason)
    
        report = {
            "best_total_return": best_val,
            "n_combinations": n_total,
            "percentiles": {"p95": p95, "p99": p99, "p99_9": p999},
            "robust_stats": {
                "median": float(med),
                "mad": float(mad),
                "scale": float(scale) if np.isfinite(scale) else None,
                "robust_z_best": robust_z
            },
            "top_cluster": {
                "ratio": wfo_top_cluster_ratio,
                "threshold": cluster_thr,
                "count": cluster_count,
                "pct": cluster_pct
            },
            "plateau": plateau,
            "equity_sweep": equity_report,
            "decision": {
                "recommend_wfo": bool(recommend_wfo),
                "signals_positive": int(positives),
                "signals_total": int(len(signals)),
                "required_positive": int(required),
                "rules": {
                    "cluster_ok": bool(cond_cluster),
                    "plateau_ok": bool(cond_plateau),
                    "outlier_ok": bool(cond_outlier_ok),
                    "equity_ok": (None if cond_equity_ok is None else bool(cond_equity_ok))
                },
                "reasons": reasons
            }
        }
        return report

    def _print_analysis(report: dict):
        import numpy as np
    
        dec = report["decision"]
        print("\n" + "=" * 80)
        print("[ANALISI OVERFITTING] Valutazione robustezza e raccomandazione WFO")
        print("=" * 80)
        print(f"Best Total Return: {report['best_total_return']:.2%}")
        print(f"Combinazioni testate: {report['n_combinations']:,}")
    
        p = report["percentiles"]
        print(f"Percentili Total Return: P95={p['p95']:.2%} | P99={p['p99']:.2%} | P99.9≈{p['p99_9']:.2%}")
    
        ts = report["top_cluster"]
        print(f"Top-cluster (>= {ts['ratio']:.0%} del best): {ts['count']:,} comb. ({ts['pct']:.2%})")
    
        pl = report["plateau"]
        if "note" in pl:
            print(f"Plateau locale: {pl.get('note')}")
        else:
            print(
                f"Plateau locale: neighbors={pl['neighbors']} | >=thr({pl['ratio_thr']:.0%})={pl['neighbors_above_thr_pct']:.2%} "
                f"| median_vicini={pl['neighbors_median']:.2%} | IQR_vicini={pl['neighbors_iqr']:.2%}"
            )
    
        rs = report["robust_stats"]
        rz = rs.get("robust_z_best", None)
        if rz is not None and np.isfinite(rz):
            print(f"Outlier robust_z(best): {rz:.2f}")
    
        # (opzionale) stampa info equity sweep in modo compatto
        eq = report.get("equity_sweep", None)
        if eq is not None:
            beat = eq.get("beat_bh_pct_by_total_return", None)
            bhtr = eq.get("bh_total_return", None)
            if isinstance(beat, (int, float)) and np.isfinite(beat):
                s = f"Equity sweep: beat_BH={float(beat):.2%}"
                if isinstance(bhtr, (int, float)) and np.isfinite(bhtr):
                    s += f" | BH_TR={float(bhtr):.2%}"
                print(s)
    
        print("-" * 80)
        print("Decisione WFO:", "SI" if dec["recommend_wfo"] else "NO")
        print(
            f"Segnali positivi: {dec['signals_positive']}/{dec.get('signals_total', 3)} "
            f"(richiesti >= {dec.get('required_positive', 2)})"
        )
        print("Motivazioni:")
        for r in dec["reasons"]:
            print(f"  - {r}")
        print("=" * 80 + "\n")

    def fingerprint_run(df, entries, exits, price_col="Open", tag=""):
        df = df.copy().sort_index()
        df = df[~df.index.duplicated(keep="first")]

        price = df[price_col]
        price_hash = int(pd.util.hash_pandas_object(price, index=True).sum())
        ent_hash   = int(pd.util.hash_pandas_object(entries.astype("int8"), index=True).sum())
        ex_hash    = int(pd.util.hash_pandas_object(exits.astype("int8"), index=True).sum())

        print(
            f"[FP {tag}] rows={len(df)} "
            f"start={df.index[0].date()} end={df.index[-1].date()} "
            f"price_hash={price_hash} entries_hash={ent_hash} exits_hash={ex_hash} "
            f"entries_sum={int(entries.sum())} exits_sum={int(exits.sum())}"
        )

    def _safe_series(x, index=None, name=None):
        """
        Rende x una pd.Series robusta e allineata a index (se fornito).
        """
        if isinstance(x, pd.Series):
            s = x.copy()
            if index is not None:
                s = s.reindex(index)
            if name is not None:
                s.name = name
            return s
        if isinstance(x, (pd.DataFrame,)):
            if x.shape[1] == 1:
                s = x.iloc[:, 0].copy()
                if index is not None:
                    s = s.reindex(index)
                if name is not None:
                    s.name = name
                return s
            raise ValueError("Impossibile convertire DataFrame multi-colonna in Series (safe).")
        arr = np.asarray(x)
        if index is None:
            s = pd.Series(arr)
        else:
            s = pd.Series(arr, index=index)
        if name is not None:
            s.name = name
        return s

    def _get_portfolio_value_series(portfolio, index):
        """
        Estrae una value series dal portfolio (vectorbt o wrapper).
        Fallback: se non disponibile, ritorna None.
        """
        v = None
        # vectorbt: .value() spesso è Series
        if hasattr(portfolio, "value") and callable(getattr(portfolio, "value")):
            try:
                v = portfolio.value()
            except Exception:
                v = None
        # alcuni wrapper espongono .value (property)
        if v is None and hasattr(portfolio, "value") and not callable(getattr(portfolio, "value")):
            try:
                v = getattr(portfolio, "value")
            except Exception:
                v = None
        if v is None:
            return None
        try:
            v = _safe_series(v, index=index, name="value")
            v = v.infer_objects(copy=False)
            return v
        except Exception:
            return None

    def _build_buy_hold_value(price: pd.Series, init_cash: float) -> pd.Series:
        p = price.astype(float).copy()
        p = p.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        p = p.infer_objects(copy=False)
        base = float(p.iloc[0]) if pd.notna(p.iloc[0]) else np.nan
        if not np.isfinite(base) or base == 0:
            return pd.Series(index=p.index, dtype=float, name="BH")
        return (init_cash * (p / base)).rename("BH")

    def _plot_equity_sweep_bundle(
        df_value_mat: pd.DataFrame,
        bh_value: pd.Series | None,
        stat: str,
        qlo: float,
        qhi: float,
        plot_all: bool,
        max_curves: int,
        alpha: float,
        title: str
    ):
        mat = df_value_mat.copy()
        if mat.empty:
            print("[EQUITY SWEEP] Nessuna equity curve disponibile per plotting.")
            return

        # curva aggregata
        stat = (stat or "mean").lower()
        if stat not in ("mean", "median"):
            stat = "mean"

        if stat == "median":
            agg = mat.median(axis=1)
            agg_name = "Median(Params)"
        else:
            agg = mat.mean(axis=1)
            agg_name = "Mean(Params)"

        qlow = mat.quantile(qlo, axis=1)
        qhigh = mat.quantile(qhi, axis=1)

        fig = go.Figure()

        # plot all (campionate)
        if plot_all:
            n = mat.shape[1]
            cols = list(mat.columns)
            if n > max_curves:
                # campionamento deterministico (stabile): prendi equispaziati
                idxs = np.linspace(0, n - 1, max_curves).round().astype(int)
                cols = [cols[i] for i in idxs]

            for c in cols:
                fig.add_trace(go.Scatter(
                    x=mat.index,
                    y=mat[c],
                    mode="lines",
                    line=dict(width=1),
                    opacity=float(alpha),
                    name=str(c),
                    showlegend=False
                ))

        # bande quantili
        fig.add_trace(go.Scatter(
            x=mat.index,
            y=qlow,
            mode="lines",
            line=dict(width=0),
            name=f"Q{int(qlo*100)}",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=mat.index,
            y=qhigh,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name=f"Q{int(qhi*100)} band",
            opacity=0.25,
            showlegend=True
        ))

        # curva aggregata
        fig.add_trace(go.Scatter(
            x=mat.index,
            y=agg,
            mode="lines",
            line=dict(width=3),
            name=agg_name
        ))

        # benchmark
        if bh_value is not None:
            fig.add_trace(go.Scatter(
                x=bh_value.index,
                y=bh_value.values,
                mode="lines",
                line=dict(width=3, dash="dash"),
                name="Buy & Hold"
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Equity (Value)",
            width=1100,
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.show()
        
    def _compute_top_mask(results_df: pd.DataFrame, metric_col: str, best_val: float, ratio: float):
        thr = ratio * best_val
        return (results_df[metric_col].astype(float) >= thr), thr
    
    def _plot_plateau_barplots(results_df: pd.DataFrame, param_keys: list, top_mask: pd.Series, title_prefix: str):
        import plotly.express as px
        df = results_df.copy()
        df["_is_top"] = top_mask.astype(int)
    
        for k in param_keys:
            hookup = df.groupby(k)["_is_top"].mean().reset_index()
            fig = px.bar(hookup, x=k, y="_is_top", title=f"{title_prefix} | % top per {k}")
            fig.update_yaxes(title="% top (>=thr)")
            fig.show()
    
    def _pairwise_top_density(results_df: pd.DataFrame, param_keys: list, top_mask: pd.Series,
                              max_pairs: int = 6, title_prefix: str = ""):
        import itertools
        import numpy as np
        import plotly.express as px
    
        df = results_df.copy()
        df["_is_top"] = top_mask.astype(int)
    
        pairs = list(itertools.combinations(param_keys, 2))[:max_pairs]
        for a, b in pairs:
            pv = df.pivot_table(index=a, columns=b, values="_is_top", aggfunc="mean")  # % top
            fig = px.imshow(
                pv,
                labels=dict(x=b, y=a, color="% top"),
                aspect="equal",
                title=f"{title_prefix} | Pairwise % top: {a} vs {b}"
            )
            fig.show()
    
    def _neighbors_for_combo(combo: dict, param_keys: list, param_ranges: dict):
        # genera vicini L∞ con passi di 1 (come _plateau_score)
        import itertools
        value_lists = {k: list(param_ranges[k]) for k in param_keys}
        idx_map = {}
        for k in param_keys:
            vals = value_lists[k]
            idx_map[k] = vals.index(combo[k]) if combo[k] in vals else None
    
        neighbors = []
        for deltas in itertools.product([-1, 0, 1], repeat=len(param_keys)):
            if all(d == 0 for d in deltas):
                continue
            cand = {}
            ok = True
            for i, k in enumerate(param_keys):
                base_idx = idx_map[k]
                if base_idx is None:
                    ok = False; break
                new_idx = base_idx + deltas[i]
                if new_idx < 0 or new_idx >= len(value_lists[k]):
                    ok = False; break
                cand[k] = value_lists[k][new_idx]
            if ok:
                neighbors.append(cand)
        # unique
        seen = set()
        uniq = []
        for d in neighbors:
            t = tuple(d[k] for k in param_keys)
            if t not in seen:
                seen.add(t)
                uniq.append(d)
        return uniq
    
    def _compute_local_stability(results_df: pd.DataFrame, param_keys: list, param_ranges: dict,
                                metric_col: str, best_val: float, ratio_thr: float):
        import numpy as np
        import pandas as pd
    
        df = results_df.copy()
        # indice lookup veloce: tuple(param_values) -> metric
        key_tuples = [tuple(row) for row in df[param_keys].values]
        metric_map = dict(zip(key_tuples, df[metric_col].astype(float).values))
    
        thr = ratio_thr * best_val
        stabilities = []
        neigh_counts = []
    
        for row in df[param_keys].to_dict("records"):
            neigh = _neighbors_for_combo(row, param_keys, param_ranges)
            vals = []
            for n in neigh:
                t = tuple(n[k] for k in param_keys)
                if t in metric_map:
                    vals.append(metric_map[t])
            if len(vals) == 0:
                stabilities.append(np.nan)
                neigh_counts.append(0)
            else:
                vals = np.asarray(vals, dtype=float)
                stabilities.append(float((vals >= thr).mean()))
                neigh_counts.append(int(len(vals)))
    
        df["_local_stability"] = pd.Series(stabilities, index=df.index)
        df["_n_neighbors"] = pd.Series(neigh_counts, index=df.index)
        return df, thr
    
    def _pairwise_stability_map(df_stab: pd.DataFrame, param_keys: list, max_pairs: int = 6, title_prefix: str = ""):
        import itertools
        import plotly.express as px
    
        pairs = list(itertools.combinations(param_keys, 2))[:max_pairs]
        for a, b in pairs:
            pv = df_stab.pivot_table(index=a, columns=b, values="_local_stability", aggfunc="mean")
            fig = px.imshow(
                pv,
                labels=dict(x=b, y=a, color="mean local stability"),
                aspect="equal",
                title=f"{title_prefix} | Pairwise local stability: {a} vs {b}"
            )
            fig.show()
    
    def _edge_bias_report(results_df: pd.DataFrame, param_keys: list, top_mask: pd.Series, param_ranges: dict):
        import pandas as pd
        out = []
        df = results_df.copy()
        df["_is_top"] = top_mask
    
        df_top = df[df["_is_top"]].copy()
        if df_top.empty:
            return pd.DataFrame(columns=["param", "edge_pct_top"])
    
        for k in param_keys:
            vals = list(param_ranges[k])
            if len(vals) < 2:
                out.append((k, 1.0))  # parametro fisso -> sempre edge
                continue
            vmin, vmax = vals[0], vals[-1]
            edge_pct = float(((df_top[k] == vmin) | (df_top[k] == vmax)).mean())
            out.append((k, edge_pct))
    
        return pd.DataFrame(out, columns=["param", "edge_pct_top"])

    # -----------------------------
    # Start: funzione principale
    # -----------------------------
    info_string = f"stock={symbol} strategy={strategy}"
    print(f"🔎 Running Overfitting optimization ([{start_date}-{end_date}] {info_string})")

    # We use raw data !
    df_daily = load_ohlcv(symbol, start_date, end_date, auto_adjust=False)
    df_daily = df_daily.sort_index()
    df_daily = df_daily[~df_daily.index.duplicated(keep="first")]

    try:
        strategy_fn = globals()[f"strategy_{strategy}"]
    except Exception:
        raise Exception(f"Funzione strategy_{strategy} non trovata")

    if not param_ranges:
        try:
            param_ranges = globals()[f"strategy_{strategy}_param_ranges"]
        except Exception:
            print(f"No default parameters strategy_{strategy}_param_ranges defined")
            param_ranges = {}
        else:
            print(f"Applico i parametri dei default: strategy_{strategy}_param_ranges")

    param_keys = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    # all_combinations = list(itertools.product(*param_values))
    grid_mode = (grid_mode or "core").lower()
    if grid_mode not in ("core", "stress"):
        raise ValueError("grid_mode must be 'core' or 'stress'")
    
    if grid_mode == "core":
        all_combinations = list(itertools.product(*param_values))
    else:
        all_combinations, _ = sample_param_combinations(
            param_ranges=param_ranges,
            n_samples=int(stress_n_samples),
            method=str(stress_method),
            seed=stress_seed,
        )
        
    results = []

    best_metric = -np.inf
    best_params = None
    portfolio_best = None

    # --- Equity sweep storage (curve value per combo)
    # user story: plot all equity curves, compute mean/median, compare vs B&H
    value_series_list = []
    value_colnames = []

    # --- Benchmark B&H (value curve)
    bh_value = None
    if equity_sweep and equity_plot_benchmark:
        if price_col not in df_daily.columns:
            raise KeyError(f"price_col='{price_col}' non presente in df_daily.columns={list(df_daily.columns)}")
        bh_value = _build_buy_hold_value(df_daily[price_col], init_cash=init_cash)

    # -----------------------------
    # Loop combinazioni
    # -----------------------------
    for combo in tqdm(all_combinations, desc="Overfitting Optimization"):
        params_dict = dict(zip(param_keys, combo))
        entries, exits = strategy_fn(df_daily, params_dict)

        portfolio = backtest_from_signals(
            data=df_daily,
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            price_col=price_col,
            freq=freq,
            debug=debug
        )

        performance = float(portfolio.total_return())
        results.append(performance)

        # --- Equity curve capture
        if equity_sweep:
            v = _get_portfolio_value_series(portfolio, index=df_daily.index)
            if v is not None:
                v = v.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                v = v.infer_objects(copy=False)
                value_series_list.append(v)
                value_colnames.append(str(params_dict))

        if performance > best_metric:
            best_metric = performance
            best_params = params_dict
            portfolio_best = portfolio

    print(f"\nMigliori parametri:\n{best_params} -> Total Return: {best_metric:.2%}\n")

    df_all = pd.DataFrame(all_combinations, columns=param_keys)
    df_all["Total Return"] = results
    results_df = df_all.sort_values("Total Return", ascending=False).reset_index(drop=True)

    if debug and best_params is not None:
        try:
            # attenzione: qui entries/exits sono quelli dell'ultima combo nel loop;
            # se vuoi fingerprint del best, ricalcola: (non obbligatorio, lo facciamo se debug)
            entries_best, exits_best = strategy_fn(df_daily, best_params)
            fingerprint_run(df_daily, entries_best, exits_best, price_col=price_col, tag=str(best_params))
        except Exception:
            pass

    # -----------------------------
    # NEW: equity sweep report + plots
    # -----------------------------
    equity_report = None
    if equity_sweep and len(value_series_list) > 0:
        # matrix: index=date, columns=combo
        df_value_mat = pd.concat(value_series_list, axis=1)
        df_value_mat.columns = value_colnames
        df_value_mat = df_value_mat.reindex(df_daily.index)
        df_value_mat = df_value_mat.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        df_value_mat = df_value_mat.infer_objects(copy=False)

        stat = (equity_stat or "mean").lower()
        if stat not in ("mean", "median"):
            stat = "mean"

        qlo, qhi = equity_quantiles if equity_quantiles else (0.10, 0.90)
        qlo = float(qlo)
        qhi = float(qhi)
        qlo = max(0.0, min(0.5, qlo))
        qhi = max(0.5, min(1.0, qhi))

        if stat == "median":
            agg_curve = df_value_mat.median(axis=1)
        else:
            agg_curve = df_value_mat.mean(axis=1)

        qlow = df_value_mat.quantile(qlo, axis=1)
        qhigh = df_value_mat.quantile(qhi, axis=1)

        # beat B&H: percentuale di combinazioni che battono B&H su metrica scelta
        beat_bh_pct = None
        bh_total_return = None
        if bh_value is not None and equity_beat_bh_metric.lower() == "total return":
            bh_total_return = float((bh_value.iloc[-1] / bh_value.iloc[0]) - 1.0) if bh_value.iloc[0] != 0 else np.nan
            beat_bh_pct = float((results_df["Total Return"].astype(float).values > bh_total_return).mean())

        equity_report = {
            "n_equity_curves": int(df_value_mat.shape[1]),
            "equity_stat": stat,
            "quantiles": {"low": qlo, "high": qhi},
            "bh_total_return": bh_total_return,
            "beat_bh_pct_by_total_return": beat_bh_pct
        }

        if plot_results:
            title = f"[EQUITY SWEEP] {symbol} - {strategy} | All curves + {stat} + quantiles vs B&H"
            _plot_equity_sweep_bundle(
                df_value_mat=df_value_mat,
                bh_value=bh_value if equity_plot_benchmark else None,
                stat=stat,
                qlo=qlo,
                qhi=qhi,
                plot_all=bool(equity_plot_all),
                max_curves=int(equity_plot_max_curves),
                alpha=float(equity_plot_alpha),
                title=title
            )

            # (opzionale) plot aggregato “equity stat” vs B&H, senza tutte le curve
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=agg_curve.index, y=agg_curve.values, mode="lines", name=f"{stat.title()}(Params)"))
            fig2.add_trace(go.Scatter(x=qlow.index, y=qlow.values, mode="lines", name=f"Q{int(qlo*100)}", line=dict(width=1)))
            fig2.add_trace(go.Scatter(x=qhigh.index, y=qhigh.values, mode="lines", name=f"Q{int(qhi*100)}", line=dict(width=1)))
            if bh_value is not None and equity_plot_benchmark:
                fig2.add_trace(go.Scatter(x=bh_value.index, y=bh_value.values, mode="lines", name="Buy & Hold", line=dict(dash="dash")))
            fig2.update_layout(
                title=f"[EQUITY AGG] {symbol} - {strategy} | {stat} + quantiles vs B&H",
                xaxis_title="Date",
                yaxis_title="Equity (Value)",
                width=1100,
                height=520
            )
            fig2.show()

            if beat_bh_pct is not None and bh_total_return is not None:
                print(f"[EQUITY SWEEP] Buy&Hold Total Return: {bh_total_return:.2%}")
                print(f"[EQUITY SWEEP] % combinazioni che battono B&H (Total Return): {beat_bh_pct:.2%}")

    elif equity_sweep:
        print("[EQUITY SWEEP] Abilitato, ma non sono riuscito a estrarre equity curves dai portfolio.")
    
    # -----------------------------
    # NEW: WFO-coherent diagnostics plots
    # -----------------------------
    if plot_results and analyze_results and len(results_df) > 0:
    
        metric_col = "Total Return"
        best_val = float(results_df.iloc[0][metric_col])
    
        # 1) Plateau globale (top cluster) -> binary
        top_mask, top_thr = _compute_top_mask(
            results_df=results_df,
            metric_col=metric_col,
            best_val=best_val,
            ratio=wfo_top_cluster_ratio
        )
    
        print(f"[WFO-PLOT] Top-cluster threshold = {top_thr:.2%} (>= {wfo_top_cluster_ratio:.0%} del best)")
    
        _plot_plateau_barplots(
            results_df=results_df,
            param_keys=param_keys,
            top_mask=top_mask,
            title_prefix=f"[WFO] {symbol}-{strategy}"
        )
    
        _pairwise_top_density(
            results_df=results_df,
            param_keys=param_keys,
            top_mask=top_mask,
            max_pairs=smart_max_pairs,
            title_prefix=f"[WFO] {symbol}-{strategy}"
        )
    
        # 2) Stabilità locale (vicinato) -> coerente con _plateau_score ma esteso a tutti i punti
        df_stab, stab_thr = _compute_local_stability(
            results_df=results_df,
            param_keys=param_keys,
            param_ranges=param_ranges,
            metric_col=metric_col,
            best_val=best_val,
            ratio_thr=smart_ratio_thr
        )
    
        print(f"[WFO-PLOT] Local stability threshold = {stab_thr:.2%} (>= {smart_ratio_thr:.0%} del best)")
    
        _pairwise_stability_map(
            df_stab=df_stab,
            param_keys=param_keys,
            max_pairs=smart_max_pairs,
            title_prefix=f"[WFO] {symbol}-{strategy}"
        )
    
        # 3) Edge bias: misura oggettiva del “best al bordo”
        edge_df = _edge_bias_report(
            results_df=results_df,
            param_keys=param_keys,
            top_mask=top_mask,
            param_ranges=param_ranges
        )
        display(edge_df.sort_values("edge_pct_top", ascending=False))

    # -----------------------------
    # Plot: heatmap / 3D / smart (originale)
    # -----------------------------
    if plot_results:

        if len(param_keys) == 2 and len(param_values) == 2:
            n1 = len(list(param_values[0]))
            n2 = len(list(param_values[1]))
            results_matrix = np.array(results).reshape(n1, n2)

            x_vals = list(param_values[1])
            y_vals = list(param_values[0])
            df_heat = pd.DataFrame(results_matrix, index=y_vals, columns=x_vals)

            fig = px.imshow(
                df_heat,
                labels=dict(x=param_keys[1], y=param_keys[0], color="Total Return"),
                x=x_vals,
                y=y_vals,
                color_continuous_scale="Viridis",
                aspect="equal"
            )

            best_val1 = best_params[param_keys[0]]
            best_val2 = best_params[param_keys[1]]
            fig.add_trace(
                go.Scatter(
                    x=[best_val2],
                    y=[best_val1],
                    mode="markers",
                    marker=dict(color="red", size=15, symbol="circle-open-dot",
                                line=dict(width=2, color="white")),
                    name="Best Combination"
                )
            )

            fig.update_layout(
                title=f"Heatmap Total Return ({symbol} - {strategy})",
                xaxis_title=param_keys[1],
                yaxis_title=param_keys[0],
                width=800,
                height=800
            )
            fig.show()

        elif len(param_keys) == 3 and len(param_values) == 3:
            df_3d = pd.DataFrame(all_combinations, columns=param_keys)
            df_3d["Total Return"] = results

            fig = px.scatter_3d(
                df_3d,
                x=param_keys[0],
                y=param_keys[1],
                z=param_keys[2],
                color="Total Return",
                color_continuous_scale="Viridis",
                hover_data={"Total Return": ':.2%'},
                title=f"Ottimizzazione Parametri 3D ({symbol} - {strategy})"
            )

            best_tuple = tuple(best_params[k] for k in param_keys)
            fig.add_trace(
                go.Scatter3d(
                    x=[best_tuple[0]],
                    y=[best_tuple[1]],
                    z=[best_tuple[2]],
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="diamond",
                                line=dict(width=2, color="white")),
                    name="Best Combination"
                )
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title=param_keys[0],
                    yaxis_title=param_keys[1],
                    zaxis_title=param_keys[2]
                ),
                width=1200,
                height=800
            )
            fig.show()

        else:
            print(f"\nTop {top_n_table} combinazioni di parametri:\n")
            display(results_df.head(top_n_table))

            if smart_plot and len(param_keys) >= 4:
                _plot_pair_heatmaps(
                    df=results_df,
                    keys=param_keys,
                    best=best_params,
                    metric_col="Total Return",
                    top_k_params=smart_top_k_params,
                    max_pairs=smart_max_pairs,
                    aggfunc=smart_aggfunc,
                    local_fix=smart_local_fix,
                    global_fallback=smart_global_fallback
                )

    # -----------------------------
    # Analisi robustezza + raccomandazione WFO (originale)
    # -----------------------------
    analysis_report = None
    if analyze_results and len(results_df) > 0:
        # --- CHIAMATA DA SOSTITUIRE (nel blocco analyze_results) ---
        analysis_report = _analyze_results(
            results_df=results_df,
            param_keys=param_keys,
            best_params=best_params,
            param_ranges=param_ranges,
            metric_col="Total Return",
            equity_report=equity_report
        )
        _print_analysis(analysis_report)

        # integra un riepilogo equity sweep dentro il report (se disponibile)
        if equity_report is not None:
            analysis_report["equity_sweep"] = equity_report

    if return_analysis:
        return portfolio_best, best_params, best_metric, results_df, analysis_report

    return portfolio_best, best_params, best_metric, results_df


def _rank_params_by_spearman(results_df: pd.DataFrame, param_keys: list, metric_col: str = "Total Return"):
    """Rank parametri per |Spearman rho| con la metrica."""
    ranks = []
    y = results_df[metric_col]
    for k in param_keys:
        # Spearman via rank-correlation
        rho = results_df[k].rank(method="average").corr(y.rank(method="average"))
        ranks.append((k, float(abs(rho)) if pd.notna(rho) else 0.0))
    return sorted(ranks, key=lambda x: x[1], reverse=True)

def _plot_pair_heatmaps_local(
    results_df: pd.DataFrame,
    param_keys: list,
    best_params: dict,
    metric_col: str = "Total Return",
    top_k_params: int = 4,
    aggfunc: str = "median",
    max_pairs: int = 6
):
    """
    Crea heatmap 2D (small multiples) sulle migliori coppie tra i top_k_params.
    Modalità "local": fissa tutti gli altri parametri ai valori best, così vedi il plateau attorno al best.
    """
    ranked = _rank_params_by_spearman(results_df, param_keys, metric_col=metric_col)
    top_params = [k for k, _ in ranked[:top_k_params]]

    pairs = list(itertools.combinations(top_params, 2))[:max_pairs]

    for a, b in pairs:
        # filtra righe dove gli altri parametri sono fissati al best
        mask = pd.Series(True, index=results_df.index)
        for k in param_keys:
            if k not in (a, b):
                mask &= (results_df[k] == best_params[k])

        df_loc = results_df[mask].copy()
        if df_loc.empty:
            # fallback: globale (senza fixing) se local non ha punti
            df_loc = results_df.copy()

        # pivot con aggregazione (median/mean)
        if aggfunc == "mean":
            pv = df_loc.pivot_table(index=a, columns=b, values=metric_col, aggfunc="mean")
        else:
            pv = df_loc.pivot_table(index=a, columns=b, values=metric_col, aggfunc="median")

        fig = px.imshow(
            pv,
            labels=dict(x=b, y=a, color=metric_col),
            aspect="equal",
            color_continuous_scale="Viridis",
            title=f"Heatmap {metric_col} (local fixing to best) — {a} vs {b}"
        )

        # marker best
        fig.add_trace(
            go.Scatter(
                x=[best_params[b]],
                y=[best_params[a]],
                mode="markers",
                marker=dict(color="red", size=14, symbol="circle-open-dot", line=dict(width=2, color="white")),
                name="Best"
            )
        )

        fig.update_layout(width=800, height=700)
        fig.show()

def _plateau_score(
    results_df: pd.DataFrame,
    param_keys: list,
    best_params: dict,
    param_ranges: dict,
    metric_col: str = "Total Return",
    ratio_thr: float = 0.80
):
    """
    Misura robustezza locale: vicini a 1-step per parametro (dove esistono).
    Restituisce un dict con statistiche di plateau.
    """
    # costruisci lookup per step discreti
    value_lists = {k: list(param_ranges[k]) for k in param_keys}

    # trova indice del valore best in ciascuna lista
    idx_map = {}
    for k in param_keys:
        vals = value_lists[k]
        try:
            idx_map[k] = vals.index(best_params[k])
        except ValueError:
            idx_map[k] = None

    # genera vicini: per ciascun parametro prova idx-1 e idx+1 (se esistono)
    neighbor_params = []
    for deltas in itertools.product([-1, 0, 1], repeat=len(param_keys)):
        if all(d == 0 for d in deltas):
            continue
        cand = {}
        ok = True
        for i, k in enumerate(param_keys):
            base_idx = idx_map[k]
            if base_idx is None:
                ok = False
                break
            new_idx = base_idx + deltas[i]
            if new_idx < 0 or new_idx >= len(value_lists[k]):
                ok = False
                break
            cand[k] = value_lists[k][new_idx]
        if ok:
            neighbor_params.append(cand)

    # dedup
    unique = []
    seen = set()
    for d in neighbor_params:
        t = tuple(d[k] for k in param_keys)
        if t not in seen:
            seen.add(t)
            unique.append(d)

    # estrai performance dei vicini
    best_val = float(results_df.loc[
        (results_df[param_keys] == pd.Series(best_params)).all(axis=1),
        metric_col
    ].iloc[0])

    neigh_vals = []
    for d in unique:
        m = (results_df[param_keys] == pd.Series(d)).all(axis=1)
        if m.any():
            neigh_vals.append(float(results_df.loc[m, metric_col].iloc[0]))

    if len(neigh_vals) == 0:
        return {
            "neighbors": 0,
            "best": best_val,
            "note": "Nessun vicino trovato (griglia troppo rada o best fuori griglia)."
        }

    neigh_vals = np.array(neigh_vals, dtype=float)
    ratio = neigh_vals / best_val

    return {
        "neighbors": int(len(neigh_vals)),
        "best": best_val,
        "neighbors_median": float(np.median(neigh_vals)),
        "neighbors_iqr": float(np.percentile(neigh_vals, 75) - np.percentile(neigh_vals, 25)),
        "neighbors_above_thr_pct": float((ratio >= ratio_thr).mean()),
        "ratio_thr": ratio_thr
    }

def k_run_portfolio(
    portfolio_cfg: dict,
    year: int | None = None,
    *,
    # --- dry-run ---
    dry_run: bool = False,

    # --- load TS batch ---
    report_end_date=None,                 # equivalente di end_date operativo (giorno "pulito")
    create_structure: bool = True,
    wfo_results_dir: str | None = None,  # (se lo vorrai riattivare)

    # --- report/email ---
    today=None,                           # None oppure 'YYYY-MM-DD' (se usato in report)
    sender_email: str = "",
    sender_password: str = "",
    recipient_email: str = "",
    subject=None,
    verbose: bool = False,

    # --- report controls ---
    check_open_trades: bool = True,
    check_close_trades: bool = True,
    generate_charts: bool = True,

    # --- attachments policy ---
    max_attachments_mb: int = 20,
    max_attachments_count: int = 10,
    attach_mode: str = "auto",
):
    """
    Esegue la pipeline operativa del portafoglio "standard" (non rotazionale),
    combinando:
      1) load_trading_systems_batch(...)
      2) send_portfolio_report(...)

    Struttura analoga a r_run_portfolio:
      - validazione minima input
      - year default
      - dry-run con stampa piano azioni
      - ritorno dict con output principali

    Dipendenze attese già disponibili nel progetto:
      - load_trading_systems_batch(...)
      - send_portfolio_report(...)
      - now()  (se vuoi uniformare year/today; qui uso pd.Timestamp)
    """
    import pandas as pd

    # --- Validazione minima portfolio_cfg ---
    if not isinstance(portfolio_cfg, dict):
        raise TypeError("portfolio_cfg deve essere un dict (config del portafoglio).")

    # --- Year default: anno corrente ---
    if year is None:
        year = int(pd.Timestamp.now().year)

    # --- end_date "pulito" (report_end_date) ---
    # Manteniamo esattamente la semantica che hai già: "passiamo SEMPRE un giorno-calendario pulito".
    # Qui normalizziamo solo se arriva Timestamp/str; altrimenti resta None e gestiscono le funzioni a valle.
    if report_end_date is not None:
        report_end_date = pd.to_datetime(report_end_date).normalize()

    # --- Portfolio title (best-effort) ---
    # Se nel cfg c'è Title, usalo; altrimenti fallback.
    portfolio_title = portfolio_cfg.get("Title", portfolio_cfg.get("title", "Portfolio"))

    # --- DRY RUN: stampa piano azioni e termina ---
    if dry_run:
        print("[DRY-RUN] k_run_portfolio")
        print(f"  - portfolio_title      : {portfolio_title}")
        print(f"  - year                 : {year}")
        print(f"  - report_end_date      : {report_end_date}")
        print(f"  - create_structure     : {create_structure}")
        print(f"  - today                : {today}")
        print("  - Azioni che verrebbero eseguite:")
        print("    1) load_trading_systems_batch(portfolio_cfg, end_date=report_end_date, ...)")
        print("    2) send_portfolio_report(portfolio_title, portfolio_ts, end_date=report_end_date, ...)")
        print("  - Email params:")
        print(f"    sender_email         : {sender_email}")
        print(f"    recipient_email      : {recipient_email}")
        print(f"    subject              : {subject}")
        print(f"    verbose              : {verbose}")
        return {
            "dry_run": True,
            "portfolio_title": portfolio_title,
            "year": year,
            "report_end_date": report_end_date,
            "portfolio_ts": None,
            "meta": None,
        }

    # ------------------------------------------------------------
    # 1) Load TS batch
    # ------------------------------------------------------------
    portfolio_ts, meta = load_trading_systems_batch(
        portfolio_cfg=portfolio_cfg,
        end_date=report_end_date,     # <-- giorno-calendario “pulito”
        verbose=verbose,
        wfo_results_dir=wfo_results_dir,
        create_structure=create_structure,
        auto_adjust=False  # Run-Time -> raw data!
    )
    
    if portfolio_ts is None or meta is None:
        raise ValueError("Errore load_trading_systems_batch. Exiting.")
    # ------------------------------------------------------------
    # 2) Send report
    # ------------------------------------------------------------
    # NB: mantengo la tua signature: la tua send_portfolio_report non riceve "today"
    # nel frammento mostrato, quindi non lo passo.
    send_portfolio_report(
        portfolio_title=portfolio_title,
        portfolio_ts=portfolio_ts,
        end_date=report_end_date,
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=recipient_email,
        subject=subject,
        verbose=verbose,
        check_open_trades=check_open_trades,
        check_close_trades=check_close_trades,
        generate_charts=generate_charts,
        max_attachments_mb=max_attachments_mb,
        max_attachments_count=max_attachments_count,
        attach_mode=attach_mode
    )

    return {
        "dry_run": False,
        "portfolio_title": portfolio_title,
        "year": year,
        "report_end_date": report_end_date,
        "portfolio_ts": portfolio_ts,
        "meta": meta,
    }

# =========================================================
# Esecuzione batch dei Trading System (versione load wfo resut)
# =========================================================

def load_trading_systems_batch(
    portfolio_cfg: dict,
    end_date: str| None = None,
    start_date: str | None = None,   # ← aggiungi
    wfo_results_dir: str = "WFO_RESULTS",
    verbose: bool = True,
    create_structure : bool = False,
    auto_adjust: bool = True 
):

    """
    Carica tutti i TS definiti in portfolio_cfg['trading_systems'] e
    restituisce la lista di portafogli TS + meta info.

    """
    # Parametri globali
    params_global = portfolio_cfg.get("params_global", {})
    title         = portfolio_cfg.get("Title", "Portfolio")
    
    # end_date risolto in modo sicuro e normalizzato
    end_date = resolve_end_date(end_date, tz_name="Europe/Rome")
    if start_date is not None:
        import pandas as pd
        start_date = pd.Timestamp(start_date).to_pydatetime().replace(tzinfo=None)
    else:
        start_date = datetime(end_date.year - 2, 1, 1)

    if verbose:
        print("Start date (batch):", start_date.date())
        print("End date   (batch):", end_date.date())



    init_cash     = params_global.get("init_cash", 100_000)
    fees          = params_global.get("fees", 0.001)
    slippage      = params_global.get("slippage", 0.0)
    # sl_stop       = params_global.get("sl_stop", None)
    price_col     = params_global.get("price_col", "Close")
    # operating_freq= params_global.get("operating_freq", "1D")
    default_ratio    = params_global.get("ratio", 4)

    if verbose:
        print(f"▶️ Running Portfolio: {title} | Date: {end_date}\n")

    portfolio_ts = []
    tickers      = []

    for ts in portfolio_cfg.get("trading_systems", []):
        symbol       = ts["symbol"]
        strategy     = ts["strategy"]
        # param_ranges = ts.get("param_ranges", [])
        ratio        = ts.get("ratio", default_ratio)
        _init_cash   = ts.get("init_cash", init_cash)
        _fees        = ts.get("fees", fees)
        _slippage    = ts.get("slippage", slippage)
        # _sl_stop     = ts.get("sl_stop", sl_stop)
        _price_col   = ts.get("price_col", price_col)
        # _freq        = ts.get("operating_freq", operating_freq)

        # ratio è stringa tipo "train:test"
        # train_window = int(str(ratio).split(":")[0])
        # start_date = datetime(end_date.year - (train_window + 2), 1, 1)

        if verbose:
            print(f"▶️ Ticker: {symbol} | Strategy: {strategy}")
            

        if strategy == "holding":
            prices = get_clean_financial_data(symbol, end=end_date, auto_adjust=auto_adjust)["Close"]
            
            pf = vbt.Portfolio.from_holding(
                prices,
                # init_cash=_init_cash,
                freq='D'
                # group_by=True,
                # fees=_fees,
                # slippage=_slippage
            )
        else:
                
            # prendo solo i risultati di WFO ma devo eseguire nuovamente
            # il backtest. In pratica una run_strategy senza WFO
            if verbose:
                print(f"  Loading WFO Results ...")

            *_,  summary_df = load_ts(symbol=symbol,
                                       strategy=strategy,
                                       ratio=ratio,
                                       wfo_results_dir=wfo_results_dir,
                                       show_result=False,
                                       create_structure=create_structure,
                                       only_wfo_results=True)
            # display(summary_df)
            if summary_df is None:
                raise ValueError("Errore load_ts (summary_df). Exiting.")
            # run_strategy con parametri WFO gia' disponibili
            if verbose:
                print(f"  Running TS ...\n")

            pf, _, _, _, _  = run_strategy(
                symbol=symbol,
                strategy_name=strategy,
                summary_df=summary_df,
                # param_ranges=param_ranges,
                start_date=start_date,
                end_date=end_date,
                # ratio=ratio,
                show_progress=False,
                init_cash=_init_cash,
                fees=_fees,
                slippage=_slippage,
                price_col=_price_col,
                verbose=False,
                save_results=False,
                auto_adjust=auto_adjust,
                run_time=True
            )

        portfolio_ts.append({
            "symbol": symbol,
            "strategy": strategy,
            "portfolio": pf,
            # "ratio": ratio,
            "performance": float(pf.total_return()),
            "returns": pf.returns().fillna(0.0)
        })
        tickers.append(symbol)
        
    meta = {
        "Title": title,
        "StartDate": start_date,
        "EndDate": end_date,
        "Tickers": tickers,
        "InitCash": init_cash,
        "Fees": fees,
        "Slippage": slippage,
        "PriceCol": price_col
        # "OperatingFreq": operating_freq
    }

    return portfolio_ts, meta

    
    
# =========================================================
# Esecuzione batch dei Trading System (versione WFO - OLD)
# =========================================================

def run_trading_systems_batch(
    portfolio_cfg: dict,
    # end_date: pd.Timestamp | datetime | None = None,
    end_date: str| None = None,
    verbose: bool = True
):
    """
    Esegue tutti i TS definiti in portfolio_cfg['trading_systems'] e
    restituisce la lista di portafogli TS + meta info.
    Usa direttamente:
      - run_wfo_ts(...)  (già definito nel tuo ambiente)
      - get_clean_financial_data(...)  (già definito nel tuo ambiente)
    """
    # Parametri globali
    params_global = portfolio_cfg.get("params_global", {})
    title         = portfolio_cfg.get("Title", "Portfolio")
    
    # end_date risolto in modo sicuro e normalizzato
    end_date = resolve_end_date(end_date, tz_name="Europe/Rome")
    
    init_cash     = params_global.get("init_cash", 100_000)
    fees          = params_global.get("fees", 0.001)
    slippage      = params_global.get("slippage", 0.0)
    sl_stop       = params_global.get("sl_stop", None)
    price_col     = params_global.get("price_col", "Close")
    operating_freq= params_global.get("operating_freq", "1D")
    default_ratio    = params_global.get("ratio", 4)

    if verbose:
        print(f"▶️ Running Portfolio: {title} | Date: {end_date}\n")

    portfolio_ts = []
    tickers      = []

    for ts in portfolio_cfg.get("trading_systems", []):
        symbol       = ts["symbol"]
        strategy     = ts["strategy"]
        param_ranges = ts.get("param_ranges", [])
        ratio        = ts.get("ratio", default_ratio)
        _init_cash   = ts.get("init_cash", init_cash)
        _fees        = ts.get("fees", fees)
        _slippage    = ts.get("slippage", slippage)
        _sl_stop     = ts.get("sl_stop", sl_stop)
        _price_col   = ts.get("price_col", price_col)
        _freq        = ts.get("operating_freq", operating_freq)

        # ratio è stringa tipo "train:test"
        train_window = int(str(ratio).split(":")[0])
        start_date = datetime(end_date.year - (train_window + 2), 1, 1)

        if verbose:
            print(f"▶️ Running TS: {symbol} | Strategy: {strategy} | params: {param_ranges}")

        if strategy == "holding":
            prices = get_clean_financial_data(symbol, start=start_date, end=end_date,auto_adjust=False)["Close"]

            # prices = load_ohlcv(symbol, start=start_date, end=end_date, auto_adjust=auto_adjust)

            pf = vbt.Portfolio.from_holding(
                prices,
                init_cash=_init_cash,
                freq='D',
                group_by=True,
                fees=_fees,
                slippage=_slippage
            )
        else:
            pf, _, _, _, _  = run_strategy(
                symbol=symbol,
                strategy_name=strategy,
                param_ranges=param_ranges,
                start_date=start_date,
                end_date=end_date,
                ratio=ratio,
                show_progress=False,
                init_cash=_init_cash,
                fees=_fees,
                slippage=_slippage,
                price_col=_price_col,
                verbose=False,
                save_results=False,
                auto_adjust=False
            )
            
        portfolio_ts.append({
            "symbol": symbol,
            "strategy": strategy,
            "portfolio": pf,
            "ratio": ratio,
            "performance": float(pf.total_return()),
            "returns": pf.returns().fillna(0.0)
        })
        tickers.append(symbol)

    meta = {
        "Title": title,
        "EndDate": end_date,
        "Tickers": tickers,
        "InitCash": init_cash,
        "Fees": fees,
        "Slippage": slippage,
        "OperatingFreq": operating_freq
    }
    return portfolio_ts, meta

def crea_portafoglio_combinato(
    portfolio_ts,
    init_cash=100_000,
    freq='D',
    normalizza_pesi=True,
    start_date=None,
    end_date=None,
    debug: bool = False
):
    """
    Crea un portafoglio sintetico combinando più portafogli individuali tramite pesi assegnati.

    Parametri:
    - portfolio_ts: lista di dizionari contenenti almeno ['portfolio', 'symbol']
    - init_cash: capitale iniziale per il portafoglio combinato
    - freq: frequenza dei dati ('D', 'W', ecc.)
    - normalizza_pesi: se True, normalizza i pesi per sommare a 1
    - start_date: data opzionale di inizio del periodo da analizzare
    - end_date: data opzionale di fine del periodo da analizzare

    Ritorna:
    - final_portfolio: portafoglio completo combinato
    - final_portfolio_period: portafoglio combinato nel periodo definito
    - portfolios: portafogli originali
    - synthetic_price_period: prezzo sintetico cumulato nel periodo
    """

    # ------------------------------------------------------------------
    # 1) Estrazione portafogli
    # ------------------------------------------------------------------
    portfolios = [ts['portfolio'] for ts in portfolio_ts]

    if len(portfolios) == 0:
        raise ValueError("portfolio_ts è vuoto: impossibile creare portafoglio combinato.")

    # ------------------------------------------------------------------
    # 2) Pesi equal-weight
    # ------------------------------------------------------------------
    weights = [1 / len(portfolios)] * len(portfolios)

    # ------------------------------------------------------------------
    # 3) Costruzione DataFrame rendimenti
    # ------------------------------------------------------------------
    returns_df = pd.DataFrame({
        f"p{i}": p.returns()
        for i, p in enumerate(portfolios)
    })

    returns_df.dropna(inplace=True)

    if returns_df.empty:
        raise ValueError("returns_df vuoto: impossibile costruire rendimenti combinati.")

    # ------------------------------------------------------------------
    # 4) Normalizzazione pesi
    # ------------------------------------------------------------------
    if normalizza_pesi:
        sum_w = sum(weights)
        if sum_w != 1.0:
            weights = [w / sum_w for w in weights]

    weight_s = pd.Series(weights, index=returns_df.columns)

    # ------------------------------------------------------------------
    # 5) Rendimenti combinati
    # ------------------------------------------------------------------
    combined_returns = (returns_df * weight_s).sum(axis=1)
    combined_returns.dropna(inplace=True)

    if combined_returns.empty:
        raise ValueError("combined_returns vuoto dopo il calcolo.")

    # ------------------------------------------------------------------
    # 6) Definizione finestra temporale (start_date / end_date)
    # ------------------------------------------------------------------
    # Normalizzazione date
    start_ts = pd.to_datetime(start_date) if start_date is not None else None
    end_ts   = pd.to_datetime(end_date)   if end_date   is not None else None

    if start_ts is None and end_ts is None:
        # comportamento legacy: YTD
        anno_corrente = datetime.now().year
        combined_returns_period = combined_returns[
            combined_returns.index.year == anno_corrente
        ]
    else:
        # slicing coerente
        if start_ts is None:
            start_ts = combined_returns.index.min()
        if end_ts is None:
            end_ts = combined_returns.index.max()

        # swap di sicurezza
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts

        combined_returns_period = combined_returns.loc[start_ts:end_ts]

    if combined_returns_period.empty:
        raise ValueError(
            f"Nessun dato nella finestra richiesta: start_date={start_date}, end_date={end_date}"
        )

    # ------------------------------------------------------------------
    # 7) Prezzo sintetico cumulato
    # ------------------------------------------------------------------
    synthetic_price = (1.0 + combined_returns).cumprod()
    synthetic_price_period = (1.0 + combined_returns_period).cumprod()

    # Forzatura primo valore a 1.0
    synthetic_price.iloc[0] = 1.0
    synthetic_price_period.iloc[0] = 1.0

    if debug:
        print("combined_returns start:", combined_returns.index.min())
        print("combined_returns first:", combined_returns.iloc[0])
        print("synthetic_price first:", synthetic_price.iloc[0])
        bad = (~np.isfinite(combined_returns)) | (combined_returns <= -1)
        print("bad returns count:", int(bad.sum()))
        if bad.any():
            print("first bad date:", combined_returns.index[bad.argmax()], "value:", combined_returns[bad].iloc[0])

    # ------------------------------------------------------------------
    # 8) Creazione portafogli VectorBT
    # ------------------------------------------------------------------
    final_portfolio = vbt.Portfolio.from_holding(
        close=synthetic_price,
        init_cash=init_cash,
        freq=freq
    )

    final_portfolio_period = vbt.Portfolio.from_holding(
        close=synthetic_price_period,
        init_cash=init_cash,
        freq=freq
    )

    return (
        final_portfolio,
        final_portfolio_period,
        portfolios,
        synthetic_price_period
    )


def generate_portfolio_report(
    portfolio_title,
    portfolio_ts,
    end_date,
    sender_email="",
    sender_password="",
    recipient_email="",
    subject=None,
    verbose=False,
    check_open_trades=False,
    check_close_trades=False,
    generate_charts=True,
    # --- NEW: controlli allegati / peso email ---
    attach_mode="signals_only",     # "signals_only" | "all_generated"
    max_attachments_mb=15,          # budget totale allegati (MB)
    max_attachments_count=10,       # numero massimo allegati
    dedupe_attachments=True,        # evita duplicati in attachments
):
    """
    Genera un report HTML unico per tutte le strategie contenute in `portfolio_ts`.
    Opzionalmente invia il report via email e allega i grafici generati.

    Struttura attesa di `portfolio_ts` (list[dict]):
      - 'symbol'    : str, ticker
      - 'strategy'  : str, nome strategia
      - 'portfolio' : oggetto portafoglio compatibile con `extract_last_trade(...)`
                      e `generate_trade_charts(...)`

    Parametri principali
    -------------------
    verbose : bool
        Se True:
          - visualizza a video la tabella riepilogativa (display(df_report))
          - visualizza i grafici generati (plot_fig.show()).
    generate_charts : bool
        Se True abilita la generazione grafici.

    check_open_trades / check_close_trades : bool
        Se True, genera grafici anche per trade "Open" / "Closed" (oltre ai segnali).

    Nuovo comportamento allegati (richiesta)
    ---------------------------------------
    - Se un grafico viene generato, può essere aggiunto agli allegati in base a `attach_mode`.
    - Se verbose=True, il grafico è SEMPRE mostrato a video (se generato), indipendentemente
      dall'allegazione.
    - Per evitare fallimenti di invio dovuti a eccesso di peso/numero allegati, l’allegazione
      è limitata da:
        * max_attachments_mb (budget totale)
        * max_attachments_count (numero massimo)
      I grafici eccedenti vengono scartati dagli allegati e segnalati nel report HTML.

    attach_mode:
      - "signals_only": allega solo i grafici dei TS che hanno segnale (signal=True).
                        (consigliato se `check_open_trades=True` per evitare email troppo pesanti)
      - "all_generated": allega ogni grafico generato (entro i limiti di size/count).

    Output
    ------
    Nessun return esplicito.
    Effetti collaterali:
      - invia email se `recipient_email` è valorizzato
      - stampa/mostra output se `verbose=True`
    """

    # end_date risolto in modo sicuro e normalizzato
    end_date = resolve_end_date(end_date, tz_name="Europe/Rome")

    # Import locali (evita dipendenze globali non garantite)
    import os

    report_data = []
    attachments = []
    n_open_trades = 0
    n_closed_trades = 0
    n_signal_trades = 0

    # Per la sezione HTML "Trading system con segnale"
    ts_with_signal = []

    # Tracking limiti allegati
    max_bytes = int(max_attachments_mb * 1024 * 1024)
    attachments_bytes = 0
    skipped_attachments = []  # elenco grafici generati ma NON allegati (per limiti)

    def _try_add_attachment(path: str) -> bool:
        """
        Aggiunge `path` agli attachments rispettando:
        - esistenza file
        - dedupe opzionale
        - max_attachments_count
        - max_attachments_mb (budget totale)
        """
        nonlocal attachments_bytes, attachments

        if not path or not isinstance(path, str) or not os.path.exists(path):
            return False

        if dedupe_attachments and path in attachments:
            return True  # già presente: lo consideriamo "aggiunto"

        if len(attachments) >= int(max_attachments_count):
            return False

        size = os.path.getsize(path)
        if attachments_bytes + size > max_bytes:
            return False

        attachments.append(path)
        attachments_bytes += size
        return True

    # --- Anagrafica società in un colpo solo (cache) ---
    tickers = [entry["symbol"] for entry in portfolio_ts]
    company_data = build_company_df_with_cache(tickers)

    # --- Loop su tutti i TS del portafoglio ---
    for ts in portfolio_ts:
        symbol = ts["symbol"]
        strategy = ts["strategy"]
        portfolio = ts["portfolio"]

        # Company robusta
        company = company_data.at[symbol, "Company"] if symbol in company_data.index else ""

        # Estrai ultimo trade e segnale
        last_trade = extract_last_trade(portfolio, end_date)
        signal, signal_msg = identify_signal(last_trade, end_date)

        # Dettagli trade
        trade_status = last_trade["Status"] if last_trade is not None else "N/A"
        trade_enter = last_trade["Entry Timestamp"] if last_trade is not None else "N/A"
        trade_exit = last_trade["Exit Timestamp"] if last_trade is not None else "N/A"

        stato_descr = "<b>in posizione</b>" if trade_status == "Open" else "non posizionato"

        # Raccogli riga report
        report_data.append(
            {
                "Ticker": symbol,
                "Company": company,
                "Strategia": strategy,
                "Stato del sistema": stato_descr,
                "Entry Timestamp": trade_enter,
                "Exit Timestamp": trade_exit,
                "Segnale": signal_msg,
            }
        )

        # Contatori
        if trade_status == "Open":
            n_open_trades += 1
        if trade_status == "Closed":
            n_closed_trades += 1
        if signal:
            n_signal_trades += 1
            ts_with_signal.append(f"{strategy} ({symbol})")

        # --- Grafici ---
        # Determina se generare un grafico
        if generate_charts:
            should_plot = (
                signal
                or (check_open_trades and trade_status == "Open")
                or (check_close_trades and trade_status == "Closed")
            )

            if should_plot:
                plot_filename, plot_fig = generate_trade_charts(
                    portfolio, symbol, strategy, titolo=symbol
                )

                # Se verbose: mostra sempre a video
                if verbose and plot_fig is not None:
                    plot_fig.show()

                # Policy allegazione
                should_attach = False
                if attach_mode == "signals_only":
                    should_attach = bool(signal)
                elif attach_mode == "all_generated":
                    should_attach = True
                else:
                    # fallback conservativo
                    should_attach = bool(signal)

                # Se dobbiamo allegare, proviamo rispettando i limiti
                if should_attach and plot_filename:
                    ok = _try_add_attachment(plot_filename)
                    if not ok:
                        skipped_attachments.append(
                            {"symbol": symbol, "strategy": strategy, "file": plot_filename}
                        )

    # --- DataFrame del report ---
    df_report = pd.DataFrame(report_data)

    # Stampa tabella a schermo
    if verbose:
        display(df_report)

    # --- Conversione HTML con stile CSS migliorato (FORMATO INVARIATO) ---
    table_html = df_report.to_html(index=False, escape=False, border=0, classes="styled-table")

    table_style = """
    <style>
    .styled-table {
        border-collapse: collapse;
        width: 80%;
        font-family: Arial, sans-serif;
        font-size: 14px;
        margin: 20px 0;
        min-width: 600px;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.15);
    }
    .styled-table th {
        text-align: left;
        padding: 12px;
        background-color: #f2f2f2;
        font-weight: bold;
        border-bottom: 1px solid #ddd;
    }
    .styled-table td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #eee;
    }
    </style>
    """

    # Sezione TS con segnale (mantiene il formato originale: lista HTML)
    ts_with_signal_html = (
        "<ul>" + "".join([f"<li>{x}</li>" for x in ts_with_signal]) + "</ul>"
        if ts_with_signal
        else "<p>Nessun segnale rilevato.</p>"
    )

    # Nota (aggiunta, ma senza alterare il layout della tabella/struttura principale)
    skipped_note = ""
    if skipped_attachments:
        skipped_note = (
            f"<p><strong>Nota allegati:</strong> alcuni grafici sono stati generati ma non allegati "
            f"per limiti di invio (attach_mode={attach_mode}, budget={max_attachments_mb}MB, "
            f"max_count={max_attachments_count}). "
            f"Scartati: <strong>{len(skipped_attachments)}</strong>.</p>"
        )

    # --- Composizione finale dell’HTML (FORMATO INVARIATO, con sola nota extra se serve) ---
    html_report = f"""
    <html>
    <head>
        <title>Trading Portfolio {portfolio_title} Report</title>
        {table_style}
    </head>
    <body>
        <h2>📈 Report di Trading - Portafoglio {portfolio_title}</h2>
                
        <h3>Sommario</h3>
        <table style="width: 40%; border-collapse: collapse; margin: 0em 0;">
          <tbody>
            <tr>
              <td style="padding: 0.5em; border-bottom: 0px solid #ccc;">Data di riferimento</td>
              <td style="padding: 0.5em; border-bottom: 0px solid #ccc;"><strong>{end_date.date()}</strong></td>
            </tr>
            <tr>
              <td style="padding: 0.5em; border-bottom: 0px solid #ccc;">In posizione</td>
              <td style="padding: 0.5em; border-bottom: 0px solid #ccc;"><strong>{n_open_trades}</strong></td>
            </tr>
            <tr>
              <td style="padding: 0.5em; border-bottom: 0px solid #ccc;">Non in posizione</td>
              <td style="padding: 0.5em; border-bottom: 0px solid #ccc;"><strong>{n_closed_trades }</strong></td>
            </tr>
            <tr>
              <td style="padding: 0.5em;">Con segnale</td>
              <td style="padding: 0.5em;"><strong>{n_signal_trades}</strong></td>
            </tr>
          </tbody>
        </table>

        <h3>Riepilogo strategico</h3>
        {table_html}

        <p><strong>Trading system con segnale</strong></p>
        {ts_with_signal_html}

        {skipped_note}

        {"<h4>Per visionare i grafici interattivi salvare l'allegato e aprire con un browser</h4>" if attachments else ""}
    </body>
    </html>
    """

    # Oggetto email predefinito se non specificato
    if subject is None:
        subject = f"[TS_LAB] Report di Portafoglio {portfolio_title} ({end_date.date()})"

    # Invio o stampa
    if recipient_email:
        send_email_report(
            sender_email,
            sender_password,
            recipient_email,
            subject,
            html_report,
            attachments
        )
    else:
        if verbose:
            print("\n*** Nessun destinatario specificato, non spedisco l'email. Report HTML: ***\n")
            print(html_report)


def extract_last_trade(portfolio, end_date=None):
    """ Estrae l'ultimo trade dal portfolio. """
    trades_info = portfolio.trades.records_readable
    if trades_info.empty:
        return None  # Nessun trade presente
    return trades_info.iloc[-1]
    
def identify_signal(last_trade, end_date):
    """ Identifica se c'è un segnale di ingresso o uscita. """
    if last_trade is None:
        return False, "No Trades"

    trade_status = last_trade["Status"]
    trade_enter = last_trade["Entry Timestamp"]
    trade_exit = last_trade["Exit Timestamp"]

    # # Simulazione: scommentare solo per simulare il segnale.
    # if trade_status == "Open" : return True, "Buy Signal"
    # if trade_status == "Closed": return True, "Sell Signal"
    # # fine simulazione

    # print(f"\ntrade_enter.date: {trade_enter.date()} trade_exit.date: {trade_exit.date()} end_date: {end_date.date()} ")

    # if trade_status == "Open" and trade_enter.date() == end_date.date():
    if trade_status == "Open" and trade_enter.date() == (end_date - timedelta(days=1)).date():
        return True, "<p style='color:green;'><b>Buy Signal</b></p>"
    # elif trade_status == "Closed" and trade_exit.date() == end_date.date():
    elif trade_status == "Closed" and trade_exit.date() == (end_date - timedelta(days=1)).date():
        return True, "<p style='color:red;'><b>Sell Signal</b></p>"
    return False, "No Signal"
    

# ===== Helpers =====
import pytz
def resolve_end_date(report_end_date=None, tz_name="Europe/Rome"):
    """
    Ritorna un pd.Timestamp NORMALIZZATO (00:00) per l'end_date da usare nel batch.
    Priorità:
      2) params_global['end_date'] se presente
      3) oggi nel fuso tz_name (default Europe/Rome)
    """
    if report_end_date is not None and str(report_end_date).strip() != "":
        return pd.Timestamp(report_end_date).normalize()


    # default: oggi nel fuso scelto (mattina EU → Europe/Rome)
    return pd.Timestamp.now(tz=pytz.timezone(tz_name)).normalize()


def today_eu_date():
    """Ritorna la data odierna (normalizzata) in fuso Europe/Rome."""
    return pd.Timestamp.now(tz=pytz.timezone("Europe/Rome")).normalize()

def _to_naive_date(s):
    """Converte a datetime naive normalizzato (rimuove timezone)."""
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()

def _prev_bar_before_date(portfolio, anchor_date: pd.Timestamp) -> pd.Timestamp:
    """
    Ritorna la massima data nell'indice del PF strettamente < anchor_date.
    Se non esistono date < anchor_date, ritorna la massima disponibile.
    """
    idx = pd.DatetimeIndex(portfolio.wrapper.index).normalize().unique()
    a = pd.Timestamp(anchor_date).normalize()
    prev = idx[idx < a]
    return prev.max() if len(prev) > 0 else idx.max()

def extract_last_trade_on_or_before(portfolio, ref_date: pd.Timestamp):
    """Ritorna l’ultimo trade con entry ≤ ref_date."""
    ti = portfolio.trades.records_readable
    if ti is None or ti.empty:
        return None
    df = ti.copy()
    df["Entry Timestamp"] = _to_naive_date(df["Entry Timestamp"])
    if "Exit Timestamp" in df.columns:
        df["Exit Timestamp"] = _to_naive_date(df["Exit Timestamp"])
    df = df[df["Entry Timestamp"] <= pd.Timestamp(ref_date).normalize()]
    if df.empty:
        return None
    return df.iloc[-1]

def is_in_position_at_date(portfolio, ref_date: pd.Timestamp) -> bool:
    """True se alla fine di ref_date il TS risulta in posizione."""
    ti = portfolio.trades.records_readable
    if ti is None or ti.empty:
        return False
    df = ti.copy()
    df["Entry Timestamp"] = _to_naive_date(df["Entry Timestamp"])
    if "Exit Timestamp" in df.columns:
        df["Exit Timestamp"] = _to_naive_date(df["Exit Timestamp"])
    d = pd.Timestamp(ref_date).normalize()
    if "Status" in df.columns:
        mask = (df["Status"].astype(str).str.lower() == "open") & (df["Entry Timestamp"] <= d)
        if mask.any():
            return True
    return bool(((df["Entry Timestamp"] <= d) & (df["Exit Timestamp"].isna() | (df["Exit Timestamp"] > d))).any())

def identify_signal_on_date_from_portfolio(portfolio, ref_date: pd.Timestamp):
    """
    Segnali generati su ref_date:
      - BUY se qualche trade ha Entry == ref_date
      - SELL se qualche trade ha Exit == ref_date
    """
    ti = portfolio.trades.records_readable
    if ti is None or ti.empty:
        return False, "No Trades"
    df = ti.copy()
    df["Entry Timestamp"] = _to_naive_date(df["Entry Timestamp"])
    if "Exit Timestamp" in df.columns:
        df["Exit Timestamp"] = _to_naive_date(df["Exit Timestamp"])
    else:
        df["Exit Timestamp"] = pd.NaT
    d = pd.Timestamp(ref_date).normalize()
    is_buy  = (df["Entry Timestamp"] == d).any()
    is_sell = (df["Exit Timestamp"]  == d).any()
    if is_buy and not is_sell: return True, "<p style='color:green;'><b>Buy Signal</b></p>"
    if is_sell and not is_buy: return True, "<p style='color:red;'><b>Sell Signal</b></p>"
    if is_buy and is_sell:     return True, "<p style='color:#a60;'><b>Buy & Sell (check)</b></p>"
    return False, "No Signal"

# ===== Report =====

def generate_portfolio_report_NEW(portfolio_title, portfolio_ts, end_date=None,
                              sender_email="", sender_password="",
                              recipient_email="", subject=None,
                              verbose=False, check_open_trades=False, 
                              check_close_trades=False,
                              generate_charts=True):
    """
    Genera un report unico per tutte le strategie contenute in portfolio_ts.
    Valuta ogni TS sulla close di ieri (ref_day = max barra < anchor_date).
    """
    report_data, attachments = [], []
    n_open_trades = n_closed_trades = n_signal_trades = 0

    # 1) Anchor date = oggi (Europe/Rome) o data passata se fornita
    anchor_date = pd.Timestamp(end_date).normalize() if end_date is not None else today_eu_date()

    tickers = [e['symbol'] for e in portfolio_ts]
    company_data = build_company_df_with_cache(tickers)

    for ts in portfolio_ts:
        symbol   = ts['symbol']
        strategy = ts['strategy']
        portfolio = ts['portfolio']
        company  = company_data.at[symbol, 'Company']

        # 2) ref_day = ultima barra completa (< anchor_date)
        ref_day = _prev_bar_before_date(portfolio, anchor_date)
        action_date = (ref_day + pd.Timedelta(days=1)).date()   # esecuzione reale

        # 3) Stato & segnale sulla ref_day
        last_trade = extract_last_trade_on_or_before(portfolio, ref_day)
        in_pos     = is_in_position_at_date(portfolio, ref_day)
        signal, signal_msg = identify_signal_on_date_from_portfolio(portfolio, ref_day)

        trade_enter = last_trade["Entry Timestamp"] if last_trade is not None else "N/A"
        trade_exit  = last_trade["Exit Timestamp"] if last_trade is not None else "N/A"
        stato_descr = "<b>in posizione</b>" if in_pos else "non posizionato"

        report_data.append({
            "Ticker": symbol,
            "Company": company,
            "Strategia": strategy,
            "Stato del sistema": stato_descr,
            "Entry Timestamp": trade_enter,
            "Exit Timestamp": trade_exit,
            "Segnale": signal_msg,
            "Data valutata": ref_day.date(),
            "Action Date": action_date
        })

        n_open_trades  += int(in_pos)
        n_closed_trades += int(not in_pos)
        n_signal_trades += int(signal)

        # 4) Grafici allegati opzionali
        if (signal and generate_charts) or (check_open_trades and in_pos) or (check_close_trades and not in_pos):
            plot_filename, plot_fig = generate_trade_charts(portfolio, symbol, strategy, titolo=symbol)
            if verbose and plot_fig is not None: plot_fig.show()
            if plot_filename and os.path.exists(plot_filename): attachments.append(plot_filename)

    df_report = pd.DataFrame(report_data)
    if verbose:
        display(df_report)

    # 5) HTML del report
    table_html = df_report.to_html(index=False, escape=False, border=0, classes="styled-table")
    table_style = """
    <style>
    .styled-table { border-collapse: collapse; width: 90%; font-family: Arial, sans-serif;
        font-size: 14px; margin: 20px 0; min-width: 600px; box-shadow: 0 0 5px rgba(0,0,0,.15);}
    .styled-table th { text-align: left; padding: 12px; background-color: #f2f2f2;
        font-weight: bold; border-bottom: 1px solid #ddd;}
    .styled-table td { padding: 10px; text-align: left; border-bottom: 1px solid #eee;}
    </style>"""

    html_report = f"""
    <html><head><title>Trading Portfolio {portfolio_title} Report</title>{table_style}</head>
    <body>
        <h2>📈 Report di Trading - Portafoglio {portfolio_title}</h2>
        <h3>Sommario</h3>
        <table style="width: 48%; border-collapse: collapse; margin: 0 0;">
          <tbody>
            <tr><td style="padding:.5em;">Anchor date (Europe/Rome)</td>
                <td style="padding:.5em;"><strong>{anchor_date.date()}</strong></td></tr>
            <tr><td style="padding:.5em;">In posizione</td>
                <td style="padding:.5em;"><strong>{n_open_trades}</strong></td></tr>
            <tr><td style="padding:.5em;">Non in posizione</td>
                <td style="padding:.5em;"><strong>{n_closed_trades}</strong></td></tr>
            <tr><td style="padding:.5em;">Con segnale (su ref_day)</td>
                <td style="padding:.5em;"><strong>{n_signal_trades}</strong></td></tr>
          </tbody>
        </table>

        <h3>Riepilogo Strategico</h3>
        {table_html}

        <p style="color:#666;margin-top:8px;">
          * Ogni TS è valutato su <b>Data valutata</b> (ultima barra completa ≤ anchor_date).<br>
          * <b>Action Date</b> = giorno di esecuzione in reale (open del giorno successivo).
        </p>
    </body></html>"""

    if subject is None:
        subject = f"[TS_LAB] Report di Portafoglio {portfolio_title} ({anchor_date.date()})"

    attachments = [p for p in attachments if isinstance(p, str) and os.path.exists(p)]
    
    if recipient_email:
        send_email_report(sender_email, sender_password, recipient_email, subject, html_report, attachments)
    else:
        if verbose:
            print(html_report)

    return df_report, html_report

send_portfolio_report = generate_portfolio_report
send_portfolio_report_new = generate_portfolio_report_NEW


def generate_trade_charts(portfolio, symbol, strategy, titolo=None, outdir=None):
    """
    Genera e salva i grafici del portafoglio su tre subplot (Orders, Trades, Cumulative Returns),
    impostando un titolo principale e sincronizzando gli assi X per permettere uno zoom simultaneo.

    Parametri:
    -----------
    - portfolio: oggetto Portfolio che espone i metodi plot_orders(), plot_trades(), plot_cum_returns().
    - symbol: stringa, ticker/identificativo del simbolo su cui hai fatto backtest.
    - strategy: stringa, nome della strategia in uso (verrà usato nel nome del file di output).
    - titolo: (opzionale) stringa da mostrare come titolo principale sopra tutti i subplot. 
              Se None, non verrà mostrato alcun titolo principale.

    Ritorna:
    --------
    - plot_filename: percorso del file HTML generato (es. "portfolio_MSFT_MyStrat_plots.html").
    - fig_combined: l'oggetto plotly.graph_objects.Figure contenente i 3 subplot.
    """

    if outdir is None:
        outdir=_TSLAB_OUTPUTS_DIR

    # 1) Genera i tre singoli Figure da vectorbt/portfolio
    fig_orders  = portfolio.plot_orders()
    fig_trades  = portfolio.plot_trades()
    fig_creturns = portfolio.plot_cum_returns()

    # 2) Crea la figura "completa" con 3 righe, 1 colonna, e assi X condivisi (shared_xaxes=True)
    #    Imposta i titoli di ciascun subplot (Orders, Trades, Cumulative Returns).
    fig_combined = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Orders", "Trades", "Cumulative Returns"]
    )

    # 3) Aggiunge tutti i trace (le linee, marker, ecc.) di ciascun fig_** nella posizione corretta
    #    NB: fig_orders.data è una lista di trace Plotly (go.Scatter, go.Bar, ecc.)
    for trace in fig_orders.data:
        fig_combined.add_trace(trace, row=1, col=1)
    for trace in fig_trades.data:
        fig_combined.add_trace(trace, row=2, col=1)
    for trace in fig_creturns.data:
        fig_combined.add_trace(trace, row=3, col=1)

    # 4) (Opzionale) Imposta un titolo principale sopra tutti i subplot, se fornito
    if titolo:
        fig_combined.update_layout(
            title={
                "text": f"Titolo: {titolo}",
                "x": 0.5,           # titolo centrato orizzontalmente
                "xanchor": "center",
                "yanchor": "top"
            }
        )

    # 5) Imposta i layout comuni (dimensioni, margini, legenda, ecc.)
    fig_combined.update_layout(
        height=1200,     # altezza complessiva (in pixel)
        showlegend=True, # mostra la legenda (di default)
        margin=dict(t=100, b=50, l=50, r=50)  # margini personalizzati (in pixel)
    )

    # 6) Sincronizza esplicitamente tutti gli assi X: 
    #    in questo modo, se fai zoom/sposti l'asse X di un grafico, 
    #    verrà mantenuta la stessa finestra temporale sugli altri subplot.
    fig_combined.update_xaxes(matches="x")


    from plotly.offline import plot  # ← Import di plot per salvare in HTML


    # 7) Salva il tutto in un unico file HTML (non apre automaticamente il browser)
    plot_filename = f"portfolio_{symbol}_{strategy}_plots.html"
    
    if outdir: plot_filename = f"{outdir}/{plot_filename}"
    
    plot(fig_combined, filename=plot_filename, auto_open=False)

    return plot_filename, fig_combined

def analyze_wfo_results(
    winners_list,
    tickers=None,
    selection=None,
    top_n: int = 10
):
    """
    Analizza i risultati delle WFO a partire da winners_list.

    Parametri
    ---------
    winners_list : list
        Lista di tuple/array con almeno:
        - w[0] = nome strategia
        - w[1] = ticker
        - (opzionale) w[2] = anno o periodo
    tickers : list or iterable, opzionale
        Universo completo di tickers usati nel test (per calcolare coperture e B&H).
    selection : str, opzionale
        Nome del selettore / criterio di ranking (solo per stampa).
    top_n : int
        Quante strategie/tickers mostrare nelle tabelle "Top".

    Ritorna
    -------
    dict con:
        - 'df_raw'
        - 'strategy_stats'
        - 'ticker_stats'
        - 'year_stats' (se anni disponibili)
        - 'bh_tickers' (se tickers forniti)
    """

    if not winners_list:
        print(f"{BOLD}{RED}Nessuna strategia vincente trovata.{RESET}")
        return {}

    # --- Costruzione DataFrame di lavoro ---
    strategies = [w[0] for w in winners_list]
    win_tickers = [w[1] for w in winners_list]

    # prova a usare w[2] come anno/periodo se esiste
    years = None
    if len(winners_list[0]) > 2:
        years = [w[2] for w in winners_list]
        data = {"strategy": strategies, "ticker": win_tickers, "year": years}
    else:
        data = {"strategy": strategies, "ticker": win_tickers}

    df = pd.DataFrame(data)

    # --- Dedup su coppie (strategia, ticker, year) per sicurezza ---
    df = df.drop_duplicates()

    total_wins = len(df)
    n_strategies = df["strategy"].nunique()
    n_tickers_win = df["ticker"].nunique()
    tickers_with_win = sorted(df["ticker"].unique())

    # --- Universo tickers e B&H ---
    bh_tickers = []
    universe_size = None
    coverage_pct = None

    if tickers is not None:
        universe = sorted(set(tickers))
        universe_size = len(universe)
        win_universe = set(df["ticker"])
        bh_tickers = [t for t in universe if t not in win_universe]

        if universe_size > 0:
            coverage_pct = 100 * n_tickers_win / universe_size

    # --- Statistiche per strategia ---
    strategy_stats = (
        df.groupby("strategy")["ticker"]
        .count()
        .rename("n_wins")
        .sort_values(ascending=False)
        .to_frame()
    )
    strategy_stats["win_pct"] = 100 * strategy_stats["n_wins"] / total_wins

    # --- Statistiche per ticker ---
    ticker_stats = (
        df.groupby("ticker")["strategy"]
        .count()
        .rename("n_wins")
        .sort_values(ascending=False)
        .to_frame()
    )
    ticker_stats["win_pct"] = 100 * ticker_stats["n_wins"] / total_wins

    # --- Statistiche per anno (se disponibili) ---
    year_stats = None
    if "year" in df.columns:
        year_stats = (
            df.groupby("year")
            .agg(
                n_wins=("ticker", "size"),
                n_strategies=("strategy", "nunique"),
                n_tickers=("ticker", "nunique"),
            )
            .sort_index()
        )
        year_stats["wins_pct"] = 100 * year_stats["n_wins"] / total_wins

    # ====================== STAMPA RISULTATI ======================

    sel_label = f" con selettore {BOLD}{selection}{RESET}" if selection else ""

    print(f"{BOLD}{GREEN}Analisi risultati WFO{RESET}{sel_label}\n")

    # --- Sintesi generale ---
    print(f"{BOLD}1) Sintesi generale{RESET}")
    print(f"- Record complessivi in winners_list: {BOLD}{total_wins}{RESET}")
    print(f"- Strategie con almeno 1 vittoria:   {BOLD}{n_strategies}{RESET}")
    print(f"- Tickers con almeno 1 vittoria:     {BOLD}{GREEN}{n_tickers_win}{RESET}")
    print(f"- Elenco tickers con almeno 1 strategia vincente ({n_tickers_win}):")
    print(tickers_with_win)

    if universe_size is not None:
        print(f"- Universo tickers testato:          {BOLD}{universe_size}{RESET}")
        print(
            f"- Copertura universo (almeno 1 strategia vincente): "
            f"{BOLD}{coverage_pct:.1f}%{RESET} "
            f"({n_tickers_win}/{universe_size})"
        )
          
    print("\n")
  
    # --- Tickes solo B&H (se universo fornito) ---
    if tickers is not None and universe_size is not None:
        n_bh = len(bh_tickers)
        pct_bh = 100 * n_bh / universe_size if universe_size > 0 else 0.0
        print(f"{BOLD}2) Tickers senza strategia vincente (B&H only){RESET}")
        print(
            f"- Numero tickers solo B&H: {BOLD}{RED}{n_bh}{RESET} "
            f"({pct_bh:.1f}% dell'universo)"
        )
        if n_bh > 0:
            print(f"- Lista (ordinata):\n{sorted(bh_tickers)}")

    print("\n")

    # --- Strategie: ranking per numero di vittorie ---
    print(f"{BOLD}3) Distribuzione vittorie per strategia{RESET}")
    print(
        strategy_stats.head(top_n)
        .assign(win_pct=lambda x: x["win_pct"].round(1))
        .to_string()
    )
    if len(strategy_stats) > top_n:
        print(f"... ({len(strategy_stats) - top_n} strategie ulteriori)\n")
    else:
        print("")

    # --- Tickers: ranking per numero di vittorie ---
    print(f"{BOLD}4) Distribuzione vittorie per ticker{RESET}")
    print(
        ticker_stats.head(top_n)
        .assign(win_pct=lambda x: x["win_pct"].round(1))
        .to_string()
    )
    if len(ticker_stats) > top_n:
        print(f"... ({len(ticker_stats) - top_n} tickers ulteriori)\n")
    else:
        print("")

    # # --- Statistiche per anno / periodo ---
    # if year_stats is not None:
    #     print(f"{BOLD}5) Distribuzione per anno/periodo{RESET}")
    #     print(
    #         year_stats.assign(
    #             wins_pct=lambda x: x["wins_pct"].round(1)
    #         ).to_string()
    #     )
    #     print("")

    # --- Ritorno oggetti utili per analisi successive ---
    results = {
        "df_raw": df,
        "strategy_stats": strategy_stats,
        "ticker_stats": ticker_stats,
        # "year_stats": year_stats,
        "bh_tickers": bh_tickers,
    }
    return results

def run_ts_portfolio_performance(
    portfolio_cfg,
    sender_email,
    sender_password,
    recipient_email,
    show_report: bool = True,
    show_plots: bool = True,
    alpha_analysis: bool = True,
    verbose: bool = False,
    create_structure: bool = True,
    wfo_results_dir: str | None = None,
    auto_adjust: bool = True,
    analisys_start_date: str | None = None,
    analisys_end_date: str | None = None
):

    if not isinstance(portfolio_cfg, dict):
        raise TypeError("portfolio_cfg deve essere un dict (config del portafoglio).")

    portfolio_title = portfolio_cfg.get("Title", portfolio_cfg.get("title", "Portfolio"))
    params_global = portfolio_cfg["params_global"]
    portfolio_benchmark = params_global.get("benchmark", None)

    if verbose:
        print(f"\nRunning load_trading_systems_batch with auto_adjust={auto_adjust} .. ")

    portfolio_ts, meta = load_trading_systems_batch(
        portfolio_cfg=portfolio_cfg,
        end_date=analisys_end_date,
        start_date=analisys_start_date,   # ← aggiungi
        verbose=verbose,
        wfo_results_dir=wfo_results_dir,
        create_structure=create_structure,
        auto_adjust=auto_adjust
    )

        
    if verbose:
        print("\nRunning crea_portafoglio_combinato ...")

    pf, pf_period, portfolios, synthetic_price_period = crea_portafoglio_combinato(
        portfolio_ts,
        start_date=analisys_start_date,
        end_date=analisys_end_date
    )

    tickers = meta["Tickers"]


    stats = pf_period.stats()
    bh_start = stats["Start"]
    bh_end = stats["End"]+ pd.Timedelta(days=1)  # yf.download esclude la end date
    # print(bh_start,bh_end)

    
    tickers_data = download_data(
        tickers,
        start_date=bh_start,
        end_date=bh_end,
        show_progress=False
    )
    # print("tickers_data", tickers_data.tail())

    benchmark_data = download_data(
        portfolio_benchmark,
        start_date=bh_start,
        end_date=bh_end,
        show_progress=False
    )

    
    pf_b_h = vbt.Portfolio.from_holding(
        tickers_data,
        freq="D",
        group_by=True
    )

    # print(pf_b_h.stats())
    
    portfolio_title = f"{portfolio_title} - Total"
    portfolio_title += " Return" if auto_adjust else " Price"

    if verbose:
        print(f"\nRunning generate_portfolio_performance with benchmark {portfolio_benchmark} ...")

    
    perf_out = generate_portfolio_performance(
        pf=pf_period,
        portfolio_title=portfolio_title,
        pf_b_h=pf_b_h,
        # pf_b_h=None,
        portfolio_ts=portfolio_ts,
        benchmark=portfolio_benchmark,
        # benchmark=None,
        benchmark_data=benchmark_data,
        show_report=show_report,
        show_plots=show_plots,
        alpha_analysis=alpha_analysis,
    )

    if verbose:
        print("\nRunning send_portfolio_performance ...")

    send_portfolio_performance(
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=recipient_email,
        assets=perf_out
    )

    return {
        "pf": pf,
        "pf_period": pf_period,
        "pf_b_h": pf_b_h,
        "portfolio_ts": portfolio_ts,
        "portfolios": portfolios,
        "synthetic_price_period": synthetic_price_period,
        "perf_out": perf_out,
    }



########################
### New Montecarlo
########################


#
# Metodo 1
#

def monte_carlo_simulation(portfolio_returns,
                           n_simulations=1000,
                           init_value=100,
                           benchmark_return=None,
                           benchmark_drawdown=None,
                           seed=None,
                           show_plots=True,
                           show_summary=True,
                           slippage=0.0,
                           shock_frequency=0.0,
                           shock_magnitude=(0.05, 0.15),
                           risk_free_rate_annual=0.0):
    """
    Monte Carlo base (bootstrap i.i.d. sui rendimenti giornalieri) con slippage e shock negativi.

    Args:
        portfolio_returns (pd.Series or np.ndarray): rendimenti giornalieri.
        n_simulations (int): numero di simulazioni.
        init_value (float): valore iniziale del portafoglio.
        benchmark_return (float, optional): total return di riferimento (es. B&H).
        benchmark_drawdown (float, optional): max drawdown di riferimento (in valore assoluto positivo, es. 0.30).
        seed (int, optional): seed random.
        show_plots (bool): se True, mostra i grafici.
        show_summary (bool): se True, stampa il riepilogo.
        slippage (float): penalizzazione giornaliera (es. 0.0001 = -0.01%).
        shock_frequency (float): quota di giorni con shock negativi (0–1).
        shock_magnitude (tuple): intervallo degli shock negativi (es. (0.05, 0.15)).
        risk_free_rate_annual (float): tasso risk-free annuale per Sharpe (default 0.0).

    Returns:
        dict: risultati aggregati e distribuzioni (final_returns, drawdowns, equity_paths, ecc.).
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(portfolio_returns, pd.Series):
        returns = portfolio_returns.dropna().values
    else:
        returns = pd.Series(portfolio_returns).dropna().values

    n_days = len(returns)

    equity_paths = []
    drawdowns = []
    final_returns = []
    sharpe_ratios = []
    volatilities = []

    below_benchmark_return_count = 0
    above_benchmark_drawdown_count = 0

    rf_daily = risk_free_rate_annual / 252.0

    for _ in range(n_simulations):
        # Bootstrap i.i.d.
        sampled_returns = np.random.choice(returns, size=n_days, replace=True)

        # Slippage
        sampled_returns = sampled_returns - slippage

        # Shock negativi
        n_shocks = int(n_days * shock_frequency)
        if n_shocks > 0:
            shock_days = np.random.choice(n_days, size=n_shocks, replace=False)
            shocks = np.random.uniform(shock_magnitude[0], shock_magnitude[1], size=n_shocks)
            sampled_returns[shock_days] -= shocks

        # Equity path
        equity = init_value * np.cumprod(1.0 + sampled_returns)
        equity_paths.append(equity)

        # Ritorni giornalieri (identici a sampled_returns per costruzione)
        daily_returns = sampled_returns

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        max_dd = np.max(1.0 - equity / running_max)

        # Total return
        total_return = equity[-1] / init_value - 1.0

        # Volatilità & Sharpe
        vol = np.std(daily_returns, ddof=1)
        excess_daily = daily_returns - rf_daily
        sharpe = np.mean(excess_daily) / (vol + 1e-8) * np.sqrt(252.0)

        drawdowns.append(max_dd)
        final_returns.append(total_return)
        volatilities.append(vol)
        sharpe_ratios.append(sharpe)

        if benchmark_return is not None and total_return < benchmark_return:
            below_benchmark_return_count += 1
        if benchmark_drawdown is not None and max_dd > benchmark_drawdown:
            above_benchmark_drawdown_count += 1

    final_returns = np.array(final_returns)
    drawdowns = np.array(drawdowns)
    sharpe_ratios = np.array(sharpe_ratios)

    results = {
        'mean_final_return': float(np.mean(final_returns)),
        'median_final_return': float(np.median(final_returns)),
        'percentile_5_return': float(np.percentile(final_returns, 5)),
        'percentile_95_return': float(np.percentile(final_returns, 95)),
        'average_max_drawdown': float(np.mean(drawdowns)),
        'average_sharpe_ratio': float(np.mean(sharpe_ratios)),
        'below_benchmark_return_count': int(below_benchmark_return_count),
        'above_benchmark_drawdown_count': int(above_benchmark_drawdown_count),
        'final_returns': final_returns,
        'drawdowns': drawdowns,
        'equity_paths': equity_paths,
    }

    if show_plots:
        # Equity paths
        plt.figure(figsize=(12, 6))
        for i in range(min(50, n_simulations)):
            plt.plot(equity_paths[i], alpha=0.2, linewidth=0.8)
        plt.title('Monte Carlo (Bootstrap i.i.d.) – Equity Curves')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()

        # Distribuzione returns
        plt.figure(figsize=(10, 5))
        sns.histplot(final_returns, bins=50, kde=True)
        plt.axvline(np.percentile(final_returns, 5), color='red', linestyle='--', label='5th Percentile')
        plt.axvline(np.mean(final_returns), color='blue', linestyle='-', label='Mean')
        plt.axvline(np.percentile(final_returns, 95), color='green', linestyle='--', label='95th Percentile')
        if benchmark_return is not None:
            plt.axvline(benchmark_return, color='purple', linestyle='-', label='Benchmark Return')
        plt.title('MC (Bootstrap i.i.d.) – Return Distribution')
        plt.xlabel('Total Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Distribuzione drawdown
        plt.figure(figsize=(10, 5))
        sns.histplot(drawdowns, bins=50, kde=True, color='salmon')
        plt.axvline(np.percentile(drawdowns, 5), color='red', linestyle='--', label='5th Percentile')
        plt.axvline(np.mean(drawdowns), color='blue', linestyle='-', label='Mean')
        plt.axvline(np.percentile(drawdowns, 95), color='green', linestyle='--', label='95th Percentile')
        if benchmark_drawdown is not None:
            plt.axvline(benchmark_drawdown, color='purple', linestyle='-', label='Benchmark Max Drawdown')
        plt.title('MC (Bootstrap i.i.d.) – Max Drawdown Distribution')
        plt.xlabel('Max Drawdown')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.show()

    if show_summary:
        print("\n================= MONTE CARLO SUMMARY – Bootstrap i.i.d. =================\n")
        print(f"Numero di simulazioni                  : {len(final_returns)}")
        print(f"Mean Final Return                      : {results['mean_final_return']:.2%}")
        print(f"Median Final Return                    : {results['median_final_return']:.2%}")
        print(f"5th Percentile Return                  : {results['percentile_5_return']:.2%}")
        print(f"95th Percentile Return                 : {results['percentile_95_return']:.2%}")
        print(f"Average Max Drawdown                   : {results['average_max_drawdown']:.2%}")
        print(f"Average Sharpe Ratio                   : {results['average_sharpe_ratio']:.2f}\n")
        if benchmark_return is not None:
            print(f"Simulazioni con rendimento < benchmark : {results['below_benchmark_return_count']}")
        if benchmark_drawdown is not None:
            print(f"Simulazioni con drawdown   > benchmark : {results['above_benchmark_drawdown_count']}")
        print("\n==========================================================================\n")

    return results

#
# Metodo 2
#    

def monte_carlo_block_bootstrap(portfolio_returns,
                                n_simulations=1000,
                                init_value=100,
                                block_size=10,
                                benchmark_return=None,
                                benchmark_drawdown=None,
                                seed=None,
                                show_plots=True,
                                show_summary=True,
                                slippage=0.0,
                                shock_frequency=0.0,
                                shock_magnitude=(0.05, 0.15),
                                risk_free_rate_annual=0.0):
    """
    Monte Carlo con Block Bootstrap: campiona blocchi contigui di rendimenti
    per preservare parzialmente autocorrelazione e clustering di volatilità.
    Parametri benchmark e logica di conteggio identici a monte_carlo_simulation.
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(portfolio_returns, pd.Series):
        returns = portfolio_returns.dropna().values
    else:
        returns = pd.Series(portfolio_returns).dropna().values

    n_days = len(returns)
    n_blocks = int(np.ceil(n_days / block_size))

    equity_paths = []
    drawdowns = []
    final_returns = []
    sharpe_ratios = []

    below_benchmark_return_count = 0
    above_benchmark_drawdown_count = 0

    rf_daily = risk_free_rate_annual / 252.0

    for _ in range(n_simulations):
        # Costruisci una sequenza di blocchi
        sampled_chunks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n_days - block_size + 1))
            block = returns[start:start + block_size]
            sampled_chunks.append(block)

        sampled_returns = np.concatenate(sampled_chunks)[:n_days]

        # Slippage
        sampled_returns = sampled_returns - slippage

        # Shock negativi
        n_shocks = int(n_days * shock_frequency)
        if n_shocks > 0:
            shock_days = np.random.choice(n_days, size=n_shocks, replace=False)
            shocks = np.random.uniform(shock_magnitude[0], shock_magnitude[1], size=n_shocks)
            sampled_returns[shock_days] -= shocks

        # Equity path
        equity = init_value * np.cumprod(1.0 + sampled_returns)
        equity_paths.append(equity)

        daily_returns = sampled_returns

        running_max = np.maximum.accumulate(equity)
        max_dd = np.max(1.0 - equity / running_max)

        total_return = equity[-1] / init_value - 1.0

        vol = np.std(daily_returns, ddof=1)
        excess_daily = daily_returns - rf_daily
        sharpe = np.mean(excess_daily) / (vol + 1e-8) * np.sqrt(252.0)

        drawdowns.append(max_dd)
        final_returns.append(total_return)
        sharpe_ratios.append(sharpe)

        if benchmark_return is not None and total_return < benchmark_return:
            below_benchmark_return_count += 1
        if benchmark_drawdown is not None and max_dd > benchmark_drawdown:
            above_benchmark_drawdown_count += 1

    final_returns = np.array(final_returns)
    drawdowns = np.array(drawdowns)
    sharpe_ratios = np.array(sharpe_ratios)

    results = {
        'mean_final_return': float(np.mean(final_returns)),
        'median_final_return': float(np.median(final_returns)),
        'percentile_5_return': float(np.percentile(final_returns, 5)),
        'percentile_95_return': float(np.percentile(final_returns, 95)),
        'average_max_drawdown': float(np.mean(drawdowns)),
        'average_sharpe_ratio': float(np.mean(sharpe_ratios)),
        'below_benchmark_return_count': int(below_benchmark_return_count),
        'above_benchmark_drawdown_count': int(above_benchmark_drawdown_count),
        'final_returns': final_returns,
        'drawdowns': drawdowns,
        'equity_paths': equity_paths,
    }

    if show_plots:
        plt.figure(figsize=(12, 6))
        for i in range(min(50, n_simulations)):
            plt.plot(equity_paths[i], alpha=0.2, linewidth=0.8)
        plt.title(f'Monte Carlo Block Bootstrap (block_size={block_size}) – Equity Curves')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.histplot(final_returns, bins=50, kde=True)
        plt.axvline(np.percentile(final_returns, 5), color='red', linestyle='--', label='5th Percentile')
        plt.axvline(np.mean(final_returns), color='blue', linestyle='-', label='Mean')
        plt.axvline(np.percentile(final_returns, 95), color='green', linestyle='--', label='95th Percentile')
        if benchmark_return is not None:
            plt.axvline(benchmark_return, color='purple', linestyle='-', label='Benchmark Return')
        plt.title('MC Block Bootstrap – Return Distribution')
        plt.xlabel('Total Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.histplot(drawdowns, bins=50, kde=True, color='salmon')
        plt.axvline(np.percentile(drawdowns, 5), color='red', linestyle='--', label='5th Percentile')
        plt.axvline(np.mean(drawdowns), color='blue', linestyle='-', label='Mean')
        plt.axvline(np.percentile(drawdowns, 95), color='green', linestyle='--', label='95th Percentile')
        if benchmark_drawdown is not None:
            plt.axvline(benchmark_drawdown, color='purple', linestyle='-', label='Benchmark Max Drawdown')
        plt.title('MC Block Bootstrap – Max Drawdown Distribution')
        plt.xlabel('Max Drawdown')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.show()

    if show_summary:
        print(f"\n================= MONTE CARLO SUMMARY – Block Bootstrap (block_size={block_size}) =================\n")
        print(f"Numero di simulazioni                  : {len(final_returns)}")
        print(f"Mean Final Return                      : {results['mean_final_return']:.2%}")
        print(f"Median Final Return                    : {results['median_final_return']:.2%}")
        print(f"5th Percentile Return                  : {results['percentile_5_return']:.2%}")
        print(f"95th Percentile Return                 : {results['percentile_95_return']:.2%}")
        print(f"Average Max Drawdown                   : {results['average_max_drawdown']:.2%}")
        print(f"Average Sharpe Ratio                   : {results['average_sharpe_ratio']:.2f}\n")
        if benchmark_return is not None:
            print(f"Simulazioni con rendimento < benchmark : {results['below_benchmark_return_count']}")
        if benchmark_drawdown is not None:
            print(f"Simulazioni con drawdown   > benchmark : {results['above_benchmark_drawdown_count']}")
        print("\n===================================================================================================\n")

    return results
#
# Metodo 3
#
def _classify_regimes(returns, window=20, low_q=0.5, high_q=0.8):
    """
    Classifica i giorni in due regimi di volatilità:
      0 = low/normal vol
      1 = high vol
    in base alla rolling std.
    """
    s = pd.Series(returns)
    rolling_vol = s.rolling(window).std()

    # Soglie
    thresh_low = rolling_vol.quantile(low_q)
    thresh_high = rolling_vol.quantile(high_q)

    regimes = np.zeros(len(returns), dtype=int)  # default 0
    regimes[rolling_vol >= thresh_high] = 1

    # I primi 'window' giorni hanno NaN rolling_vol: tienili in regime 0
    return regimes


def _estimate_transition_matrix(regimes, n_states=2):
    """
    Stima la matrice di transizione Markoviana NxN dai regimi storici.
    Se una riga è vuota, usa [0.5, 0.5] per i 2 stati.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(1, len(regimes)):
        prev_state = regimes[i - 1]
        curr_state = regimes[i]
        counts[prev_state, curr_state] += 1

    P = np.zeros_like(counts)
    for i in range(n_states):
        row_sum = counts[i].sum()
        if row_sum > 0:
            P[i] = counts[i] / row_sum
        else:
            P[i] = np.ones(n_states) / n_states  # distribuzione uniforme

    return P


def monte_carlo_regime_switching(portfolio_returns,
                                 n_simulations=1000,
                                 init_value=100,
                                 regime_window=20,
                                 low_q=0.5,
                                 high_q=0.8,
                                 benchmark_return=None,
                                 benchmark_drawdown=None,
                                 seed=None,
                                 show_plots=True,
                                 show_summary=True,
                                 slippage=0.0,
                                 shock_frequency=0.0,
                                 shock_magnitude=(0.05, 0.15),
                                 risk_free_rate_annual=0.0):
    """
    Monte Carlo con modello a regimi (2 stati: low vol / high vol):
      - classifica i giorni storici in regimi di volatilità
      - stima la matrice di transizione
      - genera una catena di regimi e campiona i rendimenti per regime.

    Confronto con benchmark identico alle altre funzioni.
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(portfolio_returns, pd.Series):
        returns = portfolio_returns.dropna().values
    else:
        returns = pd.Series(portfolio_returns).dropna().values

    n_days = len(returns)

    # Classifica regimi e stima matrice di transizione
    regimes_hist = _classify_regimes(returns, window=regime_window, low_q=low_q, high_q=high_q)
    P = _estimate_transition_matrix(regimes_hist, n_states=2)

    # Pool di rendimenti per ciascun stato
    r_state0 = returns[regimes_hist == 0]
    r_state1 = returns[regimes_hist == 1]

    # Fallback se uno dei due è vuoto
    if len(r_state0) == 0:
        r_state0 = returns
    if len(r_state1) == 0:
        r_state1 = returns

    equity_paths = []
    drawdowns = []
    final_returns = []
    sharpe_ratios = []

    below_benchmark_return_count = 0
    above_benchmark_drawdown_count = 0

    rf_daily = risk_free_rate_annual / 252.0

    for _ in range(n_simulations):
        # Simula catena di regimi
        sim_regimes = np.zeros(n_days, dtype=int)
        sim_regimes[0] = np.random.choice([0, 1])  # stato iniziale casuale

        for t in range(1, n_days):
            prev = sim_regimes[t - 1]
            sim_regimes[t] = np.random.choice([0, 1], p=P[prev])

        # Campiona i rendimenti per stato
        sampled_returns = np.empty(n_days)
        idx0 = np.where(sim_regimes == 0)[0]
        idx1 = np.where(sim_regimes == 1)[0]

        if len(idx0) > 0:
            sampled_returns[idx0] = np.random.choice(r_state0, size=len(idx0), replace=True)
        if len(idx1) > 0:
            sampled_returns[idx1] = np.random.choice(r_state1, size=len(idx1), replace=True)

        # Slippage
        sampled_returns = sampled_returns - slippage

        # Shock negativi
        n_shocks = int(n_days * shock_frequency)
        if n_shocks > 0:
            shock_days = np.random.choice(n_days, size=n_shocks, replace=False)
            shocks = np.random.uniform(shock_magnitude[0], shock_magnitude[1], size=n_shocks)
            sampled_returns[shock_days] -= shocks

        # Equity path
        equity = init_value * np.cumprod(1.0 + sampled_returns)
        equity_paths.append(equity)

        daily_returns = sampled_returns

        running_max = np.maximum.accumulate(equity)
        max_dd = np.max(1.0 - equity / running_max)

        total_return = equity[-1] / init_value - 1.0

        vol = np.std(daily_returns, ddof=1)
        excess_daily = daily_returns - rf_daily
        sharpe = np.mean(excess_daily) / (vol + 1e-8) * np.sqrt(252.0)

        drawdowns.append(max_dd)
        final_returns.append(total_return)
        sharpe_ratios.append(sharpe)

        if benchmark_return is not None and total_return < benchmark_return:
            below_benchmark_return_count += 1
        if benchmark_drawdown is not None and max_dd > benchmark_drawdown:
            above_benchmark_drawdown_count += 1

    final_returns = np.array(final_returns)
    drawdowns = np.array(drawdowns)
    sharpe_ratios = np.array(sharpe_ratios)

    results = {
        'mean_final_return': float(np.mean(final_returns)),
        'median_final_return': float(np.median(final_returns)),
        'percentile_5_return': float(np.percentile(final_returns, 5)),
        'percentile_95_return': float(np.percentile(final_returns, 95)),
        'average_max_drawdown': float(np.mean(drawdowns)),
        'average_sharpe_ratio': float(np.mean(sharpe_ratios)),
        'below_benchmark_return_count': int(below_benchmark_return_count),
        'above_benchmark_drawdown_count': int(above_benchmark_drawdown_count),
        'final_returns': final_returns,
        'drawdowns': drawdowns,
        'equity_paths': equity_paths,
        'transition_matrix': P,
    }

    if show_plots:
        plt.figure(figsize=(12, 6))
        for i in range(min(50, n_simulations)):
            plt.plot(equity_paths[i], alpha=0.2, linewidth=0.8)
        plt.title('Monte Carlo Regime Switching – Equity Curves')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.histplot(final_returns, bins=50, kde=True)
        plt.axvline(np.percentile(final_returns, 5), color='red', linestyle='--', label='5th Percentile')
        plt.axvline(np.mean(final_returns), color='blue', linestyle='-', label='Mean')
        plt.axvline(np.percentile(final_returns, 95), color='green', linestyle='--', label='95th Percentile')
        if benchmark_return is not None:
            plt.axvline(benchmark_return, color='purple', linestyle='-', label='Benchmark Return')
        plt.title('MC Regime Switching – Return Distribution')
        plt.xlabel('Total Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.histplot(drawdowns, bins=50, kde=True, color='salmon')
        plt.axvline(np.percentile(drawdowns, 5), color='red', linestyle='--', label='5th Percentile')
        plt.axvline(np.mean(drawdowns), color='blue', linestyle='-', label='Mean')
        plt.axvline(np.percentile(drawdowns, 95), color='green', linestyle='--', label='95th Percentile')
        if benchmark_drawdown is not None:
            plt.axvline(benchmark_drawdown, color='purple', linestyle='-', label='Benchmark Max Drawdown')
        plt.title('MC Regime Switching – Max Drawdown Distribution')
        plt.xlabel('Max Drawdown')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.show()

    if show_summary:
        print("\n================= MONTE CARLO SUMMARY – Regime Switching =================\n")
        print(f"Numero di simulazioni                  : {len(final_returns)}")
        print(f"Mean Final Return                      : {results['mean_final_return']:.2%}")
        print(f"Median Final Return                    : {results['median_final_return']:.2%}")
        print(f"5th Percentile Return                  : {results['percentile_5_return']:.2%}")
        print(f"95th Percentile Return                 : {results['percentile_95_return']:.2%}")
        print(f"Average Max Drawdown                   : {results['average_max_drawdown']:.2%}")
        print(f"Average Sharpe Ratio                   : {results['average_sharpe_ratio']:.2f}\n")
        if benchmark_return is not None:
            print(f"Simulazioni con rendimento < benchmark : {results['below_benchmark_return_count']}")
        if benchmark_drawdown is not None:
            print(f"Simulazioni con drawdown   > benchmark : {results['above_benchmark_drawdown_count']}")
        print("\n============================================================================\n")

    return results

#
# Wrapper: confronto di tutti i metodi vs benchmark (tabella + grafici)
#

def run_all_mc_methods(portfolio_returns,
                       init_value,
                       benchmark_return,
                       benchmark_drawdown,
                       n_simulations=10000,
                       seed=42,
                       show_method_plots=False,
                       show_method_summaries=False,
                       block_size=10,
                       regime_window=20):
    """
    Esegue le tre varianti di Monte Carlo:
      - Bootstrap i.i.d.
      - Block Bootstrap
      - Regime Switching

    e restituisce:
      - dizionario dei risultati per metodo
      - DataFrame riassuntivo con metriche chiave
      - alcuni grafici di confronto (scatter rischio/rendimento + boxplot returns)
    """

    methods_results = {}

    # 1) Bootstrap i.i.d.
    res_basic = monte_carlo_simulation(
        portfolio_returns=portfolio_returns,
        n_simulations=n_simulations,
        init_value=init_value,
        benchmark_return=benchmark_return,
        benchmark_drawdown=benchmark_drawdown,
        seed=seed,
        show_plots=show_method_plots,
        show_summary=show_method_summaries,
    )
    methods_results["Bootstrap i.i.d."] = res_basic

    # 2) Block Bootstrap
    res_block = monte_carlo_block_bootstrap(
        portfolio_returns=portfolio_returns,
        n_simulations=n_simulations,
        init_value=init_value,
        block_size=block_size,
        benchmark_return=benchmark_return,
        benchmark_drawdown=benchmark_drawdown,
        seed=seed,
        show_plots=show_method_plots,
        show_summary=show_method_summaries,
    )
    methods_results[f"Block Bootstrap ({block_size}d)"] = res_block

    # 3) Regime Switching
    res_regime = monte_carlo_regime_switching(
        portfolio_returns=portfolio_returns,
        n_simulations=n_simulations,
        init_value=init_value,
        regime_window=regime_window,
        benchmark_return=benchmark_return,
        benchmark_drawdown=benchmark_drawdown,
        seed=seed,
        show_plots=show_method_plots,
        show_summary=show_method_summaries,
    )
    methods_results[f"Regime Switching (win={regime_window})"] = res_regime

    # --- DataFrame riassuntivo ---
    rows = []
    for name, res in methods_results.items():
        n_sim = len(res["final_returns"])
        rows.append({
            "Method": name,
            "Mean Final Return": res["mean_final_return"],
            "Median Final Return": res["median_final_return"],
            "5% Return": res["percentile_5_return"],
            "95% Return": res["percentile_95_return"],
            "Avg Max DD": res["average_max_drawdown"],
            "Avg Sharpe": res["average_sharpe_ratio"],
            "Below Bench Ret (%)": (
                res["below_benchmark_return_count"] / n_sim * 100.0
                if benchmark_return is not None else np.nan
            ),
            "Above Bench DD (%)": (
                res["above_benchmark_drawdown_count"] / n_sim * 100.0
                if benchmark_drawdown is not None else np.nan
            ),
        })

    summary_df = pd.DataFrame(rows).set_index("Method")


    plot_mc_method_summary(
        methods_results=methods_results,
        summary_df=summary_df,
        benchmark_return=benchmark_return,
        benchmark_drawdown=benchmark_drawdown,
        title_prefix="TS Robustness – Monte Carlo",
        method_order=[
            "Bootstrap i.i.d.",
            "Block Bootstrap (10d)",
            "Regime Switching (win=20)"
        ],
        show_fliers=False,
        show_cdf=True
    )

    return methods_results, summary_df


def _cdf_at_value(sorted_x: np.ndarray, value: float) -> float:
    if sorted_x.size == 0 or value is None or not np.isfinite(value):
        return np.nan
    return float(np.searchsorted(sorted_x, value, side="right") / sorted_x.size)


def _annotate_point(ax, x, y, text, dy, dx=0.0):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", lw=0.6),
    )


def plot_mc_method_summary(
    methods_results: dict,
    summary_df,
    benchmark_return: float | None = None,
    benchmark_drawdown: float | None = None,
    title_prefix: str = "Monte Carlo",
    method_order: list[str] | None = None,
    show_fliers: bool = False,
    show_cdf: bool = True,
    annotate_cdf: bool = True,
    # Quantili da evidenziare
    ret_tail_q: float = 5.0,   # evidenzia q5 e q95 sui returns
    dd_tail_q: float = 95.0,   # evidenzia q95 sui drawdown
    # Layout annotazioni
    dy_prob: float = 0.05,
    annotate_q95_in_side_panel: bool = True,
    # Stampa tabella valori (gli stessi mostrati/annotati nei grafici)
    show_plot_summary_df: bool = True,
    plot_summary_title: str | None = None,
):
    """
    Diagnostica completa Monte Carlo (uniformata + leggibile) con stampa tabella
    riassuntiva dei valori effettivamente riportati sui grafici.

    Include:
      1) Scatter rischio/rendimento (summary_df)
      2) Boxplot Total Return + benchmark_return
      3) Boxplot Max Drawdown + benchmark_drawdown
      4) CDF Total Return + quantili + P_under
      5) CDF Max Drawdown + quantile + P_worseDD
      6) (opzionale) stampa DataFrame con valori "da grafico" via my_display/display

    methods_results[method] deve contenere:
      - "final_returns": array-like
      - "drawdowns": array-like
    summary_df (opzionale) deve avere index=method e colonne:
      - "Avg Max DD"
      - "Mean Final Return"
    """

    # -----------------------------
    # Determina ordine metodi
    # -----------------------------
    if method_order is None:
        method_order = list(methods_results.keys())
    else:
        method_order = [m for m in method_order if m in methods_results]

    if len(method_order) == 0:
        raise ValueError("plot_mc_method_summary: method_order vuoto o methods_results vuoto.")

    # -----------------------------
    # Helper: pulizia array
    # -----------------------------
    def _clean_arr(a):
        a = np.asarray(a, dtype=float)
        return a[np.isfinite(a)]

    # ============================================================
    # 0) Costruisci DataFrame riassuntivo "valori su grafico"
    # ============================================================
    rows = []
    for method in method_order:
        res = methods_results[method]
        rets = _clean_arr(res.get("final_returns", []))
        dds  = _clean_arr(res.get("drawdowns", []))

        rets_sorted = np.sort(rets) if rets.size else rets
        dds_sorted  = np.sort(dds) if dds.size else dds

        ret_q_low  = float(np.percentile(rets, ret_tail_q)) if rets.size else np.nan
        ret_q_high = float(np.percentile(rets, 100 - ret_tail_q)) if rets.size else np.nan
        dd_q_high  = float(np.percentile(dds, dd_tail_q)) if dds.size else np.nan

        p_under = (_cdf_at_value(rets_sorted, benchmark_return)
                   if benchmark_return is not None and rets_sorted.size else np.nan)

        if benchmark_drawdown is not None and dds_sorted.size:
            cdf_b = _cdf_at_value(dds_sorted, benchmark_drawdown)
            p_worse_dd = float(1.0 - cdf_b) if np.isfinite(cdf_b) else np.nan
        else:
            p_worse_dd = np.nan

        rows.append({
            "Method": method,
            "N": int(rets.size) if rets.size else int(dds.size) if dds.size else 0,
            f"Return_q{int(ret_tail_q)}": ret_q_low,
            f"Return_q{int(100-ret_tail_q)}": ret_q_high,
            "P_under(Return<=Bench)": p_under,
            f"DD_q{int(dd_tail_q)}": dd_q_high,
            "P_worseDD(DD>Bench)": p_worse_dd,
            "BenchReturn": benchmark_return,
            "BenchMaxDD": benchmark_drawdown,
        })

    # plot_summary_df = pd.DataFrame(rows).set_index("Method")
    
    plot_summary_df = pd.DataFrame(rows).set_index("Method")
    
    # --- Uniforma formato: valori in percentuale come nei grafici (3 decimali) ---
    pct_cols = [
        f"Return_q{int(ret_tail_q)}",
        f"Return_q{int(100-ret_tail_q)}",
        "P_under(Return<=Bench)",
        f"DD_q{int(dd_tail_q)}",
        "P_worseDD(DD>Bench)",
        "BenchReturn",
        "BenchMaxDD",
    ]
    for c in pct_cols:
        if c in plot_summary_df.columns:
            # plot_summary_df[c] = f"{(plot_summary_df[c] * 100).round(3)}%"
            plot_summary_df[c] = (plot_summary_df[c] * 100).round(3)
            # print(f"{plot_summary_df[c]}%") 

        # --- Aggiungi simbolo % ai nomi delle colonne percentuali ---
        rename_map = {c: f"{c} (%)" for c in pct_cols if c in plot_summary_df.columns}
        plot_summary_df.rename(columns=rename_map, inplace=True)

    # # stampa tabella (prima dei grafici, così ce l'hai subito disponibile)
    # if show_plot_summary_df:
    #     title = plot_summary_title or f"{title_prefix} – Plot Summary (valori su grafico)"
    #     my_display(plot_summary_df, title=title)  # usa la tua funzione se esiste

    # ============================================================
    # 1) Scatter rischio/rendimento
    # ============================================================
    if summary_df is not None and hasattr(summary_df, "empty") and not summary_df.empty:
        plt.figure(figsize=(8, 5))

        for method in method_order:
            if method not in summary_df.index:
                continue
            row = summary_df.loc[method]
            x = row.get("Avg Max DD", np.nan)
            y = row.get("Mean Final Return", np.nan)
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            plt.scatter(x, y)
            plt.text(x, y, method, fontsize=9)

        if benchmark_drawdown is not None and np.isfinite(benchmark_drawdown):
            plt.axvline(benchmark_drawdown, color="purple", linestyle="--", label="Benchmark Max DD")
        if benchmark_return is not None and np.isfinite(benchmark_return):
            plt.axhline(benchmark_return, color="green", linestyle="--", label="Benchmark Return")

        plt.xlabel("Average Max Drawdown")
        plt.ylabel("Mean Final Return")
        plt.title(f"{title_prefix} – Rischio vs Rendimento")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ============================================================
    # 2) Boxplot Total Return
    # ============================================================
    plt.figure(figsize=(10, 6))
    ret_data, ret_labels = [], []

    for method in method_order:
        arr = _clean_arr(methods_results[method].get("final_returns", []))
        if arr.size == 0:
            continue
        ret_data.append(arr)
        ret_labels.append(method)

    plt.boxplot(ret_data, labels=ret_labels, showfliers=show_fliers)

    if benchmark_return is not None and np.isfinite(benchmark_return):
        plt.axhline(benchmark_return, color="green", linestyle="--", label="Benchmark Return")

    plt.title(f"{title_prefix} – Distribuzione Total Return")
    plt.ylabel("Total Return")
    plt.grid(True, axis="y")
    plt.xticks(rotation=20)
    plt.legend()
    plt.show()

    # ============================================================
    # 3) Boxplot Max Drawdown
    # ============================================================
    plt.figure(figsize=(10, 6))
    dd_data, dd_labels = [], []

    for method in method_order:
        arr = _clean_arr(methods_results[method].get("drawdowns", []))
        if arr.size == 0:
            continue
        dd_data.append(arr)
        dd_labels.append(method)

    plt.boxplot(dd_data, labels=dd_labels, showfliers=show_fliers)

    if benchmark_drawdown is not None and np.isfinite(benchmark_drawdown):
        plt.axhline(benchmark_drawdown, color="purple", linestyle="--", label="Benchmark Max DD")

    plt.title(f"{title_prefix} – Distribuzione Max Drawdown")
    plt.ylabel("Max Drawdown")
    plt.grid(True, axis="y")
    plt.xticks(rotation=20)
    plt.legend()
    plt.show()

    if not show_cdf:
        return plot_summary_df

    # ============================================================
    # 4) CDF Total Return (quantili + P_under)
    # ============================================================
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    dy_levels = np.linspace(+dy_prob, -dy_prob, max(3, len(method_order)))
    side_y_start = 0.88
    side_step = 0.06
    side_y = side_y_start

    for i, method in enumerate(method_order):
        rets = _clean_arr(methods_results[method].get("final_returns", []))
        if rets.size == 0:
            continue

        x = np.sort(rets)
        y = np.linspace(0, 1, x.size, endpoint=True)
        plt.plot(x, y, label=method)

        if annotate_cdf:
            q_low = float(np.percentile(x, ret_tail_q))
            q_high = float(np.percentile(x, 100 - ret_tail_q))
            y_low = _cdf_at_value(x, q_low)
            y_high = _cdf_at_value(x, q_high)

            plt.scatter([q_low], [y_low], s=18, marker="o")
            plt.scatter([q_high], [y_high], s=18, marker="o")

            dy = float(dy_levels[min(i, len(dy_levels) - 1)])

            # q5: label vicino al punto (leader line)
            _annotate_point(ax, q_low, y_low, f"{method} q{ret_tail_q:.0f}={q_low:.1%}", dy=dy)

            # q95: meglio side panel
            if annotate_q95_in_side_panel:
                plt.text(
                    0.98, side_y,
                    f"{method}: q{100-ret_tail_q:.0f}={q_high:.1%}",
                    transform=ax.transAxes, ha="right", fontsize=8
                )
                side_y -= side_step
            else:
                _annotate_point(ax, q_high, y_high, f"{method} q{100-ret_tail_q:.0f}={q_high:.1%}", dy=dy * 1.2)

            # P_under: side panel
            if benchmark_return is not None and np.isfinite(benchmark_return):
                p_under = _cdf_at_value(x, benchmark_return)
                plt.text(
                    0.98, side_y,
                    f"{method}: P(Ret≤Bench)={p_under:.1%}",
                    transform=ax.transAxes, ha="right", fontsize=8
                )
                side_y -= side_step

    if benchmark_return is not None and np.isfinite(benchmark_return):
        plt.axvline(benchmark_return, color="green", linestyle="--", label="Benchmark Return")

    plt.title(f"{title_prefix} – CDF Total Return (quantili + P_under)")
    plt.xlabel("Total Return")
    plt.ylabel("Probability  P(Return ≤ x)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ============================================================
    # 5) CDF Max Drawdown (quantile + P_worseDD)
    # ============================================================
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    dy_levels = np.linspace(+dy_prob, -dy_prob, max(3, len(method_order)))
    side_y = side_y_start

    for i, method in enumerate(method_order):
        dds = _clean_arr(methods_results[method].get("drawdowns", []))
        if dds.size == 0:
            continue

        x = np.sort(dds)
        y = np.linspace(0, 1, x.size, endpoint=True)
        plt.plot(x, y, label=method)

        if annotate_cdf:
            q_high = float(np.percentile(x, dd_tail_q))
            y_high = _cdf_at_value(x, q_high)
            plt.scatter([q_high], [y_high], s=18, marker="o")

            dy = float(dy_levels[min(i, len(dy_levels) - 1)])

            # q95 DD: spesso in alto => side panel
            if annotate_q95_in_side_panel:
                plt.text(
                    0.98, side_y,
                    f"{method}: q{dd_tail_q:.0f}={q_high:.1%}",
                    transform=ax.transAxes, ha="right", fontsize=8
                )
                side_y -= side_step
            else:
                _annotate_point(ax, q_high, y_high, f"{method} q{dd_tail_q:.0f}={q_high:.1%}", dy=dy)

            # P_worseDD: side panel
            if benchmark_drawdown is not None and np.isfinite(benchmark_drawdown):
                cdf_b = _cdf_at_value(x, benchmark_drawdown)
                p_worse = 1.0 - cdf_b
                plt.text(
                    0.98, side_y,
                    f"{method}: P(DD>Bench)={p_worse:.1%}",
                    transform=ax.transAxes, ha="right", fontsize=8
                )
                side_y -= side_step

    if benchmark_drawdown is not None and np.isfinite(benchmark_drawdown):
        plt.axvline(benchmark_drawdown, color="purple", linestyle="--", label="Benchmark Max DD")

    plt.title(f"{title_prefix} – CDF Max Drawdown (quantile + P_worseDD)")
    plt.xlabel("Max Drawdown")
    plt.ylabel("Probability  P(DD ≤ x)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ============================================================
    # 6) Bad probabilities bar
    # ============================================================
    labels = []
    p_under_list = []
    p_worse_dd_list = []

    for name, res in methods_results.items():
        labels.append(name)

        if benchmark_return is not None:
            p_under = np.mean(np.asarray(res["final_returns"]) < benchmark_return)
        else:
            p_under = np.nan

        if benchmark_drawdown is not None:
            p_worse_dd = np.mean(np.asarray(res["drawdowns"]) > benchmark_drawdown)
        else:
            p_worse_dd = np.nan

        p_under_list.append(p_under)
        p_worse_dd_list.append(p_worse_dd)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, p_under_list, width, label="P(Return < Benchmark)")
    plt.bar(x + width/2, p_worse_dd_list, width, label="P(DD > BenchmarkDD)")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1)
    plt.title(f"{title_prefix} – Probabilità eventi sfavorevoli vs Benchmark")
    plt.ylabel("Probability")
    plt.grid(True, axis="y")
    plt.legend()
    plt.show()

    # stampa tabella riassuntiva alla fine dei grafici CDF
    if show_plot_summary_df:
        title = plot_summary_title or f"{title_prefix} – Plot Summary (valori su grafico)"
        my_display(plot_summary_df, title=title)  # usa la tua funzione se esiste

    return plot_summary_df
    

#
# Valutazione risultati
#
def compare_mc_methods(methods_results: dict,
                       benchmark_return: float | None,
                       benchmark_drawdown: float | None,
                       ret_quantiles=(1, 5, 10, 50, 90, 95, 99),
                       dd_quantiles=(1, 5, 10, 50, 90, 95, 99)) -> pd.DataFrame:
    """
    Tabella comparativa metodi MC con metriche vs benchmark.

    Aggiunte chiave:
      - P_dom  = P(Return > BenchReturn AND DD < BenchDD)
      - P_fail = P(Return < BenchReturn AND DD > BenchDD)
      - P_mixed1 = P(Return > BenchReturn AND DD > BenchDD)  (rendo meglio ma rischio peggio)
      - P_mixed2 = P(Return < BenchReturn AND DD < BenchDD)  (rendo peggio ma rischio meglio)
    """

    rows = []
    for name, res in methods_results.items():
        rets = np.asarray(res["final_returns"])
        dds  = np.asarray(res["drawdowns"])
        n    = len(rets)

        rq = {f"Ret_q{q:02d}": float(np.percentile(rets, q)) for q in ret_quantiles}
        dq = {f"DD_q{q:02d}": float(np.percentile(dds, q)) for q in dd_quantiles}

        # Probabilità semplici vs benchmark
        p_under = float(np.mean(rets < benchmark_return)) if benchmark_return is not None else np.nan
        p_worse_dd = float(np.mean(dds > benchmark_drawdown)) if benchmark_drawdown is not None else np.nan

        # Probabilità congiunte (solo se entrambi i benchmark sono disponibili)
        if benchmark_return is not None and benchmark_drawdown is not None:
            dom  = (rets > benchmark_return) & (dds < benchmark_drawdown)
            fail = (rets < benchmark_return) & (dds > benchmark_drawdown)
            mixed1 = (rets > benchmark_return) & (dds > benchmark_drawdown)
            mixed2 = (rets < benchmark_return) & (dds < benchmark_drawdown)

            p_dom   = float(np.mean(dom))
            p_fail  = float(np.mean(fail))
            p_mix1  = float(np.mean(mixed1))
            p_mix2  = float(np.mean(mixed2))
        else:
            p_dom = p_fail = p_mix1 = p_mix2 = np.nan

        tail_ret_5 = float(np.percentile(rets, 5))
        tail_dd_95 = float(np.percentile(dds, 95))

        rows.append({
            "Method": name,
            "N": n,
            "MeanRet": float(np.mean(rets)),
            "MedianRet": float(np.median(rets)),
            "MeanDD": float(np.mean(dds)),
            "MedianDD": float(np.median(dds)),
            "SharpeAvg": float(res.get("average_sharpe_ratio", np.nan)),

            "P(Return < Bench)": p_under,
            "P(DD > BenchDD)": p_worse_dd,

            "P_dom": p_dom,
            "P_fail": p_fail,
            "P_mixed_ret_up_dd_up": p_mix1,
            "P_mixed_ret_down_dd_down": p_mix2,

            "TailRet_5%": tail_ret_5,
            "TailDD_95%": tail_dd_95,

            **rq,
            **dq
        })

    df = pd.DataFrame(rows).set_index("Method").sort_values("MeanRet", ascending=False)
    return df
#
# Decisionale
#

def mc_deploy_recommendation(mc_compare_df: pd.DataFrame,
                             benchmark_return: float | None,
                             benchmark_drawdown: float | None,
                             rules: dict | None = None,
                             portfolio_mode: bool = True,
                             portfolio_ts_count: int = 10,
                             expected_weight: float | None = None) -> dict:
    """
    Decisione DEPLOY / NO-DEPLOY basata su confronto multi-metodo MC.

    Usa worst-case across methods per robustezza.

    portfolio_mode:
      - se True, applica solo un aggiustamento esplicito alle soglie tail
        in base a portfolio_ts_count e expected_weight (se fornito).
      - NON allenta P_fail: un TS che spesso perde su entrambe vs benchmark è tossico anche in portfolio.
    """

    if mc_compare_df is None or mc_compare_df.empty:
        return {"decision": "INSUFFICIENT_DATA", "score": 0.0,
                "reasons": ["MC comparison table is empty."], "rule_checks": {}}

    # Regole base (standalone)
    base_rules = {
        # criteri congiunti “deploy-grade”
        "min_p_dom": 0.55,      # almeno 55% dei mondi: Return>Bench & DD<Bench
        "max_p_fail": 0.20,     # al massimo 20%: Return<Bench & DD>Bench

        # guardrail su code (assoluti)
        "min_tail_return_5": -0.20,  # 5° percentile return >= -20%
        "max_tail_dd_95": 0.60,      # 95° percentile DD <= 60%

        # qualità minima
        "min_avg_sharpe": 0.20,

        # coerenza tra metodi
        "max_tail_return_5_spread": 0.15,
    }

    if rules is not None:
        base_rules.update(rules)

    r = base_rules

    has_joint = (benchmark_return is not None and benchmark_drawdown is not None
                 and "P_dom" in mc_compare_df.columns and "P_fail" in mc_compare_df.columns)

    reasons = []
    checks = {}

    # --- Portfolio-aware adjustment (esplicito, non euristico) ---
    # Se portfolio_mode e expected_weight è piccolo (es. ~1/10), possiamo permetterci
    # una soglia tail_return_5 un po’ meno severa, perché l’impatto sul portafoglio è attenuato.
    # Ma NON tocchiamo max_p_fail.
    adj_min_tail_return_5 = r["min_tail_return_5"]
    if portfolio_mode:
        # default weight se non fornito: 1/portfolio_ts_count
        w = (1.0 / max(1, portfolio_ts_count)) if expected_weight is None else float(expected_weight)
        w = max(0.0, min(1.0, w))

        # Allentamento limitato e trasparente: max 10 punti percentuali
        # Esempio: con w=0.10 -> +0.05 (da -0.20 a -0.25) oppure viceversa?
        # Nota: soglia è MIN (più alta è più severa). Allentare significa abbassarla (più negativa).
        max_relax = 0.10  # 10%
        relax = max_relax * (1.0 - min(1.0, w * portfolio_ts_count))  # se w~1/N => relax ~0; se w molto piccolo => relax positivo
        # Per evitare “magie”: se w è circa 1/N non cambia nulla.
        # Se w è molto più piccolo di 1/N (tiny allocation), concedo un piccolo relax.
        adj_min_tail_return_5 = r["min_tail_return_5"] - relax

    checks["portfolio_tail_threshold"] = {"base": r["min_tail_return_5"], "adjusted": adj_min_tail_return_5,
                                         "portfolio_mode": portfolio_mode,
                                         "portfolio_ts_count": portfolio_ts_count,
                                         "expected_weight": expected_weight}

    # --- 1) P_dom worst-case (min across methods)
    if has_joint:
        worst_p_dom = float(mc_compare_df["P_dom"].min())
        ok = worst_p_dom >= r["min_p_dom"]
        checks["P_dom_worst"] = {"value": worst_p_dom, "thr": r["min_p_dom"], "ok": ok}
        if not ok:
            reasons.append(f"Dominanza vs benchmark insufficiente: worst P_dom={worst_p_dom:.1%} < {r['min_p_dom']:.1%}.")
    else:
        checks["P_dom_worst"] = {"value": None, "thr": r["min_p_dom"], "ok": True}

    # --- 2) P_fail worst-case (max across methods)
    if has_joint:
        worst_p_fail = float(mc_compare_df["P_fail"].max())
        ok = worst_p_fail <= r["max_p_fail"]
        checks["P_fail_worst"] = {"value": worst_p_fail, "thr": r["max_p_fail"], "ok": ok}
        if not ok:
            reasons.append(f"Fail congiunto troppo frequente: worst P_fail={worst_p_fail:.1%} > {r['max_p_fail']:.1%}.")
    else:
        checks["P_fail_worst"] = {"value": None, "thr": r["max_p_fail"], "ok": True}

    # --- 3) TailRet_5% worst-case
    worst_tail_ret5 = float(mc_compare_df["TailRet_5%"].min())
    ok = worst_tail_ret5 >= adj_min_tail_return_5
    checks["tail_return_5_worst"] = {"value": worst_tail_ret5, "thr": adj_min_tail_return_5, "ok": ok}
    if not ok:
        reasons.append(f"Coda sinistra troppo pesante: worst TailRet_5%={worst_tail_ret5:.2%} < {adj_min_tail_return_5:.2%}.")

    # --- 4) TailDD_95% worst-case
    worst_tail_dd95 = float(mc_compare_df["TailDD_95%"].max())
    ok = worst_tail_dd95 <= r["max_tail_dd_95"]
    checks["tail_dd_95_worst"] = {"value": worst_tail_dd95, "thr": r["max_tail_dd_95"], "ok": ok}
    if not ok:
        reasons.append(f"Coda drawdown troppo severa: worst TailDD_95%={worst_tail_dd95:.2%} > {r['max_tail_dd_95']:.2%}.")

    # --- 5) Sharpe worst-case
    if "SharpeAvg" in mc_compare_df.columns:
        worst_sharpe = float(mc_compare_df["SharpeAvg"].min())
        ok = worst_sharpe >= r["min_avg_sharpe"]
        checks["avg_sharpe_worst"] = {"value": worst_sharpe, "thr": r["min_avg_sharpe"], "ok": ok}
        if not ok:
            reasons.append(f"Sharpe medio insufficiente: worst SharpeAvg={worst_sharpe:.2f} < {r['min_avg_sharpe']:.2f}.")
    else:
        checks["avg_sharpe_worst"] = {"value": None, "thr": r["min_avg_sharpe"], "ok": True}

    # --- 6) Spread TailRet_5% tra metodi
    tail_spread = float(mc_compare_df["TailRet_5%"].max() - mc_compare_df["TailRet_5%"].min())
    ok = tail_spread <= r["max_tail_return_5_spread"]
    checks["tail_return_5_spread"] = {"value": tail_spread, "thr": r["max_tail_return_5_spread"], "ok": ok}
    if not ok:
        reasons.append(f"Incoerenza tra modelli MC: spread TailRet_5%={tail_spread:.2%} > {r['max_tail_return_5_spread']:.2%}.")

    # Decisione: se mancano benchmark congiunti, degradare la decisione
    if not has_joint:
        # senza P_dom/P_fail non puoi applicare la policy “deploy-grade”
        decision = "INSUFFICIENT_DATA"
        score = 0.0
        reasons = ["Benchmark congiunto non disponibile: impossibile valutare P_dom/P_fail in modo deterministico."] + reasons
    else:
        all_ok = all(v.get("ok", True) for v in checks.values() if isinstance(v, dict) and "ok" in v)
        n_rules = sum(1 for v in checks.values() if isinstance(v, dict) and "ok" in v)
        n_viol = sum(1 for v in checks.values() if isinstance(v, dict) and v.get("ok") is False)

        score = max(0.0, 100.0 * (1.0 - n_viol / max(1, n_rules)))
        decision = "DEPLOY" if all_ok else "NO-DEPLOY"

        if decision == "DEPLOY":
            reasons = [
                "Robustezza Monte Carlo soddisfacente su tutti i modelli considerati (criterio worst-case).",
                f"Worst P_dom={float(mc_compare_df['P_dom'].min()):.1%}, Worst P_fail={float(mc_compare_df['P_fail'].max()):.1%}.",
                f"Worst TailRet_5%={worst_tail_ret5:.2%}, Worst TailDD_95%={worst_tail_dd95:.2%}.",
            ] + reasons

    return {
        "decision": decision,
        "score": float(score),
        "reasons": reasons,
        "rule_checks": checks,
        "rules_used": r,
    }

#
# Grafici comparativi tra metodi (con benchmark)
#
def plot_mc_method_comparison(methods_results: dict,
                              benchmark_return: float | None,
                              benchmark_drawdown: float | None,
                              title_prefix: str = "MC Comparison"):
    """
    Grafici di confronto:
      1) CDF dei final returns (per metodo) + benchmark_return
      2) CDF dei max drawdown (per metodo) + benchmark_drawdown
      3) Bar chart delle probabilità "bad": P(Return < Bench), P(DD > BenchDD)
    """

    # --- 1) CDF Returns
    plt.figure(figsize=(10, 6))
    for name, res in methods_results.items():
        x = np.sort(np.asarray(res["final_returns"]))
        y = np.linspace(0, 1, len(x), endpoint=True)
        plt.plot(x, y, label=name)
    if benchmark_return is not None:
        plt.axvline(benchmark_return, linestyle="--", label="Benchmark Return")
    plt.title(f"{title_prefix} – CDF Total Return")
    plt.xlabel("Total Return")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- 2) CDF Drawdowns
    plt.figure(figsize=(10, 6))
    for name, res in methods_results.items():
        x = np.sort(np.asarray(res["drawdowns"]))
        y = np.linspace(0, 1, len(x), endpoint=True)
        plt.plot(x, y, label=name)
    if benchmark_drawdown is not None:
        plt.axvline(benchmark_drawdown, linestyle="--", label="Benchmark Max DD")
    plt.title(f"{title_prefix} – CDF Max Drawdown")
    plt.xlabel("Max Drawdown")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- 3) Bad probabilities bar
    labels = []
    p_under_list = []
    p_worse_dd_list = []

    for name, res in methods_results.items():
        labels.append(name)

        if benchmark_return is not None:
            p_under = np.mean(np.asarray(res["final_returns"]) < benchmark_return)
        else:
            p_under = np.nan

        if benchmark_drawdown is not None:
            p_worse_dd = np.mean(np.asarray(res["drawdowns"]) > benchmark_drawdown)
        else:
            p_worse_dd = np.nan

        p_under_list.append(p_under)
        p_worse_dd_list.append(p_worse_dd)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, p_under_list, width, label="P(Return < Benchmark)")
    plt.bar(x + width/2, p_worse_dd_list, width, label="P(DD > BenchmarkDD)")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1)
    plt.title(f"{title_prefix} – Probabilità eventi sfavorevoli vs Benchmark")
    plt.ylabel("Probability")
    plt.grid(True, axis="y")
    plt.legend()
    plt.show()


#
# 5. Esempio di call completa integrata nel framework
#

# # Parametri di base
# n_simulations = 10000
# show_plots = True
# show_summary = True
# slippage = 0.0001           # -0.01% al giorno
# shock_frequency = 0.002     # 0.2% dei giorni con eventi negativi
# shock_magnitude = (0.05, 0.15)
# init_cash = 100_000

# # Returns del TS e benchmark dal B&H vectorbt
# ts_returns = portfolio.returns()          # Serie dei rendimenti giornalieri del TS
# benchmark_return = float(round(bh_portfolio.total_return(), 4))
# benchmark_drawdown = abs(float(round(bh_portfolio.max_drawdown(), 4)))  # assumo che max_drawdown sia negativo

# methods_results, summary_df = run_all_mc_methods(
#     portfolio_returns=ts_returns,
#     init_value=init_cash,
#     benchmark_return=benchmark_return,
#     benchmark_drawdown=benchmark_drawdown,
#     n_simulations=n_simulations,
#     seed=42,
#     show_method_plots=True,        # se vuoi i plot anche per il base (e per gli altri)
#     show_method_summaries=True,
#     block_size=10,
#     regime_window=20,
# )

# # Se ti serve il risultato del "basic"
# results_mc_basic = methods_results["Bootstrap i.i.d."]

# # 1) Tabella estesa confronto MC
# mc_compare_df = compare_mc_methods(
#     methods_results=methods_results,
#     benchmark_return=benchmark_return,
#     benchmark_drawdown=benchmark_drawdown
# )

# print("\n===== MC COMPARE (esteso) =====")
# display(mc_compare_df.style.format("{:.2%}", subset=[c for c in mc_compare_df.columns if c.startswith("Ret_") or c.startswith("DD_") or "Ret" in c or "DD" in c or "P(" in c]).format({
#     "SharpeAvg": "{:.2f}",
#     "N": "{:.0f}"
# }))

# # 2) Grafici comparativi
# plot_mc_method_comparison(
#     methods_results=methods_results,
#     benchmark_return=benchmark_return,
#     benchmark_drawdown=benchmark_drawdown,
#     title_prefix="TS Robustness – Monte Carlo"
# )

# # 3) Decisione deploy/no-deploy (motivata)
# deploy_report = mc_deploy_recommendation(
#     mc_compare_df=mc_compare_df,
#     benchmark_return=benchmark_return,
#     benchmark_drawdown=benchmark_drawdown,
#     rules={
#         # se vuoi essere più severo:
#         # "max_p_under_benchmark": 0.50,
#         # "max_p_worse_dd_than_bench": 0.50,
#         # "min_tail_return_5": -0.10,
#         # "max_tail_dd_95": 0.50,
#         # "min_avg_sharpe": 0.30,
#         # "max_tail_return_5_spread": 0.10,
#     }
# )

# print("\n===== DEPLOY DECISION =====")
# print(f"Decision : {deploy_report['decision']}")
# print(f"Score    : {deploy_report['score']:.1f}/100\n")

# print("Motivazioni:")
# for r in deploy_report["reasons"]:
#     print(f" - {r}")

# print("\nDettaglio regole:")
# for k, v in deploy_report["rule_checks"].items():
#     val = v["value"]
#     thr = v["thr"]
#     ok  = v["ok"]
#     if val is None:
#         print(f" - {k}: skipped (benchmark non disponibile)")
#     else:
#         print(f" - {k}: value={val:.3f} thr={thr:.3f} -> {'OK' if ok else 'FAIL'}")


########################
### Catalog functions
########################

# ============================================================
# Strategy Catalog + Pre-Analysis + WFO Joblist (PLUGIN)
# - Compatibile con get_clean_financial_data() (MultiIndex columns)
# - Auto-catalog da strategy_* già caricate nel notebook (globals())
# - Pre-analisi per ticker (trend/range + vol level)
# - Selezione strategie candidate (filtri requisiti dati + regime/vol)
# - Caching: catalog, data_map, profili, selezioni
# - Stampa risparmio combinazioni per analisi massiva
# ============================================================

# (opzionale ma utile per evitare warning FutureWarning in vari contesti pandas)
try:
    pd.set_option("future.no_silent_downcasting", True)
except Exception:
    pass


# ------------------------------------------------------------
# Caching globale (sessione notebook)
# ------------------------------------------------------------
_KS_CACHE = {
    "catalog": None,
    "catalog_source_id": None,   # id(globals()) o hash di chiavi utili
    "data_map": {},              # key -> dict[ticker->df]
    "profiles": {},              # (key, ticker) -> profile
    "selected": {},              # (key, ticker, top_k) -> list[strategy]
}

def _make_key_for_data(tickers_data: pd.DataFrame, price_group: str) -> str:
    """
    Chiave cache: basa su shape + range date + colonne MultiIndex fingerprint.
    Non è crittografica: serve solo per caching sessione.
    """
    idx = tickers_data.index
    start = str(idx[0]) if len(idx) else "NA"
    end = str(idx[-1]) if len(idx) else "NA"
    # fingerprint colonne
    cols = tickers_data.columns
    fp = f"MI{cols.nlevels}|{len(cols)}"
    return f"{price_group}|{tickers_data.shape[0]}x{tickers_data.shape[1]}|{start}|{end}|{fp}"


# ------------------------------------------------------------
# Adapter: MultiIndex OHLCV -> data_map[ticker] = df flat
# ------------------------------------------------------------
def split_multiindex_ohlcv(tickers_data: pd.DataFrame, price_group: str = "Price") -> dict[str, pd.DataFrame]:
    """
    Converte DF MultiIndex (tipico output get_clean_financial_data) in:
      data_map[ticker] -> df con colonne tra ['Open','High','Low','Close','Volume'] se presenti.

    Struttura tipica:
      columns = MultiIndex: (group, field, ticker)  es. ('Price','Close','AMD')
    """
    if not isinstance(tickers_data.columns, pd.MultiIndex):
        raise ValueError("tickers_data deve avere colonne MultiIndex (output get_clean_financial_data).")

    cols = tickers_data.columns
    if price_group not in cols.get_level_values(0):
        # fallback: usa il primo gruppo disponibile
        price_group = cols.get_level_values(0)[0]

    dfp = tickers_data.loc[:, cols.get_level_values(0) == price_group].copy()

    if dfp.columns.nlevels == 3:
        # (group, field, ticker)
        tickers = dfp.columns.get_level_values(2).unique().tolist()
        data_map: dict[str, pd.DataFrame] = {}

        for t in tickers:
            sub = dfp.xs(t, axis=1, level=2, drop_level=False)
            sub2 = sub.droplevel([0, 2], axis=1)  # lascia solo field
            sub2 = sub2.rename(columns=lambda c: str(c).title())

            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub2.columns]
            data_map[t] = sub2[keep].copy()

        return data_map

    elif dfp.columns.nlevels == 2:
        # Variante (field, ticker)
        lvl0 = [str(v) for v in dfp.columns.get_level_values(0).unique().tolist()]
        if any(x in lvl0 for x in ["Close", "Open", "High", "Low", "Volume"]):
            tickers = dfp.columns.get_level_values(1).unique().tolist()
            data_map: dict[str, pd.DataFrame] = {}
            for t in tickers:
                sub2 = dfp.xs(t, axis=1, level=1, drop_level=True).copy()
                sub2 = sub2.rename(columns=lambda c: str(c).title())
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in sub2.columns]
                data_map[t] = sub2[keep].copy()
            return data_map

        raise ValueError("Formato MultiIndex a 2 livelli non riconosciuto per OHLCV.")
    else:
        raise ValueError("Numero di livelli MultiIndex non supportato.")


def get_or_build_data_map(tickers_data: pd.DataFrame, price_group: str = "Price", use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Caching data_map per evitare riconversioni ripetute su analisi massiva.
    """
    key = _make_key_for_data(tickers_data, price_group)
    if use_cache and key in _KS_CACHE["data_map"]:
        return _KS_CACHE["data_map"][key]

    data_map = split_multiindex_ohlcv(tickers_data, price_group=price_group)
    if use_cache:
        _KS_CACHE["data_map"][key] = data_map
    return data_map


# ------------------------------------------------------------
# Catalogo strategie: auto da globals() + euristiche requisiti
# ------------------------------------------------------------
def infer_strategy_requirements_from_name(name: str) -> dict:
    n = name.lower()
    needs_volume = any(x in n for x in ["volume", "vol_", "klinger", "obv", "mfi", "cmf", "vwap"])
    needs_ohlc = any(x in n for x in ["atr", "adx", "donchian", "keltner"])  # pattern tipici OHLC
    price_only = not (needs_volume or needs_ohlc)
    return {"price_only": price_only, "needs_volume": needs_volume, "needs_ohlc": needs_ohlc}


def infer_strategy_metadata_from_name(name: str) -> dict:
    """
    Euristica minima per family/style/regime/vol_pref.
    Estendibile senza cambiare il resto del plug-in.
    """
    n = name.lower()
    mean_reverting = any(x in n for x in ["rsi", "boll", "bollinger", "zscore", "stoch", "revert", "mean"])

    if any(x in n for x in ["momentum", "roc", "breakout", "trend", "donchian", "macd", "ma_"]):
        return {
            "family": "momentum",
            "style": "trend_following",
            "market_regime": ["trend"],
            "volatility_preference": "medium",
        }

    if mean_reverting:
        return {
            "family": "mean_reversion",
            "style": "reversion",
            "market_regime": ["range"],
            "volatility_preference": "low",
        }

    return {
        "family": "unknown",
        "style": "unknown",
        "market_regime": ["any"],
        "volatility_preference": "any",
    }


def build_strategy_catalog_from_globals(
    g: dict,
    use_cache: bool = True,
    require_param_grid: bool = False
) -> dict[str, dict]:
    """
    Costruisce catalogo leggendo l'ambiente del notebook (globals()) dopo che
    k_strategies è stato eseguito/importato.

    require_param_grid:
      - False: include tutte le strategy_*
      - True : include solo quelle che hanno anche strategy_<name>_param_ranges
    """
    # fingerprint semplice: numero di chiavi strategy_ e param_ranges
    strategy_keys = [k for k, v in g.items() if callable(v) and isinstance(k, str) and k.startswith("strategy_")]
    grid_keys = [k for k, v in g.items() if isinstance(v, dict) and isinstance(k, str) and k.startswith("strategy_") and k.endswith("_param_ranges")]
    source_id = f"{len(strategy_keys)}|{len(grid_keys)}"

    if use_cache and _KS_CACHE["catalog"] is not None and _KS_CACHE["catalog_source_id"] == source_id:
        return _KS_CACHE["catalog"]

    # map griglie
    grids = {k: g[k] for k in grid_keys}

    catalog: dict[str, dict] = {}
    for fn_name in strategy_keys:
        base = fn_name.replace("strategy_", "")
        grid_name = f"strategy_{base}_param_ranges"
        has_grid = grid_name in grids

        if require_param_grid and not has_grid:
            continue

        meta = infer_strategy_metadata_from_name(base)
        meta.update(infer_strategy_requirements_from_name(base))

        # campi utili
        meta.update({
            "name": base,
            "strategy_fn_name": fn_name,
            "has_param_grid": has_grid,
            "param_grid_name": grid_name if has_grid else None,
            "param_grid": grids.get(grid_name),
            "direction": "long_only",
            "holding_profile": "swing",
            "complexity": "low",
        })

        catalog[base] = meta

    if use_cache:
        _KS_CACHE["catalog"] = catalog
        _KS_CACHE["catalog_source_id"] = source_id

    return catalog


# ------------------------------------------------------------
# Pre-analisi sottostante (su df ticker con Close e opzionali)
# ------------------------------------------------------------
def analyze_asset_profile_from_df(
    df: pd.DataFrame,
    lookback_trend: int = 126,
    lookback_vol: int = 63,
    slope_threshold: float = 0.02,
    r2_threshold: float = 0.20
) -> dict:
    """
    Profilo:
      - regime: trend | range | unknown
      - vol_level: low | medium | high | unknown
      - trend_dir: up | down | flat | unknown
    """
    if df is None or df.empty or "Close" not in df.columns:
        return {"regime": "unknown", "vol_level": "unknown", "trend_dir": "unknown"}

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.shape[0] < max(lookback_trend, lookback_vol) + 5:
        return {"regime": "unknown", "vol_level": "unknown", "trend_dir": "unknown"}

    y = np.log(close.iloc[-lookback_trend:].values)
    x = np.arange(len(y)).astype(float)

    b, a = np.polyfit(x, y, 1)
    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else np.nan)

    approx_ret = float(np.exp(b * lookback_trend) - 1.0)
    regime = "trend" if (abs(approx_ret) >= slope_threshold and (not np.isnan(r2)) and r2 >= r2_threshold) else "range"
    trend_dir = "up" if approx_ret > 0 else ("down" if approx_ret < 0 else "flat")

    rets = np.log(close).diff()
    vol_annual = float(rets.iloc[-lookback_vol:].std(ddof=0) * np.sqrt(252))

    if np.isnan(vol_annual):
        vol_level = "unknown"
    elif vol_annual < 0.20:
        vol_level = "low"
    elif vol_annual < 0.40:
        vol_level = "medium"
    else:
        vol_level = "high"

    return {"regime": regime, "vol_level": vol_level, "trend_dir": trend_dir, "vol_annual": vol_annual, "r2": float(r2) if not np.isnan(r2) else np.nan}


# ------------------------------------------------------------
# Selezione strategie: filtri requisiti dati + regime/vol
# ------------------------------------------------------------
def select_strategies_for_asset_df(
    df: pd.DataFrame,
    catalog: dict[str, dict],
    top_k: int = 10,
    allow_unknown_profile: bool = False
) -> tuple[list[str], dict]:
    profile = analyze_asset_profile_from_df(df)
    reg = profile.get("regime", "unknown")
    vol = profile.get("vol_level", "unknown")

    if (reg == "unknown" or vol == "unknown") and not allow_unknown_profile:
        return [], profile

    has_volume = "Volume" in df.columns
    has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])

    scored = []
    for sname, meta in catalog.items():
        # requisiti dati
        if meta.get("needs_volume", False) and not has_volume:
            continue
        if meta.get("needs_ohlc", False) and not has_ohlc:
            continue

        # regime
        regimes = meta.get("market_regime", ["any"])
        if reg != "unknown" and (reg not in regimes and "any" not in regimes):
            continue

        # volatilità
        pref = meta.get("volatility_preference", "any")
        if vol != "unknown" and (pref != "any" and pref != vol):
            continue

        # scoring leggero
        score = 0
        if reg in regimes:
            score += 2
        if pref == "any" or pref == vol:
            score += 1
        if reg == "trend" and meta.get("style") == "trend_following":
            score += 1

        scored.append((sname, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    selected = [s for s, _ in scored][: int(top_k)]
    return selected, profile


# ------------------------------------------------------------
# Joblist builder: ticker list -> (Ticker, Strategy, profilo)
# con caching profili/selezioni per analisi massiva
# ------------------------------------------------------------
def build_wfo_joblist_from_clean_data(
    tickers: list[str],
    tickers_data: pd.DataFrame,
    catalog: dict[str, dict] | None = None,
    top_k_per_ticker: int = 10,
    price_group: str = "Price",
    use_cache: bool = True,
    show_progress: bool = True,
    allow_unknown_profile: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    """
    Crea una joblist (Ticker, Strategy) usando l'output di get_clean_financial_data.

    Note:
    - Si aspetta normalmente `tickers_data` con colonne MultiIndex del tipo:
      (Group, Ticker, Field)
    - In alcuni casi la funzione get_or_build_data_map può restituire una mappa
      indicizzata per Field invece che per Ticker (tipicamente nel caso single-ticker).
      In tal caso viene applicato un fallback che ricostruisce correttamente la mappa
      ticker -> DataFrame OHLCV.
    """
    def _jdebug(*args, **kwargs):
       if debug:
           print(*args, **kwargs)
        
    if catalog is None:
        catalog = build_strategy_catalog_from_globals(
            globals(),
            use_cache=True,
            require_param_grid=False
        )

    data_key = _make_key_for_data(tickers_data, price_group)
    data_map = get_or_build_data_map(
        tickers_data,
        price_group=price_group,
        use_cache=use_cache
    )

    # --------------------------------------------------------------
    # Fallback robusto: se data_map non è indicizzata per ticker ma
    # per field (es. Close/High/Low/Open/Volume), ricostruiscila.
    # --------------------------------------------------------------
    rebuild_data_map = False

    if not isinstance(data_map, dict):
        rebuild_data_map = True
    else:
        missing_tickers = [t for t in tickers if t not in data_map]
        if missing_tickers:
            rebuild_data_map = True

    if rebuild_data_map:
        rebuilt_map: dict[str, pd.DataFrame] = {}

        # Caso MultiIndex atteso: (Group, Ticker, Field)
        if isinstance(tickers_data.columns, pd.MultiIndex):
            col_names = list(tickers_data.columns.names)

            try:
                group_level = col_names.index("Group")
                ticker_level = col_names.index("Ticker")
                field_level = col_names.index("Field")
            except ValueError:
                # fallback posizionale
                group_level, ticker_level, field_level = 0, 1, 2

            available_groups = set(tickers_data.columns.get_level_values(group_level))
            if price_group in available_groups:
                td = tickers_data.xs(price_group, axis=1, level=group_level)
            else:
                td = tickers_data.copy()

            # td ora dovrebbe avere colonne (Ticker, Field)
            if isinstance(td.columns, pd.MultiIndex):
                available_tickers = set(td.columns.get_level_values(0))
                for t in tickers:
                    if t in available_tickers:
                        sub = td.xs(t, axis=1, level=0)
                        if isinstance(sub, pd.Series):
                            sub = sub.to_frame()
                        sub = sub.dropna(how="all")
                        if not sub.empty:
                            rebuilt_map[t] = sub.copy()

        # Caso flat columns e singolo ticker
        elif len(tickers) == 1:
            t = str(tickers[0])
            df_single = tickers_data.copy()
            df_single = df_single.dropna(how="all")
            if not df_single.empty:
                rebuilt_map[t] = df_single

        data_map = rebuilt_map

    rows = []
    it = tickers
    if show_progress:
        it = tqdm(tickers, desc="Pre-analysis + strategy selection", leave=False)

    for t in it:
        df = data_map.get(t)

        if df is None or df.empty:
            continue

        # caching profilo
        pkey = (data_key, t)
        if use_cache and pkey in _KS_CACHE["profiles"]:
            profile = _KS_CACHE["profiles"][pkey]
        else:
            profile = analyze_asset_profile_from_df(df)
            if use_cache:
                _KS_CACHE["profiles"][pkey] = profile

        # caching selezione
        skey = (data_key, t, int(top_k_per_ticker), bool(allow_unknown_profile))
        if use_cache and skey in _KS_CACHE["selected"]:
            selected = _KS_CACHE["selected"][skey]
        else:
            selected, _ = select_strategies_for_asset_df(
                df=df,
                catalog=catalog,
                top_k=top_k_per_ticker,
                allow_unknown_profile=allow_unknown_profile
            )
            if use_cache:
                _KS_CACHE["selected"][skey] = selected

        for s in selected:
            rows.append({
                "Ticker": t,
                "Strategy": s,
                "Regime": profile.get("regime"),
                "VolLevel": profile.get("vol_level"),
                "TrendDir": profile.get("trend_dir"),
            })

    return pd.DataFrame(rows)
    

# ------------------------------------------------------------
# Report risparmio combinazioni + helper per analisi massiva
# ------------------------------------------------------------
def print_joblist_summary(
    tickers: list[str],
    catalog: dict[str, dict],
    joblist: pd.DataFrame,
    title: str | None = None
) -> None:
    """
    Stampa statistiche di riduzione combinazioni e copertura.
    """
    if title:
        print(f"\n=== {title} ===")

    n_t = len(tickers)
    n_s = len(catalog)
    naive = n_t * n_s

    n_jobs = int(joblist.shape[0])
    kept_pct = (n_jobs / naive * 100.0) if naive > 0 else 0.0
    saved = naive - n_jobs
    saved_pct = (saved / naive * 100.0) if naive > 0 else 0.0

    covered_tickers = joblist["Ticker"].nunique() if "Ticker" in joblist.columns else 0
    uncovered = n_t - int(covered_tickers)

    print(f"Tickers: {n_t}")
    print(f"Strategie in catalogo: {n_s}")
    print(f"Combinazioni naive (tickers×strategie): {naive:,}")
    print(f"Job selezionati (pre-WFO): {n_jobs:,}")
    print(f"Riduzione: {saved:,} ({saved_pct:.2f}%) | Retained: {kept_pct:.2f}%")
    print(f"Tickers con almeno 1 strategia candidata: {covered_tickers} | senza candidati: {uncovered}")

    # breakdown rapido
    if n_jobs > 0 and "Strategy" in joblist.columns:
        top_strats = joblist["Strategy"].value_counts().head(10)
        print("\nTop 10 strategie per frequenza in joblist:")
        for k, v in top_strats.items():
            print(f"  {k}: {int(v)}")

    if n_jobs > 0 and "Regime" in joblist.columns:
        print("\nDistribuzione Regime (tickers selezionati):")
        print(joblist.groupby("Ticker")["Regime"].first().value_counts().to_string())


def build_massive_joblists(
    universes: dict[str, list[str]],
    start_date: str,
    end_date: str,
    get_clean_financial_data_fn,
    price_group: str = "Price",
    top_k_per_ticker: int = 10,
    require_param_grid: bool = False,
    show_progress: bool = True,
    allow_unknown_profile: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Pipeline completa per analisi massiva:
      - costruisce catalogo (una volta)
      - per ciascun universo:
          scarica dati con get_clean_financial_data_fn(tickers, start, end)
          crea joblist
          stampa summary
      - ritorna dict universo -> joblist

    NOTA: get_clean_financial_data_fn è la tua funzione get_clean_financial_data.
    """
    catalog = build_strategy_catalog_from_globals(globals(), use_cache=True, require_param_grid=require_param_grid)

    out = {}
    uni_iter = universes.items()
    if show_progress:
        uni_iter = tqdm(list(uni_iter), desc="Universes", leave=True)

    for uni_name, tickers in uni_iter:
        tickers_data = get_clean_financial_data_fn(tickers, start_date, end_date)

        joblist = build_wfo_joblist_from_clean_data(
            tickers=tickers,
            tickers_data=tickers_data,
            catalog=catalog,
            top_k_per_ticker=top_k_per_ticker,
            price_group=price_group,
            use_cache=True,
            show_progress=show_progress,
            allow_unknown_profile=allow_unknown_profile,
        )

        print_joblist_summary(
            tickers=tickers,
            catalog=catalog,
            joblist=joblist,
            title=f"Universe: {uni_name}"
        )

        out[uni_name] = joblist

    return out


# ------------------------------------------------------------
# ESEMPIO USO (commentato)
# ------------------------------------------------------------
# universes = {
#     "sp100_top": sp100_top_tickers,
#     "nasdaq100_top": nasdaq100_top_tickers,
#     "euro_top": euro_top_tickers,
# }
#
# # Assumendo che:
# # - k_strategies.ipynb sia già eseguito/importato (quindi strategy_* in globals())
# # - tu abbia la funzione get_clean_financial_data disponibile
# joblists = build_massive_joblists(
#     universes=universes,
#     start_date="2015-01-01",
#     end_date="2026-01-02",
#     get_clean_financial_data_fn=get_clean_financial_data,
#     top_k_per_ticker=10,
#     require_param_grid=False,
#     show_progress=True,
#     allow_unknown_profile=False
# )
#
# # joblists["sp100_top"] è un DF pronto per alimentare il tuo loop WFO




