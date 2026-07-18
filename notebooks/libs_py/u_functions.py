"""
u_functions.py — Common Utilities Functions
Refactored from notebooks/libs/u_functions.ipynb
Excludes: deprecated versions (_BAD, _R1), commented-out code,
          notebook-only deps (ace_tools_open, itables),
          cell 9 (notebook init code).
"""

from __future__ import annotations

import os
import re
import json
import glob
import time
import shutil
import smtplib
import ssl
import html as _html
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import reduce
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Figure as Figure
from plotly.subplots import make_subplots
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_TSLAB_INPUNTS_DIR = os.environ.get("IQ_INPUTS_DIR",  "../../inputs/")
_TSLAB_OUTPUTS_DIR = os.environ.get("IQ_OUTPUTS_DIR", "../../outputs")
_TSLAB_CACHE_DIR   = os.environ.get("IQ_CACHE_DIR",   "../../cache")

_TSLAB_RUNTIME_T_WFO_RESULTS_DIR = f"{_TSLAB_INPUNTS_DIR}/WFO_T_RUN_RESULTS"
_TSLAB_RUNTIME_R_WFO_RESULTS_DIR = f"{_TSLAB_INPUNTS_DIR}/WFO_R_RUN_RESULTS"
_TSLAB_DEV_T_WFO_RESULTS_DIR     = f"{_TSLAB_OUTPUTS_DIR}/WFO_T_DEV_RESULTS"
_TSLAB_DEV_R_WFO_RESULTS_DIR     = f"{_TSLAB_OUTPUTS_DIR}/WFO_R_DEV_RESULTS"
_TSLAB_K_PANEL_EXPORTS_DIR       = f"{_TSLAB_OUTPUTS_DIR}/k_panel_exports"
_TSLAB_L_PANEL_EXPORTS_DIR       = f"{_TSLAB_OUTPUTS_DIR}/l_panel_exports"

def get_analysis_output_dir(
    category: str,
    ptf_name: str = None,
    profilo: str = None,
    timestamp: str = None,
) -> "Path":
    """
    Calcola il path di output canonico per un'analisi CLI o notebook.

    Parameters
    ----------
    category  : "r_analysis" | "k_analysis" | "l_analysis"
    ptf_name  : sottocartella portafoglio (solo R-portfolio la usa)
    profilo   : profilo di rischio, es. "satellite" | "core" (solo R-portfolio)
                Path: <category>/<ptf_name>/<profilo>/<timestamp>/
    timestamp : stringa timestamp; se None genera datetime.now() con
                formato "%Y%m%d_%H%M%S"

    Returns
    -------
    pathlib.Path — <TSLAB_OUTPUTS_DIR>/<category>[/<ptf_name>][/<profilo>]/<timestamp>
    """
    from pathlib import Path
    from datetime import datetime
    _valid = ("r_analysis", "k_analysis", "l_analysis")
    if category not in _valid:
        raise ValueError(f"get_analysis_output_dir: category deve essere uno di {_valid}, ricevuto '{category}'")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(_TSLAB_OUTPUTS_DIR) / category
    if ptf_name:
        base = base / ptf_name
    if profilo:
        base = base / profilo
    return base / timestamp


def update_latest_symlink(run_dir) -> bool:
    """
    Aggiorna il symlink `latest` nella directory padre di run_dir,
    puntando a run_dir (relativo), solo se almeno un CSV esiste in run_dir.

    Ritorna True se il symlink è stato aggiornato, False altrimenti.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    if not list(run_dir.glob("*.csv")):
        return False
    parent = run_dir.parent
    symlink = parent / "latest"
    target = run_dir.name
    try:
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()
        symlink.symlink_to(target)
        return True
    except OSError as e:
        print(f"[WARN] update_latest_symlink: impossibile aggiornare symlink: {e}")
        return False


vbt_plot_width = 1100

# ---------------------------------------------------------------------------
# Cosmesi
# ---------------------------------------------------------------------------
class Emoji:
    UP = "📈"; DOWN = "📉"; FIRE = "🔥"; MONEY = "💰"; ALERT = "⚠️"
    WARNING = "⚠️"; INFO = "ℹ️"; CHECK = "✅"; SEARCH = "🔎"; STAR = "⭐"
    CHART = "📊"; STRATEGY = "🧪"; RISK = "⚠️"; WIN = "🏆"; FAIL = "❌"
    ROCKET = "🚀"; PARAMS = "🛠️"; MACRO = "🏛️"; MOMENTUM = "⏳"; REBALANCE = "🔄"
    DIVIETO = "🚫"; RUN = "▶️"; FAST_RUN = "⏩"; POWER_RUN = "⚡"; LOOP = "🔄"
    EXECUTE = "🚀"; PROCESS = "🏃"; SETUP = "🛠️"; CALENDAR = "📅"

RESET = "\033[0m";   BOLD = "\033[1m";    DIM = "\033[2m"
ITALIC = "\033[3m";  UNDER = "\033[4m";   BLINK = "\033[5m"; REVERSE = "\033[7m"
BLACK = "\033[30m";  RED = "\033[31m";    GREEN = "\033[32m"; YELLOW = "\033[33m"
BLUE = "\033[34m";   MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"
BRIGHT_BLACK = "\033[90m";  BRIGHT_RED = "\033[91m";  BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"; BRIGHT_BLUE = "\033[94m"; BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m";   BRIGHT_WHITE = "\033[97m"

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def now() -> datetime:
    return datetime.now()

today = now

def ytd_date() -> str:
    return f"{now().year}-01-01"

ytd = ytd_date


# ---------------------------------------------------------------------------
# Code-check utilities
# ---------------------------------------------------------------------------
def delete_paths(paths, dry_run=True):
    deleted, failed = [], []
    for p in map(Path, paths):
        try:
            if not p.exists():
                failed.append((p, "not found")); continue
            if dry_run:
                deleted.append((p, "would delete dir" if p.is_dir() else "would delete file")); continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            deleted.append((p, "deleted"))
        except Exception as e:
            failed.append((p, repr(e)))
    return deleted, failed


_DEF_RE = re.compile(r'^def\s+([A-Za-z_]\w*)\s*\(')


def _load_ipynb_code_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    lines = []
    for ci, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.splitlines(True)
        for li, line in enumerate(src, start=1):
            if line.lstrip().startswith(("%", "!", "?")):
                continue
            lines.append((line.rstrip("\n"), ci, li))
    return lines


def _load_py_code_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read().splitlines()
    return [(line, None, i + 1) for i, line in enumerate(src)]


def find_duplicate_function_defs(path, prefix=None, top_level_only=True, show=True):
    path = os.fspath(path)
    lines = _load_ipynb_code_lines(path) if path.endswith(".ipynb") else _load_py_code_lines(path)
    seen = {}
    for line, cell_idx, line_no in lines:
        if top_level_only and (line.startswith(" ") or line.startswith("\t")):
            continue
        m = _DEF_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if prefix and not name.startswith(prefix):
            continue
        pos = f"C{cell_idx}:L{line_no}" if cell_idx is not None else f"L{line_no}"
        seen.setdefault(name, []).append(pos)
    duplicates = {n: locs for n, locs in seen.items() if len(locs) > 1}
    if show:
        base = os.path.basename(path)
        if duplicates:
            print(f"Duplicati trovati in {base}:")
            for n, locs in sorted(duplicates.items()):
                print(f"  {n}: definita {len(locs)} volte a {locs}")
        else:
            print(f"Nessuna funzione duplicata in {base}")
    return duplicates


def _scan_file(path, prefix=None, top_level_only=True, collect_all=False):
    lines = _load_ipynb_code_lines(path) if path.endswith(".ipynb") else _load_py_code_lines(path)
    seen = {}
    for line, cell_idx, line_no in lines:
        if top_level_only and (line.startswith(" ") or line.startswith("\t")):
            continue
        m = _DEF_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if prefix and not name.startswith(prefix):
            continue
        pos = f"C{cell_idx}:L{line_no}" if cell_idx is not None else f"L{line_no}"
        seen.setdefault(name, []).append(pos)
    duplicates = {n: locs for n, locs in seen.items() if len(locs) > 1}
    return duplicates, seen if collect_all else None


def find_duplicate_function_defs_multi(patterns, prefix=None, top_level_only=True,
                                        recursive=True, show=True):
    if isinstance(patterns, str):
        pats = [p.strip() for p in re.split(r'[,\s]+', patterns) if p.strip()]
    else:
        pats = list(patterns)
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=recursive))
    files = sorted(set(f for f in files if f.endswith((".ipynb", ".py"))))
    per_file = {}
    name_positions_global = {}
    for path in files:
        dups, name_pos = _scan_file(path, prefix=prefix, top_level_only=top_level_only, collect_all=True)
        per_file[path] = dups
        for name, positions in name_pos.items():
            for pos in positions:
                name_positions_global.setdefault(name, []).append((path, pos))
    cross_file = {n: locs for n, locs in name_positions_global.items()
                  if len({fp for fp, _ in locs}) > 1}
    if show:
        print("Verifica naming di funzioni...\n")
        for path in files:
            base = os.path.basename(path)
            d = per_file[path]
            if d:
                print(f"Duplicati in {base}:")
                for n, locs in sorted(d.items()):
                    print(f"  {n}: {locs}")
            else:
                print(f"Nessun duplicato in {base}")
        if cross_file:
            print("\nFunzioni con LO STESSO NOME presenti in file diversi:")
            for n, locs in sorted(cross_file.items()):
                print(f"  {n}: {[f'{os.path.basename(fp)}:{pos}' for fp, pos in locs]}")
        else:
            print("\nNessun nome di funzione ripetuto su file diversi.")
    return per_file, cross_file


# ---------------------------------------------------------------------------
# Download / Financial Data
# ---------------------------------------------------------------------------
def _load_nav_from_cache(ticker: str, cache_dir: str,
                          start=None, end=None) -> "pd.Series | None":
    """Cerca in cache_dir qualunque *.csv il cui nome contenga ticker (case-insensitive)."""
    import glob as _glob, re as _re
    all_csv = _glob.glob(os.path.join(cache_dir, "*.csv"))
    tk_lower = ticker.lower()
    matches = [p for p in all_csv if tk_lower in os.path.basename(p).lower()]
    if not matches:
        return None
    def _sort_key(p):
        # Preferisce data ISO nel nome; fallback a mtime
        m = _re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(p))
        return m.group(1) if m else f"mtime:{os.path.getmtime(p):.6f}"
    best = max(matches, key=_sort_key)
    try:
        nav_df = pd.read_csv(best, sep=';', decimal=',')
        nav_df['Data'] = pd.to_datetime(nav_df['Data'], format='%d/%m/%Y')
        nav_series = nav_df.set_index('Data')['NAV'].astype(float).sort_index()
        if start:
            nav_series = nav_series[nav_series.index >= pd.Timestamp(start)]
        if end:
            nav_series = nav_series[nav_series.index < pd.Timestamp(end)]
        return nav_series if not nav_series.empty else None
    except Exception:
        return None

def load_ohlcv(symbol: str, start: str = None, end: str = None,
               show_progress: bool = False, auto_adjust: bool = True,
               multi_level_index: bool = False, interval: str = "1d") -> pd.DataFrame:
    """Scarica dati OHLCV da yfinance con indice DatetimeIndex.
    Fallback su CSV locale per simboli non su yfinance (es. fondi con NAV):
    cerca {IQ_CACHE_DIR}/{symbol}-NAV_History-*.csv e usa il NAV come OHLC.
    """
    df = yf.download(symbol, start=start, end=end, multi_level_index=multi_level_index,
                     auto_adjust=auto_adjust, progress=show_progress, interval=interval)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # --- Cache fallback: fondi NAV non disponibili su yfinance ---
    symbols = list(symbol) if isinstance(symbol, (list, tuple)) else [symbol]
    if isinstance(df.columns, pd.MultiIndex):
        # Batch 2+ ticker → MultiIndex (Price, Ticker)
        existing_tickers = set(df.columns.get_level_values(1).unique())
        # caso (a): presente ma Close tutto NaN; caso (b): assente dal MultiIndex
        failed = [tk for tk in symbols
          if tk not in existing_tickers
          or (('Close', tk) in df.columns and df[('Close', tk)].isna().mean() > 0.5)]
        # failed = [tk for tk in symbols
        #           if tk not in existing_tickers
        #           or (('Close', tk) in df.columns and df[('Close', tk)].isna().all())]
        for tk in failed:
            nav = _load_nav_from_cache(tk, _TSLAB_CACHE_DIR, start, end)
            if nav is None:
                print(f"[load_ohlcv] {tk}: nessun dato su yfinance né in cache locale — colonna rimarrà vuota/NaN.")
                continue
            nav = nav.reindex(df.index, method='ffill')  # allinea; ffill per gap nel CSV NAV
            for price in ('Close', 'Open', 'High', 'Low'):
                col = (price, tk)
                df[col] = nav
            vol_col = ('Volume', tk)
            df[vol_col] = 0
            # Rimuove l'artifact 'Adj Close' che yfinance aggiunge solo ai ticker falliti
            if ('Adj Close', tk) in df.columns:
                df = df.drop(columns=[('Adj Close', tk)])
            print(f"[load_ohlcv] {tk}: dati non disponibili su yfinance, recuperati da cache locale "
                  f"({nav.notna().sum()} righe valide, {nav.first_valid_index().date()} → {nav.last_valid_index().date()}).")
    elif df.empty and len(symbols) == 1:
        # Stringa singola o lista con 1 elemento fallito → df vuoto, flat columns
        tk = symbols[0]
        nav = _load_nav_from_cache(tk, _TSLAB_CACHE_DIR, start, end)
        if nav is not None:
            df = pd.DataFrame({'Close': nav, 'Open': nav, 'High': nav, 'Low': nav, 'Volume': 0})
            df.index.name = 'Date'
            print(f"[load_ohlcv] {tk}: dati non disponibili su yfinance, recuperati da cache locale "
                  f"({nav.notna().sum()} righe valide, {nav.first_valid_index().date()} → {nav.last_valid_index().date()}).")
        else:
            print(f"[load_ohlcv] {tk}: nessun dato su yfinance né in cache locale — DataFrame vuoto.")
    return df

get_clean_financial_data = load_ohlcv


def download_data(tickers, start_date=None, end_date=None, auto_adjust=True, show_progress=False):
    return load_ohlcv(tickers, start=start_date, end=end_date, auto_adjust=auto_adjust,
                      show_progress=show_progress).Close


def load_isin_overrides(path: str = None) -> dict:
    if path is None:
        path = f"{_TSLAB_CACHE_DIR}/ticker_isin_overrides.csv"
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        if "Ticker" not in df.columns or "ISIN" not in df.columns:
            return {}
        df["Ticker"] = df["Ticker"].str.strip()
        df["ISIN"]   = df["ISIN"].str.strip()
        df = df.dropna(subset=["Ticker", "ISIN"])
        df = df[(df["Ticker"] != "") & (df["ISIN"] != "")]
        return dict(zip(df["Ticker"], df["ISIN"]))
    except Exception:
        return {}


def build_company_df_with_cache(tickers, cache_path: str = None, expire_days=250,
                                 max_retries=3, backoff_factor=1.5):
    """
    Costruisce o ricarica da cache un DataFrame con Company, marketCap e ISIN.
    Priorità ISIN: 1) ticker_isin_overrides.csv  2) cache  3) yfinance.isin
    """
    if cache_path is None:
        cache_path = f"{_TSLAB_CACHE_DIR}/company_cache.csv"
    isin_overrides = load_isin_overrides()
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, parse_dates=["DateFetched"], index_col="Ticker")
        for col in ("Company", "marketCap", "ISIN"):
            if col not in cache.columns:
                cache[col] = pd.NA
        if "DateFetched" not in cache.columns:
            cache["DateFetched"] = pd.NaT
    else:
        cache = pd.DataFrame(columns=["Company", "marketCap", "ISIN", "DateFetched"])
        cache.index.name = "Ticker"

    cache_changed = False
    for tk in tickers:
        if tk in isin_overrides and tk in cache.index:
            ov = isin_overrides[tk]
            if ov and pd.notna(ov) and (pd.isna(cache.at[tk, "ISIN"]) or cache.at[tk, "ISIN"] != ov):
                cache.at[tk, "ISIN"] = ov; cache_changed = True
    if cache_changed:
        cache.to_csv(cache_path)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=expire_days)
    to_fetch = [tk for tk in tickers
                if tk not in cache.index or pd.isna(cache.at[tk, "DateFetched"])
                or cache.at[tk, "DateFetched"] < cutoff or pd.isna(cache.at[tk, "ISIN"])]

    def fetch_info(obj):
        delay = 1.0
        for _ in range(max_retries):
            try: return obj.info
            except Exception: time.sleep(delay); delay *= backoff_factor
        return {}

    def fetch_isin(obj, tk):
        if tk in isin_overrides:
            v = isin_overrides[tk]
            if v and pd.notna(v): return v
        if obj is None: return None
        try:
            isin = obj.isin
            if isin and isin != '-': return isin
        except Exception: pass
        return None

    if to_fetch:
        try:
            tk_objs = yf.Tickers(" ".join(to_fetch)).tickers
            fetched = {tk: {"Company": (info := fetch_info(obj) or {}).get("longName") or info.get("shortName"),
                            "marketCap": info.get("marketCap"),
                            "ISIN": fetch_isin(obj, tk),
                            "DateFetched": pd.Timestamp.now()}
                       for tk, obj in tk_objs.items()}
        except Exception:
            fetched = {}
            for tk in to_fetch:
                try: obj = yf.Ticker(tk); info = fetch_info(obj) or {}
                except Exception: obj = None; info = {}
                fetched[tk] = {"Company": info.get("longName") or info.get("shortName"),
                               "marketCap": info.get("marketCap"),
                               "ISIN": fetch_isin(obj, tk),
                               "DateFetched": pd.Timestamp.now()}
        fetched_df = pd.DataFrame.from_dict(fetched, orient="index")
        fetched_df.index.name = "Ticker"
        cache.update(fetched_df)
        new_idx = fetched_df.index.difference(cache.index)
        if not new_idx.empty:
            cache = pd.concat([cache, fetched_df.loc[new_idx]])
        cache.to_csv(cache_path)

    return cache.reindex(tickers)[["Company", "marketCap", "ISIN"]]


def fetch_data_and_companies(tickers, start_date=None, end_date=None, show_progress=False,
                              normalize: bool = False, min_overlap_cols: int = 2,
                              verbose: bool = True, auto_adjust: bool = True,
                              enrich_companies: bool = False,
                              enrich_fields: tuple = ("marketCap", "priceToBook", "bookValue",
                                                       "trailingPE", "forwardPE", "enterpriseValue"),
                              enrich_pause: float = 0.15, enrich_max_retries: int = 2):
    stocks_data = download_data(tickers, start_date=start_date, end_date=end_date,
                                show_progress=show_progress, auto_adjust=auto_adjust)
    common_start_date = common_end_date = None

    if normalize and isinstance(stocks_data, pd.DataFrame):
        df = stocks_data.copy().sort_index()
        first_valid = df.apply(lambda s: s.first_valid_index())
        last_valid  = df.apply(lambda s: s.last_valid_index())
        fully_nan = first_valid[first_valid.isna()].index.tolist()
        if fully_nan:
            if verbose: print(f"[Normalizzazione] Rimossi: {fully_nan}")
            df = df.drop(columns=fully_nan)
            if df.empty:
                return df, build_company_df_with_cache(tickers), None, None
            first_valid = df.apply(lambda s: s.first_valid_index())
            last_valid  = df.apply(lambda s: s.last_valid_index())

        keep_cols = df.columns.tolist(); changed = True
        while changed and len(keep_cols) >= min_overlap_cols:
            changed = False
            fv = first_valid[keep_cols]; lv = last_valid[keep_cols]
            if max(fv.dropna()) <= min(lv.dropna()): break
            worst_fv = fv.idxmax(); worst_lv = lv.idxmin()
            to_drop = worst_fv if fv[worst_fv] >= lv[worst_lv] else worst_lv
            keep_cols.remove(to_drop); changed = True
            if verbose: print(f"[Normalizzazione] Rimuovo '{to_drop}'. Rimaste: {len(keep_cols)}")

        if len(keep_cols) >= min_overlap_cols:
            fv = first_valid[keep_cols]; lv = last_valid[keep_cols]
            common_start = max(fv.dropna()); common_end = min(lv.dropna())
            df = df[keep_cols].loc[common_start:common_end]
            if int(df.isna().sum().sum()) > 0: df = df.bfill().ffill()
            common_start_date = df.index.min() if not df.empty else None
            common_end_date   = df.index.max() if not df.empty else None
        else:
            df = df.bfill().ffill()
            common_start_date = df.index.min() if not df.empty else None
            common_end_date   = df.index.max() if not df.empty else None
        stocks_data = df

    company_data = build_company_df_with_cache(tickers)

    if enrich_companies:
        if verbose: print(f"[Companies] Enrichment: {enrich_fields}")
        if 'Ticker' in company_data.columns:
            company_data = company_data.set_index('Ticker')
        for tk in tickers:
            for attempt in range(enrich_max_retries):
                try:
                    info = yf.Ticker(tk).info or {}
                    for fld in enrich_fields:
                        company_data.loc[tk, fld] = info.get(fld, np.nan)
                    break
                except Exception as e:
                    if attempt + 1 >= enrich_max_retries and verbose:
                        print(f"[Companies] Enrich failed for {tk}: {e}")
                    time.sleep(enrich_pause)
            time.sleep(enrich_pause)
        for c in company_data.columns:
            if c in enrich_fields or c == 'marketCap':
                company_data[c] = pd.to_numeric(company_data[c], errors='ignore')

    if normalize:
        return stocks_data, company_data, common_start_date, common_end_date
    return stocks_data, company_data


# ---------------------------------------------------------------------------
# Performance / Statistics
# ---------------------------------------------------------------------------
def _normalize_series_idx(s: pd.Series) -> pd.Series:
    s = s.copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.index = pd.to_datetime(s.index).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _pf_returns_series(pf) -> pd.Series:
    r = pf.returns(group_by=True)
    if isinstance(r, pd.DataFrame):
        try:
            if r.shape[1] == 1: r = r.iloc[:, 0]
            else:
                v = pf.value()
                v_sum = v.sum(axis=1) if isinstance(v, pd.DataFrame) else v
                return _normalize_series_idx(v_sum).pct_change().dropna()
        except Exception:
            v = pf.value()
            v_sum = v.sum(axis=1) if isinstance(v, pd.DataFrame) else v
            return _normalize_series_idx(v_sum).pct_change().dropna()
    return _normalize_series_idx(r.dropna().astype(float))


def _prices_to_returns_align(prices: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    p = _normalize_series_idx(prices.dropna().astype(float))
    return p.pct_change().dropna().reindex(idx, method="ffill").fillna(0.0)


def resolve_benchmark_returns(pf, benchmark_mode: str = "internal",
                               benchmark_name: str = "Benchmark",
                               benchmark_data: Optional[pd.Series] = None,
                               benchmark_portfolio=None) -> Tuple[pd.Series, dict]:
    pf_ret = _pf_returns_series(pf)
    idx = pf_ret.index
    meta = {"benchmark_source": None, "benchmark_name": benchmark_name, "notes": ""}

    if benchmark_mode == "portfolio":
        if benchmark_portfolio is None:
            raise ValueError("benchmark_mode='portfolio' requires benchmark_portfolio.")
        bm = benchmark_portfolio.returns(group_by=True)
        if isinstance(bm, pd.DataFrame): bm = bm.sum(axis=1)
        bm = _normalize_series_idx(bm.dropna().astype(float)).reindex(idx, method="ffill").fillna(0.0)
        meta["benchmark_source"] = "portfolio"; return bm, meta

    if benchmark_mode == "external":
        if benchmark_data is None:
            raise ValueError("benchmark_mode='external' requires benchmark_data.")
        meta["benchmark_source"] = "external"
        return _prices_to_returns_align(benchmark_data, idx), meta

    try:
        bm = pf.benchmark_returns(group_by=True)
    except Exception: raise
    if isinstance(bm, pd.DataFrame): bm = bm.sum(axis=1)
    bm = _normalize_series_idx(bm.dropna().astype(float)).reindex(idx, method="ffill").fillna(0.0)
    meta["benchmark_source"] = "internal(pf.benchmark_returns)"
    return bm, meta


def capm_alpha_beta(ret: pd.Series, bm: pd.Series, risk_free_rate: float = 0.02,
                    annualization: int = 252, min_obs: int = 30) -> dict:
    rf_daily = (1 + risk_free_rate) ** (1 / annualization) - 1
    df = pd.concat({"ret": ret, "bm": bm}, axis=1).dropna()
    if df.shape[0] < min_obs:
        return {"alpha_daily": np.nan, "alpha_ann_pct": np.nan, "beta": np.nan,
                "t_alpha": np.nan, "p_alpha": np.nan, "te_ann": np.nan,
                "corr": np.nan, "n_obs": int(df.shape[0])}
    df["excess_ret"] = df["ret"] - rf_daily
    df["excess_bm"]  = df["bm"]  - rf_daily
    model = sm.OLS(df["excess_ret"], sm.add_constant(df["excess_bm"])).fit()
    alpha = float(model.params["const"]); beta = float(model.params["excess_bm"])
    return {"alpha_daily": alpha, "alpha_ann_pct": alpha * annualization * 100.0,
            "beta": beta, "t_alpha": float(model.tvalues["const"]),
            "p_alpha": float(model.pvalues["const"]),
            "te_ann": float(model.resid.std() * np.sqrt(annualization)),
            "corr": float(df["ret"].corr(df["bm"])), "n_obs": int(df.shape[0])}


def rolling_capm_alpha_beta(ret: pd.Series, bm: pd.Series, window: int = 252,
                             min_periods: Optional[int] = None, risk_free_rate: float = 0.02,
                             annualization: int = 252) -> pd.DataFrame:
    if min_periods is None: min_periods = max(60, window // 3)
    df = pd.concat({"ret": ret, "bm": bm}, axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=["alpha_ann_pct", "beta", "t_alpha", "p_alpha"])
    idx = df.index; rows = []; rf_daily = (1 + risk_free_rate) ** (1 / annualization) - 1
    for end in range(window, len(idx) + 1):
        sub = df.loc[idx[end - window:end]]
        if sub.shape[0] < min_periods: continue
        y = sub["ret"] - rf_daily; x = sub["bm"] - rf_daily
        m = sm.OLS(y, sm.add_constant(x)).fit()
        rows.append([idx[end-1], float(m.params["const"]) * annualization * 100.0,
                     float(m.params.iloc[1]), float(m.tvalues["const"]), float(m.pvalues["const"])])
    return (pd.DataFrame(rows, columns=["date", "alpha_ann_pct", "beta", "t_alpha", "p_alpha"])
            .set_index("date"))


def rolling_alpha_section(pf, benchmark_returns: pd.Series, window: int = 252,
                           risk_free_rate: float = 0.02, annualization: int = 252,
                           show_plot: bool = False) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    df_roll = rolling_capm_alpha_beta(_pf_returns_series(pf), benchmark_returns,
                                      window=window, risk_free_rate=risk_free_rate,
                                      annualization=annualization)
    fig = None
    if show_plot and not df_roll.empty:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(df_roll.index, df_roll["alpha_ann_pct"], linewidth=1)
        ax.axhline(0, linestyle="--", linewidth=0.8)
        sig_mask = df_roll["p_alpha"] < 0.05
        if sig_mask.any():
            ax.scatter(df_roll.index[sig_mask], df_roll["alpha_ann_pct"][sig_mask], s=22, zorder=3)
        ax.set_ylabel("Alpha annuo (%)"); ax.set_title("Rolling CAPM Alpha vs Benchmark")
        ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    return df_roll, fig


def create_portfolio_summary_refactored(
    pf, *, sel_tickers=None, alpha_analysis: bool = True,
    risk_free_rate: float = 0.02, benchmark_mode: str = "internal",
    benchmark_name: str = "Benchmark", benchmark_data: Optional[pd.Series] = None,
    benchmark_portfolio=None, annualization: int = 252,
    rolling_window: Optional[int] = 252, show: bool = False,
    return_formatted: bool = False,
):
    from tabulate import tabulate
    stats = pf.stats()
    start_date = stats.get("Start"); end_date = stats.get("End")
    init_invested = stats.get("Start Value"); final_value = stats.get("End Value")
    try: start_str = pd.to_datetime(start_date).date().isoformat()
    except Exception: start_str = str(start_date)
    try: end_str = pd.to_datetime(end_date).date().isoformat()
    except Exception: end_str = str(end_date)
    period_str = f"{start_str} -> {end_str}"

    period_val = stats.get("Period")
    period_days = max(int(period_val.days if hasattr(period_val, "days") else int(period_val)), 1)

    ret_pf = _pf_returns_series(pf)
    cagr = (final_value / init_invested) ** (annualization / period_days) - 1
    total_return = float(pf.total_return(group_by=True))
    volatility   = float(pf.annualized_volatility(group_by=True, freq="1D"))
    max_dd       = float(abs(pf.max_drawdown(group_by=True)))
    rf_daily     = (1 + risk_free_rate) ** (1 / annualization) - 1
    mean_daily   = float(ret_pf.mean()); std_daily = float(ret_pf.std())
    sharpe = (mean_daily - rf_daily) / std_daily * np.sqrt(annualization) if std_daily != 0 else np.nan
    adj_car_calmar = (cagr / max_dd) if max_dd != 0 else np.nan

    if sel_tickers is None:
        total_trades = stats.get("Total Trades", np.nan)
    else:
        total_trades = 0
        dates = sel_tickers.index
        for i in range(1, len(dates)):
            prev_set = set(sel_tickers.loc[dates[i - 1], "tickers"])
            curr_set  = set(sel_tickers.loc[dates[i], "tickers"])
            total_trades += len(curr_set - prev_set) + len(prev_set - curr_set)
    total_months = period_days / 21
    month_op = (float(total_trades) / total_months * (2 if sel_tickers is None else 1)
                if total_months > 0 else np.nan)

    try: exposure = float(pf.gross_exposure().mean())
    except Exception: exposure = np.nan

    def _min_days_to_positive(ret):
        try: L = find_min_positive_period(ret); return "nessuno" if L is None else L
        except Exception: return "n/a"

    def _uw_stats(ret):
        try:
            r = ret.dropna()
            if r.empty: return "n/a", [], "n/a", "n/a"
            eq = (1.0 + r).cumprod(); peak = eq.cummax(); is_uw = (eq / peak - 1.0) < 0
            starts = [i for i in range(1, len(is_uw)) if is_uw.iloc[i] and not is_uw.iloc[i-1]]
            ends   = [i for i in range(1, len(is_uw)) if not is_uw.iloc[i] and is_uw.iloc[i-1]]
            uw_max = max((int((next((c for c in ends if c > s), len(is_uw)-1)) - s) for s in starts), default=0)
            rec_lengths = [int(ends[k] - starts[k]) for k in range(min(len(starts), len(ends)))
                           if ends[k] > starts[k]]
            uw_now = bool(is_uw.iloc[-1])
            uw_now_days = int((len(is_uw)-1) - next((s for s in reversed(starts)), 0)) if uw_now else 0
            return uw_max, rec_lengths, uw_now, uw_now_days
        except Exception: return "n/a", [], "n/a", "n/a"

    L_min = _min_days_to_positive(ret_pf)
    uw_max, rec_lengths, uw_now, uw_now_days = _uw_stats(ret_pf)
    if isinstance(rec_lengths, list) and len(rec_lengths) > 0:
        rec_min = int(np.min(rec_lengths)); rec_med = int(np.median(rec_lengths))
        rec_p90 = int(np.percentile(rec_lengths, 90)); rec_max = int(np.max(rec_lengths))
    else:
        rec_min = rec_med = rec_p90 = rec_max = "nessuno"

    bm_ret, bm_meta = resolve_benchmark_returns(pf, benchmark_mode=benchmark_mode,
                                                 benchmark_name=benchmark_name,
                                                 benchmark_data=benchmark_data,
                                                 benchmark_portfolio=benchmark_portfolio)
    capm = capm_alpha_beta(ret_pf, bm_ret, risk_free_rate=risk_free_rate,
                           annualization=annualization) if alpha_analysis else None
    rolling_df = None
    if alpha_analysis and rolling_window:
        rolling_df = rolling_capm_alpha_beta(ret_pf, bm_ret, window=int(rolling_window),
                                              risk_free_rate=risk_free_rate, annualization=annualization)

    data_entries = [
        ("Periodo", period_str, period_str),
        ("Benchmark source", bm_meta.get("benchmark_source"), bm_meta.get("benchmark_source")),
        ("Benchmark name", bm_meta.get("benchmark_name"), bm_meta.get("benchmark_name")),
        ("Importo investito (€)", init_invested, f"{init_invested:.2f}"),
        ("Valore patrimoniale netto (€)", final_value, f"{final_value:.2f}"),
        ("Giorni di trading", period_days, period_days),
        ("Ritorno totale", total_return, f"{total_return:.2%}"),
        ("CAGR", cagr, f"{cagr:.2%}"),
        ("Max Drawdown", max_dd, f"{max_dd:.2%}"),
        ("Volatilità annua", volatility, f"{volatility:.2%}"),
        ("Rapporto di Sharpe", sharpe, round(sharpe, 2) if pd.notna(sharpe) else "n/a"),
        ("Adjusted CAR (Sharpe-style)", sharpe, round(sharpe, 2) if pd.notna(sharpe) else "n/a"),
        ("Adjusted CAR (Calmar-style)", adj_car_calmar, round(adj_car_calmar, 2) if pd.notna(adj_car_calmar) else "n/a"),
        ("Operazioni al mese", month_op, round(month_op, 2) if pd.notna(month_op) else "n/a"),
        ("Market Exposure (avg gross)", exposure, f"{exposure:.2%}" if pd.notna(exposure) else "n/a"),
        ("Durata minima in guadagno (giorni)", L_min, L_min),
        ("Max Underwater Duration (giorni)", uw_max, uw_max),
        ("Recovery Duration Min (giorni)", rec_min, rec_min),
        ("Recovery Duration Median (giorni)", rec_med, rec_med),
        ("Recovery Duration P90 (giorni)", rec_p90, rec_p90),
        ("Recovery Duration Max (giorni)", rec_max, rec_max),
        ("Underwater Now?", uw_now, uw_now),
        ("Underwater Days Now", uw_now_days, uw_now_days),
    ]
    if alpha_analysis and capm is not None:
        data_entries.extend([
            ("Alpha (giornaliero, assoluto)", capm["alpha_daily"],
             f"{capm['alpha_daily']:.6f}" if pd.notna(capm["alpha_daily"]) else "n/a"),
            ("Alpha annualizzato (%)", capm["alpha_ann_pct"],
             f"{capm['alpha_ann_pct']:.2f}%" if pd.notna(capm["alpha_ann_pct"]) else "n/a"),
            ("Beta (vs benchmark)", capm["beta"], round(capm["beta"], 4) if pd.notna(capm["beta"]) else "n/a"),
            ("Correlazione (vs benchmark)", capm["corr"], round(capm["corr"], 4) if pd.notna(capm["corr"]) else "n/a"),
            ("T-stat Alpha", capm["t_alpha"], round(capm["t_alpha"], 2) if pd.notna(capm["t_alpha"]) else "n/a"),
            ("P-value Alpha", capm["p_alpha"], float(f"{capm['p_alpha']:.2e}") if pd.notna(capm["p_alpha"]) else "n/a"),
            ("Tracking Error (ann)", capm["te_ann"], round(capm["te_ann"], 4) if pd.notna(capm["te_ann"]) else "n/a"),
            ("Obs (alpha reg)", capm["n_obs"], capm["n_obs"]),
        ])

    df_raw = pd.DataFrame({"Valore": [v for _, v, _ in data_entries]},
                           index=[k for k, _, _ in data_entries])
    df_fmt = pd.DataFrame({"Valore": [pv for _, _, pv in data_entries]},
                           index=[k for k, _, _ in data_entries])
    out = {"stats_df_raw": df_raw,
           "stats_df": (df_fmt if return_formatted else df_raw),
           "capm": capm, "rolling_capm": rolling_df,
           "benchmark_meta": bm_meta, "benchmark_returns": bm_ret}
    if show:
        from IPython.display import display
        display(df_fmt)
    return out


def comment_alpha_diagnostics(capm: dict | None, roll: pd.DataFrame | None) -> str:
    def _fmt(x, fmt):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)): return "n/a"
            return format(float(x), fmt)
        except Exception: return "n/a"

    def _is_num(x):
        try: return x is not None and not (isinstance(x, float) and np.isnan(x))
        except Exception: return False

    if capm is None:
        return f"{BOLD}Alpha CAPM non disponibile{RESET}."

    a = capm.get("alpha_ann_pct", np.nan); t = capm.get("t_alpha", np.nan)
    p = capm.get("p_alpha", np.nan);      b = capm.get("beta", np.nan)
    te = capm.get("te_ann", np.nan);      nobs = capm.get("n_obs", np.nan)
    lines = []
    if pd.isna(a) or pd.isna(p) or pd.isna(t):
        lines.append(f"{BOLD}Alpha/Beta non stimabili{RESET}.")
    else:
        sig_05 = p < 0.05; sig_10 = p < 0.10
        if a > 0 and sig_05:    base = f"Alpha {BOLD}positivo e significativo{RESET}."
        elif a > 0 and sig_10:  base = f"Alpha {BOLD}positivo e debolmente significativo{RESET}."
        elif a > 0:             base = f"Alpha {BOLD}positivo ma non significativo{RESET}."
        elif a < 0 and sig_05:  base = f"Alpha {BOLD}negativo e significativo{RESET}."
        elif a < 0 and sig_10:  base = f"Alpha {BOLD}negativo e debolmente significativo{RESET}."
        elif a < 0:             base = f"Alpha {BOLD}negativo ma non significativo{RESET}."
        else:                   base = f"Alpha {BOLD}~0{RESET}."
        beta_msg = f" Beta b={b:.2f}." if pd.notna(b) else ""
        stats_msg = (f" alpha_ann={BOLD}{a:.2f}%{RESET}, T={BOLD}{t:.2f}{RESET}, "
                     f"P={BOLD}{_fmt(p,'.2e')}{RESET}, TE={BOLD}{_fmt(te,'.3f')}{RESET}, "
                     f"Obs={BOLD}{_fmt(nobs,'.0f')}{RESET}.")
        lines.append(base + beta_msg + stats_msg)

    if roll is None or roll.empty:
        lines.append(f"Rolling alpha {BOLD}non disponibile{RESET}.")
    else:
        r = roll.dropna()
        if r.empty: lines.append(f"Rolling alpha {BOLD}tutto NaN{RESET}.")
        else:
            last_a = float(r["alpha_ann_pct"].iloc[-1]) if "alpha_ann_pct" in r.columns else np.nan
            med_a  = float(r["alpha_ann_pct"].median())  if "alpha_ann_pct" in r.columns else np.nan
            sig_pct = float((r["p_alpha"] < 0.05).mean() * 100.0) if "p_alpha" in r.columns else np.nan
            if _is_num(last_a) and _is_num(med_a):
                if last_a > med_a + 0.5:   trend = f"Rolling alpha {BOLD}in miglioramento{RESET}."
                elif last_a < med_a - 0.5: trend = f"Rolling alpha {BOLD}in peggioramento{RESET}."
                else:                       trend = f"Rolling alpha {BOLD}stabile{RESET}."
            else: trend = "Rolling alpha: trend non stimabile."
            nums = f" Ultimo={BOLD}{last_a:.2f}%{RESET}, mediana={med_a:.2f}%." if _is_num(last_a) and _is_num(med_a) else ""
            sig_msg = f" p<0.05 nel {sig_pct:.1f}% dei giorni." if _is_num(sig_pct) else ""
            lines.append(trend + nums + sig_msg)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio weights
# ---------------------------------------------------------------------------
def _compute_weights_from_pf(pf) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pf is None: raise ValueError("pf is None")
    qty = pf.assets()
    if isinstance(qty, pd.Series): qty = qty.to_frame()
    prices = pf.close
    if isinstance(prices, pd.Series): prices = prices.to_frame()
    qty, prices = qty.align(prices, join="inner", axis=0)
    qty, prices = qty.align(prices, join="inner", axis=1)
    asset_values = qty * prices
    total_value = pf.value().reindex(asset_values.index) if hasattr(pf, "value") else asset_values.sum(axis=1)
    weights = asset_values.div(total_value, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if hasattr(pf, "cash"):
        cash = pf.cash().reindex(asset_values.index).fillna(0.0)
        weights["CASH"] = (cash / total_value).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        asset_values["CASH"] = cash
    return weights, asset_values


def _collapse_weights_row(w_row: pd.Series, top_n: int = 12, keep: tuple = ("CASH",)) -> pd.Series:
    w = w_row.copy(); w = w[w.abs() > 0]
    forced = pd.Series(dtype=float)
    for k in keep:
        if k in w.index: forced.loc[k] = w.loc[k]; w = w.drop(index=k)
    w = w.sort_values(ascending=False); top = w.iloc[:max(top_n, 0)]; other = w.iloc[max(top_n, 0):].sum()
    out = pd.concat([top, forced])
    if other > 0: out.loc["OTHER"] = other
    return out.sort_values(ascending=False)


def _snapshot_yearly(weights: pd.DataFrame, shift_trading_days: int = 2,
                     current_label: str = "CURRENT") -> pd.DataFrame:
    idx = weights.index; years = pd.Index(idx.year).unique(); snap_dates = []
    for y in years:
        idx_year = idx[idx.year == y]
        if len(idx_year) == 0: continue
        snap_dates.append(idx_year[min(shift_trading_days, len(idx_year)-1)])
    snap = weights.loc[snap_dates].copy()
    snap.index = pd.Index([d.year for d in snap.index], name="Year")
    current_row = weights.loc[[weights.index[-1]]].copy()
    current_row.index = pd.Index([current_label], name="Year")
    return pd.concat([snap, current_row], axis=0)


def _cash_metrics(cash_w: pd.Series, thresholds=(0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    cw = cash_w.dropna().astype(float)
    stats_dict = {"Cash mean (%)": cw.mean()*100, "Cash median (%)": cw.median()*100,
                  "Cash max (%)": cw.max()*100, "Cash p90 (%)": cw.quantile(0.90)*100,
                  "Cash p95 (%)": cw.quantile(0.95)*100, "Cash p99 (%)": cw.quantile(0.99)*100}
    for th in thresholds:
        stats_dict[f"Days cash > {th*100:.1f}%"] = int((cw > th).sum())
        stats_dict[f"Pct days cash > {th*100:.1f}%"] = (cw > th).mean() * 100
    arr = (cw > 0.01).to_numpy(dtype=bool); best = cur = 0
    for v in arr: cur = (cur+1) if v else 0; best = max(best, cur)
    stats_dict["Max consecutive days cash > 1.0%"] = best
    df = pd.DataFrame.from_dict(stats_dict, orient="index", columns=["Value"])
    df["Value"] = df["Value"].astype(float).round(3)
    return df


def _cash_by_year(cash_w: pd.Series) -> pd.DataFrame:
    cw = cash_w.dropna().astype(float)
    df = pd.DataFrame({"cash_w": cw}); df["year"] = df.index.year
    out = df.groupby("year")["cash_w"].agg(["mean", "max"])
    out = out.rename(columns={"mean": "Cash mean (%)", "max": "Cash max (%)"})
    out["Cash mean (%)"] = (out["Cash mean (%)"]*100).round(2)
    out["Cash max (%)"]  = (out["Cash max (%)"]*100).round(2)
    out.index.name = "Year"
    current = pd.DataFrame({"Cash mean (%)": [np.nan], "Cash max (%)": [round(cw.iloc[-1]*100, 2)]},
                            index=pd.Index(["CURRENT"], name="Year"))
    return pd.concat([out, current], axis=0)


def visualize_portfolio_weights(pf, title="Portfolio Weights", show_report: bool = True,
                                 vbt_plot_width: int | None = None, auto_threshold: int = 12,
                                 top_n: int = 12, shift_trading_days: int = 2, fig_height: int = 1150):
    try: weights, _ = _compute_weights_from_pf(pf)
    except Exception: return None
    if weights.empty: return None

    n_cols = weights.shape[1]; full_mode = n_cols <= auto_threshold
    snap = _snapshot_yearly(weights, shift_trading_days=shift_trading_days)

    if full_mode:
        snap_table = (snap * 100).round(2); weights_plot = weights
    else:
        collapsed = [(_collapse_weights_row(snap.loc[idx_row], top_n=top_n, keep=("CASH",))).rename(idx_row)
                     for idx_row in snap.index]
        snap_c = pd.DataFrame(collapsed).fillna(0.0); snap_c.index.name = "Year"
        snap_table = (snap_c * 100).round(2)
        mean_w = weights.drop(columns=["CASH"], errors="ignore").mean().sort_values(ascending=False)
        top_cols = list(mean_w.iloc[:top_n].index)
        cols_keep = top_cols + (["CASH"] if "CASH" in weights.columns else [])
        w_sub = weights[cols_keep].copy()
        other_cols = [c for c in weights.columns if c not in cols_keep]
        if other_cols: w_sub["OTHER"] = weights[other_cols].sum(axis=1)
        weights_plot = w_sub.div(w_sub.sum(axis=1), axis=0).fillna(0.0)

    if show_report:
        print(snap_table.to_string())
        if "CASH" in weights.columns:
            print(_cash_metrics(weights["CASH"]).to_string())
            print(_cash_by_year(weights["CASH"]).to_string())

    pie_data = {}
    for lbl in snap.index:
        row = snap.loc[lbl]
        row = (row[row.abs() > 0].sort_values(ascending=False) if full_mode
               else _collapse_weights_row(row, top_n=top_n, keep=("CASH",)))
        pie_data[str(lbl)] = (row.index.astype(str).tolist(), row.values.tolist())

    years_only = sorted([k for k in pie_data if k != "CURRENT"], key=lambda x: int(x))
    pie_keys = years_only + (["CURRENT"] if "CURRENT" in pie_data else [])
    default_lbl = "CURRENT" if "CURRENT" in pie_data else pie_keys[-1]
    def _pie_title(lbl): return f"{title} - Allocation - showing: {lbl}"

    fig = make_subplots(rows=3, cols=1,
                        specs=[[{"type": "xy"}], [{"type": "domain"}], [{"type": "xy"}]],
                        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.14,
                        subplot_titles=(f"{title} - Weights evolution", _pie_title(default_lbl),
                                        f"{title} - CASH (%) over time"))
    for col in weights_plot.columns:
        fig.add_trace(go.Scatter(x=weights_plot.index, y=weights_plot[col], mode="lines",
                                  stackgroup="one", name=str(col)), row=1, col=1)
    fig.update_yaxes(title_text="Weight", tickformat=".0%", row=1, col=1)
    fig.add_trace(go.Pie(labels=pie_data[default_lbl][0], values=pie_data[default_lbl][1],
                          hole=0.45, sort=False, textinfo="label+percent", showlegend=False), row=2, col=1)
    if "CASH" in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights["CASH"]*100, mode="lines", name="CASH (%)"),
                      row=3, col=1)
        fig.update_yaxes(title_text="CASH (%)", ticksuffix="%", row=3, col=1)

    active_idx = pie_keys.index(default_lbl) if default_lbl in pie_keys else 0
    pie_ann_idx = next((i for i, a in enumerate(fig.layout.annotations or [])
                        if isinstance(getattr(a, "text", None), str) and "Allocation" in a.text), 1)
    fig.update_layout(
        updatemenus=[dict(type="dropdown", direction="down", x=0.5, xanchor="center",
                          y=0.53, yanchor="top", active=active_idx,
                          buttons=[dict(label=str(lbl), method="update",
                                        args=[{"labels": [pie_data[str(lbl)][0]], "values": [pie_data[str(lbl)][1]]},
                                              {f"annotations[{pie_ann_idx}].text": _pie_title(str(lbl))}])
                                   for lbl in pie_keys])],
        autosize=vbt_plot_width is None, width=int(vbt_plot_width) if vbt_plot_width else None,
        height=int(fig_height), title=dict(text=title, x=0.5),
        hovermode="x unified", legend_title_text="Ticker", margin=dict(l=60, r=60, t=120, b=60))
    return fig


# ---------------------------------------------------------------------------
# Selection comparison
# ---------------------------------------------------------------------------

###############################################################################
# UTILITY: CREAZIONE DELLA MASCHERA DI OPERATIVITÀ
###############################################################################
    
def compare_selection_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str = "Top_Tickers",
    label_a: str = "Selezione A",
    label_b: str = "Selezione B",
    compare_only_common_dates: bool = True,
    sort_table_by_diff: bool = False,
    display_table: bool = True,
):
    """
    Confronta due DataFrame contenenti una colonna di selezioni ticker.

    Ogni cella della colonna deve idealmente contenere una lista di ticker,
    ma la funzione gestisce anche:
      - NaN / None
      - stringhe singole
      - tuple / set / np.ndarray / pd.Index

    Parametri
    ---------
    df1, df2 : pd.DataFrame
        DataFrame da confrontare.

    column : str
        Nome della colonna che contiene le selezioni ticker.

    label_a, label_b : str
        Etichette descrittive delle due selezioni.
        Esempio:
            label_a="con risk on/off"
            label_b="senza risk on/off"

    compare_only_common_dates : bool
        Se True, confronta solo le date presenti in entrambi i DataFrame.
        Se False, mantiene l'unione degli indici e converte eventuali NaN in liste vuote.

    sort_table_by_diff : bool
        Se True, la tabella visualizzata viene ordinata per Diff_Count decrescente.

    display_table : bool
        Se True, visualizza la tabella con ace_tools_open.

    Ritorna
    -------
    df_compare : pd.DataFrame
        DataFrame con selezioni e metriche di confronto.
    """

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from IPython.display import display

    if column not in df1.columns:
        raise ValueError(f"Colonna '{column}' non presente in df1.")

    if column not in df2.columns:
        raise ValueError(f"Colonna '{column}' non presente in df2.")

    def _to_list(x):
        if x is None:
            return []

        if isinstance(x, float) and np.isnan(x):
            return []

        if isinstance(x, (list, tuple, set, np.ndarray, pd.Index)):
            return [item for item in list(x) if pd.notna(item)]

        if isinstance(x, str):
            return [x]

        return []

    # ------------------------------------------------------------
    # Allineamento indici
    # ------------------------------------------------------------
    if compare_only_common_dates:
        common_index = df1.index.intersection(df2.index)

        df_compare = pd.concat(
            [
                df1.loc[common_index, column],
                df2.loc[common_index, column],
            ],
            axis=1
        )
    else:
        df_compare = pd.concat(
            [
                df1[column],
                df2[column],
            ],
            axis=1
        )

    # Colonne tecniche interne
    df_compare.columns = ["Sel_A", "Sel_B"]

    # ------------------------------------------------------------
    # Normalizzazione celle
    # ------------------------------------------------------------
    df_compare["Sel_A"] = df_compare["Sel_A"].apply(_to_list)
    df_compare["Sel_B"] = df_compare["Sel_B"].apply(_to_list)

    set_a = df_compare["Sel_A"].apply(set)
    set_b = df_compare["Sel_B"].apply(set)

    # ------------------------------------------------------------
    # Metriche di confronto
    # ------------------------------------------------------------
    df_compare["In_Common"] = [
        sorted(a & b) for a, b in zip(set_a, set_b)
    ]

    df_compare[f"Solo in {label_a}"] = [
        sorted(a - b) for a, b in zip(set_a, set_b)
    ]

    df_compare[f"Solo in {label_b}"] = [
        sorted(b - a) for a, b in zip(set_a, set_b)
    ]

    df_compare[f"N {label_a}"] = df_compare["Sel_A"].apply(len)
    df_compare[f"N {label_b}"] = df_compare["Sel_B"].apply(len)

    df_compare["N_Common"] = df_compare["In_Common"].apply(len)
    df_compare[f"N solo {label_a}"] = df_compare[f"Solo in {label_a}"].apply(len)
    df_compare[f"N solo {label_b}"] = df_compare[f"Solo in {label_b}"].apply(len)

    df_compare["Union_Count"] = [
        len(a | b) for a, b in zip(set_a, set_b)
    ]

    df_compare["Diff_Count"] = [
        len(a ^ b) for a, b in zip(set_a, set_b)
    ]

    df_compare["Jaccard"] = [
        len(a & b) / len(a | b) if len(a | b) > 0 else np.nan
        for a, b in zip(set_a, set_b)
    ]

    # ------------------------------------------------------------
    # Grafico 1: Jaccard Similarity
    # ------------------------------------------------------------
    fig_jaccard = go.Figure()

    fig_jaccard.add_trace(
        go.Scatter(
            x=df_compare.index,
            y=df_compare["Jaccard"],
            mode="lines+markers",
            name="Similarità Jaccard",
            hovertemplate=(
                "Data: %{x}<br>"
                "Similarità Jaccard: %{y:.2f}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_jaccard.update_layout(
        title=f"Similarità tra selezioni ({column})",
        xaxis_title="Data di Ribilanciamento",
        yaxis_title="Similarità Jaccard",
        yaxis=dict(range=[0, 1.05]),
        width=1000,
        height=470,
        margin=dict(t=60, b=95),
        annotations=[
            dict(
                text=(
                    "Jaccard Similarity = ticker in comune / ticker totali unici tra le due selezioni. "
                    "Valore 1.00 = selezioni identiche; valore 0.00 = nessun ticker in comune."
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=-0.28,
                showarrow=False,
                align="left",
                font=dict(size=11)
            )
        ]
    )

    display(fig_jaccard)

    # ------------------------------------------------------------
    # Grafico 2: composizione differenze
    # ------------------------------------------------------------
    x_labels = df_compare.index.strftime("%Y-%m-%d")

    fig_stack = go.Figure()

    fig_stack.add_trace(
        go.Bar(
            x=x_labels,
            y=df_compare["N_Common"],
            name="In comune",
            hovertemplate=(
                "Data: %{x}<br>"
                "Ticker comuni: %{y}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_stack.add_trace(
        go.Bar(
            x=x_labels,
            y=df_compare[f"N solo {label_a}"],
            name=f"Solo in {label_a}",
            hovertemplate=(
                "Data: %{x}<br>"
                f"Solo in {label_a}: "
                "%{y}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_stack.add_trace(
        go.Bar(
            x=x_labels,
            y=df_compare[f"N solo {label_b}"],
            name=f"Solo in {label_b}",
            hovertemplate=(
                "Data: %{x}<br>"
                f"Solo in {label_b}: "
                "%{y}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_stack.update_layout(
        title=f"Composizione differenze tra selezioni ({column})",
        barmode="stack",
        xaxis_title="Data di Ribilanciamento",
        yaxis_title="Numero ticker",
        width=1100,
        height=500,
        legend_title="Categoria",
    )

    fig_stack.update_xaxes(
        type="category",
        tickangle=45,
    )

    display(fig_stack)

    # ------------------------------------------------------------
    # Tabella dettagliata con nomi leggibili
    # ------------------------------------------------------------
    df_table = df_compare[
        [
            "Sel_A",
            "Sel_B",
            "In_Common",
            f"Solo in {label_a}",
            f"Solo in {label_b}",
            f"N {label_a}",
            f"N {label_b}",
            "N_Common",
            f"N solo {label_a}",
            f"N solo {label_b}",
            "Diff_Count",
            "Union_Count",
            "Jaccard",
        ]
    ].copy()

    df_table = df_table.rename(
        columns={
            "Sel_A": label_a,
            "Sel_B": label_b,
            "In_Common": "In comune",
            "N_Common": "N in comune",
            "Diff_Count": "N diversi",
            "Union_Count": "N unici totali",
            "Jaccard": "Similarità Jaccard",
        }
    )

    if sort_table_by_diff:
        df_table = df_table.sort_values(
            ["N diversi", "Similarità Jaccard"],
            ascending=[False, True]
        )

    if display_table:
        try:
            import ace_tools_open as tools
            tools.display_dataframe_to_user(
                name=f"Confronto selezioni {column}",
                dataframe=df_table
            )
        except ImportError:
            display(df_table)

    return df_compare

# ---------------------------------------------------------------------------
# Min positive period
# ---------------------------------------------------------------------------
def find_min_positive_period(returns: pd.Series) -> int:
    for L in range(1, len(returns) + 1):
        rolling_cum = returns.rolling(window=L).apply(lambda r: np.prod(1+r)-1, raw=True).dropna()
        if (rolling_cum > 0).all(): return L
    return None


# ---------------------------------------------------------------------------
# Wikipedia ticker extraction
# ---------------------------------------------------------------------------
def extract_tickers_from_wikipedia(index: str, exclude: Iterable[str] | None = None,
                                    rename: Mapping[str, str] | None = None) -> List[str]:
    urls = {
        'sp100': "https://en.wikipedia.org/wiki/S%26P_100",
        'sp500': "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        'nasdaq100': "https://en.wikipedia.org/wiki/Nasdaq-100",
        'ftsemib': "https://en.wikipedia.org/wiki/FTSE_MIB",
        'dax': "https://en.wikipedia.org/wiki/DAX",
        'eurostoxx50': "https://en.wikipedia.org/wiki/EURO_STOXX_50",
        'cac40': "https://en.wikipedia.org/wiki/CAC_40",
        'ibex35': "https://en.wikipedia.org/wiki/IBEX_35",
        'nikkei': "https://www.tradingview.com/symbols/TVC-NI225/components",
        'sse50': "https://en.wikipedia.org/wiki/SSE_50_Index",
        'hangseng': "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        'nifty50': "https://en.wikipedia.org/wiki/NIFTY_50",
        'kospi200': "https://en.wikipedia.org/wiki/KOSPI_200",
    }
    candidate_columns = {
        'sp100': ['Symbol','Ticker','Code'], 'sp500': ['Symbol','Ticker'],
        'nasdaq100': ['Ticker','Symbol'],    'ftsemib': ['Ticker','Symbol'],
        'dax': ['Ticker','Symbol'],          'eurostoxx50': ['Ticker','Symbol'],
        'cac40': ['Ticker','Symbol'],        'ibex35': ['Ticker','Symbol'],
        'nikkei': ['Symbol','Code'],         'sse50': ['Ticker symbol','Symbol'],
        'hangseng': ['Ticker','Symbol','Code'], 'nifty50': ['Symbol','Ticker'],
        'kospi200': ['Symbol','Ticker'],
    }
    UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    exclude_set = {str(x).strip().upper() for x in (exclude or []) if x is not None}
    rename_map  = {str(k).strip().upper(): str(v).strip().upper()
                   for k, v in (rename or {}).items() if k and v}

    def http_get(url, tries=3, sleep_s=1.0):
        sess = requests.Session()
        headers = {'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.9'}
        last_err = None
        for i in range(tries):
            try:
                ru = url + ("?action=render" if "wikipedia.org" in url and "?" not in url else "")
                r = sess.get(ru, headers=headers, timeout=20); r.raise_for_status(); return r.text
            except Exception as e: last_err = e; time.sleep(sleep_s*(i+1))
        raise last_err

    def find_table_with_columns(html, cand_cols):
        tables = pd.read_html(StringIO(html), flavor='lxml')
        low_cands = [c.lower() for c in cand_cols]
        for df in tables:
            if any(c in [str(x).strip().lower() for x in df.columns] for c in low_cands):
                return df
        return tables[0] if tables else None

    def resolve_column_name(df, cand_cols):
        cols_map = {str(c).strip().lower(): c for c in df.columns}
        for c in cand_cols:
            if c.lower() in cols_map: return cols_map[c.lower()]
        return df.columns[0]

    def clean_symbols(raw_list):
        out = [re.sub(r"\[[^\]]*\]", "", str(t)).strip().upper() for t in raw_list if t]
        seen, dedup = set(), []
        for t in out:
            if t and t not in seen: seen.add(t); dedup.append(t)
        return dedup

    def postprocess(index_key, syms):
        if index_key == 'sse50':
            return [s.replace("SSE:", "").replace("SSE", "").strip() + ".SS" for s in syms]
        if index_key == 'nifty50':
            return [s if s.endswith(".NS") else s + ".NS" for s in syms]
        if index_key == 'kospi200':
            return [s if s.endswith(".KS") else s + ".KS" for s in syms]
        if index_key == 'nikkei':
            out = [f"{s[:4]}.T" for s in syms if s[:4].isdigit()]
            if not out:
                out = [f"{m.group(1)}.T" for s in syms
                       for m in [re.search(r"\b(\d{4})\b", s)] if m]
            seen, clean = set(), []
            for x in out:
                x = x.strip().upper()
                if x and x not in seen: seen.add(x); clean.append(x)
            return clean
        return syms

    if index not in urls:
        raise ValueError(f"Indice non supportato. Scegli tra: {', '.join(sorted(urls.keys()))}")
    try:
        html = http_get(urls[index])
        df = find_table_with_columns(html, candidate_columns.get(index, []))
        if df is None or df.empty: return []
        col = resolve_column_name(df, candidate_columns.get(index, []))
        syms = clean_symbols(df[col].dropna().astype(str).tolist())
        syms = postprocess(index, syms)
        syms = [rename_map.get(s, s) for s in syms if s and s not in exclude_set]
        return list(dict.fromkeys(syms))
    except Exception as e:
        print(f"Errore nell'estrazione del ticker per {index}: {e}")
        return []


# ---------------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------------
def plot_multiple_portfolios(portfolios: dict, title: str = None, benchmark: str = None,
                              benchmark_data: pd.Series = None, start_date: str = None,
                              end_date: str = None, base: float = 1.0) -> go.Figure:
    def _normalize_dt_index(s):
        s = s.copy()
        try: s.index = pd.DatetimeIndex(s.index)
        except Exception: s.index = pd.to_datetime(s.index, errors="coerce")
        try:
            if getattr(s.index, "tz", None) is not None: s.index = s.index.tz_localize(None)
        except Exception: pass
        return s.normalize() if hasattr(s.index, 'normalize') else s

    clean = {}
    for name, ret in (portfolios or {}).items():
        if ret is None or getattr(ret, "empty", True): continue
        r = ret.copy()
        if isinstance(r, pd.DataFrame):
            if r.shape[1] == 1: r = r.iloc[:, 0]
            else: raise ValueError(f"Portfolio '{name}': returns must be pd.Series")
        r = pd.to_numeric(r, errors="coerce").replace([np.inf, -np.inf], np.nan)
        try:
            if getattr(r.index, "tz", None) is not None: r.index = r.index.tz_localize(None)
        except Exception: pass
        r.index = pd.to_datetime(r.index).normalize()
        if start_date or end_date: r = r.loc[start_date:end_date]
        if not r.empty: clean[name] = r

    if not clean: raise ValueError("Nessun portafoglio valido")
    idxs = [r.index for r in clean.values()]
    port_common_idx = reduce(lambda a, b: a.intersection(b), idxs).sort_values()
    if port_common_idx.empty: raise ValueError("Nessuna data comune")

    if start_date is not None:
        t0_req = pd.Timestamp(start_date).normalize()
        valid = port_common_idx[port_common_idx >= t0_req]
        if valid.empty: raise ValueError("start_date oltre l'ultima data comune")
        t0 = valid[0]
    else:
        t0 = port_common_idx[0]

    bench_ret = None
    if benchmark_data is not None:
        br = benchmark_data.copy()
        if isinstance(br, pd.DataFrame):
            if br.shape[1] == 1: br = br.iloc[:, 0]
        br = pd.to_numeric(br, errors="coerce").replace([np.inf, -np.inf], np.nan)
        try:
            if getattr(br.index, "tz", None) is not None: br.index = br.index.tz_localize(None)
        except Exception: pass
        br.index = pd.to_datetime(br.index).normalize()
        bench_ret = br.pct_change().reindex(port_common_idx, method="ffill").fillna(0.0)
    elif benchmark:
        df_bench = download_data(benchmark, start_date, end_date)
        bench_ret = df_bench.pct_change().reindex(port_common_idx, method="ffill").fillna(0.0)

    fig = go.Figure()
    for name, r in clean.items():
        r_aligned = r.reindex(port_common_idx, method="ffill").fillna(0.0)
        t0_loc = r_aligned.index.get_loc(t0)
        r_aligned.iloc[t0_loc] = 0.0
        eq = (1.0 + r_aligned).cumprod()
        y = (eq - 1.0) * 100 if base == 0.0 else eq * (100 if base == 100 else base)
        fig.add_trace(go.Scatter(x=eq.index, y=y, mode="lines", name=name))

    if bench_ret is not None:
        t0_loc = bench_ret.index.get_loc(t0)
        bench_ret.iloc[t0_loc] = 0.0
        eq_b = (1.0 + bench_ret).cumprod()
        y_b = (eq_b - 1.0) * 100 if base == 0.0 else eq_b * (100 if base == 100 else base)
        fig.add_trace(go.Scatter(x=eq_b.index, y=y_b, mode="lines",
                                  name=benchmark or "Benchmark", line=dict(dash="dash")))
    if title: fig.update_layout(title=title)
    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


def plot_monthly_returns(pf, eoy: bool = True, title: str = "Monthly Returns (%)",
                          width: int | None = None, height: int | None = None,
                          auto_height: bool = True, cell_h: int = 28,
                          min_h: int = 300, max_h: int = 900) -> go.Figure:
    v = pf.value()
    if isinstance(v, pd.DataFrame): v = v.sum(axis=1)
    v = v.dropna().copy()
    try:
        if v.index.tz is not None: v.index = v.index.tz_localize(None)
    except Exception: pass
    v.index = pd.to_datetime(v.index).normalize(); v = v.sort_index()
    if v.index.duplicated().any(): v = v[~v.index.duplicated(keep="last")]
    if v.empty: raise ValueError("pf.value() vuoto.")

    monthly_rows = [(y, m, float(vm.iloc[-1]/vm.iloc[0]-1.0) if len(vm) >= 2 else np.nan)
                    for (y, m), vm in v.groupby([v.index.year, v.index.month])]
    df_m = pd.DataFrame(monthly_rows, columns=["Year", "Month", "Ret"])
    heat = df_m.pivot_table(index="Year", columns="Month", values="Ret", aggfunc="first")
    month_map = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
                 7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
    heat = heat.reindex(columns=list(range(1, 13)))
    heat.columns = [month_map[c] for c in heat.columns]
    years_idx = pd.Index(sorted(v.index.year.unique()), name="Year")
    heat = heat.reindex(index=years_idx)
    if eoy:
        heat["EOY"] = [float(v[v.index.year==y].iloc[-1]/v[v.index.year==y].iloc[0]-1.0)
                       if len(v[v.index.year==y]) >= 2 else np.nan for y in years_idx]
    n_years = len(heat.index)
    if auto_height and height is None:
        height = int(np.clip(120 + cell_h * max(1, n_years), min_h, max_h))
    z = (heat.values * 100.0).astype(float)
    text = np.where(np.isnan(z), "", np.round(z, 2).astype(object))
    fig = go.Figure(data=go.Heatmap(z=z, x=heat.columns.tolist(), y=heat.index.astype(str).tolist(),
                                     colorscale="RdYlGn", zmid=0, colorbar_title="%",
                                     text=text, texttemplate="%{text}"))
    fig.update_layout(title=title, width=width, height=height, showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_annual_return_triangle(pf, resample_freq: str = "YE", run_as_app: bool = False):
    dr = pf.returns()
    if isinstance(dr, pd.DataFrame): dr = dr.iloc[:, 0]
    price_index = (1 + dr).cumprod()
    freq = {"A": "YE", "Y": "YE", "YE": "YE"}.get(resample_freq, "YE")
    yearly_price = price_index.resample(freq).last().to_frame("Price")
    annual_ret = np.log(yearly_price["Price"] / yearly_price["Price"].shift(1)).dropna().to_frame("Return")
    annual_ret.index = (pd.to_datetime(annual_ret.index) - pd.Timedelta(days=1)).year
    annual_ret.index.name = "Year"
    total_years = len(annual_ret)
    windows = list(range(total_years, 0, -1))
    for n in windows:
        annual_ret[f"{n}Y"] = annual_ret["Return"].rolling(window=n).mean()
    triangle_df = annual_ret.drop(columns="Return")
    msg = "Triangolo dei rendimenti medi annualizzati (CAGR)."
    if not run_as_app: print(msg)
    fig = plotly_heatmap_triangle(triangle_df, vmin=-0.2, vmax=0.2, colorscale='RdYlGn', width=900, height=700)
    cols = list(triangle_df.columns)
    tick_colors = ["red" if (triangle_df[col] < 0).any() else "green" for col in cols]
    fig.update_xaxes(tickvals=cols,
                     ticktext=[f"<span style='color:{c}'>{v}</span>" for v, c in zip(cols, tick_colors)])
    return (fig, triangle_df, msg) if run_as_app else (fig, triangle_df)


def plotly_heatmap_triangle(triangle_df: pd.DataFrame, vmin: Optional[float] = None,
                             vmax: Optional[float] = None, colorscale: str = 'RdYlGn',
                             title: Optional[str] = None, width: int = 900, height: int = 700) -> go.Figure:
    df = triangle_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    z = df.values * 100
    text = []
    for i, row in enumerate(z):
        txt_row = []
        end_year = int(df.index[i])
        for j, val in enumerate(row):
            if pd.isna(df.iat[i, j]): txt_row.append("")
            else:
                n = int(str(df.columns[j]).replace("Y", ""))
                txt_row.append(f"{val:.1f}%<br>{end_year - n + 1}-{end_year}")
        text.append(txt_row)
    fig = go.Figure(data=go.Heatmap(z=z, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
                                     text=text, texttemplate="%{text}", colorscale=colorscale,
                                     zmin=(vmin*100) if vmin is not None else None,
                                     zmax=(vmax*100) if vmax is not None else None,
                                     colorbar=dict(title="%")))
    plot_title = title or dict(text="Triangolo dei rendimenti medi annui (CAGR)", x=0.5, xanchor="center")
    fig.update_layout(title=plot_title, width=width, height=height, margin=dict(l=100, r=40, t=80, b=80))
    fig.update_xaxes(title_text="Finestra Mobile (anni)", ticks="outside")
    fig.update_yaxes(title_text="Anno di Fine Finestra", autorange="reversed", ticks="outside", side="left")
    return fig


def plot_ts_portfolio(final_portfolio, portfolio_ts, portfolio_title: str,
                       width: int = 1100, start_date=None, end_date=None) -> go.Figure:
    cumulative_intermedi = pd.DataFrame()
    for ts in portfolio_ts:
        symbol = ts["symbol"]; p = ts["portfolio"]
        cum = 1.0 + p.cumulative_returns()
        if start_date is not None:
            cum = cum[cum.index >= pd.to_datetime(start_date)]
            if not cum.empty: cum /= cum.iloc[0]
        if end_date is not None:
            cum = cum[cum.index <= pd.to_datetime(end_date)]
        cumulative_intermedi[symbol] = cum

    cum_final = 1.0 + final_portfolio.cumulative_returns()
    if start_date is not None:
        cum_final = cum_final[cum_final.index >= pd.to_datetime(start_date)]
        if not cum_final.empty: cum_final /= cum_final.iloc[0]
    if end_date is not None:
        cum_final = cum_final[cum_final.index <= pd.to_datetime(end_date)]

    fig = go.Figure()
    for col in cumulative_intermedi.columns:
        fig.add_trace(go.Scatter(x=cumulative_intermedi.index, y=cumulative_intermedi[col],
                                  mode='lines', name=f"TS - {col}", line=dict(width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=cum_final.index, y=cum_final, mode='lines',
                              name=f"Portfolio {portfolio_title}", line=dict(width=4, color='blue')))
    fig.add_hline(y=1, line=dict(color='red', width=2, dash='dash'))
    fig.update_layout(title=f"Portfolio {portfolio_title} e Trading System", width=width,
                       height=600, template="plotly_white",
                       legend=dict(x=1.02, y=1, xanchor='left', yanchor='auto'))
    return fig


def print_dict_kv(d: dict, indent: int = 0):
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict): print(f"{pad}{k}:"); print_dict_kv(v, indent+2)
        else: print(f"{pad}{k}: {BOLD}{v}{RESET}")


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
def load_email_credentials(secrets_file: str = 'config/tslab_secrets.json') -> tuple:
    import json as _json, os as _os
    _release_config = _os.path.join(_os.environ.get('IQ_INPUTS_DIR', ''), '..', 'config', 'tslab_secrets.json')
    for candidate in [secrets_file, _os.path.join('../../', secrets_file), _release_config]:
    # for candidate in [secrets_file, _os.path.join('../../', secrets_file)]:
        if _os.path.exists(candidate):
            sec = _json.load(open(candidate))
            return sec['sender_email'], sec['sender_password']
    print(f"WARNING: {secrets_file} non trovato.")
    return (_os.environ.get('TSLAB_SENDER_EMAIL', ''), _os.environ.get('TSLAB_SENDER_PASSWORD', ''))


def send_report_via_gmail(sender_email: str, sender_password: str, recipient_email: str,
                           subject: str, body_text: str, attachments: list = None):
    if attachments is None: attachments = []
    msg = MIMEMultipart()
    msg['From'] = sender_email; msg['To'] = recipient_email; msg['Subject'] = subject
    msg.attach(MIMEText(body_text, 'html'))
    for file_path in attachments:
        try:
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={file_path.split('/')[-1]}")
            msg.attach(part)
        except Exception as e:
            print(f"Impossibile allegare {file_path}: {e}")
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        try:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print("Mail inviata.")
        except Exception: print("Invio mail fallito.")


def send_email_report(sender_email, sender_password, recipient_email, subject, body_text, attachments):
    if recipient_email:
        for to in recipient_email.split(','):
            send_report_via_gmail(sender_email=sender_email, sender_password=sender_password,
                                   recipient_email=to, subject=subject, body_text=body_text,
                                   attachments=attachments)


def save_figures_to_html_sequential(figs: list, output_filename: str,
                                     figure_titles: list | None = None,
                                     fig_width: int | None = None, fig_height: int | None = None):
    n = len(figs)
    titles = (figure_titles or [])[:n] + [""] * max(0, n - len(figure_titles or []))
    valid_pairs = [(fig, title) for fig, title in zip(figs, titles)
                   if fig is not None and hasattr(fig, 'data')]
    if not valid_pairs: raise ValueError("Nessuna figura Plotly valida in input")
    html = ["<html>", "<head>", "  <meta charset='utf-8'/>",
            "  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            "</head>", "<body>"]
    for idx, (fig, title) in enumerate(valid_pairs):
        if title: html.append(f"<h2>{title}</h2>")
        if fig_width or fig_height:
            fig.update_layout(width=fig_width or fig.layout.width, height=fig_height or fig.layout.height)
        html.append(fig.to_html(full_html=False, include_plotlyjs='cdn' if idx == 0 else False))
        html.append("<hr style='margin:40px 0;'/>")
    html += ["</body>", "</html>"]
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML Plots salvato in: {output_filename}")


def send_portfolio_performance(sender_email: str, sender_password: str, recipient_email: str, *,
                                assets: dict, html_output: str | None = None,
                                figure_titles: list | None = None, subject: str | None = None):
    if html_output is None:
        html_output = f"{_TSLAB_OUTPUTS_DIR}/report_figures.html"
    figs = assets.get("figs", []); header = assets.get("header", "Report performance")
    stats_df = assets["stats_df"]; sintesi_df = assets["sintesi_df"]
    performance_info = assets.get("performance_info"); performance_tables = assets.get("performance_tables")
    if subject is None: subject = f"[TS_LAB] {header}"
    save_figures_to_html_sequential(figs=figs, output_filename=html_output, figure_titles=figure_titles)
    attach_name = os.path.basename(html_output)

    def _render_table(df, *, index=True):
        return (df.to_html(index=index, border=0, escape=False)
                .replace('<table ', '<table style="border-collapse:collapse; margin:1em 0;" ')
                .replace('<th>', '<th style="background:#f4f4f4; padding:6px; border:1px solid #ccc;">')
                .replace('<td>', '<td style="padding:6px; border:1px solid #ccc;">'))

    def _render_pi(info):
        if info is None: return ""
        if isinstance(info, str): return f"<p>{_html.escape(info).replace(chr(10), '<br>')}</p>"
        if isinstance(info, (list, tuple)):
            return "<ul>" + "".join(f"<li>{_html.escape(str(x))}</li>" for x in info if x) + "</ul>"
        return f"<p>{_html.escape(str(info))}</p>"

    def _render_pt(tables):
        if not isinstance(tables, dict) or not tables: return ""
        return "".join(f"<h4>{_html.escape(str(t))}</h4>" +
                       (_render_table(df) if isinstance(df, pd.DataFrame) else f"<p>{_html.escape(str(df))}</p>")
                       for t, df in tables.items())

    pi = _render_pi(performance_info); pt = _render_pt(performance_tables)
    perf_section = f"<h3>Note di lettura performance</h3>{pi}{pt}" if pi or pt else ""

    html_report = (f"<html><head><meta charset='utf-8'></head><body>"
                   f"<h2>{header}</h2><h3>Sintesi</h3>{_render_table(sintesi_df, index=False)}"
                   f"<h3>Dettaglio</h3>{_render_table(stats_df, index=True)}{perf_section}"
                   f"<h3>Grafici interattivi</h3><p>In allegato <b>{attach_name}</b>.</p>"
                   f"</body></html>")
    send_email_report(sender_email, sender_password, recipient_email, subject, html_report, [html_output])


# ---------------------------------------------------------------------------
# Portfolio performance wrappers (public API)
# ---------------------------------------------------------------------------


def generate_portfolio_performance(pf, portfolio_title: str, pf_b_h=None,
                                    portfolio_ts=None, alpha_analysis: bool = True,
                                    benchmark=None, benchmark_data=None,
                                    plot_start_date=None, plot_end_date=None,
                                    show_report: bool = True, show_plots: bool = False) -> dict:
    return _generate_portfolio_performance_core_refactored(
        pf=pf, portfolio_title=portfolio_title, mode="standard", portfolio_ts=portfolio_ts,
        pf_b_h=pf_b_h, benchmark_mode=("external" if benchmark_data is not None else "internal"),
        benchmark=benchmark, benchmark_data=benchmark_data, alpha_analysis=alpha_analysis,
        show_report=show_report, show_plots=show_plots)


def generate_rotational_portfolio_performance(pf=None, portfolio_title: str = None, sel_tickers=None,
                                               results_pipeline=None, engine=None,
                                               variant_scelta: str = "RISK_ON_OFF",
                                               benchmark: str = 'SPY', benchmark_data=None,
                                               plot_start_date=None, plot_end_date=None,
                                               method=None, freq=None, alpha_analysis: bool = True,
                                               show_report: bool = True, show_plots: bool = False,
                                               universe=[]) -> dict:
    # Risoluzione pf/sel_tickers da results_pipeline+engine, se forniti (alternativa
    # a passare pf/sel_tickers gia' estratti a mano dal chiamante).
    if results_pipeline is not None and engine is not None:
        _valid_variants = {"RISK_ON_OFF", "BASE"}
        if variant_scelta not in _valid_variants:
            raise ValueError(
                f"variant_scelta='{variant_scelta}' non valido. "
                f"Valori possibili: {sorted(_valid_variants)}"
            )
        if engine not in results_pipeline:
            raise ValueError(
                f"engine {engine!r} non presente in results_pipeline "
                f"(disponibili: {list(results_pipeline.keys())})"
            )
        if variant_scelta == "BASE":
            pf = results_pipeline[engine]["pf_rot_base"]
            sel_tickers = results_pipeline[engine]["sel_tickers_base"]
        else:
            pf = results_pipeline[engine]["pf_rot"]
            sel_tickers = results_pipeline[engine]["sel_tickers"]
    elif pf is None:
        raise ValueError(
            "generate_rotational_portfolio_performance: fornire pf esplicito "
            "oppure results_pipeline + engine."
        )

    return _generate_portfolio_performance_core_refactored(
        pf=pf, portfolio_title=portfolio_title, mode="rotational", sel_tickers=sel_tickers,
        pf_b_h=None, benchmark_mode=("external" if benchmark_data is not None else "internal"),
        benchmark=benchmark, benchmark_data=benchmark_data, alpha_analysis=alpha_analysis,
        show_report=show_report, show_plots=show_plots, universe=universe)


def generate_lazy_portfolio_performance(pf, portfolio_title: str, benchmark: str = 'SPY',
                                         benchmark_data=None, method=None, freq=None,
                                         alpha_analysis: bool = True, show_report: bool = True,
                                         show_plots: bool = False) -> dict:
    return _generate_portfolio_performance_core_refactored(
        pf=pf, portfolio_title=portfolio_title, mode="lazy", pf_b_h=None,
        benchmark_mode=("external" if benchmark_data is not None else "internal"),
        benchmark=benchmark, benchmark_data=benchmark_data, alpha_analysis=alpha_analysis,
        show_report=show_report, show_plots=show_plots)
"""
u_functions_patch.py
Patch per u_functions.py:
  1. Funzioni grafiche mancanti (plot_cumulative_and_rolling_returns, 
     plot_annual_performance, plot_year_returns_histogram,
     plot_ticker_frequencies, plot_total_return_per_ticker)
  2. Versione completa di _generate_portfolio_performance_core_refactored

Istruzioni:
  - Appendi questo file a notebooks/libs_py/u_functions.py
  - Poi commenta/rimuovi la vecchia _generate_portfolio_performance_core_refactored
    (circa riga 1240-1370 dell'attuale u_functions.py)
"""

# ============================================================
# SEZIONE 1: Funzioni grafiche mancanti (da cell 6 originale)
# ============================================================

###############################################################################
# Grafica: funzioni di plot
###############################################################################
def plot_multiple_portfolios(
    portfolios: dict[str, "pd.Series"],
    title: str = None,
    benchmark: str = None,
    benchmark_data: "pd.Series" = None,
    start_date: str = None,
    end_date: str = None,
    base: float = 1.0,   # 1.0=rebased to 1, 0.0=cum return, 100=base 100
) -> "go.Figure":
    """
    Confronta più portafogli e, opzionalmente, un benchmark.

    FIX principali rispetto alla versione precedente:
    - NON droppa i NaN dei returns in input (evita di spostare t0 in avanti)
    - costruisce un indice comune e allinea TUTTE le serie su tale indice
    - impone r[t0] = 0.0 per far partire l'equity-index esattamente dal punto base
    - sceglie t0 coerente con start_date se fornita (così la curva finale combacia con le stats)
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from functools import reduce

    # -------------------------
    # 0) Helper: normalize index
    # -------------------------
    def _normalize_dt_index(s: pd.Series) -> pd.Series:
        s = s.copy()
        try:
            s.index = pd.DatetimeIndex(s.index)
        except Exception:
            s.index = pd.to_datetime(s.index, errors="coerce")
        try:
            # rimuovi tz se presente
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
        except Exception:
            pass
        s.index = pd.DatetimeIndex(s.index).normalize()
        return s

    # -------------------------
    # 1) Pulisci i returns dei portafogli (SENZA dropna)
    # -------------------------
    clean: dict[str, pd.Series] = {}
    for name, ret in (portfolios or {}).items():
        if ret is None or getattr(ret, "empty", True):
            continue

        r = ret.copy()

        # se arriva DataFrame a 1 colonna, schiaccia
        if isinstance(r, pd.DataFrame):
            if r.shape[1] == 1:
                r = r.iloc[:, 0]
            else:
                raise ValueError(f"Portafoglio '{name}': returns devono essere pd.Series (non DataFrame multi-colonna)")

        # numeric + pulizia inf
        r = pd.to_numeric(r, errors="coerce").replace([np.inf, -np.inf], np.nan)

        # normalize index + slice
        r = _normalize_dt_index(r)
        if start_date or end_date:
            r = r.loc[start_date:end_date]

        # NON dropna: il primo NaN è normale nei returns
        if not r.empty:
            clean[name] = r

    if not clean:
        raise ValueError("Nessun portafoglio valido da plottare")

    # -------------------------
    # 2) Indice comune solo fra portafogli (intersection)
    # -------------------------
    idxs = [r.index for r in clean.values()]
    port_common_idx = reduce(lambda a, b: a.intersection(b), idxs).sort_values()

    if port_common_idx.empty:
        raise ValueError("Nessuna data comune fra i portafogli")

    # -------------------------
    # 2b) t0 coerente con start_date se fornita
    # -------------------------
    if start_date is not None:
        t0_req = pd.Timestamp(start_date)
        try:
            if getattr(t0_req, "tzinfo", None) is not None:
                t0_req = t0_req.tz_localize(None)
        except Exception:
            pass
        t0_req = pd.Timestamp(t0_req).normalize()

        # prima data disponibile >= richiesta
        valid = port_common_idx[port_common_idx >= t0_req]
        if valid.empty:
            raise ValueError("start_date oltre l'ultima data comune fra i portafogli")
        t0 = valid[0]
    else:
        t0 = port_common_idx[0]

    # -------------------------
    # 3) Carica e riallinea il benchmark (se richiesto)
    # -------------------------
    bench_ret = None
    if benchmark_data is not None:
        br = benchmark_data.copy()

        if isinstance(br, pd.DataFrame):
            if br.shape[1] == 1:
                br = br.iloc[:, 0]
            else:
                raise ValueError("benchmark_data deve essere pd.Series (non DataFrame multi-colonna)")

        br = pd.to_numeric(br, errors="coerce").replace([np.inf, -np.inf], np.nan)
        br = _normalize_dt_index(br)

        # se benchmark_data è PRICE, trasformo in returns; se è già returns, l'utente deve passarli coerenti.
        # Qui mantengo la logica originale: assumo PRICE.
        br = br.pct_change()

        br_aligned = br.reindex(port_common_idx, method="ffill")
        br_aligned = br_aligned.fillna(0.0)
        bench_ret = br_aligned.rename("Benchmark")

    elif benchmark:
        df_bench = download_data(benchmark, start_date, end_date)  # assume esista nel tuo contesto
        br = df_bench.copy()
        if isinstance(br, pd.DataFrame):
            # prova a scegliere la colonna più probabile
            if "Close" in br.columns:
                br = br["Close"]
            else:
                br = br.iloc[:, 0]
        br = pd.to_numeric(br, errors="coerce").replace([np.inf, -np.inf], np.nan)
        br = _normalize_dt_index(br)
        br = br.pct_change()

        br_aligned = br.reindex(port_common_idx, method="ffill")
        br_aligned = br_aligned.fillna(0.0)
        bench_ret = br_aligned.rename(benchmark)

    # -------------------------
    # 4) Costruisci figura
    # -------------------------
    fig = go.Figure()

    # -------------------------
    # 4b) Helper: curva rebased coerente con t0 (e con Total Return)
    # -------------------------
    def _rebased_curve_from_returns(r: pd.Series) -> pd.Series:
        # allinea PRIMA
        r = r.reindex(port_common_idx)

        # forza r[t0] = 0 per partire dal base
        if t0 in r.index:
            r.loc[t0] = 0.0

        # buchi -> 0%
        r = r.fillna(0.0)

        # equity index
        eq = (1.0 + r).cumprod()

        # rebased a 1 su t0
        eq0 = eq.loc[t0]
        if eq0 == 0 or np.isnan(eq0):
            eq0 = 1.0
        reb1 = eq / eq0

        # convert to requested base
        if base == 1.0:
            y = reb1
        elif base == 0.0:
            y = reb1 - 1.0
        else:
            y = reb1 * float(base)

        return y

    # -------------------------
    # 5) Aggiungi portafogli
    # -------------------------
    for name, r in clean.items():
        y = _rebased_curve_from_returns(r)
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values,
            mode="lines", name=f"Portfolio ({name})"
        ))

    # -------------------------
    # 6) Aggiungi benchmark
    # -------------------------
    if bench_ret is not None:
        yb = _rebased_curve_from_returns(bench_ret)
        fig.add_trace(go.Scatter(
            x=yb.index, y=yb.values,
            mode="lines",
            name=f"Benchmark ({benchmark})" if benchmark else "Benchmark",
            line=dict(color="silver"),
            opacity=0.8
        ))

    # -------------------------
    # 7) Linea base coerente
    # -------------------------
    hline_y = 1.0 if base == 1.0 else (0.0 if base == 0.0 else float(base))
    fig.add_hline(
        y=hline_y,
        line=dict(color="red", width=2, dash="dash"),
    )

    # -------------------------
    # 8) Asse Y: formattazione coerente con base
    # -------------------------
    if base == 0.0:
        y_title = "Cumulative Return (%)"
        fig.update_yaxes(tickformat=".1%")
    elif base == 1.0:
        y_title = "Cumulative Return (rebased to 1)"
        fig.update_yaxes(tickformat=".3f")
    else:
        y_title = f"Index (base {base:g})"
        fig.update_yaxes(tickformat=".1f" if float(base) % 1 else ".0f")

    fig.update_layout(
        title=title or ("Performance vs Benchmark" if bench_ret is not None else "Performance Portafogli"),
        xaxis_title="Data",
        yaxis_title=y_title,
        hovermode="x unified",
        height=600,
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="All")
            ])
        ),
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="auto"
        ),
        margin=dict(t=100),
        template="plotly_white"
    )

    return fig
    
    
def plot_multiple_portfolios_R1(
    portfolios: dict[str, pd.Series],
    title: str = None,
    benchmark: str = None,
    benchmark_data: pd.Series = None,
    start_date: str = None,
    end_date: str = None,
    base: float = 1.0,   # <<< NEW: valore base (default 1.0)
) -> go.Figure:
    """
    Confronta più portafogli e, opzionalmente, un benchmark.
    Le curve dei portafogli sono ribasate al primo giorno comune fra tutti i portafogli.

    base:
      - 1.0   => "1 unità investita" (default storico)
      - 0.0   => cumulative return (0 = 0%)
      - 100.0 => indice base 100
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from functools import reduce

    # 1) Pulisci i returns dei portafogli
    clean = {}
    for name, ret in portfolios.items():
        if ret is None or getattr(ret, "empty", True):
            continue
        # r = ret.dropna().copy()
        r = ret.copy()
        r = r.replace([np.inf, -np.inf], np.nan)
        # NON droppare: il primo NaN dei returns è normale e serve per l'allineamento
        try:
            r.index = r.index.tz_localize(None)
        except Exception:
            pass
        r.index = r.index.normalize()
        if start_date or end_date:
            r = r.loc[start_date:end_date]
        if not r.empty:
            clean[name] = r

    if not clean:
        raise ValueError("Nessun portafoglio valido da plottare")

    # 2) Indice comune solo fra portafogli
    idxs = [r.index for r in clean.values()]
    port_common_idx = reduce(lambda a, b: a.intersection(b), idxs).sort_values()
    if port_common_idx.empty:
        raise ValueError("Nessuna data comune fra i portafogli")

    
    t0 = port_common_idx[0]
    
    # 3) Carica e riallinea il benchmark (se richiesto)
    bench_ret = None
    if benchmark_data is not None:
        br = benchmark_data.pct_change().dropna()
        try:
            br.index = br.index.tz_localize(None)
        except Exception:
            pass
        br.index = br.index.normalize()
        br_aligned = br.reindex(port_common_idx, method="ffill").fillna(0.0)
        bench_ret = br_aligned.rename("Benchmark")
    elif benchmark:
        df_bench = download_data(benchmark, start_date, end_date)
        br = df_bench.pct_change().dropna()
        try:
            br.index = br.index.tz_localize(None)
        except Exception:
            pass
        br.index = br.index.normalize()
        br_aligned = br.reindex(port_common_idx, method="ffill").fillna(0.0)
        bench_ret = br_aligned.rename(benchmark)

    # 4) Costruisci figura
    fig = go.Figure()

    def _rebased_curve_from_returns(r: pd.Series) -> pd.Series:
        # forza Series
        if isinstance(r, pd.DataFrame):
            if r.shape[1] == 1:
                r = r.iloc[:, 0]
            else:
                raise ValueError("plot_multiple_portfolios: returns devono essere pd.Series")
    
        # allinea PRIMA
        r = r.reindex(port_common_idx)
    
        # il primo return deve essere 0 per far partire l'indice da 1 (o base)
        # (tipicamente è NaN perché non esiste t-1)
        if len(r) > 0:
            r.iloc[0] = 0.0
    
        # buchi -> 0% (coerenza)
        r = r.fillna(0.0)
    
        # equity index (parte da 1)
        eq = (1.0 + r).cumprod()
    
        # converti base
        if base == 1.0:
            y = eq
        elif base == 0.0:
            y = eq - 1.0
        else:
            y = eq * float(base)
    
        return y    
    # def _rebased_curve_from_returns(r: pd.Series) -> pd.Series:
    #     full_cum = (1 + r).cumprod()
    #     # rebased to 1 at t0
    #     reb1 = full_cum / full_cum.loc[t0]
    #     # convert to requested base
    #     if base == 1.0:
    #         y = reb1
    #     elif base == 0.0:
    #         y = reb1 - 1.0
    #     else:
    #         y = reb1 * float(base)
    #     return y.reindex(port_common_idx)

    # 5) Aggiungi portafogli
    for name, r in clean.items():
        y = _rebased_curve_from_returns(r)
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values,
            mode="lines", name=f"Portfolio ({name})"
        ))

    # 6) Aggiungi benchmark
    if bench_ret is not None:
        yb = _rebased_curve_from_returns(bench_ret)
        fig.add_trace(go.Scatter(
            x=yb.index, y=yb.values,
            mode="lines",
            name=f"Benchmark ({benchmark})" if benchmark else "Benchmark",
            line=dict(color="silver"),
            opacity=0.8
        ))

    # --- linea base coerente ---
    hline_y = 1.0 if base == 1.0 else (0.0 if base == 0.0 else float(base))
    fig.add_hline(
        y=hline_y,
        line=dict(color="red", width=2, dash="dash"),
    )

    # --- asse Y: formattazione coerente con base ---
    if base == 0.0:
        y_title = "Cumulative Return (%)"
        fig.update_yaxes(tickformat=".1%")
    elif base == 1.0:
        y_title = "Cumulative Return (rebased to 1)"
        fig.update_yaxes(tickformat=".3f")
    else:
        y_title = f"Index (base {base:g})"
        fig.update_yaxes(tickformat=".1f" if float(base) % 1 else ".0f")

    fig.update_layout(
        title=title or ("Performance vs Benchmark" if bench_ret is not None else "Performance Portafogli"),
        xaxis_title="Data",
        yaxis_title=y_title,
        hovermode="x unified",
        # width=vbt_plot_width,
        height=600,
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="All")
            ])
        ),
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="auto"
        ),
        margin=dict(t=100),
        template="plotly_white"
    )

    return fig

def plot_annual_performance(
    portfolios_returns: dict,
    benchmark: str = None,
    benchmark_data: pd.Series = None,
    title: str = None,
    risk_metric: str = "vol",   # "vol" | "downside" | "maxdd"
    trading_days: int = 252
) -> go.Figure:
    """
    Rendimenti annuali (%) da returns + istogramma rischio annuale.

    Regola CORRETTA (intra-year):
      - per ogni anno, compone SOLO i returns il cui giorno precedente è nello stesso anno
        => esclude il primo return dell'anno (cross-year).
    Nessun allineamento tra serie.

    risk_metric:
      - "vol": volatilità annualizzata su returns intra-year
      - "downside": downside deviation annualizzata (solo returns < 0)
      - "maxdd": max drawdown intra-year (equity costruita da returns intra-year)
    """

    data = portfolios_returns.copy()

    # if title is None:
    #     title = "Rendimenti annuali (%) + Rischio annuale"

    if title is None:
        title = f"Rendimenti annuali (%) + Rischio annuale ({risk_metric})"

    # --- benchmark (se fornito come prezzi) -> returns ---
    bench_name = None
    if benchmark_data is not None:
        bh_ret = benchmark_data.pct_change().dropna()
        bench_name = f"Benchmark ({benchmark})" if benchmark else "Benchmark"
        bh_ret.name = bench_name
        data[bench_name] = bh_ret
    elif benchmark:
        try:
            idx_all = pd.DatetimeIndex(
                np.concatenate([s.index.values for s in portfolios_returns.values()
                                if s is not None and len(s) > 0])
            )
            start = idx_all.min().strftime("%Y-%m-%d")
            end   = (idx_all.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            bh_close = download_data(benchmark, start_date=start, end_date=end)
            bh_ret   = bh_close.pct_change().dropna()
            bench_name = benchmark
            bh_ret.name = bench_name
            data[bench_name] = bh_ret
        except Exception:
            pass

    def _clean_returns(x):
        if x is None:
            return pd.Series(dtype=float)

        # DataFrame -> mean(axis=1) (compat originale)
        if isinstance(x, pd.DataFrame):
            s = x.mean(axis=1).dropna()
        else:
            s = x.dropna().copy()

        # normalize index
        try:
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
        except Exception:
            pass
        s.index = pd.to_datetime(s.index).normalize()

        # sort + drop dup dates
        s = s.sort_index()
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep="last")]

        # numeric
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s

    def _intra_year_mask(r: pd.Series) -> np.ndarray:
        """True solo per returns il cui giorno precedente è nello stesso anno."""
        yrs = r.index.year
        prev_yrs = pd.Series(yrs, index=r.index).shift(1)
        return (pd.Series(yrs, index=r.index) == prev_yrs).fillna(False).values

    def _annual_intra_year_return(r: pd.Series) -> pd.Series:
        """Compound per anno escludendo il cross-year return (primo return dell'anno)."""
        if r is None or r.empty:
            return pd.Series(dtype=float)

        mask = _intra_year_mask(r)
        r_intra = r[mask]
        if r_intra.empty:
            return pd.Series(dtype=float)

        ann = r_intra.groupby(r_intra.index.year).apply(lambda seg: (1.0 + seg).prod() - 1.0)
        ann.index = ann.index.astype(int)
        return ann.sort_index()

    def _annual_intra_year_risk(r: pd.Series) -> pd.Series:
        """Rischio per anno su returns intra-year."""
        if r is None or r.empty:
            return pd.Series(dtype=float)

        mask = _intra_year_mask(r)
        r_intra = r[mask]
        if r_intra.empty:
            return pd.Series(dtype=float)

        def _year_risk(seg: pd.Series) -> float:
            seg = seg.dropna()
            if len(seg) < 2:
                return np.nan

            if risk_metric == "vol":
                return float(seg.std(ddof=1) * np.sqrt(trading_days))

            if risk_metric == "downside":
                dn = seg[seg < 0]
                if len(dn) < 2:
                    return 0.0
                return float(dn.std(ddof=1) * np.sqrt(trading_days))

            if risk_metric == "maxdd":
                eq = (1.0 + seg).cumprod()
                dd = (eq / eq.cummax()) - 1.0
                return float(dd.min())  # valore negativo (es: -0.18)

            raise ValueError("risk_metric deve essere: 'vol', 'downside' o 'maxdd'")

        out = r_intra.groupby(r_intra.index.year).apply(_year_risk)
        out.index = out.index.astype(int)
        return out.sort_index()

    # --- calcolo annuale per tutti (portafogli + benchmark) ---
    annual_ret = {}
    annual_risk = {}
    for name, series in data.items():
        r = _clean_returns(series)
        annual_ret[name] = _annual_intra_year_return(r)
        annual_risk[name] = _annual_intra_year_risk(r)

    annual_ret_df = pd.DataFrame(annual_ret).sort_index()
    annual_risk_df = pd.DataFrame(annual_risk).sort_index()

    # --- filtro anni presenti in almeno un PORTAFOGLIO (escludi anni solo benchmark) ---
    port_names = list(portfolios_returns.keys())
    port_years = set()
    for nm in port_names:
        if nm in annual_ret_df.columns:
            port_years |= set(annual_ret_df.index[~annual_ret_df[nm].isna()])
    if port_years:
        years_sorted = sorted(port_years)
        annual_ret_df = annual_ret_df.loc[annual_ret_df.index.isin(years_sorted)]
        annual_risk_df = annual_risk_df.loc[annual_risk_df.index.isin(years_sorted)]

    # --- labels rischio ---
    if risk_metric == "vol":
        risk_title = f"Risk (Vol ann., √{trading_days})"
        risk_text_fmt = lambda v: f"{v*100:.2f}%" if pd.notna(v) else ""
    elif risk_metric == "downside":
        risk_title = f"Risk (Downside ann., √{trading_days})"
        risk_text_fmt = lambda v: f"{v*100:.2f}%" if pd.notna(v) else ""
    else:  # maxdd
        risk_title = "Risk (Max Drawdown intra-year)"
        risk_text_fmt = lambda v: f"{v*100:.2f}%" if pd.notna(v) else ""

    # --- plot (2 pannelli) ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Annual Return", risk_title),
    )

    # pannello 1: return
    for col in annual_ret_df.columns:
        y = annual_ret_df[col]
        fig.add_trace(
            go.Bar(
                x=annual_ret_df.index.astype(str),
                y=y,
                name=col if col != bench_name else f"{col}",
                text=(y * 100).round(2).astype(str) + "%",
                textposition="outside",
                opacity=1.0 if col not in [bench_name] else 0.6
            ),
            row=1, col=1
        )

    # pannello 2: risk
    for col in annual_risk_df.columns:
        y = annual_risk_df[col]
        fig.add_trace(
            go.Bar(
                x=annual_risk_df.index.astype(str),
                y=y,
                name=col if col != bench_name else f"{col}",
                text=[risk_text_fmt(v) for v in y.values],
                textposition="outside",
                opacity=1.0 if col not in [bench_name] else 0.6,
                showlegend=False  # evita doppia legenda (già sopra)
            ),
            row=2, col=1
        )
    # if title is None:
    #     title = f"Rendimenti annuali (%) + Rischio annuale ({risk_title})"

    fig.update_layout(
        barmode="group",
        title=title,
        template="plotly_white",
        margin=dict(t=120),
        height=900,
    )
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Risk", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    return fig

    

def plot_monthly_returns_histogram(
    pf,
    title: str = "Istogramma dei rendimenti mensili",
    top_n: int = 3,
    width: int = 1200,
    height: int = 640,
    *,
    panel_width: float = 0.36,   # larghezza pannello destro (0..1)
    gap: float = 0.02,           # gap tra istogramma e pannello
    hist_fill: float = 0.96,     # frazione dello spazio sinistro occupata dall’istogramma
    left_margin_px: int = 28,    # margine sinistro per label Y
):
    """
    Istogramma dei rendimenti mensili (Plotly) + pannello destro.
    Calcolo MENSILE = intra-month compounding sui returns del portafoglio:
        m = prod(1 + r_intra_month) - 1
    dove r_intra_month esclude SEMPRE il primo return del mese (carry-over dal mese precedente).
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # ------------------------------------------------------------
    # 1) Daily returns ROBUSTI dal value totale (coerente con core)
    # ------------------------------------------------------------
    v = pf.value()
    if isinstance(v, pd.DataFrame):
        v = v.sum(axis=1)
    v = v.dropna().copy()

    try:
        if v.index.tz is not None:
            v.index = v.index.tz_localize(None)
    except Exception:
        pass

    v.index = pd.to_datetime(v.index).normalize()
    v = v.sort_index()
    if v.index.duplicated().any():
        v = v[~v.index.duplicated(keep="last")]

    r = v.pct_change().dropna()
    if r.empty:
        raise ValueError("Nessun rendimento disponibile.")

    # ------------------------------------------------------------
    # 2) Monthly returns INTRA-MONTH (escludi primo return del mese)
    # ------------------------------------------------------------
    monthly_list = []
    for (y, m), rm in r.groupby([r.index.year, r.index.month]):
        # rm contiene TUTTI i daily returns nel mese, incluso il primo (carry-over).
        if len(rm) >= 2:
            rm_intra = rm.iloc[1:]
            ret_m = (1.0 + rm_intra).prod() - 1.0
            monthly_list.append((y, m, float(ret_m)))
        else:
            monthly_list.append((y, m, np.nan))

    monthly = pd.Series(
        [x[2] for x in monthly_list],
        index=pd.PeriodIndex([f"{x[0]}-{x[1]:02d}" for x in monthly_list], freq="M"),
        dtype=float
    ).dropna()

    if monthly.empty:
        raise ValueError("Nessun rendimento mensile disponibile.")

    mean_monthly = float(monthly.mean())

    # ------------------------------------------------------------
    # 3) Statistiche e top/bottom
    # ------------------------------------------------------------
    monthly_named = monthly.copy()
    monthly_named.index = monthly_named.index.astype(str)  # "YYYY-MM"

    years = pd.Index([int(s.split("-")[0]) for s in monthly_named.index])
    start_year, end_year = int(years.min()), int(years.max())

    total_months = int(monthly.shape[0])
    up_months = int((monthly > 0).sum())
    pct_up = 100.0 * up_months / total_months if total_months else np.nan

    top_s = monthly_named.sort_values(ascending=False).head(top_n)
    bot_s = monthly_named.sort_values(ascending=True).head(top_n)

    # bins simmetrici (1%)
    max_abs = float(np.ceil(np.max(np.abs(monthly.values)) * 100.0))
    max_abs = max(max_abs, 6.0)
    xbins = dict(start=-max_abs/100.0, end=max_abs/100.0, size=0.01)

    x_pos = monthly[monthly >= 0].values
    x_neg = monthly[monthly < 0].values

    # ------------------------------------------------------------
    # 4) Istogramma
    # ------------------------------------------------------------
    fig = go.Figure()
    fig.add_histogram(
        x=x_pos, xbins=xbins, marker=dict(color="rgb(34,139,34)"),
        hovertemplate="Mesi: %{y}<extra></extra>", showlegend=False
    )
    fig.add_histogram(
        x=x_neg, xbins=xbins, marker=dict(color="rgb(203,67,53)"),
        hovertemplate="Mesi: %{y}<extra></extra>", showlegend=False
    )
    fig.add_vline(x=0.0, line_width=2, line_dash="solid", line_color="rgba(80,80,80,0.6)")

    fig.update_xaxes(
        tickformat=".0%", title_text="Rendimento mensile (intra-month)",
        zeroline=False, range=[xbins["start"], xbins["end"]],
        showgrid=True, gridcolor="rgba(0,0,0,0.06)"
    )
    fig.update_yaxes(
        title_text="Numero di mesi",
        rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.06)"
    )
    fig.update_layout(bargap=0.05)

    # ------------------------------------------------------------
    # 5) Pannello destro (paper coords)
    # ------------------------------------------------------------
    GREEN_BG, GREEN_ACC = "#e8f5e9", "#2e7d32"
    RED_BG, RED_ACC = "#fdecea", "#c62828"
    GRAY = "#6b7280"
    IT_MONTHS = ["gennaio","febbraio","marzo","aprile","maggio","giugno",
                 "luglio","agosto","settembre","ottobre","novembre","dicembre"]

    def pct_str(x, d=1): return f"{x*100:.{d}f}%".replace(".", ",")

    def fmt_month(ym: str):
        # ym: "YYYY-MM"
        y, m = ym.split("-")
        return f"{IT_MONTHS[int(m)-1]} {y}"

    left_space = 1.0 - panel_width - gap
    hist_width = max(0.1, left_space * float(hist_fill))
    hist_left = (left_space - hist_width) / 2.0
    hist_right = hist_left + hist_width
    fig.update_xaxes(domain=[hist_left, hist_right])

    px0 = hist_right + gap
    px1 = px0 + panel_width

    def rect(y0, y1, color):
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=px0, x1=px1, y0=y0, y1=y1,
            fillcolor=color, line=dict(color="rgba(0,0,0,0)")
        )

    def ann(x, y, text, size=14, color="#111", bold=False, anchor="left"):
        fig.add_annotation(
            xref="paper", yref="paper", x=x, y=y,
            xanchor=anchor, yanchor="middle",
            text=(f"<b>{text}</b>" if bold else text),
            showarrow=False, font=dict(size=size, color=color), align="left"
        )

    # riepilogo + media mensile
    rect(0.74, 0.96, GREEN_BG)
    ann(px0+0.02, 0.92, "Il portafoglio ha avuto un rendimento positivo", size=16, color=GREEN_ACC, bold=True)
    ann(
        px0+0.02, 0.86,
        f"durante <b>{up_months}</b> dei <b>{total_months}</b> mesi (<b>{int(round(pct_up))}%</b>) tra il {start_year} e il {end_year}.",
        size=14
    )
    ann(px0+0.02, 0.80, f"Media rendimento mensile: <b>{pct_str(mean_monthly, 2)}</b>", size=14, color="#0f5132")

    def triplet(title_txt, series, y0, y1, bg, accent):
        rect(y0, y1, bg)
        title_y = y1 - 0.05
        month_y = title_y - 0.055
        value_y = month_y - 0.045

        ann(px0 + 0.02, title_y, title_txt, size=16, color=accent, bold=True)

        w = (px1 - px0)
        xs = [px0 + w * 0.17, px0 + w * 0.50, px0 + w * 0.83]

        for i, (idx, val) in enumerate(series.items()):
            if i > 2:
                break
            ann(xs[i], month_y, fmt_month(str(idx)), size=12, anchor="center")
            ann(xs[i], value_y, pct_str(float(val), 1), size=16, color=accent, bold=True, anchor="center")

    triplet("I mesi migliori", top_s, y0=0.48, y1=0.70, bg=GREEN_BG, accent=GREEN_ACC)
    triplet("I mesi peggiori", bot_s, y0=0.22, y1=0.44, bg=RED_BG,   accent=RED_ACC)
    ann(px0+0.02, 0.08, "ℹ️ L'istogramma mostra la frequenza dei rendimenti mensili (intra-month).", size=13, color=GRAY)

    fig.update_layout(
        title=title, title_x=0.5,
        width=width, height=height,
        margin=dict(l=left_margin_px, r=54, t=70, b=48),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def plot_year_returns_histogram(
    pf,
    title: str = "Istogramma dei rendimenti annuali",
    top_n: int = 3,
    width: int = 1200,
    height: int = 520,
    *,
    panel_width: float = 0.36,
    gap: float = 0.02,
    hist_fill: float = 0.96,
    left_margin_px: int = 28,
    min_years: int = 2
):
    """
    Istogramma dei rendimenti annuali con pannello testuale a destra (Plotly).
    Calcolo ANNUALE = intra-year compounding sui returns del portafoglio:
        ann = prod(1 + r_intra_year) - 1
    dove r_intra_year esclude SEMPRE il primo return dell'anno (carry-over dall'anno precedente).
    Se il portafoglio ha meno di `min_years` anni, ritorna None.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # --- daily returns ROBUSTI dal value totale ---
    v = pf.value()
    if isinstance(v, pd.DataFrame):
        v = v.sum(axis=1)
    v = v.dropna().copy()

    try:
        if v.index.tz is not None:
            v.index = v.index.tz_localize(None)
    except Exception:
        pass

    v.index = pd.to_datetime(v.index).normalize()
    v = v.sort_index()
    if v.index.duplicated().any():
        v = v[~v.index.duplicated(keep="last")]

    r = v.pct_change().dropna()

    # --- annual returns INTRA-YEAR (escludi primo return dell'anno) ---
    yearly_list = []
    for y in sorted(r.index.year.unique()):
        ry = r[r.index.year == y]
        if len(ry) >= 2:
            ry_intra = ry.iloc[1:]  # esclude carry-over da anno precedente
            ann = (1.0 + ry_intra).prod() - 1.0
            yearly_list.append((y, float(ann)))

    yearly = pd.Series(dict(yearly_list), dtype=float).sort_index()
    if len(yearly) < min_years:
        return None

    mean_yearly = float(yearly.mean())

    # Serie con indice "YYYY" per etichette
    yearly_named = yearly.copy()
    yearly_named.index = yearly_named.index.astype(str)

    start_year, end_year = int(yearly.index.min()), int(yearly.index.max())
    total_years = int(yearly.shape[0])
    up_years = int((yearly > 0).sum())
    pct_up = 100.0 * up_years / total_years if total_years else np.nan

    # Top/Bottom anni
    top_s = yearly_named.sort_values(ascending=False).head(top_n)
    bot_s = yearly_named.sort_values(ascending=True).head(top_n)

    # Bins simmetrici (2% di passo)
    max_abs = float(np.ceil(np.max(np.abs(yearly.values)) * 100.0))
    max_abs = max(max_abs, 10.0)
    step = 2.0
    max_abs = step * np.ceil(max_abs / step)
    xbins = dict(start=-max_abs/100.0, end=max_abs/100.0, size=step/100.0)

    x_pos = yearly[yearly >= 0].values
    x_neg = yearly[yearly < 0].values

    fig = go.Figure()
    fig.add_histogram(
        x=x_pos, xbins=xbins, marker=dict(color="rgb(34,139,34)"),
        hovertemplate="Anni: %{y}<extra></extra>", showlegend=False
    )
    fig.add_histogram(
        x=x_neg, xbins=xbins, marker=dict(color="rgb(203,67,53)"),
        hovertemplate="Anni: %{y}<extra></extra>", showlegend=False
    )
    fig.add_vline(x=0.0, line_width=2, line_dash="solid", line_color="rgba(80,80,80,0.6)")

    fig.update_xaxes(
        tickformat=".0%", title_text="Rendimento annuale",
        zeroline=False, range=[xbins["start"], xbins["end"]],
        showgrid=True, gridcolor="rgba(0,0,0,0.06)"
    )
    fig.update_yaxes(
        title_text="Numero di anni",
        rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.06)"
    )
    fig.update_layout(bargap=0.05)

    GREEN_BG, GREEN_ACC = "#e8f5e9", "#2e7d32"
    RED_BG, RED_ACC = "#fdecea", "#c62828"
    GRAY = "#6b7280"

    def pct_str(x, d=1): return f"{x*100:.{d}f}%".replace(".", ",")

    left_space = 1.0 - panel_width - gap
    hist_width = max(0.1, left_space * float(hist_fill))
    hist_left = (left_space - hist_width) / 2.0
    hist_right = hist_left + hist_width
    fig.update_xaxes(domain=[hist_left, hist_right])

    px0 = hist_right + gap
    px1 = px0 + panel_width

    def rect(y0, y1, color):
        fig.add_shape(type="rect", xref="paper", yref="paper",
                      x0=px0, x1=px1, y0=y0, y1=y1,
                      fillcolor=color, line=dict(color="rgba(0,0,0,0)"))

    def ann(x, y, text, size=14, color="#111", bold=False, anchor="left"):
        fig.add_annotation(xref="paper", yref="paper", x=x, y=y,
                           xanchor=anchor, yanchor="middle",
                           text=(f"<b>{text}</b>" if bold else text),
                           showarrow=False, font=dict(size=size, color=color), align="left")

    rect(0.68, 0.94, GREEN_BG)
    ann(px0+0.02, 0.90, "Il portafoglio ha avuto un rendimento positivo", size=16, color=GREEN_ACC, bold=True)
    ann(px0+0.02, 0.84, f"durante <b>{up_years}</b> dei <b>{total_years}</b> anni (<b>{int(round(pct_up))}%</b>) tra il {start_year} e il {end_year}.", size=14)
    ann(px0+0.02, 0.78, f"Media rendimento annuo: <b>{pct_str(mean_yearly, 2)}</b>", size=14, color="#0f5132")

    def triplet(title, series, y0, y1, bg, accent):
        rect(y0, y1, bg)
        title_y = y1 - 0.05
        year_y  = title_y - 0.055
        value_y = year_y  - 0.045
        ann(px0 + 0.02, title_y, title, size=16, color=accent, bold=True)
        w = (px1 - px0)
        xs = [px0 + w*0.17, px0 + w*0.50, px0 + w*0.83]
        for i, (idx, val) in enumerate(series.items()):
            if i > 2: break
            ann(xs[i], year_y, str(idx), size=12, anchor="center")
            ann(xs[i], value_y, pct_str(val, 1), size=16, color=accent, bold=True, anchor="center")

    triplet("Gli anni migliori", top_s, y0=0.42, y1=0.64, bg=GREEN_BG, accent=GREEN_ACC)
    triplet("Gli anni peggiori", bot_s, y0=0.18, y1=0.40, bg=RED_BG,   accent=RED_ACC)

    ann(px0+0.02, 0.08, "ℹ️ L'istogramma mostra la frequenza dei rendimenti annuali (intra-year).", size=13, color=GRAY)

    fig.update_layout(
        title=title, title_x=0.5,
        width=width, height=height,
        margin=dict(l=left_margin_px, r=54, t=64, b=48),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

    
def plot_monthly_returns(
    pf,
    eoy: bool = True,
    title: str = "Monthly Returns (%)",
    width: int | None = None,
    height: int | None = None,
    auto_height: bool = True,
    cell_h: int = 28,
    min_h: int = 300,
    max_h: int = 900
) -> go.Figure:
    """
    Heatmap 'anno x mese' dei rendimenti mensili (in %) del portafoglio.

    Calcolo su VALUE (robusto):
    - Mensile (intra-month): last_value_month / first_value_month - 1
    - EOY/YTD (intra-year): last_value_year  / first_value_year  - 1

    Nota: così GEN e EOY sono sempre calcolabili anche senza mese/anno precedente.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # --- value totale robusto ---
    v = pf.value()
    if isinstance(v, pd.DataFrame):
        v = v.sum(axis=1)
    v = v.dropna().copy()

    try:
        if v.index.tz is not None:
            v.index = v.index.tz_localize(None)
    except Exception:
        pass

    v.index = pd.to_datetime(v.index).normalize()
    v = v.sort_index()
    if v.index.duplicated().any():
        v = v[~v.index.duplicated(keep="last")]

    if v.empty:
        raise ValueError("Nessun valore disponibile (pf.value() vuoto).")

    # --- rendimenti mensili intra-month: last/first - 1 ---
    monthly_rows = []
    for (y, m), vm in v.groupby([v.index.year, v.index.month]):
        if len(vm) >= 2:
            ret_m = float(vm.iloc[-1] / vm.iloc[0] - 1.0)
        else:
            ret_m = np.nan
        monthly_rows.append((y, m, ret_m))

    df_m = pd.DataFrame(monthly_rows, columns=["Year", "Month", "Ret"])
    heat = df_m.pivot_table(index="Year", columns="Month", values="Ret", aggfunc="first")

    month_map = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
                 7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
    all_months = [1,2,3,4,5,6,7,8,9,10,11,12]
    heat = heat.reindex(columns=all_months)
    heat.columns = [month_map[c] for c in heat.columns]

    # anni presenti nel value
    years_idx = pd.Index(sorted(v.index.year.unique()), name="Year")
    heat = heat.reindex(index=years_idx)

    # --- EOY/YTD intra-year: last/first - 1 ---
    if eoy:
        yrets = []
        for y in years_idx:
            vy = v[v.index.year == y]
            if len(vy) >= 2:
                yret = float(vy.iloc[-1] / vy.iloc[0] - 1.0)
            else:
                yret = np.nan
            yrets.append(yret)
        heat["EOY"] = yrets

    # --- dimensioni figura ---
    n_years = len(heat.index)
    if auto_height and height is None:
        height = int(np.clip(120 + cell_h * max(1, n_years), min_h, max_h))

    z = (heat.values * 100.0).astype(float)
    text = np.where(np.isnan(z), "", np.round(z, 2).astype(object))

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=heat.columns.tolist(),
            y=heat.index.astype(str).tolist(),
            colorscale="RdYlGn",
            zmid=0,
            colorbar_title="%",
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10 if n_years <= 6 else 9 if n_years <= 12 else 8),
            hovertemplate="Year %{y}<br>%{x}: %{z:.2f}%<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        width=width, height=height,
        margin=dict(l=50, r=30, t=60, b=40),
        showlegend=False
    )
    fig.update_yaxes(autorange="reversed")
    return fig


#
# Strategies Rotationals
#

def plot_weights_heatmap(df_weights, title="Allocazioni medie (Walk-Forward)"):
    import seaborn as sns, matplotlib.pyplot as plt
    df_avg = df_weights.mean(axis=1).sort_values(ascending=False)
    plt.figure(figsize=(10, 0.4 * len(df_avg)))
    sns.heatmap(df_avg.to_frame().T, cmap="Blues", annot=True, fmt=".1%", cbar=False)
    plt.title(title); plt.yticks([]); plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.show()

def plot_strategy_pie(df_summary, title="Distribuzione strategie selezionate"):
    import matplotlib.pyplot as plt
    counts = df_summary["Method"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title(title); plt.tight_layout(); plt.show()

#
# Momentum Rotationals
#
def plot_ticker_frequencies(
    sel_tickers: 'pd.DataFrame',
    start_date: 'pd.Timestamp | str | None' = None,
    end_date: 'pd.Timestamp | str | None' = None,
    include_prev: bool = True,
    universe: 'Iterable[str] | None' = None,
    title: str = "Frequenza di selezione dei titoli",
    width: int = 1000,
    height: int = 600,
):
    """
    Istogramma Plotly dei ticker più ricorrenti (migliorata).

    Parametri
    ---------
    sel_tickers : pd.DataFrame
        DataFrame indicizzato in datetime con colonna 'Top_Tickers' (liste o stringhe come "AAPL,MSFT").
    start_date, end_date : str|Timestamp|None
        Finestra da considerare (inclusiva). Se None -> usa tutto.
    include_prev : bool
        Se True include l'ultima selezione strettamente precedente a start_date (utile per "set iniziale").
    universe : iterable[str] | None
        Lista/Index/set dell'universo completo dei titoli. Se fornito, la funzione calcola tic
        ker mai selezionati e la percentuale di non-selezionati.
    title, width, height : grafica

    Restituisce
    ----------
    fig : plotly.graph_objects.Figure
        Istogramma interattivo.
    freq_df : pd.DataFrame
        DataFrame con colonne ['Ticker','Frequenza'] ordinato per frequenza decrescente.
    freq_full_df : pd.DataFrame
        Se universe fornito: DataFrame con tutte le tickers dell'universo e la frequenza (0=mai selezionato).
        Se universe None: None.
    unselected_df : pd.DataFrame
        Se universe fornito: DataFrame dei tickers mai selezionati e una riga-sintesi con la percentuale.
        Se universe None: None.
    """
    from collections.abc import Iterable

    # universe += ["ZZZ_TEST_NOT_SELECTED"] # test per la selezione

    # -----------------------
    # Validazioni e copia
    # -----------------------
    if sel_tickers is None or sel_tickers.empty:
        raise ValueError("sel_tickers è vuoto: impossibile calcolare le frequenze.")

    if "Top_Tickers" not in sel_tickers.columns:
        raise KeyError("sel_tickers deve contenere la colonna 'Top_Tickers'.")

    df = sel_tickers.copy()

    # --- normalizza indice datetime ---
    try:
        idx = pd.to_datetime(df.index)
    except Exception as e:
        raise TypeError("sel_tickers.index non è convertibile a datetime.") from e

    # rimuovi tz e normalizza (solo data)
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    df.index = idx.normalize()
    df = df.sort_index()

    # --- parse date bounds ---
    def _to_ts(x):
        if x is None:
            return None
        t = pd.to_datetime(x)
        try:
            if getattr(t, "tz", None) is not None:
                t = t.tz_localize(None)
        except Exception:
            pass
        return pd.Timestamp(t).normalize()

    s = _to_ts(start_date)
    e = _to_ts(end_date)

    # --- filtro finestra ---
    df_win = df
    if s is not None and e is not None:
        if s > e:
            s, e = e, s
        df_win = df.loc[s:e]
    elif s is not None:
        df_win = df.loc[s:]
    elif e is not None:
        df_win = df.loc[:e]

    # --- include prev selection (ultima riga strettamente prima di s) ---
    if include_prev and s is not None:
        # tutte le righe fino a s (inclusive) -> prendo l'ultima con index < s
        df_before = df.loc[:s]
        # rimuovo eventuale riga con index == s perché vogliamo strettamente precedente
        df_before = df_before.loc[df_before.index < s]
        if not df_before.empty:
            prev_row = df_before.iloc[[-1]]
            df_win = pd.concat([prev_row, df_win], axis=0)
            df_win = df_win[~df_win.index.duplicated(keep="last")].sort_index()

    if df_win.empty:
        raise ValueError(
            f"Nessun dato sel_tickers nella finestra richiesta (start={start_date}, end={end_date})."
        )

    # --- esplodi tickers ---
    exploded = df_win["Top_Tickers"].explode()

    # normalizza eventuali stringhe tipo "AAPL,MSFT" e rimuovi NaN
    def _norm(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, str):
            # split su virgole o punto e virgola
            if "," in x or ";" in x:
                parts = [p.strip() for p in re.split(r"[,;]+", x) if p.strip()]
                return parts
            # se stringa singola
            return x.strip()
        # se già lista/iterable (ma non stringa), ritorna come è
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return list(x)
        return x

    import re
    exploded = exploded.apply(_norm).explode().dropna().astype(str).str.upper().str.strip()

    # Conta frequenze
    freq_series = exploded.value_counts()
    freq_df = freq_series.reset_index()
    freq_df.columns = ["Ticker", "Frequenza"]

    # ----- se fornisco universe: costruisco freq_full_df e unselected_df -----
    freq_full_df = None
    unselected_df = None
    if universe is not None:
        # normalizza universe in lista di stringhe uppercase
        if isinstance(universe, (pd.Index, list, set, tuple)):
            uni_list = [str(x).upper().strip() for x in list(universe)]
        else:
            # se passato un single stringo separato da virgole
            try:
                uni_list = [p.strip().upper() for p in re.split(r"[,;]+", str(universe)) if p.strip()]
            except Exception:
                raise TypeError("Parametro 'universe' non è un iterable riconosciuto.")
        uni_index = pd.Index(sorted(set(uni_list)), name="Ticker")

        # freq_full: merge universo con freq (assegno 0 ai non presenti)
        freq_full_df = pd.DataFrame(index=uni_index).reset_index()
        freq_full_df = freq_full_df.merge(freq_df, on="Ticker", how="left").fillna({"Frequenza": 0})
        freq_full_df["Frequenza"] = freq_full_df["Frequenza"].astype(int)
        freq_full_df = freq_full_df.sort_values("Frequenza", ascending=False).reset_index(drop=True)

        # tickers non selezionati
        unselected = freq_full_df.loc[freq_full_df["Frequenza"] == 0, "Ticker"].tolist()
        pct_unselected = (len(unselected) / len(uni_index)) if len(uni_index) > 0 else np.nan

        unselected_df = pd.DataFrame({
            "Ticker": unselected
        })
        # aggiungo riga di sintesi (facoltativa, utile per visual)
        summary = pd.DataFrame([{
            "Ticker": "<summary>",
            "Frequenza": len(unselected),
            "Pct_unselected": pct_unselected
        }])
        # non concateno la summary alle ticker list automaticamente; lascio separate ma ritorno il valore
        # per comodità aggiungo la percentuale a freq_full_df
        freq_full_df["Pct_universe"] = (freq_full_df["Frequenza"] > 0).astype(int)  # 1 se selezionato almeno 1 volta
        # Aggiungo percentuale colonna per chiarezza (0/1) non la percentuale reale per ticker
        # Fornisco pct_unselected separatamente come float

    # ----- Istogramma -----
    # Se freq_df è vuoto (improbabile qui), creiamo fig vuota altrimenti grafico bar
    if freq_df.empty:
        fig = px.bar(title=title)
        fig.update_layout(width=width, height=height, template="plotly_white")
    else:
        fig = px.bar(
            freq_df,
            x="Ticker",
            y="Frequenza",
            title=title,
            labels={"Frequenza": "Numero di occorrenze"},
            text="Frequenza",
        )
        fig.update_traces(marker_color="blue", textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, template="plotly_white", width=width, height=height)

    # --- return (compatibilità col vecchio API) ---
    return fig, freq_df, freq_full_df, (unselected_df, (len(unselected) if universe is not None else None),
                                        (pct_unselected if universe is not None else None))

    
def plot_total_return_per_ticker(
    returns,
    title: str = "Rendimenti totali per titolo (%)",
    start_date=None,
    end_date=None,
    label_decimals: int = 2,
    highlight_zero: bool = True 
) -> go.Figure:
    """
    Crea un bar chart orizzontale dei rendimenti totali per ticker usando Plotly,
    con etichette percentuali poste all'esterno delle barre.

    Parameters
    ----------
    returns : pd.Series or dict of pd.Series
        - Se è una pd.Series indicizzata con ticker e valori numerici:
          si assume che siano già i rendimenti totali in %.
        - Se è una pd.Series “object” (mapping ticker → pd.Series):
          calcola il total return da ciascuna serie.
    title : str
        Titolo del grafico.
    label_decimals : int
        Numero di decimali da mostrare sulle etichette.
    highlight_zero : bool
        Se True, colora in grigio i rendimenti pari a 0.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Se returns è già una serie di floats, salto il calcolo
    if not (returns.dtype == object and isinstance(returns.iloc[0], (pd.Series, list, np.ndarray))):
        total_return = returns.copy()
    else:
        # Applichiamo filtro temporale e normalizzazione
        if start_date is not None:
            returns = pd.Series(
                {sym: ser[ser.index >= start_date] for sym, ser in returns.items()},
                name='returns',
                dtype=object
            )
        if end_date is not None:
            returns = pd.Series(
                {sym: ser[ser.index <= end_date] for sym, ser in returns.items()},
                name='returns',
                dtype=object
            )

        # calcola il total return per ogni ticker
        total_return = pd.Series(
            {symbol: (1 + series).prod() - 1
             for symbol, series in returns.items()},
            name='total_return'
        )

    # Se era in frazione, lo porto a percentuale
    if total_return.abs().max() <= 1:
        total_return = total_return * 100

    # Ordina i rendimenti (escludendo gli zero, se ci sono)
    perf_sorted = total_return[total_return != 0].sort_values()

    # Determina colori
    def get_color(v):
        if v > 0:
            return "green"
        elif v < 0:
            return "red"
        else:
            return "lightgray" if highlight_zero else "black"

    colors = [get_color(v) for v in perf_sorted.values]

    # Prepara le etichette di testo
    texts = [f"{v:.{label_decimals}f}%" for v in perf_sorted.values]

    # Calcola dinamicamente altezza in pixel (min 400px, ~30px per ticker)
    height = max(400, len(perf_sorted) * 30)

    # Estendi i limiti dell'asse X di ±5 punti
    x_min = perf_sorted.min() - 5
    x_max = perf_sorted.max() + 5

    tickers = perf_sorted.index
    values = perf_sorted.values

    # Recupera i nomi aziendali, se disponibili
    company_data = build_company_df_with_cache(tickers)
    def truncate(s, n):
        return s if len(s) <= n else '…' + s[-n:]

    n=70
    company_data['Company'] = company_data['Company'].apply(lambda s: truncate(s, n))

    labels = [
        f"{company_data.loc[t, 'Company'] if t in company_data.index else 'N/D'} ({t})"
        for t in tickers
    ]

    # Costruisci la figura
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=texts,
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
    ))
    
    # Rimuovo il titolo y tradizionale e imposto solo l’asse x
    fig.update_layout(
        title=title,
        xaxis_title="Rendimento (%)",
        yaxis_title=None,
        xaxis=dict(range=[x_min, x_max], showgrid=True, gridcolor='lightgray'),
        margin=dict(l=120, r=40, t=80, b=40),
        height=height
    )
    
    # Mantengo l’ordine delle categorie sull’asse y
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=list(perf_sorted.index),
        showticklabels=True
    )
    
    # Aggiungo l’annotazione in alto a sinistra dentro l’area del plot (paper coords)
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0, y=1.02,               # 0% da sinistra, 102% in alto (leggermente sopra)
        xanchor='left',
        text="Companies (ticker)",
        showarrow=False,
        font=dict(size=12)
    )


    return fig

def _pf_from_equity_curve(eq, *, init_cash=100_000):
    """
    Converte una equity line (Series) in un vbt.Portfolio
    usando from_holding (compatibile col tuo ambiente).
    """
    import pandas as pd
    import numpy as np

    eq = pd.Series(eq).dropna().copy()

    # pulizia index
    try:
        if eq.index.tz is not None:
            eq.index = eq.index.tz_localize(None)
    except Exception:
        pass
    eq.index = pd.to_datetime(eq.index).normalize()
    eq = eq.sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]

    # guard-rail: equity deve essere positiva
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()
    eq = eq[eq > 0]

    import vectorbt as vbt
    pf_tmp = vbt.Portfolio.from_holding(
        eq,
        init_cash=init_cash,
        freq="D",
        group_by=False
    )
    return pf_tmp


def build_rolling_summaries_table(
    pf,
    *,
    horizons_years=(1, 2, 3, 5),
    annual_trading_days=252,
    risk_free_rate=0.02,
    alpha_analysis=False,
    init_cash=100_000,
    asof_date=None,
):
    import pandas as pd
    import numpy as np
    import vectorbt as vbt

    # --- daily returns base ---
    r = pf.returns()
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    r = r.dropna().copy()
    try:
        if r.index.tz is not None:
            r.index = r.index.tz_localize(None)
    except Exception:
        pass
    r.index = pd.to_datetime(r.index).normalize()
    r = r.sort_index()
    r = r[~r.index.duplicated(keep="last")]

    if r.empty:
        raise ValueError("Rendimenti vuoti.")

    if asof_date is None:
        asof_date = r.index.max()
    else:
        asof_date = pd.to_datetime(asof_date).normalize()

    r = r.loc[:asof_date]
    if r.empty:
        raise ValueError("asof_date fuori range.")
    asof_date = r.index.max()

    # ---- Totale (full history) usando vbt/summary esistente ----
    # (qui va bene usare la tua create_portfolio_summary sul pf totale)
    df_total = create_portfolio_summary(
        pf,
        benchmark_portfolio=None,
        sel_tickers=None,
        alpha_analysis=alpha_analysis,
        risk_free_rate=risk_free_rate,
        show=False,
        run_as_app=False
    )
    rows = {"Totale": df_total["Valore"]}

    # ---- helper: summary rolling "DA RETURNS" (coerente col grafico) ----
    def _summary_from_returns(ret: pd.Series) -> pd.Series:
        ret = ret.dropna()
        n = int(ret.shape[0])
        if n <= 1:
            return pd.Series(dtype=float)

        # coerente col grafico rolling: prod(1+r)-1
        total_ret = float((1.0 + ret).prod() - 1.0)

        # equity curve (base 1)
        eq = (1.0 + ret).cumprod()
        dd = (eq / eq.cummax()) - 1.0
        max_dd = float(abs(dd.min())) if not dd.empty else np.nan

        cagr = float((1.0 + total_ret) ** (annual_trading_days / n) - 1.0)

        vol = float(ret.std(ddof=0) * np.sqrt(annual_trading_days))

        rf_daily = (1 + risk_free_rate) ** (1 / annual_trading_days) - 1
        std = float(ret.std(ddof=0))
        sharpe = float(((ret.mean() - rf_daily) / std) * np.sqrt(annual_trading_days)) if std > 0 else np.nan

        final_value = float(init_cash * (1.0 + total_ret))

        return pd.Series({
            "Periodo": f"{ret.index.min().date().isoformat()} → {ret.index.max().date().isoformat()}",
            "Importo investito (€)": float(init_cash),
            "Valore patrimoniale netto (€)": final_value,
            "Giorni di trading": n,
            "Ritorno totale": total_ret,
            "CAGR": cagr,
            "Max Drawdown": max_dd,
            "Volatilità annua": vol,
            "Rapporto di Sharpe": sharpe,
            # non ha senso per finestre rolling “sintetiche”:
            "Operazioni al mese": np.nan,
            "Market Time Exposure": np.nan,
        })

    # ---- Rolling: ULTIMA finestra, ma:
    # 1) solo se esiste nel grafico (len(r) >= ndays)
    # 2) skip se coincide col Totale (ndays >= len(r))
    n_total = int(len(r))

    for y in horizons_years:
        ndays = int(annual_trading_days * y)

        # se rolling sarebbe uguale al Totale, non mostrarlo
        if ndays >= n_total:
            continue

        # se non ho abbastanza dati, nel grafico rolling è NaN -> skip
        if n_total < ndays:
            continue

        r_win = r.iloc[-ndays:].copy()
        s = _summary_from_returns(r_win)
        if not s.empty:
            rows[f"Rolling {y}y (last window)"] = s

    out = pd.DataFrame(rows).T

    # ---- formatting (uguale al tuo) ----
    wanted = [
        "Periodo",
        "Importo investito (€)",
        "Valore patrimoniale netto (€)",
        "Giorni di trading",
        "Ritorno totale",
        "CAGR",
        "Max Drawdown",
        "Volatilità annua",
        "Rapporto di Sharpe",
        "Operazioni al mese",
        "Market Time Exposure",
    ]
    out = out.reindex(columns=wanted)

    fmt = out.copy()

    pct_cols = ["Ritorno totale", "CAGR", "Max Drawdown", "Volatilità annua", "Market Time Exposure"]
    num_cols = ["Rapporto di Sharpe", "Operazioni al mese"]
    money_cols = ["Importo investito (€)", "Valore patrimoniale netto (€)"]

    for c in pct_cols:
        if c in fmt.columns:
            fmt[c] = fmt[c].apply(lambda x: "n/a" if pd.isna(x) else f"{x*100:.2f}%")

    for c in num_cols:
        if c in fmt.columns:
            fmt[c] = fmt[c].apply(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")

    for c in money_cols:
        if c in fmt.columns:
            fmt[c] = fmt[c].apply(lambda x: "n/a" if pd.isna(x) else f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + "€")

    if "Giorni di trading" in fmt.columns:
        fmt["Giorni di trading"] = fmt["Giorni di trading"].apply(lambda x: "n/a" if pd.isna(x) else str(int(x)))

    return fmt, out
    
def analyze_rolling_horizons(
    roll_cum_df: pd.DataFrame,
    loss_threshold: float = 0.0,             # compat: la vecchia soglia “principale”
    *,
    loss_thresholds: tuple[float, ...] | None = None,  # nuovo: più soglie
    min_obs: int = 30,
    target_prob: float | None = 0.0          # opzionale: “min safe” su soglia 0
) -> dict:
    import numpy as np
    import pandas as pd

    # 1) orizzonti (es: '1y','2y'...)
    cols = [c for c in roll_cum_df.columns if isinstance(c, str) and c.endswith("y")]
    if not cols:
        raise ValueError("roll_cum_df non ha colonne tipo '1y','2y',...")

    horizons = sorted([(int(c[:-1]), c) for c in cols], key=lambda x: x[0])

    # 2) soglie
    if loss_thresholds is None:
        loss_thresholds = (float(loss_threshold),)
    else:
        loss_thresholds = tuple(float(x) for x in loss_thresholds)

    # 3) calcolo probabilità P(R < soglia) per ogni orizzonte e soglia
    prob_rows = []
    for h, col in horizons:
        s = roll_cum_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(s))
        row = {"horizon_years": h, "n_obs": n}
        for thr in loss_thresholds:
            row[f"p_lt_{thr}"] = float((s < thr).mean()) if n >= min_obs else np.nan
        prob_rows.append(row)

    prob_df = pd.DataFrame(prob_rows).set_index("horizon_years").sort_index()

    # 4) “min safe horizon” sulla soglia 0 (se presente e target_prob valorizzato)
    min_safe = None
    if target_prob is not None:
        # cerco la colonna per thr=0.0 (attenzione a float repr)
        # quindi la ricavo cercando la soglia “più vicina a 0”
        thr0 = min(loss_thresholds, key=lambda x: abs(x - 0.0))
        col0 = f"p_lt_{thr0}"
        if abs(thr0) < 1e-12 and col0 in prob_df.columns:
            for h in prob_df.index:
                p = prob_df.loc[h, col0]
                if pd.notna(p) and p <= float(target_prob):
                    min_safe = int(h)
                    break

    return {
        "prob_df": prob_df,                 # matrice delle probabilità
        "loss_thresholds": loss_thresholds,
        "loss_threshold": float(loss_threshold),  # compat
        "target_prob": target_prob,
        "min_safe_horizon": min_safe,
        "min_obs": int(min_obs),
    }



def plot_loss_probability_curve(
    analysis: dict,
    *,
    title: str = "Probabilità di perdita (rolling < soglia) vs orizzonte",
    show_point_labels: bool = True,
    height: int = 420
):

    prob_df: pd.DataFrame = analysis["prob_df"]
    thresholds = analysis["loss_thresholds"]
    min_safe = analysis.get("min_safe_horizon", None)
    target_prob = analysis.get("target_prob", None)

    # --- X NUMERICO (fix: evita asse categorico) ---
    # prob_df.index può essere [1,2,3,5] oppure ["1","2","3","5"] ecc.
    xs = pd.to_numeric(pd.Index(prob_df.index), errors="coerce").astype(float).to_list()

    fig = go.Figure()

    for thr in thresholds:
        col = f"p_lt_{thr}"
        if col not in prob_df.columns:
            continue

        ys = prob_df[col].astype(float).values
        name = f"P(R < {thr:.0%})" if abs(thr) > 1e-12 else "P(R < 0%)"

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=name,
            hovertemplate="Orizzonte: %{x}y<br>P: %{y:.2%}<extra></extra>"
        ))

        if show_point_labels:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="text",
                text=[f"{v*100:.1f}%" if pd.notna(v) else "" for v in ys],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip"
            ))


    if min_safe is not None:
        fig.add_vline(x=min_safe, line_width=1, line_dash="dot")
    
        if target_prob is not None:
            dx=0.3
            x_text = min_safe + dx

            fig.add_annotation(
                x=x_text,
                y=0.12,          # più alto: 12% dal fondo del pannello
                xref="x",
                yref="paper",
                text=f"Min safe: {min_safe} (P≤{float(target_prob):.0%})",
                showarrow=False,
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(80,80,80,0.35)",
                borderwidth=0.9,
                font=dict(size=14, color="green")

            )


    fig.update_layout(
        title=title,
        xaxis_title="Orizzonte (anni)",
        yaxis_title="Probabilità",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        height=height,
        hovermode="x unified",
    )

    # FIX CRITICO: asse X lineare (non categorie)
    fig.update_xaxes(type="linear")

    return fig

def compute_rolling_extrema_ranges(roll_cum_df: pd.DataFrame,
                                   daily_index: pd.DatetimeIndex,
                                   annual_trading_days: int = 252) -> pd.DataFrame:
    """
    Per ogni colonna di roll_cum_df (es. '1y','2y',...) trova:
      - il massimo osservato -> (start_date, end_date, return, duration_days)
      - il minimo osservato -> (start_date, end_date, return, duration_days)

    roll_cum_df: DataFrame con index DatetimeIndex e colonne '1y','2y',...
                 i valori sono rolling total returns (es. prod(1+rets)-1).
    daily_index: DatetimeIndex giornaliero corrispondente ai returns usati per il rolling
                 (di solito daily_rets.index, normalizzato a midnight).
    annual_trading_days: numero di trading days/anno usato per calcolare la finestra.
    """
    rows = []
    # normalizza index a DatetimeIndex senza tz
    idx = pd.to_datetime(daily_index).normalize()
    # mapping index -> posizione per lookup rapido
    pos_map = {d: i for i, d in enumerate(idx)}

    for col in roll_cum_df.columns:
        series = roll_cum_df[col].dropna()
        if series.empty:
            rows.append({
                "horizon": col,
                "max_start": pd.NaT, "max_end": pd.NaT, "max_return": np.nan, "max_days": np.nan,
                "min_start": pd.NaT, "min_end": pd.NaT, "min_return": np.nan, "min_days": np.nan,
            })
            continue

        # determina ndays usati nella rolling (es. '1y' -> 1 * annual_trading_days)
        # supporta etichette come '1y' o '2y' o numeriche
        try:
            if isinstance(col, str) and col.endswith("y"):
                years = int(col[:-1])
            else:
                years = int(col)
        except Exception:
            # fallback: stima window dalla differenza di posizioni utili (non ideale)
            # mettiamo il valore default = annual_trading_days
            years = 1

        ndays = int(round(years * annual_trading_days))

        # ---- massimo ----
        max_end = series.idxmax()               # data di fine finestra per il massimo
        max_val = float(series.loc[max_end])
        # posizione corrispondente in daily_index
        pos_end = pos_map.get(pd.to_datetime(max_end).normalize(), None)
        if pos_end is None:
            # se non trovi pos (possibile per differenze minime), usa get_indexer
            pos_end = idx.get_indexer([pd.to_datetime(max_end).normalize()])[0]
        start_pos = pos_end - (ndays - 1)
        if start_pos >= 0:
            max_start = idx[start_pos]
            max_days = (max_end - max_start).days + 1
        else:
            max_start = pd.NaT
            max_days = np.nan

        # ---- minimo ----
        min_end = series.idxmin()
        min_val = float(series.loc[min_end])
        pos_end = pos_map.get(pd.to_datetime(min_end).normalize(), None)
        if pos_end is None:
            pos_end = idx.get_indexer([pd.to_datetime(min_end).normalize()])[0]
        start_pos = pos_end - (ndays - 1)
        if start_pos >= 0:
            min_start = idx[start_pos]
            min_days = (min_end - min_start).days + 1
        else:
            min_start = pd.NaT
            min_days = np.nan

        rows.append({
            "horizon": col,
            "max_start": pd.to_datetime(max_start) if not pd.isna(max_start) else pd.NaT,
            "max_end": pd.to_datetime(max_end),
            "max_return": max_val,
            "max_days": int(max_days) if not np.isnan(max_days) else np.nan,
            "min_start": pd.to_datetime(min_start) if not pd.isna(min_start) else pd.NaT,
            "min_end": pd.to_datetime(min_end),
            "min_return": min_val,
            "min_days": int(min_days) if not np.isnan(min_days) else np.nan,
        })

    out = pd.DataFrame(rows)
    # ordina per horizon se necessario (es. 1y,2y,...)
    def _hkey(x):
        try:
            if isinstance(x, str) and x.endswith("y"):
                return int(x[:-1])
            return int(x)
        except:
            return 999
    out = out.sort_values(by="horizon", key=lambda s: s.map(_hkey)).reset_index(drop=True)
    return out
    
def plot_cumulative_and_rolling_returns(
    pf,
    horizons_years=None,
    annual_trading_days=252,
    title="Rendimenti cumulati e rolling del portafoglio",
    height=700,
    annotate_extrema=True,
    show_controls=True,
    show_rolling_summary: bool = True,
    rolling_summary_risk_free_rate: float = 0.02,
    # --- fan chart ---
    add_fan_chart: bool = True,
    fan_horizon: str = "1y",
    fan_window_days: int = 252,
    fan_percentiles: tuple = (5, 25, 50, 75, 95),
    # --- heatmap ---
    add_heatmap: bool = True,
    return_heatmap: bool = False,
    # --- horizon analysis ---
    add_horizon_analysis: bool = True,
    loss_threshold: float = 0.0,
    min_obs: int = 30,
    # --- loss_probability_vs_horizon ---
    add_loss_prob_curve: bool = True,
    loss_thresholds: tuple = (0.0, -0.05, -0.10),
    loss_target_prob: float | None = 0.0,
    show_loss_point_labels: bool = True,
    # --- NEW: tabella finestre best/worst per rolling ---
    show_windows_table: bool = True,
    # --- return esteso ---
    return_extras: bool = False,
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if horizons_years is None:
        horizons_years = [1, 2, 3, 5]

    # -----------------------------
    # Base data
    # -----------------------------
    daily_rets = pf.returns()
    cum_rets_tot = pf.cumulative_returns()

    if isinstance(daily_rets, pd.DataFrame):
        daily_rets = daily_rets.iloc[:, 0]
    if isinstance(cum_rets_tot, pd.DataFrame):
        cum_rets_tot = cum_rets_tot.iloc[:, 0]

    # pulizia minima
    daily_rets = daily_rets.dropna().copy().sort_index()
    cum_rets_tot = cum_rets_tot.dropna().copy().sort_index()

    if daily_rets.empty or cum_rets_tot.empty:
        raise ValueError("Serie returns/cumulative_returns vuote: impossibile plottare rolling.")

    analysis_start = daily_rets.index.min()
    analysis_end = daily_rets.index.max()

    # indice trading effettivo
    idx = daily_rets.index

    # finestra usata per linee/slider (evita sconfini oltre ultimo giorno disponibile)
    idx_window = idx[(idx >= analysis_start) & (idx <= analysis_end)]
    if len(idx_window) == 0:
        raise ValueError("Finestra indice vuota: controlla analysis_start/analysis_end.")

    # -----------------------------
    # Rolling cumulative returns
    # -----------------------------
    roll_cum_dict = {}
    window_days_map = {}  # label -> ndays
    for y in horizons_years:
        ndays = int(annual_trading_days * y)
        label = f"{y}y"
        roll = (1 + daily_rets).rolling(ndays).apply(np.prod, raw=True) - 1
        roll_cum_dict[label] = roll
        window_days_map[label] = ndays

    roll_cum_df = pd.DataFrame(roll_cum_dict, index=daily_rets.index)

    # -----------------------------
    # Main figure
    # -----------------------------
    fig = go.Figure()

    color_map = {
        "Totale": "blue",
        "1y": "orange",
        "2y": "purple",
        "3y": "green",
        "5y": "red",
    }

    # Totale
    cum_slice = cum_rets_tot.loc[analysis_start:analysis_end]
    fig.add_trace(go.Scatter(
        x=cum_slice.index,
        y=cum_slice.values,
        name="Totale",
        mode="lines",
        line=dict(width=2, color=color_map["Totale"]),
        hovertemplate="%{x|%Y-%m-%d}<br>Totale: %{y:.2%}<extra></extra>",
        legendgroup="lg_Totale",
        showlegend=True,
    ))

    # -----------------------------
    # Helper per start-date finestra (trading-days)
    # -----------------------------
    def _start_from_end(end_ts: pd.Timestamp, n: int) -> pd.Timestamp | None:
        """Start della finestra che termina in end_ts e contiene n osservazioni."""
        try:
            pos = idx.get_loc(end_ts)
            if isinstance(pos, slice):
                pos = pos.stop - 1
            start_pos = int(pos) - int(n) + 1
            if start_pos < 0:
                return None
            return idx[start_pos]
        except Exception:
            return None

    # -----------------------------
    # Rolling curves (+ windows table)
    # -----------------------------
    windows_rows = []

    for label in roll_cum_df.columns:
        s = roll_cum_df[label].loc[analysis_start:analysis_end].copy()
        ss = s.dropna()
        if ss.empty:
            # niente dati validi -> non aggiungere nulla (niente legenda)
            continue

        c = color_map.get(label, "gray")
        legend_name = f"Rolling {label}"
        lg = f"lg_{label}"

        # 1) Traccia principale (in legenda)
        fig.add_trace(go.Scatter(
            x=ss.index,
            y=ss.values,
            name=legend_name,
            mode="lines",
            line=dict(width=1.6, color=c),
            hovertemplate="%{x|%Y-%m-%d}<br>" + f"{legend_name}: %{{y:.2%}}<extra></extra>",
            legendgroup=lg,
            showlegend=True,
        ))

        # 2) Area negativa (ancorata al gruppo, non in legenda)
        neg = ss.copy()
        neg[neg > 0] = 0.0
        fig.add_trace(go.Scatter(
            x=neg.index,
            y=neg.values,
            fill="tozeroy",
            mode="none",
            hoverinfo="skip",
            showlegend=False,
            fillcolor="rgba(255,0,0,0.18)",
            legendgroup=lg,
        ))

        # 3) Marker massimo/minimo + tabella finestre
        if annotate_extrema:
            max_date = ss.idxmax()
            min_date = ss.idxmin()
            max_val = float(ss.loc[max_date])
            min_val = float(ss.loc[min_date])

            fig.add_trace(go.Scatter(
                x=[max_date], y=[max_val],
                mode="markers+text",
                text=[f"▲ {max_val:.2%}"],
                textposition="top center",
                marker=dict(color="green", size=8),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Max: %{y:.2%}<extra></extra>",
                legendgroup=lg,
            ))

            fig.add_trace(go.Scatter(
                x=[min_date], y=[min_val],
                mode="markers+text",
                text=[f"▼ {min_val:.2%}"],
                textposition="bottom center",
                marker=dict(color="red", size=8),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Min: %{y:.2%}<extra></extra>",
                legendgroup=lg,
            ))

            ndays = int(window_days_map.get(label, annual_trading_days))
            best_start = _start_from_end(max_date, ndays)
            worst_start = _start_from_end(min_date, ndays)

            windows_rows.append({
                "Orizzonte": label,
                "Trading_days": ndays,
                "Best_start": best_start,
                "Best_end": max_date,
                "Best_return": max_val,
                "Worst_start": worst_start,
                "Worst_end": min_date,
                "Worst_return": min_val,
            })

    windows_df = pd.DataFrame(windows_rows)

    # zero line (tagliata alla finestra reale)
    fig.add_trace(go.Scatter(
        x=idx_window,
        y=[0] * len(idx_window),
        mode="lines",
        line=dict(color="red", dash="dot"),
        name="Soglia 0%",
        hoverinfo="skip",
        legendgroup="lg_Zero",
        showlegend=True
    ))

    # -----------------------------
    # Fan chart (rolling percentiles)
    # -----------------------------
    if add_fan_chart and fan_horizon in roll_cum_df.columns:
        sr = roll_cum_df[fan_horizon].loc[analysis_start:analysis_end].dropna()
        if not sr.empty:
            bands = {}
            for p in fan_percentiles:
                bands[p] = sr.rolling(fan_window_days).quantile(p / 100.0)
            bands = pd.DataFrame(bands)

            if {5, 25, 50, 75, 95}.issubset(bands.columns):
                lg_fan = f"lg_fan_{fan_horizon}"

                fig.add_trace(go.Scatter(
                    x=bands.index, y=bands[5],
                    line=dict(width=0),
                    showlegend=True,
                    name=f"{fan_horizon} P5–P95",
                    legendgroup=lg_fan
                ))
                fig.add_trace(go.Scatter(
                    x=bands.index, y=bands[95],
                    fill="tonexty",
                    line=dict(width=0),
                    fillcolor="rgba(120,120,120,0.18)",
                    showlegend=False,
                    legendgroup=lg_fan,
                    hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=bands.index, y=bands[25],
                    line=dict(width=0),
                    showlegend=True,
                    name=f"{fan_horizon} P25–P75",
                    legendgroup=lg_fan
                ))
                fig.add_trace(go.Scatter(
                    x=bands.index, y=bands[75],
                    fill="tonexty",
                    line=dict(width=0),
                    fillcolor="rgba(120,120,120,0.30)",
                    showlegend=False,
                    legendgroup=lg_fan,
                    hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=bands.index, y=bands[50],
                    line=dict(width=1, dash="dot", color="black"),
                    name=f"{fan_horizon} Mediana",
                    legendgroup=lg_fan,
                    showlegend=True,
                    hoverinfo="skip"
                ))

    # -----------------------------
    # Layout (+ togglegroup)
    # -----------------------------
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(title="Rendimento cumulato (%)", tickformat=".0%"),
        xaxis=dict(
            title="Data",
            range=[analysis_start, analysis_end],
            autorange=False,
            rangeslider=dict(
                visible=True,
                autorange=False,
                range=[analysis_start, analysis_end]
            ),
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
            groupclick="togglegroup"
        )
    )

    # -----------------------------
    # Tabella finestre best/worst (display)
    # -----------------------------
    if show_windows_table and windows_df is not None and not windows_df.empty:
        try:
            from IPython.display import display, HTML
    
            df_show = windows_df.copy()
    
            # date pulite
            for c in ["Best_start", "Best_end", "Worst_start", "Worst_end"]:
                df_show[c] = pd.to_datetime(df_show[c]).dt.date
    
            # rendimenti in percentuale
            df_show["Best_return"] = (df_show["Best_return"] * 100).round(2)
            df_show["Worst_return"] = (df_show["Worst_return"] * 100).round(2)
    
            # titolo tabella
            display(HTML(
                "<h4 style='margin-top:15px'>"
                "Finestre rolling migliori e peggiori per orizzonte"
                "</h4>"
                "<p style='color:gray; font-size:12px'>"
                "Per ogni orizzonte (1y, 2y, 3y, …) sono riportati il periodo "
                "che ha generato il rendimento massimo e minimo (Total Return rolling)."
                "</p>"
            ))
    
            display(df_show)
    
        except Exception:
            pass

    # -----------------------------
    # Rolling heatmap
    # -----------------------------
    fig_hm = None
    if add_heatmap:
        hm = roll_cum_df.loc[analysis_start:analysis_end].replace([np.inf, -np.inf], np.nan)

        # se tutto NaN evita crash su nanmin/nanmax
        finite = np.isfinite(hm.values)
        if not finite.any():
            fig_hm = None
        else:
            zmin = float(np.nanmin(hm.values))
            zmax = float(np.nanmax(hm.values))
        
            # --- Colorscale robusta ---
            if zmin >= 0:
                # solo positivi: 0 è il minimo "logico"
                zmin = 0.0
                colorscale = [
                    [0.0, "rgb(255,255,255)"],
                    [1.0, "rgb(0,104,55)"],
                ]
            elif zmax <= 0:
                # solo negativi: 0 è il massimo "logico"
                zmax = 0.0
                colorscale = [
                    [0.0, "rgb(165,0,38)"],
                    [1.0, "rgb(255,255,255)"],
                ]
            else:
                # misto: scala divergente centrata su 0
                p0 = (0.0 - zmin) / (zmax - zmin)
                colorscale = [
                    [0.0, "rgb(165,0,38)"],
                    [p0,  "rgb(255,255,255)"],
                    [1.0, "rgb(0,104,55)"],
                ]

        fig_hm = go.Figure(go.Heatmap(
            z=hm.T.values,
            x=hm.index,
            y=hm.columns,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Rolling Return", tickformat=".0%"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y}: %{z:.2%}<extra></extra>"
        ))

        fig_hm.update_layout(
            title=dict(
                text=(
                    "Mappa temporale dei rendimenti rolling (Total Return)"
                    "<br><span style='font-size:12px;color:gray'>"
                    "Rendimenti totali osservati ex-post su finestre mobili di diversa durata."
                    "</span>"
                ),
                x=0.5
            ),
            height=420,
            template="plotly_white"
        )

    # -----------------------------
    # Horizon analysis
    # -----------------------------
    analysis = None
    fig_loss = None
    if add_horizon_analysis:
        analysis = analyze_rolling_horizons(
            roll_cum_df.loc[analysis_start:analysis_end],
            loss_threshold=loss_threshold,
            loss_thresholds=loss_thresholds,
            min_obs=min_obs,
            target_prob=loss_target_prob
        )

        if add_loss_prob_curve:
            fig_loss = plot_loss_probability_curve(
                analysis,
                show_point_labels=show_loss_point_labels
            )

    # -----------------------------
    # Return
    # -----------------------------
    if return_extras:
        return {
            "fig": fig,
            "fig_hm": fig_hm,
            "fig_loss": fig_loss,
            "analysis": analysis,
            "roll_cum_df": roll_cum_df,
            "windows_df": windows_df,
        }

    if return_heatmap:
        return fig, fig_hm

    return fig
    
def plot_annual_return_triangle(
    pf,
    resample_freq: str = "YE",
    run_as_app: bool = False
) -> Union[Tuple, Tuple[object, pd.DataFrame, str]]:
    """
    Calcola un "triangolo" di rendimenti medi annuali rolling e lo plotta
    con etichette X colorate: rosso se la colonna ha almeno un rendimento negativo,
    verde altrimenti.

    Returns:
        (fig, triangle_df) oppure (fig, triangle_df, msg) se run_as_app=True
    """
    # 1) Prendi i rendimenti giornalieri e ricostruisci price index
    dr = pf.returns()
    if isinstance(dr, pd.DataFrame):
        dr = dr.iloc[:, 0]
    price_index = (1 + dr).cumprod()

    # 2) Prezzo di fine anno
    alias_map = {"A": "YE", "Y": "YE", "YE": "YE"}
    freq = alias_map.get(resample_freq, "YE")
    yearly_price = price_index.resample(freq).last().to_frame("Price")

    # 3) Log-return annuale
    annual_ret = np.log(yearly_price["Price"] / yearly_price["Price"].shift(1)).dropna().to_frame("Return")

    # --- FIX ROBUSTO: anno di fine finestra corretto (gestisce anche timestamp al 01/01) ---
    annual_ret.index = (pd.to_datetime(annual_ret.index) - pd.Timedelta(days=1)).year
    annual_ret.index.name = "Year"

    # 4) Rolling mean su finestre nY
    total_years = len(annual_ret)
    windows = list(range(total_years, 0, -1))
    for n in windows:
        annual_ret[f"{n}Y"] = annual_ret["Return"].rolling(window=n).mean()
    triangle_df = annual_ret.drop(columns="Return")

    # 5) Orizzonte minimo consigliato
    recommended = None
    for n in sorted(windows):
        vals = triangle_df[f"{n}Y"].dropna()
        if len(vals) > 0 and (vals > 0).all():
            recommended = n
            break

    if recommended:
        if run_as_app:
            msg = (
                "Triangolo dei rendimenti medi annualizzati (CAGR). "
                "Analisi ex-post su finestre discrete di ingresso e uscita. "
                "La valutazione dell’orizzonte minimo di investimento è fornita "
                "dalla mappa rolling e dalla curva di probabilità di perdita."
            )
        else:
            msg = (
                "Triangolo dei rendimenti medi annualizzati (CAGR).\n"
                "Analisi ex-post su finestre discrete di ingresso e uscita.\n"
                "La valutazione dell’orizzonte minimo di investimento è fornita "
                "dalla mappa rolling e dalla curva di probabilità di perdita."
            )
    else:
        msg = (
            "Triangolo dei rendimenti medi annualizzati (CAGR).\n"
            "Analisi ex-post su finestre discrete di ingresso e uscita.\n"
            "La valutazione dell’orizzonte minimo di investimento è fornita "
            "dalla mappa rolling e dalla curva di probabilità di perdita."
        )

    if run_as_app:
        import streamlit as st
        st.markdown(msg,unsafe_allow_html=False)
    else:
        print(msg)

    # 6) Plot con plotly_heatmap_triangle
    fig = plotly_heatmap_triangle(
        triangle_df,
        vmin=-0.2,
        vmax=0.2,
        colorscale='RdYlGn',
        # title="Triangolo dei rendimenti medi annuali (rolling log-return)",
        width=900,
        height=700
    )

    # 7) Colora le tick labels sull'asse X
    cols = list(triangle_df.columns)
    # per ogni col, se almeno un valore < 0 → rosso, else verde
    tick_colors = [
        "red" if (triangle_df[col] < 0).any() else "green"
        for col in cols
    ]
    # creiamo ticktext con span colorato
    tickvals = cols
    ticktext = [
        f"<span style='color:{c}'>{val}</span>"
        for val, c in zip(cols, tick_colors)
    ]
    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext
    )

    if run_as_app:
        return fig, triangle_df, msg
    else:
        return fig, triangle_df
    
def plotly_heatmap_triangle(
    triangle_df: pd.DataFrame,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscale: str = 'RdYlGn',
    title: Optional[str] = None,   # <-- default None
    width: int = 900,
    height: int = 700
) -> go.Figure:
    # 1) Drop righe/colonne piene di NaN
    df = triangle_df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # 2) Prepara z
    z = df.values * 100
        
    # 3) Prepara text ma sostituisci NaN con stringa vuota
    text = []
    for i, row in enumerate(z):
        txt_row = []
        end_year = int(df.index[i])
        for j, val in enumerate(row):
            if pd.isna(df.iat[i, j]):
                txt_row.append("")
            else:
                n = int(str(df.columns[j]).replace("Y", ""))
                start_year = end_year - n + 1
                txt_row.append(f"{val:.1f}%<br>{start_year}-{end_year}")
        text.append(txt_row)

    # 4) Etichette assi
    x_labels = [str(col) for col in df.columns]
    y_labels = [str(idx) for idx in df.index]

    # 5) Costruisci heatmap
    heatmap = go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        text=text,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmin=(vmin * 100) if vmin is not None else None,
        zmax=(vmax * 100) if vmax is not None else None,
        colorbar=dict(title="%")
    )

    fig = go.Figure(data=heatmap)

    # 6) Layout generale

    if title is None:
        plot_title = dict(
            text=(
                "Triangolo dei rendimenti medi annui (CAGR)"
                "<br><span style='font-size:12px;color:gray'>"
                "Rendimenti medi annui calcolati tra date discrete di ingresso e uscita."
                "</span>"
            ),
            x=0.5,
            xanchor="center"
        )
    else:
        plot_title = title

    fig.update_layout(
        # title=title,
        title=plot_title,
        width=width,
        height=height,
        margin=dict(l=100, r=40, t=80, b=80),
    )
    # 7) Asse X
    fig.update_xaxes(
        title_text="Finestra Mobile (anni)",
        ticks="outside",
        tickangle=0,
    )

    # 8) Asse Y spostato a sinistra
    fig.update_yaxes(
        title_text="Anno di Fine Finestra",
        autorange="reversed",
        ticks="outside",
        tickangle=0,
        side="left"
    )

    return fig
    
def build_and_plot_portfolio_contributions(
    portfolio: 'vbt.Portfolio',
    title: str,
    benchmark: str = "SPY",
    benchmark_data: pd.Series | None = None,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    show_report: bool = True,
):
    """
    Costruisce e plotta i contributi al portafoglio.

    - Curve per singolo asset (contributi)
    - Curva aggregata di portafoglio
    - Eventuale benchmark (interno o esterno)

    Robustezza:
    - start_date/end_date accettano str/Timestamp/datetime/None
    - finestra clampata sul range dati disponibile
    - skip pulito se la finestra non interseca i dati

    NOTE CHIAVE:
    - NESSUN download da yfinance.
    - Il benchmark viene SEMPRE iniettato come serie di returns nel dict `portfolios_returns`.
      * Se `benchmark_data` è fornito => usato come PREZZI (Close) esterni.
      * Altrimenti => usato benchmark interno vectorbt: `portfolio.benchmark_returns(...)`.
    - `plot_multiple_portfolios` viene chiamata con benchmark=None, benchmark_data=None per evitare
      qualsiasi interpretazione del benchmark come ticker.
    """

    import pandas as pd
    import numpy as np

    # -----------------------------
    # Utility
    # -----------------------------
    def _to_ts(x):
        if x is None:
            return None
        try:
            return pd.to_datetime(x)
        except Exception:
            return None

    def _normalize_index(s: pd.Series) -> pd.Series:
        s = s.dropna().copy()
        try:
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
        except Exception:
            pass
        s.index = pd.to_datetime(s.index).normalize()
        s = s[~s.index.duplicated(keep="last")]
        return s.sort_index()

    def _resolve_window(idx: pd.Index, start, end):
        if idx is None or len(idx) == 0:
            return (None, None)

        s = _to_ts(start)
        e = _to_ts(end)

        idx_min = pd.to_datetime(idx.min())
        idx_max = pd.to_datetime(idx.max())

        if s is None:
            s = idx_min
        if e is None:
            e = idx_max

        if s > e:
            s, e = e, s

        # clamp
        if s < idx_min:
            s = idx_min
        if e > idx_max:
            e = idx_max

        if s > idx_max or e < idx_min:
            return (None, None)

        return (s, e)

    def _prices_to_returns(prices: pd.Series, target_idx: pd.DatetimeIndex) -> pd.Series:
        p = _normalize_index(prices)
        r = p.pct_change().dropna()
        target = pd.to_datetime(target_idx).normalize()
        r = r.reindex(target, method="ffill").fillna(0.0)
        return r

    # -----------------------------
    # Returns per-asset
    # -----------------------------
    try:
        asset_returns = portfolio.returns(group_by=False)
    except Exception:
        if show_report:
            print(f"ℹ️  Skip contributions '{title}': impossibile calcolare returns per-asset")
        return None

    if asset_returns is None or getattr(asset_returns, "empty", False):
        if show_report:
            print(f"ℹ️  Skip contributions '{title}': returns per-asset vuoti")
        return None

    # -----------------------------
    # Returns aggregati portafoglio
    # -----------------------------
    try:
        pf_returns = portfolio.returns()
    except Exception:
        if show_report:
            print(f"ℹ️  Skip contributions '{title}': impossibile calcolare portfolio.returns()")
        return None

    if pf_returns is None or getattr(pf_returns, "empty", False):
        if show_report:
            print(f"ℹ️  Skip contributions '{title}': returns portafoglio vuoti")
        return None

    # -----------------------------
    # Finestra robusta
    # -----------------------------
    s, e = _resolve_window(asset_returns.index, start_date, end_date)
    if s is None:
        if show_report:
            dr_min = asset_returns.index.min()
            dr_max = asset_returns.index.max()
            print(
                f"ℹ️  Skip contributions '{title}': finestra fuori range "
                f"(start={start_date}, end={end_date}, data_range={dr_min}→{dr_max})"
            )
        return None

    asset_returns_w = asset_returns.loc[s:e].dropna(how="all")
    pf_returns_w = pf_returns.loc[s:e].dropna()

    if asset_returns_w.empty and pf_returns_w.empty:
        if show_report:
            print(f"ℹ️  Skip contributions '{title}': nessun dato nella finestra {s}→{e}")
        return None

    # -----------------------------
    # Dict per plot (asset + portfolio hero)
    # -----------------------------
    portfolios_returns = {}
    for t in getattr(asset_returns_w, "columns", []):
        portfolios_returns[str(t)] = asset_returns_w[t].dropna()

    # label sentinella per riconoscere SEMPRE il portafoglio
    _PORTFOLIO_LABEL = f"__PORTFOLIO__::{title}"
    portfolios_returns[_PORTFOLIO_LABEL] = pf_returns_w

    # -----------------------------
    # BENCHMARK (interno o esterno) -> SEMPRE come serie returns
    # -----------------------------
    _BENCH_LABEL = f"__BENCH__::{benchmark}".strip() if benchmark else "__BENCH__"

    bench_ret = None
    try:
        if benchmark_data is not None:
            # benchmark esterno: benchmark_data sono PREZZI (Close)
            bench_ret = _prices_to_returns(benchmark_data, pf_returns_w.index)
        else:
            # benchmark interno vectorbt: returns già pronti
            br = portfolio.benchmark_returns(group_by=True)
            if isinstance(br, pd.DataFrame):
                br = br.mean(axis=1)
            br = _normalize_index(br)
            br = br.reindex(pd.to_datetime(pf_returns_w.index).normalize(), method="ffill").fillna(0.0)
            bench_ret = br
    except Exception:
        bench_ret = None

    if bench_ret is not None and not bench_ret.empty:
        portfolios_returns[_BENCH_LABEL] = bench_ret

    # -----------------------------
    # Plot (NESSUN ticker passato!)
    # -----------------------------
    fig = plot_multiple_portfolios(
        portfolios_returns,
        title=f"Contributi al portafoglio: {title}",
        benchmark=None,
        benchmark_data=None,
        start_date=s,
        end_date=e
    )

    # -----------------------------
    # Styling: asset grigi + top-N colorati, portfolio/benchmark hero
    # -----------------------------
    PORTFOLIO_COLOR = "blue"
    BENCHMARK_COLOR = "gray"
    PORTFOLIO_WIDTH = 3.5
    BENCHMARK_WIDTH = 3.0

    ASSET_GRAY = "#C9D1D9"
    ASSET_WIDTH = 1.0
    ASSET_OPACITY = 0.25

    # --- Top-N assets (colorati) ---
    TOPN = 5
    TOPN_WIDTH = 1.8
    TOPN_OPACITY = 0.80
    TOPN_PALETTE = ["#FF6B6B", "#F7B801", "#2EC4B6", "#9B5DE5", "#00BBF9", "#F15BB5"]

    asset_total_ret = (1.0 + asset_returns_w).prod(axis=0) - 1.0
    asset_total_ret = asset_total_ret.replace([np.inf, -np.inf], np.nan).dropna()
    top_assets = asset_total_ret.abs().sort_values(ascending=False).head(TOPN).index.tolist()
    top_color_map = {a: TOPN_PALETTE[i % len(TOPN_PALETTE)] for i, a in enumerate(top_assets)}

    def _is_portfolio_trace(tr):
        n = getattr(tr, "name", "") or ""
        return _PORTFOLIO_LABEL in n

    def _is_benchmark_trace(tr):
        n = getattr(tr, "name", "") or ""
        return _BENCH_LABEL in n

    def _normalize_asset_label(trace_name: str) -> str:
        """
        Normalizza i nomi trace generati da plot_multiple_portfolios.
        Esempi:
          'Portfolio (DTE.DE)' -> 'DTE.DE'
          'Portfolio (DHL.DE (contributo))' -> 'DHL.DE'
        """
        n = (trace_name or "").strip()

        # non toccare le label sentinella
        if n.startswith("__PORTFOLIO__::") or n.startswith("__BENCH__::"):
            return n

        if n.startswith("Portfolio (") and n.endswith(")"):
            n = n[len("Portfolio ("):-1].strip()

        n = n.replace("(contributo)", "").strip()
        return n

    for tr in fig.data:
        name = getattr(tr, "name", "") or ""

        # default: asset grigi
        tr.opacity = ASSET_OPACITY
        if hasattr(tr, "line") and tr.line is not None:
            tr.line.color = ASSET_GRAY
            tr.line.width = ASSET_WIDTH

        # portfolio hero
        if _is_portfolio_trace(tr):
            tr.opacity = 1.0
            if hasattr(tr, "line") and tr.line is not None:
                tr.line.color = PORTFOLIO_COLOR
                tr.line.width = PORTFOLIO_WIDTH
            tr.name = title
            continue

        # benchmark hero
        if _is_benchmark_trace(tr):
            tr.opacity = 1.0
            if hasattr(tr, "line") and tr.line is not None:
                tr.line.color = BENCHMARK_COLOR
                tr.line.width = BENCHMARK_WIDTH
            tr.name = benchmark if benchmark else "Benchmark"
            continue

        # top-N: colorati e più visibili
        asset_key = _normalize_asset_label(name)
        if asset_key in top_color_map:
            tr.opacity = TOPN_OPACITY
            if hasattr(tr, "line") and tr.line is not None:
                tr.line.color = top_color_map[asset_key]
                tr.line.width = TOPN_WIDTH

    return fig
    
def plot_ts_portfolio(
    final_portfolio, 
    portfolio_ts, 
    portfolio_title="Composito", 
    width=1000,
    start_date=None,
    end_date=None,    
    # # nuovo parametro: mostrare le figure a video
    # show_report: bool = True,
):
    """
    Plotta i rendimenti cumulativi dei TS (ricavati da portfolio_ts) e del portafoglio finale,
    normalizzando a 1.0 alla start_date se specificata.

    Parametri:
    - final_portfolio: oggetto vectorbt.Portfolio aggregato
    - portfolio_ts: lista di dict, ognuno con chiave 'portfolio' e 'symbol'
    - portfolio_title: nome da visualizzare per il portafoglio finale
    - width: larghezza grafico in pixel
    - start_date: data iniziale per il filtro (datetime o stringa 'YYYY-MM-DD')
    - end_date: data finale per il filtro (datetime o stringa 'YYYY-MM-DD')
    """
    # 1. Costruzione DataFrame rendimenti cumulati dei TS
    cumulative_intermedi = pd.DataFrame()

    for ts in portfolio_ts:
        symbol = ts["symbol"]
        p = ts["portfolio"]
        cum_returns = 1.0 + p.cumulative_returns()
        cum_returns = cum_returns.copy()

        # Applichiamo filtro temporale e normalizzazione
        if start_date is not None:
            cum_returns = cum_returns[cum_returns.index >= pd.to_datetime(start_date)]
            if not cum_returns.empty:
                cum_returns /= cum_returns.iloc[0]  # normalizza a 1.0

        if end_date is not None:
            cum_returns = cum_returns[cum_returns.index <= pd.to_datetime(end_date)]

        cumulative_intermedi[symbol] = cum_returns

    # 2. Portafoglio finale
    cum_returns_final = 1.0 + final_portfolio.cumulative_returns()
    cum_returns_final = cum_returns_final.copy()

    if start_date is not None:
        cum_returns_final = cum_returns_final[cum_returns_final.index >= pd.to_datetime(start_date)]
        if not cum_returns_final.empty:
            cum_returns_final /= cum_returns_final.iloc[0]  # normalizza a 1.0

    if end_date is not None:
        cum_returns_final = cum_returns_final[cum_returns_final.index <= pd.to_datetime(end_date)]

    # 3. Costruzione grafico
    fig = go.Figure()

    # Tracciati TS
    for col in cumulative_intermedi.columns:
        fig.add_trace(
            go.Scatter(
                x=cumulative_intermedi.index,
                y=cumulative_intermedi[col],
                mode='lines',
                name=f"TS - {col}",
                line=dict(width=1),
                opacity=0.5
            )
        )

    # Tracciato finale
    fig.add_trace(
        go.Scatter(
            x=cum_returns_final.index,
            y=cum_returns_final,
            mode='lines',
            name=f"Portfolio {portfolio_title}",
            line=dict(width=4, color='blue'),
            opacity=1.0
        )
    )
    # 6.1) Aggiungi linea orizzontale rossa a y=1
    fig.add_hline(
        y=1,
        line=dict(color='red', width=2, dash='dash'),
    )

    # Layout
    fig.update_layout(
        title=f"Portfolio {portfolio_title} e Trading System",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all", label="All")
                ])
            ),
        ),
        width=width,
        height=600,
        template="plotly_white",
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='auto')
    )

    # if show_report: fig.show()

    return fig #, cum_returns_final

# ============================================================
# SEZIONE 2: _generate_portfolio_performance_core_refactored completa
# ============================================================

def _generate_portfolio_performance_core_refactored(
    # Portfolios
    pf: 'vbt.Portfolio',
    portfolio_title: str,
    pf_b_h: 'vbt.Portfolio' = None,
    
    # switch comportamento
    mode: str = "standard",  # standard|rotational|lazy
    
    # caratteristiche "standard"
    portfolio_ts: list[dict] | None = None,
    
    # caratteristiche "rotational"
    sel_tickers: list | None = None,

    # benchmark
    benchmark_mode: str = "internal",
    benchmark: str = "Benchmark",
    benchmark_data: Optional[pd.Series] = None,   # prices if external
    
    alpha_analysis: bool = True,
    risk_free_rate: float = 0.02,
    rolling_window: Optional[int] = 252,
    show_report: bool = True,
    show_plots: bool = False,
    vbt_plot_width: Optional[int] = 1000,
    universe: Optional[list] = None,
):
    """
    Refactoring a blocchi:
      - header PRIMA di tutto
      - STATISTICHE complete + SINTESI boxed
      - Rolling CAPM alpha+beta (Plotly) subito dopo i plot vbt
      - Ripristina rolling panels (cum/rolling + heatmap + prob loss + triangle)
      - Ripristina grafico rendimenti totali per titolo
      - Benchmark interno: evita yfinance nei contributions (crea benchmark_data sintetico da bm_ret)
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from tabulate import tabulate

    # -------------------------
    # Helpers: show
    # -------------------------
    def _maybe_show_fig(fig):
        if not show_plots:
            return
        try:
            fig.show()
        except Exception:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception:
                pass

    # -------------------------
    # Helpers: dates
    # -------------------------
    def _safe_period_str(stats_obj) -> tuple[str, str, str]:
        try:
            s = pd.to_datetime(stats_obj.get("Start")).date().isoformat()
        except Exception:
            s = str(stats_obj.get("Start", ""))
        try:
            e = pd.to_datetime(stats_obj.get("End")).date().isoformat()
        except Exception:
            e = str(stats_obj.get("End", ""))
        return s, e, f"{s} → {e}"

    def _ytd_start():
        now = pd.Timestamp.now()
        return pd.Timestamp(now.year, 1, 1)

    # -------------------------
    # Helpers: benchmark “prices” from returns (to avoid yfinance)
    # -------------------------
    def _returns_to_price(ret: pd.Series, base: float = 100.0) -> pd.Series:
        r = ret.copy().dropna()
        if r.empty:
            return r
        try:
            if getattr(r.index, "tz", None) is not None:
                r.index = r.index.tz_localize(None)
        except Exception:
            pass
        r.index = pd.to_datetime(r.index).normalize()
        return (1.0 + r).cumprod() * float(base)

    def is_one_ticker(pf) -> bool:
        """
        Ritorna True se il Portfolio contiene un solo ticker/colonna.
        Robusto a pf.assets() che può essere Series o DataFrame.
        """
        assets = pf.assets()
    
        # Caso più comune: DataFrame (multi-ticker)
        if isinstance(assets, pd.DataFrame):
            return assets.shape[1] == 1
    
        # Caso: Series (spesso single-ticker o single-group)
        if isinstance(assets, pd.Series):
            # Se è una Series con name che rappresenta un ticker -> considerala 1 ticker.
            # Se è una Series "group" aggregata, comunque lato tickers è 1 (perché non hai breakdown).
            return True
    
        # Fallback: prova ad inferire da pf.close (di solito è coerente)
        close = getattr(pf, "close", None)
        if close is not None:
            if isinstance(close, pd.DataFrame):
                return close.shape[1] == 1
            if isinstance(close, pd.Series):
                return True
    
        # Ultimo fallback: non lo sappiamo => assume multi per prudenza
        return False

    # -------------------------
    # Block 0: header + summary build + print tables
    # -------------------------
    def _build_summary_and_print():
        stats_tmp = pf.stats()
        start_str, end_str, period_str = _safe_period_str(stats_tmp)

        one_ticker = is_one_ticker(pf)
        portfolio_type = "Titolo" if one_ticker else "Portfolio"

    
        header = f"📈 Statistiche {portfolio_type} {portfolio_title} ({period_str})"

        # header = f"📈 Statistiche Portfolio {portfolio_title} ({period_str})"
        
        if show_report:
            print(header)

        summary = create_portfolio_summary_refactored(
            pf,
            sel_tickers=sel_tickers,
            alpha_analysis=alpha_analysis,
            risk_free_rate=risk_free_rate,
            # benchmark_mode=benchmark_mode if mode != "standard" else ("portfolio" if pf_b_h is not None else benchmark_mode),
            benchmark_mode=benchmark_mode,
            benchmark_name=(f"{benchmark}" if benchmark else "Benchmark"),
            benchmark_data=benchmark_data,
            benchmark_portfolio=pf_b_h,
            annualization=252,
            rolling_window=rolling_window,
            show=False,
            return_formatted=True
        )

        stats_df = summary["stats_df"]
        stats_df_raw = summary.get("stats_df_raw", stats_df)
        bm_ret = summary.get("benchmark_returns", None)
        bm_meta = summary.get("benchmark_meta", {"benchmark_source": None, "benchmark_name": benchmark})
        capm = summary.get("capm")
        rolling_capm = summary.get("rolling_capm")

        # STATISTICHE complete
        if show_report:
            print("\n================= STATISTICHE ==================")
            try:
                display(stats_df.rename(columns={"Valore": ""}))
            except Exception:
                print(stats_df.rename(columns={"Valore": ""}))

        # SINTESI boxed (stesso formato)
        if show_report:
            print("\n================= SINTESI ==================")
            df_printing = stats_df.copy()

            headers_raw = [
                "Importo investito", "Valore finale netto", "CAGR (252)",
                "Max Drawdown", "Deviazione standard", "Rapporto di Sharpe",
                "Operazioni al mese", "Durata minima in guadagno"
            ]

            def _find(key, default="n/a"):
                return df_printing.loc[key, "Valore"] if key in df_printing.index else default

            vals_ordered = [
                _find("Importo investito (€)"),
                _find("Valore patrimoniale netto (€)"),
                _find("CAGR"),
                _find("Max Drawdown"),
                _find("Volatilità annua"),
                _find("Rapporto di Sharpe"),
                _find("Operazioni al mese"),
                _find("Durata minima in guadagno (giorni)")
            ]
            tab = tabulate([vals_ordered], headers=headers_raw, tablefmt="fancy_grid",
                           colalign=("center",) * len(headers_raw))
            print(tab)

        # sintesi_df (come prima)
        def _pick(key):
            return stats_df.loc[key, "Valore"] if key in stats_df.index else "n/a"

        sintesi_df = pd.DataFrame([{
            "Importo investito": _pick("Importo investito (€)"),
            "Valore finale netto": _pick("Valore patrimoniale netto (€)"),
            "CAGR (252)": _pick("CAGR"),
            "Max Drawdown": _pick("Max Drawdown"),
            "Deviazione standard": _pick("Volatilità annua"),
            "Rapporto di Sharpe": _pick("Rapporto di Sharpe"),
            "Operazioni al mese": _pick("Operazioni al mese"),
            "Durata minima in guadagno": _pick("Durata minima in guadagno (giorni)"),
        }])

        return {
            "header": header,
            "stats_tmp": stats_tmp,
            "stats_df": stats_df,
            "stats_df_raw": stats_df_raw,
            "sintesi_df": sintesi_df,
            "bm_ret": bm_ret,
            "bm_meta": bm_meta,
            "capm": capm,
            "rolling_capm": rolling_capm,
        }

    # -------------------------
    # Block 1: vbt base plots
    # -------------------------
    def _plot_vbt_base(figs: list):
        try:
            subplots = ['orders', 'trade_pnl'] if mode == "standard" else []    
            subplots.extend(['cum_returns', 'drawdowns', 'underwater', 'gross_exposure'])
            fig_value = pf.plot(
                width=vbt_plot_width,
                subplots=subplots
            )
            figs.append(fig_value)
            _maybe_show_fig(fig_value)
        except Exception:
            pass
            
    # -------------------------
    # Block 1.1: Time-series portfolio (solo standard) — RIPRISTINATO
    # -------------------------
    def _plot_ts_portfolio_standard(figs: list):
        if mode != "standard":
            return
        if not portfolio_ts:
            return

        try:
            fig_ts = plot_ts_portfolio(
                final_portfolio=pf,
                portfolio_ts=portfolio_ts,
                portfolio_title=portfolio_title,
                # width=vbt_plot_width,
                width=None,
                start_date=ytd(),          # come originale
                # show_report=False
            )
            figs.append(fig_ts)
            _maybe_show_fig(fig_ts)
        except Exception as e:
            if show_report:
                print("Warning: plot_ts_portfolio failed:", str(e))
    # -------------------------
    # Block 2: rolling CAPM alpha+beta (Plotly) right after vbt
    # -------------------------
    def _plot_rolling_capm_alpha_beta(figs: list, bm_ret: pd.Series | None):
    
        rolling_alpha_df = None  # <-- FIX: evita UnboundLocalError in ogni path
    
        def _diag_bm(bm: pd.Series | None) -> str:
            if bm is None:
                return "bm_ret=None"
            try:
                n = int(bm.shape[0])
                nn = int(bm.dropna().shape[0])
                idx = bm.index
                i0 = pd.to_datetime(idx.min()) if n else None
                i1 = pd.to_datetime(idx.max()) if n else None
                return f"bm_ret: n={n}, nonnull={nn}, range={i0}→{i1}"
            except Exception:
                return "bm_ret: (diagnostica non disponibile)"
    
        def _diag_df(df: pd.DataFrame | None, window: int) -> str:
            if df is None:
                return "rolling_alpha_df=None"
            if getattr(df, "empty", True):
                return "rolling_alpha_df empty=True"
    
            cols = list(df.columns)
            msg = [f"rolling_alpha_df: n={len(df)}, cols={cols}"]
            try:
                msg.append(f"range={pd.to_datetime(df.index.min())}→{pd.to_datetime(df.index.max())}")
            except Exception:
                pass
    
            expected = ["alpha_ann_pct", "p_alpha", "beta"]
            missing = [c for c in expected if c not in df.columns]
            if missing:
                msg.append(f"missing_cols={missing}")
    
            for c in ["alpha_ann_pct", "p_alpha", "beta"]:
                if c in df.columns:
                    s = df[c]
                    msg.append(f"{c}: nonnull={int(s.notna().sum())}/{len(s)}")
    
            if "alpha_ann_pct" in df.columns and "beta" in df.columns:
                valid = df[["alpha_ann_pct", "beta"]].dropna()
                msg.append(f"valid_rows(alpha+beta)= {len(valid)}/{len(df)}")
    
            msg.append(f"rolling_window={int(window)}")
            return " | ".join(msg)
    
        # -------------------------
        # Guard-rail benchmark
        # -------------------------
        if bm_ret is None:
            if show_report:
                print(f"ℹ️  Rolling CAPM skipped: benchmark returns missing ({_diag_bm(bm_ret)})")
            return None
    
        try:
            bm_len = int(len(bm_ret))
        except Exception:
            bm_len = 0
    
        if bm_len == 0:
            if show_report:
                print(f"ℹ️  Rolling CAPM skipped: benchmark returns empty ({_diag_bm(bm_ret)})")
            return None
    
        # Normalizza index benchmark
        try:
            bm_ret = bm_ret.copy()
            if bm_ret.index.tz is not None:
                bm_ret.index = bm_ret.index.tz_localize(None)
            bm_ret.index = pd.to_datetime(bm_ret.index).normalize()
        except Exception:
            pass
    
        w = int(rolling_window or 252)
    
        try:
            bm_nonnull = int(bm_ret.dropna().shape[0])
        except Exception:
            bm_nonnull = 0
    
        if bm_nonnull < max(30, w):
            if show_report:
                print(
                    "ℹ️  Rolling CAPM skipped: benchmark too short for rolling window. "
                    f"need≥{max(30, w)} non-null obs, got {bm_nonnull}. ({_diag_bm(bm_ret)})"
                )
            return None
    
        # -------------------------
        # Compute rolling alpha/beta
        # -------------------------
        try:
            rolling_alpha_df, _ = rolling_alpha_section(
                pf,
                bm_ret,
                window=w,
                risk_free_rate=risk_free_rate,
                annualization=252,
                show_plot=False
            )
        except Exception as e:
            if show_report:
                print("⚠️  Rolling CAPM failed inside rolling_alpha_section.")
                print("    Error:", str(e))
                print("    " + _diag_bm(bm_ret))
            return None
    
        # -------------------------
        # Validate output before plotting
        # -------------------------
        if rolling_alpha_df is None or rolling_alpha_df.empty:
            if show_report:
                print("ℹ️  Rolling CAPM skipped: rolling_alpha_df is None/empty.")
                print("    " + _diag_bm(bm_ret))
                print("    " + _diag_df(rolling_alpha_df, w))
            return rolling_alpha_df
    
        needed_cols = ["alpha_ann_pct", "p_alpha", "beta"]
        missing = [c for c in needed_cols if c not in rolling_alpha_df.columns]
        if missing:
            if show_report:
                print("ℹ️  Rolling CAPM skipped: rolling_alpha_df missing required columns:", missing)
                print("    " + _diag_df(rolling_alpha_df, w))
            return rolling_alpha_df
    
        valid_df = rolling_alpha_df[["alpha_ann_pct", "beta", "p_alpha"]].dropna(subset=["alpha_ann_pct", "beta"])
        if valid_df.empty:
            if show_report:
                print("ℹ️  Rolling CAPM skipped: no valid rows after dropna(alpha_ann_pct,beta).")
                print("    " + _diag_df(rolling_alpha_df, w))
                try:
                    display(rolling_alpha_df.tail(5))
                except Exception:
                    pass
            return rolling_alpha_df
    
        # -------------------------
        # Plot (plotly)
        # -------------------------
        try:
            fig_roll = go.Figure()
    
            fig_roll.add_trace(go.Scatter(
                x=rolling_alpha_df.index,
                y=rolling_alpha_df["alpha_ann_pct"],
                mode="lines",
                name="Alpha annuo (%)",
                line=dict(width=2),
                hovertemplate="Date: %{x}<br>Alpha annuo: %{y:.3f}%<br>p: %{customdata[0]:.3g}",
                customdata=rolling_alpha_df[["p_alpha"]].values
            ))
    
            sig_mask = (rolling_alpha_df["p_alpha"] < 0.05) & rolling_alpha_df["p_alpha"].notna()
            if sig_mask.any():
                fig_roll.add_trace(go.Scatter(
                    x=rolling_alpha_df.index[sig_mask],
                    y=rolling_alpha_df.loc[sig_mask, "alpha_ann_pct"],
                    mode="markers",
                    name="Alpha signif. (p<0.05)",
                    marker=dict(size=8, symbol="circle-open"),
                    hovertemplate="Date: %{x}<br>Alpha annuo: %{y:.3f}%<br>p: %{customdata[0]:.3g}",
                    customdata=rolling_alpha_df.loc[sig_mask, ["p_alpha"]].values
                ))
    
            fig_roll.add_trace(go.Scatter(
                x=rolling_alpha_df.index,
                y=rolling_alpha_df["beta"],
                mode="lines",
                name="Beta (rolling)",
                line=dict(width=1, dash="dot"),
                yaxis="y2",
                hovertemplate="Date: %{x}<br>Beta: %{y:.3f}"
            ))
    
            fig_roll.update_layout(
                title="Rolling CAPM — Alpha annuo (%) e Beta (rolling)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Alpha annuo (%)", zeroline=True),
                yaxis2=dict(title="Beta (rolling)", overlaying="y", side="right", showgrid=False),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    font=dict(size=12),
                    itemsizing="constant"
                ),
                height=420,
                margin=dict(l=50, r=70, t=60, b=40)
            )
    
            fig_roll.add_hline(y=0, line=dict(dash="dash", width=1),
                               annotation_text="0", annotation_position="top left")
    
            figs.append(fig_roll)
            _maybe_show_fig(fig_roll)
    
        except Exception as e:
            if show_report:
                print("⚠️  Rolling CAPM plotly build failed.")
                print("    Error:", str(e))
                print("    " + _diag_df(rolling_alpha_df, w))
    
        return rolling_alpha_df  
    

    # -------------------------
    # Block 3: alpha diagnostics text (after rolling plot)
    # -------------------------
    def _print_alpha_diagnostics(capm, rolling_alpha_df):
        if show_report and alpha_analysis:
            print("\n================= ANALISI ALPHA ==================")
            print(comment_alpha_diagnostics(capm, rolling_alpha_df))

    # -------------------------
    # Block 4: build portfolios_returns + vs benchmark plots (incl YTD)
    # -------------------------
    def _plot_vs_benchmark(figs: list, stats_tmp, bm_ret: pd.Series | None, bm_meta: dict):
        # returns strategy
        try:
            returns_strat = pf.returns()
        except Exception:
            returns_strat = _pf_returns_series(pf)
    
        portfolios_returns = {f"{portfolio_title} (Strategy)": returns_strat}
    
        # optional B&H portfolio
        if pf_b_h is not None:
            try:
                portfolios_returns[f"{portfolio_title} (B&H)"] = pf_b_h.returns()
            except Exception:
                pass
    
        # add benchmark RETURNS directly (no yfinance)
        if bm_ret is not None and len(bm_ret) > 0:
            bench_name = (bm_meta.get("benchmark_name") or "Benchmark")
            portfolios_returns[str(bench_name)] = bm_ret
    
        # --- normalize full-period window (for de-dup YTD) ---
        p_start = stats_tmp.get("Start")
        p_end   = stats_tmp.get("End")
        
        try:
            p_start_n = pd.to_datetime(p_start).tz_localize(None) if getattr(pd.to_datetime(p_start), "tzinfo", None) else pd.to_datetime(p_start)
            p_end_n   = pd.to_datetime(p_end).tz_localize(None)   if getattr(pd.to_datetime(p_end), "tzinfo", None)   else pd.to_datetime(p_end)
            p_start_n = pd.Timestamp(p_start_n).normalize()
            p_end_n   = pd.Timestamp(p_end_n).normalize()
        except Exception:
            p_start_n = None
            p_end_n = None
    
        # full period plot
        try:
            start_year = getattr(p_start, "year", None)
            curr_year = pd.Timestamp.now().year
            base = 0.0 if (start_year == curr_year) else 1.0
    
            fig_vs = plot_multiple_portfolios(
                portfolios_returns,
                benchmark=None,
                title="Performance vs Benchmark",
                benchmark_data=None,
                # start_date=p_start,
                # end_date=p_end,
                base=base
            )
            figs.append(fig_vs)
            _maybe_show_fig(fig_vs)
        except Exception:
            pass
    
        # YTD plot (only if effective window differs from full-period effective window)
        try:
            strat_series = portfolios_returns.get(f"{portfolio_title} (Strategy)", None)
            if strat_series is not None:
                idx = pd.to_datetime(strat_series.index)
                try:
                    if getattr(idx, "tz", None) is not None:
                        idx = idx.tz_localize(None)
                except Exception:
                    pass
                idx = pd.DatetimeIndex(idx).normalize()
    
                # requested windows
                ytd_start_req = _ytd_start().normalize()
                ytd_end_req   = idx.max()
    
                # full requested window (fallback to idx bounds)
                full_start_req = p_start_n if p_start_n is not None else idx.min()
                full_end_req   = p_end_n   if p_end_n   is not None else idx.max()
    
                # ---- EFFECTIVE windows after clamp to available data ----
                eff_full_start = max(full_start_req, idx.min())
                eff_full_end   = min(full_end_req,   idx.max())
    
                eff_ytd_start  = max(ytd_start_req,  idx.min())
                eff_ytd_end    = min(ytd_end_req,    idx.max())
    
                same_as_full = (eff_full_start == eff_ytd_start) and (eff_full_end == eff_ytd_end)
    
                if (eff_ytd_end >= eff_ytd_start) and (not same_as_full):
                    fig_vs_ytd = plot_multiple_portfolios(
                        portfolios_returns,
                        title="Performance vs Benchmark (YTD)",
                        benchmark=None,
                        benchmark_data=None,
                        start_date=eff_ytd_start,
                        end_date=eff_ytd_end,
                        base=base
                    )
                    figs.append(fig_vs_ytd)
                    _maybe_show_fig(fig_vs_ytd)
    
        except Exception as e:
            if show_report:
                print("Warning: YTD plot failed:", str(e))    
                
        return portfolios_returns

    

    # -------------------------
    # Block 5: periodic plots (annual/monthly)
    # -------------------------
    def _plot_periodic(figs: list, portfolios_returns: dict):
        try:
            fig_annual = plot_annual_performance(portfolios_returns, benchmark=None, benchmark_data=None)
            figs.append(fig_annual); _maybe_show_fig(fig_annual)
        except Exception:
            pass

        try:
            fig_annual_hist = plot_year_returns_histogram(
                pf,
                title=f"Istogramma dei rendimenti annuali – {portfolio_title}",
                panel_width=0.34, gap=0.02, hist_fill=0.95, min_years=2
            )
            if fig_annual_hist is not None:
                figs.append(fig_annual_hist); _maybe_show_fig(fig_annual_hist)
        except Exception:
            pass

        try:
            fig_monthly = plot_monthly_returns(
                pf,
                eoy=True,
                title=f"Portfolio {portfolio_title} - Rendimenti mensili (%)",
                width=vbt_plot_width if mode == "standard" else None
            )
            figs.append(fig_monthly); _maybe_show_fig(fig_monthly)
        except Exception:
            pass

        try:
            fig_monthly_hist = plot_monthly_returns_histogram(
                pf,
                title=f"Istogramma dei rendimenti mensili – {portfolio_title}"
            )
            figs.append(fig_monthly_hist); _maybe_show_fig(fig_monthly_hist)
        except Exception:
            pass

    # -------------------------
    # Block 6: rolling panels (restore original “rolling” section)
    # -------------------------
    def _plot_rolling_panels(figs: list, stats_tmp):
        # solo rotational/lazy e solo se non siamo all’anno corrente (come tua logica originale)
        if mode not in ("rotational", "lazy"):
            return

        try:
            p_start = stats_tmp.get("Start")
            start_year = getattr(p_start, "year", None)
            curr_year = pd.Timestamp.now().year
            if start_year == curr_year:
                return
        except Exception:
            # se non riesce, proviamo comunque
            pass

        try:
            out_roll = plot_cumulative_and_rolling_returns(
                pf,
                horizons_years=[1, 2, 3, 4, 5],
                add_fan_chart=False,
                add_heatmap=True,
                return_extras=True,
                add_horizon_analysis=True,
            )
            if isinstance(out_roll, dict) and out_roll.get("fig") is not None:
                figs.append(out_roll["fig"]); _maybe_show_fig(out_roll["fig"])
            if isinstance(out_roll, dict) and out_roll.get("fig_hm") is not None:
                figs.append(out_roll["fig_hm"]); _maybe_show_fig(out_roll["fig_hm"])
            if isinstance(out_roll, dict) and out_roll.get("fig_loss") is not None:
                figs.append(out_roll["fig_loss"]); _maybe_show_fig(out_roll["fig_loss"])
        except Exception:
            pass

        try:
            fig_triangle, _ = plot_annual_return_triangle(pf, resample_freq="YE")
            figs.append(fig_triangle); _maybe_show_fig(fig_triangle)
        except Exception:
            pass

    # -------------------------
    # Block 7: selection frequencies (rotational)
    # -------------------------
    def _plot_selection_freq(figs: list, stats_tmp):
        freq_df = None
        if mode == "rotational" and sel_tickers is not None:
            try:
                fig_sel_tickers, freq_df, freq_full_df, unselected_info = plot_ticker_frequencies(
                    sel_tickers,
                    start_date=stats_tmp.get("Start"),
                    end_date=stats_tmp.get("End"),
                    include_prev=True,
                    universe=universe or []
                )
                figs.append(fig_sel_tickers); _maybe_show_fig(fig_sel_tickers)

                unselected_df, n_unselected, pct_unselected = unselected_info
                if show_report:
                    print(f"{n_unselected} tickers su {len(universe or [])} mai selezionati ({pct_unselected:.1%})")
            except Exception:
                pass
        return freq_df

    # -------------------------
    # Block 8: total return per ticker — RIPRISTINATO (standard + rotational)
    # -------------------------
    def _plot_total_return_per_ticker(figs: list):
        try:
            if mode == "rotational":
                performance = pf.total_return(group_by=False) * 100
                fig_tickers_perf = plot_total_return_per_ticker(performance[performance != 0])
                figs.append(fig_tickers_perf)
                _maybe_show_fig(fig_tickers_perf)

            elif mode == "standard":
                if portfolio_ts is None or len(portfolio_ts) == 0:
                    return

                # portfolio_ts atteso: list[dict] con chiavi "symbol" e "returns"
                # Esempio: [{"symbol":"XYZ","returns":0.12}, ...]
                returns = pd.Series({d["symbol"]: d["returns"] for d in portfolio_ts if "symbol" in d and "returns" in d})

                if returns.empty:
                    return

                fig_tickers_perf = plot_total_return_per_ticker(
                    returns,
                    start_date=_ytd_start(),
                    end_date=None
                )
                figs.append(fig_tickers_perf)
                _maybe_show_fig(fig_tickers_perf)

        except Exception as e:
            if show_report:
                print("Warning: plot_total_return_per_ticker failed:", str(e))    
                
    # -------------------------
    # Block 9: contributions (fix internal benchmark without changing signature)
    # -------------------------
    def _plot_contributions(figs: list, stats_tmp, bm_ret: pd.Series | None, bm_meta: dict):
        # Se benchmark_data non è fornito e benchmark_mode è internal, costruisci prezzi sintetici da bm_ret
        _bench_data = benchmark_data
        _bench_name = benchmark

        if (_bench_data is None) and (bm_ret is not None) and (benchmark_mode == "internal"):
            # price-like series -> evita yfinance
            _bench_data = _returns_to_price(bm_ret, base=100.0)
            _bench_name = str(bm_meta.get("benchmark_name") or benchmark or "Internal BM")

        try:
            fig_assets = build_and_plot_portfolio_contributions(
                pf,
                title=portfolio_title,
                benchmark=_bench_name,
                benchmark_data=_bench_data,
                start_date=stats_tmp.get("Start"),
                end_date=stats_tmp.get("End"),
                show_report=show_report
            )
            if fig_assets is not None:
                figs.append(fig_assets); _maybe_show_fig(fig_assets)
        except Exception:
            pass

        # YTD contributions (come prima)
        try:
            p_start = stats_tmp.get("Start")
            start_year = getattr(p_start, "year", None)
            curr_year = pd.Timestamp.now().year
            if start_year != curr_year:
                fig_assets_ytd = build_and_plot_portfolio_contributions(
                    pf,
                    title=portfolio_title,
                    benchmark=_bench_name,
                    benchmark_data=_bench_data,
                    start_date=_ytd_start(),
                    end_date=stats_tmp.get("End"),
                    show_report=show_report
                )
                if fig_assets_ytd is not None:
                    figs.append(fig_assets_ytd); _maybe_show_fig(fig_assets_ytd)
        except Exception:
            pass

    # -------------------------
    # Block 10: weights
    # -------------------------
    def _plot_weights(figs: list,benchmark: str):
        
        if str(mode).lower() != "standard" or benchmark == None:
            try:
                fig_weights = visualize_portfolio_weights(pf=pf, title=portfolio_title)
                if fig_weights is not None:
                    figs.append(fig_weights); _maybe_show_fig(fig_weights)
            except Exception:
                pass

    # -------------------------
    # RUN: pipeline
    # -------------------------
    summary_pack = _build_summary_and_print()

    header = summary_pack["header"]
    stats_tmp = summary_pack["stats_tmp"]
    stats_df = summary_pack["stats_df"]
    stats_df_raw = summary_pack["stats_df_raw"]
    sintesi_df = summary_pack["sintesi_df"]
    bm_ret = summary_pack["bm_ret"]
    bm_meta = summary_pack["bm_meta"]
    capm = summary_pack["capm"]
    rolling_capm = summary_pack["rolling_capm"]

    figs: list = []

    _plot_vbt_base(figs)
    _plot_ts_portfolio_standard(figs)

    rolling_alpha_df = _plot_rolling_capm_alpha_beta(figs, bm_ret)    
    _print_alpha_diagnostics(capm, rolling_alpha_df)

    portfolios_returns = _plot_vs_benchmark(figs, stats_tmp, bm_ret, bm_meta)
    _plot_periodic(figs, portfolios_returns)

    _plot_rolling_panels(figs, stats_tmp)
    freq_df = _plot_selection_freq(figs, stats_tmp)

    _plot_total_return_per_ticker(figs)
    _plot_contributions(figs, stats_tmp, bm_ret, bm_meta)
    _plot_weights(figs,benchmark)

    out = {
        "figs": figs,
        "header": header,
        "stats_df": stats_df,
        "stats_df_raw": stats_df_raw,
        "sintesi_df": sintesi_df,
        "capm": capm,
        "rolling_capm": rolling_capm,
        "rolling_alpha": rolling_alpha_df,
        "benchmark_meta": bm_meta,
        "benchmark_returns": bm_ret
    }

    if mode == "rotational" and freq_df is not None:
        try:
            out.update({
                "performance_info": "Informazioni sulla selezione dei titoli:",
                "performance_tables": {
                    "Sequenza di selezioni": sel_tickers,
                    "Frequenze di selezione": freq_df.set_index("Ticker")
                }
            })
        except Exception:
            pass

    return out
    
# -------------------------
# Backwards-compatible wrappers matching your original names
# -------------------------

def my_display(data: pd.DataFrame, title: str = ""):
    if title:
        print(title)
    try:
        from IPython.display import display
        display(data)
    except ImportError:
        print(data)
def print_summary(
    portfolio,
    benchmark_portfolio=None,
    sel_tickers=None,
    alpha_analysis=True,
    risk_free_rate=0.02,
):
    return create_portfolio_summary_refactored(
        portfolio,
        benchmark_portfolio=benchmark_portfolio,
        sel_tickers=sel_tickers,
        alpha_analysis=alpha_analysis,
        risk_free_rate=risk_free_rate,
        show=True,
    )
