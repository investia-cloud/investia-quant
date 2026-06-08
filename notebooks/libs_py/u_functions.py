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
def load_ohlcv(symbol: str, start: str = None, end: str = None,
               show_progress: bool = False, auto_adjust: bool = True,
               multi_level_index: bool = False, interval: str = "1d") -> pd.DataFrame:
    """Scarica dati OHLCV da yfinance con indice DatetimeIndex."""
    df = yf.download(symbol, start=start, end=end, multi_level_index=multi_level_index,
                     auto_adjust=auto_adjust, progress=show_progress, interval=interval)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
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
        ("Importo investito (EUR)", init_invested, f"{init_invested:.2f}"),
        ("Valore patrimoniale netto (EUR)", final_value, f"{final_value:.2f}"),
        ("Giorni di trading", period_days, period_days),
        ("Ritorno totale", total_return, f"{total_return:.2%}"),
        ("CAGR", cagr, f"{cagr:.2%}"),
        ("Max Drawdown", max_dd, f"{max_dd:.2%}"),
        ("Volatilita annua", volatility, f"{volatility:.2%}"),
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
def compare_selection_columns(df1: pd.DataFrame, df2: pd.DataFrame,
                               column: str = "Top_Tickers", label_a: str = "Selezione A",
                               label_b: str = "Selezione B",
                               compare_only_common_dates: bool = True,
                               sort_table_by_diff: bool = False,
                               display_table: bool = True):
    if column not in df1.columns: raise ValueError(f"Colonna '{column}' non presente in df1.")
    if column not in df2.columns: raise ValueError(f"Colonna '{column}' non presente in df2.")

    def _to_list(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return []
        if isinstance(x, (list, tuple, set, np.ndarray, pd.Index)):
            return [item for item in list(x) if pd.notna(item)]
        if isinstance(x, str): return [x]
        return []

    common_index = df1.index.intersection(df2.index) if compare_only_common_dates else None
    df_compare = pd.concat([
        (df1.loc[common_index, column] if common_index is not None else df1[column]),
        (df2.loc[common_index, column] if common_index is not None else df2[column])
    ], axis=1)
    df_compare.columns = ["Sel_A", "Sel_B"]
    df_compare["Sel_A"] = df_compare["Sel_A"].apply(_to_list)
    df_compare["Sel_B"] = df_compare["Sel_B"].apply(_to_list)
    set_a = df_compare["Sel_A"].apply(set); set_b = df_compare["Sel_B"].apply(set)

    df_compare["In_Common"]          = [sorted(a & b) for a, b in zip(set_a, set_b)]
    df_compare[f"Solo in {label_a}"] = [sorted(a - b) for a, b in zip(set_a, set_b)]
    df_compare[f"Solo in {label_b}"] = [sorted(b - a) for a, b in zip(set_a, set_b)]
    df_compare[f"N {label_a}"]       = df_compare["Sel_A"].apply(len)
    df_compare[f"N {label_b}"]       = df_compare["Sel_B"].apply(len)
    df_compare["N_Common"]            = df_compare["In_Common"].apply(len)
    df_compare[f"N solo {label_a}"]  = df_compare[f"Solo in {label_a}"].apply(len)
    df_compare[f"N solo {label_b}"]  = df_compare[f"Solo in {label_b}"].apply(len)
    df_compare["Union_Count"]         = [len(a | b) for a, b in zip(set_a, set_b)]
    df_compare["Diff_Count"]          = [len(a ^ b) for a, b in zip(set_a, set_b)]
    df_compare["Jaccard"]             = [len(a & b)/len(a | b) if len(a | b) > 0 else np.nan
                                         for a, b in zip(set_a, set_b)]
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
    for candidate in [secrets_file, _os.path.join('../../', secrets_file)]:
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
def _generate_portfolio_performance_core_refactored(
    pf, portfolio_title: str, pf_b_h=None, mode: str = "standard",
    portfolio_ts: list | None = None, sel_tickers: list | None = None,
    benchmark_mode: str = "internal", benchmark: str = "Benchmark",
    benchmark_data: Optional[pd.Series] = None,
    alpha_analysis: bool = True, risk_free_rate: float = 0.02,
    rolling_window: Optional[int] = 252, show_report: bool = True,
    show_plots: bool = False, vbt_plot_width: Optional[int] = 1000,
    universe: Optional[list] = None,
):
    def _maybe_show(fig):
        if show_plots:
            try: fig.show()
            except Exception: pass

    stats_tmp = pf.stats()
    try: s = pd.to_datetime(stats_tmp.get("Start")).date().isoformat()
    except Exception: s = str(stats_tmp.get("Start", ""))
    try: e = pd.to_datetime(stats_tmp.get("End")).date().isoformat()
    except Exception: e = str(stats_tmp.get("End", ""))
    period_str = f"{s} -> {e}"
    header = f"Statistiche Portfolio {portfolio_title} ({period_str})"
    if show_report: print(header)

    summary = create_portfolio_summary_refactored(
        pf, sel_tickers=sel_tickers, alpha_analysis=alpha_analysis,
        risk_free_rate=risk_free_rate, benchmark_mode=benchmark_mode,
        benchmark_name=benchmark or "Benchmark", benchmark_data=benchmark_data,
        benchmark_portfolio=pf_b_h, annualization=252, rolling_window=rolling_window,
        show=False, return_formatted=True)

    stats_df = summary["stats_df"]; stats_df_raw = summary.get("stats_df_raw", stats_df)
    bm_ret = summary.get("benchmark_returns"); bm_meta = summary.get("benchmark_meta", {})
    capm = summary.get("capm"); rolling_df = summary.get("rolling_capm")

    def _pick(df, key): return df.loc[key, "Valore"] if key in df.index else "n/a"
    sintesi_df = pd.DataFrame([{
        "Importo investito": _pick(stats_df, "Importo investito (EUR)"),
        "Valore finale netto": _pick(stats_df, "Valore patrimoniale netto (EUR)"),
        "CAGR (252)": _pick(stats_df, "CAGR"),
        "Max Drawdown": _pick(stats_df, "Max Drawdown"),
        "Deviazione standard": _pick(stats_df, "Volatilita annua"),
        "Rapporto di Sharpe": _pick(stats_df, "Rapporto di Sharpe"),
        "Operazioni al mese": _pick(stats_df, "Operazioni al mese"),
        "Durata minima in guadagno": _pick(stats_df, "Durata minima in guadagno (giorni)"),
    }])

    figs = []
    try:
        fig_value = pf.plot(width=vbt_plot_width,
                            subplots=['cum_returns', 'drawdowns', 'underwater', 'gross_exposure'])
        figs.append(fig_value); _maybe_show(fig_value)
    except Exception: pass

    return {
        "figs": figs, "header": header, "stats_df": stats_df, "stats_df_raw": stats_df_raw,
        "sintesi_df": sintesi_df, "capm": capm, "rolling_capm": rolling_df,
        "rolling_alpha": None, "benchmark_meta": bm_meta, "benchmark_returns": bm_ret,
    }


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


def generate_rotational_portfolio_performance(pf, portfolio_title: str, sel_tickers=None,
                                               benchmark: str = 'SPY', benchmark_data=None,
                                               plot_start_date=None, plot_end_date=None,
                                               method=None, freq=None, alpha_analysis: bool = True,
                                               show_report: bool = True, show_plots: bool = False,
                                               universe=[]) -> dict:
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
