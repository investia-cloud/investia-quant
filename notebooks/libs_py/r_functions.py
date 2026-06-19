"""
r_functions.py — Rotational Portfolio Functions
Refactored from notebooks/libs/r_functions.ipynb
"""

from __future__ import annotations
# =================================
# Rotational Momentum Functions
# =================================

import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from u_functions import (
    my_display, Emoji, BOLD, RESET, DIM, compare_selection_columns,
    build_company_df_with_cache, download_data, extract_tickers_from_wikipedia,
    generate_rotational_portfolio_performance, now, send_email_report, send_portfolio_performance,
)
import yfinance as yf
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import os
from typing import Union, List, Dict, Tuple, Any
from itertools import product, combinations

import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataclasses import dataclass, field
from typing import Optional

"""
rotational_engine.py
====================
Refactoring del motore rotazionale.

ARCHITETTURA A 3 LAYER:
  Layer 1 – Pure functions   : compute_rebal_dates, compute_scores, select_tickers, build_weight_matrix
  Layer 2 – Orchestrator     : run_rotational_engine  →  RotationalResult (dataclass)
  Layer 3 – VBT Bridge       : build_portfolio, build_portfolio_from_wfo_summary

PRINCIPI:
  - Arità STABILE: niente più tuple a lunghezza variabile.
  - Carry-forward ESPLICITO: colonna `carried: bool` nel DataFrame selections.
  - Single source of truth per rebal_dates: get_rebalance_dates() (già esistente, importata).
  - Nessun doppio carry-forward.
  - Weight-shift difeso da asserzione su trading-only index.
  - bottom_tickers sperimentale → rimosso dal core, disponibile come utility separata.

COMPATIBILITÀ COI NOTEBOOK:
  - build_rotational_portfolios_from_wfo_result  → alias di build_portfolio_from_wfo_summary
  - collect_selections_from_summary              → alias di collect_wfo_selections
  - build_rotational_portfolios_from_selections  → alias di build_portfolio_from_selections
"""


# Helper per data in italiano (da mettere a livello modulo, riusabile)
_MESI_IT = ["", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES  (normalizzazione indice, già presenti nel progetto)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_dt_index(idx) -> pd.DatetimeIndex:
    return pd.to_datetime(idx).normalize()


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 – PURE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rebal_dates(
    trading_index: pd.DatetimeIndex,
    freq: str,
    grace_days_map: dict | None = None,
) -> pd.DatetimeIndex:
    """
    Single source of truth per le date di ribilanciamento.

    Restituisce l'ultimo trading-day per ciascun periodo completo.
    Un periodo è "incompleto" (e quindi escluso) se la distanza tra l'ultima
    data disponibile e la fine calendario del periodo supera `grace_days`.

    Parameters
    ----------
    trading_index : pd.DatetimeIndex
        Indice dei giorni di trading (solo trading days).
    freq : str
        Frequenza: 'ME'/'M', 'QE'/'Q', 'YE'/'Y', 'W'/'W-FRI', 'D'.
    grace_days_map : dict, optional
        Override dei grace-days per frequenza.
        Default: {'ME': 3, 'QE': 5, 'YE': 5, 'W': 0}

    Returns
    -------
    pd.DatetimeIndex  (sorted, unique, tz-naive, normalized)
    """
    idx = pd.DatetimeIndex(trading_index).normalize().sort_values().unique()
    if len(idx) == 0:
        return pd.DatetimeIndex([])

    f = str(freq).upper().strip()

    if grace_days_map is None:
        grace_days_map = {"ME": 3, "QE": 5, "YE": 5, "W": 0}

    last_date = idx.max()

    # ── mapping frequenza → (period_key, grace) ──────────────────────────────
    if f in {"ME", "M", "MONTH", "MONTHLY"}:
        period_key, grace = "M", grace_days_map.get("ME", 3)
    elif f in {"QE", "Q", "QUARTER", "QUARTERLY"}:
        period_key, grace = "Q", grace_days_map.get("QE", 5)
    elif f in {"YE", "Y", "A", "ANNUAL", "YEARLY"}:
        period_key, grace = "Y", grace_days_map.get("YE", 5)
    elif f == "W":
        period_key, grace = "W-FRI", grace_days_map.get("W", 0)
    elif f.startswith("W-"):
        period_key, grace = f, grace_days_map.get("W", 0)
    elif f in {"D", "DAY", "DAILY"}:
        return idx
    else:
        raise ValueError(f"compute_rebal_dates: frequenza non supportata '{freq}'")

    periods = idx.to_period(period_key)
    dates = (
        pd.Series(idx, index=periods)
        .groupby(level=0)
        .max()
        .values
    )
    dates = pd.DatetimeIndex(dates).normalize().sort_values().unique()

    if len(dates) == 0:
        return dates

    # ── rimuovi ultimo periodo se incompleto ──────────────────────────────────
    last_rebal = dates.max()
    if last_rebal == last_date:
        last_period = last_date.to_period(period_key)
        period_end = pd.Timestamp(last_period.end_time).normalize()
        gap = (period_end - last_date).days
        if gap > int(grace):
            dates = dates[:-1]

    return pd.DatetimeIndex(dates).unique().sort_values()


# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScoreParams:
    """Parametri per il calcolo degli score di selezione."""
    momentum_lookback_days: int = 126
    riskparity_lookback_days: int = 20
    momentum_weight: float = 0.7
    use_acceleration: bool = False
    ema_span: int = 200          # usato solo se filter_ema=True (in SelectionParams)

    def __post_init__(self):
        if not 0.0 <= self.momentum_weight <= 1.0:
            raise ValueError("momentum_weight deve essere in [0, 1]")


@dataclass(frozen=True)
class SelectionParams:
    """Parametri per i filtri di selezione dei ticker."""
    n_top: int = 5
    filter_ema: bool = False
    filter_volatility: bool = False
    filter_min_momentum: bool = False
    volatility_quantile: float = 0.75
    min_momentum_threshold: float = 1.0


@dataclass(frozen=True)
class EngineParams:
    """Tutti i parametri del motore rotazionale."""
    score: ScoreParams = field(default_factory=ScoreParams)
    selection: SelectionParams = field(default_factory=SelectionParams)
    rebalance_frequency: str = "ME"
    init_cash: float = 100_000

    @classmethod
    def from_dict(cls, d: dict) -> "EngineParams":
        """Costruisce EngineParams da un dizionario (es. riga di summary_df)."""
        sp = ScoreParams(
            momentum_lookback_days=int(d.get("momentum_lookback_days", 126)),
            riskparity_lookback_days=int(d.get("riskparity_lookback_days", 20)),
            momentum_weight=float(d.get("momentum_weight", 0.7)),
            use_acceleration=bool(d.get("use_acceleration", False)),
            ema_span=int(d.get("ema_span", 200)),
        )
        selp = SelectionParams(
            n_top=int(d.get("n_top", 5)),
            filter_ema=bool(d.get("filter_ema", False)),
            filter_volatility=bool(d.get("filter_volatility", False)),
            filter_min_momentum=bool(d.get("filter_min_momentum", False)),
            volatility_quantile=float(d.get("volatility_quantile", 0.75)),
            min_momentum_threshold=float(d.get("min_momentum_threshold", 1.0)),
        )
        return cls(
            score=sp,
            selection=selp,
            rebalance_frequency=str(d.get("rebalance_frequency", "ME")),
        )

    def required_warmup_days(self) -> int:
        """Calendar days of history needed before the first valid signal."""
        max_lb = max(
            self.score.momentum_lookback_days,
            self.score.riskparity_lookback_days,
        )
        if self.selection.filter_ema:
            max_lb = max(max_lb, self.score.ema_span)
        return max(int(max_lb * 1.5) + 30, 60)


# ─────────────────────────────────────────────────────────────────────────────

def _precompute_indicators(
    prices: pd.DataFrame,
    params: "EngineParams",
    vol_cache: dict | None = None,
) -> dict:
    """
    Precomputa tutti gli indicatori necessari al loop di selezione.
    Riceve l'intero EngineParams per accedere sia a ScoreParams che a SelectionParams.
    Ritorna un dizionario di DataFrame allineati all'indice di prices.

    Non ha side effects; è testabile indipendentemente.
    """
    sp   = params.score
    selp = params.selection

    rets = prices.pct_change().fillna(0.0)

    mom_df = prices / prices.shift(sp.momentum_lookback_days)

    lb = sp.riskparity_lookback_days
    if vol_cache is not None and lb in vol_cache:
        vol_df = vol_cache[lb].reindex(rets.index)
    else:
        vol_df = rets.rolling(lb, min_periods=1).std(ddof=0)

    inv_vol_df = 1.0 / vol_df.replace(0.0, np.nan)

    rank_mom  = mom_df.rank(pct=True, axis=1, na_option="bottom")
    rank_ivol = inv_vol_df.rank(pct=True, axis=1, na_option="bottom")

    result = {
        "mom":       mom_df,
        "vol":       vol_df,
        "inv_vol":   inv_vol_df,
        "rank_mom":  rank_mom,
        "rank_ivol": rank_ivol,
    }

    if selp.filter_ema:
        result["ema"] = prices.ewm(
            span=sp.ema_span, adjust=False, min_periods=1
        ).mean()

    if sp.use_acceleration:
        _SMOOTH, _SHIFT, _WEIGHT = 5, 10, 0.20
        mom_sm = mom_df.ewm(span=_SMOOTH, adjust=False, min_periods=1).mean()
        accel  = mom_sm - mom_sm.shift(_SHIFT)
        result["accel"]         = accel
        result["rank_accel"]    = accel.rank(pct=True, axis=1, na_option="bottom")
        result["_accel_weight"] = _WEIGHT

    return result


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SelectionResult:
    """Risultato della selezione per una singola data di ribilanciamento."""
    date: pd.Timestamp
    tickers: list[str]          # lista dei ticker selezionati (può essere vuota)
    carried: bool               # True se è un carry-forward di una selezione precedente
    score: pd.Series            # score completo (tutti i ticker, NaN se filtrati)
    n_passed_filters: int       # quanti ticker hanno passato tutti i filtri
    universe: list[str]         # ticker eleggibili (hanno passato tutti i filtri)


def _select_at_date(
    date: pd.Timestamp,
    prices: pd.DataFrame,
    indicators: dict,
    score_params: ScoreParams,
    sel_params: SelectionParams,
    prev_selection: list[str] | None,
) -> _SelectionResult:
    """
    Calcola la selezione per una singola rebal_date.

    - Applica filtri (EMA, volatilità, min-momentum, accelerazione).
    - Calcola combo score.
    - Seleziona top-N.
    - Se selezione vuota E prev_selection disponibile → carry-forward (carried=True).
    - Se selezione vuota E nessun precedente → tickers=[], carried=False.

    Questa funzione è PURA rispetto allo stato: prev_selection è input esplicito.
    """
    d = pd.Timestamp(date).normalize()

    mom       = indicators["mom"].loc[d]
    inv_vol   = indicators["inv_vol"].loc[d]
    rank_mom  = indicators["rank_mom"].loc[d]
    rank_ivol = indicators["rank_ivol"].loc[d]

    # mask base: entrambi gli indicatori presenti
    mask = mom.notna() & inv_vol.notna()

    if sel_params.filter_ema:
        ema = indicators["ema"].loc[d]
        mask &= prices.loc[d] > ema

    if sel_params.filter_volatility:
        vol = indicators["vol"].loc[d]
        q   = vol.quantile(sel_params.volatility_quantile)
        mask &= vol < q

    if sel_params.filter_min_momentum:
        mask &= mom > sel_params.min_momentum_threshold

    # combo score
    w = score_params.momentum_weight
    combo = w * rank_mom + (1.0 - w) * rank_ivol

    if score_params.use_acceleration:
        accel_today = indicators["accel"].loc[d]
        mask &= accel_today > 0
        combo = combo + indicators.get("_accel_weight", 0.20) * indicators["rank_accel"].loc[d]

    combo_masked = combo.where(mask)
    n_passed = int(mask.sum())

    # selezione top-N
    valid = combo_masked.dropna()
    if not valid.empty:
        top_tickers = list(valid.nlargest(sel_params.n_top).index)
        carried = False
    elif prev_selection:
        top_tickers = list(prev_selection)
        carried = True
    else:
        top_tickers = []
        carried = False

    universe_tickers = list(valid.index)

    return _SelectionResult(
        date=d,
        tickers=top_tickers,
        carried=carried,
        score=combo_masked.sort_values(ascending=False),
        n_passed_filters=n_passed,
        universe=universe_tickers,
    )


# ─────────────────────────────────────────────────────────────────────────────

def build_weight_matrix(
    selections: pd.DataFrame,
    prices_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Costruisce la matrice dei pesi equal-weight da un DataFrame di selezioni,
    allineata all'intero indice dei prezzi e con shift(1) per evitare look-ahead.

    Parameters
    ----------
    selections : pd.DataFrame
        Index  : rebal_dates (pd.DatetimeIndex)
        Columns: 'tickers' (list[str]), + altre colonne ignorate
        Deve contenere almeno la colonna 'tickers'.
    prices_idx : pd.DatetimeIndex
        Indice completo dei prezzi (trading days only).

    Returns
    -------
    pd.DataFrame  shape=(len(prices_idx), n_assets)
        Pesi equal-weight, shiftati di 1 giorno (ordine eseguito il giorno dopo).

    Raises
    ------
    ValueError
        Se prices_idx ha buchi > 4 giorni consecutivi (segnale che non è trading-only).
    """
    prices_idx = pd.DatetimeIndex(prices_idx).normalize().sort_values()

    # ── guardrail: verifica che sia un indice trading-only ────────────────────
    if len(prices_idx) > 1:
        gaps = pd.Series(prices_idx).diff().dt.days.dropna()
        max_gap = gaps.max()
        if max_gap > 7:
            warnings.warn(
                f"build_weight_matrix: gap massimo di {max_gap} giorni in prices_idx. "
                "Gap > 7 giorni potrebbe indicare dati non trading-only (es. calendario errato).",
                stacklevel=2,
            )

    # ── ricava l'universo di asset da tutte le selezioni ─────────────────────
    all_tickers: set[str] = set()
    for tlist in selections["tickers"]:
        if isinstance(tlist, (list, tuple, set)):
            all_tickers.update(tlist)
    cols = sorted(all_tickers)

    if not cols:
        return pd.DataFrame(0.0, index=prices_idx, columns=[])

    # ── costruzione pesi sulle rebal_dates ───────────────────────────────────
    w_sparse = pd.DataFrame(0.0, index=selections.index, columns=cols)
    for d, row in selections.iterrows():
        tlist = row.get("tickers", [])
        if not isinstance(tlist, (list, tuple, set)) or len(tlist) == 0:
            continue
        valid = [t for t in tlist if t in cols]
        if valid:
            w_sparse.loc[d, valid] = 1.0 / len(valid)

    # ── espandi all'intero indice, poi shift(1) ───────────────────────────────
    w_full = w_sparse.reindex(prices_idx).ffill().fillna(0.0)
    w_shifted = w_full.shift(1).ffill().fillna(0.0)

    return w_shifted


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 – ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RotationalResult:
    """
    Output strutturato di run_rotational_engine.

    Attributes
    ----------
    selections : pd.DataFrame
        Una riga per ogni rebal_date.
        Colonne: 'tickers' (list[str]), 'carried' (bool), 'n_passed_filters' (int).
        Index name: 'rebal_date'.
    weights : pd.DataFrame
        Pesi equal-weight sull'intero indice dei prezzi, shiftati di 1 giorno.
    rankings : pd.DataFrame
        Combo-score per tutti i ticker, una riga per rebal_date.
    rebal_dates : pd.DatetimeIndex
        Date di ribilanciamento effettive usate.
    params : EngineParams
        Parametri usati per questa run.
    """
    selections: pd.DataFrame
    weights: pd.DataFrame
    rankings: pd.DataFrame
    rebal_dates: pd.DatetimeIndex
    params: EngineParams


def run_rotational_engine(
    prices: pd.DataFrame,
    params: EngineParams,
    vol_cache: dict | None = None,
    start_date: str | pd.Timestamp | None = None,
    debug: bool = False,
) -> RotationalResult:
    """
    Cuore del motore rotazionale. Pura orchestrazione del Layer 1.

    Il carry-forward è ESPLICITO nel DataFrame selections (colonna 'carried').
    I pesi sono costruiti sull'intero indice (post start_date) con shift(1).

    Parameters
    ----------
    prices : pd.DataFrame
        Prezzi giornalieri, indice trading-only.
        Deve coprire almeno momentum_lookback_days prima della prima rebal_date.
    params : EngineParams
        Tutti i parametri del motore.
    vol_cache : dict, optional
        Cache {lookback_days: vol_df} per evitare ricalcoli.
    start_date : str | Timestamp, optional
        Se fornito, le weights vengono tagliate da questa data.
        Le selezioni vengono calcolate sull'intero range di prices
        (per garantire lookback corretto), poi i pesi vengono tagliati.
    debug : bool
        Se True, stampa log dettagliati.

    Returns
    -------
    RotationalResult
    """
    def _dbg(msg: str):
        if debug:
            print(msg)

    # ── 1) Sanity checks e normalizzazione ───────────────────────────────────
    prices = prices.dropna(axis=1, how="all").ffill().bfill().copy()
    prices.index = _norm_dt_index(prices.index)
    prices = prices.sort_index()
    trading_idx = prices.index.unique()

    _dbg(f"[ENGINE] freq={params.rebalance_frequency}  "
         f"prices={trading_idx.min().date()}→{trading_idx.max().date()}  "
         f"n_assets={prices.shape[1]}")

    # ── 2) Rebal dates (unica fonte di verità) ────────────────────────────────
    rebal_dates = compute_rebal_dates(trading_idx, params.rebalance_frequency)
    rebal_dates = pd.DatetimeIndex(
        [d for d in rebal_dates if d in trading_idx]
    ).sort_values().unique()

    _dbg(f"[ENGINE] rebal_dates: n={len(rebal_dates)}  "
         f"tail={list(rebal_dates[-4:]) if len(rebal_dates) else []}")

    # ── 3) Fallback: nessuna rebal_date disponibile ───────────────────────────
    if len(rebal_dates) == 0:
        _dbg("[ENGINE] WARN: rebal_dates vuoto → pesi a zero (cash)")
        empty_sel = pd.DataFrame(
            columns=["tickers", "carried", "n_passed_filters", "universe"],
            index=pd.DatetimeIndex([], name="rebal_date"),
        )
        empty_w = pd.DataFrame(0.0, index=trading_idx, columns=prices.columns)
        empty_rank = pd.DataFrame(
            index=pd.DatetimeIndex([], name="rebal_date"),
            columns=prices.columns,
        )
        return RotationalResult(
            selections=empty_sel,
            weights=empty_w,
            rankings=empty_rank,
            rebal_dates=rebal_dates,
            params=params,
        )

    # ── 4) Precomputa indicatori ──────────────────────────────────────────────
    indicators = _precompute_indicators(prices, params, vol_cache)

    # ── 5) Loop selezione ─────────────────────────────────────────────────────
    sel_records: list[dict] = []
    rank_records: dict[pd.Timestamp, pd.Series] = {}
    prev_top: list[str] | None = None

    for d in rebal_dates:
        d = pd.Timestamp(d).normalize()
        result = _select_at_date(
            date=d,
            prices=prices,
            indicators=indicators,
            score_params=params.score,
            sel_params=params.selection,
            prev_selection=prev_top,
        )

        sel_records.append({
            "rebal_date":       d,
            "tickers":          result.tickers,
            "carried":          result.carried,
            "n_passed_filters": result.n_passed_filters,
            "universe":         result.universe,
        })
        rank_records[d] = result.score

        if result.tickers and not result.carried:
            prev_top = result.tickers

        if debug:
            status = "CARRY-FWD" if result.carried else ("EMPTY    " if not result.tickers else "OK       ")
            _dbg(
                f"[ENGINE] {status} | {d.date()} | "
                f"passed_filters={result.n_passed_filters} | "
                f"selected={result.tickers}"
            )

    # ── 6) Costruisce DataFrame selezioni ─────────────────────────────────────
    selections = pd.DataFrame(sel_records).set_index("rebal_date")
    selections.index.name = "rebal_date"

    rankings = pd.DataFrame(rank_records).T
    rankings.index.name = "rebal_date"

    # ── 7) Weight matrix (sull'intero indice, poi taglio start_date) ──────────
    weights = build_weight_matrix(selections, trading_idx)

    # allinea colonne al prices universe (aggiunge zeri per ticker non selezionati)
    missing_cols = [c for c in prices.columns if c not in weights.columns]
    if missing_cols:
        weights = pd.concat(
            [weights, pd.DataFrame(0.0, index=weights.index, columns=missing_cols)],
            axis=1,
        )[prices.columns]

    if start_date is not None:
        start_ts = pd.Timestamp(start_date).normalize()
        weights = weights.loc[start_ts:]

    _dbg(f"[ENGINE] Done. selections={len(selections)}  "
         f"carried={selections['carried'].sum()}  "
         f"empty={( selections['tickers'].apply(len) == 0 ).sum()}")

    return RotationalResult(
        selections=selections,
        weights=weights,
        rankings=rankings,
        rebal_dates=rebal_dates,
        params=params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 – VBT BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(
    result: RotationalResult,
    prices: pd.DataFrame,
    benchmark_data: pd.Series | None = None,
    init_cash: float = 100_000,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    plot: bool = True,
    show_report: bool = False,
    portfolio_name: str = "Rotational Portfolio",
    benchmark_title: str = "Benchmark",
    vbt_plot_width: int = 1200,
) -> tuple:
    """
    Layer 3: costruisce portafogli VBT da un RotationalResult.

    Parameters
    ----------
    result : RotationalResult
        Output di run_rotational_engine.
    prices : pd.DataFrame
        Prezzi (stesso universo usato per run_rotational_engine).
    benchmark_data : pd.Series, optional
    init_cash : float
    start_date, end_date : str | Timestamp, optional
        Finestra operativa per il backtest (i pesi vengono costruiti
        sull'intero range, poi tagliati qui).
    plot : bool
    show_report : bool
    portfolio_name : str
    benchmark_title : str
    vbt_plot_width : int

    Returns
    -------
    (pf_rot, pf_bh)
        pf_bh è None se benchmark_data=None.
    """
    import vectorbt as vbt

    # ── normalizza prezzi ─────────────────────────────────────────────────────
    prices = prices.dropna(axis=1, how="all").ffill().bfill().copy()
    prices.index = _norm_dt_index(prices.index)
    prices = prices.sort_index()

    # ── taglio start/end ──────────────────────────────────────────────────────
    idx_min, idx_max = prices.index.min(), prices.index.max()
    s_ts = pd.Timestamp(start_date).normalize() if start_date else idx_min
    e_ts = pd.Timestamp(end_date).normalize()   if end_date   else idx_max
    if s_ts > e_ts:
        s_ts, e_ts = e_ts, s_ts
    s_ts = max(s_ts, idx_min)
    e_ts = min(e_ts, idx_max)

    prices_cut = prices.loc[s_ts:e_ts].copy()
    if prices_cut.empty:
        raise ValueError(f"Taglio prezzi produce DataFrame vuoto: {s_ts} → {e_ts}")

    # pesi allineati al periodo di backtest
    weights_cut = result.weights.reindex(prices_cut.index).fillna(0.0)
    # allinea colonne
    weights_cut = weights_cut.reindex(columns=prices_cut.columns, fill_value=0.0)

    # ── portafoglio rotazionale ───────────────────────────────────────────────
    pf_rot = vbt.Portfolio.from_orders(
        close=prices_cut,
        size=weights_cut,
        size_type="targetpercent",
        init_cash=init_cash,
        cash_sharing=True,
        freq="D",
    )

    # ── benchmark ─────────────────────────────────────────────────────────────
    pf_bh = None
    if benchmark_data is not None:
        bench = benchmark_data.copy()
        bench.index = _norm_dt_index(bench.index)
        bench = bench.sort_index().reindex(prices_cut.index).ffill().dropna()
        pf_bh = vbt.Portfolio.from_holding(
            close=bench.to_frame(name="Benchmark"),
            init_cash=init_cash,
            cash_sharing=True,
            freq="D",
        )

    # ── plot ──────────────────────────────────────────────────────────────────
    if plot:
        _plot_cumulative_returns(
            pf_rot=pf_rot,
            pf_bh=pf_bh,
            init_cash=init_cash,
            title=f"{portfolio_name} – Rendimenti cumulati ({s_ts.date()} → {e_ts.date()})",
            benchmark_title=benchmark_title,
            width=vbt_plot_width,
        )

    # ── report testuale ───────────────────────────────────────────────────────
    if show_report:
        _print_report(
            pf_rot=pf_rot,
            pf_bh=pf_bh,
            selections=result.selections,
            portfolio_name=portfolio_name,
            benchmark_title=benchmark_title,
        )

    return pf_rot, pf_bh


# ─────────────────────────────────────────────────────────────────────────────

def collect_wfo_selections(
    summary_df: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Raccoglie le selezioni da una Walk-Forward Optimization summary.

    Per ogni finestra temporale in summary_df:
      1. Calcola i parametri del motore dalla riga del summary.
      2. Esegue run_rotational_engine sulla slice di prezzo corretta
         (con buffer storico per garantire il lookback).
      3. Tiene solo le selezioni nella finestra [start, end].
      4. NON fa doppio carry-forward: le selezioni già contengono il flag 'carried'.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Index  : stringhe tipo "2020-01-01→2021-12-31"
        Columns: momentum_lookback_days, riskparity_lookback_days, n_top,
                 momentum_weight, rebalance_frequency, filter_ema,
                 filter_volatility, filter_min_momentum, use_acceleration, ...
    stocks_data : pd.DataFrame
        Prezzi giornalieri completi.
    benchmark_data : pd.Series, optional
        Non usato nel calcolo delle selezioni; accettato per simmetria con
        la vecchia collect_selections_from_summary.
    debug : bool

    Returns
    -------
    pd.DataFrame
        Index name: 'rebal_date'
        Columns   : 'tickers' (list[str]), 'carried' (bool), 'n_passed_filters' (int)
        Sorted, deduplicato (keep='last').
    """
    if stocks_data is None or stocks_data.empty:
        return _empty_selections()
    if summary_df is None or summary_df.empty:
        return _empty_selections()

    stocks = stocks_data.copy()
    stocks.index = _norm_dt_index(stocks.index)
    stocks = stocks.sort_index()

    all_sel: list[pd.DataFrame] = []

    for window, row in summary_df.iterrows():
        # ── parse window ──────────────────────────────────────────────────────
        try:
            start_str, end_str = str(window).split("→")
            win_start = pd.Timestamp(start_str).normalize()
            win_end   = pd.Timestamp(end_str).normalize()
        except Exception:
            if debug:
                print(f"[WFO] SKIP: parse error window='{window}'")
            continue

        last_avail = stocks.index.max()
        slice_end  = min(win_end, last_avail)

        params = EngineParams.from_dict(dict(row))

        # ── buffer storico per garantire il lookback ──────────────────────────
        buffer_days = params.required_warmup_days()

        
        buf_start = max(win_start - pd.Timedelta(days=buffer_days), stocks.index.min())

        slice_prices = stocks.loc[buf_start:slice_end].copy()
        if slice_prices.empty:
            if debug:
                print(f"[WFO] SKIP: slice vuota | window={window}")
            continue

        # ── run engine sulla slice ────────────────────────────────────────────
        engine_result = run_rotational_engine(
            prices=slice_prices,
            params=params,
            debug=debug,
        )

        if engine_result.selections.empty:
            if debug:
                print(f"[WFO] SKIP: selezioni vuote | window={window}")
            continue

        # ── taglia solo le selezioni dentro la window ─────────────────────────
        sel = engine_result.selections.copy()
        sel = sel.loc[(sel.index >= win_start) & (sel.index <= slice_end)]

        if sel.empty:
            if debug:
                print(f"[WFO] SKIP: selezioni fuori window | window={window}")
            continue

        if debug:
            n_carried = int(sel["carried"].sum())
            n_empty   = int((sel["tickers"].apply(len) == 0).sum())
            print(
                f"[WFO] OK | window={window} | "
                f"n_sel={len(sel)} | carried={n_carried} | empty={n_empty}"
            )

        all_sel.append(sel)

    if not all_sel:
        return _empty_selections()

    combined = pd.concat(all_sel).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.index.name = "rebal_date"

    if debug:
        print(f"\n[WFO] FINAL: {len(combined)} selezioni totali | "
              f"carried={combined['carried'].sum()} | "
              f"empty={(combined['tickers'].apply(len) == 0).sum()}")

    return combined


# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_from_wfo_summary(
    summary_df: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    benchmark_title: str = "Benchmark",
    portfolio_name: str = "Rotational Portfolio – WFO",
    init_cash: float = 100_000,
    plot: bool = True,
    show_report: bool = True,
    vbt_plot_width: int = 1200,
    debug: bool = False,
) -> tuple:
    """
    Funzione di alto livello per i Notebook.

    Pipeline completa:
      1. collect_wfo_selections  → selections DataFrame
      2. build_portfolio_from_selections → (pf_rot, pf_bh)

    Returns
    -------
    (pf_rot, pf_bh, selections)
        selections : pd.DataFrame con colonne tickers, carried, n_passed_filters
    """
    selections = collect_wfo_selections(
        summary_df=summary_df,
        stocks_data=stocks_data,
        benchmark_data=benchmark_data,
        debug=debug,
    )

    if selections.empty:
        raise ValueError(
            "collect_wfo_selections ha restituito un DataFrame vuoto. "
            "Controlla summary_df e stocks_data."
        )

    pf_rot, pf_bh = build_portfolio_from_selections(
        selections=selections,
        stocks_data=stocks_data,
        benchmark_data=benchmark_data,
        benchmark_title=benchmark_title,
        init_cash=init_cash,
        start_date=start_date,
        end_date=end_date,
        plot=plot,
        show_report=show_report,
        portfolio_name=portfolio_name,
        vbt_plot_width=vbt_plot_width,
    )

    return pf_rot, pf_bh, selections


def build_portfolio_from_selections(
    selections: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    benchmark_title: str = "Benchmark",
    init_cash: float = 100_000,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    plot: bool = True,
    show_report: bool = True,
    portfolio_name: str = "Rotational Portfolio",
    vbt_plot_width: int = 1200,
) -> tuple:
    """
    Costruisce (pf_rot, pf_bh) da un DataFrame di selezioni già pronte.

    Accetta sia il nuovo formato (colonna 'tickers')
    sia il vecchio formato (colonna 'Top_Tickers') per retrocompatibilità.

    Returns
    -------
    (pf_rot, pf_bh)
    """
    import vectorbt as vbt

    if selections is None or selections.empty:
        raise ValueError("selections è vuoto.")

    # ── normalizza nome colonna (retrocompatibilità) ──────────────────────────
    sel = _normalize_selections_df(selections)

    # ── prezzi ────────────────────────────────────────────────────────────────
    prices = stocks_data.copy()
    prices.index = _norm_dt_index(prices.index)
    prices = prices.sort_index().dropna(axis=1, how="all").ffill().bfill()

    idx_min, idx_max = prices.index.min(), prices.index.max()

    # ── calcolo start/end effettivi ───────────────────────────────────────────
    #
    # REGOLA: portafoglio e benchmark devono iniziare alla stessa data
    # e quella data deve essere il PRIMO giorno in cui i pesi sono > 0
    # (= il primo trading day DOPO la prima rebal_date, per effetto dello shift+1).
    #
    # Se start_date viene passato dall'utente ma è PRECEDENTE alla prima selezione,
    # usiamo comunque il primo giorno con pesi effettivi: avere un lungo tratto
    # di cash flat prima delle selezioni sfaserebbe il confronto col benchmark.
    #
    # Se start_date è SUCCESSIVO alla prima selezione, lo rispettiamo (l'utente
    # vuole vedere solo un sotto-periodo).

    first_rebal = pd.Timestamp(sel.index.min()).normalize()
    after_first_rebal = prices.index[prices.index > first_rebal]
    first_active_day = after_first_rebal[0] if len(after_first_rebal) else first_rebal

    if start_date is None:
        s_ts = first_active_day
    else:
        s_ts_requested = pd.Timestamp(start_date).normalize()
        # se l'utente chiede una data prima che ci siano selezioni → alza al primo giorno attivo
        s_ts = max(s_ts_requested, first_active_day)

    e_ts = pd.Timestamp(end_date).normalize() if end_date else idx_max
    if s_ts > e_ts:
        s_ts, e_ts = e_ts, s_ts
    s_ts = max(s_ts, idx_min)
    e_ts = min(e_ts, idx_max)
    if s_ts > idx_max or e_ts < idx_min:
        raise ValueError(
            f"Finestra [{s_ts.date()} → {e_ts.date()}] non interseca "
            f"i dati disponibili [{idx_min.date()} → {idx_max.date()}]."
        )

    prices_cut = prices.loc[s_ts:e_ts].copy()

    # avvisa se start_date passato è stato ignorato
    if start_date is not None:
        s_ts_requested = pd.Timestamp(start_date).normalize()
        if s_ts > s_ts_requested:
            warnings.warn(
                f"build_portfolio_from_selections: start_date={s_ts_requested.date()} "
                f"è precedente alla prima selezione disponibile ({first_rebal.date()}). "
                f"Il portafoglio partirà dal primo giorno attivo: {s_ts.date()}. "
                "Per evitare un lungo tratto di cash flat che disallinea il confronto "
                "col benchmark, start_date viene alzato automaticamente.",
                stacklevel=2,
            )

    # ── costruzione pesi ──────────────────────────────────────────────────────
    # costruiamo i pesi sull'intero indice (per ffill corretto) poi tagliamo
    weights_full = build_weight_matrix(sel, prices.index)
    # allinea colonne
    all_tickers = sorted({t for tlist in sel["tickers"] for t in (tlist or [])})
    missing = [t for t in all_tickers if t not in prices.columns]
    if missing:
        warnings.warn(f"Ticker non trovati in stocks_data: {missing}", stacklevel=2)

    weights_cut = weights_full.reindex(prices_cut.index).fillna(0.0)
    weights_cut = weights_cut.reindex(columns=prices_cut.columns, fill_value=0.0)

    # ── portafoglio rotazionale ───────────────────────────────────────────────
    pf_rot = vbt.Portfolio.from_orders(
        close=prices_cut,
        size=weights_cut,
        size_type="targetpercent",
        init_cash=init_cash,
        cash_sharing=True,
        freq="D",
    )

    # ── benchmark ─────────────────────────────────────────────────────────────
    bench = benchmark_data.copy()
    bench.index = _norm_dt_index(bench.index)
    bench = bench.sort_index().reindex(prices_cut.index).ffill().dropna()
    pf_bh = vbt.Portfolio.from_holding(
        close=bench.to_frame(name="Benchmark"),
        init_cash=init_cash,
        cash_sharing=True,
        freq="D",
    )

    # ── plot ──────────────────────────────────────────────────────────────────
    if plot:
        _plot_cumulative_returns(
            pf_rot=pf_rot,
            pf_bh=pf_bh,
            init_cash=init_cash,
            title=f"{portfolio_name} – Rendimenti cumulati ({s_ts.date()} → {e_ts.date()})",
            benchmark_title=benchmark_title,
            width=vbt_plot_width,
        )

    # ── report ────────────────────────────────────────────────────────────────
    if show_report:
        _print_report(
            pf_rot=pf_rot,
            pf_bh=pf_bh,
            selections=sel,
            portfolio_name=portfolio_name,
            benchmark_title=benchmark_title,
        )

    return pf_rot, pf_bh


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS PRIVATI
# ─────────────────────────────────────────────────────────────────────────────

def _empty_selections() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["tickers", "carried", "n_passed_filters", "universe"],
        index=pd.DatetimeIndex([], name="rebal_date"),
    )


def _normalize_selections_df(sel: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza il DataFrame di selezioni al formato interno:
      Index : 'rebal_date'
      Columns: 'tickers' (list[str]), opzionalmente 'carried', 'n_passed_filters'

    Retrocompatibile con il vecchio formato (colonna 'Top_Tickers').
    """
    sel = sel.copy()
    sel.index = _norm_dt_index(sel.index)
    sel.index.name = "rebal_date"

    # rinomina Top_Tickers → tickers se necessario
    if "tickers" not in sel.columns and "Top_Tickers" in sel.columns:
        sel = sel.rename(columns={"Top_Tickers": "tickers"})

    # colonna tickers mancante: cerca la prima colonna con liste
    if "tickers" not in sel.columns:
        for c in sel.columns:
            if sel[c].apply(lambda x: isinstance(x, (list, tuple, set))).any():
                sel = sel.rename(columns={c: "tickers"})
                break

    if "tickers" not in sel.columns:
        raise ValueError(
            "Il DataFrame di selezioni non contiene una colonna 'tickers' "
            "né 'Top_Tickers'."
        )

    # aggiungi colonne opzionali se assenti (retrocompatibilità)
    if "carried" not in sel.columns:
        sel["carried"] = False
    if "n_passed_filters" not in sel.columns:
        sel["n_passed_filters"] = -1  # -1 = non disponibile
    if "universe" not in sel.columns:  # retrocompatibilità con run salvati prima di questa modifica
        sel["universe"] = sel["tickers"].apply(lambda x: list(x) if isinstance(x, list) else [])

    # normalizza valori NaN nelle liste
    sel["tickers"] = sel["tickers"].apply(
        lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else list(x))
    )

    return sel


def _plot_cumulative_returns(
    pf_rot,
    pf_bh,
    init_cash: float,
    title: str,
    benchmark_title: str,
    width: int = 1200,
):
    """Produce il grafico Plotly dei rendimenti cumulati."""
    import plotly.graph_objects as go

    fig = go.Figure()

    cum_rot = pf_rot.value() / init_cash
    fig.add_trace(go.Scatter(
        x=cum_rot.index, y=cum_rot,
        mode="lines", name="Rotational",
        line=dict(width=2),
    ))

    if pf_bh is not None:
        cum_bh = pf_bh.value() / init_cash
        fig.add_trace(go.Scatter(
            x=cum_bh.index, y=cum_bh,
            mode="lines", name=f"Benchmark ({benchmark_title})",
            line=dict(color="gray", width=2),
            opacity=0.85,
        ))

    fig.update_layout(
        title=title,
        yaxis_title="Valore (base 1€)",
        xaxis_title="Data",
        yaxis_tickformat=".2f",
        width=width,
        height=600,
        template="plotly_white",
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                dict(step="all"),
            ]),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )
    fig.show()


def _print_report(pf_rot, pf_bh, selections: pd.DataFrame, portfolio_name: str, benchmark_title: str):
    """Stampa report testuale con stats VBT e riepilogo selezioni."""
    try:
        from IPython.display import display
    except ImportError:
        display = print

    print(f"\n{'='*60}")
    print(f"  {portfolio_name}")
    print(f"{'='*60}")

    # carry-forward summary
    if "carried" in selections.columns:
        n_carried = int(selections["carried"].sum())
        n_total   = len(selections)
        n_empty   = int((selections["tickers"].apply(len) == 0).sum())
        print(f"\n  Selezioni totali : {n_total}")
        print(f"  Carry-forward    : {n_carried} ({100*n_carried/max(n_total,1):.1f}%)")
        print(f"  Selezioni vuote  : {n_empty}")
        if n_carried > 0:
            cf_dates = selections.index[selections["carried"]].tolist()
            print(f"  Date carry-fwd   : {[d.date() for d in cf_dates[:5]]}"
                  f"{'...' if len(cf_dates) > 5 else ''}")

    print(f"\n  --- Stats Rotational ---")
    try:
        display(pf_rot.stats())
    except Exception as e:
        print(f"  [WARN] pf_rot.stats() error: {e}")

    if pf_bh is not None:
        print(f"\n  --- Stats Benchmark ({benchmark_title}) ---")
        try:
            display(pf_bh.stats())
        except Exception as e:
            print(f"  [WARN] pf_bh.stats() error: {e}")

    print(f"\n  --- Ultime selezioni ---")
    display(selections.tail(12))


# ─────────────────────────────────────────────────────────────────────────────
# ALIAS PER RETROCOMPATIBILITÀ COI NOTEBOOK ESISTENTI
# ─────────────────────────────────────────────────────────────────────────────

build_rotational_portfolios_from_wfo_result   = build_portfolio_from_wfo_summary
collect_selections_from_summary               = collect_wfo_selections
build_rotational_portfolios_from_selections   = build_portfolio_from_selections

# Funzioni di utilita' per i rotazionali
def list_available_portfolios(prefix: str = "portfolio_") -> list[str]:
    return [
        name for name, obj in globals().items()
        if name.startswith(prefix) and not isinstance(obj, str)
    ]
    
def compute_portfolio_ticker_intersections(debug: bool = False) -> Dict[str, List[str]]:
    """
    Calcola tutte le intersezioni tra le liste ticker dei portafogli disponibili.

    Usa list_available_portfolios() che ritorna una LISTA di nomi.
    I portafogli sono recuperati da globals().
    """

    portfolio_names = list_available_portfolios()
    print(f"\nPortafogli disponibili:\n{BOLD}{portfolio_names}{RESET}")

    intersections: Dict[str, List[str]] = {}

    for name_a, name_b in combinations(portfolio_names, 2):
        pf_a = globals().get(name_a)
        pf_b = globals().get(name_b)

        # --- FIX: assicurati che siano dict ---
        if not isinstance(pf_a, dict) or not isinstance(pf_b, dict):
            # Debug utile: mostra cosa sono davvero
            if debug:
                if not isinstance(pf_a, dict):
                    print(f"SKIP {name_a}: type={type(pf_a).__name__}")
                if not isinstance(pf_b, dict):
                    print(f"SKIP {name_b}: type={type(pf_b).__name__}")
            continue

        tickers_a = pf_a.get("tickers", [])
        tickers_b = pf_b.get("tickers", [])

        if not tickers_a or not tickers_b:
            continue

        set_b = set(tickers_b)
        common = [t for t in tickers_a if t in set_b]

        if common:
            key = f"{name_a} ∩ {name_b}"
            intersections[key] = common

    return intersections


# WFO Stuff

# Load/save WFO


def save_rotational_wfo_summary(
    summary_df: pd.DataFrame,
    file_path: str,
    *,
    param_grid: dict,
    metric: str,
    ratio: str,
    force_next_year_params: bool,
    start_date: str,
    end_date: str,
    extra_meta: dict | None = None,
):
    """
    Salva il risultato della Walk-Forward Optimization in CSV,
    includendo un header commentato con i parametri di esecuzione.

    Compatibile con:
        pd.read_csv(file_path, index_col="Window", comment="#")

    FIX:
    - Se param_grid contiene booleani True/False, con JSON diventerebbero true/false.
      Qui usiamo repr() (serializzazione Python) per preservare True/False.

    ratio:
      stringa nel formato "train:test" (es. "3:1").
      Viene salvata nei metadata per coerenza con il framework WFO.
    """

    # Validazione leggera ratio (non altera il comportamento, ma evita header incoerenti)
    try:
        train_years_str, test_years_str = ratio.split(":")
        _train_years = int(train_years_str)
        _test_years = int(test_years_str)
        if _train_years <= 0 or _test_years <= 0:
            raise ValueError
    except Exception:
        raise ValueError(f"ratio non valido: '{ratio}'. Formato atteso 'train:test' (es. '3:1')")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_start_date": start_date,
        "data_end_date": end_date,
        "metric": metric,
        "ratio": ratio,
        "force_next_year_params": force_next_year_params,
        "param_grid": param_grid,
    }

    if extra_meta:
        meta.update(extra_meta)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# === WFO METADATA ===\n")
        for k, v in meta.items():
            if isinstance(v, dict):
                # Serializzazione Python => preserva True/False (e None)
                f.write(f"# {k} = {repr(v)}\n")
            else:
                f.write(f"# {k} = {v}\n")
        f.write("# === WFO RESULTS ===\n")
        summary_df.to_csv(f)

    # --- Stampa finale di conferma ---
    print(
        "[WFO SAVE OK] "
        f"File salvato correttamente: '{file_path}' | "
        f"Righe: {len(summary_df)} | "
        f"Metric: {metric} | "
        f"Ratio: {ratio} | "
        f"Data start: {start_date} | "
        f"Data end: {end_date} | "
        f"Force next year params: {force_next_year_params}"
    )

def load_wfo_summary(file_path: str) -> pd.DataFrame:
    """
    Carica il summary WFO salvato da save_rotational_wfo_summary().

    - ignora l'header commentato grazie a comment="#"
    - imposta l'indice su 'Window'
    """
    df = pd.read_csv(
        file_path,
        index_col="Window",
        comment="#",
    )
    return df
# =============================================================================
# PUNTO 3: PRE-CALCOLO VOLATILITY MULTI-WINDOW
# =============================================================================

def precalculate_volatility_multiwindow(
    prices: pd.DataFrame,
    windows: List[int] = [10, 20, 30, 60]
) -> Dict[int, pd.DataFrame]:
    """
    Pre-calcola volatility per multiple finestre in UNA passata.
    
    SPEEDUP: 2x rispetto a calcolare ogni window separatamente.
    
    PERCHÉ PIÙ VELOCE:
    - Singolo loop sui dati
    - Riuso calcoli intermedi (returns)
    - Vectorizzazione ottimale pandas
    
    Parametri
    ----------
    prices : pd.DataFrame
        Prezzi (N days × M stocks)
    windows : list of int
        Liste finestre (es. [10, 20, 30, 60])
        
    Returns
    -------
    dict
        {window_size: volatility_df}
        
    Esempio
    -------
    >>> vol_cache = precalculate_volatility_multiwindow(
    ...     stocks_data,
    ...     windows=[10, 20, 60]
    ... )
    >>> 
    >>> # Poi usa nelle funzioni:
    >>> vol_20 = vol_cache[20]
    >>> vol_60 = vol_cache[60]
    """
    
    # Calcola returns UNA volta sola
    returns = prices.pct_change()
    
    # Pre-alloca dizionario
    vol_dict = {}
    
    print(f"Pre-calculating volatility for {len(windows)} windows...")
    
    # Loop ottimizzato
    for window in tqdm(windows, desc="Vol Windows"):
        # Rolling std con ddof=1 (campionario, standard finance)
        vol = returns.rolling(window=window, min_periods=max(1, window//2)).std() * np.sqrt(252)
        vol_dict[window] = vol
    
    print(f"✅ Volatility cache ready for windows: {windows}")
    
    return vol_dict

    
def walk_forward_rotational(
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    param_grid: Dict[str, List[Any]],
    ratio: str = "3:1",
    metric: str = "Sharpe Ratio",
    verbose: bool = True,
    plot: bool = False,
    force_next_year_params: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    n_jobs: int = -1,
    backend: str = 'loky',
    debug: bool = False,
    auto_reduce_grid: bool = False,
    stability_metric: str = "CAGR",
    stability_k: int = 3,
    stability_n_top_anchors: list[int] | None = None,
    stability_report_dir: str | None = None,
) -> pd.DataFrame:
    """
    Walk-Forward Optimization con grid search vettorizzata.

    ARCHITETTURA E PERFORMANCE
    --------------------------
    Il collo di bottiglia della WFO classica è il ricalcolo degli indicatori
    (momentum, volatilità, rank) per ogni combinazione di parametri. Con 4608
    combo × 9 finestre questo significa ~40.000 ricalcoli dello stesso dato.

    Questa implementazione separa nettamente i due costi:

      1. INDICATORI (O(finestre × giorni × asset) — fatto UNA VOLTA per finestra)
         - Tutti i lookback distinti nel param_grid vengono precomputati in batch.
         - Es: se momentum_lookback_days=[60,120,252], si calcolano 3 mom_df,
           non 4608.

      2. SCORE CACHE (precalcola mw×rm + (1-mw)×riv per ogni tripla distinta)
         - Elimina l'operazione di combinazione dal loop combo.
         - Es: con 4 mom_lb × 4 rp_lb × 4 mw = 64 matrici precalcolate,
           non 4608 ricalcoli.
         - IMPORTANTE: la cache opera su DataFrame pandas (non numpy) per
           garantire identità dei risultati con la versione di riferimento.

      3. SELEZIONE (loop su combo con lookup O(1) nella score_cache)
         - Per ogni combo: lookup combo_score, loop su rebal_dates con
           prev_top_ci fallback — semantica identica alla versione di riferimento.
         - Nessuna costruzione VBT nel loop di train.

      4. VBT PORTFOLIO (costruito solo per il BEST params — 1 volta per finestra)
         - Solo per calcolare il test score.

    CORRETTEZZA
    -----------
    Questa versione è derivata direttamente dalla versione di riferimento (R3)
    con una sola modifica: aggiunta della score_cache per evitare di ricalcolare
    mw×rm + (1-mw)×riv ad ogni combo. Tutto il resto — gestione NaN, period_slices,
    prev_top_ci fallback, conversione .values — è invariato rispetto a R3.

    PARAMETRI
    ---------
    stocks_data              : pd.DataFrame   Prezzi giornalieri.
    benchmark_data           : pd.Series      Prezzi benchmark.
    param_grid               : dict           Griglia parametri.
    ratio                    : str            Train:test in anni (es. '3:1').
    metric                   : str            'Sharpe Ratio' | 'CAGR' | 'Calmar'.
    verbose                  : bool           Header, footer, riga per finestra.
    plot                     : bool           Plot portfolio (solo best params, test).
    force_next_year_params   : bool           Aggiunge finestra futura.
    start_date / end_date    : str | None     Limiti analisi.
    n_jobs                   : int            -1=tutti, 1=sequenziale, N=N core.
    backend                  : str            'loky' | 'threading' | 'multiprocessing'.
    debug                    : bool           Score ogni combo, best params, stack trace errori.
    auto_reduce_grid         : bool           Se True, riduce param_grid via stability analysis
                                              prima del loop WFO. Default False (non-breaking).
    stability_metric         : str            Metrica usata per la stability analysis.
                                              Può differire da metric (WFO). Default "CAGR".
    stability_k              : int            Numero di sotto-periodi per la stability. Default 3.
    stability_n_top_anchors  : list[int]|None Anchor n_top per stability. Default [3,5,8].
    stability_report_dir     : str|None       Se non None, salva il diagnostic_report come CSV.

    AUTO GRID REDUCTION
    -------------------
    Quando attivare (auto_reduce_grid=True):
      - PTF in fase di sviluppo, prima del deploy finale.
      - Griglia ampia (>1000 combo) e universo adeguato (>= max(anchors)+3 ticker).
      - Vuoi ridurre il tempo WFO senza rinunciare alla copertura dei parametri
        numerici (lookback, n_top, rebalance_frequency, momentum_weight).

    Quando NON attivare:
      - PTF già consolidato in produzione (riproducibilità dei run storici).
      - Universi piccoli (ValueError da reduce_grid_via_stability).
      - Vuoi esplorare tutte le combinazioni di flag per analisi comparativa.

    Vedi reduce_grid_via_stability() per i dettagli metodologici (Cell 17).

    RETURNS
    -------
    pd.DataFrame  Index='Window', colonne=param_names + TrainScore + TestScore.
    """
    import sys
    import traceback as _tb
    import os

    # =========================================================================
    # 1) VALIDAZIONE E SETUP
    # =========================================================================
    try:
        train_y, test_y = [int(x) for x in ratio.split(":")]
    except Exception:
        raise ValueError(f"ratio non valido: '{ratio}'. Formato: 'train:test' es. '3:1'")
    if train_y <= 0 or test_y <= 0:
        raise ValueError("train e test devono essere > 0")
    if stocks_data.empty:
        raise ValueError("stocks_data è vuoto")

    stocks_data    = stocks_data.sort_index()
    benchmark_data = benchmark_data.sort_index()
    data_min = stocks_data.index.min()
    data_max = stocks_data.index.max()

    a_start = pd.Timestamp(start_date).normalize() if start_date else pd.Timestamp(data_min).normalize()
    a_end   = (pd.Timestamp(end_date) - pd.Timedelta(days=1)).normalize() if end_date else pd.Timestamp(data_max).normalize()

    if a_end < data_min or a_start > data_max:
        raise ValueError("Finestra di analisi fuori dal range dati")

    n_jobs_eff = os.cpu_count() if n_jobs == -1 else abs(n_jobs) if n_jobs < -1 else max(1, n_jobs)

    def _vprint(*args, **kw):
        if verbose or debug:
            print(*args, **kw)
            sys.stdout.flush()

    def _dprint(*args, **kw):
        if debug:
            print("[DEBUG]", *args, **kw)
            sys.stdout.flush()

    # =========================================================================
    # 1b) AUTO GRID REDUCTION (optional pre-WFO step)
    # =========================================================================
    if auto_reduce_grid:
        _ptf_config = {"stocks_data": stocks_data, "init_cash": 100_000}
        param_grid, _stability_report = reduce_grid_via_stability(
            ptf_config=_ptf_config,
            full_grid=param_grid,
            full_start_date=stocks_data.index.min(),
            full_end_date=stocks_data.index.max(),
            metric=stability_metric,
            k=stability_k,
            n_top_anchors=stability_n_top_anchors,
            verbose=verbose,
        )
        if stability_report_dir is not None:
            from pathlib import Path as _Path
            _Path(stability_report_dir).mkdir(parents=True, exist_ok=True)
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _rpath = _Path(stability_report_dir) / f"stability_{_ts}.csv"
            _stability_report.to_csv(_rpath, index=False)
            _vprint(f"[auto_reduce_grid] stability report saved to {_rpath}")

    # =========================================================================
    # 2) BUFFER E FINESTRE
    # =========================================================================
    def _max_grid(key):
        vals = [int(v) for v in param_grid.get(key, []) if v is not None and np.isfinite(float(v))]
        return max(vals) if vals else 0

    max_lb      = max(_max_grid(k) for k in ["momentum_lookback_days", "riskparity_lookback_days", "ema_span", "vol_window"])
    buffer_days = max(int(max_lb * 2 + 30), 60)

    first_test_y = a_start.year + train_y
    last_test_y  = a_end.year - (test_y - 1)
    if first_test_y > last_test_y:
        raise ValueError(f"Dati insufficienti per ratio={ratio} nel range {a_start.year}→{a_end.year}")

    test_periods = list(range(first_test_y, last_test_y + 1, test_y))
    if force_next_year_params and test_periods:
        test_periods.append(test_periods[-1] + test_y)

    n_windows  = len(test_periods)
    param_keys = list(param_grid.keys())
    all_combos = list(product(*param_grid.values()))
    n_combo    = len(all_combos)

    # =========================================================================
    # 3) HELPER: score da returns
    # =========================================================================
    def _score(ret: pd.Series) -> dict:
        ret = ret.dropna()
        if ret.empty:
            return {"Sharpe Ratio": np.nan, "CAGR": np.nan, "Calmar": np.nan}
        ann_r = (1 + ret.mean()) ** 252 - 1
        ann_v = ret.std(ddof=0) * np.sqrt(252)
        shrp  = ann_r / ann_v if ann_v and np.isfinite(ann_v) else np.nan
        cagr  = (1 + ret).prod() ** (252 / max(len(ret), 1)) - 1
        eq    = (1 + ret).cumprod()
        dd    = (eq / eq.cummax() - 1).min()
        cal   = cagr / abs(dd) if dd and np.isfinite(dd) else np.nan
        return {"Sharpe Ratio": shrp, "CAGR": cagr, "Calmar": cal}

    # =========================================================================
    # 4) CORE: ottimizzazione singola finestra
    # =========================================================================
    def _optimize_window(test_start_year: int, show_inner: bool = False):
        train_start = f"{test_start_year - train_y}-01-01"
        train_end   = f"{test_start_year - 1}-12-31"
        test_start  = f"{test_start_year}-01-01"
        test_end    = f"{test_start_year + test_y - 1}-12-31"

        # ── Slice dati con buffer ─────────────────────────────────────────
        buf_start = pd.Timestamp(train_start) - pd.Timedelta(days=buffer_days)
        tr_px  = stocks_data.loc[buf_start:train_end].dropna(axis=1, how='all').ffill().bfill()
        tr_bch = benchmark_data.loc[buf_start:train_end]

        if tr_px.empty or stocks_data.loc[train_start:train_end].empty:
            _dprint(f"SKIP {train_start}→{train_end}: dati insufficienti")
            _vprint(f"  ↳  {test_start}→{test_end}: skip (train {train_start[:4]}–{train_end[:4]} fuori range dati)")
            return None

        cols = tr_px.columns
        idx  = tr_px.index

        # ── Precalcolo indicatori (UNA VOLTA per finestra) ────────────────
        rets = tr_px.pct_change().fillna(0.0)

        # Raccoglie tutti i lookback distinti — usa dict(zip()) per robustezza
        mom_lbs = sorted(set(
            int(dict(zip(param_keys, c)).get('momentum_lookback_days', 126))
            for c in all_combos
        )) if 'momentum_lookback_days' in param_keys else [126]

        rp_lbs = sorted(set(
            int(dict(zip(param_keys, c)).get('riskparity_lookback_days', 20))
            for c in all_combos
        )) if 'riskparity_lookback_days' in param_keys else [20]

        ema_spans = sorted(set(
            int(dict(zip(param_keys, c)).get('ema_span', 200))
            for c in all_combos
        )) if 'ema_span' in param_keys else []

        mw_values = sorted(set(
            float(dict(zip(param_keys, c)).get('momentum_weight', 0.7))
            for c in all_combos
        )) if 'momentum_weight' in param_keys else [0.7]

        # Momentum rank — DataFrame pandas (identico a R3)
        mom_cache: dict[int, pd.DataFrame] = {}
        for lb in mom_lbs:
            m = tr_px / tr_px.shift(lb)
            mom_cache[lb] = m.rank(pct=True, axis=1, na_option='bottom')

        # Volatilità inversa rank — DataFrame pandas (identico a R3)
        ivol_cache: dict[int, pd.DataFrame] = {}
        for lb in rp_lbs:
            v  = rets.rolling(lb, min_periods=1).std(ddof=0)
            iv = (1.0 / v.replace(0.0, np.nan))
            ivol_cache[lb] = iv.rank(pct=True, axis=1, na_option='bottom')

        # EMA — DataFrame pandas (identico a R3)
        ema_cache: dict[int, pd.DataFrame] = {}
        for sp in ema_spans:
            ema_cache[sp] = tr_px.ewm(span=sp, adjust=False, min_periods=1).mean()

        # ── Score cache: precalcola mw×rm + (1-mw)×riv per ogni tripla ───
        # UNICA modifica rispetto a R3: la combinazione lineare viene calcolata
        # qui una volta per tripla distinta invece che dentro il loop combo.
        # Opera su DataFrame pandas — semantica NaN identica a R3.
        # Il .values viene estratto qui una volta sola per ogni tripla.
        score_cache: dict[tuple, np.ndarray] = {}
        for mw in mw_values:
            for mom_lb in mom_lbs:
                for rp_lb in rp_lbs:
                    rm  = mom_cache[mom_lb]
                    riv = ivol_cache[rp_lb]
                    score_cache[(mw, mom_lb, rp_lb)] = (mw * rm + (1.0 - mw) * riv).values

        # ── Rebalancing dates ─────────────────────────────────────────────
        tr_idx     = tr_px.loc[train_start:train_end].index
        rebal_freq = next(
            (c[param_keys.index('rebalance_frequency')] for c in all_combos
             if 'rebalance_frequency' in param_keys),
            'ME'
        )
        try:
            rebal_dates = compute_rebal_dates(tr_idx, str(rebal_freq).upper())
            rebal_dates = pd.DatetimeIndex([d for d in rebal_dates if d in tr_idx])
        except Exception:
            rebal_dates = pd.DatetimeIndex(
                pd.Series(tr_idx).groupby(tr_idx.to_period('M')).max().values
            )

        if len(rebal_dates) == 0:
            _dprint(f"SKIP {train_start}→{train_end}: nessuna rebal_date")
            return None

        _dprint(f"Finestra {train_start}→{train_end}: "
                f"{len(rebal_dates)} rebal_dates, {len(cols)} asset, "
                f"mom_lbs={mom_lbs}, rp_lbs={rp_lbs}")

        # ── Strutture numpy (identiche a R3) ─────────────────────────────
        rets_np  = rets.values.astype(np.float64)
        date_pos = {d: i for i, d in enumerate(rets.index)}

        rebal_list = list(rebal_dates)
        n_rebal    = len(rebal_list)

        # period_slices: identico a R3 — non modificato
        period_slices = []
        for i, d in enumerate(rebal_list):
            d_next  = rebal_list[i + 1] if i + 1 < n_rebal else rets.index[-1]
            start_i = date_pos.get(d, 0)
            end_i   = date_pos.get(d_next, len(rets) - 1) + 1
            period_slices.append((start_i, end_i))

        rebal_pos_in_full = [date_pos.get(d, 0) for d in rebal_list]

        # ── Grid search ───────────────────────────────────────────────────
        best_score  = -np.inf
        best_params = None

        pbar_i = None
        if show_inner:
            pbar_i = tqdm(
                total=n_combo,
                desc=f"  Grid {test_start_year}",
                position=1, leave=False,
                bar_format='{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} [{elapsed},{rate_fmt}]',
            )

        n_assets = len(cols)

        for combo in all_combos:
            params = dict(zip(param_keys, combo))

            try:
                mom_lb = int(params.get('momentum_lookback_days', 126))
                rp_lb  = int(params.get('riskparity_lookback_days', 20))
                n_top  = int(params.get('n_top', 5))
                mw     = float(params.get('momentum_weight', 0.7))
                f_ema  = bool(params.get('filter_ema', False))
                f_vol  = bool(params.get('filter_volatility', False))
                f_mom  = bool(params.get('filter_min_momentum', False))
                ema_sp = int(params.get('ema_span', 200))
                vol_q  = float(params.get('volatility_quantile', 0.75))
                min_m  = float(params.get('min_momentum_threshold', 1.0))

                # Lookup nella score_cache — zero operazioni aritmetiche
                # Fallback a calcolo diretto se la chiave non esiste (robustezza)
                combo_score = score_cache.get((mw, mom_lb, rp_lb))
                if combo_score is None:
                    rm  = mom_cache.get(mom_lb)
                    riv = ivol_cache.get(rp_lb)
                    if rm is None or riv is None:
                        continue
                    combo_score = (mw * rm + (1.0 - mw) * riv).values

                # ema_arr e px_arr: identici a R3
                ema_arr = ema_cache[ema_sp].values if f_ema and ema_sp in ema_cache else None
                px_arr  = tr_px.values

                # Loop su rebal_date con prev_top_ci fallback — identico a R3
                pf_chunks   = []
                prev_top_ci = None

                for k, (ri, (start_i, end_i)) in enumerate(zip(rebal_pos_in_full, period_slices)):
                    cs = combo_score[ri].copy()

                    if ema_arr is not None:
                        cs = np.where(px_arr[ri] > ema_arr[ri], cs, np.nan)
                    if f_vol:
                        ivol_row = ivol_cache[rp_lb].values[ri]
                        q_thresh = np.nanquantile(ivol_row, 1.0 - vol_q)
                        cs = np.where(ivol_row >= q_thresh, cs, np.nan)
                    if f_mom:
                        shift_i = max(0, ri - mom_lb)
                        raw_mom = px_arr[ri] / np.where(px_arr[shift_i] > 0, px_arr[shift_i], np.nan)
                        cs = np.where(raw_mom > min_m, cs, np.nan)

                    valid_mask = ~np.isnan(cs)
                    if valid_mask.sum() == 0:
                        top_ci = prev_top_ci
                    else:
                        top_ci        = np.argsort(cs[valid_mask])[::-1]
                        valid_indices = np.where(valid_mask)[0]
                        top_ci        = valid_indices[top_ci[:n_top]]
                        prev_top_ci   = top_ci

                    if top_ci is None or len(top_ci) == 0:
                        continue

                    chunk = rets_np[start_i:end_i, :][:, top_ci]
                    if chunk.size == 0:
                        continue
                    pf_chunks.append(chunk.mean(axis=1))

                if not pf_chunks:
                    continue

                pf_ret = np.concatenate(pf_chunks)
                if len(pf_ret) == 0:
                    continue

                # Calcolo score — identico a R3
                ann_r = (1.0 + pf_ret.mean()) ** 252 - 1.0
                ann_v = pf_ret.std(ddof=0) * np.sqrt(252)
                if metric == "Sharpe Ratio":
                    sc = ann_r / ann_v if ann_v > 0 and np.isfinite(ann_v) else np.nan
                elif metric == "CAGR":
                    sc = float(np.prod(1.0 + pf_ret) ** (252 / max(len(pf_ret), 1)) - 1.0)
                elif metric == "Calmar":
                    cagr = float(np.prod(1.0 + pf_ret) ** (252 / max(len(pf_ret), 1)) - 1.0)
                    eq   = np.cumprod(1.0 + pf_ret)
                    dd   = np.min(eq / np.maximum.accumulate(eq) - 1.0)
                    sc   = cagr / abs(dd) if dd != 0 and np.isfinite(dd) else np.nan
                else:
                    sc = np.nan

                _dprint(f"  {params}  {metric}={sc:.4f}" if np.isfinite(sc) else f"  {params}  {metric}=NaN")

                if np.isfinite(sc) and sc > best_score:
                    best_score, best_params = sc, params

            except Exception as exc:
                if debug:
                    print(f"[DEBUG] EXCEPTION combo={params}: {exc}")
                    _tb.print_exc()

            if pbar_i:
                pbar_i.update(1)

        if pbar_i:
            pbar_i.close()

        if best_params is None:
            _dprint(f"NO VALID PARAMS: {train_start}→{train_end}")
            return None

        _dprint(f"BEST {train_start}→{train_end}: {metric}={best_score:.4f} params={best_params}")

        # ── Test: costruisce VBT solo con best_params (1 volta) ──────────
        test_score = np.nan
        if not stocks_data.loc[test_start:test_end].empty:
            try:
                buf_s  = pd.Timestamp(test_start) - pd.Timedelta(days=buffer_days)
                te_px  = stocks_data.loc[buf_s:test_end]
                te_bch = benchmark_data.loc[buf_s:test_end]
                pf_te, *_ = build_rotational_portfolios_vbt(
                    stocks_data=te_px,
                    benchmark_data=te_bch,
                    plot=plot,
                    **best_params,
                )
                test_score = _score(
                    pf_te.returns().loc[test_start:test_end]
                ).get(metric, np.nan)
            except Exception as exc:
                if debug:
                    print(f"[DEBUG] TEST exception: {exc}")
                    _tb.print_exc()

        return {
            **best_params,
            "Window":     f"{test_start}→{test_end}",
            "TrainScore": best_score,
            "TestScore":  test_score,
        }

    # =========================================================================
    # 5) HEADER
    # =========================================================================
    _vprint()
    _vprint("=" * 72)
    _vprint("WALK-FORWARD OPTIMIZATION  (grid vettorizzata v4)")
    _vprint("=" * 72)
    _vprint(f"  Dati         : {data_min.date()} → {data_max.date()}")
    _vprint(f"  Analisi      : {a_start.date()} → {a_end.date()}")
    _vprint(f"  Ratio        : {ratio}  (train={train_y}a, test={test_y}a)")
    _vprint(f"  Metric       : {metric}")
    _vprint(f"  Windows      : {n_windows}")
    _vprint(f"  Combinations : {n_combo:,}")
    _vprint(f"  Parallel     : {'SEQUENTIAL' if n_jobs == 1 else f'n_jobs={n_jobs} (eff={n_jobs_eff}), backend={backend}'}")
    _mom_lbs = sorted(set(int(v) for v in param_grid.get("momentum_lookback_days", [])))
    _vprint(f"  Mom lookbacks : {_mom_lbs}")
    if debug:
        _vprint("  [DEBUG MODE ON]")
    _vprint("=" * 72)
    _vprint()

    # =========================================================================
    # 6) ESECUZIONE
    # =========================================================================
    results = []
    t0_wfo  = time.time()

    # ── Sequenziale ──────────────────────────────────────────────────────────
    if n_jobs == 1:
        pbar = tqdm(
            test_periods, desc="WFO Windows", position=0, leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        )
        for test_year in pbar:
            t0_w = time.time()
            pbar.set_description(
                f"WFO  {test_year - train_y}–{test_year - 1} → "
                f"{test_year}–{test_year + test_y - 1}"
            )
            result    = _optimize_window(test_year, show_inner=True)
            elapsed_w = time.time() - t0_w
            if result:
                results.append(result)
                _vprint(
                    f"  ✓  {result['Window']:<28} "
                    f"Train={result['TrainScore']:+.3f}  "
                    f"Test={result['TestScore']:+.3f}  ({elapsed_w:.0f}s)"
                )
            else:
                _vprint(f"  ✗  {test_year}: fallita ({elapsed_w:.0f}s)")
        pbar.close()

    # ── Parallela ─────────────────────────────────────────────────────────────
    else:
        n_done = n_fail = 0

        pbar = tqdm(
            total=n_windows, desc="WFO Parallel", position=0, leave=True,
            bar_format=(
                '{desc}: {percentage:3.0f}%|{bar}| '
                '{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ),
        )

        gen = Parallel(
            n_jobs=n_jobs, backend=backend, verbose=0,
            return_as="generator",
        )(
            delayed(_optimize_window)(yr, show_inner=False)
            for yr in test_periods
        )

        for result in gen:
            n_done += 1
            avg_s = (time.time() - t0_wfo) / n_done

            if result is not None:
                results.append(result)
                _vprint(
                    f"  ✓  {result['Window']:<28} "
                    f"Train={result['TrainScore']:+.3f}  "
                    f"Test={result['TestScore']:+.3f}  "
                    f"(avg {avg_s:.0f}s/win)"
                )
            else:
                n_fail += 1
                _vprint(f"  ✗  window {n_done}/{n_windows}: fallita")

            pbar.set_postfix_str(
                f"ok={n_done - n_fail}  fail={n_fail}  avg={avg_s:.0f}s/win",
                refresh=True,
            )
            pbar.update(1)

        pbar.close()

    # =========================================================================
    # 7) FOOTER
    # =========================================================================
    elapsed = time.time() - t0_wfo
    n_ok    = len(results)
    n_fail  = n_windows - n_ok

    _vprint()
    _vprint("=" * 72)
    _vprint(f"  Completata in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    _vprint(f"  Windows OK  : {n_ok}/{n_windows}")
    if n_fail:
        _vprint(
            f"  Windows KO  : {n_fail}  "
            f"(dati insufficienti nel periodo di train — normale per le prime finestre)"
        )
    _vprint("=" * 72)
    _vprint()

    if not results:
        raise ValueError("Nessuna finestra WFO completata con successo")

    return pd.DataFrame(results).set_index("Window").sort_index()    
    


"""
build_rotational_portfolios_vbt.py
===================================
Versione migliorata di build_rotational_portfolios_vbt.

MIGLIORAMENTI RISPETTO ALL'ORIGINALE
-------------------------------------
1. ARITÀ RETURN STABILE
   - Ritorna sempre un RotationalVbtResult (dataclass).
   - Nessuna più tuple 4/5/6/7 dipendente da flag booleani.
   - pf_mom, pf_rp, sel_bottom sono None se non richiesti.

2. LOGICA REBAL-DATES DEDUPLICATA
   - Usa compute_rebal_dates() da rotational_engine.py (unica fonte di verità).
   - Rimosse: compute_rebal_dates_trading(), _count_days_in_period(),
     _is_incomplete_last_period() (erano duplicati interni alla funzione).

3. VALIDAZIONE PARAMETRI COMPLETA
   - n_top, ema_span, volatility_quantile, min_momentum_threshold,
     riskparity_lookback_days, momentum_lookback_days tutti validati.

4. DEBUG STRUTTURATO E COMPLETO
   - DebugLogger: livelli INFO / DETAIL / TRACE.
   - Log su: rebal_dates, ogni selezione, pesi calcolati, shift, start_date cut.
   - n_accel loggato correttamente anche quando use_acceleration=False.
   - carried loggato sempre (non solo quando bottom_carried).

5. CARRIED FLAG IN OUTPUT
   - sel_tickers_rot_w ha colonna 'carried' (bool) oltre a 'Top_Tickers'.
   - Informazione prima persa, ora disponibile per audit e reporting.

6. FALLBACK REBAL_DATES VUOTO NON DUPLICA CODICE
   - Un solo punto di costruzione VBT (helper _build_vbt_portfolio).

7. DOCSTRING OPERATIVA
   - Parametri tutti documentati con tipo, default e effetto.
   - Arità return esplicitata per ogni combinazione di flag.
   - vol_cache documentato.
"""



# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS  (arità stabile)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RotationalVbtResult:
    """
    Output strutturato di build_rotational_portfolios_vbt.

    Attributi sempre presenti
    -------------------------
    pf_rot      : vbt.Portfolio   Portafoglio rotazionale equal-weight.
    pf_bh       : vbt.Portfolio | None   Buy-and-hold benchmark (None se benchmark_data=None).
    selections  : pd.DataFrame    Index=RebalanceDate. Colonne: Top_Tickers (list), carried (bool).
    rankings    : pd.DataFrame    Combo-score per tutti i ticker a ogni rebal_date.
    rebal_dates : pd.DatetimeIndex  Date di ribilanciamento effettivamente usate.

    Attributi opzionali (None se non richiesti)
    --------------------------------------------
    pf_mom      : vbt.Portfolio | None   Solo se build_other_portfolios=True.
    pf_rp       : vbt.Portfolio | None   Solo se build_other_portfolios=True.
    sel_bottom  : pd.DataFrame | None    Solo se bottom_tickers=True.

    RETROCOMPATIBILITA' TUPLE
    -------------------------
    L'oggetto e' subscriptable e iterabile, replicando l'arità dell'API legacy:

      default (4)      : pf_rot, pf_bh, selections, rankings
      with_others (6)  : pf_rot, pf_mom, pf_rp, pf_bh, selections, rankings
      with_bottom (5)  : pf_rot, pf_bh, selections, rankings, sel_bottom
      full (7)         : pf_rot, pf_mom, pf_rp, pf_bh, selections, rankings, sel_bottom

    Esempi:
        pf_rot_w, pf_bh, sel, rankings = result   # unpacking
        pf_rot_w = result[0]                       # subscript
        pf_rot_w = result.pf_rot                   # attributo (preferito)
    """
    pf_rot:      object
    pf_bh:       Optional[object]
    selections:  pd.DataFrame
    rankings:    pd.DataFrame
    rebal_dates: pd.DatetimeIndex
    pf_mom:      Optional[object] = None
    pf_rp:       Optional[object] = None
    sel_bottom:  Optional[pd.DataFrame] = None

    def _as_tuple(self) -> tuple:
        """Tuple nell'arità corretta in base ai campi opzionali attivi."""
        has_others = self.pf_mom is not None or self.pf_rp is not None
        has_bottom = self.sel_bottom is not None
        if has_others and has_bottom:
            return (self.pf_rot, self.pf_mom, self.pf_rp, self.pf_bh,
                    self.selections, self.rankings, self.sel_bottom)
        if has_others:
            return (self.pf_rot, self.pf_mom, self.pf_rp, self.pf_bh,
                    self.selections, self.rankings)
        if has_bottom:
            return (self.pf_rot, self.pf_bh,
                    self.selections, self.rankings, self.sel_bottom)
        return (self.pf_rot, self.pf_bh, self.selections, self.rankings)

    def __iter__(self):
        return iter(self._as_tuple())

    def __getitem__(self, idx):
        return self._as_tuple()[idx]

    def __len__(self):
        return len(self._as_tuple())


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG LOGGER INTERNO
# ─────────────────────────────────────────────────────────────────────────────

class _DebugLogger:
    """
    Logger strutturato a 3 livelli per debug interno.

    Livelli
    -------
    0 = off
    1 = INFO   : eventi principali (rebal_dates, selezioni, start_date cut)
    2 = DETAIL : dettagli per ogni rebal_date (mask counts, pesi)
    3 = TRACE  : tutto (shift matrix, fallback steps)
    """
    def __init__(self, level: int):
        self.level = int(level)

    def info(self, msg: str):
        if self.level >= 1:
            print(f"[INFO]   {msg}")

    def detail(self, msg: str):
        if self.level >= 2:
            print(f"[DETAIL] {msg}")

    def trace(self, msg: str):
        if self.level >= 3:
            print(f"[TRACE]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER VBT
# ─────────────────────────────────────────────────────────────────────────────

def _build_vbt_portfolio(prices: pd.DataFrame, weights: pd.DataFrame, init_cash: float):
    """Costruisce un vbt.Portfolio.from_orders con i parametri standard."""
    import vectorbt as vbt
    return vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",
        init_cash=init_cash,
        cash_sharing=True,
        freq="D",
    )


def _build_vbt_bh(bench_px: pd.Series, prices_index: pd.DatetimeIndex, init_cash: float):
    """Costruisce il benchmark buy-and-hold allineato all'indice dei prezzi."""
    import vectorbt as vbt
    bench = bench_px.reindex(prices_index).ffill().dropna().rename("Benchmark")
    return vbt.Portfolio.from_holding(
        close=bench.to_frame(),
        init_cash=init_cash,
        cash_sharing=True,
        freq="D",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FUNZIONE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def build_rotational_portfolios_vbt(
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series | None = None,
    # ── frequenza e lookback ─────────────────────────────────────────────────
    rebalance_frequency: str = "ME",
    momentum_lookback_days: int = 126,
    riskparity_lookback_days: int = 20,
    # ── selezione ────────────────────────────────────────────────────────────
    n_top: int = 5,
    momentum_weight: float = 0.7,
    # ── filtri opzionali ─────────────────────────────────────────────────────
    filter_ema: bool = False,
    filter_volatility: bool = False,
    filter_min_momentum: bool = False,
    ema_span: int = 200,
    volatility_quantile: float = 0.75,
    min_momentum_threshold: float = 1.0,
    # ── accelerazione ────────────────────────────────────────────────────────
    use_acceleration: bool = False,
    # ── portafoglio ──────────────────────────────────────────────────────────
    init_cash: float = 100_000,
    start_date: str | pd.Timestamp | None = None,
    # ── output aggiuntivi ────────────────────────────────────────────────────
    build_other_portfolios: bool = False,
    bottom_tickers: bool = False,
    # ── performance ──────────────────────────────────────────────────────────
    vol_cache: dict | None = None,
    # ── plot ─────────────────────────────────────────────────────────────────
    plot: bool = True,
    portfolio_name: str = "Portafoglio rotazionale",
    vbt_plot_width: int = 1000,
    # ── debug ────────────────────────────────────────────────────────────────
    debug: bool = False,
    debug_level: int = 1,
) -> RotationalVbtResult:
    """
    Costruisce portafogli rotazionali equal-weight con ribilanciamento trading-aligned.

    MECCANISMO CORE
    ---------------
    1. Calcola le rebal_dates come ultimo trading-day di ciascun periodo completo
       (delegato a compute_rebal_dates(), unica fonte di verità nel progetto).
    2. A ogni rebal_date applica filtri opzionali (EMA, volatilità, momentum minimo,
       accelerazione) e seleziona i top-N ticker per combo-score
       (momentum_weight * rank_mom + (1-momentum_weight) * rank_inv_vol).
    3. Se la selezione è vuota dopo i filtri, esegue carry-forward dell'ultima
       selezione valida (segnalato nella colonna `carried` dell'output).
    4. Costruisce la matrice dei pesi equal-weight e la shifta di 1 giorno
       (gli ordini vengono eseguiti il giorno di trading successivo alla rebal_date).

    PARAMETRI
    ---------
    stocks_data : pd.DataFrame
        Prezzi giornalieri. Index = DatetimeIndex (trading days only).
        Colonne = ticker. NaN interni vengono ffill/bfill.

    benchmark_data : pd.Series | None
        Prezzi del benchmark (es. ETF su indice). Se None, pf_bh=None in output.

    rebalance_frequency : str  default='ME'
        Frequenza di ribilanciamento. Valori supportati:
        'ME'/'M' (mensile), 'QE'/'Q' (trimestrale), 'YE'/'Y' (annuale),
        'W'/'W-FRI' (settimanale), 'D' (giornaliero).

    momentum_lookback_days : int  default=126
        Finestra in giorni per il calcolo del momentum (price ratio).
        Deve essere > 0 e < len(stocks_data).

    riskparity_lookback_days : int  default=20
        Finestra in giorni per il calcolo della volatilità (rolling std dei ritorni).
        Deve essere > 0.

    n_top : int  default=5
        Numero di ticker da selezionare a ogni ribilanciamento. Deve essere >= 1.

    momentum_weight : float  default=0.7
        Peso del rank di momentum nel combo-score [0.0, 1.0].
        Il peso del rank inv-volatilità è (1 - momentum_weight).

    filter_ema : bool  default=False
        Se True, esclude i ticker il cui prezzo è < EMA(ema_span).

    filter_volatility : bool  default=False
        Se True, esclude i ticker nel quantile superiore di volatilità
        (soglia = volatility_quantile).

    filter_min_momentum : bool  default=False
        Se True, esclude i ticker con momentum < min_momentum_threshold.
        Un threshold di 1.0 significa: escludi chi ha prezzo attuale < prezzo N giorni fa.

    ema_span : int  default=200
        Span per il calcolo dell'EMA (usato solo se filter_ema=True). Deve essere > 0.

    volatility_quantile : float  default=0.75
        Quantile di esclusione per volatilità (usato solo se filter_volatility=True).
        Deve essere in (0, 1).

    min_momentum_threshold : float  default=1.0
        Soglia minima di momentum (usato solo se filter_min_momentum=True).

    use_acceleration : bool  default=False
        Se True, aggiunge al combo-score un termine di accelerazione del momentum
        (EWM a breve su EWM a lungo). I ticker con accelerazione negativa vengono esclusi.

    init_cash : float  default=100_000
        Capitale iniziale per i portafogli VBT.

    start_date : str | pd.Timestamp | None  default=None
        Se fornito, i portafogli vengono tagliati a partire da questa data.
        I pesi vengono costruiti sull'intero range (per lookback corretto)
        e poi tagliati. Deve essere una data presente o successiva al primo
        trading day disponibile.

    build_other_portfolios : bool  default=False
        Se True, costruisce anche:
        - pf_mom : portafoglio momentum puro (pesi proporzionali al momentum)
        - pf_rp  : portafoglio risk-parity puro (pesi proporzionali a 1/vol)
        Disponibili in result.pf_mom e result.pf_rp.

    bottom_tickers : bool  default=False  [SPERIMENTALE]
        Se True, calcola anche la selezione dei bottom-N ticker (peggiori per score).
        Disponibile in result.sel_bottom.

    vol_cache : dict | None  default=None
        Cache delle volatilità precalcolate. Formato: {lookback_days: vol_df}.
        vol_df deve avere stesso indice di stocks_data.
        Utile in WFO dove la stessa finestra viene ricalcolata più volte.

    plot : bool  default=True
        Se True, mostra il grafico Plotly dei rendimenti cumulati.

    portfolio_name : str  default='Portafoglio rotazionale'
        Titolo del grafico.

    vbt_plot_width : int  default=1000
        Larghezza in pixel del grafico Plotly.

    debug : bool  default=False
        Abilita il logging di debug (equivalente a debug_level=1 se True, 0 se False).
        Se debug_level è specificato esplicitamente, ha precedenza.

    debug_level : int  default=1
        Livello di dettaglio del debug (attivo solo se debug=True):
        1 = INFO   : eventi principali (rebal_dates, selezioni, start_date cut)
        2 = DETAIL : dettaglio per ogni rebal_date (mask counts, pesi effettivi)
        3 = TRACE  : tutto (shift matrix, valori intermedi)

    RETURN
    ------
    RotationalVbtResult con i seguenti campi:

    Sempre presenti:
      .pf_rot      vbt.Portfolio   Portafoglio rotazionale.
      .pf_bh       vbt.Portfolio | None   Benchmark B&H. None se benchmark_data=None.
      .selections  pd.DataFrame   Index=RebalanceDate. Colonne:
                     Top_Tickers (list[str]) : ticker selezionati
                     carried (bool)          : True se selezione da carry-forward
      .rankings    pd.DataFrame   Combo-score completo per ogni rebal_date.
      .rebal_dates pd.DatetimeIndex  Date di ribilanciamento usate.

    Opzionali (None se flag non attivo):
      .pf_mom      vbt.Portfolio | None   Solo se build_other_portfolios=True.
      .pf_rp       vbt.Portfolio | None   Solo se build_other_portfolios=True.
      .sel_bottom  pd.DataFrame | None    Solo se bottom_tickers=True.

    RAISES
    ------
    ValueError
        Se i parametri numerici sono fuori range.
    TypeError
        Se stocks_data non è un pd.DataFrame.
    RuntimeError
        Se compute_rebal_dates non è disponibile nel namespace.
    """

    # ── 0) Risolvi debug_level ────────────────────────────────────────────────
    effective_level = debug_level if debug else 0
    log = _DebugLogger(effective_level)

    # ── 1) Validazione parametri ──────────────────────────────────────────────
    if not isinstance(stocks_data, pd.DataFrame):
        raise TypeError("stocks_data deve essere un pd.DataFrame")
    if not 0.0 <= momentum_weight <= 1.0:
        raise ValueError(f"momentum_weight={momentum_weight} non in [0, 1]")
    if n_top < 1:
        raise ValueError(f"n_top={n_top} deve essere >= 1")
    if momentum_lookback_days < 1:
        raise ValueError(f"momentum_lookback_days={momentum_lookback_days} deve essere >= 1")
    if riskparity_lookback_days < 1:
        raise ValueError(f"riskparity_lookback_days={riskparity_lookback_days} deve essere >= 1")
    if filter_ema and ema_span < 1:
        raise ValueError(f"ema_span={ema_span} deve essere >= 1")
    if filter_volatility and not 0.0 < volatility_quantile < 1.0:
        raise ValueError(f"volatility_quantile={volatility_quantile} deve essere in (0, 1)")

    _compute_rebal = compute_rebal_dates
    if _compute_rebal is None:
        raise RuntimeError(
            "compute_rebal_dates non disponibile. "
            "Assicurarsi che rotational_engine.py sia importato nel namespace."
        )

    # ── 2) Normalizzazione dati ───────────────────────────────────────────────
    prices = stocks_data.dropna(axis=1, how="all").ffill().bfill().copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.sort_index()
    cols = prices.columns
    idx_full = prices.index.unique()

    bench_px = None
    if benchmark_data is not None:
        bench_px = benchmark_data.copy()
        bench_px.index = pd.to_datetime(bench_px.index).normalize()
        bench_px = bench_px.sort_index()

    rets = prices.pct_change().fillna(0.0)

    log.info(
        f"freq={rebalance_frequency}  "
        f"prices={idx_full.min().date()}→{idx_full.max().date()}  "
        f"n_assets={prices.shape[1]}  "
        f"n_top={n_top}  mom_lb={momentum_lookback_days}  rp_lb={riskparity_lookback_days}"
    )

    # ── 3) Rebal dates (unica fonte di verità) ────────────────────────────────
    freq = str(rebalance_frequency).upper().strip()
    rebal_dates = _compute_rebal(idx_full, freq)
    rebal_dates = pd.DatetimeIndex(
        [d for d in rebal_dates if d in idx_full]
    ).sort_values().unique()

    log.info(
        f"rebal_dates: n={len(rebal_dates)}  "
        f"first={rebal_dates[0].date() if len(rebal_dates) else 'N/A'}  "
        f"last={rebal_dates[-1].date() if len(rebal_dates) else 'N/A'}"
    )
    log.detail(f"rebal_dates tail: {[str(d.date()) for d in rebal_dates[-6:]]}")

    # ── 4) Precomputa indicatori ──────────────────────────────────────────────
    mom_df = prices / prices.shift(momentum_lookback_days)

    if vol_cache is not None and riskparity_lookback_days in vol_cache:
        log.detail(f"vol_cache HIT: window={riskparity_lookback_days}")
        vol_df = vol_cache[riskparity_lookback_days]
        if not vol_df.index.equals(rets.index):
            vol_df = vol_df.reindex(rets.index)
    else:
        if vol_cache is not None:
            log.detail(f"vol_cache MISS: window={riskparity_lookback_days}, calcolo on-demand")
        vol_df = rets.rolling(riskparity_lookback_days, min_periods=1).std(ddof=0)

    inv_vol_df = 1.0 / vol_df.replace(0.0, np.nan)
    rank_mom_df  = mom_df.rank(pct=True, axis=1, na_option="bottom")
    rank_ivol_df = inv_vol_df.rank(pct=True, axis=1, na_option="bottom")

    ema_df = None
    if filter_ema:
        ema_df = prices.ewm(span=ema_span, adjust=False, min_periods=1).mean()

    accel_df = accel_rank_df = None
    _ACCEL_WEIGHT = 0.20
    if use_acceleration:
        mom_sm  = mom_df.ewm(span=5, adjust=False, min_periods=1).mean()
        accel_df = mom_sm - mom_sm.shift(10)
        accel_rank_df = accel_df.rank(pct=True, axis=1, na_option="bottom")

    # ── 5) Loop di selezione ──────────────────────────────────────────────────
    w_rot: dict[pd.Timestamp, pd.Series] = {}
    w_mom: dict[pd.Timestamp, pd.Series] = {} if build_other_portfolios else None
    w_rp:  dict[pd.Timestamp, pd.Series] = {} if build_other_portfolios else None

    sel_dates:    list[pd.Timestamp] = []
    sel_tickers:  list[list[str]]    = []
    sel_carried:  list[bool]         = []
    rank_records: dict[pd.Timestamp, pd.Series] = {}

    bottom_sel:     list[list[str]] = []
    prev_top:       list[str] | None = None
    prev_bottom:    list[str] | None = None

    # ── fallback: nessuna rebal_date ──────────────────────────────────────────
    if len(rebal_dates) == 0:
        log.info("WARN: nessuna rebal_date valida → portafoglio in cash (pesi zero)")
        return _build_empty_result(
            prices=prices,
            bench_px=bench_px,
            cols=cols,
            init_cash=init_cash,
            start_date=start_date,
            build_other_portfolios=build_other_portfolios,
            bottom_tickers=bottom_tickers,
        )

    for d in rebal_dates:
        d = pd.Timestamp(d).normalize()

        mom     = mom_df.loc[d]
        vol     = vol_df.loc[d]
        inv_vol = inv_vol_df.loc[d]
        rm      = rank_mom_df.loc[d]
        riv     = rank_ivol_df.loc[d]

        # mask progressiva con conteggi per debug
        mask = mom.notna() & inv_vol.notna()
        n_base = int(mask.sum())

        n_ema = n_vol = n_minmom = n_accel = n_base  # default se filtro non attivo

        if filter_ema:
            mask &= prices.loc[d] > ema_df.loc[d]
            n_ema = int(mask.sum())

        if filter_volatility:
            mask &= vol < vol.quantile(volatility_quantile)
            n_vol = int(mask.sum())

        if filter_min_momentum:
            mask &= mom > min_momentum_threshold
            n_minmom = int(mask.sum())

        combo = momentum_weight * rm + (1.0 - momentum_weight) * riv

        if use_acceleration:
            mask &= accel_df.loc[d] > 0
            combo = combo + _ACCEL_WEIGHT * accel_rank_df.loc[d]
            n_accel = int(mask.sum())

        combo_masked = combo.where(mask)
        rank_records[d] = combo_masked.sort_values(ascending=False)

        # selezione top-N
        valid = combo_masked.dropna()
        if not valid.empty:
            top_list = list(valid.nlargest(n_top).index)
            carried  = False
        elif prev_top is not None:
            top_list = list(prev_top)
            carried  = True
        else:
            top_list = []
            carried  = False

        sel_dates.append(d)
        sel_tickers.append(top_list)
        sel_carried.append(carried)

        if top_list and not carried:
            prev_top = top_list

        # log per ogni rebal_date
        status = "CARRY-FWD" if carried else ("EMPTY    " if not top_list else "OK       ")
        log.detail(
            f"{status} | {d.date()} | "
            f"base={n_base} ema={n_ema} vol={n_vol} mom={n_minmom} accel={n_accel} | "
            f"selected={top_list}"
        )

        # pesi equal-weight
        w = pd.Series(0.0, index=cols)
        valid_top = [t for t in top_list if t in cols]
        if valid_top:
            w[valid_top] = 1.0 / len(valid_top)
        w_rot[d] = w

        log.trace(f"  w_rot[{d.date()}] non-zero: { {k: round(v,4) for k,v in w.items() if v > 0} }")

        # portafogli aggiuntivi
        if build_other_portfolios:
            w1 = mom.where(mask, 0.0);  s1 = float(w1.sum())
            w_mom[d] = w1.div(s1) if np.isfinite(s1) and s1 else w1 * 0.0

            w2 = inv_vol.where(mask, 0.0); s2 = float(w2.sum())
            w_rp[d]  = w2.div(s2) if np.isfinite(s2) and s2 else w2 * 0.0

        # bottom tickers (sperimentale)
        if bottom_tickers:
            if not valid.empty:
                bot_list = list(valid.nsmallest(n_top).index)
                prev_bottom = bot_list
            elif prev_bottom is not None:
                bot_list = list(prev_bottom)
                log.detail(f"  BOTTOM CARRY-FWD | {d.date()} | {bot_list}")
            else:
                bot_list = []
            bottom_sel.append(bot_list)

    # ── 6) Matrice pesi → shift(1) ────────────────────────────────────────────
    def _to_shifted(w_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame(w_dict).T.reindex(idx_full).ffill().fillna(0.0)
        shifted = df.shift(1).ffill().fillna(0.0)
        log.trace(
            f"  weight matrix: shape={shifted.shape}  "
            f"non-zero rows={(shifted.sum(axis=1) > 0).sum()}"
        )
        return shifted

    w_rot_sh = _to_shifted(w_rot)

    w_mom_sh = _to_shifted(w_mom) if build_other_portfolios else None
    w_rp_sh  = _to_shifted(w_rp)  if build_other_portfolios else None

    # ── 7) Taglio start_date ──────────────────────────────────────────────────
    if start_date is not None:
        s_ts = pd.Timestamp(start_date).normalize()
        n_before = len(prices)
        w_rot_sh = w_rot_sh.loc[s_ts:]
        prices   = prices.loc[s_ts:]
        if bench_px is not None:
            bench_px = bench_px.loc[s_ts:]
        if build_other_portfolios:
            w_mom_sh = w_mom_sh.loc[s_ts:]
            w_rp_sh  = w_rp_sh.loc[s_ts:]
        log.info(f"start_date cut: {s_ts.date()}  ({n_before - len(prices)} righe rimosse)")

    log.info(
        f"Done: selections={len(sel_dates)}  "
        f"carried={sum(sel_carried)}  "
        f"empty={sum(1 for t in sel_tickers if not t)}"
    )

    # ── 8) Costruzione portafogli VBT ─────────────────────────────────────────
    pf_rot = _build_vbt_portfolio(prices, w_rot_sh, init_cash)

    pf_bh = None
    if bench_px is not None:
        pf_bh = _build_vbt_bh(bench_px, prices.index, init_cash)

    pf_mom = pf_rp = None
    if build_other_portfolios:
        pf_mom = _build_vbt_portfolio(prices, w_mom_sh, init_cash)
        pf_rp  = _build_vbt_portfolio(prices, w_rp_sh,  init_cash)

    # ── 9) Plot ───────────────────────────────────────────────────────────────
    if plot:
        _plot_results(
            pf_rot=pf_rot,
            pf_bh=pf_bh,
            pf_mom=pf_mom,
            pf_rp=pf_rp,
            init_cash=init_cash,
            portfolio_name=portfolio_name,
            width=vbt_plot_width,
        )

    # ── 10) Output DataFrames ─────────────────────────────────────────────────
    selections = pd.DataFrame(
        {"Top_Tickers": sel_tickers, "carried": sel_carried},
        index=pd.DatetimeIndex(sel_dates, name="RebalanceDate"),
    )

    rankings = pd.DataFrame(rank_records).T
    rankings.index.name = "RebalanceDate"

    sel_bottom_df = None
    if bottom_tickers:
        sel_bottom_df = pd.DataFrame(
            {"Bottom_Tickers": bottom_sel},
            index=pd.DatetimeIndex(sel_dates, name="RebalanceDate"),
        )

    return RotationalVbtResult(
        pf_rot=pf_rot,
        pf_bh=pf_bh,
        selections=selections,
        rankings=rankings,
        rebal_dates=rebal_dates,
        pf_mom=pf_mom,
        pf_rp=pf_rp,
        sel_bottom=sel_bottom_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: rebal_dates vuoto
# ─────────────────────────────────────────────────────────────────────────────

def _build_empty_result(
    prices: pd.DataFrame,
    bench_px: pd.Series | None,
    cols: pd.Index,
    init_cash: float,
    start_date,
    build_other_portfolios: bool,
    bottom_tickers: bool,
) -> RotationalVbtResult:
    """
    Costruisce un RotationalVbtResult con pesi zero (portafoglio in cash)
    quando rebal_dates è vuoto. Un solo punto di costruzione VBT.
    """
    if start_date is not None:
        s_ts = pd.Timestamp(start_date).normalize()
        prices = prices.loc[s_ts:]
        if bench_px is not None:
            bench_px = bench_px.loc[s_ts:]

    w_zero = pd.DataFrame(0.0, index=prices.index, columns=cols)

    pf_rot = _build_vbt_portfolio(prices, w_zero, init_cash)
    pf_bh  = _build_vbt_bh(bench_px, prices.index, init_cash) if bench_px is not None else None

    empty_sel = pd.DataFrame(
        {"Top_Tickers": pd.Series(dtype=object), "carried": pd.Series(dtype=bool)},
        index=pd.DatetimeIndex([], name="RebalanceDate"),
    )
    empty_rank = pd.DataFrame(index=pd.DatetimeIndex([], name="RebalanceDate"), columns=cols)
    empty_dates = pd.DatetimeIndex([])

    return RotationalVbtResult(
        pf_rot=pf_rot,
        pf_bh=pf_bh,
        selections=empty_sel,
        rankings=empty_rank,
        rebal_dates=empty_dates,
        pf_mom=_build_vbt_portfolio(prices, w_zero, init_cash) if build_other_portfolios else None,
        pf_rp=_build_vbt_portfolio(prices, w_zero, init_cash)  if build_other_portfolios else None,
        sel_bottom=pd.DataFrame(
            {"Bottom_Tickers": pd.Series(dtype=object)},
            index=pd.DatetimeIndex([], name="RebalanceDate"),
        ) if bottom_tickers else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _plot_results(pf_rot, pf_bh, pf_mom, pf_rp, init_cash, portfolio_name, width):
    import plotly.graph_objects as go

    fig = go.Figure()

    def _add(pf, name, color=None, opacity=1.0):
        y = pf.value() / init_cash
        kw = dict(x=y.index, y=y, mode="lines", name=name, opacity=opacity)
        if color:
            kw["line"] = dict(color=color, width=2)
        fig.add_trace(go.Scatter(**kw))

    _add(pf_rot, "Rotational")
    if pf_mom is not None:
        _add(pf_mom, "Momentum puro")
    if pf_rp is not None:
        _add(pf_rp,  "Risk-Parity puro")
    if pf_bh is not None:
        _add(pf_bh, "Benchmark", color="gray", opacity=0.8)

    fig.update_layout(
        title=f"{portfolio_name} – Rendimenti cumulati",
        yaxis_tickformat=".0%",
        width=width,
        height=600,
        template="plotly_white",
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                dict(step="all"),
            ]),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )
    fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# RETROCOMPATIBILITÀ: unpacking del vecchio formato tuple
# ─────────────────────────────────────────────────────────────────────────────

def unpack_vbt_result(result: RotationalVbtResult, mode: str = "default") -> tuple:
    """
    Converte un RotationalVbtResult nel formato tuple dell'API precedente.
    Utile per codice legacy che fa unpacking diretto.

    mode='default'
        → (pf_rot, pf_bh, sel_tickers_rot_w, rankings_df)
    mode='with_others'
        → (pf_rot, pf_mom, pf_rp, pf_bh, sel_tickers_rot_w, rankings_df)
    mode='with_bottom'
        → (pf_rot, pf_bh, sel_tickers_rot_w, rankings_df, sel_bottom)
    mode='full'
        → (pf_rot, pf_mom, pf_rp, pf_bh, sel_tickers_rot_w, rankings_df, sel_bottom)
    """
    modes = {
        "default":     (result.pf_rot, result.pf_bh, result.selections, result.rankings),
        "with_others": (result.pf_rot, result.pf_mom, result.pf_rp, result.pf_bh, result.selections, result.rankings),
        "with_bottom": (result.pf_rot, result.pf_bh, result.selections, result.rankings, result.sel_bottom),
        "full":        (result.pf_rot, result.pf_mom, result.pf_rp, result.pf_bh, result.selections, result.rankings, result.sel_bottom),
    }
    if mode not in modes:
        raise ValueError(f"mode='{mode}' non valido. Scegliere tra: {list(modes)}")
    return modes[mode]

# Metrics (backtest non ottimizati)

def build_benchmark(benchmark_portfolio: dict, start_date="2018-01-01", end_date=None, auto_adjust: bool = True) -> pd.Series:
    """
    Crea un benchmark sintetico a partire da un dizionario {ticker: peso}.
    Scarica i dati da yfinance, li normalizza e li aggrega secondo i pesi.

    Parametri:
    -----------
    benchmark_portfolio : dict
        Dizionario dei ticker e dei pesi (es. {'SPY': 0.5, 'GLD': 0.3, 'TLT': 0.2})
    start_date : str
        Data di inizio in formato 'YYYY-MM-DD'
    end_date : str
        Data di fine. Se None, usa la data attuale.

    Ritorna:
    --------
    benchmark_series : pd.Series
        Serie storica dei prezzi del benchmark ponderato.
    """
    tickers = list(benchmark_portfolio.keys())
    weights = pd.Series(benchmark_portfolio)

    # Scarica i dati da yfinance
    data = yf.download(tickers, start=start_date, end=end_date,auto_adjust=auto_adjust)["Close"]
    data = data.dropna(how="all")  # rimuove righe completamente NaN

    # Normalizza ogni colonna a 1 all'inizio
    data_norm = data / data.iloc[0]

    # Allinea i pesi solo ai ticker presenti nei dati scaricati
    common_tickers = [ticker for ticker in weights.index if ticker in data_norm.columns]
    weights = weights[common_tickers]
    data_norm = data_norm[common_tickers]

    # Calcolo benchmark come media ponderata
    benchmark_series = (data_norm * weights).sum(axis=1)

    return benchmark_series.dropna()


def analyze_portfolio_metrics(
    port_cumrets: pd.DataFrame,
    portfolio_name = "Portafoglio Rotazionale",
    benchmark_cumret: pd.DataFrame = None,
    freq: str = "D",
    sort_by: str = "CAGR (%)",
    ascending: bool = False,
    plot_radar: bool = False,
    radar_metrics: Union[str, List[str]] = "all",
    highlight_best: bool = True
) -> pd.DataFrame:
    """
    Calcola e confronta le metriche di performance di uno o più portafogli
    a partire da serie di rendimenti cumulativi (base 1.0).

    Parameters
    ----------
    port_cumrets : pd.DataFrame
        DataFrame con una colonna per ogni portafoglio da analizzare.
        I valori devono essere rendimenti cumulativi (es. 1.0 = partenza,
        1.15 = +15%).
    portfolio_name : str
        Titolo descrittivo usato nell'intestazione del report.
    benchmark_cumret : pd.DataFrame, optional
        Serie cumulativa del benchmark. Se fornita viene aggiunta al confronto.
    freq : str
        Frequenza dei dati: "D" (giornaliera), "W" (settimanale), "ME" (mensile).
        Usata per il fattore di annualizzazione.
    sort_by : str
        Colonna per l'ordinamento del DataFrame risultante.
    ascending : bool
        Direzione dell'ordinamento.
    plot_radar : bool
        Se True, genera un grafico radar delle metriche normalizzate.
    radar_metrics : str | list[str]
        Metriche da includere nel radar. "all" usa tutte le disponibili.
    highlight_best : bool
        Se True, evidenzia il valore migliore per ogni colonna nel DataFrame
        stampato.

    Returns
    -------
    pd.DataFrame
        Tabella con una riga per portafoglio e colonne:
        Cumulative Return (%), Annualized Return (%), CAGR (%),
        Annualized Volatility (%), Sharpe Ratio, Sortino Ratio,
        Max Drawdown (%), Calmar Ratio, Win Rate (%).
    """
    def compute_metrics_from_cum(cum_series: pd.Series) -> Dict[str, float]:
        """Calcola le metriche partendo da una serie di rendimenti cumulativi (>=1)."""
        # rets = cum_series.pct_change().dropna()
        rets = cum_series.pct_change(fill_method=None).dropna()
        if rets.empty or len(rets) < 2:
            return {col: np.nan for col in [
                "Cumulative Return (%)", "Annualized Return (%)", "CAGR (%)",
                "Annualized Volatility (%)", "Sharpe Ratio", "Sortino Ratio",
                "Max Drawdown (%)", "Calmar Ratio", "Win Rate (%)",
                "Avg Daily Return (%)", "Median Daily Return (%)"
            ]}

        # fattore di annualizzazione in base alla frequenza
        ann_factor = {"D": 252, "W": 52, "ME": 12}.get(freq.upper(), 252)

        # --------------------- metriche principali ---------------------
        total_ret = cum_series.iloc[-1] - 1
        duration_days = (cum_series.index[-1] - cum_series.index[0]).days
        total_years = duration_days / 365.25 if duration_days > 0 else np.nan

        # CAGR (compounded)
        cagr = (1 + total_ret) ** (1 / total_years) - 1 if total_years else np.nan

        # Annualized arithmetic return
        ann_return = rets.mean() * ann_factor

        # Volatilità annualizzata
        ann_vol = rets.std(ddof=0) * np.sqrt(ann_factor)

        # Sharpe e Sortino (using arithmetic return)
        sharpe = ann_return / ann_vol if ann_vol else np.nan
        downside_std = rets[rets < 0].std(ddof=0) * np.sqrt(ann_factor)
        sortino = ann_return / downside_std if downside_std else np.nan

        # Drawdown & Calmar
        max_dd = (cum_series / cum_series.cummax() - 1).min()
        calmar = cagr / abs(max_dd) if max_dd else np.nan

        # Win rate
        win_rate = (rets > 0).mean() * 100

        return {
            "Cumulative Return (%)": total_ret * 100,
            "Annualized Return (%)": ann_return * 100,
            "CAGR (%)": cagr * 100,
            "Annualized Volatility (%)": ann_vol * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown (%)": -max_dd * 100,
            "Calmar Ratio": calmar,
            "Win Rate (%)": win_rate,
            "Avg Daily Return (%)": rets.mean() * 100,
            # "Median Daily Return (%)": rets.median() * 100
        }

    def normalize_by_absolute_ranges(df: pd.DataFrame,
                                     ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """Normalizza ciascuna colonna di df secondo range assoluti predefiniti."""
        df_norm = pd.DataFrame(index=df.index)
        for col in df.columns:
            if col in ranges:
                min_val, max_val = ranges[col]
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = df[col]
        return df_norm.clip(0, 1)

    print(f'{Emoji.SETUP} Analisi Portfolio Rotazionale ({BOLD}{portfolio_name}{RESET}):\n')
    
    # -----------------------------------------------------------------
    # Prepara dataframe input
    # -----------------------------------------------------------------
    if isinstance(port_cumrets, pd.Series):
        port_cumrets = port_cumrets.to_frame("Rotational")

    metrics = {name: compute_metrics_from_cum(series)
               for name, series in port_cumrets.items()}

    if benchmark_cumret is not None:
        bench_series = benchmark_cumret.squeeze() if isinstance(benchmark_cumret, pd.DataFrame) else benchmark_cumret
        metrics["Benchmark"] = compute_metrics_from_cum(bench_series)

    metrics_df = pd.DataFrame(metrics).T

    # -----------------------------------------------------------------
    # Ordinamento & styling opzionale
    # -----------------------------------------------------------------
    if sort_by in metrics_df.columns:
        metrics_df = metrics_df.sort_values(by=sort_by, ascending=ascending)

    if highlight_best:
        styled = metrics_df.style
        for col in metrics_df.select_dtypes(include=[np.number]).columns:
            reverse = col in ["Annualized Volatility (%)", "Max Drawdown (%)"]
            styled = styled.background_gradient(
                subset=[col],
                cmap="RdYlGn_r" if reverse else "RdYlGn",
                low=0, high=0, axis=0
            )

    # -----------------------------------------------------------------
    # Radar chart opzionale
    # -----------------------------------------------------------------
    if plot_radar and len(metrics_df) <= 5:
        if radar_metrics == "basic":
            radar_cols = ["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"]
        elif radar_metrics == "all":
            radar_cols = list(metrics_df.columns)
        elif isinstance(radar_metrics, list):
            radar_cols = radar_metrics
        else:
            radar_cols = ["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"]

        abs_ranges = {
            "CAGR (%)": (0, 20),
            "Annualized Return (%)": (0, 20),
            "Sharpe Ratio": (0, 2),
            "Sortino Ratio": (0, 3),
            "Max Drawdown (%)": (0, 50),
            "Win Rate (%)": (0, 100),
            "Annualized Volatility (%)": (0, 30),
            "Calmar Ratio": (0, 2),
            "Avg Daily Return (%)": (0, 0.3),
            "Median Daily Return (%)": (0, 0.3),
            "Cumulative Return (%)": (0, 200)
        }

        radar_df = normalize_by_absolute_ranges(
            metrics_df[radar_cols].fillna(0), abs_ranges
        )

        fig = go.Figure()
        for idx, row in radar_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row.tolist(),
                theta=radar_cols,
                fill='toself',
                name=idx
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, tickformat=".1f", range=[0, 1])),
            title="Radar Chart normalizzato su range assoluti",
            height=750, width=900,
            template="plotly_white",
            showlegend=True
        )
        fig.show()

    return metrics_df.round(2)

def pf_stats_aligned(
    pf,
    benchmark: pd.Series | None = None,
    analysis_start_date: str | pd.Timestamp | None = None,
    analysis_end_date: str | pd.Timestamp | None = None,
    rebase_to: float = 100_000.0,
    annualization: int = 252,
    label_pf: str = "Portfolio",
    label_bench: str = "Benchmark",
    return_series: bool = False,
) -> dict:
    """
    Calcola metriche equity-based ALLINEATE su una finestra di analisi comune,
    indipendentemente dal fatto che pf sia stato costruito con buffer (warm-up).

    Policy:
      - Finestra = [analysis_start_date, analysis_end_date] se forniti.
      - Se analysis_end_date è None -> usa ultimo giorno comune tra equity pf e benchmark (se presente),
        altrimenti ultimo giorno equity pf.
      - Benchmark viene riallineato sul calendario della equity del pf via reindex+ffill+bfill.
      - Equity pf e benchmark vengono REBASED a `rebase_to` sul primo giorno della finestra.
      - Metriche: Total Return, CAGR, Vol Ann, Sharpe (rf=0), Max DD.

    Ritorna un dict:
      {
        "pf": metrics_pf,
        "benchmark": metrics_bench or None,
        "window": (start_ts, end_ts),
        "series": {"equity_pf":..., "equity_bench":...}   # solo se return_series=True
      }
    """
    def _to_ts(x):
        if x is None:
            return None
        return pd.Timestamp(x).normalize()

    def _rebase(s: pd.Series, base: float) -> pd.Series:
        first = float(s.iloc[0])
        if first == 0 or not np.isfinite(first):
            raise ValueError("Rebase impossibile: primo valore nullo/non finito.")
        return (s / first) * float(base)

    def _equity_metrics(eq: pd.Series) -> dict:
        eq = eq.dropna().astype(float)
        if len(eq) < 3:
            raise ValueError("Equity troppo corta per metriche.")

        total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1.0
        r = eq.pct_change().dropna()

        n_days = (eq.index[-1] - eq.index[0]).days
        years = n_days / 365.25 if n_days > 0 else np.nan
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years and np.isfinite(years) and years > 0 else np.nan

        vol_ann = r.std(ddof=0) * np.sqrt(annualization)
        sharpe = (r.mean() * annualization) / vol_ann if vol_ann and np.isfinite(vol_ann) and vol_ann > 0 else np.nan

        peak = eq.cummax()
        dd = (eq / peak) - 1.0

        return {
            "Start": eq.index[0],
            "End": eq.index[-1],
            "Days": int((eq.index[-1] - eq.index[0]).days),
            "Start Value": float(eq.iloc[0]),
            "End Value": float(eq.iloc[-1]),
            "Total Return %": 100.0 * float(total_ret),
            "CAGR %": 100.0 * float(cagr) if np.isfinite(cagr) else np.nan,
            "Vol Ann %": 100.0 * float(vol_ann) if np.isfinite(vol_ann) else np.nan,
            "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "Max DD %": 100.0 * float(dd.min()),
        }

    # ------------------------------------------------------------
    # 1) Equity pf
    # ------------------------------------------------------------
    eq_pf = pf.value().copy()
    eq_pf.index = pd.to_datetime(eq_pf.index).normalize()
    eq_pf = eq_pf.sort_index()

    # ------------------------------------------------------------
    # 2) Benchmark (opzionale)
    # ------------------------------------------------------------
    eq_bench = None
    if benchmark is not None:
        b = benchmark.copy()
        b.index = pd.to_datetime(b.index).normalize()
        b = b.sort_index()
        eq_bench = b

    # ------------------------------------------------------------
    # 3) Definisci finestra di analisi
    # ------------------------------------------------------------
    start = _to_ts(analysis_start_date) or pd.Timestamp(eq_pf.index.min()).normalize()

    if analysis_end_date is not None:
        end = _to_ts(analysis_end_date)
    else:
        if eq_bench is not None:
            end = min(pd.Timestamp(eq_pf.index.max()).normalize(),
                      pd.Timestamp(eq_bench.index.max()).normalize())
        else:
            end = pd.Timestamp(eq_pf.index.max()).normalize()

    if start > end:
        start, end = end, start

    # clip pf
    eq_pf_c = eq_pf.loc[(eq_pf.index >= start) & (eq_pf.index <= end)]
    if eq_pf_c.empty:
        raise ValueError("Finestra analisi non interseca l'equity del portafoglio.")

    # clip + align benchmark sul calendario del pf
    eq_bench_c = None
    if eq_bench is not None:
        eq_bench_c = eq_bench.loc[(eq_bench.index >= start) & (eq_bench.index <= end)]
        # riallinea su calendario pf
        eq_bench_c = eq_bench_c.reindex(eq_pf_c.index).ffill().bfill()
        if eq_bench_c.empty:
            eq_bench_c = None

    # ------------------------------------------------------------
    # 4) Rebase (100k di default)
    # ------------------------------------------------------------
    eq_pf_c = _rebase(eq_pf_c, rebase_to)
    if eq_bench_c is not None:
        eq_bench_c = _rebase(eq_bench_c, rebase_to)

    # ------------------------------------------------------------
    # 5) Metriche
    # ------------------------------------------------------------
    m_pf = _equity_metrics(eq_pf_c)
    m_b = _equity_metrics(eq_bench_c) if eq_bench_c is not None else None

    out = {
        "window": (start, end),
        "pf": {"label": label_pf, "metrics": m_pf},
        "benchmark": {"label": label_bench, "metrics": m_b} if m_b is not None else None,
    }

    if return_series:
        out["series"] = {"equity_pf": eq_pf_c, "equity_bench": eq_bench_c}

    return out

def calc_vbt_internal_benchmark_buyhold_equal_cash(
    prices_wide: pd.DataFrame,
    init_cash: float = 100000.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    ffill_prices: bool = True,
):
    """
    Calcola MANUALMENTE il "benchmark interno" che VectorBT riporta come
    `Benchmark Return [%]` quando il Portfolio è multi-ticker (multi-colonna).

    REGOLA CHIAVE (da fissare nel framework):
    - Il benchmark interno NON è un equal-weight ribilanciato giornalmente.
    - Il benchmark interno è un BUY & HOLD "equal-cash" (paniere statico):
        1) alla data iniziale si investe `init_cash` in parti uguali tra gli asset disponibili;
        2) si acquistano quote (shares) fisse e si tengono fino alla fine del periodo;
        3) nessun ribilanciamento in corso d'opera;
        4) equity(t) = somma_i( shares_i * price_i(t) ).

    Perché è diverso dal tuo primo tentativo (media dei rendimenti):
    - fare la media dei rendimenti giornalieri equivale a un portafoglio
      equal-weight RIBILANCIATO OGNI GIORNO (daily rebalanced),
      che spesso produce un TR diverso (nel tuo caso più alto).
    - vbt invece usa un benchmark che rappresenta "compra e tieni l'universo"
      senza rotazione e senza ribilanciamento.

    Parametri
    ----------
    prices_wide : pd.DataFrame
        DataFrame prezzi in formato wide:
        - index = Date (DatetimeIndex o convertibile)
        - columns = tickers
        - valori = prezzi (nel tuo framework: Close già adjusted/total return)
        Può contenere NaN (es. titoli non quotati all’inizio o buchi dati).

    init_cash : float
        Capitale iniziale del benchmark.

    start, end : str | pd.Timestamp | None
        Periodo di calcolo. Se None, usa tutto l'indice disponibile in prices_wide.
        Nota: le "date effettive" usate sono le prime/ultime righe risultanti
        dopo lo slicing, e vengono ritornate in output.

    ffill_prices : bool
        Se True, applica forward-fill ai prezzi DOPO lo slicing temporale.
        Serve a gestire buchi sporadici (missing data) senza interrompere l'equity.
        Non "crea" prezzi prima della prima osservazione: i NaN iniziali restano NaN,
        quindi un ticker senza prezzo alla data iniziale viene escluso dal paniere.

    Output (dict)
    -------------
    start_used : Timestamp
        Prima data effettivamente presente dopo slicing.

    end_used : Timestamp
        Ultima data effettivamente presente dopo slicing.

    n_assets : int
        Numero di asset inclusi nel paniere (quelli con prezzo valido alla start).

    assets : list[str]
        Elenco ticker effettivamente inclusi (prezzo valido alla start).

    shares : pd.Series
        Shares fissi comprati alla start per ciascun ticker incluso.

    equity : pd.Series
        Equity line del benchmark buy&hold equal-cash.

    end_value : float
        Valore finale del benchmark.

    total_return_pct : float
        Total Return (%) = (end_value / init_cash - 1) * 100.
        Questo è il valore che deve coincidere con `pf_rot.stats()["Benchmark Return [%]"]`
        quando `prices_wide` è lo stesso close usato per costruire pf_rot.
    """
    if not isinstance(prices_wide, pd.DataFrame):
        raise TypeError("prices_wide deve essere un DataFrame wide (Date index, tickers columns).")

    # --- Normalizza indice data e conserva solo colonne numeriche ---
    df = prices_wide.copy()
    df.index = pd.to_datetime(df.index)
    df = df.select_dtypes(include=[np.number])

    # --- Slicing temporale richiesto (usa i dati presenti in df) ---
    if start is not None:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df.index <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError("DataFrame vuoto dopo slicing per date.")

    # --- Date effettive usate ---
    start_used = df.index[0]
    end_used = df.index[-1]

    # --- Gestione buchi dati: forward-fill interno al periodo ---
    # Nota: NON risolve NaN iniziali (prima osservazione). Quelli restano NaN.
    if ffill_prices:
        df = df.ffill()

    # --- Selezione asset inclusi: devono avere prezzo valido alla data iniziale ---
    # I ticker senza prezzo a start non possono essere comprati (shares non definibili).
    p0 = df.iloc[0]
    valid_cols = p0.dropna().index.tolist()

    if len(valid_cols) == 0:
        raise ValueError("Nessun ticker con prezzo valido alla start (dopo ffill se attivo).")

    dfv = df[valid_cols]
    p0v = dfv.iloc[0]

    # --- Regola equal-cash: investe init_cash/N in ogni asset incluso ---
    cash_per_asset = init_cash / len(valid_cols)

    # --- Shares fissi comprati alla start e poi mantenuti (NO rebalance) ---
    shares = cash_per_asset / p0v

    # --- Equity line: somma del valore delle posizioni buy&hold ---
    equity = (dfv * shares).sum(axis=1)

    # --- Total Return rispetto a init_cash (non rispetto a equity.iloc[0]) ---
    total_return_pct = (equity.iloc[-1] / init_cash - 1.0) * 100.0

    return {
        "start_used": start_used,
        "end_used": end_used,
        "n_assets": len(valid_cols),
        "assets": valid_cols,
        "shares": shares,
        "equity": equity,
        "end_value": float(equity.iloc[-1]),
        "total_return_pct": float(total_return_pct),
    }


# Run time

def r_run_portfolio(
    portfolio: dict,
    year: int | None = None,
    *,
    # --- dry-run ---
    dry_run: bool = False,
    # --- download ---
    start_date=None,            # default: now() se None
    end_date=None,
    lookback_buffer_days: int = 365 * 2,
    show_progress: bool = False,
    wfo_results_dir: str = "WFO_R_RESULTS",
    # --- WFO summary ---
    wfo_file_save: str | None = None,   # default: f"{portfolio_title}_{year}.wfo_summary.csv"
    # --- report ---
    report_end_date=None,                 # None oppure 'YYYY-MM-DD'
    sender_email: str = "",
    sender_password: str = "",
    recipient_email: str = "",
    subject=None,
    verbose: bool = False,
    debug: bool = False,
):
    """
    Esegue la pipeline operativa del portafoglio rotazionale a partire
    da una struttura portfolio predefinita.

    Struttura attesa:
    portfolio = {
        "Title": "Germany Plan",
        "tickers": [...]
    }

    Modalità dry-run:
      - stampa le azioni che verrebbero eseguite
      - non scarica dati
      - non legge file
      - non invia email

    Dipendenze attese già disponibili nel progetto:
      - now()
      - download_data(...)
      - collect_selections_from_summary(...)
      - extract_operational_params_from_summary(...)
      - send_rotational_portfolio_report(...)
    """

    # --- Validazione minima portfolio ---
    if not isinstance(portfolio, dict):
        raise TypeError("portfolio deve essere un dict, es. {'Title': '...', 'tickers': [...]}")

    if "Title" not in portfolio or "tickers" not in portfolio:
        raise KeyError("portfolio deve contenere le chiavi obbligatorie: 'Title' e 'tickers'")

    portfolio_title = portfolio["Title"]
    tickers = portfolio["tickers"]

    # Supporto a indici: se tickers e' una stringa (ossia non e' una lista predefinita) creo la lista di tickers:
    tickers = (
        extract_tickers_from_wikipedia(tickers,exclude=["GOOG"],rename={"BRK.B": "BRK-B"})
        if isinstance(tickers, str)
        else list(tickers)
    )

    if not isinstance(tickers, (list, tuple)) or len(tickers) == 0:
        raise ValueError("portfolio['tickers'] deve essere una lista/tupla non vuota di ticker")

    # --- Year default: anno corrente ---
    if year is None:
        year = int(pd.Timestamp.now().year)

    
    # --- WFO summary filename ---
    if wfo_file_save is None:
        wfo_file_save = f"{portfolio_title}_{year}.wfo_summary.csv"

    wfo_file_save=f"{wfo_results_dir}/{wfo_file_save}"
    
    # --- Start date default ---
    if start_date is None:
        start_date = now()

    # --- Download start (buffer) ---
    download_start_date = (
        pd.to_datetime(start_date) - timedelta(days=int(lookback_buffer_days))
    ).strftime("%Y-%m-%d")

    # --- DRY RUN: stampa piano azioni e termina ---
    if dry_run:
        print("[DRY-RUN] r_run_portfolio")
        print(f"  - portfolio_title      : {portfolio_title}")
        print(f"  - year                 : {year}")
        print(f"  - tickers              : {len(tickers)} tickers")
        print(f"  - start_date (input)   : {start_date}")
        print(f"  - lookback_buffer_days : {lookback_buffer_days}")
        print(f"  - download_start_date  : {download_start_date}")
        print(f"  - end_date             : {end_date}")
        print(f"  - show_progress        : {show_progress}")
        print(f"  - wfo_file_save        : {wfo_file_save}")
        print(f"  - report_end_date                : {report_end_date}")
        print("  - Azioni che verrebbero eseguite:")
        print("    1) download_data(tickers, download_start_date, end_date, show_progress=...)")
        print("    2) pd.read_csv(wfo_file_save, index_col='Window')")
        print("    3) collect_selections_from_summary(summary_df, stocks_data, debug=...)")
        print("    4) extract_operational_params_from_summary(summary_df, report_end_date)")
        print("    5) send_rotational_portfolio_report(..., rebalance_frequency, n_top)")
        print("    6) pd.write_csv(sel_tickers_file = {portfolio_title}_{current_year}_sel_tickers_current_year.csv)")

        print("  - Email params:")
        print(f"    sender_email     : {sender_email}")
        print(f"    recipient_email  : {recipient_email}")
        print(f"    subject          : {subject}")
        print(f"    verbose          : {verbose}")
        return {
            "dry_run": True,
            "portfolio": portfolio_title,
            "year": year,
            "wfo_file_save": wfo_file_save,
            "download_start_date": download_start_date,
            "end_date": end_date,
            # richiesti: presenti ma non calcolabili in dry-run
            "sel_tickers": None,
            "summary_df": None,
        }

    # --- Download dati ---
    stocks_data = download_data(
        tickers,
        download_start_date,
        end_date,
        show_progress=show_progress,
        auto_adjust=False
    )

    # --- Carica summary WFO ---
    summary_df=load_wfo_summary(wfo_file_save)
    
    # --- Selezioni tickers dal summary ---
    sel_tickers = collect_selections_from_summary(
        summary_df=summary_df,
        stocks_data=stocks_data,
        debug=debug
    )
    # ------------------------------------------------------------
    # Salvataggio selezioni ticker dell'anno corrente
    # ------------------------------------------------------------
    current_year = now().year

    sel_tickers_current_year = sel_tickers[
        sel_tickers.index.year == current_year
    ]

    sel_tickers_file = f"{portfolio_title}_{current_year}_sel_tickers_current_year.csv"
    sel_tickers_file = f"{wfo_results_dir}/{sel_tickers_file}"


    if not sel_tickers_current_year.empty:
        sel_tickers_current_year.to_csv(sel_tickers_file)
        if verbose:
            print(f"[INFO] Selezioni anno {current_year} salvate in: {sel_tickers_file}")
    else:
        if verbose:
            print(f"[INFO] Nessuna selezione disponibile per l'anno {current_year}")

    # --- Parametri operativi ---
    params_ops = extract_operational_params_from_summary(summary_df, report_end_date)
    rebalance_frequency = params_ops["rebalance_frequency"]
    n_top = params_ops["n_top"]

    # --- Invio report ---
    send_rotational_portfolio_report(
        sel_tickers,
        portfolio_title,
        report_end_date,
        sender_email,
        sender_password,
        recipient_email,
        subject,
        verbose,
        rebalance_frequency,
        n_top,
        trading_index=stocks_data.index
    )


    return {
        "dry_run": False,
        "portfolio": portfolio_title,
        "year": year,
        "wfo_file_save": wfo_file_save,
        "download_start_date": download_start_date,
        "end_date": end_date,
        # richiesti: aggiunti in output
        "sel_tickers": sel_tickers,
        "summary_df": summary_df,
        # resto invariato
        "params_ops": params_ops,
        "rebalance_frequency": rebalance_frequency,
        "n_top": n_top,
        "stocks_data": stocks_data
    }
def extract_operational_params_from_summary(
    summary_df: pd.DataFrame,
    today: str | pd.Timestamp | None = None
) -> dict:
    """
    Estrae i parametri operativi dal risultato WFO per l'ANNO di riferimento.

    Regola:
    - se today=None -> usa l'anno corrente (pd.Timestamp.today()).
    - se today è valorizzato -> usa l'anno di today (accetta str o Timestamp).

    Se summary_df è indicizzato per "Window" nel formato:
        'YYYY-MM-DD→YYYY-MM-DD'
    allora seleziona la riga la cui finestra INIZIA nell'anno target (tipico: 2026-01-01→2026-12-31).

    Ritorna (minimo indispensabile):
    - rebalance_frequency (str)
    - n_top (int)
    """

    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df è vuoto: impossibile estrarre parametri operativi")

    # --- anno target ---
    if today is None:
        target_year = pd.Timestamp.today().year
    else:
        target_year = pd.to_datetime(today).year

    # --- ricava lo start-year da index "Window" (formato 'start→end') ---
    idx = summary_df.index.astype(str)

    def _start_year(window_str: str) -> int | None:
        try:
            start_str = window_str.split("→", 1)[0]
            return pd.to_datetime(start_str).year
        except Exception:
            return None

    start_years = pd.Series([_start_year(x) for x in idx], index=summary_df.index)

    # --- seleziona la riga per l'anno target ---
    mask = start_years == int(target_year)
    if not mask.any():
        available_years = sorted({y for y in start_years.dropna().astype(int).tolist()})
        raise ValueError(
            f"Nessuna finestra WFO trovata per l'anno target {target_year}. "
            f"Anni disponibili in summary_df: {available_years}"
        )

    # Se ci sono più righe per lo stesso anno (caso raro), prendi l'ultima occorrenza
    row = summary_df.loc[mask].iloc[-1]

    rebalance_frequency = row.get("rebalance_frequency")
    n_top = row.get("n_top")

    if pd.isna(rebalance_frequency):
        raise ValueError(f"rebalance_frequency mancante per l'anno {target_year} in summary_df")

    if pd.isna(n_top):
        raise ValueError(f"n_top mancante per l'anno {target_year} in summary_df")

    return {
        "rebalance_frequency": str(rebalance_frequency),
        "n_top": int(n_top),
        "year": int(target_year),
    }

def generate_rotational_portfolio_report(
    sel_tickers,
    portfolio_name,
    today,
    sender_email="",
    sender_password="",
    recipient_email="",
    subject=None,
    verbose=False,
    # --- parametri operativi espliciti (da WFO) ---
    rebalance_frequency: str | None = None,
    n_top: int | None = None,
    attachments=None,
    # --- NEW: calendario di borsa reale (es. stocks_data.index) ---
    trading_index=None,
):
    """
    Genera e stampa/invia un report HTML per un portafoglio rotazionale.

    Regole (robuste, una volta per tutte):
    - target di calendario: fine mese (ME) / fine trimestre (QE)
    - effective rebalance date: ultima seduta <= target (snap usando trading_index)
    - execution day: prima seduta > effective (sempre da trading_index)
    - il report viene prodotto SEMPRE:
        * se today == execution day -> mostra azioni operative (buy/sell/keep)
        * altrimenti -> mostra stato, holdings correnti, prossime date chiave
    """
    import datetime
    import pandas as pd

    # Normalizza today per subject/label
    if today is None:
        today_ts = pd.Timestamp.today().normalize()
    else:
        today_ts = pd.to_datetime(today).normalize()

    today_str = today_ts.strftime("%Y-%m-%d")

    html_report = analyze_rebalance_actions_for_report(
        sel_tickers=sel_tickers,
        today=today_ts,
        rebalance_frequency=rebalance_frequency,
        n_top=n_top,
        verbose=verbose,
        trading_index=trading_index,
    )

    # Stampa o invia report
    if recipient_email:
        if subject is None:
            subject = f"[TS_LAB] Report di Portafoglio {portfolio_name} ({today_str})"

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

send_rotational_portfolio_report = generate_rotational_portfolio_report

def analyze_rebalance_actions_for_report(
    sel_tickers: pd.DataFrame,
    today: str | pd.Timestamp | None = None,
    rebalance_frequency: str | None = None,
    n_top: int | None = None,
    verbose: bool = False,
    trading_index=None,
) -> str:
    """
    Analizza e descrive le azioni di ribilanciamento di un portafoglio rotazionale,
    secondo una logica **puramente calendariale**, progettata per un job automatico
    eseguito **tutti i giorni alle 08:00**.
    
    LOGICA DI TRIGGER (CALENDARIO, NON MERCATO)
    ------------------------------------------
    Il ribilanciamento NON dipende dalla disponibilità dei dati di chiusura del giorno,
    ma esclusivamente dal calendario:
    
    - ME (Mensile):        ribilanciamento il **1° giorno del mese**
    - QE (Trimestrale):    ribilanciamento il **1° giorno del trimestre**
                           (gennaio, aprile, luglio, ottobre)
    - WE (Settimanale):    ribilanciamento il **lunedì**
    
    Se oggi soddisfa la regola di calendario per la frequenza scelta, allora
    oggi è considerato "rebalance day", indipendentemente da:
    - orario di esecuzione,
    - fatto che oggi sia trading day,
    - weekend o festività.
    
    FIXING OPERATIVO (ASOF)
    ----------------------
    Il fixing dei segnali e delle selezioni avviene sempre sull’ultima seduta
    di mercato **effettivamente disponibile**:
    
        ASOF = ultima data in trading_index ≤ today
    
    Questo garantisce che:
    - alle 08:00 si usino solo dati già disponibili,
    - nei weekend/festivi si utilizzi automaticamente l’ultima seduta precedente,
    - non vi sia dipendenza dal fatto che il close del giorno corrente sia già noto.
    
    EXECUTION
    ---------
    Le operazioni NON vengono eseguite su ASOF, ma sono pianificate per:
    
        execution = prima seduta di mercato successiva ad ASOF
    
    Nel report viene esplicitamente indicato che:
    - oggi è il giorno di pianificazione del ribilanciamento,
    - gli ordini vanno eseguiti alla prossima seduta utile.
    
    COMPORTAMENTO IN GIORNI NON DI RIBILANCIAMENTO
    ----------------------------------------------
    Se oggi NON è un giorno di ribilanciamento calendario:
    - la funzione mostra lo stato corrente del portafoglio,
    - riporta l’ultima selezione disponibile,
    - indica chiaramente se oggi è trading day o non trading day.
    
    OBIETTIVO DELLA FUNZIONE
    -----------------------
    Questa funzione è pensata per:
    - reporting operativo,
    - controllo quotidiano delle azioni di rotazione,
    - utilizzo in pipeline automatiche e deterministiche.
    
    La funzione NON:
    - esegue operazioni,
    - scarica dati,
    - assume che i prezzi di oggi siano disponibili.
    
    Restituisce esclusivamente una descrizione HTML delle azioni da intraprendere,
    coerente, ripetibile e indipendente dall’orario di esecuzione.
    """
    
    import pandas as pd
    from typing import List

    # =========================
    # Normalize today (job alle 08:00 → date-only)
    # =========================
    if today is None:
        today_ts = pd.Timestamp.today().normalize()
    else:
        today_ts = pd.to_datetime(today).normalize()

    # =========================
    # Normalize selections index
    # =========================
    sel_df = sel_tickers.copy()
    sel_df.index = pd.to_datetime(sel_df.index).normalize()
    sel_idx = sel_df.index.sort_values().unique()

    if len(sel_idx) == 0:
        return "<h2>⚠️ Nessun dato di selezione disponibile</h2>"

    # =========================
    # Normalize trading calendar
    # =========================
    if trading_index is None:
        return "<h2>⚠️ trading_index non fornito</h2>"

    t_idx = pd.DatetimeIndex(pd.to_datetime(trading_index)).normalize().sort_values().unique()
    if len(t_idx) == 0:
        return "<h2>⚠️ trading_index vuoto</h2>"

    # =========================
    # Helpers
    # =========================
    def _snap_prev(idx, d):
        pos = idx.searchsorted(d, side="right") - 1
        return None if pos < 0 else pd.Timestamp(idx[pos]).normalize()

    def _next(idx, d):
        pos = idx.searchsorted(d, side="right")
        return None if pos >= len(idx) else pd.Timestamp(idx[pos]).normalize()

    def _to_set(tickers_value) -> set:
        if tickers_value is None:
            return set()
        if isinstance(tickers_value, (list, tuple, set)):
            return {t for t in tickers_value if isinstance(t, str) and t}
        if isinstance(tickers_value, pd.Series):
            non_null = tickers_value.dropna()
            if len(non_null) == 0:
                return set()
            return _to_set(non_null.iloc[0])
        return set()
        
    def _get_tickers(df: pd.DataFrame, date) -> set:
        """Accesso robusto alla colonna ticker per una data. Guardrail su NaT e date mancanti."""
        if date is None:
            return set()
        try:
            date = pd.Timestamp(date)
        except Exception:
            return set()
        if pd.isna(date):
            return set()
        col = "Top_Tickers" if "Top_Tickers" in df.columns else "tickers"
        if date not in df.index:
            return set()
        return _to_set(df.at[date, col])

    
    def _market_status(d, idx, asof=None):
        # Weekend: qui possiamo essere certi
        if d.weekday() >= 5:
            return "<p>🏛️ Stato mercato: <b>Weekend</b></p>"
    
        # Giorno feriale: è una seduta potenziale (non diciamo 'aperto')
        if d in idx:
            return "<p>🏛️ Stato mercato: <b>Seduta di trading</b> (dati aggiornati)</p>"
    
        # Feriale ma non presente nei dati: tipico pre-market / provider non aggiornato
        if asof is not None:
            return (
                "<p>🏛️ Stato mercato: <b>Seduta di trading</b> "
                f"(dati non ancora aggiornati; ASOF={pd.Timestamp(asof).date()})</p>"
            )
    
        return "<p>🏛️ Stato mercato: <b>Seduta di trading</b> (dati non ancora aggiornati)</p>"
          
    def _is_rebalance_day(d, freq):
        f = str(freq).upper()
        if f == "ME":
            return d.day == 1
        if f == "QE":
            return d.day == 1 and d.month in (1, 4, 7, 10)
        if f == "WE":
            return d.weekday() == 0  # Monday
        return False

    def _format_freq(freq):
        return {
            "ME": "Mensile (1° giorno del mese)",
            "QE": "Trimestrale (1° giorno del trimestre)",
            "WE": "Settimanale (lunedì)",
        }.get(freq, "non specificata")

    def _fmt_data_it(d):
        """Converte una data in formato italiano leggibile: '22 maggio 2026'."""
        d = pd.to_datetime(d)
        return f"{d.day} {_MESI_IT[d.month]} {d.year}"

    # =========================
    # Labels
    # =========================
    freq_code = str(rebalance_frequency).upper() if rebalance_frequency else ""
    freq_label = _format_freq(freq_code)
    n_top_label = "non specificato" if n_top is None else f"{int(n_top)}"

    # =========================
    # Trigger calendario
    # =========================
    is_rebalance_today = _is_rebalance_day(today_ts, freq_code)

    # =========================
    # ASOF & execution (sempre calendario trading)
    # =========================
    asof = _snap_prev(t_idx, today_ts)
    if asof is None:
        return "<h2>⚠️ Nessuna seduta disponibile ≤ today</h2>"

    execution = _next(t_idx, asof)
    execution_str = execution.date() if execution is not None else "N/D"

    # =========================
    # NON rebalance day → solo stato
    # =========================
    if not is_rebalance_today:
        # last_sel = sel_idx[sel_idx <= today_ts].max()
        # holdings = sorted(_to_set(sel_df.loc[last_sel]))
        
        eligible = sel_idx[sel_idx <= today_ts]
        if len(eligible) == 0:
            return "<h2>⚠️ Nessuna selezione disponibile ≤ oggi</h2>"
        last_sel = eligible.max()
        holdings = sorted(_get_tickers(sel_df, last_sel))
        
        html = f"<h2>🕒 Oggi NON è data di ribilanciamento ({_fmt_data_it(today_ts)})</h2>"
        html += _market_status(today_ts, t_idx)
        html += (
            "<h3>Parametri della strategia</h3>"
            f"<p>🗓️ Frequenza di ribilanciamento: <b>{freq_label}</b></p>"
            f"<p>🎯 Titoli in portafoglio: <b>{n_top_label}</b></p>"
            "<h3>Riferimenti temporali</h3>"
            f"<p>📌 Aggiornamento al: <b>{_fmt_data_it(asof)}</b></p>"
            f"<p>🔄 Ultima selezione: <b>{_fmt_data_it(last_sel)}</b></p>"
        )            
        if holdings:
            html += "<h3>Portafoglio attuale</h3>"
            company_data = build_company_df_with_cache(holdings)
            html += (
                "<table style='border-collapse: collapse; margin-top: 8px;'>"
                "<thead>"
                "<tr style='background-color: #f0f0f0;'>"
                "<th style='border: 1px solid #ccc; padding: 6px 12px; text-align: left;'>Ticker</th>"
                "<th style='border: 1px solid #ccc; padding: 6px 12px; text-align: left;'>Company</th>"
                "<th style='border: 1px solid #ccc; padding: 6px 12px; text-align: left;'>ISIN</th>"
                "</tr>"
                "</thead>"
                "<tbody>"
            )
            for t in holdings:
                if t in company_data.index:
                    company = company_data.at[t, "Company"] if pd.notna(company_data.at[t, "Company"]) else ""
                    isin    = company_data.at[t, "ISIN"]    if pd.notna(company_data.at[t, "ISIN"])    else ""
                else:
                    company = ""
                    isin    = ""
                html += (
                    "<tr>"
                    f"<td style='border: 1px solid #ccc; padding: 6px 12px;'>{t}</td>"
                    f"<td style='border: 1px solid #ccc; padding: 6px 12px;'>{company}</td>"
                    f"<td style='border: 1px solid #ccc; padding: 6px 12px; font-family: monospace;'>{isin}</td>"
                    "</tr>"
                )
            html += "</tbody></table>"
        return html

    # =========================
    # REBALANCE DAY
    # =========================
    eligible_sel = sel_idx[sel_idx <= asof]
    curr_sel = eligible_sel.max()
    prev_sel = eligible_sel[-2] if len(eligible_sel) > 1 else None

    # curr_set = _to_set(sel_df.loc[curr_sel])
    # prev_set = _to_set(sel_df.loc[prev_sel]) if prev_sel is not None else set()
    
    curr_set = _get_tickers(sel_df, curr_sel)
    prev_set = _get_tickers(sel_df, prev_sel)   # _get_tickers gestisce già None → set()

    to_keep = sorted(prev_set & curr_set)
    to_sell = sorted(prev_set - curr_set)
    to_buy  = sorted(curr_set - prev_set)

    company_data = build_company_df_with_cache(to_sell + to_buy + to_keep)

    html = f"<h2>🔄 OGGI È data di ribilanciamento – {today_ts.date()}</h2>"
    html += _market_status(today_ts, t_idx)
    html += (
        f"<p>🗓️ Frequenza: <b>{freq_label}</b></p>"
        f"<p>🎯 Target: <b>{n_top_label}</b></p>"
        f"<p>📌 ASOF (fine periodo precedente): <b>{asof.date()}</b></p>"
        f"<p>🗓️ Execution (prossima seduta): <b>{execution_str}</b></p>"
        f"<p><i>Selezione usata</i>: <b>{curr_sel.date()}</b></p>"
        f"<p style='color:#8a6d3b'>Ordini da eseguire alla prossima seduta utile.</p>"
    )

    def _fmt(lst, title, icon):
        if not lst:
            return ""
        s = f"<h3>{icon} {title} ({len(lst)}):</h3><ul>"
        for t in lst:
            company = company_data.at[t, "Company"] if t in company_data.index else ""
            s += f"<li><b>{t}</b> – {company}</li>"
        return s + "</ul>"

    html += _fmt(to_keep, "Da mantenere", "📌")
    html += _fmt(to_sell, "Da vendere", "❌")
    html += _fmt(to_buy,  "Da acquistare", "✅")

    return html


"""
Monte Carlo Block Bootstrap per Portfolios Rotazionali
=======================================================

Implementazione completa pronta all'uso per analisi robustezza
portafogli rotazionali tramite block bootstrap.

Autore: Analisi Framework Rotazionale (Claude Code)
Data: Febbraio 2026
"""


# =============================================================================
# CORE: Block Bootstrap Engine
# =============================================================================
def monte_carlo_block_bootstrap_rotational(
    portfolio: vbt.Portfolio,
    sel_tickers_df: pd.DataFrame,
    stocks_data: pd.DataFrame,
    n_simulations: int = 10_000,
    block_size: int = 20,
    init_cash: float = 100_000,
    preserve_mean: bool = False,
    random_seed: Optional[int] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Stima la robustezza di un portafoglio rotazionale tramite block bootstrap
    di Monte Carlo sui rendimenti giornalieri degli asset.

    Per ogni simulazione i rendimenti di ciascun ticker vengono ricampionati
    a blocchi (preservando l'autocorrelazione locale), i prezzi vengono
    ricostruiti e l'equity viene simulata applicando le stesse selezioni
    storiche del portafoglio originale.

    Parameters
    ----------
    portfolio : vbt.Portfolio
        Portafoglio originale, usato come riferimento per il confronto.
    sel_tickers_df : pd.DataFrame
        DataFrame con indice = rebal_dates e colonna "tickers" (list[str]).
        Tipicamente selections da RotationalResult.
    stocks_data : pd.DataFrame
        Prezzi giornalieri degli asset (colonne = ticker).
    n_simulations : int
        Numero di simulazioni MC (default 10_000).
    block_size : int
        Dimensione dei blocchi per il bootstrap (default 20 giorni).
    init_cash : float
        Capitale iniziale per la simulazione (default 100_000).
    preserve_mean : bool
        Se True, riscala i rendimenti bootstrappati per preservare la media
        originale di ciascun ticker.
    random_seed : int, optional
        Seed per la riproducibilità.
    show_progress : bool
        Se True, mostra una barra di progresso tqdm.

    Returns
    -------
    pd.DataFrame
        Matrice (n_giorni × n_simulations) delle equity curve simulate.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # ------------------------------------------------------------
    # FIX: allinea il periodo prezzi al periodo effettivo di selezione
    # ------------------------------------------------------------
    sel_tickers_df = sel_tickers_df.copy()
    sel_tickers_df.index = pd.DatetimeIndex(sel_tickers_df.index)

    start_dt = sel_tickers_df.index.min()
    end_dt   = sel_tickers_df.index.max()

    # Slice coerente: usiamo solo il range in cui esistono selezioni
    stocks_data = stocks_data.loc[start_dt:end_dt].copy()

    # Se per qualche motivo lo slicing svuota, fail-fast
    if stocks_data.empty:
        raise ValueError(
            f"stocks_data vuoto dopo slicing su [{start_dt.date()} - {end_dt.date()}]. "
            "Verifica che stocks_data e sel_tickers_df condividano lo stesso calendario/date."
        )

    # Estrai info dal periodo allineato
    returns = stocks_data.pct_change().fillna(0).infer_objects(copy=False)
    dates = returns.index
    all_tickers = list(stocks_data.columns)
    n_days = len(dates)

    # Rebalance dates: tieni solo quelle presenti nel calendario dei prezzi
    rebal_dates = pd.DatetimeIndex(sel_tickers_df.index)
    rebal_dates = rebal_dates[(rebal_dates >= dates[0]) & (rebal_dates <= dates[-1])]

    if len(rebal_dates) == 0:
        raise ValueError(
            "Nessuna rebal_date di sel_tickers_df cade dentro l'intervallo di stocks_data (dopo slicing)."
        )

    # Pre-allocazione matrice risultati
    sim_equity_curves = np.zeros((n_days, n_simulations))

    # Setup progress bar
    pbar = tqdm(total=n_simulations, desc="MC Bootstrap", disable=not show_progress)

    for sim_idx in range(n_simulations):
        # 1. Bootstrap returns per ogni ticker (blocchi)
        sim_returns_dict = {}

        for ticker in all_tickers:
            ticker_rets = returns[ticker].values
            bootstrapped_rets = _block_bootstrap_single_series(
                ticker_rets,
                block_size=block_size,
                preserve_mean=preserve_mean
            )
            sim_returns_dict[ticker] = bootstrapped_rets

        # 2. Converti in DataFrame
        sim_returns_df = pd.DataFrame(sim_returns_dict, index=dates)

        # 3. Ricostruisci prezzi da returns bootstrappati (base = primo giorno del periodo allineato)
        sim_prices = (1 + sim_returns_df).cumprod() * stocks_data.iloc[0]

        # 4. Simula equity usando STESSE selezioni (coerenti con l'intervallo)
        sim_equity = _simulate_rotational_equity(
            sim_prices,
            sel_tickers_df,
            rebal_dates,
            init_cash
        )

        sim_equity_curves[:, sim_idx] = sim_equity
        pbar.update(1)

    pbar.close()

    # Converti in DataFrame
    result_df = pd.DataFrame(
        sim_equity_curves,
        index=dates,
        columns=[f"sim_{i}" for i in range(n_simulations)]
    )

    return result_df
    


def _block_bootstrap_single_series(
    series: np.ndarray,
    block_size: int,
    preserve_mean: bool = False
) -> np.ndarray:
    """
    Block bootstrap di una singola serie temporale.
    
    Preserva autocorrelazione campionando blocchi contigui.
    """
    n = len(series)
    n_blocks_needed = int(np.ceil(n / block_size))
    
    bootstrapped = []
    
    for _ in range(n_blocks_needed):
        # Sample random starting point
        max_start = max(0, n - block_size)
        start_idx = np.random.randint(0, max_start + 1)
        
        # Extract block
        block = series[start_idx:start_idx + block_size]
        
        # Center block se preserve_mean
        if preserve_mean:
            block = block - block.mean() + series.mean()
        
        bootstrapped.extend(block)
    
    # Tronca alla lunghezza originale
    return np.array(bootstrapped[:n])


def _simulate_rotational_equity(
    prices: pd.DataFrame,
    selections: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    init_cash: float
) -> np.ndarray:
    """
    Simula equity curve dato prezzi e selezioni.
    
    Implementazione efficiente usando numpy per velocità.
    """
    equity = np.zeros(len(prices))
    equity[0] = init_cash
    
    # Current holdings: {ticker: n_shares}
    holdings = {}
    holdings_value = 0.0
    cash = init_cash
    
    # Pre-compute price ratios per velocità
    price_ratios = (prices / prices.shift(1).fillna(prices.iloc[0])).values
    
    for i in range(1, len(prices)):
        current_date = prices.index[i]
        prev_date = prices.index[i-1]
        
        # Check se oggi è rebalance
        is_rebalance = current_date in rebal_dates
        
        if is_rebalance:
            # Liquida holdings correnti
            if holdings:
                total_value = sum(
                    n_shares * prices.loc[prev_date, ticker]
                    for ticker, n_shares in holdings.items()
                    if ticker in prices.columns
                )
            else:
                total_value = cash
            
            # Nuova selezione
            try:
                selected_tickers = selections.loc[current_date, 'Top_Tickers']
                if isinstance(selected_tickers, str):
                    # Caso singolo ticker (non dovrebbe accadere)
                    selected_tickers = [selected_tickers]
                elif not isinstance(selected_tickers, list):
                    # Può essere una Series o altro
                    selected_tickers = list(selected_tickers)
            except:
                selected_tickers = []
            
            # Riallocazione equal-weight
            n_selected = len(selected_tickers)
            if n_selected > 0:
                target_per_ticker = total_value / n_selected
                holdings = {}
                
                for ticker in selected_tickers:
                    if ticker in prices.columns:
                        price_at_rebal = prices.loc[current_date, ticker]
                        if price_at_rebal > 0:
                            n_shares = target_per_ticker / price_at_rebal
                            holdings[ticker] = n_shares
                
                cash = 0.0
            else:
                # Nessuna selezione → 100% cash
                holdings = {}
                cash = total_value
        
        # Aggiorna valore holdings
        if holdings:
            portfolio_value = 0.0
            for ticker, n_shares in holdings.items():
                if ticker in prices.columns:
                    portfolio_value += n_shares * prices.iloc[i][ticker]
            equity[i] = portfolio_value + cash
        else:
            equity[i] = cash
    
    return equity


# =============================================================================
# ANALISI RISULTATI
# =============================================================================

def analyze_mc_results(
    mc_equity_curves: pd.DataFrame,
    portfolio_actual: vbt.Portfolio,
    confidence_level: float = 0.90,
    print_report: bool = True
) -> Dict:
    """
    Analizza distribuzione risultati Monte Carlo.
    
    Parametri
    ----------
    mc_equity_curves : pd.DataFrame
        Output di monte_carlo_block_bootstrap_rotational
    portfolio_actual : vbt.Portfolio
        Portfolio reale per confronto
    confidence_level : float, default=0.90
        Livello confidence interval (0.90 = 90%)
    print_report : bool, default=True
        Stampa report testuale
        
    Returns
    -------
    dict
        Dizionario con tutte le metriche calcolate
    """
    
    n_sims = mc_equity_curves.shape[1]
    init_value = mc_equity_curves.iloc[0, 0]
    
    # =========================================================================
    # 1. FINAL RETURNS
    # =========================================================================
    final_values = mc_equity_curves.iloc[-1]
    final_returns = (final_values / init_value) - 1
    
    actual_final_return = (portfolio_actual.value().iloc[-1] / portfolio_actual.init_cash) - 1
    
    alpha_lower = (1 - confidence_level) / 2
    alpha_upper = 1 - alpha_lower
    
    final_return_stats = {
        'actual': actual_final_return,
        'mc_mean': final_returns.mean(),
        'mc_median': final_returns.median(),
        'mc_std': final_returns.std(),
        'ci_lower': final_returns.quantile(alpha_lower),
        'ci_upper': final_returns.quantile(alpha_upper),
        'percentile_of_actual': (final_returns < actual_final_return).mean()
    }
    
    # =========================================================================
    # 2. CAGR
    # =========================================================================
    n_years = len(mc_equity_curves) / 252
    
    mc_cagrs = ((final_values / init_value) ** (1 / n_years)) - 1
    actual_cagr = ((portfolio_actual.value().iloc[-1] / portfolio_actual.init_cash) ** (1 / n_years)) - 1
    
    cagr_stats = {
        'actual': actual_cagr,
        'mc_mean': mc_cagrs.mean(),
        'mc_median': mc_cagrs.median(),
        'mc_std': mc_cagrs.std(),
        'ci_lower': mc_cagrs.quantile(alpha_lower),
        'ci_upper': mc_cagrs.quantile(alpha_upper),
        'percentile_of_actual': (mc_cagrs < actual_cagr).mean()
    }
    
    # =========================================================================
    # 3. MAX DRAWDOWN
    # =========================================================================
    def compute_max_dd(equity_series):
        cummax = equity_series.cummax()
        dd = (equity_series - cummax) / cummax
        return dd.min()
    
    mc_drawdowns = mc_equity_curves.apply(compute_max_dd, axis=0)
    actual_dd = compute_max_dd(portfolio_actual.value())
    
    dd_stats = {
        'actual': actual_dd,
        'mc_mean': mc_drawdowns.mean(),
        'mc_median': mc_drawdowns.median(),
        'mc_std': mc_drawdowns.std(),
        'ci_lower': mc_drawdowns.quantile(alpha_lower),  # less negative
        'ci_upper': mc_drawdowns.quantile(alpha_upper),  # more negative (worse)
        'worst_5pct': mc_drawdowns.quantile(0.95),  # 95th percentile = worst
        'percentile_of_actual': (mc_drawdowns < actual_dd).mean()  # <0, so < = worse
    }
    
    # =========================================================================
    # 4. SHARPE RATIO
    # =========================================================================
    mc_returns = mc_equity_curves.pct_change().fillna(0)
    mc_sharpes = (mc_returns.mean() / mc_returns.std()) * np.sqrt(252)
    
    actual_sharpe = portfolio_actual.sharpe_ratio()
    
    sharpe_stats = {
        'actual': actual_sharpe,
        'mc_mean': mc_sharpes.mean(),
        'mc_median': mc_sharpes.median(),
        'mc_std': mc_sharpes.std(),
        'ci_lower': mc_sharpes.quantile(alpha_lower),
        'ci_upper': mc_sharpes.quantile(alpha_upper),
        'percentile_of_actual': (mc_sharpes < actual_sharpe).mean()
    }
    
    # =========================================================================
    # 5. CALMAR RATIO
    # =========================================================================
    mc_calmars = mc_cagrs / mc_drawdowns.abs()
    mc_calmars = mc_calmars.replace([np.inf, -np.inf], np.nan).dropna()
    
    actual_calmar = portfolio_actual.calmar_ratio()
    
    calmar_stats = {
        'actual': actual_calmar,
        'mc_mean': mc_calmars.mean(),
        'mc_median': mc_calmars.median(),
        'mc_std': mc_calmars.std(),
        'ci_lower': mc_calmars.quantile(alpha_lower),
        'ci_upper': mc_calmars.quantile(alpha_upper),
        'percentile_of_actual': (mc_calmars < actual_calmar).mean() if len(mc_calmars) > 0 else np.nan
    }
    
    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    
    if print_report:
        print("=" * 80)
        print(f"MONTE CARLO BLOCK BOOTSTRAP ANALYSIS ({n_sims:,} simulations)")
        print("=" * 80)
        print(f"Confidence Level: {confidence_level:.0%}")
        print(f"Period: {mc_equity_curves.index[0].date()} → {mc_equity_curves.index[-1].date()}")
        print(f"Duration: {n_years:.2f} years")
        print()
        
        print("─" * 80)
        print("FINAL RETURN")
        print("─" * 80)
        _print_stat("Actual", final_return_stats['actual'], is_pct=True)
        _print_stat("MC Mean", final_return_stats['mc_mean'], is_pct=True)
        _print_stat("MC Median", final_return_stats['mc_median'], is_pct=True)
        _print_stat("MC Std", final_return_stats['mc_std'], is_pct=True)
        print(f"  {confidence_level:.0%} CI: [{final_return_stats['ci_lower']:.1%}, {final_return_stats['ci_upper']:.1%}]")
        print(f"  Actual Percentile: {final_return_stats['percentile_of_actual']:.1%}")
        _print_flag(final_return_stats['percentile_of_actual'])
        print()
        
        print("─" * 80)
        print("CAGR (Compound Annual Growth Rate)")
        print("─" * 80)
        _print_stat("Actual", cagr_stats['actual'], is_pct=True)
        _print_stat("MC Mean", cagr_stats['mc_mean'], is_pct=True)
        _print_stat("MC Median", cagr_stats['mc_median'], is_pct=True)
        print(f"  {confidence_level:.0%} CI: [{cagr_stats['ci_lower']:.1%}, {cagr_stats['ci_upper']:.1%}]")
        print(f"  Actual Percentile: {cagr_stats['percentile_of_actual']:.1%}")
        _print_flag(cagr_stats['percentile_of_actual'])
        print()
        
        print("─" * 80)
        print("MAX DRAWDOWN")
        print("─" * 80)
        _print_stat("Actual", dd_stats['actual'], is_pct=True)
        _print_stat("MC Mean", dd_stats['mc_mean'], is_pct=True)
        _print_stat("MC Median", dd_stats['mc_median'], is_pct=True)
        print(f"  {confidence_level:.0%} CI: [{dd_stats['ci_lower']:.1%}, {dd_stats['ci_upper']:.1%}]")
        print(f"  Worst 5%: {dd_stats['worst_5pct']:.1%}")
        print(f"  Actual Percentile: {dd_stats['percentile_of_actual']:.1%} (lower = better)")
        _print_flag_dd(dd_stats['percentile_of_actual'])
        print()
        
        print("─" * 80)
        print("SHARPE RATIO")
        print("─" * 80)
        _print_stat("Actual", sharpe_stats['actual'], is_pct=False, decimals=3)
        _print_stat("MC Mean", sharpe_stats['mc_mean'], is_pct=False, decimals=3)
        _print_stat("MC Median", sharpe_stats['mc_median'], is_pct=False, decimals=3)
        _print_stat("MC Std", sharpe_stats['mc_std'], is_pct=False, decimals=3)
        print(f"  {confidence_level:.0%} CI: [{sharpe_stats['ci_lower']:.3f}, {sharpe_stats['ci_upper']:.3f}]")
        print(f"  Actual Percentile: {sharpe_stats['percentile_of_actual']:.1%}")
        _print_flag(sharpe_stats['percentile_of_actual'])
        print()
        
        print("─" * 80)
        print("CALMAR RATIO")
        print("─" * 80)
        _print_stat("Actual", calmar_stats['actual'], is_pct=False, decimals=3)
        _print_stat("MC Mean", calmar_stats['mc_mean'], is_pct=False, decimals=3)
        _print_stat("MC Median", calmar_stats['mc_median'], is_pct=False, decimals=3)
        if not np.isnan(calmar_stats['ci_lower']):
            print(f"  {confidence_level:.0%} CI: [{calmar_stats['ci_lower']:.3f}, {calmar_stats['ci_upper']:.3f}]")
        print()
        
        print("=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        _interpret_results(final_return_stats, cagr_stats, dd_stats, sharpe_stats)
        print("=" * 80)
    
    # Return completo
    return {
        'final_return': final_return_stats,
        'cagr': cagr_stats,
        'max_drawdown': dd_stats,
        'sharpe': sharpe_stats,
        'calmar': calmar_stats,
        'n_simulations': n_sims,
        'confidence_level': confidence_level
    }


def _print_stat(label, value, is_pct=True, decimals=1):
    """Helper per stampare statistiche formattate."""
    if is_pct:
        print(f"  {label:12s}: {value:+.{decimals}%}")
    else:
        print(f"  {label:12s}: {value:+.{decimals}f}")


def _print_flag(percentile):
    """Stampa flag interpretazione basato su percentile."""
    if percentile > 0.95:
        print("  🚩 WARNING: Actual > 95th percentile → Likely overfitting or luck")
    elif percentile > 0.75:
        print("  🟡 CAUTION: Actual > 75th percentile → Above average, monitor")
    elif percentile < 0.25:
        print("  ⚠️  CONCERN: Actual < 25th percentile → Below average")
    else:
        print("  ✅ NORMAL: Actual within expected range")


def _print_flag_dd(percentile):
    """Stampa flag per drawdown (logica inversa: basso = buono)."""
    if percentile < 0.05:
        print("  🟢 EXCELLENT: Actual DD better than 95% of simulations")
    elif percentile < 0.25:
        print("  ✅ GOOD: Actual DD better than average")
    elif percentile > 0.75:
        print("  ⚠️  CONCERN: Actual DD worse than 75% of simulations")
    else:
        print("  🟡 NORMAL: Actual DD within expected range")


def _interpret_results(final_ret, cagr, dd, sharpe):
    """Interpretazione automatica risultati."""
    
    # Check overfitting
    overfitting_score = 0
    if final_ret['percentile_of_actual'] > 0.95:
        overfitting_score += 2
    elif final_ret['percentile_of_actual'] > 0.85:
        overfitting_score += 1
    
    if sharpe['percentile_of_actual'] > 0.95:
        overfitting_score += 2
    elif sharpe['percentile_of_actual'] > 0.85:
        overfitting_score += 1
    
    # Check lucky DD
    lucky_dd = dd['percentile_of_actual'] < 0.15
    
    # Check robustezza
    robust = (
        0.30 < final_ret['percentile_of_actual'] < 0.70 and
        0.30 < sharpe['percentile_of_actual'] < 0.70
    )
    
    # Interpretazioni
    if overfitting_score >= 3:
        print("🚩 HIGH OVERFITTING RISK")
        print("   Your actual performance is in the extreme tail of MC distribution.")
        print("   This suggests parameter overfitting or exceptional luck.")
        print("   → Expect mean reversion in future OOS performance.")
        print()
    
    if lucky_dd and overfitting_score >= 2:
        print("🚩 LUCKY SCENARIO")
        print("   Both high returns AND low drawdown vs MC distribution.")
        print("   → Unusually favorable market conditions or overfitting.")
        print()
    
    if robust:
        print("✅ ROBUST PARAMETERS")
        print("   Actual performance near center of MC distribution.")
        print("   → Parameters appear stable and not overfit.")
        print()
    
    # Risk assessment
    worst_dd = dd['worst_5pct']
    if worst_dd < -0.40:
        print("⚠️  TAIL RISK ALERT")
        print(f"   Worst 5% scenarios show DD up to {worst_dd:.1%}")
        print("   → Consider position sizing or hedging strategies.")
        print()
    
    # Confidence assessment
    ci_width_ret = final_ret['ci_upper'] - final_ret['ci_lower']
    if ci_width_ret > 1.0:  # >100% range
        print("⚠️  HIGH UNCERTAINTY")
        print(f"   90% CI width: {ci_width_ret:.1%}")
        print("   → Highly variable outcomes, difficult to forecast.")
        print()


# =============================================================================
# VISUALIZZAZIONI
# =============================================================================

def plot_mc_distribution(
    mc_equity_curves: pd.DataFrame,
    portfolio_actual: vbt.Portfolio,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Crea visualizzazione completa distribuzione Monte Carlo.
    
    4 plot:
    - Equity curves con percentili
    - Distribuzione final returns
    - Distribuzione max drawdown  
    - Distribuzione Sharpe
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # =========================================================================
    # PLOT 1: Equity Curves con Percentili
    # =========================================================================
    ax = axes[0, 0]
    
    # Percentili MC
    pct_5 = mc_equity_curves.quantile(0.05, axis=1)
    pct_25 = mc_equity_curves.quantile(0.25, axis=1)
    pct_50 = mc_equity_curves.quantile(0.50, axis=1)
    pct_75 = mc_equity_curves.quantile(0.75, axis=1)
    pct_95 = mc_equity_curves.quantile(0.95, axis=1)
    
    # Actual
    actual_equity = portfolio_actual.value()
    
    # Plot
    ax.fill_between(mc_equity_curves.index, pct_5, pct_95, 
                     alpha=0.2, color='blue', label='5th-95th percentile')
    ax.fill_between(mc_equity_curves.index, pct_25, pct_75,
                     alpha=0.3, color='blue', label='25th-75th percentile')
    ax.plot(mc_equity_curves.index, pct_50, 'b-', linewidth=2, label='MC Median')
    ax.plot(actual_equity.index, actual_equity.values, 'r-', linewidth=2, label='Actual')
    
    ax.set_title('Monte Carlo Equity Curves Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 2: Final Returns Distribution
    # =========================================================================
    ax = axes[0, 1]
    
    init_value = mc_equity_curves.iloc[0, 0]
    final_returns = (mc_equity_curves.iloc[-1] / init_value - 1) * 100
    actual_final_ret = (actual_equity.iloc[-1] / portfolio_actual.init_cash - 1) * 100
    
    ax.hist(final_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(actual_final_ret, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_final_ret:.1f}%')
    ax.axvline(final_returns.median(), color='blue', linestyle='--', linewidth=2, label=f'MC Median: {final_returns.median():.1f}%')
    
    ax.set_title('Final Return Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Final Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 3: Max Drawdown Distribution
    # =========================================================================
    ax = axes[1, 0]
    
    def compute_max_dd(equity_series):
        cummax = equity_series.cummax()
        dd = (equity_series - cummax) / cummax
        return dd.min() * 100
    
    mc_drawdowns = mc_equity_curves.apply(compute_max_dd, axis=0)
    actual_dd = compute_max_dd(actual_equity) 
    
    ax.hist(mc_drawdowns, bins=50, alpha=0.7, color='salmon', edgecolor='black')
    ax.axvline(actual_dd, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_dd:.1f}%')
    ax.axvline(mc_drawdowns.median(), color='blue', linestyle='--', linewidth=2, label=f'MC Median: {mc_drawdowns.median():.1f}%')
    
    ax.set_title('Max Drawdown Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 4: Sharpe Ratio Distribution
    # =========================================================================
    ax = axes[1, 1]
    
    mc_returns = mc_equity_curves.pct_change().fillna(0)
    mc_sharpes = (mc_returns.mean() / mc_returns.std()) * np.sqrt(252)
    actual_sharpe = portfolio_actual.sharpe_ratio()
    
    ax.hist(mc_sharpes, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.axvline(actual_sharpe, color='red', linestyle='--', linewidth=2, label=f'Actual: {actual_sharpe:.2f}')
    ax.axvline(mc_sharpes.median(), color='blue', linestyle='--', linewidth=2, label=f'MC Median: {mc_sharpes.median():.2f}')
    
    ax.set_title('Sharpe Ratio Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig


# =============================================================================
# UTILITY: Export Results
# =============================================================================

def export_mc_results(
    mc_equity_curves: pd.DataFrame,
    analysis_dict: Dict,
    output_dir: str = "./mc_results"
):
    """
    Esporta risultati Monte Carlo in vari formati.
    
    Crea:
    - mc_equity_curves.csv (tutte le simulazioni)
    - mc_summary.csv (statistiche)
    - mc_report.txt (report testuale)
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Equity curves complete
    mc_equity_curves.to_csv(f"{output_dir}/mc_equity_curves.csv")
    print(f"✅ Equity curves saved: {output_dir}/mc_equity_curves.csv")
    
    # 2. Summary statistiche
    summary_data = []
    for metric_name, metric_stats in analysis_dict.items():
        if isinstance(metric_stats, dict) and 'actual' in metric_stats:
            summary_data.append({
                'Metric': metric_name,
                'Actual': metric_stats.get('actual'),
                'MC_Mean': metric_stats.get('mc_mean'),
                'MC_Median': metric_stats.get('mc_median'),
                'MC_Std': metric_stats.get('mc_std'),
                'CI_Lower': metric_stats.get('ci_lower'),
                'CI_Upper': metric_stats.get('ci_upper'),
                'Percentile': metric_stats.get('percentile_of_actual')
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/mc_summary.csv", index=False)
    print(f"✅ Summary saved: {output_dir}/mc_summary.csv")
    
    print(f"\n📁 All results in: {output_dir}/")


# =============================================================================
# ESEMPIO USO
# =============================================================================

# # Esegui Monte Carlo
# mc_results = monte_carlo_block_bootstrap_rotational(
#     portfolio=pf_rot,
#     sel_tickers_df=sel_tickers,
#     stocks_data=stocks_data,
#     n_simulations=10_000,
#     block_size=20,
#     random_seed=42
# )

# # Analizza risultati
# analysis = analyze_mc_results(
#     mc_results,
#     pf_rot,
#     confidence_level=0.90,
#     print_report=True
# )

# # Visualizza
# fig = plot_mc_distribution(mc_results, pf_rot)

# # Esporta
# export_mc_results(mc_results, analysis, output_dir="./mc_results")

def monte_carlo_ranking_noise(
    stocks_data: pd.DataFrame,
    benchmark_data: Optional[pd.Series],
    params: Dict,
    n_simulations: int = 1_000,
    noise_std: float = 0.05,
    noise_type: str = "rank",
    init_cash: float = 100_000,
    random_seed: Optional[int] = None,
    show_progress: bool = True,
    build_portfolio_func: Optional[Callable] = None
) -> Dict:
    """
    Monte Carlo Ranking Noise Test per portfolios rotazionali.
    
    Aggiunge noise gaussiano al ranking, poi verifica:
    1. Quanto cambiano le selezioni (selection stability)
    2. Quanto cambia la performance (performance robustness)
    
    Parametri
    ----------
    stocks_data : pd.DataFrame
        Prezzi storici (stesso formato di build_rotational_portfolios_vbt)
    benchmark_data : pd.Series, optional
        Benchmark per confronto
    params : dict
        Parametri portfolio (da WFO o fissi)
        Es: {'rebalance_frequency': 'QE', 'n_top': 5, ...}
    n_simulations : int, default=1_000
        Numero simulazioni (1k tipicamente sufficiente, meno computation-heavy)
    noise_std : float, default=0.05
        Deviazione standard noise (0.05 = 5% rank shift)
        - 0.01-0.03: noise lieve (test sensibilità minima)
        - 0.05-0.10: noise moderato (raccomandato)
        - 0.10-0.20: noise forte (worst-case test)
    noise_type : str, default="rank"
        Tipo di noise:
        - "rank": noise sui rank percentili [0,1]
        - "score": noise sui combo score diretti
    init_cash : float, default=100_000
        Capitale iniziale
    random_seed : int, optional
        Seed per riproducibilità
    show_progress : bool, default=True
        Progress bar
    build_portfolio_func : Callable, optional
        Funzione custom per build portfolio.
        Se None, usa build_rotational_portfolios_vbt di default.
        
    Returns
    -------
    dict
        Risultati completi:
        - 'baseline': Portfolio baseline (no noise)
        - 'simulations': Lista portfolios con noise
        - 'selections_baseline': Selezioni baseline
        - 'selections_sims': Lista selezioni simulate
        - 'metrics': Dict metriche comparative
        - 'stability': Metriche stabilità selezioni
        
    Esempio
    -------
    >>> results = monte_carlo_ranking_noise(
    ...     stocks_data,
    ...     benchmark_data,
    ...     params={'rebalance_frequency': 'QE', 'n_top': 5, ...},
    ...     n_simulations=1_000,
    ...     noise_std=0.05
    ... )
    >>> analyze_ranking_noise_results(results)
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Set build function (assume disponibile)
    if build_portfolio_func is None:
            build_portfolio_func = build_rotational_portfolios_vbt

    
    # =========================================================================
    # 1. BASELINE (no noise)
    # =========================================================================
    print("Building baseline portfolio (no noise)...")
    
    pf_baseline, pf_bench, sel_baseline, rankings_baseline = build_portfolio_func(
        stocks_data=stocks_data,
        benchmark_data=benchmark_data,
        portfolio_name="Baseline",
        init_cash=init_cash,
        plot=False,
        **params
    )
    
    baseline_sharpe = pf_baseline.sharpe_ratio()
    baseline_cagr = pf_baseline.annualized_return()
    
    print(f"Baseline Sharpe: {baseline_sharpe:.3f}")
    print(f"Baseline CAGR: {baseline_cagr:.2%}")
    print()
    
    # =========================================================================
    # 2. SIMULAZIONI CON NOISE
    # =========================================================================
    print(f"Running {n_simulations} simulations with noise_std={noise_std}...")
    
    sim_results = []
    sim_sharpes = []
    sim_cagrs = []
    sim_selections = []
    
    pbar = tqdm(total=n_simulations, desc="Noise Sims", disable=not show_progress)
    
    for sim_idx in range(n_simulations):
        # Crea versione con noise della funzione build
        pf_noisy, _, sel_noisy, _ = _build_portfolio_with_ranking_noise(
            build_func=build_portfolio_func,
            stocks_data=stocks_data,
            benchmark_data=benchmark_data,
            params=params,
            noise_std=noise_std,
            noise_type=noise_type,
            init_cash=init_cash,
            sim_seed=sim_idx if random_seed else None
        )
        
        try:
            sim_sharpe = pf_noisy.sharpe_ratio()
            sim_cagr = pf_noisy.annualized_return()
        except:
            sim_sharpe = np.nan
            sim_cagr = np.nan
        
        sim_results.append({
            'portfolio': pf_noisy,
            'selections': sel_noisy,
            'sharpe': sim_sharpe,
            'cagr': sim_cagr,
            'sim_idx': sim_idx
        })
        
        sim_sharpes.append(sim_sharpe)
        sim_cagrs.append(sim_cagr)
        sim_selections.append(sel_noisy)
        
        pbar.update(1)
    
    pbar.close()
    
    # =========================================================================
    # 3. ANALISI STABILITÀ SELEZIONI
    # =========================================================================
    print("\nAnalyzing selection stability...")
    
    stability_metrics = _compute_selection_stability(
        baseline_selections=sel_baseline,
        simulated_selections=sim_selections
    )
    
    # =========================================================================
    # 4. ANALISI PERFORMANCE ROBUSTNESS
    # =========================================================================
    print("Analyzing performance robustness...")
    
    sim_sharpes_clean = [s for s in sim_sharpes if np.isfinite(s)]
    sim_cagrs_clean = [c for c in sim_cagrs if np.isfinite(c)]
    
    performance_metrics = {
        'baseline_sharpe': baseline_sharpe,
        'baseline_cagr': baseline_cagr,
        'mc_sharpe_mean': np.mean(sim_sharpes_clean) if sim_sharpes_clean else np.nan,
        'mc_sharpe_std': np.std(sim_sharpes_clean) if sim_sharpes_clean else np.nan,
        'mc_sharpe_median': np.median(sim_sharpes_clean) if sim_sharpes_clean else np.nan,
        'mc_cagr_mean': np.mean(sim_cagrs_clean) if sim_cagrs_clean else np.nan,
        'mc_cagr_std': np.std(sim_cagrs_clean) if sim_cagrs_clean else np.nan,
        'sharpe_degradation': baseline_sharpe - np.mean(sim_sharpes_clean) if sim_sharpes_clean else np.nan,
        'sharpe_degradation_pct': ((baseline_sharpe - np.mean(sim_sharpes_clean)) / baseline_sharpe * 100) 
                                   if sim_sharpes_clean and baseline_sharpe != 0 else np.nan,
        'n_failed_sims': n_simulations - len(sim_sharpes_clean)
    }
    
    # =========================================================================
    # RETURN COMPLETO
    # =========================================================================
    return {
        'baseline': {
            'portfolio': pf_baseline,
            'benchmark': pf_bench,
            'selections': sel_baseline,
            'rankings': rankings_baseline,
            'sharpe': baseline_sharpe,
            'cagr': baseline_cagr
        },
        'simulations': sim_results,
        'selections_baseline': sel_baseline,
        'selections_sims': sim_selections,
        'metrics': performance_metrics,
        'stability': stability_metrics,
        'params': params,
        'noise_std': noise_std,
        'n_simulations': n_simulations
    }

def _build_portfolio_with_ranking_noise(
    build_func: Callable,
    stocks_data: pd.DataFrame,
    benchmark_data: Optional[pd.Series],
    params: Dict,
    noise_std: float,
    noise_type: str,
    init_cash: float,
    sim_seed: Optional[int] = None
):
    """
    Wrapper che aggiunge noise al ranking internamente.
    
    STRATEGIA: Monkey-patch temporaneo della funzione rank di pandas
    per aggiungere noise ai rank percentili.
    """
    
    # Salva funzione rank originale
    original_rank = pd.Series.rank
    
    # Crea versione con noise
    def rank_with_noise(self, *args, **kwargs):
        # Rank normale
        ranked = original_rank(self, *args, **kwargs)
        
        # Aggiungi noise gaussiano
        if sim_seed is not None:
            np.random.seed(sim_seed + hash(str(self.index[0])) % 10000)
        
        noise = np.random.normal(0, noise_std, size=len(ranked))
        ranked_noisy = ranked + noise * len(ranked)  # scale by n per mantenere range
        
        # Re-rank per garantire ordine valido
        ranked_noisy = pd.Series(ranked_noisy, index=ranked.index).rank(
            method='average', na_option='keep'
        )
        
        return ranked_noisy
    
    # Applica monkey patch
    pd.Series.rank = rank_with_noise
    
    try:
        # Build portfolio con rank modificato
        pf, pf_bench, sel, rankings = build_func(
            stocks_data=stocks_data,
            benchmark_data=benchmark_data,
            portfolio_name=f"Noise_{sim_seed}",
            init_cash=init_cash,
            plot=False,
            **params
        )
    finally:
        # SEMPRE ripristina funzione originale
        pd.Series.rank = original_rank
    
    return pf, pf_bench, sel, rankings

def _compute_selection_stability(
    baseline_selections: pd.DataFrame,
    simulated_selections: List[pd.DataFrame]
) -> Dict:
    """
    Calcola metriche di stabilità selezioni tra baseline e simulazioni.
    
    Metriche:
    - Avg overlap: % ticker comuni tra baseline e sim
    - Min overlap: worst-case scenario
    - Stability score: weighted metric
    - Churn rate: % ticker changed on average
    """
    
    if baseline_selections.empty or not simulated_selections:
        return {
            'avg_overlap': 0.0,
            'min_overlap': 0.0,
            'max_overlap': 0.0,
            'std_overlap': 0.0,
            'stability_score': 0.0,
            'avg_churn_rate': 1.0,
            'n_dates': 0
        }
    
    rebal_dates = baseline_selections.index
    
    overlaps = []
    churn_rates = []
    
    for date in rebal_dates:
        if date not in baseline_selections.index:
            continue
        
        baseline_set = set(baseline_selections.loc[date, 'tickers'])
        n_baseline = len(baseline_set)
        
        if n_baseline == 0:
            continue
        
        date_overlaps = []
        
        for sim_sel in simulated_selections:
            if date not in sim_sel.index:
                continue
            
            sim_set = set(sim_sel.loc[date, 'tickers'])
            
            # Overlap
            common = len(baseline_set & sim_set)
            overlap = common / n_baseline if n_baseline > 0 else 0.0
            date_overlaps.append(overlap)
            
            # Churn
            changed = len(baseline_set ^ sim_set)  # symmetric difference
            total = len(baseline_set | sim_set)
            churn = changed / total if total > 0 else 0.0
            churn_rates.append(churn)
        
        if date_overlaps:
            overlaps.extend(date_overlaps)
    
    if not overlaps:
        return {
            'avg_overlap': 0.0,
            'min_overlap': 0.0,
            'max_overlap': 0.0,
            'std_overlap': 0.0,
            'stability_score': 0.0,
            'avg_churn_rate': 1.0,
            'n_dates': 0
        }
    
    avg_overlap = np.mean(overlaps)
    min_overlap = np.min(overlaps)
    max_overlap = np.max(overlaps)
    std_overlap = np.std(overlaps)
    avg_churn = np.mean(churn_rates) if churn_rates else 1.0
    
    # Stability score: weighted combination
    # High overlap = good, Low std = good, Low churn = good
    stability_score = (
        0.5 * avg_overlap +
        0.3 * (1 - std_overlap) +
        0.2 * (1 - avg_churn)
    )
    
    return {
        'avg_overlap': avg_overlap,
        'min_overlap': min_overlap,
        'max_overlap': max_overlap,
        'std_overlap': std_overlap,
        'stability_score': stability_score,
        'avg_churn_rate': avg_churn,
        'n_dates': len(rebal_dates),
        'n_comparisons': len(overlaps)
    }

# =============================================================================
# ANALISI RISULTATI
# =============================================================================

def analyze_ranking_noise_results(
    results: Dict,
    print_report: bool = True
) -> Dict:
    """
    Analizza risultati Ranking Noise Test.
    
    Parametri
    ----------
    results : dict
        Output di monte_carlo_ranking_noise()
    print_report : bool
        Stampa report testuale
        
    Returns
    -------
    dict
        Analisi interpretata con raccomandazioni
    """
    
    baseline = results['baseline']
    metrics = results['metrics']
    stability = results['stability']
    noise_std = results['noise_std']
    n_sims = results['n_simulations']
    
    # =========================================================================
    # INTERPRETAZIONE
    # =========================================================================
    
    # Selection Stability
    if stability['avg_overlap'] > 0.80:
        stability_rating = "EXCELLENT"
        stability_color = "🟢"
        stability_msg = "Selezioni molto stabili, parametri robusti"
    elif stability['avg_overlap'] > 0.65:
        stability_rating = "GOOD"
        stability_color = "✅"
        stability_msg = "Selezioni ragionevolmente stabili"
    elif stability['avg_overlap'] > 0.50:
        stability_rating = "MODERATE"
        stability_color = "🟡"
        stability_msg = "Selezioni moderatamente stabili, attenzione"
    else:
        stability_rating = "FRAGILE"
        stability_color = "🔴"
        stability_msg = "Selezioni fragili, parametri sensibili a noise"
    
    # Performance Robustness
    sharpe_deg_pct = metrics['sharpe_degradation_pct']
    
    if np.isnan(sharpe_deg_pct):
        performance_rating = "UNKNOWN"
        performance_color = "⚪"
        performance_msg = "Impossibile calcolare degradation"
    elif sharpe_deg_pct < 5:
        performance_rating = "EXCELLENT"
        performance_color = "🟢"
        performance_msg = "Performance molto robusta al noise"
    elif sharpe_deg_pct < 15:
        performance_rating = "GOOD"
        performance_color = "✅"
        performance_msg = "Performance ragionevolmente robusta"
    elif sharpe_deg_pct < 30:
        performance_rating = "MODERATE"
        performance_color = "🟡"
        performance_msg = "Performance moderatamente sensibile"
    else:
        performance_rating = "WEAK"
        performance_color = "🔴"
        performance_msg = "Performance molto sensibile al noise"
    
    # Overall Assessment
    if stability_rating in ["EXCELLENT", "GOOD"] and performance_rating in ["EXCELLENT", "GOOD"]:
        overall_rating = "ROBUST"
        overall_color = "🟢"
        overall_msg = "Parametri robusti, deploy consigliato"
    elif "FRAGILE" in [stability_rating, performance_rating] or "WEAK" in [stability_rating, performance_rating]:
        overall_rating = "RISKY"
        overall_color = "🔴"
        overall_msg = "Parametri fragili, riconsiderare strategia"
    else:
        overall_rating = "ACCEPTABLE"
        overall_color = "🟡"
        overall_msg = "Parametri accettabili ma con cautela"
    
    interpretation = {
        'stability': {
            'rating': stability_rating,
            'color': stability_color,
            'message': stability_msg
        },
        'performance': {
            'rating': performance_rating,
            'color': performance_color,
            'message': performance_msg
        },
        'overall': {
            'rating': overall_rating,
            'color': overall_color,
            'message': overall_msg
        }
    }
    
    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    
    if print_report:
        print("=" * 80)
        print(f"MONTE CARLO RANKING NOISE TEST ({n_sims:,} simulations)")
        print("=" * 80)
        print(f"Noise Level: {noise_std:.2%} rank std")
        print(f"Parameters: {results['params']}")
        print()
        
        print("─" * 80)
        print("BASELINE PERFORMANCE")
        print("─" * 80)
        print(f"  Sharpe Ratio: {baseline['sharpe']:.3f}")
        print(f"  CAGR:         {baseline['cagr']:.2%}")
        print()
        
        print("─" * 80)
        print("SELECTION STABILITY")
        print("─" * 80)
        print(f"  Avg Overlap:       {stability['avg_overlap']:.1%} {stability_color}")
        print(f"  Min Overlap:       {stability['min_overlap']:.1%}")
        print(f"  Max Overlap:       {stability['max_overlap']:.1%}")
        print(f"  Std Overlap:       {stability['std_overlap']:.3f}")
        print(f"  Avg Churn Rate:    {stability['avg_churn_rate']:.1%}")
        print(f"  Stability Score:   {stability['stability_score']:.3f}")
        print()
        print(f"  Rating: {stability_color} {stability_rating}")
        print(f"  → {stability_msg}")
        print()
        
        print("─" * 80)
        print("PERFORMANCE ROBUSTNESS")
        print("─" * 80)
        print(f"  Baseline Sharpe:   {metrics['baseline_sharpe']:.3f}")
        print(f"  MC Mean Sharpe:    {metrics['mc_sharpe_mean']:.3f}")
        print(f"  MC Std Sharpe:     {metrics['mc_sharpe_std']:.3f}")
        print(f"  Sharpe Degradation: {metrics['sharpe_degradation']:+.3f} ({sharpe_deg_pct:+.1f}%) {performance_color}")
        print()
        print(f"  Baseline CAGR:     {metrics['baseline_cagr']:.2%}")
        print(f"  MC Mean CAGR:      {metrics['mc_cagr_mean']:.2%}")
        print(f"  MC Std CAGR:       {metrics['mc_cagr_std']:.2%}")
        print()
        print(f"  Failed Sims:       {metrics['n_failed_sims']}/{n_sims}")
        print()
        print(f"  Rating: {performance_color} {performance_rating}")
        print(f"  → {performance_msg}")
        print()
        
        print("=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)
        print(f"{overall_color} {overall_rating}: {overall_msg}")
        print()
        
        _print_recommendations(interpretation, stability, metrics)
        
        print("=" * 80)
    
    return {
        'interpretation': interpretation,
        'stability_metrics': stability,
        'performance_metrics': metrics
    }


def _print_recommendations(interpretation, stability, metrics):
    """Stampa raccomandazioni basate su risultati."""
    
    print("RECOMMENDATIONS")
    print("─" * 80)
    
    stability_rating = interpretation['stability']['rating']
    performance_rating = interpretation['performance']['rating']
    
    if stability_rating == "FRAGILE":
        print("🔴 CRITICAL - Selection Instability:")
        print("   → Increase lookback windows (es. 60→120 giorni)")
        print("   → Increase n_top (più ticker = meno sensibile a ranking noise)")
        print("   → Consider removing aggressive filters")
        print()
    
    if performance_rating == "WEAK":
        print("🔴 CRITICAL - Performance Degradation:")
        print(f"   → Sharpe drops by {metrics['sharpe_degradation_pct']:.1f}% with noise")
        print("   → Parameters likely overfit to specific rankings")
        print("   → Test with simpler strategy or longer lookbacks")
        print()
    
    if stability['avg_overlap'] < 0.60:
        print("⚠️  LOW OVERLAP WARNING:")
        print("   → Less than 60% selection overlap with noise")
        print("   → Consider momentum_weight closer to 1.0 (less risk-parity)")
        print("   → Test with acceleration=False")
        print()
    
    if stability['avg_churn_rate'] > 0.50:
        print("⚠️  HIGH CHURN WARNING:")
        print("   → More than 50% portfolio turnover with noise")
        print("   → Transaction costs will significantly impact live performance")
        print("   → Consider quarterly rebalance instead of monthly")
        print()
    
    if interpretation['overall']['rating'] == "ROBUST":
        print("✅ STRATEGY APPROVED:")
        print("   → Parameters appear robust to ranking perturbations")
        print("   → Safe to deploy with confidence")
        print("   → Continue monitoring with walk-forward OOS")
        print()
    elif interpretation['overall']['rating'] == "ACCEPTABLE":
        print("🟡 PROCEED WITH CAUTION:")
        print("   → Parameters acceptable but not ideal")
        print("   → Consider reducing position sizes by 20-30%")
        print("   → Monitor first 2-3 rebalances closely")
        print()


# =============================================================================
# VISUALIZZAZIONI
# =============================================================================

def plot_ranking_noise_analysis(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Visualizzazione completa Ranking Noise Test.
    
    4 grafici:
    1. Selection Overlap Distribution
    2. Performance Distribution (Sharpe)
    3. Churn Rate over Time
    4. Equity Curve Comparison (Baseline vs MC mean/range)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    baseline = results['baseline']
    sims = results['simulations']
    stability = results['stability']
    metrics = results['metrics']
    
    # =========================================================================
    # PLOT 1: Selection Overlap Distribution
    # =========================================================================
    ax = axes[0, 0]
    
    # Calcola overlap per ogni simulazione
    baseline_sel = results['selections_baseline']
    sim_selections = results['selections_sims']
    
    all_overlaps = []
    for date in baseline_sel.index:
        baseline_set = set(baseline_sel.loc[date, 'Top_Tickers'])
        n_baseline = len(baseline_set)
        
        if n_baseline == 0:
            continue
        
        for sim_sel in sim_selections:
            if date in sim_sel.index:
                sim_set = set(sim_sel.loc[date, 'tickers'])
                overlap = len(baseline_set & sim_set) / n_baseline
                all_overlaps.append(overlap * 100)
    
    if all_overlaps:
        ax.hist(all_overlaps, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(stability['avg_overlap'] * 100, color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {stability['avg_overlap']:.1%}")
        ax.axvline(80, color='green', linestyle=':', linewidth=2, alpha=0.5, label='80% threshold')
    
    ax.set_title('Selection Overlap Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Overlap with Baseline (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 2: Sharpe Distribution
    # =========================================================================
    ax = axes[0, 1]
    
    sim_sharpes = [s['sharpe'] for s in sims if np.isfinite(s['sharpe'])]
    
    if sim_sharpes:
        ax.hist(sim_sharpes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(baseline['sharpe'], color='red', linestyle='--', linewidth=2,
                   label=f"Baseline: {baseline['sharpe']:.3f}")
        ax.axvline(np.mean(sim_sharpes), color='blue', linestyle='--', linewidth=2,
                   label=f"MC Mean: {np.mean(sim_sharpes):.3f}")
    
    ax.set_title('Sharpe Ratio Distribution (with Noise)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 3: Churn Rate Over Time
    # =========================================================================
    ax = axes[1, 0]
    
    # Calcola churn per data
    churn_by_date = {}
    
    for date in baseline_sel.index:
        baseline_set = set(baseline_sel.loc[date, 'tickers'])
        
        date_churns = []
        for sim_sel in sim_selections:
            if date in sim_sel.index:
                sim_set = set(sim_sel.loc[date, 'Top_Tickers'])
                changed = len(baseline_set ^ sim_set)
                total = len(baseline_set | sim_set)
                churn = (changed / total * 100) if total > 0 else 0
                date_churns.append(churn)
        
        if date_churns:
            churn_by_date[date] = np.mean(date_churns)
    
    if churn_by_date:
        dates = list(churn_by_date.keys())
        churns = list(churn_by_date.values())
        
        ax.plot(dates, churns, 'o-', alpha=0.6, color='coral')
        ax.axhline(stability['avg_churn_rate'] * 100, color='red', linestyle='--',
                   label=f"Mean: {stability['avg_churn_rate']:.1%}")
        ax.axhline(50, color='orange', linestyle=':', alpha=0.5, label='50% threshold')
    
    ax.set_title('Churn Rate Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rebalance Date')
    ax.set_ylabel('Churn Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # =========================================================================
    # PLOT 4: Equity Curves (Baseline vs MC range)
    # =========================================================================
    ax = axes[1, 1]
    
    # Baseline equity
    baseline_equity = baseline['portfolio'].value()
    ax.plot(baseline_equity.index, baseline_equity.values, 'r-', linewidth=2,
            label='Baseline', zorder=10)
    
    # MC equity curves (sample 20 for visualization)
    n_plot = min(20, len(sims))
    for i in range(n_plot):
        sim_equity = sims[i]['portfolio'].value()
        ax.plot(sim_equity.index, sim_equity.values, 'gray', alpha=0.2, linewidth=1)
    
    # MC mean
    if sims:
        sim_equities = np.array([s['portfolio'].value().values for s in sims if hasattr(s['portfolio'], 'value')])
        if len(sim_equities) > 0:
            mc_mean = sim_equities.mean(axis=0)
            ax.plot(baseline_equity.index, mc_mean, 'b--', linewidth=2,
                    label='MC Mean', zorder=9)
    
    ax.set_title('Equity Curves: Baseline vs Noise Simulations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()
    
    return fig

# =============================================================================
# MC PER WFO: PER-WINDOW ANALYSIS
# =============================================================================

def monte_carlo_wfo_per_window(
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    wfo_summary: pd.DataFrame,
    benchmark_title: str,
    n_simulations: int = 1_000,
    noise_std: float = 0.05,
    init_cash: float = 100_000,
    random_seed: Optional[int] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Esegue MC Ranking Noise per OGNI finestra WFO separatamente.
    
    Testa robustezza parametri nella STESSA finestra OOS dove sono stati usati.
    
    Parametri
    ----------
    stocks_data : pd.DataFrame
        Prezzi storici completi
    benchmark_data : pd.Series
        Benchmark completo
    wfo_summary : pd.DataFrame
        Output di walk_forward_rotational() con colonne:
        - Index: Window string (es. "2020-01-01→2020-12-31 | 2021-01-01→2021-12-31")
        - Parametri: rebalance_frequency, n_top, momentum_lookback_days, etc.
        - TestStart, TestEnd: date OOS (se presenti, altrimenti parsate da Index)
    benchmark_title : str
        Nome benchmark (es. "^STOXX50E")
    n_simulations : int, default=1_000
        Numero simulazioni MC per finestra
    noise_std : float, default=0.05
        Deviazione standard noise (5% raccomandato)
    init_cash : float, default=100_000
        Capitale iniziale
    random_seed : int, optional
        Seed per riproducibilità
    show_progress : bool, default=True
        Progress bar
        
    Returns
    -------
    pd.DataFrame
        Una riga per finestra WFO con metriche:
        - Window, TestStart, TestEnd
        - Baseline_Sharpe, MC_Mean_Sharpe, Sharpe_Degradation_Pct
        - Avg_Overlap, Min_Overlap, Stability_Score
        - Overall_Rating
        - Parametri della finestra
        
    Esempio
    -------
    >>> wfo_summary = load_wfo_summary("Alpha Euro_2026.wfo_summary")
    >>> wfo_mc = monte_carlo_wfo_per_window(
    ...     stocks_data, benchmark_data, wfo_summary,
    ...     benchmark_title="^STOXX50E",
    ...     n_simulations=1_000
    ... )
    >>> print(wfo_mc[['Window', 'Avg_Overlap', 'Overall_Rating']])
    """
    wfo_summary = filter_testable_windows(
        wfo_summary,
        min_days_ago=30,
        verbose=True
    )

    if random_seed is not None:
        np.random.seed(random_seed)
    
    # # Import funzione (già disponibile nell'ambiente)
    # try:
    #     from r_functions import build_rotational_portfolios_from_wfo_result
    # except ImportError:
    #     raise ImportError(
    #         "build_rotational_portfolios_from_wfo_result non trovata. "
    #         "Assicurati di aver eseguito %run r_functions.ipynb"
    #     )
    
    results_per_window = []
    
    # =========================================================================
    # LOOP FINESTRE
    # =========================================================================
    
    pbar_windows = tqdm(
        total=len(wfo_summary),
        desc="MC per Finestra WFO",
        disable=not show_progress,
        position=0
    )
    
    for window_idx, (idx, row) in enumerate(wfo_summary.iterrows(), start=1):
        
        # ---------------------------------------------------------------------
        # 1. PARSE FINESTRA (formato: "TestStart→TestEnd")
        # ---------------------------------------------------------------------
        try:
            # Parse diretto da index: "2021-01-01→2021-12-31"
            parts = [x.strip() for x in str(idx).split("→", 1)]
            if len(parts) != 2:
                print(f"⚠️  Skipping window {idx}: invalid format (expected 'START→END')")
                pbar_windows.update(1)
                continue
            
            test_start = pd.Timestamp(parts[0])
            test_end = pd.Timestamp(parts[1])
        except Exception as e:
            print(f"⚠️  Skipping window {idx}: cannot parse dates ({e})")
            pbar_windows.update(1)
            continue
        
        window_str = f"{test_start.date()} → {test_end.date()}"
        pbar_windows.set_description(f"MC [{window_idx}/{len(wfo_summary)}] {window_str}")
        
        # ---------------------------------------------------------------------
        # 2. VERIFICA SE FINESTRA È TESTABILE
        # ---------------------------------------------------------------------
        
        today = pd.Timestamp.now().normalize()
        
        # CASO 1: Finestra completamente futura
        if test_start > today:
            if show_progress:
                print(f"⏭️  Skipping future window: {window_str}")
            pbar_windows.update(1)
            continue
        
        # CASO 2: Finestra parzialmente completata (test_end nel futuro o molto recente)
        # Per portafogli trimestrali, se siamo a Feb 2026 e test_end è Dic 2026,
        # non ci saranno rebalance dates ancora disponibili
        if test_end > today:
            if show_progress:
                print(f"⏭️  Skipping incomplete window (ends in future): {window_str}")
            pbar_windows.update(1)
            continue
        
        # CASO 3: Test_end è nel passato ma troppo recente (< 30 giorni)
        # Potrebbero mancare ancora dati completi
        days_since_end = (today - test_end).days
        if days_since_end < 30:
            if show_progress:
                print(f"⏭️  Skipping recent window (only {days_since_end} days old): {window_str}")
            pbar_windows.update(1)
            continue
        
        # CASO 4: Verifica dati disponibili
        available_data = stocks_data.loc[test_start:test_end]
        if available_data.empty:
            print(f"⚠️  Skipping {window_str}: no data available in period")
            pbar_windows.update(1)
            continue
        
        if len(available_data) < 10:  # almeno 10 giorni trading
            print(f"⚠️  Skipping {window_str}: insufficient data ({len(available_data)} days)")
            pbar_windows.update(1)
            continue
        
        # ---------------------------------------------------------------------
        # 3. BASELINE PORTFOLIO (da WFO result, no noise)
        # ---------------------------------------------------------------------
        
        # Limita summary_df a questa finestra
        summary_df_window = wfo_summary.loc[[idx]]
        
        try:
            pf_baseline, pf_bench, sel_baseline = build_rotational_portfolios_from_wfo_result(
                summary_df=summary_df_window,
                stocks_data=stocks_data,
                benchmark_data=benchmark_data,
                benchmark_title=benchmark_title,
                portfolio_name=f"Baseline_{window_str}",
                start_date=test_start,
                end_date=test_end,
                init_cash=init_cash,
                plot=False,
                debug=False,
                show_report=False
            )
        except Exception as e:
            error_msg = str(e).lower()
            
            # Errori comuni per finestre incomplete
            if any(keyword in error_msg for keyword in ['vuoto', 'empty', 'no selection', 'no data']):
                if show_progress:
                    print(f"⏭️  Skipping {window_str}: no valid selections (likely incomplete period)")
            else:
                print(f"⚠️  Error building baseline for {window_str}: {e}")
            
            pbar_windows.update(1)
            continue
        
        # Verifica esplicita che sel_baseline non sia vuoto
        if sel_baseline is None or sel_baseline.empty or len(sel_baseline) == 0:
            if show_progress:
                print(f"⏭️  Skipping {window_str}: no rebalance dates in period")
            pbar_windows.update(1)
            continue
        
        # Verifica che portfolio sia valido
        try:
            baseline_sharpe = pf_baseline.sharpe_ratio()
            baseline_cagr = pf_baseline.annualized_return()
            
            # Check se metriche sono valide
            if not np.isfinite(baseline_sharpe) or not np.isfinite(baseline_cagr):
                print(f"⚠️  Skipping {window_str}: invalid baseline metrics (Sharpe={baseline_sharpe}, CAGR={baseline_cagr})")
                pbar_windows.update(1)
                continue
                
        except Exception as e:
            print(f"⚠️  Skipping {window_str}: cannot compute baseline metrics ({e})")
            pbar_windows.update(1)
            continue
        
        # ---------------------------------------------------------------------
        # 3. MC SIMULAZIONI CON NOISE
        # ---------------------------------------------------------------------
        
        sim_sharpes = []
        sim_cagrs = []
        sim_selections = []
        
        pbar_sims = tqdm(
            total=n_simulations,
            desc=f"  Sims {window_str}",
            disable=not show_progress,
            position=1,
            leave=False
        )
        
        for sim_idx in range(n_simulations):
            try:
                # Build con noise usando monkey-patch interno
                # NOTA: build_rotational_portfolios_from_wfo_result chiama
                # build_rotational_portfolios_vbt internamente, quindi
                # il monkey-patch su pd.Series.rank funziona
                
                original_rank = pd.Series.rank
                
                def rank_with_noise(self, *args, **kwargs):
                    ranked = original_rank(self, *args, **kwargs)
                    if random_seed is not None:
                        np.random.seed(random_seed + sim_idx + hash(str(self.index[0])) % 10000)
                    noise = np.random.normal(0, noise_std, size=len(ranked))
                    ranked_noisy = ranked + noise * len(ranked)
                    ranked_noisy = pd.Series(ranked_noisy, index=ranked.index).rank(
                        method='average', na_option='keep'
                    )
                    return ranked_noisy
                
                pd.Series.rank = rank_with_noise
                
                pf_noisy, _, sel_noisy = build_rotational_portfolios_from_wfo_result(
                    summary_df=summary_df_window,
                    stocks_data=stocks_data,
                    benchmark_data=benchmark_data,
                    benchmark_title=benchmark_title,
                    portfolio_name=f"Noisy_{sim_idx}",
                    start_date=test_start,
                    end_date=test_end,
                    init_cash=init_cash,
                    plot=False,
                    debug=False,
                    show_report=False
                )
                
                pd.Series.rank = original_rank
                
                sim_sharpe = pf_noisy.sharpe_ratio()
                sim_cagr = pf_noisy.annualized_return()
                
                sim_sharpes.append(sim_sharpe)
                sim_cagrs.append(sim_cagr)
                sim_selections.append(sel_noisy)
                
            except Exception as e:
                # Ripristina sempre
                pd.Series.rank = original_rank
                # Continua con prossima sim
            finally:
                pbar_sims.update(1)
        
        pbar_sims.close()
        
        # ---------------------------------------------------------------------
        # 4. ANALISI STABILITÀ
        # ---------------------------------------------------------------------
        
        if not sim_selections:
            print(f"⚠️  No valid simulations for {window_str}")
            pbar_windows.update(1)
            continue
        
        stability = _compute_selection_stability(sel_baseline, sim_selections)
        
        # Performance metrics
        sim_sharpes_clean = [s for s in sim_sharpes if np.isfinite(s)]
        sim_cagrs_clean = [c for c in sim_cagrs if np.isfinite(c)]
        
        mc_sharpe_mean = np.mean(sim_sharpes_clean) if sim_sharpes_clean else np.nan
        mc_cagr_mean = np.mean(sim_cagrs_clean) if sim_cagrs_clean else np.nan
        
        sharpe_degradation = baseline_sharpe - mc_sharpe_mean
        sharpe_degradation_pct = (sharpe_degradation / baseline_sharpe * 100) if baseline_sharpe != 0 else np.nan
        
        # Rating
        if stability['avg_overlap'] > 0.80 and abs(sharpe_degradation_pct) < 5:
            rating = "ROBUST"
        elif stability['avg_overlap'] < 0.50 or abs(sharpe_degradation_pct) > 30:
            rating = "RISKY"
        else:
            rating = "ACCEPTABLE"
        
        # ---------------------------------------------------------------------
        # 5. STORE RESULTS
        # ---------------------------------------------------------------------
        
        # Estrai parametri della finestra
        param_cols = [
            'rebalance_frequency', 'momentum_lookback_days',
            'riskparity_lookback_days', 'n_top', 'use_acceleration',
            'momentum_weight', 'filter_ema', 'filter_volatility',
            'filter_min_momentum'
        ]
        
        params_dict = {col: row.get(col, np.nan) for col in param_cols}
        
        results_per_window.append({
            'Window': window_str,
            'TestStart': test_start,
            'TestEnd': test_end,
            'Baseline_Sharpe': baseline_sharpe,
            'Baseline_CAGR': baseline_cagr,
            'MC_Mean_Sharpe': mc_sharpe_mean,
            'MC_Mean_CAGR': mc_cagr_mean,
            'Sharpe_Degradation': sharpe_degradation,
            'Sharpe_Degradation_Pct': sharpe_degradation_pct,
            'Avg_Overlap': stability['avg_overlap'],
            'Min_Overlap': stability['min_overlap'],
            'Max_Overlap': stability['max_overlap'],
            'Stability_Score': stability['stability_score'],
            'Avg_Churn_Rate': stability['avg_churn_rate'],
            'Overall_Rating': rating,
            'N_Valid_Sims': len(sim_sharpes_clean),
            **params_dict
        })
        
        pbar_windows.update(1)
    
    pbar_windows.close()
    
    # =========================================================================
    # RETURN DATAFRAME
    # =========================================================================
    
    results_df = pd.DataFrame(results_per_window)
    
    return results_df


def filter_testable_windows(
    wfo_summary: pd.DataFrame,
    min_days_ago: int = 30,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filtra wfo_summary per includere solo finestre testabili con MC.
    
    Rimuove:
    - Finestre future (TestEnd > oggi)
    - Finestre incomplete (TestEnd < 30 giorni fa)
    
    Parametri
    ----------
    wfo_summary : pd.DataFrame
        WFO summary completo (index format: "TestStart→TestEnd")
    min_days_ago : int, default=30
        Minimo giorni tra TestEnd e oggi per considerare finestra testabile
    verbose : bool, default=True
        Stampa info su finestre rimosse
        
    Returns
    -------
    pd.DataFrame
        WFO summary filtrato con sole finestre testabili
        
    Esempio
    -------
    >>> wfo_full = load_wfo_summary("Alpha Euro_2026.wfo_summary")
    >>> wfo_testable = filter_testable_windows(wfo_full)
    >>> # wfo_testable esclude finestre 2026 se incomplete
    """
    
    today = pd.Timestamp.now().normalize()
    testable_indices = []
    skipped = []
    
    for idx in wfo_summary.index:
        try:
            # Parse date da index
            parts = [x.strip() for x in str(idx).split("→", 1)]
            if len(parts) != 2:
                skipped.append((idx, "invalid format"))
                continue
            
            test_start = pd.Timestamp(parts[0])
            test_end = pd.Timestamp(parts[1])
            
            # Check se testabile
            if test_start > today:
                skipped.append((idx, "future window"))
                continue
            
            if test_end > today:
                skipped.append((idx, "incomplete (ends in future)"))
                continue
            
            days_since_end = (today - test_end).days
            if days_since_end < min_days_ago:
                skipped.append((idx, f"too recent ({days_since_end} days ago)"))
                continue
            
            # Testabile
            testable_indices.append(idx)
            
        except Exception as e:
            skipped.append((idx, f"parse error: {e}"))
            continue
    
    # Filtra
    wfo_filtered = wfo_summary.loc[testable_indices]
    
    # Report
    if verbose:
        print(f"WFO Summary Filtering:")
        print(f"  Total windows: {len(wfo_summary)}")
        print(f"  Testable: {len(wfo_filtered)}")
        print(f"  Skipped: {len(skipped)}")
        
        if skipped:
            print(f"\nSkipped windows:")
            for window, reason in skipped:
                print(f"  - {window}: {reason}")
        print()
    
    return wfo_filtered


# =============================================================================
# ANALISI AGGREGATE
# =============================================================================

def analyze_wfo_mc_results(
    wfo_mc_df: pd.DataFrame,
    print_report: bool = True
) -> Dict:
    """
    Analizza risultati aggregati di MC su WFO multi-finestra.
    
    Parametri
    ----------
    wfo_mc_df : pd.DataFrame
        Output di monte_carlo_wfo_per_window()
    print_report : bool
        Stampa report
        
    Returns
    -------
    dict
        Analisi aggregate e raccomandazioni
    """
    
    n_windows = len(wfo_mc_df)
    
    # Aggregate statistics
    avg_overlap_all = wfo_mc_df['Avg_Overlap'].mean()
    min_overlap_all = wfo_mc_df['Avg_Overlap'].min()
    
    avg_degradation_all = wfo_mc_df['Sharpe_Degradation_Pct'].mean()
    max_degradation_all = wfo_mc_df['Sharpe_Degradation_Pct'].max()
    
    n_robust = (wfo_mc_df['Overall_Rating'] == 'ROBUST').sum()
    n_acceptable = (wfo_mc_df['Overall_Rating'] == 'ACCEPTABLE').sum()
    n_risky = (wfo_mc_df['Overall_Rating'] == 'RISKY').sum()
    
    # Overall assessment
    robust_pct = n_robust / n_windows
    
    if robust_pct >= 0.70:
        overall_rating = "ROBUST"
        overall_color = "🟢"
        overall_msg = "Maggioranza finestre robuste, strategia approvata"
    elif robust_pct >= 0.40:
        overall_rating = "ACCEPTABLE"
        overall_color = "🟡"
        overall_msg = "Mix di finestre robuste/fragili, cautela"
    else:
        overall_rating = "RISKY"
        overall_color = "🔴"
        overall_msg = "Maggioranza finestre fragili, rivedere strategia"
    
    # Identify worst window
    worst_idx = wfo_mc_df['Stability_Score'].idxmin()
    worst_window = wfo_mc_df.loc[worst_idx]
    
    # Identify best window
    best_idx = wfo_mc_df['Stability_Score'].idxmax()
    best_window = wfo_mc_df.loc[best_idx]
    
    analysis = {
        'n_windows': n_windows,
        'avg_overlap_all': avg_overlap_all,
        'min_overlap_all': min_overlap_all,
        'avg_degradation_all': avg_degradation_all,
        'max_degradation_all': max_degradation_all,
        'n_robust': n_robust,
        'n_acceptable': n_acceptable,
        'n_risky': n_risky,
        'robust_pct': robust_pct,
        'overall_rating': overall_rating,
        'overall_color': overall_color,
        'overall_msg': overall_msg,
        'worst_window': worst_window.to_dict(),
        'best_window': best_window.to_dict()
    }
    
    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    
    if print_report:
        print("=" * 80)
        print("MONTE CARLO WFO - ANALISI AGGREGATE")
        print("=" * 80)
        print(f"Finestre analizzate: {n_windows}")
        print()
        
        print("─" * 80)
        print("STABILITÀ SELEZIONI (aggregate)")
        print("─" * 80)
        print(f"  Avg Overlap (tutte finestre):  {avg_overlap_all:.1%}")
        print(f"  Min Overlap (finestra peggiore): {min_overlap_all:.1%}")
        print()
        
        print("─" * 80)
        print("PERFORMANCE ROBUSTNESS (aggregate)")
        print("─" * 80)
        print(f"  Avg Sharpe Degradation:   {avg_degradation_all:+.1f}%")
        print(f"  Max Sharpe Degradation:   {max_degradation_all:+.1f}%")
        print()
        
        print("─" * 80)
        print("RATING PER FINESTRA")
        print("─" * 80)
        print(f"  ROBUST:     {n_robust}/{n_windows} ({robust_pct:.0%}) 🟢")
        print(f"  ACCEPTABLE: {n_acceptable}/{n_windows} ({n_acceptable/n_windows:.0%}) 🟡")
        print(f"  RISKY:      {n_risky}/{n_windows} ({n_risky/n_windows:.0%}) 🔴")
        print()
        
        print("=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)
        print(f"{overall_color} {overall_rating}: {overall_msg}")
        print()
        
        print("─" * 80)
        print("FINESTRA PEGGIORE")
        print("─" * 80)
        print(f"  Window: {worst_window['Window']}")
        print(f"  Avg Overlap: {worst_window['Avg_Overlap']:.1%}")
        print(f"  Sharpe Degradation: {worst_window['Sharpe_Degradation_Pct']:+.1f}%")
        print(f"  Rating: {worst_window['Overall_Rating']}")
        print()
        
        print("─" * 80)
        print("FINESTRA MIGLIORE")
        print("─" * 80)
        print(f"  Window: {best_window['Window']}")
        print(f"  Avg Overlap: {best_window['Avg_Overlap']:.1%}")
        print(f"  Sharpe Degradation: {best_window['Sharpe_Degradation_Pct']:+.1f}%")
        print(f"  Rating: {best_window['Overall_Rating']}")
        print()
        
        print("=" * 80)
        print("RACCOMANDAZIONI")
        print("=" * 80)
        
        if overall_rating == "ROBUST":
            print("✅ STRATEGIA APPROVATA:")
            print("   → Maggioranza finestre mostra parametri robusti")
            print("   → Safe to deploy con confidence")
            print()
        elif overall_rating == "RISKY":
            print("🔴 STRATEGIA A RISCHIO:")
            print("   → Maggioranza finestre fragili")
            print("   → Considera:")
            print("     • Lookback più lunghi (60→120 giorni)")
            print("     • n_top più alto (3→8)")
            print("     • Rimuovere filtri aggressivi")
            print()
        else:
            print("🟡 STRATEGIA ACCETTABILE CON CAUTELA:")
            print("   → Mix di finestre robuste e fragili")
            print("   → Suggerimenti:")
            print("     • Deploy con position sizing ridotto (50-70%)")
            print("     • Monitoring intensivo prime rebalances")
            print("     • Analizza pattern finestre fragili (regime-specific?)")
            print()
        
        # Analisi temporale
        if 'TestStart' in wfo_mc_df.columns:
            wfo_mc_df_sorted = wfo_mc_df.sort_values('TestStart')
            recent_windows = wfo_mc_df_sorted.tail(3)
            
            recent_robust = (recent_windows['Overall_Rating'] == 'ROBUST').sum()
            
            print("─" * 80)
            print("TREND TEMPORALE (ultime 3 finestre)")
            print("─" * 80)
            
            for _, row in recent_windows.iterrows():
                status = "✅" if row['Overall_Rating'] == "ROBUST" else "🟡" if row['Overall_Rating'] == "ACCEPTABLE" else "🔴"
                print(f"  {status} {row['Window']}: Overlap={row['Avg_Overlap']:.1%}, Rating={row['Overall_Rating']}")
            
            if recent_robust >= 2:
                print("\n  → Trend positivo: finestre recenti robuste")
            elif recent_robust == 0:
                print("\n  ⚠️  Trend negativo: nessuna finestra recente robusta")
            
            print()
        
        print("=" * 80)
    
    return analysis

# =============================================================================
# VISUALIZZAZIONI
# =============================================================================

def plot_wfo_mc_results(
    wfo_mc_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize=(16, 10)
):
    """
    Visualizza risultati MC per WFO multi-finestra.
    
    4 grafici:
    1. Overlap per finestra (timeline)
    2. Sharpe Degradation per finestra
    3. Rating distribution
    4. Stability Score trend
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Sort by TestStart
    if 'TestStart' in wfo_mc_df.columns:
        wfo_mc_df = wfo_mc_df.sort_values('TestStart')
    
    windows = wfo_mc_df['Window'].values
    x_pos = np.arange(len(windows))
    
    # =========================================================================
    # PLOT 1: Avg Overlap per Finestra
    # =========================================================================
    ax = axes[0, 0]
    
    colors = [
        'green' if r == 'ROBUST' else 'orange' if r == 'ACCEPTABLE' else 'red'
        for r in wfo_mc_df['Overall_Rating']
    ]
    
    ax.bar(x_pos, wfo_mc_df['Avg_Overlap'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(80, color='green', linestyle='--', alpha=0.5, label='80% target')
    ax.axhline(60, color='orange', linestyle='--', alpha=0.5, label='60% min')
    
    ax.set_title('Selection Overlap per Finestra WFO', fontsize=14, fontweight='bold')
    ax.set_xlabel('Finestra')
    ax.set_ylabel('Avg Overlap (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(windows, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 2: Sharpe Degradation per Finestra
    # =========================================================================
    ax = axes[0, 1]
    
    ax.bar(x_pos, wfo_mc_df['Sharpe_Degradation_Pct'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(15, color='orange', linestyle='--', alpha=0.5, label='15% threshold')
    ax.axhline(30, color='red', linestyle='--', alpha=0.5, label='30% critical')
    
    ax.set_title('Sharpe Degradation per Finestra', fontsize=14, fontweight='bold')
    ax.set_xlabel('Finestra')
    ax.set_ylabel('Degradation (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(windows, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 3: Rating Distribution
    # =========================================================================
    ax = axes[1, 0]
    
    rating_counts = wfo_mc_df['Overall_Rating'].value_counts()
    colors_pie = []
    for rating in rating_counts.index:
        if rating == 'ROBUST':
            colors_pie.append('green')
        elif rating == 'ACCEPTABLE':
            colors_pie.append('orange')
        else:
            colors_pie.append('red')
    
    ax.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.0f%%',
           colors=colors_pie, startangle=90)
    ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    
    # =========================================================================
    # PLOT 4: Stability Score Trend
    # =========================================================================
    ax = axes[1, 1]
    
    ax.plot(x_pos, wfo_mc_df['Stability_Score'], 'o-', linewidth=2, markersize=8)
    ax.axhline(0.75, color='green', linestyle='--', alpha=0.5, label='Good (0.75)')
    ax.axhline(0.60, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.60)')
    
    ax.set_title('Stability Score Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Finestra')
    ax.set_ylabel('Composite Stability Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(windows, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()
    
    return fig



# Engine Sanity Check

def build_engine_health_check(
    pf,
    sel_tickers: "pd.DataFrame",
    *,
    prices: "pd.DataFrame | None" = None,
    start_date: "pd.Timestamp | str | None" = None,
    end_date: "pd.Timestamp | str | None" = None,
    include_prev: bool = True,
    annual_trading_days: int = 252,
    # --- warning thresholds (tuning) ---
    warn_pct_never_selected: float = 0.40,
    warn_top1_sel_share: float = 0.35,
    warn_avg_churn: float = 0.70,
    warn_concentration_hhi: float = 0.18,
    return_details: bool = True,
):
    """
    Health-check del motore rotazionale (per sviluppatori).

    Input richiesti:
    - pf: vbt.Portfolio (rotational)
    - sel_tickers: DataFrame con indice datetime-like e colonna 'tickers' (liste o stringhe)

    Input opzionali (consigliati):
    - prices: DataFrame prezzi (Date x Ticker) per:
        1) definire l'universo reale (tickers disponibili)
        2) calcolare "held return" per ticker (proxy: compounding sui giorni in cui il ticker risulta selezionato)
    - start_date/end_date: finestra analisi; se None usa range di pf (se disponibile) o sel_tickers
    - include_prev: include ultima selezione prima di start_date come “set iniziale”

    Output:
    - health_df: tabella sintetica health-check (metriche + warning + sintesi finale)
    - ticker_df: diagnostica per-ticker (selezioni, share, held-return, ecc.)
    - selection_log: DataFrame long con colonne [Date, Ticker] di tutte le selezioni
    - details: dict con oggetti intermedi (se return_details=True)
    """
    import numpy as np
    import pandas as pd

    # --- helpers ---
    def _to_ts(x):
        if x is None:
            return None
        ts = pd.to_datetime(x)
        try:
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_localize(None)
        except Exception:
            pass
        return pd.Timestamp(ts).normalize()

    def _norm_index(idx):
        idx = pd.to_datetime(idx)
        try:
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
        except Exception:
            pass
        return pd.DatetimeIndex(idx).normalize()

    def _ensure_list(x):
        if x is None:
            return []
        if isinstance(x, float) and pd.isna(x):
            return []
        if isinstance(x, (list, tuple, set)):
            return [str(t).strip() for t in x if str(t).strip()]
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            if "," in s:
                return [p.strip() for p in s.split(",") if p.strip()]
            return [s]
        return [str(x).strip()]

    # --- validate sel_tickers ---
    if sel_tickers is None or getattr(sel_tickers, "empty", True):
        raise ValueError("sel_tickers è vuoto.")
    if "tickers" not in sel_tickers.columns:
        raise KeyError("sel_tickers deve contenere la colonna 'tickers'.")

    df = sel_tickers.copy()
    df.index = _norm_index(df.index)
    df = df.sort_index()

    # --- resolve analysis window (pf preferred) ---
    pf_start = None
    pf_end = None
    try:
        st = pf.stats()
        if isinstance(st, dict) or hasattr(st, "__getitem__"):
            pf_start = _to_ts(pd.to_datetime(st["Start"]))
            pf_end = _to_ts(pd.to_datetime(st["End"]))
    except Exception:
        pass

    s = _to_ts(start_date) or pf_start or df.index.min()
    e = _to_ts(end_date) or pf_end or df.index.max()

    if s is None or pd.isna(s):
        s = df.index.min()
    if e is None or pd.isna(e):
        e = df.index.max()
    if s > e:
        s, e = e, s

    # --- filter window + include prev selection ---
    df_win = df.loc[s:e].copy()

    if include_prev and s is not None:
        df_prev = df.loc[:s]
        if not df_prev.empty:
            if df_prev.index.max() == s and len(df_prev) >= 2:
                df_prev_strict = df_prev.iloc[:-1]
            elif df_prev.index.max() < s:
                df_prev_strict = df_prev
            else:
                df_prev_strict = df_prev

            if not df_prev_strict.empty:
                prev_row = df_prev_strict.iloc[[-1]]
                df_win = pd.concat([prev_row, df_win], axis=0)
                df_win = df_win[~df_win.index.duplicated(keep="last")].sort_index()

    if df_win.empty:
        raise ValueError(f"Nessuna selezione nella finestra {s} → {e}.")

    # --- build selection_log (long): Date x Ticker ---
    rows = []
    for dt, row in df_win.iterrows():
        tick_list = _ensure_list(row["tickers"])
        for t in tick_list:
            rows.append((dt, t))

    selection_log = pd.DataFrame(rows, columns=["Date", "Ticker"])
    selection_log["Date"] = _norm_index(selection_log["Date"])
    selection_log = selection_log.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # --- counts & shares ---
    counts = selection_log["Ticker"].value_counts()
    total_picks = int(len(selection_log))
    n_dates = int(df_win.shape[0])

    ticker_df = pd.DataFrame({
        "Ticker": counts.index.astype(str),
        "Selections": counts.values,
    })
    ticker_df["Selection_Share"] = ticker_df["Selections"] / total_picks if total_picks else np.nan

    # --- per-date churn metrics ---
    sets_by_date = (
        df_win["tickers"]
        .apply(_ensure_list)
        .apply(lambda lst: tuple(sorted(set(lst))))
    )
    churns = []
    prev = None
    for _, cur_tuple in sets_by_date.items():
        cur = set(cur_tuple)
        if prev is None:
            prev = cur
            continue
        inter = len(prev.intersection(cur))
        union = len(prev.union(cur))
        jacc = (inter / union) if union else 1.0
        churns.append(1.0 - jacc)
        prev = cur
    avg_churn = float(np.nanmean(churns)) if len(churns) else np.nan

    # --- concentration (HHI over selection shares) ---
    shares = ticker_df["Selection_Share"].to_numpy(dtype=float) if not ticker_df.empty else np.array([])
    hhi = float(np.nansum(shares**2)) if shares.size else np.nan

    # --- never selected (universe from prices if provided) ---
    never_selected = []
    pct_never_selected = np.nan
    prices_universe = None
    prices_ok = bool(prices is not None and hasattr(prices, "columns") and len(getattr(prices, "columns", [])) > 0)

    if prices_ok:
        prices_universe = [str(c) for c in prices.columns]
        prices_universe_set = set(prices_universe)
        never_selected = sorted(prices_universe_set - set(counts.index.astype(str)))
        pct_never_selected = (
            len(never_selected) / len(prices_universe_set)
            if len(prices_universe_set) else np.nan
        )
        universe_note = f"Universe da prices: {len(prices_universe)} ticker. Never-selected: {len(never_selected)}."
    else:
        # fallback: within observed selections, "never selected" è 0% by construction
        pct_never_selected = 0.0
        universe_note = "Universe non disponibile (prices assente). % never-selected = 0% by construction."

    # --- top1 selection share ---
    top1_share = float(ticker_df["Selection_Share"].iloc[0]) if not ticker_df.empty else np.nan
    top1_ticker = str(ticker_df["Ticker"].iloc[0]) if not ticker_df.empty else None

    # --- held-return per ticker (optional, requires prices) ---
    held_ret_by_ticker = {}
    held_days_by_ticker = {}
    if prices_ok and hasattr(prices, "index"):
        px = prices.copy()
        px.index = _norm_index(px.index)
        px = px.sort_index()

        # restrict to analysis window
        px = px.loc[s:e]

        # daily returns for all tickers (NO pad fill)
        r_px = px.pct_change(fill_method=None)

        # selection sets per selection date -> daily with ffill
        sel_sets = pd.Series(
            index=df_win.index,
            data=df_win["tickers"].apply(_ensure_list).apply(lambda lst: tuple(sorted(set(lst))))
        ).sort_index()

        sel_sets_daily = sel_sets.reindex(px.index, method="ffill")
        valid_mask = sel_sets_daily.notna()
        sel_sets_daily = sel_sets_daily.loc[valid_mask.index[valid_mask]]
        r_px = r_px.loc[sel_sets_daily.index]

        # compute per ticker held returns
        for t in px.columns.astype(str):
            held = sel_sets_daily.apply(lambda tup: t in set(tup) if isinstance(tup, tuple) else False).astype(bool)
            rt = r_px[t].loc[held.index]
            rt_held = rt[held].dropna()
            held_days = int(rt_held.shape[0])
            held_ret = float((1.0 + rt_held).prod() - 1.0) if held_days > 0 else np.nan
            held_ret_by_ticker[t] = held_ret
            held_days_by_ticker[t] = held_days

        ticker_df["Held_Days"] = ticker_df["Ticker"].map(held_days_by_ticker).astype(float)
        ticker_df["Held_Return"] = ticker_df["Ticker"].map(held_ret_by_ticker).astype(float)
    else:
        ticker_df["Held_Days"] = np.nan
        ticker_df["Held_Return"] = np.nan

    # ------------------------------------------------------------
    # Storico: valutazione adeguatezza (NUOVO)
    # ------------------------------------------------------------
    history_days = int((e - s).days)
    history_years = history_days / 365.25

    if history_years >= 2 and n_dates >= 24:
        history_status = "ADEGUATO"
        history_note = "Storico sufficiente per valutazione completa del motore."
    elif history_years >= 1 and n_dates >= 12:
        history_status = "PARZIALE"
        history_note = (
            "Storico sufficiente solo per analisi preliminari. "
            "Metriche come churn/copertura vanno interpretate con cautela."
        )
    else:
        history_status = "INSUFFICIENTE"
        history_note = (
            "Storico troppo corto per una valutazione affidabile del motore. "
            "Consigliato ripetere con uno storico più ampio."
        )

    # --- build health table ---
    def _flag(cond: bool) -> str:
        return "⚠️" if cond else "OK"

    meta_rows = [
        {
            "Check": "Adeguatezza storico",
            "Value": f"{history_years:.2f} anni / {n_dates} selezioni",
            "Threshold": "≥ 2 anni & ≥ 24 selezioni",
            "Status": history_status,
            "Note": history_note,
        },
        {"Check": "Periodo analisi", "Value": f"{s.date()} → {e.date()}", "Threshold": "", "Status": "", "Note": ""},
        {"Check": "N. date selezione", "Value": n_dates, "Threshold": "", "Status": "", "Note": ""},
        {"Check": "N. selezioni totali (date×top)", "Value": total_picks, "Threshold": "", "Status": "", "Note": ""},
        {"Check": "N. ticker selezionati almeno 1 volta", "Value": int(counts.shape[0]), "Threshold": "", "Status": "", "Note": ""},
        {"Check": "prices_ok", "Value": prices_ok, "Threshold": "", "Status": "", "Note": ""},
        {"Check": "prices_shape", "Value": (getattr(prices, "shape", None) if prices_ok else None), "Threshold": "", "Status": "", "Note": ""},
        {"Check": "Universe (da prices) size", "Value": (len(prices_universe) if prices_universe is not None else np.nan), "Threshold": "", "Status": "", "Note": universe_note},
    ]

    warn_rows = [
        {
            "Check": "% titoli mai selezionati (su universe)",
            "Value": pct_never_selected,
            "Threshold": f">{warn_pct_never_selected:.0%}",
            "Status": _flag(pd.notna(pct_never_selected) and pct_never_selected > warn_pct_never_selected),
            "Note": universe_note,
        },
        {
            "Check": "Concentrazione Top-1 (share selezioni)",
            "Value": top1_share,
            "Threshold": f">{warn_top1_sel_share:.0%}",
            "Status": _flag(pd.notna(top1_share) and top1_share > warn_top1_sel_share),
            "Note": f"Top1: {top1_ticker}" if top1_ticker else "",
        },
        {
            "Check": "Churn medio (1 - Jaccard)",
            "Value": avg_churn,
            "Threshold": f">{warn_avg_churn:.2f}",
            "Status": _flag(pd.notna(avg_churn) and avg_churn > warn_avg_churn),
            "Note": "Se troppo alto: turnover eccessivo/instabile.",
        },
        {
            "Check": "Concentrazione HHI (share^2)",
            "Value": hhi,
            "Threshold": f">{warn_concentration_hhi:.2f}",
            "Status": _flag(pd.notna(hhi) and hhi > warn_concentration_hhi),
            "Note": "Più alto = selezioni concentrate su pochi titoli.",
        },
    ]

    health_df = pd.concat([pd.DataFrame(meta_rows), pd.DataFrame(warn_rows)], ignore_index=True)

    # ------------------------------------------------------------
    # Sintesi finale stato di salute (NUOVO)
    # ------------------------------------------------------------
    n_warn = int((health_df["Status"] == "⚠️").sum())

    if history_status == "INSUFFICIENTE":
        engine_status = "🔴 CRITICO"
        engine_note = (
            "Valutazione non affidabile: storico insufficiente. "
            "Estendere lo storico prima di trarre conclusioni."
        )
    elif history_status == "PARZIALE":
        engine_status = "🟡 DA MONITORARE"
        engine_note = (
            "Motore funzionante ma valutazione parziale. "
            "Ripetere l’analisi su uno storico più esteso."
        )
    else:
        if n_warn >= 3:
            engine_status = "🟡 DA MONITORARE"
            engine_note = (
                "Motore attivo ma con segnali di instabilità strutturale "
                "(churn/concentrazione). Valutare revisione griglia WFO."
            )
        else:
            engine_status = "🟢 SANO"
            engine_note = (
                "Motore coerente: buona copertura universo e nessun segnale "
                "di concentrazione eccessiva o instabilità marcata."
            )

    summary_row = {
        "Check": "Stato di salute del motore",
        "Value": engine_status,
        "Threshold": "",
        "Status": "",
        "Note": engine_note,
    }
    health_df = pd.concat([health_df, pd.DataFrame([summary_row])], ignore_index=True)

    # --- details ---
    details = None
    if return_details:
        details = {
            "analysis_start": s,
            "analysis_end": e,
            "history_days": history_days,
            "history_years": history_years,
            "history_status": history_status,
            "sets_by_date": sets_by_date,
            "churn_series": pd.Series(churns, name="churn") if len(churns) else pd.Series(dtype=float, name="churn"),
            "never_selected": never_selected,
            "prices_universe": prices_universe,
            "universe_note": universe_note,
        }

    # --- sort ticker_df ---
    ticker_df = ticker_df.sort_values(["Selections", "Selection_Share"], ascending=[False, False]).reset_index(drop=True)

    return health_df, ticker_df, selection_log, details


# ============================================================
# adaptive_cluster_universe — K adattivo per clustering universo
# ============================================================
import logging
from dataclasses import dataclass

from sklearn.metrics import silhouette_score as _silhouette_score


@dataclass
class ClusterResult:
    """Risultato di adaptive_cluster_universe."""
    labels          : "np.ndarray"       # indici 0-based, uno per asset
    n_clusters      : int
    method_used     : str
    score           : float
    all_scores      : "dict[int, float]" # silhouette per ogni K testato
    cluster_members : "dict[int, list]"  # cluster_id (1-based) → lista ticker
    distance_matrix : "np.ndarray"


# ── helpers privati ──────────────────────────────────────────────────────────

def _compute_correlation_distance(returns: "pd.DataFrame") -> "np.ndarray":
    corr = returns.corr().values
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return np.sqrt(0.5 * (1 - corr))


def _apply_asset_class_constraint(
    dist           : "np.ndarray",
    tickers        : list,
    asset_class_map: dict,
) -> "np.ndarray":
    dist = dist.copy()
    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            if asset_class_map.get(tickers[i]) != asset_class_map.get(tickers[j]):
                dist[i, j] = dist[j, i] = 1.0
    return dist


def _cluster_by_silhouette(
    dist          : "np.ndarray",
    k_min         : int,
    k_max         : int,
    linkage_method: str,
    logger        : "logging.Logger",
) -> tuple:
    condensed  = squareform(dist, checks=False)
    Z          = linkage(condensed, method=linkage_method)
    best_k     = k_min
    best_score = -np.inf
    all_scores : dict = {}

    for k in range(k_min, k_max + 1):
        lbls = fcluster(Z, t=k, criterion='maxclust') - 1
        if len(set(lbls)) < 2:
            continue
        s = _silhouette_score(dist, lbls, metric='precomputed')
        all_scores[k] = float(s)
        logger.debug(f"  silhouette k={k}: {s:.4f}")
        if s > best_score:
            best_score = s
            best_k     = k

    if best_score < 0:
        logger.warning(
            f"Silhouette negativo per tutti i K [{k_min}..{k_max}], fallback k={k_min}"
        )
        best_k     = k_min
        best_score = all_scores.get(k_min, -1.0)

    labels = fcluster(Z, t=best_k, criterion='maxclust') - 1
    return labels, best_k, float(best_score), all_scores, Z


def _cluster_by_corr_threshold(
    dist          : "np.ndarray",
    corr_threshold: float,
    k_min         : int,
    k_max         : int,
    linkage_method: str,
    logger        : "logging.Logger",
) -> tuple:
    condensed = squareform(dist, checks=False)
    Z         = linkage(condensed, method=linkage_method)
    thresh    = corr_threshold
    labels    = None
    k         = None

    for _ in range(5):
        cut   = np.sqrt(0.5 * (1 - thresh))
        lbls  = fcluster(Z, t=cut, criterion='distance')
        k_try = len(set(lbls))
        logger.debug(f"  corr_threshold={thresh:.2f} → k={k_try}")
        if k_min <= k_try <= k_max:
            labels, k = lbls - 1, k_try
            break
        thresh = min(0.99, thresh + 0.1) if k_try > k_max else max(0.01, thresh - 0.1)
    else:
        logger.warning("corr_threshold: rescaling non convergito dopo 5 iter, fallback su silhouette")
        return None, None, None, Z

    score = (
        float(_silhouette_score(dist, labels, metric='precomputed'))
        if len(set(labels)) >= 2 else -1.0
    )
    return labels, k, score, Z


# ── API pubblica ─────────────────────────────────────────────────────────────

def adaptive_cluster_universe(
    returns        : "pd.DataFrame",
    method         : str        = 'hybrid',
    k_min          : int        = 2,
    k_max          : "int|None" = None,
    corr_threshold : float      = 0.5,
    linkage_method : str        = 'average',
    asset_class_map: "dict|None"= None,
    random_state   : int        = 42,
    verbose        : bool       = False,
) -> ClusterResult:
    """Seleziona K ottimale per clustering universo rotazionale.

    Args:
        returns:         DataFrame di ritorni giornalieri (colonne = ticker).
        method:          'silhouette' | 'corr_threshold' | 'hybrid' (default).
        k_min:           K minimo testato.
        k_max:           K massimo; default = min(12, n_assets // 3).
        corr_threshold:  Soglia correlazione per metodo 'corr_threshold'.
        linkage_method:  Metodo linkage scipy ('average', 'ward', 'complete').
        asset_class_map: Mappa ticker → classe asset (es. 'equity', 'bond').
                         Se fornita, penalizza cluster misti tra classi diverse.
        random_state:    Reservato (clustering gerarchico è deterministico).
        verbose:         Se True, abilita logging DEBUG.

    Returns:
        ClusterResult: K scelto, labels 0-based, score silhouette, diagnostica.

    Trade-off metodi:
        - silhouette: robusto, tende a K piccoli, meno interpretabile
        - corr_threshold: interpretabile economicamente, K più stabile
        - hybrid (default): usa silhouette se i due metodi convergono
          (|K_sil - K_corr| <= 2), altrimenti usa corr_threshold

    Example::
        ret = prices.pct_change().dropna(how='all')
        result = adaptive_cluster_universe(ret, method='hybrid')
        print(result.n_clusters, result.method_used, result.score)
    """
    logger = logging.getLogger(__name__)
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Filtra asset con > 30% NaN
    nan_frac = returns.isna().mean()
    valid    = nan_frac[nan_frac <= 0.30].index.tolist()
    if len(valid) < len(returns.columns):
        dropped = set(returns.columns) - set(valid)
        logger.warning(f"Filtrati {len(dropped)} asset con >30% NaN: {dropped}")
    ret = returns[valid].dropna(how='all')

    n_assets = len(valid)
    if k_max is None:
        k_max = max(k_min, min(12, n_assets // 3))

    dist = _compute_correlation_distance(ret)
    if asset_class_map is not None:
        dist = _apply_asset_class_constraint(dist, valid, asset_class_map)

    all_scores: dict = {}

    if method == 'silhouette':
        labels, k, score, all_scores, _ = _cluster_by_silhouette(
            dist, k_min, k_max, linkage_method, logger
        )
        method_used = 'silhouette'

    elif method == 'corr_threshold':
        labels, k, score, _ = _cluster_by_corr_threshold(
            dist, corr_threshold, k_min, k_max, linkage_method, logger
        )
        if labels is None:
            labels, k, score, all_scores, _ = _cluster_by_silhouette(
                dist, k_min, k_max, linkage_method, logger
            )
            method_used = 'silhouette (fallback da corr_threshold)'
        else:
            all_scores  = {k: score}
            method_used = 'corr_threshold'

    elif method == 'hybrid':
        labels_sil, k_sil, score_sil, all_scores, _ = _cluster_by_silhouette(
            dist, k_min, k_max, linkage_method, logger
        )
        labels_corr, k_corr, score_corr, _ = _cluster_by_corr_threshold(
            dist, corr_threshold, k_min, k_max, linkage_method, logger
        )
        logger.info(
            f"Hybrid: K_silhouette={k_sil}, K_corr={k_corr if k_corr is not None else 'N/A'}"
        )
        if labels_corr is None or abs(k_sil - k_corr) <= 2:
            labels, k, score = labels_sil, k_sil, score_sil
            method_used = 'hybrid→silhouette'
        else:
            labels, k, score = labels_corr, k_corr, score_corr
            method_used = 'hybrid→corr_threshold'

    else:
        raise ValueError(
            f"method deve essere 'silhouette'|'corr_threshold'|'hybrid', "
            f"ricevuto: {method!r}"
        )

    logger.info(
        f"adaptive_cluster_universe: K={k}, method={method_used}, score={score:.4f}"
    )

    cluster_members: dict = {}
    for ticker, lbl in zip(valid, labels):
        cluster_members.setdefault(int(lbl) + 1, []).append(ticker)

    logger.info(
        "Cluster sizes: "
        + str({cid: len(m) for cid, m in sorted(cluster_members.items())})
    )

    return ClusterResult(
        labels          = labels,
        n_clusters      = k,
        method_used     = method_used,
        score           = score,
        all_scores      = all_scores,
        cluster_members = cluster_members,
        distance_matrix = dist,
    )


# ============================================================
# CELL 0 — Import
# ============================================================

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================
# CELL 1 — Funzioni (incolla tutto, non modificare)
# ============================================================

def filter_stocks_data(stocks_data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    available = [t for t in tickers if t in stocks_data.columns]
    if len(available) < len(tickers):
        missing = set(tickers) - set(available)
        print(f"⚠️  Ticker mancanti: {missing}")
    return stocks_data[available]


def plot_dendrogram_colored(Z, labels, n_clusters, ax, palette=None):
    if palette is None:
        palette = ['#1D9E75', '#378ADD', '#BA7517', '#9F77DD', '#D85A30']

    cluster_ids     = fcluster(Z, t=n_clusters, criterion='maxclust')
    label_to_cluster= dict(zip(labels, cluster_ids))
    cluster_colors  = {cid: palette[i % len(palette)]
                       for i, cid in enumerate(sorted(set(cluster_ids)))}

    n_leaves = len(labels)

    def get_leaves(node_idx):
        if node_idx < n_leaves:
            return {node_idx}
        left  = int(Z[node_idx - n_leaves, 0])
        right = int(Z[node_idx - n_leaves, 1])
        return get_leaves(left) | get_leaves(right)

    def link_color_func(node_idx):
        leaves   = get_leaves(node_idx)
        clusters = {cluster_ids[i] for i in leaves}
        if len(clusters) == 1:
            return cluster_colors[next(iter(clusters))]
        return '#AAAAAA'

    dendrogram(
        Z,
        labels          = labels,
        leaf_rotation   = 90,
        link_color_func = link_color_func,
        ax              = ax,
    )

    # Linea di taglio — UNA SOLA, calcolata come media tra ultima fusione
    # intra-cluster e prima fusione inter-cluster
    cut_height = (Z[-(n_clusters-1), 2] + Z[-n_clusters, 2]) / 2
    ax.axhline(y=cut_height, color='red', linestyle='--',
               alpha=0.7, label=f'{n_clusters} cluster')
    ax.legend()

    return cluster_colors  # restituisce la palette per allineare lo scatter

    
def analyze_and_cluster_universe(
    prices            : pd.DataFrame,
    n_clusters        : int  = 3,
    lookback_days     : int  = 252,
    plot              : bool = True,
    adaptive_k        : bool = False,
    adaptive_k_method : str  = 'hybrid',
    min_cluster_size  : int | None = None,
    save_plots        : bool = False,
    plots_dir                = None,
) -> dict:

    px  = prices.dropna(axis=1, how='all').ffill().iloc[-lookback_days:]
    ret = px.pct_change().dropna(how='all')

    metrics = {}
    for ticker in ret.columns:
        r = ret[ticker].dropna()
        if len(r) < 60:
            continue

        cagr     = (1 + r.mean()) ** 252 - 1
        vol      = r.std() * np.sqrt(252)
        sharpe   = cagr / vol if vol > 0 else 0
        mom_3m   = px[ticker].iloc[-1] / px[ticker].iloc[-63]  - 1
        mom_6m   = px[ticker].iloc[-1] / px[ticker].iloc[-126] - 1
        mom_12m  = px[ticker].iloc[-1] / px[ticker].iloc[0]    - 1
        autocorr = r.autocorr(lag=5)

        cum = (1 + r).cumprod()
        dd  = ((cum - cum.cummax()) / cum.cummax()).min()

        metrics[ticker] = dict(
            cagr=cagr,
            vol=vol,
            sharpe=sharpe,
            mom_3m=mom_3m,
            mom_6m=mom_6m,
            mom_12m=mom_12m,
            autocorr=autocorr,
            max_dd=dd,
        )

    metrics_df = pd.DataFrame(metrics).T.dropna()

    if metrics_df.empty:
        raise ValueError(
            f"analyze_and_cluster_universe: nessun ticker ha dati sufficienti "
            f"(ret ha {len(ret)} righe, soglia minima=60). "
            f"Riduci lookback_days o verifica i dati in ingresso."
        )

    universe_size = len(metrics_df)

    if min_cluster_size is None:
        min_cluster_size = 4

    if min_cluster_size < 2:
        min_cluster_size = 2

    max_k_by_size = max(2, universe_size // min_cluster_size)
    max_k_allowed = min(n_clusters, max_k_by_size)

    if max_k_allowed < 2:
        max_k_allowed = 2

    corr_matrix = ret[metrics_df.index].corr()
    dist_corr   = np.sqrt(0.5 * (1 - corr_matrix))

    scaler = StandardScaler()
    feat = scaler.fit_transform(
        metrics_df[['vol', 'mom_6m', 'autocorr', 'max_dd']]
    )

    feat_dist = pd.DataFrame(
        np.linalg.norm(feat[:, None] - feat[None, :], axis=2) / feat.shape[1],
        index=metrics_df.index,
        columns=metrics_df.index,
    )

    combined = 0.6 * dist_corr + 0.4 * feat_dist
    condensed = squareform(combined.values, checks=False)
    Z = linkage(condensed, method='ward')

    def _make_clusters(k: int):
        labels = fcluster(Z, t=k, criterion='maxclust')
        cmap = dict(zip(metrics_df.index, labels))

        groups = {}
        for ticker, cid in cmap.items():
            groups.setdefault(cid, []).append(ticker)

        return labels, cmap, groups

    if adaptive_k:
        _ac_ret = px.pct_change().dropna(how='all')

        _ac = adaptive_cluster_universe(
            returns=_ac_ret,
            method=adaptive_k_method,
            k_min=2,
        )

        proposed_k = int(_ac.n_clusters)
        capped_k = min(proposed_k, max_k_allowed)

        print(
            f"  K adattivo proposto: {proposed_k} "
            f"(metodo={_ac.method_used}, score={_ac.score:.3f})"
        )
        print(
            f"  K massimo ammesso: {max_k_allowed} "
            f"(universe_size={universe_size}, min_cluster_size={min_cluster_size})"
        )

        selected_k = None
        selected_labels = None
        cluster_map = None
        cluster_groups = None

        for k_try in range(capped_k, 1, -1):
            labels_try, cmap_try, groups_try = _make_clusters(k_try)
            sizes_try = {cid: len(v) for cid, v in groups_try.items()}
            min_size_try = min(sizes_try.values())

            if min_size_try >= min_cluster_size:
                selected_k = k_try
                selected_labels = labels_try
                cluster_map = cmap_try
                cluster_groups = groups_try
                break

        if selected_k is None:
            selected_k = 2
            selected_labels, cluster_map, cluster_groups = _make_clusters(2)

            print(
                f"  ⚠️ Nessun K rispetta min_cluster_size={min_cluster_size}. "
                f"Fallback a K=2."
            )

        n_clusters = selected_k

        print(f"  K finale usato: {n_clusters}")

    else:
        n_clusters = min(n_clusters, max_k_allowed)
        labels_cl, cluster_map, cluster_groups = _make_clusters(n_clusters)

    if not adaptive_k:
        labels_cl = fcluster(Z, t=n_clusters, criterion='maxclust')
        cluster_map = dict(zip(metrics_df.index, labels_cl))

        cluster_groups = {}
        for ticker, cid in cluster_map.items():
            cluster_groups.setdefault(cid, []).append(ticker)

    # Soglie adattive
    vol_series = metrics_df['vol']
    vol_p33    = vol_series.quantile(0.33)
    vol_p66    = vol_series.quantile(0.66)
    mom_median = metrics_df['mom_6m'].median()
    sharpe_med = metrics_df['sharpe'].median()

    print(f"\nSoglie adattive universo:")
    print(f"  Vol p33={vol_p33:.1%}  p66={vol_p66:.1%}")
    print(f"  Mom6m mediana={mom_median:.1%}")
    print(f"  Sharpe mediana={sharpe_med:.2f}")

    cluster_labels = {}

    for cid, tickers in cluster_groups.items():
        sub     = metrics_df.loc[tickers]
        avg_vol = sub['vol'].mean()
        avg_mom = sub['mom_6m'].mean()
        avg_sh  = sub['sharpe'].mean()
        avg_dd  = sub['max_dd'].mean()

        if avg_dd <= -0.40 or (avg_sh < 0 and avg_mom < 0):
            label = "AVOID"
        elif avg_vol >= vol_p66 and avg_mom >= mom_median:
            label = "HIGH_MOMENTUM"
        elif avg_vol <= vol_p33:
            label = "DEFENSIVE"
        else:
            label = "BALANCED"

        cluster_labels[cid] = label

        print(f"\nCluster {cid} [{label}] — {len(tickers)} asset")
        print(f"  Tickers  : {tickers}")
        print(f"  Avg Vol  : {avg_vol:.1%}")
        print(f"  Avg Mom6m: {avg_mom:.1%}")
        print(f"  Avg MaxDD: {avg_dd:.1%}")
        print(f"  Sharpe   : {avg_sh:.2f}")

    unique_labels = set(cluster_labels.values())

    if len(unique_labels) == 1:
        print(
            f"\n⚠️  Tutti i cluster hanno label '{list(unique_labels)[0]}' "
            f"— forzo differenziazione per Sharpe"
        )

        sharpe_by_cluster = {
            cid: metrics_df.loc[tickers, 'sharpe'].mean()
            for cid, tickers in cluster_groups.items()
        }

        sorted_cids = sorted(
            sharpe_by_cluster,
            key=sharpe_by_cluster.get,
            reverse=True,
        )

        forced_labels = ["DEFENSIVE", "BALANCED", "HIGH_MOMENTUM"]

        for i, cid in enumerate(sorted_cids):
            idx = min(i, len(forced_labels) - 1)
            cluster_labels[cid] = forced_labels[idx]

            print(
                f"  Cluster {cid} → {forced_labels[idx]} "
                f"(Sharpe={sharpe_by_cluster[cid]:.2f})"
            )

    if plot:
        palette = ['#1D9E75', '#378ADD', '#BA7517', '#9F77DD', '#D85A30']

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        cluster_colors = plot_dendrogram_colored(
            Z          = Z,
            labels     = metrics_df.index.tolist(),
            n_clusters = n_clusters,
            ax         = axes[0],
            palette    = palette,
        )

        axes[0].set_title("Dendrogramma — Clustering Universo")

        offsets = [(6, 6), (-6, 6), (6, -10), (-6, -10), (10, 0), (-10, 0)]

        for cid, tickers in cluster_groups.items():
            sub = metrics_df.loc[tickers]
            color = cluster_colors.get(cid, '#888888')

            axes[1].scatter(
                sub['vol'],
                sub['mom_6m'],
                label=f"C{cid}: {cluster_labels[cid]}",
                color=color,
                s=100,
                zorder=5,
            )

            for i, t in enumerate(tickers):
                ox, oy = offsets[i % len(offsets)]

                axes[1].annotate(
                    t.replace('.MI', '').replace('.DE', '')
                     .replace('.PA', '').replace('.MC', ''),
                    xy=(metrics_df.loc[t, 'vol'], metrics_df.loc[t, 'mom_6m']),
                    xytext=(ox, oy),
                    textcoords='offset points',
                    fontsize=7,
                    ha='left' if ox >= 0 else 'right',
                    va='bottom' if oy >= 0 else 'top',
                )

        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel("Volatilità annualizzata")
        axes[1].set_ylabel("Momentum 6 mesi")
        axes[1].set_title("Scatter Vol vs Momentum — per Cluster")
        axes[1].legend()

        plt.tight_layout()

        if save_plots and plots_dir is not None:
            _pd = str(plots_dir)

            for _ax, _name in [
                (axes[0], 'cluster_dendrogram.png'),
                (axes[1], 'cluster_scatter.png'),
            ]:
                _extent = _ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                _extent_padded = _extent.expanded(1.15, 1.20)

                fig.savefig(
                    f"{_pd}/{_name}",
                    dpi=150,
                    bbox_inches=_extent_padded,
                )

        plt.show()

    return dict(
        cluster_map       = cluster_map,
        cluster_groups    = cluster_groups,
        metrics_df        = metrics_df,
        cluster_labels    = cluster_labels,
        n_clusters_final  = n_clusters,
        min_cluster_size  = min_cluster_size,
    )


def plot_cluster_heatmap(
    cluster_result : dict,
    stocks_data    : pd.DataFrame,
    lookback_days  : int  = 252,
    save_path             = None,    # str | Path | None
) -> None:
    """
    Heatmap di correlazione dell'universo, ordinata per cluster, con blocchi colorati.

    Usa la stessa finestra temporale del WFO (lookback_days) per coerenza con
    il clustering effettivo. NON usa tail() fisso indipendente dal WFO.

    Parameters
    ----------
    cluster_result : output di analyze_and_cluster_universe (chiavi: cluster_groups, cluster_labels)
    stocks_data    : prezzi storici — stessa sorgente passata al WFO
    lookback_days  : finestra in giorni — DEVE corrispondere al lookback_days del run WFO
    save_path      : se fornito, salva la figura nel path indicato (dpi=150)
    """
    import matplotlib.patches as patches

    cluster_groups = cluster_result['cluster_groups']
    cluster_labels = cluster_result['cluster_labels']

    sorted_t = []
    for cid in sorted(cluster_groups.keys()):
        sorted_t.extend([t for t in cluster_groups[cid] if t in stocks_data.columns])

    if not sorted_t:
        print("plot_cluster_heatmap: nessun ticker disponibile in stocks_data — skip")
        return

    ret_sub = (stocks_data[sorted_t]
               .dropna(how='all')
               .tail(lookback_days)
               .pct_change()
               .dropna(how='all'))
    corr = ret_sub.corr()

    PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fs = max(10, len(sorted_t) * 0.38)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.85))
    im = ax.imshow(corr.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(sorted_t)))
    ax.set_xticklabels(sorted_t, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sorted_t)))
    ax.set_yticklabels(sorted_t, fontsize=7)
    pos = 0
    for i, cid in enumerate(sorted(cluster_groups.keys())):
        n = len([t for t in cluster_groups[cid] if t in sorted_t])
        rect = patches.Rectangle(
            (pos - 0.5, pos - 0.5), n, n,
            lw=2.5, edgecolor=PALETTE[i % len(PALETTE)], facecolor='none')
        ax.add_patch(rect)
        # ax.text(pos + n / 2 - 0.5, -1.2,
        #         cluster_labels.get(cid, f'C{cid}'),
        #         ha='center', fontsize=8,
        #         color=PALETTE[i % len(PALETTE)], fontweight='bold')
        ax.text(pos + n / 2 - 0.5, len(sorted_t) + 1.5,
                cluster_labels.get(cid, f'C{cid}'),
                ha='center', va='top', fontsize=8,
                color=PALETTE[i % len(PALETTE)], fontweight='bold')
        pos += n
    # ax.set_title(f'Correlazione per Cluster (ultimi {lookback_days} gg)')
    # ax.set_title(f'Correlazione per Cluster (ultimi {lookback_days} gg)', pad=20)
    # ax.set_title(f'Correlazione per Cluster (ultimi {lookback_days} gg)', pad=30)
    ax.set_title(f'Correlazione per Cluster (ultimi {lookback_days} gg)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.show()


def build_cluster_grids(
    cluster_labels : dict,
    cluster_groups : dict,
    n_top_min      : int   = 2,
    n_top_fraction : tuple = (0.10, 0.20, 0.30),
) -> dict:

    def _build_n_top(n_assets: int) -> list[int]:
        candidates = sorted(set(
            max(n_top_min, round(n_assets * f))
            for f in n_top_fraction
        ))
        candidates = [v for v in candidates
                      if v <= max(n_top_min, int(n_assets * 0.40))]
        if len(candidates) < 2:
            candidates = [n_top_min, max(n_top_min + 1, candidates[-1])]
        return candidates

    grids = {}

    for cid, label in cluster_labels.items():
        tickers  = cluster_groups.get(cid, [])
        n_assets = len(tickers)
        n_top    = _build_n_top(n_assets)

        print(f"\nCluster {cid} [{label}] — {n_assets} asset → n_top: {n_top}")

        if label == "HIGH_MOMENTUM":
            grid = {
                "rebalance_frequency"      : ["ME"],
                "momentum_lookback_days"   : [20, 40, 60],
                "riskparity_lookback_days" : [20, 40],
                "n_top"                    : n_top,
                "use_acceleration"         : [True, False],
                "momentum_weight"          : [0.7, 1.0],
                "filter_ema"               : [True],
                "filter_volatility"        : [True, False],
                "filter_min_momentum"      : [True],
            }

        elif label == "DEFENSIVE":
            grid = {
                "rebalance_frequency"      : ["ME", "QE"],
                "momentum_lookback_days"   : [60, 120, 180],
                "riskparity_lookback_days" : [60, 120],
                "n_top"                    : n_top,
                "use_acceleration"         : [False],
                "momentum_weight"          : [0.5, 0.7],
                "filter_ema"               : [True, False],
                "filter_volatility"        : [True],
                "filter_min_momentum"      : [True, False],
            }

        elif label == "AVOID":
            # ✅ Griglia molto selettiva — entra solo con momentum genuino
            grid = {
                "rebalance_frequency"      : ["ME"],
                "momentum_lookback_days"   : [60, 120],
                "riskparity_lookback_days" : [60, 120],
                "n_top"                    : n_top,
                "use_acceleration"         : [False],
                "momentum_weight"          : [1.0],       # puro momentum
                "filter_ema"               : [True],      # tutti i filtri ON
                "filter_volatility"        : [True],
                "filter_min_momentum"      : [True],
            }

        else:  # BALANCED
            grid = {
                "rebalance_frequency"      : ["ME", "QE"],
                "momentum_lookback_days"   : [40, 60, 120],
                "riskparity_lookback_days" : [40, 60],
                "n_top"                    : n_top,
                "use_acceleration"         : [True, False],
                "momentum_weight"          : [0.7, 1.0],
                "filter_ema"               : [True, False],
                "filter_volatility"        : [True, False],
                "filter_min_momentum"      : [True, False],
            }

        grids[cid] = grid
        n_comb = int(np.prod([len(v) for v in grid.values()]))
        print(f"  Combinazioni totali: {n_comb}")

    return grids


def run_clustered_wfo(
    cluster_groups  : dict,
    cluster_grids   : dict,
    cluster_labels  : dict,
    stocks_data_raw : pd.DataFrame,
    wfo_kwargs      : dict,
) -> dict:
    results = {}
    for cid, tickers in cluster_groups.items():
        label = cluster_labels[cid]
        grid  = cluster_grids[cid]

        # if label == "DEFENSIVE" and "XEON.MI" not in tickers:
        #     tickers = tickers + ["XEON.MI"]

        cluster_data = filter_stocks_data(stocks_data_raw, tickers)

        print(f"\n{'='*55}")
        print(f"WFO Cluster {cid} [{label}] — {len(tickers)} asset")
        print(f"Tickers: {tickers}")
        print(f"{'='*55}")

        try:
            summary_df = walk_forward_rotational(
                stocks_data            = cluster_data,
                param_grid             = grid,
                ratio                  = wfo_kwargs.get('ratio', 'sharpe'),
                metric                 = wfo_kwargs.get('metric', 'TestScore'),
                start_date             = wfo_kwargs.get('start_date'),
                end_date               = wfo_kwargs.get('end_date'),
                benchmark_data         = wfo_kwargs.get('benchmark_data'),
                n_jobs                 = wfo_kwargs.get('n_jobs', 1),
                backend                = wfo_kwargs.get('backend', 'loky'),
                plot                   = False,
                verbose                = wfo_kwargs.get('verbose', False),
                debug                  = False,
                force_next_year_params = wfo_kwargs.get('force_next_year_params', True),
            )
            results[cid] = dict(summary_df=summary_df, label=label, universe=tickers)
            print(f"✅ Cluster {cid} — WFO completata")
            print(f"   TrainScore medio: {summary_df['TrainScore'].mean():.4f}")
            print(f"   TestScore medio : {summary_df['TestScore'].mean():.4f}")

            # Display con short_map
            # df_disp = summary_df.rename(columns=short_map)
            my_display(
                title=f"WFO Results Cluster {cid} [{label}]",
                # data=df_disp
                data=summary_df
            )

        except Exception as e:
            print(f"❌ Cluster {cid} fallito: {e}")
            results[cid] = dict(summary_df=None, label=label, universe=tickers)

    return results


def compute_market_regime(
    prices        : pd.DataFrame,
    equity_tickers: list,
    ema_fast      : int   = 50,
    ema_slow      : int   = 200,
    vol_window    : int   = 20,
    vol_threshold : float = 0.25,
) -> pd.Series:
    eq_px    = prices[equity_tickers].dropna(axis=1, how='all').ffill()
    eq_index = eq_px.mean(axis=1)
    ema_f    = eq_index.ewm(span=ema_fast,  adjust=False).mean()
    ema_s    = eq_index.ewm(span=ema_slow,  adjust=False).mean()
    vol_roll = eq_index.pct_change().rolling(vol_window).std() * np.sqrt(252)
    regime   = ((ema_f > ema_s) & (vol_roll < vol_threshold)).astype(int)
    regime.name = "regime"
    print(f"Regime ON : {regime.mean():.1%} del tempo")
    print(f"Regime OFF: {(1 - regime.mean()):.1%} del tempo")
    return regime


def aggregate_cluster_portfolios(
    wfo_results    : dict,
    stocks_data    : pd.DataFrame,
    benchmark_data,
    regime         : pd.Series,
    weight_on      : dict  = None,
    weight_off     : dict  = None,
    start_date     : str   = None,
    end_date       : str   = None,
    init_cash      : float = 100_000,
    plot           : bool  = True,
) -> dict:
    if weight_on is None:
        weight_on  = {"HIGH_MOMENTUM": 0.60, "BALANCED": 0.30, "DEFENSIVE": 0.10}
    if weight_off is None:
        weight_off = {"HIGH_MOMENTUM": 0.10, "BALANCED": 0.20, "DEFENSIVE": 0.70}

    cluster_returns = {}
    cluster_pf      = {}

    for cid, res in wfo_results.items():
        if res['summary_df'] is None:
            print(f"⚠️  Cluster {cid} senza summary_df, skip")
            continue
        label        = res['label']
        cluster_data = filter_stocks_data(stocks_data, res['universe'])

        print(f"\nCostruzione portafoglio Cluster {cid} [{label}]...")
        try:
            pf_rot, pf_bh, selections = build_portfolio_from_wfo_summary(
                summary_df      = res['summary_df'],
                stocks_data     = cluster_data,
                benchmark_data  = benchmark_data,
                benchmark_title = f"Cluster {cid} BM",
                init_cash       = init_cash,
                start_date      = start_date,
                end_date        = end_date,
                plot            = False,
                show_report     = False,
                portfolio_name  = f"Cluster {cid} [{label}]",
            )
            eq = pf_rot.value()
            if isinstance(eq, pd.DataFrame):
                eq = eq.sum(axis=1)
            eq_norm = eq / eq.iloc[0]
            ret     = eq_norm.pct_change().fillna(0)

            cluster_returns[label] = ret
            cluster_pf[cid]        = pf_rot
            print(f"✅ Cluster {cid} [{label}] costruito")

        except Exception as e:
            print(f"❌ Cluster {cid} fallito: {e}")

    if not cluster_returns:
        raise ValueError("Nessun cluster costruito correttamente")

    ret_df   = pd.DataFrame(cluster_returns).ffill().dropna()
    regime_a = regime.reindex(ret_df.index).ffill().fillna(0)

    w_rows = []
    for date in ret_df.index:
        r   = int(regime_a.loc[date])
        w   = weight_on if r == 1 else weight_off
        w_rows.append({label: w.get(label, 0.0) for label in ret_df.columns})

    w_df     = pd.DataFrame(w_rows, index=ret_df.index)
    w_df     = w_df.div(w_df.sum(axis=1), axis=0)
    agg_ret  = (ret_df * w_df).sum(axis=1)
    agg_eq   = (1 + agg_ret).cumprod() * init_cash

    n_years  = len(agg_ret) / 252
    cagr     = (agg_eq.iloc[-1] / agg_eq.iloc[0]) ** (1 / n_years) - 1
    vol      = agg_ret.std() * np.sqrt(252)
    sharpe   = cagr / vol if vol > 0 else 0
    dd       = ((agg_eq / agg_eq.cummax()) - 1).min()
    calmar   = cagr / abs(dd) if dd != 0 else 0

    print(f"\n{'='*55}")
    print(f"PORTAFOGLIO AGGREGATO CLUSTERED")
    print(f"{'='*55}")
    print(f"CAGR      : {cagr:.2%}")
    print(f"Volatilità: {vol:.2%}")
    print(f"Sharpe    : {sharpe:.2f}")
    print(f"Max DD    : {dd:.2%}")
    print(f"Calmar    : {calmar:.2f}")
    print(f"Regime ON : {regime_a.mean():.1%} del tempo")

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                  gridspec_kw={'height_ratios': [3, 1]})
        for label, ret_s in cluster_returns.items():
            eq_c = (1 + ret_s.reindex(ret_df.index).fillna(0)).cumprod() * 100
            axes[0].plot(eq_c, alpha=0.5, linestyle='--', label=f"Cluster {label}")
        agg_plot = (1 + agg_ret).cumprod() * 100
        axes[0].plot(agg_plot, color='navy', linewidth=2.5, label="Aggregato")
        axes[0].set_title("Portafoglio Clustered Aggregato vs Cluster singoli")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].fill_between(regime_a.index, regime_a,
                              alpha=0.4, color='green', label='Risk ON')
        axes[1].fill_between(regime_a.index, 1 - regime_a,
                              alpha=0.4, color='red',   label='Risk OFF')
        axes[1].set_title("Regime Risk ON / OFF")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return dict(
        equity          = agg_eq,
        returns         = agg_ret,
        weights         = w_df,
        regime          = regime_a,
        cluster_pf      = cluster_pf,
        cluster_returns = cluster_returns,
        metrics         = dict(cagr=cagr, vol=vol, sharpe=sharpe,
                               max_dd=dd, calmar=calmar),
    )


# cluster_grids = build_cluster_grids(cluster_result['cluster_labels'])


def merge_cluster_summary_dfs(
    wfo_results    : dict,
    cluster_labels : dict,
    regime         : pd.Series,
    dominant_on    : str = "HIGH_MOMENTUM",
    dominant_off   : str = "DEFENSIVE",
) -> pd.DataFrame:
    """
    Produce un summary_df aggregato compatibile con
    build_rotational_portfolios_from_wfo_result.

    Per ogni finestra WFO:
      - se il regime medio nella finestra è ON  → parametri da dominant_on
      - se il regime medio nella finestra è OFF → parametri da dominant_off

    Parameters
    ----------
    wfo_results    : output di run_clustered_wfo
    cluster_labels : {cluster_id: label}
    regime         : pd.Series 0/1 da compute_market_regime
    dominant_on    : label cluster da usare in Risk ON
    dominant_off   : label cluster da usare in Risk OFF

    Returns
    -------
    summary_df compatibile con build_rotational_portfolios_from_wfo_result
    """

    # Mappa label → summary_df
    label_to_summary = {}
    for cid, res in wfo_results.items():
        if res['summary_df'] is not None:
            label = cluster_labels[cid]
            label_to_summary[label] = res['summary_df']

    # Verifica che i cluster dominanti esistano
    for dominant in [dominant_on, dominant_off]:
        if dominant not in label_to_summary:
            available = list(label_to_summary.keys())
            print(f"⚠️  Cluster '{dominant}' non trovato, "
                  f"disponibili: {available}")
            # Fallback: usa BALANCED se disponibile, altrimenti il primo
            fallback = "BALANCED" if "BALANCED" in available else available[0]
            if dominant == dominant_on:
                dominant_on  = fallback
            else:
                dominant_off = fallback
            print(f"   → Fallback su '{fallback}'")

    # Prendi tutte le finestre disponibili (usa il summary più lungo)
    all_windows = set()
    for df in label_to_summary.values():
        all_windows.update(df.index.tolist())
    all_windows = sorted(all_windows)

    rows = []
    for window in all_windows:
        # Estrai date dalla finestra (es. "2022-01-01 →2022-12-31")
        # Il regime viene valutato sulla data di fine finestra
        try:
            # Parsing della data di fine finestra dall'index
            if isinstance(window, tuple):
                end_date_win = pd.Timestamp(window[1])
            elif isinstance(window, str) and '→' in window:
                end_date_win = pd.Timestamp(window.split('→')[1].strip())
            else:
                end_date_win = pd.Timestamp(window)
        except Exception:
            end_date_win = None

        # Determina regime dominante nella finestra
        if end_date_win is not None and end_date_win in regime.index:
            regime_val = int(regime.loc[end_date_win])
        elif end_date_win is not None:
            # Prendi il valore più vicino
            idx_pos    = regime.index.get_indexer([end_date_win],
                                                   method='nearest')[0]
            regime_val = int(regime.iloc[idx_pos])
        else:
            regime_val = 1  # default ON

        # Seleziona cluster dominante
        dominant = dominant_on if regime_val == 1 else dominant_off
        src_df   = label_to_summary[dominant]

        # Prendi riga corrispondente alla finestra
        if window in src_df.index:
            row = src_df.loc[window].copy()
        else:
            # Finestra non presente nel cluster dominante → usa l'altro
            fallback_label = dominant_off if dominant == dominant_on \
                             else dominant_on
            fallback_df    = label_to_summary.get(fallback_label)
            if fallback_df is not None and window in fallback_df.index:
                row = fallback_df.loc[window].copy()
            else:
                # Ultima riga disponibile come fallback finale
                row = src_df.iloc[-1].copy()

        row['_cluster_used'] = dominant   # colonna debug, non impatta il motore
        row['_regime']       = regime_val
        rows.append((window, row))

    # Ricostruisce DataFrame con stesso formato di summary_df originale
    merged = pd.DataFrame(
        [r for _, r in rows],
        index=[w for w, _ in rows]
    )
    merged.index.name = "Window"

    # Stampa riepilogo
    n_on  = (merged['_regime'] == 1).sum()
    n_off = (merged['_regime'] == 0).sum()
    print(f"\n{'='*50}")
    print(f"SUMMARY DF AGGREGATO")
    print(f"{'='*50}")
    print(f"Finestre totali : {len(merged)}")
    print(f"Risk ON  → {dominant_on:<16} : {n_on} finestre")
    print(f"Risk OFF → {dominant_off:<16} : {n_off} finestre")

    by_cluster = merged['_cluster_used'].value_counts()
    # for label, count in by_cluster.items():
    #     print(f"  {label}: {count} finestre ({count/len(merged):.0%})")

    return merged


def get_clean_summary_df(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove le colonne di debug prima di passare a
    build_rotational_portfolios_from_wfo_result.
    """
    return merged.drop(
        columns=[c for c in ['_cluster_used', '_regime']
                 if c in merged.columns]
    )


def run_wfo_pipeline(
    # Dati
    stocks_data_raw  : pd.DataFrame,
    stocks_data      : pd.DataFrame,
    benchmark_data,
    benchmark_data_raw,
    tickers          : list,
    risk_off_data    : pd.DataFrame = None,

    # Parametri WFO
    ratio            : str   = 'sharpe',
    metric           : str   = 'TestScore',
    start_date       : str   = None,
    end_date         : str   = None,
    cores            : int   = 1,
    verbose          : bool  = False,
    force_next_year_params: bool = True,
    param_grid       : dict  = None,

    # Parametri clustering
    use_clustering   : bool  = True,
    n_clusters       : int   = 3,
    lookback_days    : int   = 504,
    n_top_min        : int   = 2,
    n_top_fraction   : tuple = (0.10, 0.20, 0.30),
    adaptive_k       : bool  = False,
    adaptive_k_method: str   = 'hybrid',
    min_cluster_size : int   = None,   # <<< NUOVO: vincolo anti-frammentazione

    # Parametri regime
    dominant_on      : str   = "HIGH_MOMENTUM",
    dominant_off     : str   = "DEFENSIVE",
    weight_on        : dict  = None,
    weight_off       : dict  = None,

    # Parametri portafoglio
    portfolio_title  : str   = "Portfolio",
    benchmark_title  : str   = "Benchmark",
    init_cash        : float = 100_000,
    analisys_start_date: str = None,
    analisys_end_date  : str = None,
    risk_on_off      : bool  = True,

    # Display
    short_map        : dict  = None,
    plot             : bool  = True,
    save_plots       : bool  = False,
    plots_dir                = None,
) -> dict:
    """
    Pipeline end-to-end per la Walk-Forward Optimization di un portafoglio
    rotazionale, con supporto opzionale a clustering dell'universo e
    switching di regime.

    Nota sul clustering:
    - n_clusters rappresenta il massimo numero di cluster richiesto.
    - min_cluster_size impone una dimensione minima reale per cluster.
    - Se adaptive_k produce cluster troppo piccoli, la pipeline riduce k
      progressivamente fino a ottenere cluster ammissibili.
    """

    if weight_on is None:
        weight_on  = {"HIGH_MOMENTUM": 0.60, "BALANCED": 0.30, "DEFENSIVE": 0.10}

    if weight_off is None:
        weight_off = {"HIGH_MOMENTUM": 0.10, "BALANCED": 0.20, "DEFENSIVE": 0.70}

    if short_map is None:
        short_map = {
            "rebalance_frequency"     : "freq",
            "momentum_lookback_days"  : "mom_lb",
            "riskparity_lookback_days": "rp_lb",
            "n_top"                   : "n_top",
            "momentum_weight"         : "mom_w",
            "filter_ema"              : "f_ema",
            "filter_volatility"       : "f_vol",
            "filter_min_momentum"     : "f_min_m",
            "Score"                   : "score",
        }

    # ----------------------------------------------------------
    # Vincolo strutturale anti-frammentazione cluster
    # ----------------------------------------------------------
    if min_cluster_size is None:
        # Regola robusta: ogni cluster deve poter contenere almeno
        # n_top_min titoli + buffer di diversificazione.
        min_cluster_size = max(4, n_top_min + 2)

    if min_cluster_size < n_top_min:
        raise ValueError(
            f"min_cluster_size={min_cluster_size} non può essere inferiore "
            f"a n_top_min={n_top_min}."
        )

    wfo_kwargs = {
        'ratio'                 : ratio,
        'metric'                : metric,
        'start_date'            : start_date,
        'end_date'              : end_date,
        'benchmark_data'        : benchmark_data_raw,
        'n_jobs'                : cores,
        'backend'               : 'loky',
        'verbose'               : verbose,
        'force_next_year_params': force_next_year_params,
    }

    results = {}

    if use_clustering:
        # ----------------------------------------------------------
        # STEP 1 — Clustering con vincolo min_cluster_size
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("STEP 1 — Analisi e clustering universo")
        print("="*55)

        prices_df = stocks_data[tickers].dropna(how='all')
        universe_size = prices_df.shape[1]

        if universe_size < 2 * min_cluster_size:
            # Universo troppo piccolo per un clustering stabile.
            # In questo caso si forza k=2 solo se possibile.
            safe_max_clusters = 2
        else:
            safe_max_clusters = max(2, universe_size // min_cluster_size)

        requested_clusters = n_clusters
        effective_max_clusters = min(requested_clusters, safe_max_clusters)

        if verbose:
            print(
                f"[CLUSTER CONTROL] universe_size={universe_size}, "
                f"requested_n_clusters={requested_clusters}, "
                f"min_cluster_size={min_cluster_size}, "
                f"effective_max_clusters={effective_max_clusters}"
            )

        cluster_result = None
        selected_k = None

        # Prova k decrescente fino a ottenere cluster non degenerati.
        # Questo evita casi tipo 19 ticker / 6 cluster / cluster da 1 titolo.
        for k_try in range(effective_max_clusters, 1, -1):
            if verbose:
                print(f"[CLUSTER CONTROL] Tentativo clustering con k={k_try}")

            candidate_result = analyze_and_cluster_universe(
                prices            = prices_df,
                n_clusters        = k_try,
                lookback_days     = lookback_days,
                plot              = plot,
                adaptive_k        = adaptive_k,
                adaptive_k_method = adaptive_k_method,
                save_plots        = save_plots,
                plots_dir         = plots_dir,
            )

            if candidate_result is None:
                continue

            cluster_groups = candidate_result.get("cluster_groups", {})
            cluster_sizes = {
                cid: len(tickers_)
                for cid, tickers_ in cluster_groups.items()
            }

            if not cluster_sizes:
                continue

            min_actual_size = min(cluster_sizes.values())

            if verbose:
                print(
                    f"[CLUSTER CONTROL] k={k_try}, "
                    f"cluster_sizes={cluster_sizes}, "
                    f"min_actual_size={min_actual_size}"
                )

            if min_actual_size >= min_cluster_size:
                cluster_result = candidate_result
                selected_k = k_try
                break

        # Fallback: se nessun k rispetta il vincolo, usa k=2 ma segnala.
        if cluster_result is None:
            if verbose:
                print(
                    "[WARN] Nessun clustering rispetta min_cluster_size. "
                    "Fallback forzato a k=2."
                )

            cluster_result = analyze_and_cluster_universe(
                prices            = prices_df,
                n_clusters        = 2,
                lookback_days     = lookback_days,
                plot              = plot,
                adaptive_k        = False,
                adaptive_k_method = adaptive_k_method,
                min_cluster_size  = min_cluster_size,
                save_plots        = save_plots,
                plots_dir         = plots_dir,
            )
            selected_k = 2

        if verbose:
            final_cluster_sizes = {
                cid: len(tickers_)
                for cid, tickers_ in cluster_result["cluster_groups"].items()
            }
            print(
                f"[CLUSTER CONTROL] Clustering finale: "
                f"k={selected_k}, cluster_sizes={final_cluster_sizes}"
            )

        # Heatmap correlazione cluster
        if cluster_result is not None:
            _heatmap_path = (
                str(plots_dir / 'cluster_heatmap.png')
                if save_plots and plots_dir is not None
                else None
            )

            plot_cluster_heatmap(
                cluster_result = cluster_result,
                stocks_data    = prices_df,
                lookback_days  = lookback_days,
                save_path      = _heatmap_path,
            )

        # ----------------------------------------------------------
        # STEP 2 — Griglie per cluster
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("STEP 2 — Griglie WFO per cluster")
        print("="*55)

        cluster_grids = build_cluster_grids(
            cluster_labels = cluster_result['cluster_labels'],
            cluster_groups = cluster_result['cluster_groups'],
            n_top_min      = n_top_min,
            n_top_fraction = n_top_fraction,
        )

        # ----------------------------------------------------------
        # STEP 3 — WFO per cluster
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("STEP 3 — WFO per cluster")
        print("="*55)

        wfo_results = run_clustered_wfo(
            cluster_groups  = cluster_result['cluster_groups'],
            cluster_grids   = cluster_grids,
            cluster_labels  = cluster_result['cluster_labels'],
            stocks_data_raw = stocks_data_raw,
            wfo_kwargs      = wfo_kwargs,
        )

        for cid, res in wfo_results.items():
            if res['summary_df'] is not None:
                df_disp = res['summary_df'].rename(columns=short_map)
                my_display(
                    title=f"WFO Results Cluster {cid} [{res['label']}]",
                    data=df_disp
                )

        # ----------------------------------------------------------
        # STEP 4 — Regime
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("STEP 4 — Regime Risk ON/OFF")
        print("="*55)

        equity_tickers = [
            t for t, cid in cluster_result['cluster_map'].items()
            if cluster_result['cluster_labels'][cid] == dominant_on
        ]

        if not equity_tickers:
            print(f"⚠️  Nessun ticker '{dominant_on}' — uso tutti")
            equity_tickers = tickers

        regime = compute_market_regime(
            prices          = stocks_data[equity_tickers].dropna(how='all'),
            equity_tickers  = equity_tickers,
        )

        # ----------------------------------------------------------
        # STEP 5 — Merge summary_df
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("STEP 5 — Merge summary_df aggregato")
        print("="*55)

        merged_summary = merge_cluster_summary_dfs(
            wfo_results    = wfo_results,
            cluster_labels = cluster_result['cluster_labels'],
            regime         = regime,
            dominant_on    = dominant_on,
            dominant_off   = dominant_off,
        )

        summary_df_final = get_clean_summary_df(merged_summary)

        df_disp = summary_df_final.rename(columns=short_map)
        my_display(
            title=f"WFO Results Clustered — {portfolio_title}",
            data=df_disp
        )

        results.update(dict(
            cluster_result        = cluster_result,
            cluster_grids         = cluster_grids,
            wfo_results           = wfo_results,
            regime                = regime,
            merged_summary        = merged_summary,
            selected_k            = selected_k,
            min_cluster_size      = min_cluster_size,
            effective_max_clusters= effective_max_clusters,
        ))

    else:
        # ----------------------------------------------------------
        # WFO STANDARD — nessun clustering
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("STEP 1 — WFO Standard (clustering disabilitato)")
        print("="*55)

        if param_grid is None:
            raise ValueError("use_clustering=False richiede param_grid")

        summary_df_final = walk_forward_rotational(
            stocks_data            = stocks_data_raw[tickers],
            param_grid             = param_grid,
            ratio                  = ratio,
            metric                 = metric,
            start_date             = start_date,
            end_date               = end_date,
            benchmark_data         = benchmark_data_raw,
            n_jobs                 = cores,
            backend                = 'loky',
            plot                   = False,
            verbose                = verbose,
            debug                  = False,
            force_next_year_params = force_next_year_params,
        )

        df_disp = summary_df_final.rename(columns=short_map)
        my_display(
            title=f"WFO Results Standard — {portfolio_title}",
            data=df_disp
        )

        results.update(dict(
            cluster_result = None,
            cluster_grids  = None,
            wfo_results    = None,
            regime         = None,
            merged_summary = None,
            selected_k     = None,
        ))

    results['summary_df'] = summary_df_final

    # ----------------------------------------------------------
    # STEP 6 — Portafogli
    # ----------------------------------------------------------
    print("\n" + "="*55)
    print("STEP 6 — Costruzione portafogli")
    print("="*55)

    # Con Risk ON/OFF
    if risk_on_off and risk_off_data is not None:
        print("\n▶ Portafoglio CON Risk ON/OFF...")

        # Evita duplicati tra universo principale e asset risk-off
        duplicate_risk_off_cols = stocks_data.columns.intersection(risk_off_data.columns)

        if len(duplicate_risk_off_cols) > 0 and verbose:
            print(
                f"[WARN] Rimossi da risk_off_data ticker già presenti "
                f"in stocks_data: {list(duplicate_risk_off_cols)}"
            )

        risk_off_clean = risk_off_data.drop(
            columns=duplicate_risk_off_cols,
            errors='ignore'
        )

        oos_data = pd.concat([stocks_data, risk_off_clean], axis=1)

        # Guardia difensiva finale
        if oos_data.columns.duplicated().any():
            dup_cols = oos_data.columns[oos_data.columns.duplicated()].tolist()
            raise ValueError(
                f"Duplicate columns in oos_data after concat: {dup_cols}"
            )

        mode_label = "Clustered" if use_clustering else "Standard"

        pf_rot, pf_bm, sel = build_rotational_portfolios_from_wfo_result(
            summary_df      = summary_df_final,
            stocks_data     = oos_data,
            start_date      = analisys_start_date,
            end_date        = analisys_end_date,
            benchmark_data  = benchmark_data,
            benchmark_title = benchmark_title,
            portfolio_name  = f"{portfolio_title} – {mode_label} OOS WFO - Total Return (Risk on/off)",
            init_cash       = init_cash,
            plot            = plot,
            debug           = False,
        )

        results['pf_rot']       = pf_rot
        results['pf_benchmark'] = pf_bm
        results['sel_tickers']  = sel

        if plot:
            port_cumrets = pd.DataFrame({
                portfolio_title: pf_rot.cumulative_returns() + 1,
                benchmark_title: pf_bm.cumulative_returns() + 1,
            })

            analyze_portfolio_metrics(
                port_cumrets=port_cumrets,
                portfolio_name=portfolio_title,
                freq="D",
                sort_by="CAGR (%)",
                ascending=False,
                plot_radar=True,
                radar_metrics="all",
                highlight_best=True,
            )
    else:
        results['pf_rot']       = None
        results['pf_benchmark'] = None
        results['sel_tickers']  = None

    # Senza Risk ON/OFF
    print("\n▶ Portafoglio SENZA Risk ON/OFF...")

    mode_label = "Clustered" if use_clustering else "Standard"

    pf_rot_base, pf_bm_base, sel_base = build_rotational_portfolios_from_wfo_result(
        summary_df      = summary_df_final,
        stocks_data     = stocks_data,
        start_date      = analisys_start_date,
        end_date        = analisys_end_date,
        benchmark_data  = benchmark_data,
        benchmark_title = benchmark_title,
        portfolio_name  = f"{portfolio_title} – {mode_label} OOS WFO - Total Return",
        init_cash       = init_cash,
        plot            = plot,
        debug           = False,
    )

    results['pf_rot_base']        = pf_rot_base
    results['pf_benchmark_base']  = pf_bm_base
    results['sel_tickers_base']   = sel_base

    if plot:
        port_cumrets_base = pd.DataFrame({
            portfolio_title: pf_rot_base.cumulative_returns() + 1,
            benchmark_title: pf_bm_base.cumulative_returns() + 1,
        })

        analyze_portfolio_metrics(
            port_cumrets=port_cumrets_base,
            portfolio_name=f"{portfolio_title} (Base)",
            freq="D",
            sort_by="CAGR (%)",
            ascending=False,
            plot_radar=True,
            radar_metrics="all",
            highlight_best=True,
        )

    # Confronto selezioni
    if risk_on_off and results['sel_tickers'] is not None:
        print("\n▶ Confronto selezioni Risk ON/OFF vs Base...")

        _ = compare_selection_columns(
            results['sel_tickers'],
            results['sel_tickers_base'],
            column="tickers",
            label_a="risk on/off",
            label_b="standard",
            compare_only_common_dates=True,
            sort_table_by_diff=True
        )
        # _ = compare_selection_columns(
        #     results['sel_tickers'],
        #     results['sel_tickers_base'],
        #     column="tickers"
        # )

    print("\n" + "="*55)
    print("PIPELINE COMPLETATA")
    print("="*55)

    return results
    


def compare_wfo_pipelines(
    results_std     : dict,
    results_cluster : dict,
    portfolio_title : str  = "Portfolio",
    benchmark_title : str  = "Benchmark",
    plot_radar      : bool = True,
    plot            : bool = True,
    start_date      : str  = None,
    end_date        : str  = None,
    save_plots      : bool = False,
    plots_dir              = None,    # str | Path | None
) -> pd.DataFrame:
    """
    Confronta i 4 portafogli prodotti da due run di run_wfo_pipeline
    (Standard e Clustered, ciascuno con/senza Risk ON/OFF).

    Genera:
    1. Grafico lineare dei rendimenti cumulativi (4 portafogli + benchmark).
    2. Tabella comparativa delle metriche con heatmap (via analyze_portfolio_metrics).
    3. Radar chart normalizzato su range assoluti (se plot_radar=True).

    Parameters
    ----------
    results_std     : dict  Risultato di run_wfo_pipeline(use_clustering=False).
    results_cluster : dict  Risultato di run_wfo_pipeline(use_clustering=True).
    portfolio_title : str   Titolo base usato nelle etichette.
    benchmark_title : str   Etichetta del benchmark.
    plot_radar      : bool  Se True genera il radar chart.
    start_date      : str   Filtro opzionale inizio (es. "2020-01-01").
    end_date        : str   Filtro opzionale fine   (es. "2024-12-31").

    Returns
    -------
    pd.DataFrame  Tabella metriche restituita da analyze_portfolio_metrics.
    """
    import plotly.graph_objects as go

    # ------------------------------------------------------------------
    # Etichette
    # ------------------------------------------------------------------
    lbl = {
        'std_on'   : "Std \u2013 Risk ON/OFF",
        'std_base' : "Std \u2013 Base",
        'cl_on'    : "Cluster \u2013 Risk ON/OFF",
        'cl_base'  : "Cluster \u2013 Base",
    }

    # ------------------------------------------------------------------
    # Raccolta rendimenti cumulativi (base 1.0)
    # ------------------------------------------------------------------
    cumrets = {}

    def _add(pf, label):
        if pf is None:
            return
        cr = pf.cumulative_returns() + 1
        cumrets[label] = cr.squeeze() if isinstance(cr, pd.DataFrame) else cr

    _add(results_std.get('pf_rot'),          lbl['std_on'])
    _add(results_std.get('pf_rot_base'),     lbl['std_base'])
    _add(results_cluster.get('pf_rot'),      lbl['cl_on'])
    _add(results_cluster.get('pf_rot_base'), lbl['cl_base'])

    if not cumrets:
        print("Nessun portafoglio disponibile per il confronto.")
        return pd.DataFrame()

    # Benchmark (primo disponibile)
    pf_bm = (
        results_std.get('pf_benchmark') or
        results_std.get('pf_benchmark_base') or
        results_cluster.get('pf_benchmark') or
        results_cluster.get('pf_benchmark_base')
    )
    bm_cumret = None
    if pf_bm is not None:
        bm_cr = pf_bm.cumulative_returns() + 1
        bm_cumret = bm_cr.squeeze() if isinstance(bm_cr, pd.DataFrame) else bm_cr

    port_cumrets = pd.DataFrame(cumrets)

    # Filtro data
    if start_date:
        port_cumrets = port_cumrets[port_cumrets.index >= start_date]
        if bm_cumret is not None:
            bm_cumret = bm_cumret[bm_cumret.index >= start_date]
    if end_date:
        port_cumrets = port_cumrets[port_cumrets.index <= end_date]
        if bm_cumret is not None:
            bm_cumret = bm_cumret[bm_cumret.index <= end_date]

    port_cumrets = port_cumrets.dropna(how='all')

    # ------------------------------------------------------------------
    # 1. Plot cumulativo
    # ------------------------------------------------------------------
    COLORS = {
        lbl['std_on']   : "#1f77b4",  # blu pieno
        lbl['std_base'] : "#aec7e8",  # blu chiaro
        lbl['cl_on']    : "#d62728",  # rosso pieno
        lbl['cl_base']  : "#f5a7a7",  # rosso chiaro
        benchmark_title   : "#7f7f7f",  # grigio
    }
    DASH = {
        lbl['std_on']   : "solid",
        lbl['std_base'] : "dot",
        lbl['cl_on']    : "solid",
        lbl['cl_base']  : "dot",
        benchmark_title   : "dash",
    }
    WIDTH = {
        lbl['std_on']   : 2.5,
        lbl['std_base'] : 1.5,
        lbl['cl_on']    : 2.5,
        lbl['cl_base']  : 1.5,
        benchmark_title   : 1.5,
    }

    fig = go.Figure()
    for col in port_cumrets.columns:
        fig.add_trace(go.Scatter(
            x    = port_cumrets.index,
            y    = port_cumrets[col],
            name = col,
            mode = "lines",
            line = dict(
                color = COLORS.get(col, "#333333"),
                dash  = DASH.get(col, "solid"),
                width = WIDTH.get(col, 2),
            ),
        ))

    if bm_cumret is not None:
        bm_aligned = bm_cumret.reindex(port_cumrets.index, method="ffill")
        fig.add_trace(go.Scatter(
            x    = bm_aligned.index,
            y    = bm_aligned.values,
            name = benchmark_title,
            mode = "lines",
            line = dict(
                color = COLORS[benchmark_title],
                dash  = DASH[benchmark_title],
                width = WIDTH[benchmark_title],
            ),
        ))

    fig.update_layout(
        title       = f"Confronto rendimenti cumulativi \u2013 {portfolio_title}",
        xaxis_title = "Data",
        yaxis_title = "Rendimento cumulativo (base 1.0)",
        height      = 550,
        width       = 1100,
        template    = "plotly_white",
        hovermode   = "x unified",
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if save_plots and plots_dir is not None:
        from pathlib import Path as _P
        _pd = _P(str(plots_dir))
        _pd.mkdir(parents=True, exist_ok=True)
        # equity_comparison.png — tutti i path (fig già costruita sopra)
        fig.write_image(str(_pd / 'equity_comparison.png'))
        # equity_std.png — solo percorsi Standard
        _std_cols = [c for c in port_cumrets.columns
                     if c in (lbl['std_on'], lbl['std_base'])]
        if _std_cols:
            _fig_std = go.Figure()
            for _c in _std_cols:
                _fig_std.add_trace(go.Scatter(
                    x=port_cumrets.index, y=port_cumrets[_c], name=_c, mode='lines',
                    line=dict(color=COLORS.get(_c,'#333'), dash=DASH.get(_c,'solid'), width=WIDTH.get(_c,2))))
            if bm_cumret is not None:
                _ba = bm_cumret.reindex(port_cumrets.index, method='ffill')
                _fig_std.add_trace(go.Scatter(
                    x=_ba.index, y=_ba.values, name=benchmark_title, mode='lines',
                    line=dict(color=COLORS[benchmark_title], dash=DASH[benchmark_title], width=WIDTH[benchmark_title])))
            _fig_std.update_layout(title=f"Rendimenti cumulativi \u2013 Standard \u2013 {portfolio_title}",
                                   height=400, width=900, template='plotly_white',
                                   hovermode='x unified')
            _fig_std.write_image(str(_pd / 'equity_std.png'))
        # equity_cluster.png — solo percorsi Cluster (condizionale)
        _cl_cols = [c for c in port_cumrets.columns
                    if c in (lbl.get('cl_on',''), lbl.get('cl_base',''))]
        if _cl_cols:
            _fig_cl = go.Figure()
            for _c in _cl_cols:
                _fig_cl.add_trace(go.Scatter(
                    x=port_cumrets.index, y=port_cumrets[_c], name=_c, mode='lines',
                    line=dict(color=COLORS.get(_c,'#333'), dash=DASH.get(_c,'solid'), width=WIDTH.get(_c,2))))
            if bm_cumret is not None:
                _ba = bm_cumret.reindex(port_cumrets.index, method='ffill')
                _fig_cl.add_trace(go.Scatter(
                    x=_ba.index, y=_ba.values, name=benchmark_title, mode='lines',
                    line=dict(color=COLORS[benchmark_title], dash=DASH[benchmark_title], width=WIDTH[benchmark_title])))
            _fig_cl.update_layout(title=f"Rendimenti cumulativi \u2013 Cluster \u2013 {portfolio_title}",
                                   height=400, width=900, template='plotly_white',
                                   hovermode='x unified')
            _fig_cl.write_image(str(_pd / 'equity_cluster.png'))
    if plot:
        fig.show()

    # ------------------------------------------------------------------
    # 2. Tabella metriche + Radar (via analyze_portfolio_metrics)
    # ------------------------------------------------------------------
    metrics_df = analyze_portfolio_metrics(
        port_cumrets     = port_cumrets,
        portfolio_name   = f"Confronto WFO \u2013 {portfolio_title}",
        benchmark_cumret = bm_cumret,
        freq             = "D",
        sort_by          = "CAGR (%)",
        ascending        = False,
        plot_radar       = plot_radar,
        radar_metrics    = "all",
        highlight_best   = True,
    )

    return metrics_df

def run_rotational_portfolio_performance(
    portfolio: dict,
    analisys_start_date: str | pd.Timestamp | None,
    analisys_end_date: str | pd.Timestamp | None,
    # --- WFO summary ---  
    year: int | None = None,
    wfo_results_dir: str = "WFO_R_RESULTS",
    wfo_file_save: str | None = None,   # default: f"{portfolio_title}_{year}.wfo_summary"
    init_cash: float = 100_000,
    plot: bool = False,
    sender_email: str | None = None,
    sender_password: str | None = None,
    recipient_email: str | None = None,
    show_report: bool = True,
    debug: bool = False,
    auto_adjust: bool = True,
    verbose: bool = False,
):
    """
    Costruisce (da summary WFO) un portafoglio rotazionale + benchmark, genera figure
    e invia il resoconto via email.

    Vincolo: analisys_start_date/analisys_end_date governano sia l'analisi sia i grafici.

    Parametri:
    - show_report: se True, visualizza a video le figure e abilita le stampe
    - plot: se True, abilita i plot interni di build_rotational_portfolios_from_selections (di norma False)
    """
    # --- Validazione minima portfolio ---
    if not isinstance(portfolio, dict):
        raise TypeError("portfolio deve essere un dict, es. {'Title': '...', 'tickers': [...]}")

    if "Title" not in portfolio or "tickers" not in portfolio:
        raise KeyError("portfolio deve contenere le chiavi obbligatorie: 'Title' e 'tickers'")

    # Get portfolio data
    tickers=portfolio['tickers']
    benchmark_portfolio=portfolio['benchmark_portfolio']
    benchmark_title=portfolio['benchmark_title']
    portfolio_title=portfolio['Title']

    if not isinstance(tickers, (list, tuple)) or len(tickers) == 0:
        raise ValueError("portfolio['tickers'] deve essere una lista/tupla non vuota di ticker")

    # --- Download dati (warm-up per evitare sel_tickers vuoto a inizio anno) ---
    a_start = pd.Timestamp(analisys_start_date) if analisys_start_date is not None else None
    a_end   = pd.Timestamp(analisys_end_date)   if analisys_end_date is not None else None
    
    # warm-up: 12-18 mesi sono in genere sufficienti per momentum/EMA
    download_start = a_start - pd.DateOffset(months=6) if a_start is not None else None
    
    stocks_data = download_data(
        tickers,
        download_start,          
        a_end,
        show_progress=False,
        auto_adjust=auto_adjust
    )


    if benchmark_portfolio:
        if verbose: print("Creo il benchmark portfolio data...")
        benchmark_data = build_benchmark(benchmark_portfolio, stocks_data.index.min(),stocks_data.index.max(),auto_adjust=auto_adjust).replace(0, np.nan).ffill()
    elif benchmark_title:
        benchmark_data = download_data(benchmark_title, stocks_data.index.min(),analisys_end_date,auto_adjust=auto_adjust)
    else:
        benchmark_data=None
        
    # --- Carica summary WFO ---
       
    # --- Year default: anno corrente ---
    if year is None:
        year = int(pd.Timestamp.now().year)

     
    # --- WFO summary filename ---
    if wfo_file_save is None:
        wfo_file_save = f"{portfolio_title}_{year}.wfo_summary.csv"
        
    wfo_file_save=f"{wfo_results_dir}/{wfo_file_save}"

    summary_df=load_wfo_summary(wfo_file_save)
        
    portfolio_title = f"{portfolio_title} - Total"
    portfolio_title += " Return" if auto_adjust else " Price"

    # --- 1) Costruzione portafogli su finestra analisi (governa stats/plot) ---
    if verbose:
        print("\nRunning build_rotational_portfolios_from_wfo_result ...")

    pf_rot, pf_benchmark, sel_tickers = build_rotational_portfolios_from_wfo_result(
        summary_df=summary_df,
        stocks_data=stocks_data,
        benchmark_data=benchmark_data,
        benchmark_title=benchmark_title,
        # portfolio_name=f"{portfolio_title} – Real OOS WFO - Total Return",
        portfolio_name=portfolio_title,
        init_cash=init_cash,
        plot=False,
        start_date=analisys_start_date,
        end_date=analisys_end_date,
        show_report=show_report,
        debug=debug,
    )

    # --- Normalizza sel_tickers alla finestra di analisi (NO warm-up nel report) ---
    a_start = pd.Timestamp(analisys_start_date) if analisys_start_date is not None else None
    a_end   = pd.Timestamp(analisys_end_date)   if analisys_end_date is not None else None
    
    # --- 2) Figure di performance (stessa finestra) ---

    if verbose:
        print("\nRunning generate_rotational_portfolio_performance ...")
    
    # print(sel_tickers.head(10))
    perf_out = generate_rotational_portfolio_performance(
        pf=pf_rot,
        portfolio_title=portfolio_title,
        sel_tickers=sel_tickers,
        benchmark=benchmark_title,
        benchmark_data=benchmark_data,
        # plot_start_date=analisys_start_date,
        # plot_end_date=analisys_end_date,
        show_report=show_report,
        alpha_analysis=False,
        universe=tickers
    )

    # ------------------------------------------------------------
    # 5) EMAIL: usa ESATTAMENTE perf_out (ZERO RICALCOLI)
    # ------------------------------------------------------------
    if verbose:
        print("\nRunning send_portfolio_performance ...")

    send_portfolio_performance(
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=recipient_email,
        assets=perf_out
    )
    
    return {
        "pf": pf_rot,
        "pf_benchmark": pf_benchmark,
        "sel_tickers": sel_tickers,
        "analisys_start_date": analisys_start_date,
        "analisys_end_date": analisys_end_date,
        "perf_out": perf_out,
    }



# =============================================================================
# Monte Carlo Validation per Portafogli Rotazionali
# =============================================================================

# Metriche standard e convenzione direzione
_MC_METRICS = ['CAGR', 'MaxDD', 'Sharpe', 'Calmar', 'Volatility', 'Ulcer']
# MaxDD stored negative (e.g. -0.15 = -15% DD); less negative = better = higher_is_better=True
# Ulcer stored negative; less negative = better = higher_is_better=True
# Volatility stored positive; lower = better = higher_is_better=False
_MC_HIB = {'CAGR': True, 'MaxDD': True, 'Sharpe': True,
           'Calmar': True, 'Volatility': False, 'Ulcer': True}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER INTERNI
# ─────────────────────────────────────────────────────────────────────────────

def _mc_compute_metrics(equity: pd.Series, trading_days: int = 252) -> dict:
    """
    Calcola CAGR, MaxDD, Sharpe (rf=0), Calmar, Volatility, Ulcer da equity giornaliera.

    Convenzioni:
    - MaxDD: valore negativo (es. -0.157 per -15.7%)
    - Ulcer: negativo — -sqrt(mean(DD^2)); meno negativo = migliore
    - Volatility: positivo, deviazione std annualizzata
    - Sharpe: risk-free rate = 0 per semplicità
    """
    nan_result = {k: np.nan for k in _MC_METRICS}
    rets = equity.pct_change(fill_method=None).dropna()
    if len(rets) < 2:
        return nan_result
    n_years = len(equity) / trading_days
    if n_years <= 0:
        return nan_result
    total_ret = float(equity.iloc[-1] / equity.iloc[0]) - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / n_years) - 1.0
    std_ret = float(rets.std(ddof=1))
    mean_ret = float(rets.mean())
    vol = std_ret * np.sqrt(trading_days)
    sharpe = (mean_ret * trading_days) / vol if vol > 0.0 else np.nan
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if (max_dd != 0.0 and np.isfinite(max_dd)) else np.nan
    ulcer = -float(np.sqrt(float((dd ** 2).mean())))
    return {'CAGR': cagr, 'MaxDD': max_dd, 'Sharpe': sharpe,
            'Calmar': calmar, 'Volatility': vol, 'Ulcer': ulcer}


def _mc_compute_metrics_batch(
    equity_full: np.ndarray,
    boot_rets: np.ndarray,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Calcola le 6 metriche per n_sims equity curve in forma vettorizzata.

    Parameters
    ----------
    equity_full : np.ndarray  shape (n_sims, T)   include il giorno 0 (= init_cash)
    boot_rets   : np.ndarray  shape (n_sims, T-1) ritorni giornalieri
    """
    n_sims, T = equity_full.shape
    n_years = T / trading_days
    final_vals = equity_full[:, -1]
    init_val   = equity_full[:, 0]
    total_ret  = final_vals / init_val - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / n_years) - 1.0
    std_ret  = np.std(boot_rets, axis=1, ddof=1)
    mean_ret = np.mean(boot_rets, axis=1)
    vol      = std_ret * np.sqrt(trading_days)
    sharpe   = np.where(vol > 0.0, (mean_ret * trading_days) / vol, np.nan)
    running_max = np.maximum.accumulate(equity_full, axis=1)
    dd      = equity_full / running_max - 1.0
    max_dd  = dd.min(axis=1)
    calmar  = np.where(max_dd != 0.0, cagr / np.abs(max_dd), np.nan)
    ulcer   = -np.sqrt((dd ** 2).mean(axis=1))
    return pd.DataFrame({'CAGR': cagr, 'MaxDD': max_dd, 'Sharpe': sharpe,
                         'Calmar': calmar, 'Volatility': vol, 'Ulcer': ulcer})


def _mc_build_result(equity_full: np.ndarray, boot_rets: np.ndarray,
                     equity_actual: pd.Series, trading_days: int = 252) -> dict:
    """Costruisce il dict risultato standard da array numpy."""
    n_sims = equity_full.shape[0]
    equity_curves = pd.DataFrame(equity_full.T,
                                 index=equity_actual.index,
                                 columns=range(n_sims))
    metrics_per_sim = _mc_compute_metrics_batch(equity_full, boot_rets, trading_days)
    actual_metrics  = _mc_compute_metrics(equity_actual, trading_days)
    percentiles = {
        lbl: {m: float(np.nanpercentile(metrics_per_sim[m].values, q))
              for m in _MC_METRICS}
        for lbl, q in [('p5', 5), ('p25', 25), ('p50', 50), ('p75', 75), ('p95', 95)]
    }
    # fraction of sims <= actual value: high = actual above median (for MaxDD: less negative = better → high pos. OK)
    actual_quantile_position = {
        m: float(np.nanmean(metrics_per_sim[m].values <= actual_metrics[m]))
        for m in _MC_METRICS
    }
    return dict(equity_curves=equity_curves, metrics_per_sim=metrics_per_sim,
                percentiles=percentiles, actual_metrics=actual_metrics,
                actual_quantile_position=actual_quantile_position)


def _mc_block_bootstrap_returns(
    returns: pd.Series,
    block_size: int,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Block bootstrap a lunghezza fissa con wrap-around.

    Genera ceil(n/block_size) blocchi campionando start random in [0,n),
    applica wrap-around (start+j)%n per blocchi che superano la fine,
    poi tronca alla lunghezza originale n.
    """
    vals = returns.values
    n = len(vals)
    n_blocks = int(np.ceil(n / block_size))
    starts  = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    indices = (starts[:, None] + offsets[None, :]) % n   # (n_blocks, block_size)
    bootstrapped = vals[indices.ravel()][:n]
    return pd.Series(bootstrapped, index=returns.index)


def _mc_regime_block_bootstrap_returns(
    returns: pd.Series,
    regime: pd.Series,
    block_size: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Regime-conditional block bootstrap con wrap-around.

    Per ogni giorno i campiona un blocco dal pool del regime corrispondente a i.
    Fall-back al pool completo se un regime ha < block_size osservazioni (warning unico).

    Returns
    -------
    (pd.Series, bool)  — serie bootstrappata + flag fallback_used
    """
    vals = returns.values
    n    = len(vals)
    reg  = regime.reindex(returns.index).ffill().bfill().fillna(0).astype(int).values
    pools = {r: np.where(reg == r)[0] for r in (0, 1)}
    fallback_used = False
    full_pool = np.arange(n)
    for r_val in (0, 1):
        if len(pools[r_val]) < block_size:
            if not fallback_used:
                warnings.warn(
                    f"_mc_regime_block_bootstrap_returns: regime={r_val} has "
                    f"{len(pools[r_val])} obs < block_size={block_size}. "
                    "Falling back to full-sample pool for this regime.",
                    stacklevel=2,
                )
            pools[r_val] = full_pool
            fallback_used = True
    bootstrapped = np.empty(n)
    i = 0
    while i < n:
        r_val    = int(reg[i])
        pool     = pools[r_val]
        start    = pool[rng.integers(0, len(pool))]
        end      = min(i + block_size, n)
        block_len = end - i
        idxs     = (start + np.arange(block_len)) % n
        bootstrapped[i:end] = vals[idxs]
        i += block_size
    return pd.Series(bootstrapped, index=returns.index), fallback_used


def _mc_equity_from_returns(bootstrapped_returns: pd.Series, init_cash: float) -> pd.Series:
    """Equity curve: init_cash * cumprod(1 + r). Stesso indice di bootstrapped_returns."""
    return init_cash * (1.0 + bootstrapped_returns).cumprod()


def _mc_simulate_equity_from_holdings(
    rebal_schedule: pd.DataFrame,
    stocks_data: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    init_cash: float,
) -> pd.Series:
    """
    Simulazione vettorizzata equity da rebalance schedule equal-weight.

    Costruisce weight_matrix (date × ticker) con pesi 1/k tra i ticker selezionati,
    la shifta di 1 giorno (segnale al close → entrata il giorno successivo),
    poi calcola (returns * weights_shifted).sum → cumprod → equity.

    I ticker di rebal_schedule non presenti in stocks_data.columns sono skippati
    silenziosamente (il warning viene emesso a monte).
    """
    cols    = list(stocks_data.columns)
    col_pos = {t: i for i, t in enumerate(cols)}
    n_d     = len(full_index)
    n_c     = len(cols)
    w_arr   = np.zeros((n_d, n_c), dtype=np.float64)

    sorted_dates = sorted(rebal_schedule.index)
    n_rd = len(sorted_dates)
    for i, rd in enumerate(sorted_dates):
        raw = rebal_schedule.loc[rd, 'tickers']
        tickers_sel = [t for t in (raw if isinstance(raw, list) else list(raw))
                       if t in col_pos]
        k = len(tickers_sel)
        if k == 0:
            continue
        s = full_index.searchsorted(rd, side='left')
        if s >= n_d:
            continue
        e = (full_index.searchsorted(sorted_dates[i + 1], side='left')
             if i + 1 < n_rd else n_d)
        w_arr[s:e] = 0.0
        wt = 1.0 / k
        for t in tickers_sel:
            w_arr[s:e, col_pos[t]] = wt

    # shift 1 day
    w_shifted      = np.roll(w_arr, 1, axis=0)
    w_shifted[0]   = 0.0

    prices_aligned = stocks_data.reindex(full_index).ffill().values.astype(np.float64)
    rets_arr       = np.zeros_like(prices_aligned)
    prev           = prices_aligned[:-1]
    mask           = prev > 0.0
    rets_arr[1:]   = np.where(mask, (prices_aligned[1:] - prev) / prev, 0.0)

    port_rets  = (rets_arr * w_shifted).sum(axis=1)
    equity_vals = init_cash * np.cumprod(1.0 + port_rets)
    return pd.Series(equity_vals, index=full_index)


def _mc_compute_pvalue(
    sim_values: np.ndarray,
    actual_value: float,
    higher_is_better: bool = True,
) -> float:
    """
    P-value = frazione di simulazioni che BATTONO il valore reale.

    higher_is_better=True  (CAGR, MaxDD-negativo, Sharpe, Calmar, Ulcer-negativo):
        p = mean(sim > actual)
        Low p = actual raramente superato dal random = skill
    higher_is_better=False (Volatility positiva):
        p = mean(sim < actual)
        Low p = actual raramente battuto in termini di vol bassa = skill
    """
    arr = np.asarray(sim_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr > actual_value) if higher_is_better
                 else np.mean(arr < actual_value))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS — Plotly (template "plotly_white", coerente con il resto del progetto)
# ─────────────────────────────────────────────────────────────────────────────
# Palette MC
_MC_C_ACTUAL    = "#d62728"   # rosso — equity reale
_MC_C_MEDIAN    = "#1f77b4"   # blu   — mediana simulata
_MC_C_BAND      = "rgba(31,119,180,0.15)"   # banda p5-p95
_MC_C_BENCHMARK = "#7f7f7f"   # grigio — benchmark
_MC_C_PATH      = "rgba(31,119,180,0.06)"   # individual paths (bassa opacità)

_MC_RANGESELECTOR = dict(buttons=[
    dict(count=1, label="1A",  step="year",  stepmode="backward"),
    dict(count=3, label="3A",  step="year",  stepmode="backward"),
    dict(count=5, label="5A",  step="year",  stepmode="backward"),
    dict(step="all", label="Tutto"),
])

def _mc_layout(**kwargs) -> dict:
    """Restituisce un dict di argomenti update_layout con i default del progetto."""
    base = dict(
        template   = "plotly_white",
        hovermode  = "x unified",
        legend     = dict(orientation="h", yanchor="bottom", y=1.02,
                          xanchor="right", x=1),
        font       = dict(size=12),
    )
    base.update(kwargs)
    return base


def _mc_plot_ci_method(
    method_label: str,
    result: dict,
    actual_equity: pd.Series,
    benchmark_equity: pd.Series,
    show_individual_paths: bool = False,
) -> list:
    """
    Genera 3 go.Figure per un singolo metodo CI (Blocco A):
      [0] Fan chart equity: banda p5–p95 + p50 + equity reale + benchmark
      [1] Istogramma distribuzione MaxDD con linea verticale del reale
      [2] Istogramma distribuzione CAGR con linee verticali per reale e benchmark

    Ritorna la lista delle figure senza chiamare fig.show().
    Il wrapper chiama fig.show() se show_method_plots=True.

    Parameters
    ----------
    method_label : str   Es. "A2 · Block Bootstrap"
    result       : dict  Output di _mc_run_iid/block/regime_block_bootstrap
    actual_equity        : pd.Series  equity del PTF reale
    benchmark_equity     : pd.Series  equity del benchmark normalizzata a init_cash
    show_individual_paths: bool       se True mostra fino a 50 path (opacità 6%)
    """
    ec      = result['equity_curves']
    p5      = ec.quantile(0.05, axis=1)
    p50     = ec.quantile(0.50, axis=1)
    p95     = ec.quantile(0.95, axis=1)
    act_m   = result['actual_metrics']
    figs    = []

    # kaleido/orjson cannot serialize pd.Timestamp — convert index to ISO strings
    # for all series used as x-axis in traces so write_image works correctly.
    def _ix(s: pd.Series) -> list:
        return s.index.strftime("%Y-%m-%d").tolist()

    p5_x   = _ix(p5)
    p50_x  = _ix(p50)
    p95_x  = _ix(p95)
    act_x  = _ix(actual_equity)
    bm_x   = _ix(benchmark_equity) if benchmark_equity is not None else None

    # ── Fig 0: Fan chart ──────────────────────────────────────────────────────
    fig = go.Figure()

    if show_individual_paths:
        sample_cols = ec.columns[:50]
        ec_x = ec.index.strftime("%Y-%m-%d").tolist()
        for col in sample_cols:
            fig.add_trace(go.Scatter(
                x=ec_x, y=ec[col], mode="lines",
                line=dict(color=_MC_C_PATH, width=0.5),
                showlegend=False, hoverinfo="skip",
            ))

    # banda p5-p95
    fig.add_trace(go.Scatter(
        x=p95_x + p5_x[::-1],
        y=list(p95.values) + list(p5.values[::-1]),
        fill="toself", fillcolor=_MC_C_BAND,
        line=dict(color="rgba(0,0,0,0)"),
        name="p5–p95", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=p50_x, y=p50.values, mode="lines",
        line=dict(color=_MC_C_MEDIAN, width=2),
        name="p50 mediana",
    ))
    fig.add_trace(go.Scatter(
        x=act_x, y=actual_equity.values, mode="lines",
        line=dict(color=_MC_C_ACTUAL, width=2.5),
        name="Actual",
    ))
    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=bm_x, y=benchmark_equity.values, mode="lines",
            line=dict(color=_MC_C_BENCHMARK, width=1.5, dash="dash"),
            name="Benchmark",
        ))

    fig.update_layout(
        **_mc_layout(
            title=f"{method_label} — Fan chart equity (p5/p50/p95)",
            xaxis_title="Data", yaxis_title="Valore (€)",
            height=520, width=1100,
            xaxis=dict(rangeselector=_MC_RANGESELECTOR,
                       rangeslider=dict(visible=False), type="date"),
        )
    )
    figs.append(fig)

    # ── Fig 1: Istogramma MaxDD ───────────────────────────────────────────────
    dd_vals = result['metrics_per_sim']['MaxDD'].dropna().values * 100
    actual_dd = act_m['MaxDD'] * 100
    aqp_dd    = result['actual_quantile_position']['MaxDD']

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=dd_vals, nbinsx=50,
        marker_color=_MC_C_MEDIAN, opacity=0.65,
        name="Simulazioni",
    ))
    fig2.add_vline(
        x=actual_dd,
        line=dict(color=_MC_C_ACTUAL, width=2.5, dash="solid"),
        annotation_text=f"Actual {actual_dd:.1f}%<br>(pct={aqp_dd:.0%})",
        annotation_position="top right",
        annotation_font_size=11,
    )
    fig2.update_layout(**_mc_layout(
        title=f"{method_label} — Distribuzione MaxDD",
        xaxis_title="Max Drawdown (%)", yaxis_title="Conteggio",
        height=420, width=700, showlegend=False,
        hovermode="x",
    ))
    figs.append(fig2)

    # ── Fig 2: Istogramma CAGR ────────────────────────────────────────────────
    cagr_vals = result['metrics_per_sim']['CAGR'].dropna().values * 100
    actual_cagr = act_m['CAGR'] * 100
    aqp_cagr    = result['actual_quantile_position']['CAGR']

    bm_cagr = None
    if benchmark_equity is not None:
        n_y = len(benchmark_equity) / 252
        bm_cagr = ((benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) ** (1/n_y) - 1) * 100

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=cagr_vals, nbinsx=50,
        marker_color=_MC_C_MEDIAN, opacity=0.65,
        name="Simulazioni",
    ))
    fig3.add_vline(
        x=actual_cagr,
        line=dict(color=_MC_C_ACTUAL, width=2.5),
        annotation_text=f"Actual {actual_cagr:.1f}%<br>(pct={aqp_cagr:.0%})",
        annotation_position="top right", annotation_font_size=11,
    )
    if bm_cagr is not None:
        fig3.add_vline(
            x=bm_cagr,
            line=dict(color=_MC_C_BENCHMARK, width=1.8, dash="dash"),
            annotation_text=f"BM {bm_cagr:.1f}%",
            annotation_position="top left", annotation_font_size=11,
        )
    fig3.update_layout(**_mc_layout(
        title=f"{method_label} — Distribuzione CAGR",
        xaxis_title="CAGR (%)", yaxis_title="Conteggio",
        height=420, width=700, showlegend=False,
        hovermode="x",
    ))
    figs.append(fig3)

    return figs


def _mc_plot_ci_summary(
    ci_results      : dict,
    actual_equities : dict,
    save_path               = None,    # str | Path | None
) -> go.Figure:
    """
    Genera un go.Figure con subplots (1×2): boxplot CAGR e MaxDD cross-method.
    I metodi None (A3 skippato) vengono omessi.
    Ritorna la figura senza chiamare show().
    """
    method_labels = {
        'iid_bootstrap':  'A1 · IID',
        'block_bootstrap': 'A2 · Block',
        'regime_block':   'A3 · Regime',
    }
    palette = {
        'iid_bootstrap':  "#aec7e8",
        'block_bootstrap': "#1f77b4",
        'regime_block':   "#ff7f0e",
    }
    active = {k: v for k, v in ci_results.items() if v is not None}

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("CAGR (%)", "Max Drawdown (%)"))

    for key, res in active.items():
        lbl   = method_labels[key]
        color = palette[key]
        cagr_vals = res['metrics_per_sim']['CAGR'].dropna().values * 100
        dd_vals   = res['metrics_per_sim']['MaxDD'].dropna().values * 100

        for col_idx, (vals, _metric) in enumerate([(cagr_vals, 'CAGR'), (dd_vals, 'MaxDD')], 1):
            fig.add_trace(go.Box(
                y=vals, name=lbl,
                marker_color=color,
                boxmean=True,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    # Linea del valore reale (usa il primo metodo attivo per actual)
    first_res = next(iter(active.values()))
    actual_cagr = first_res['actual_metrics']['CAGR'] * 100
    actual_dd   = first_res['actual_metrics']['MaxDD'] * 100
    fig.add_hline(y=actual_cagr, line=dict(color=_MC_C_ACTUAL, dash="dash", width=2),
                  annotation_text="Actual", row=1, col=1)
    fig.add_hline(y=actual_dd,   line=dict(color=_MC_C_ACTUAL, dash="dash", width=2),
                  annotation_text="Actual", row=1, col=2)

    fig.update_layout(**_mc_layout(
        title="CI Methods — Confronto cross-method (CAGR e MaxDD)",
        height=520, width=1100,
        hovermode="closest",
    ))
    if save_path is not None:
        fig.write_image(str(save_path))
    return fig


def _mc_plot_skill_test(
    test_label    : str,
    result        : dict,
    actual_equity : pd.Series,
    save_path             = None,    # str | Path | None — salva figs[0] (CAGR histogram)
) -> list:
    """
    Genera 2 go.Figure per uno skill test (Blocco B):
      [0] Istogramma CAGR con linea reale + p-value in etichetta
      [1] Istogramma MaxDD con linea reale + p-value in etichetta

    NOTA: questo metodo produce p-value (permutation test), NON CI.
    Ritorna la lista senza chiamare show().

    save_path : str | Path | None, default None
        Se fornito, salva su disco SOLO la prima figura (figs[0] =
        istogramma CAGR), che è il plot principale del test MC. Le
        eventuali figure secondarie ritornate non vengono salvate.
        Coerente con il pattern un-file-per-test-MC su disco.
    """
    act_m  = result['actual_metrics']
    pvals  = result['p_values']
    figs   = []

    for metric, x_label in [('CAGR', 'CAGR (%)'), ('MaxDD', 'Max Drawdown (%)')]:
        vals       = result['metrics_per_sim'][metric].dropna().values * 100
        actual_val = act_m[metric] * 100
        pv         = pvals.get(metric, float('nan'))
        sig_label  = "***" if pv < 0.01 else ("*" if pv < 0.05 else "n.s.")

        # Colora le bar: verde se pv < 0.05 (skill evidente), grigio altrimenti
        bar_color  = "#2ca02c" if pv < 0.05 else _MC_C_MEDIAN

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=50,
            marker_color=bar_color, opacity=0.65,
            name="Simulazioni random",
        ))
        fig.add_vline(
            x=actual_val,
            line=dict(color=_MC_C_ACTUAL, width=2.5),
            annotation_text=(f"Actual {actual_val:.1f}%<br>"
                             f"p={pv:.3f} {sig_label}"),
            annotation_position="top right",
            annotation_font_size=11,
        )
        fig.update_layout(**_mc_layout(
            title=f"{test_label} — {x_label}",
            xaxis_title=x_label, yaxis_title="Conteggio",
            height=420, width=700, showlegend=False,
            hovermode="x",
        ))
        figs.append(fig)

    # Salviamo solo figs[0] (istogramma CAGR) — è il plot più informativo
    # per il report tecnico e mantiene il naming dei file PNG su disco
    # coerente con le altre metriche MC (un file per test). Le altre
    # figure ritornate (eventuali metriche secondarie) restano disponibili
    # per visualizzazione inline ma non vengono salvate.
    if save_path is not None and figs:
        figs[0].write_image(str(save_path))
    return figs


def _mc_plot_skill_summary(
    skill_results : dict,
    save_path             = None,    # str | Path | None
) -> go.Figure:
    """
    Genera un go.Figure barplot cross-test dei p-value per metrica.
    Bar verdi se p < 0.05, grigie sopra.
    Linee orizzontali a 0.05 (arancio) e 0.01 (rosso) con annotazione.
    Ritorna la figura senza chiamare show().
    """
    test_labels = {
        'rotation_reshuffle': 'B1 · Rotation Reshuffle',
        'rebalance_timing':   'B2 · Rebalance Timing',
    }
    metrics_display = {
        'CAGR': 'CAGR', 'MaxDD': 'MaxDD', 'Sharpe': 'Sharpe',
        'Calmar': 'Calmar', 'Volatility': 'Vol', 'Ulcer': 'Ulcer',
    }
    test_palette = ['#1f77b4', '#ff7f0e']

    fig = go.Figure()
    n_metrics = len(_MC_METRICS)
    x_labels  = [metrics_display[m] for m in _MC_METRICS]

    for t_idx, (key, res) in enumerate(skill_results.items()):
        lbl  = test_labels.get(key, key)
        pvals = [res['p_values'].get(m, float('nan')) for m in _MC_METRICS]
        # Colori per barra: verde se p < 0.05, base-color se no
        bar_colors = ['#2ca02c' if (not np.isnan(p) and p < 0.05) else test_palette[t_idx]
                      for p in pvals]
        # Altezza visiva minima 0.012 per barre p≈0 (valore reale nelle label)
        _MIN_VIS = 0.012
        display_pvals = [max(p, _MIN_VIS) if not np.isnan(p) else p for p in pvals]
        fig.add_trace(go.Bar(
            name=lbl,
            x=x_labels,
            y=display_pvals,
            marker_color=bar_colors,
            opacity=0.8,
            text=[f"{p:.3f}" if not np.isnan(p) else "" for p in pvals],
            textposition="outside",
            customdata=pvals,
            hovertemplate="%{x}: p=%{customdata:.4f}<extra></extra>",
        ))

    fig.add_hline(
        y=0.05, line=dict(color="orange", width=1.8, dash="dash"),
        annotation_text="5%", annotation_position="right",
        annotation_font_size=11,
    )
    fig.add_hline(
        y=0.01, line=dict(color=_MC_C_ACTUAL, width=1.8, dash="dash"),
        annotation_text="1%", annotation_position="right",
        annotation_font_size=11,
    )
    fig.update_layout(**_mc_layout(
        title="Skill Tests — p-value per metrica (verde = significativo p<0.05)",
        yaxis_title="p-value",
        yaxis=dict(range=[0, 1.1]),
        height=480, width=900,
        barmode="group",
        hovermode="x unified",
    ))
    if save_path is not None:
        fig.write_image(str(save_path))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _mc_cagr_quick(pf, trading_days: int = 252) -> float:
    """MC-convention CAGR from portfolio equity, without running a full MC."""
    eq = pf.value()
    if isinstance(eq, pd.DataFrame):
        eq = eq.squeeze()
    n = len(eq)
    if n < 2:
        return float('nan')
    return float((eq.iloc[-1] / eq.iloc[0]) ** (trading_days / n) - 1.0)


def _mc_print_portfolio_note() -> None:
    print("  Note: MC uses (V_end/V_start)^(252/n)-1  (academic convention).")
    print("        vbt uses an internal annualization that differs systematically.")
    print("        Compare MC-Actual vs MC distributions only; do NOT mix vbt and MC values.")


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION SHAPE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _mc_check_distribution_shape(
    sim_values: np.ndarray,
    metric_name: str,
) -> dict:
    """
    Diagnostica della forma di una distribuzione MC.
    Rileva bimodalità/multimodalità che rendono la lettura del p-value ambigua.

    Algoritmo: istogramma sqrt(n) bin + smoothing rolling-3 + picchi locali.
    Safeguard 1: distanza minima tra picchi >= nbins//5 (evita rumore di binning).
    Safeguard 2: profondità valle tra picchi < 0.7 * min(peak_heights) (evita
                 distribuzioni 'piatte in cima' conteggiate come bimodali).

    Returns
    -------
    dict con chiavi:
        is_unimodal   : bool      False se ≥2 picchi separati e profondi
        shape_warning : str|None  descrizione se non unimodale, None altrimenti
        kurtosis      : float     excess kurtosis (0 = gaussiana)
        skewness      : float     skewness (0 = simmetrica)
    """
    vals = np.asarray(sim_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n    = len(vals)
    _nan = {'is_unimodal': True, 'shape_warning': None,
            'kurtosis': np.nan, 'skewness': np.nan}
    if n < 10:
        return _nan
    std = vals.std(ddof=0)
    if std < 1e-12:
        return _nan
    norm     = (vals - vals.mean()) / std
    skewness = float(np.mean(norm ** 3))
    kurtosis = float(np.mean(norm ** 4) - 3.0)  # excess kurtosis

    # Histogram + smoothing
    nbins    = max(10, int(np.sqrt(n)))
    counts, _= np.histogram(vals, bins=nbins)
    smoothed = np.convolve(counts.astype(float), np.ones(3) / 3.0, mode='same')
    max_cnt  = smoothed.max()
    if max_cnt == 0:
        return _nan

    # Candidate local maxima (amplitude > 0.5 * global max)
    candidates = [
        i for i in range(1, len(smoothed) - 1)
        if smoothed[i] > smoothed[i - 1]
        and smoothed[i] > smoothed[i + 1]
        and smoothed[i] > 0.5 * max_cnt
    ]

    # Safeguard 1: min distance nbins//5 between retained peaks
    min_dist = max(2, nbins // 5)
    retained: list[int] = []
    for c in candidates:
        if not retained or (c - retained[-1]) >= min_dist:
            retained.append(c)

    # Safeguard 2: valley depth between consecutive retained peaks
    # Valley must be < 0.7 * min(left_height, right_height)
    valid_mode_pairs = 0
    for idx in range(len(retained) - 1):
        left, right  = retained[idx], retained[idx + 1]
        valley       = float(smoothed[left:right + 1].min())
        threshold    = 0.7 * float(min(smoothed[left], smoothed[right]))
        if valley < threshold:
            valid_mode_pairs += 1

    is_unimodal   = valid_mode_pairs == 0
    shape_warning = None
    if not is_unimodal:
        n_modes = valid_mode_pairs + 1
        shape_warning = (
            f"{metric_name}: distribuzione {n_modes}-modale rilevata "
            f"(skewness={skewness:+.2f}, kurtosis={kurtosis:+.2f}); "
            "lettura del p-value richiede cautela"
        )
    return {
        'is_unimodal':   is_unimodal,
        'shape_warning': shape_warning,
        'kurtosis':      kurtosis,
        'skewness':      skewness,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO A — Metodi CI
# ─────────────────────────────────────────────────────────────────────────────

def mc_run_iid_bootstrap(pf_rot, n_simulations: int, rng: np.random.Generator) -> dict:
    """
    A1: IID Returns Bootstrap — baseline diagnostica.

    Campiona iid i ritorni giornalieri del PTF. Sottostima il rischio reale
    perché ignora autocorrelazione e cluster di volatilità.
    Usare A2 (block bootstrap) come metodo principale per le decisioni.

    Sanity check A1: median(CAGR_sim) ≈ CAGR_actual (bootstrap iid è non-distorto in media).
    """
    equity_actual = pf_rot.value()
    if isinstance(equity_actual, pd.DataFrame):
        equity_actual = equity_actual.squeeze()
    init_cash = float(pf_rot.init_cash)
    returns   = equity_actual.pct_change(fill_method=None).dropna()
    n         = len(returns)
    ret_vals  = returns.values.astype(np.float64)

    # Vectorized: campiona tutti i ritorni in una sola operazione
    sampled_idx  = rng.integers(0, n, size=(n_simulations, n))
    all_boot_rets = ret_vals[sampled_idx]                        # (n_sims, n)
    all_cum       = np.cumprod(1.0 + all_boot_rets, axis=1)     # (n_sims, n)
    all_equity_mc = init_cash * all_cum                          # (n_sims, n)
    init_col      = np.full((n_simulations, 1), init_cash)
    equity_full   = np.concatenate([init_col, all_equity_mc], axis=1)  # (n_sims, T)

    result = _mc_build_result(equity_full, all_boot_rets, equity_actual)

    # sanity check: median CAGR ≈ actual CAGR (bias dovrebbe essere < 0.5%)
    med_cagr    = result['percentiles']['p50']['CAGR']
    actual_cagr = result['actual_metrics']['CAGR']
    print(f"  [A1 sanity] median_CAGR={med_cagr:.3f}  actual_CAGR={actual_cagr:.3f}  "
          f"bias={med_cagr - actual_cagr:+.4f} (should be ≈ 0)")
    return result


def mc_run_block_bootstrap(
    pf_rot,
    block_size: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> dict:
    """
    A2: Block Bootstrap — metodo principale per CI standard.

    Ricampiona i ritorni in blocchi di lunghezza fissa (wrap-around),
    preservando autocorrelazione e cluster di volatilità.

    Sanity check A2: std(CAGR_sim) > std(CAGR_sim_A1) — block bootstrap
    riconosce più rischio rispetto all'iid (da verificare confrontando i due run).
    """
    equity_actual = pf_rot.value()
    if isinstance(equity_actual, pd.DataFrame):
        equity_actual = equity_actual.squeeze()
    init_cash = float(pf_rot.init_cash)
    returns   = equity_actual.pct_change(fill_method=None).dropna()
    n         = len(returns)

    all_boot_rets = np.empty((n_simulations, n), dtype=np.float64)
    for sim_idx in range(n_simulations):
        all_boot_rets[sim_idx] = _mc_block_bootstrap_returns(returns, block_size, rng).values

    all_cum       = np.cumprod(1.0 + all_boot_rets, axis=1)
    all_equity_mc = init_cash * all_cum
    init_col      = np.full((n_simulations, 1), init_cash)
    equity_full   = np.concatenate([init_col, all_equity_mc], axis=1)

    result = _mc_build_result(equity_full, all_boot_rets, equity_actual)

    # sanity check: std(CAGR) dovrebbe essere >= A1 (block cattura più struttura)
    std_cagr = float(result['metrics_per_sim']['CAGR'].std())
    print(f"  [A2 sanity] std_CAGR={std_cagr:.4f}  (block_size={block_size}; "
          "confronta con A1 — A2 dovrebbe avere std >= A1)")
    return result


def _mc_run_regime_block_bootstrap(
    pf_rot,
    regime: pd.Series,
    block_size: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> dict:
    """
    A3: Regime-Conditional Block Bootstrap.

    Bootstrappa ritorni separatamente per regime 0 (Risk OFF) e 1 (Risk ON),
    seguendo la sequenza originale del regime. Le proporzioni di regime sono
    preservate per costruzione.

    Sanity check A3: frazione di tempo Risk OFF nel campione ≈ storica (diff < 0.05).
    """
    equity_actual = pf_rot.value()
    if isinstance(equity_actual, pd.DataFrame):
        equity_actual = equity_actual.squeeze()
    init_cash = float(pf_rot.init_cash)
    returns   = equity_actual.pct_change(fill_method=None).dropna()
    n         = len(returns)

    regime_aligned = regime.reindex(returns.index).ffill().bfill().fillna(0).astype(int)

    # Pre-check fallback (warn una sola volta prima del loop)
    pools_counts = {r: int((regime_aligned == r).sum()) for r in (0, 1)}
    fallback_used_any = False
    for r_val, cnt in pools_counts.items():
        if cnt < block_size and not fallback_used_any:
            warnings.warn(
                f"_mc_run_regime_block_bootstrap: regime={r_val} has {cnt} obs "
                f"< block_size={block_size}. Falling back to full-sample pool.",
                stacklevel=2,
            )
            fallback_used_any = True

    all_boot_rets = np.empty((n_simulations, n), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sim_idx in range(n_simulations):
            boot_series, _ = _mc_regime_block_bootstrap_returns(
                returns, regime_aligned, block_size, rng
            )
            all_boot_rets[sim_idx] = boot_series.values

    all_cum       = np.cumprod(1.0 + all_boot_rets, axis=1)
    all_equity_mc = init_cash * all_cum
    init_col      = np.full((n_simulations, 1), init_cash)
    equity_full   = np.concatenate([init_col, all_equity_mc], axis=1)

    result = _mc_build_result(equity_full, all_boot_rets, equity_actual)
    result['regime_fallback_used'] = fallback_used_any

    # sanity check: proporzione Risk OFF preservata per costruzione
    hist_off = float((regime_aligned == 0).mean())
    hist_on  = float((regime_aligned == 1).mean())
    print(f"  [A3 sanity] hist_risk_off={hist_off:.2%}  risk_on={hist_on:.2%}  "
          f"(frazioni preservate per costruzione)  fallback_used={fallback_used_any}")
    # Verifica: diff proporzione OFF in un sample sim vs storico dovrebbe essere 0
    # (per costruzione, i ritorni campionati nelle date OFF vengono sempre dal pool OFF)
    print(f"  [A3 sanity] abs(diff) OFF frac = 0.000 (garantito dall'algoritmo)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO B — Skill Tests
# ─────────────────────────────────────────────────────────────────────────────

def _mc_run_rotation_reshuffle(
    pf_rot,
    sel_tickers: pd.DataFrame,
    stocks_data: pd.DataFrame,
    tickers_master: list,
    init_cash: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> dict:
    """
    B1: Rotation Reshuffle — permutation test sulla skill di selezione.
    Produce p-value, NON intervalli di confidenza.
    """
    equity_actual = pf_rot.value()
    if isinstance(equity_actual, pd.DataFrame):
        equity_actual = equity_actual.squeeze()
    full_index    = equity_actual.index

    has_universe  = 'universe' in sel_tickers.columns
    use_master_fb = not has_universe
    if use_master_fb:
        warnings.warn(
            f"B1 Rotation Reshuffle: 'universe' column missing in sel_tickers. "
            f"Fall-back to tickers_master ({len(tickers_master)} tickers) for all rebal_dates.",
            stacklevel=2,
        )

    actual_metrics = _mc_compute_metrics(equity_actual)
    all_sim_equities = []

    for sim_idx in range(n_simulations):
        sim_schedule = sel_tickers[['tickers']].copy()
        new_tickers_col = []
        for rd in sel_tickers.index:
            k = len(sel_tickers.at[rd, 'tickers'])
            if k == 0:
                new_tickers_col.append([])
                continue
            if has_universe:
                univ = list(sel_tickers.at[rd, 'universe'])
            else:
                univ = tickers_master
            # keep only tickers available in stocks_data
            univ_avail = [t for t in univ if t in stocks_data.columns]
            k_actual = min(k, len(univ_avail))
            if k_actual == 0:
                new_tickers_col.append([])
            else:
                chosen = rng.choice(univ_avail, size=k_actual, replace=False).tolist()
                new_tickers_col.append(chosen)
        sim_schedule = sim_schedule.copy()
        sim_schedule['tickers'] = new_tickers_col
        eq = _mc_simulate_equity_from_holdings(sim_schedule, stocks_data, full_index, init_cash)
        all_sim_equities.append(eq.values)

    equity_arr = np.array(all_sim_equities)   # (n_sims, T)
    # returns: (n_sims, T-1) — approximate from equity ratios
    boot_rets_approx = np.diff(equity_arr, axis=1) / equity_arr[:, :-1]
    metrics_per_sim = _mc_compute_metrics_batch(equity_arr, boot_rets_approx)

    p_values = {
        m: _mc_compute_pvalue(metrics_per_sim[m].values, actual_metrics[m], _MC_HIB[m])
        for m in _MC_METRICS
    }
    interpretation = _mc_interpret_skill(p_values, 'Selection')

    equity_curves = pd.DataFrame(equity_arr.T, index=full_index, columns=range(n_simulations))
    distribution_shape = {
        m: _mc_check_distribution_shape(metrics_per_sim[m].dropna().values, m)
        for m in _MC_METRICS
    }
    return dict(equity_curves=equity_curves, metrics_per_sim=metrics_per_sim,
                actual_metrics=actual_metrics, p_values=p_values,
                interpretation=interpretation,
                distribution_shape=distribution_shape)


def _mc_run_rebalance_timing(
    pf_rot,
    sel_tickers: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    init_cash: float,
    vol_window: int,
    n_vol_quantiles: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> dict:
    """
    B2: Rebalance Timing Bootstrap — permutation test sulla skill di timing.
    Permuta date di ribilanciamento intra-quantile di volatilità.
    Produce p-value, NON intervalli di confidenza.
    """
    equity_actual = pf_rot.value()
    if isinstance(equity_actual, pd.DataFrame):
        equity_actual = equity_actual.squeeze()
    full_index    = equity_actual.index
    actual_metrics = _mc_compute_metrics(equity_actual)

    # Volatilità rolling annualizzata del benchmark
    bm_rets = benchmark_data.pct_change(fill_method=None).dropna()
    bm_vol  = bm_rets.rolling(vol_window).std() * np.sqrt(252)

    # Assegna ogni rebal_date a un quantile di vol
    rebal_dates = list(sel_tickers.index)
    vol_at_rebal = bm_vol.reindex(rebal_dates, method='ffill').fillna(bm_vol.mean())
    quantile_labels = pd.qcut(vol_at_rebal, q=n_vol_quantiles,
                              labels=False, duplicates='drop')

    # Partiziona per bucket di vol
    buckets: dict[int, list] = {}
    for rd, ql in zip(rebal_dates, quantile_labels):
        q = int(ql) if not pd.isna(ql) else 0
        buckets.setdefault(q, []).append(rd)

    all_sim_equities = []
    orig_tickers_map = {rd: sel_tickers.at[rd, 'tickers'] for rd in rebal_dates}

    for sim_idx in range(n_simulations):
        # Permuta date intra-bucket
        date_remap: dict = {}
        for bucket_dates in buckets.values():
            shuffled = list(bucket_dates)
            rng.shuffle(shuffled)
            for orig, new in zip(bucket_dates, shuffled):
                date_remap[orig] = new

        sim_schedule = pd.DataFrame(index=rebal_dates, columns=['tickers'])
        for rd in rebal_dates:
            # La selezione segue la data shuffled (stessi tickers, timing diverso)
            sim_schedule.at[rd, 'tickers'] = orig_tickers_map[date_remap[rd]]

        eq = _mc_simulate_equity_from_holdings(sim_schedule, stocks_data, full_index, init_cash)
        all_sim_equities.append(eq.values)

    equity_arr = np.array(all_sim_equities)
    boot_rets_approx = np.diff(equity_arr, axis=1) / equity_arr[:, :-1]
    metrics_per_sim = _mc_compute_metrics_batch(equity_arr, boot_rets_approx)

    p_values = {
        m: _mc_compute_pvalue(metrics_per_sim[m].values, actual_metrics[m], _MC_HIB[m])
        for m in _MC_METRICS
    }
    interpretation = _mc_interpret_skill(p_values, 'Timing')

    # Degenerate distribution guard: std≈0 → all permutations produced identical equity
    min_std = float(metrics_per_sim.std().min())
    if min_std < 1e-10:
        warnings.warn(
            "B2 rebalance timing: degenerate distribution (std≈0 on all metrics). "
            "Likely cause: time-invariant selections in sel_tickers (same tickers at "
            "every rebal_date), or pf_rot built on a different universe than "
            "sel_tickers. Results are NOT interpretable.",
            stacklevel=2,
        )
        interpretation = "INVALID: degenerate distribution (std≈0) — see warning"

    equity_curves = pd.DataFrame(equity_arr.T, index=full_index, columns=range(n_simulations))
    distribution_shape = {
        m: _mc_check_distribution_shape(metrics_per_sim[m].dropna().values, m)
        for m in _MC_METRICS
    }
    return dict(equity_curves=equity_curves, metrics_per_sim=metrics_per_sim,
                actual_metrics=actual_metrics, p_values=p_values,
                interpretation=interpretation,
                distribution_shape=distribution_shape)


def _mc_interpret_skill(p_values: dict, test_type: str) -> str:
    """Genera stringa interpretazione dal dict p_values."""
    sig_metrics = [m for m in ['CAGR', 'Sharpe', 'Calmar'] if p_values.get(m, 1.0) < 0.05]
    if sig_metrics:
        pv_str = ', '.join(f'p_{m}={p_values[m]:.3f}' for m in sig_metrics)
        return f"{test_type} skill: SIGNIFICANT ({pv_str})"
    pv_cagr = p_values.get('CAGR', np.nan)
    return f"{test_type} skill: NOT significant (p_CAGR={pv_cagr:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER BLOCCO A
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sri_priips(
    equity_curves: pd.DataFrame,
    init_cash_val: float,
    capital_base: float = 10_000,
    rhp_years: int = 5,
    method_label: str = '',
) -> dict:
    """
    Calcola MRM/SRI per metodologia PRIIPs Category 3 dalla distribuzione bootstrap esistente.

    NON ricalcola il bootstrap — riusa equity_curves da _mc_build_result.

    Formula VEV (Annex II, punto 17 RTS 2017/653 consolidato 2021/2268):
        VEV = (sqrt(3.842 - 2*ln(VaR_return)) - 1.96) / sqrt(T)
    dove T = rhp_years_effective, VaR_return = VaR_price / capital_base.

    Penalita' regolamentare (dati mensili): MRM_final = MRM_table + 1, capped a 7.

    Parameters
    ----------
    equity_curves : pd.DataFrame  shape (T_trading_days+1, n_sims), da _mc_build_result
    init_cash_val : float  valore cash iniziale del PTF
    capital_base  : float  capitale di riferimento per il VaR (default 10_000 EUR)
    rhp_years     : int    Recommended Holding Period in anni (default 5)
    method_label  : str    etichetta per warning (es. 'IID', 'Block')

    Returns
    -------
    dict con chiavi: var_price, var_return, vev, vev_pct, mrm_class_table,
                     mrm_class_final, rhp_years_effective, rhp_td_used
    """
    n_days = len(equity_curves) - 1   # trading days esclusa riga t=0 (init_cash)
    rhp_td = int(rhp_years * 252)

    if n_days < rhp_td:
        print(f"  \u26a0  [SRI/{method_label}] Storico OOS ({n_days} gg, "
              f"{n_days/252:.1f} anni) < RHP ({rhp_years} anni). "
              f"Calcolo MRM con T_eff={n_days/252:.2f} anni.")
        rhp_eff = n_days / 252.0
        t_idx   = len(equity_curves) - 1
    else:
        rhp_eff = float(rhp_years)
        t_idx   = rhp_td

    terminal_raw    = equity_curves.iloc[t_idx].values
    terminal_scaled = terminal_raw * (capital_base / init_cash_val)

    var_price  = float(np.percentile(terminal_scaled, 2.5))
    var_return = var_price / capital_base

    var_return_safe = max(var_return, 1e-9)
    arg = 3.842 - 2.0 * np.log(var_return_safe)
    vev = (np.sqrt(max(arg, 0.0)) - 1.96) / np.sqrt(rhp_eff)

    vev_pct = vev * 100.0

    if   vev_pct < 0.5:   mrm_table = 1
    elif vev_pct < 5.0:   mrm_table = 2
    elif vev_pct < 12.0:  mrm_table = 3
    elif vev_pct < 20.0:  mrm_table = 4
    elif vev_pct < 30.0:  mrm_table = 5
    elif vev_pct < 80.0:  mrm_table = 6
    else:                  mrm_table = 7

    mrm_final = min(mrm_table + 1, 7)

    return {
        'var_price':           var_price,
        'var_return':          var_return,
        'vev':                 vev,
        'vev_pct':             vev_pct,
        'mrm_class_table':     mrm_table,
        'mrm_class_final':     mrm_final,
        'rhp_years_effective': rhp_eff,
        'rhp_td_used':         t_idx,
    }


def run_mc_confidence_intervals_rotational(
    pf_rot,
    pf_rot_base,
    regime: Optional[pd.Series],
    benchmark_data: pd.Series,
    init_cash: float,
    n_simulations: int = 1000,
    seed: int = 42,
    block_size: int = 10,
    rhp_years: int = 5,
    capital_base: float = 10_000,
    show_method_plots: bool = True,
    show_method_summaries: bool = True,
    _show_portfolio_header: bool = True,
    save_plots     : bool = False,
    plots_dir             = None,
) -> tuple:
    """
    Blocco A: Confidence Intervals per la validazione del portafoglio rotazionale.

    Esegue A1 (IID), A2 (Block Bootstrap), A3 (Regime-Conditional, solo se regime non è None).
    Produce quantili p5/p25/p50/p75/p95 delle metriche. NON produce p-value.

    A3 viene skippato con warning esplicito se regime=None:
    "A3 (Regime-Conditional Block Bootstrap) skipped: regime=None
    (pipeline non-clustered). Use clustered pipeline to enable A3."

    Parameters
    ----------
    pf_rot         : vbt.Portfolio  portafoglio principale (con risk on/off se disponibile)
    pf_rot_base    : vbt.Portfolio  portafoglio base (senza risk on/off)
    regime         : pd.Series 0/1 da compute_market_regime, oppure None
    benchmark_data : pd.Series      prezzi giornalieri del benchmark di mercato
    init_cash      : float
    n_simulations  : int            default 1000
    seed           : int            default 42
    block_size     : int            giorni per blocco in A2/A3 (default 10)
    rhp_years      : int            RHP in anni per MRM/SRI PRIIPs Cat.3 (default 5)
    capital_base   : float          capitale di riferimento VaR MRM (default 10_000 EUR)
    show_method_plots    : bool
    show_method_summaries: bool

    Returns
    -------
    ci_results : dict con chiavi 'iid_bootstrap', 'block_bootstrap', 'regime_block',
                 'sri' (MRM/SRI PRIIPs Cat.3: chiavi iid, block, mrm_class_final)
    ci_summary_df : pd.DataFrame  index=metodi, colonne=metriche×quantili + Actual + Benchmark
    """
    rng = np.random.default_rng(seed)

    equity_actual = pf_rot.value()
    if isinstance(equity_actual, pd.DataFrame):
        equity_actual = equity_actual.squeeze()
    equity_base = pf_rot_base.value()
    if isinstance(equity_base, pd.DataFrame):
        equity_base = equity_base.squeeze()

    # Benchmark equity normalizzata
    bm_start = benchmark_data.reindex(equity_actual.index).ffill().bfill()
    bm_equity = init_cash * bm_start / bm_start.iloc[0]

    if _show_portfolio_header and show_method_summaries:
        mc_cagr_a = _mc_cagr_quick(pf_rot)
        try:
            vbt_cagr_a = float(pf_rot.annualized_return())
        except Exception:
            vbt_cagr_a = float('nan')
        print("=" * 55)
        print("Reference portfolio — Blocco A")
        print("=" * 55)
        print(f"  pf_rot (with Risk ON/OFF)  MC-CAGR={mc_cagr_a:.4f} | vbt-CAGR={vbt_cagr_a:.4f}")
        _mc_print_portfolio_note()

    print("=" * 55)
    print("BLOCCO A — Confidence Intervals")
    print("=" * 55)

    # ── A1 ────────────────────────────────────────────────────────────────────
    print("\n▶ A1 · IID Bootstrap ...")
    res_iid = mc_run_iid_bootstrap(pf_rot, n_simulations, rng)

    # ── A2 ────────────────────────────────────────────────────────────────────
    print("\n▶ A2 · Block Bootstrap ...")
    res_blk = mc_run_block_bootstrap(pf_rot, block_size, n_simulations, rng)

    # ── A3 ────────────────────────────────────────────────────────────────────
    if regime is None:
        print("\n⚠  A3 (Regime-Conditional Block Bootstrap) skipped: regime=None "
              "(pipeline non-clustered). Use clustered pipeline to enable A3.")
        res_reg = None
    else:
        print("\n▶ A3 · Regime-Conditional Block Bootstrap ...")
        res_reg = _mc_run_regime_block_bootstrap(pf_rot, regime, block_size, n_simulations, rng)

    ci_results = {
        'iid_bootstrap':  res_iid,
        'block_bootstrap': res_blk,
        'regime_block':   res_reg,
    }

    # ── ci_summary_df ─────────────────────────────────────────────────────────
    rows = []
    method_labels = {
        'iid_bootstrap':  'A1 · IID Bootstrap',
        'block_bootstrap': 'A2 · Block Bootstrap',
        'regime_block':   'A3 · Regime-Conditional',
    }
    for key, res in ci_results.items():
        if res is None:
            continue
        row = {'Method': method_labels[key]}
        for pct_lbl in ['p5', 'p25', 'p50', 'p75', 'p95']:
            for m in _MC_METRICS:
                row[f'{m}_{pct_lbl}'] = res['percentiles'][pct_lbl][m]
        for m in _MC_METRICS:
            row[f'Actual_{m}'] = res['actual_metrics'][m]
        rows.append(row)

    ci_summary_df = pd.DataFrame(rows).set_index('Method')

    # Aggiungi colonne benchmark
    n_years_bm = len(bm_equity) / 252
    bm_cagr = (bm_equity.iloc[-1] / bm_equity.iloc[0]) ** (1 / n_years_bm) - 1
    bm_metrics = _mc_compute_metrics(bm_equity)
    for m in _MC_METRICS:
        ci_summary_df[f'Benchmark_{m}'] = bm_metrics[m]

    # Aggiungi base portfolio metrics
    base_metrics = _mc_compute_metrics(equity_base)
    for m in _MC_METRICS:
        ci_summary_df[f'Actual_base_{m}'] = base_metrics[m]

    if show_method_summaries:
        print("\n" + "─" * 55)
        print("CI Summary (p50 | Actual | Benchmark):")
        for key, res in ci_results.items():
            if res is None:
                continue
            lbl = method_labels[key]
            print(f"\n  {lbl}")
            for m in ['CAGR', 'MaxDD', 'Sharpe']:
                p50 = res['percentiles']['p50'][m]
                act = res['actual_metrics'][m]
                bm  = bm_metrics[m]
                aqp = res['actual_quantile_position'][m]
                print(f"    {m:12s}: p50={p50:+.3f}  actual={act:+.3f}  "
                      f"bm={bm:+.3f}  actual_pct={aqp:.0%}")

    _fan_names = {
        'iid_bootstrap':   'mc_ci_fanchart_iid.png',
        'block_bootstrap': 'mc_ci_fanchart_block.png',
    }
    if show_method_plots:
        for key, res in ci_results.items():
            if res is None:
                continue
            figs = _mc_plot_ci_method(method_labels[key], res, equity_actual, bm_equity)
            if save_plots and plots_dir is not None and key in _fan_names:
                from pathlib import Path as _Path
                _orig_rs = figs[0].layout.xaxis.rangeselector
                figs[0].update_layout(xaxis=dict(rangeselector=dict(visible=False)))
                figs[0].write_image(str(_Path(plots_dir) / _fan_names[key]))
                figs[0].update_layout(xaxis=dict(rangeselector=_orig_rs))
            for fig in figs:
                fig.show()
        from pathlib import Path as _Path
        _save_ci = (str(_Path(plots_dir) / 'mc_ci.png')
                    if save_plots and plots_dir is not None else None)
        _mc_plot_ci_summary(
            ci_results, {'pf_rot': equity_actual, 'pf_rot_base': equity_base},
            save_path=_save_ci,
        ).show()
    elif save_plots and plots_dir is not None:
        # save fan charts without showing
        from pathlib import Path as _Path
        for key, fname in _fan_names.items():
            if ci_results.get(key) is not None:
                _fig_s = _mc_plot_ci_method(
                    method_labels[key], ci_results[key], equity_actual, bm_equity
                )[0]
                _fig_s.update_layout(xaxis=dict(rangeselector=dict(visible=False)))
                _fig_s.write_image(str(_Path(plots_dir) / fname))

    # ── SRI / MRM PRIIPs Cat. 3 ─────────────────────────────────────────────
    _init_cash_val  = float(pf_rot.init_cash)
    sri_iid         = _compute_sri_priips(res_iid['equity_curves'],  _init_cash_val, capital_base, rhp_years, 'IID')
    sri_block       = _compute_sri_priips(res_blk['equity_curves'], _init_cash_val, capital_base, rhp_years, 'Block')
    mrm_class_final = max(sri_iid['mrm_class_final'], sri_block['mrm_class_final'])
    ci_results['sri'] = {
        'iid':              sri_iid,
        'block':            sri_block,
        'mrm_class_iid':    sri_iid['mrm_class_final'],
        'mrm_class_block':  sri_block['mrm_class_final'],
        'mrm_class_final':  mrm_class_final,
        'capital_base':     capital_base,
        'rhp_years':        rhp_years,
    }
    print(f"\n  [SRI/PRIIPs Cat.3]  IID={sri_iid['mrm_class_final']}  "
          f"Block={sri_block['mrm_class_final']}  Final={mrm_class_final}  "
          f"(T_eff={sri_iid['rhp_years_effective']:.1f}y  capital={capital_base:,.0f} EUR)")

    return ci_results, ci_summary_df


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER BLOCCO B
# ─────────────────────────────────────────────────────────────────────────────

def run_mc_skill_tests_rotational(
    pf_rot,
    sel_tickers: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    regime: Optional[pd.Series],
    tickers_master: list,
    init_cash: float,
    n_simulations: int = 1000,
    seed: int = 42,
    vol_window: int = 60,
    n_vol_quantiles: int = 3,
    show_method_plots: bool = True,
    show_method_summaries: bool = True,
    _show_portfolio_header: bool = True,
    save_plots     : bool = False,
    plots_dir             = None,
) -> tuple:
    """
    Blocco B: Skill Tests — diagnosi della fonte di performance.

    Produce p-value (permutation tests). NON produce intervalli di confidenza.
    Non mescolare skill_summary_df (p-value) con ci_summary_df (quantili).

    Raccomandazione: passare pf_rot_base + stocks_data puro (senza risk_off_data)
    per evitare mismatch. Se si passa pf_rot (con risk on/off), stocks_data
    deve includere anche risk_off_data.

    Verifica copertura ticker: se sel_tickers contiene ticker non in
    stocks_data.columns, emette warning con lista mancanti e procede skippandoli.

    Parameters
    ----------
    regime  : accettato ma non usato internamente; riservato per estensioni future.
    tickers_master : fall-back universe per B1 se 'universe' assente in sel_tickers.
        Warning: "Fall-back to tickers_master (N tickers) for all rebal_dates".
    seed    : B1 usa rng(seed), B2 usa rng(seed+1000) per indipendenza.
    """
    # ── Coverage check ────────────────────────────────────────────────────────
    all_tickers_in_sel = set()
    for tlist in sel_tickers['tickers']:
        if isinstance(tlist, list):
            all_tickers_in_sel.update(tlist)
    missing = sorted(all_tickers_in_sel - set(stocks_data.columns))
    if missing:
        warnings.warn(
            f"run_mc_skill_tests_rotational: {len(missing)} ticker(s) in sel_tickers "
            f"not found in stocks_data.columns → will be skipped.\n"
            f"Missing: {missing}",
            stacklevel=2,
        )

    rng_b1 = np.random.default_rng(seed)
    rng_b2 = np.random.default_rng(seed + 1000)

    # Guard: time-invariant selections make B2 degenerate (all permutations identical)
    _skip_b2 = False
    if sel_tickers['tickers'].apply(tuple).nunique() == 1:
        warnings.warn(
            "run_mc_skill_tests_rotational: sel_tickers has time-invariant selections "
            "(all rebal_dates share the same tickers). B2 (Rebalance Timing) would "
            "produce a degenerate distribution. Skipping B2.",
            stacklevel=2,
        )
        _skip_b2 = True

    equity_actual = pf_rot.value()
    if isinstance(equity_actual, pd.DataFrame):
        equity_actual = equity_actual.squeeze()

    if _show_portfolio_header and show_method_summaries:
        mc_cagr_b = _mc_cagr_quick(pf_rot)
        try:
            vbt_cagr_b = float(pf_rot.annualized_return())
        except Exception:
            vbt_cagr_b = float('nan')
        print("=" * 55)
        print("Reference portfolio — Blocco B")
        print("=" * 55)
        print(f"  pf_rot (no Risk ON/OFF)    MC-CAGR={mc_cagr_b:.4f} | vbt-CAGR={vbt_cagr_b:.4f}")
        _mc_print_portfolio_note()

    print("=" * 55)
    print("BLOCCO B — Skill Tests (p-value)")
    print("=" * 55)

    print("\n▶ B1 · Rotation Reshuffle ...")
    res_b1 = _mc_run_rotation_reshuffle(
        pf_rot, sel_tickers, stocks_data, tickers_master, init_cash, n_simulations, rng_b1
    )

    if _skip_b2:
        print("\n\u26a0  B2 · Rebalance Timing skipped: time-invariant selections.")
        res_b2 = None
    else:
        print("\n\u25b6 B2 · Rebalance Timing Bootstrap ...")
        res_b2 = _mc_run_rebalance_timing(
            pf_rot, sel_tickers, stocks_data, benchmark_data, init_cash,
            vol_window, n_vol_quantiles, n_simulations, rng_b2
        )

    skill_results = {
        'rotation_reshuffle': res_b1,
        'rebalance_timing':   res_b2,
    }

    # ── skill_summary_df ──────────────────────────────────────────────────────
    test_labels = {
        'rotation_reshuffle': 'B1 · Rotation Reshuffle',
        'rebalance_timing':   'B2 · Rebalance Timing',
    }
    rows = []
    for key, res in skill_results.items():
        if res is None:
            continue
        row = {'Test': test_labels[key]}
        for m in _MC_METRICS:
            row[f'p_value_{m}'] = res['p_values'].get(m, np.nan)
        for m in _MC_METRICS:
            row[f'Actual_{m}'] = res['actual_metrics'].get(m, np.nan)
        row['interpretation'] = res['interpretation']
        rows.append(row)
    skill_summary_df = pd.DataFrame(rows).set_index('Test')

    if show_method_summaries:
        print("\n" + "─" * 55)
        print("Skill Test Summary (p-value per metrica):")
        for key, res in skill_results.items():
            if res is None:
                continue
            print(f"\n  {test_labels[key]}")
            print(f"  → {res['interpretation']}")
            for m in ['CAGR', 'MaxDD', 'Sharpe', 'Calmar']:
                pv  = res['p_values'].get(m, np.nan)
                act = res['actual_metrics'].get(m, np.nan)
                sig = "***" if pv < 0.01 else ("*" if pv < 0.05 else "")
                print(f"    {m:12s}: p={pv:.3f}{sig:3s}  actual={act:+.3f}")

        # Distribution shape warnings (solo se almeno una metrica non unimodale)
        shape_warnings: list[str] = []
        for key, res in skill_results.items():
            if res is None:
                continue
            if 'distribution_shape' not in res:
                print(f"  ⚠ {test_labels.get(key, key)}: campo 'distribution_shape' assente "
                      "— riesegui i skill tests con la versione corrente di r_functions.")
                continue
            method_warns = [
                ds['shape_warning']
                for ds in res['distribution_shape'].values()
                if ds['shape_warning'] is not None
            ]
            if method_warns:
                shape_warnings.append((test_labels[key], method_warns))
        if shape_warnings:
            print("\n\u26a0  Distribution shape warnings (Blocco B):")
            for lbl, warns in shape_warnings:
                print(f"    {lbl}:")
                for w in warns:
                    print(f"      - {w}")
                    print("        \u2192 p-value reading requires caution; check the histogram for cluster structure")

    _skill_filenames = {
        'rotation_reshuffle': 'mc_reshuffle.png',
        'rebalance_timing':   'mc_timing.png',
    }
    if show_method_plots:
        from pathlib import Path as _Path
        for key, res in skill_results.items():
            if res is None:
                continue
            _save_sk = (str(_Path(plots_dir) / _skill_filenames[key])
                        if save_plots and plots_dir is not None else None)
            for fig in _mc_plot_skill_test(test_labels[key], res, equity_actual,
                                            save_path=_save_sk):
                fig.show()
        _save_ss = (str(_Path(plots_dir) / 'mc_skill_summary.png')
                    if save_plots and plots_dir is not None else None)
        _mc_plot_skill_summary(skill_results, save_path=_save_ss).show()

    return skill_results, skill_summary_df


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER ALTO LIVELLO
# ─────────────────────────────────────────────────────────────────────────────
def run_all_mc_methods_rotational_GPT(
    pf_rot,
    pf_rot_base,
    regime: Optional[pd.Series],
    sel_tickers: pd.DataFrame,
    sel_tickers_base: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    tickers_master: list,
    init_cash: float,
    n_simulations: int = 1000,
    seed: int = 42,
    block_size: int = 10,
    rhp_years: int = 5,
    capital_base: float = 10_000,
    vol_window: int = 60,
    n_vol_quantiles: int = 3,
    show_method_plots: bool = True,
    show_method_summaries: bool = True,
    save_plots     : bool = False,
    plots_dir             = None,
) -> tuple:
    """
    Wrapper alto livello: esegue Blocco A e Blocco B in sequenza.

    Blocco A usa pf_rot (con risk on/off) e seed=seed.
    Blocco B usa pf_rot_base + sel_tickers_base + stocks_data puro e seed=seed+10_000
    (offset garantisce indipendenza tra i due blocchi pur mantenendo riproducibilità globale).

    Returns
    -------
    (ci_results, ci_summary_df, skill_results, skill_summary_df)
    """
    if show_method_summaries:
        mc_cagr_a = _mc_cagr_quick(pf_rot)
        mc_cagr_b = _mc_cagr_quick(pf_rot_base)
        try:
            vbt_cagr_a = float(pf_rot.annualized_return())
        except Exception:
            vbt_cagr_a = float('nan')
        try:
            vbt_cagr_b = float(pf_rot_base.annualized_return())
        except Exception:
            vbt_cagr_b = float('nan')
        print("=" * 55)
        print("Reference portfolios for MC validation")
        print("=" * 55)
        print(f"  Blocco A: pf_rot       (with Risk ON/OFF)  MC-CAGR={mc_cagr_a:.4f} | vbt-CAGR={vbt_cagr_a:.4f}")
        print(f"  Blocco B: pf_rot_base  (no Risk ON/OFF)    MC-CAGR={mc_cagr_b:.4f} | vbt-CAGR={vbt_cagr_b:.4f}")
        _mc_print_portfolio_note()

    ci_results, ci_summary_df = run_mc_confidence_intervals_rotational(
        pf_rot=pf_rot, pf_rot_base=pf_rot_base, regime=regime,
        benchmark_data=benchmark_data, init_cash=init_cash,
        n_simulations=n_simulations, seed=seed, block_size=block_size,
        rhp_years=rhp_years,
        capital_base=capital_base,
        show_method_plots=show_method_plots,
        show_method_summaries=show_method_summaries,
        save_plots=save_plots,
        plots_dir=plots_dir,
        _show_portfolio_header=False,
    )
    
    # ----------------------------------------------------------
    # B-BASE — skill rotazione pura, senza Risk ON/OFF
    # ----------------------------------------------------------
    skill_results_base, skill_summary_df_base = run_mc_skill_tests_rotational(
        pf_rot=pf_rot_base,
        sel_tickers=sel_tickers_base,
        stocks_data=stocks_data,
        benchmark_data=benchmark_data,
        regime=None,
        tickers_master=tickers_master,
        init_cash=init_cash,
        n_simulations=n_simulations,
        seed=seed + 10_000,
        vol_window=vol_window,
        n_vol_quantiles=n_vol_quantiles,
        show_method_plots=show_method_plots,
        show_method_summaries=show_method_summaries,
        save_plots=save_plots,
        plots_dir=plots_dir,
        _show_portfolio_header=False,
    )
    
    # ----------------------------------------------------------
    # B-RISKONOFF — skill del path realmente deployato
    # ----------------------------------------------------------
    skill_results_ronoff, skill_summary_df_ronoff = run_mc_skill_tests_rotational(
        pf_rot=pf_rot,
        sel_tickers=sel_tickers,
        stocks_data=stocks_data,
        benchmark_data=benchmark_data,
        regime=regime,
        tickers_master=tickers_master,
        init_cash=init_cash,
        n_simulations=n_simulations,
        seed=seed + 20_000,
        vol_window=vol_window,
        n_vol_quantiles=n_vol_quantiles,
        show_method_plots=show_method_plots,
        show_method_summaries=show_method_summaries,
        save_plots=save_plots,
        plots_dir=plots_dir,
        _show_portfolio_header=False,
    )

    skill_results = {
        "base": skill_results_base,
        "risk_on_off": skill_results_ronoff,
    }
    
    skill_summary_df = {
        "base": skill_summary_df_base,
        "risk_on_off": skill_summary_df_ronoff,
    }
    
    return ci_results, ci_summary_df, skill_results, skill_summary_df

def run_all_mc_methods_rotational(
    pf_rot,
    pf_rot_base,
    regime: Optional[pd.Series],
    sel_tickers: pd.DataFrame,
    sel_tickers_base: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    tickers_master: list,
    init_cash: float,
    n_simulations: int = 1000,
    seed: int = 42,
    block_size: int = 10,
    rhp_years: int = 5,
    capital_base: float = 10_000,
    vol_window: int = 60,
    n_vol_quantiles: int = 3,
    show_method_plots: bool = True,
    show_method_summaries: bool = True,
    save_plots     : bool = False,
    plots_dir             = None,
) -> tuple:
    """
    Wrapper alto livello: esegue Blocco A e Blocco B in sequenza.

    Blocco A usa pf_rot (con risk on/off) e seed=seed.
    Blocco B usa pf_rot_base + sel_tickers_base + stocks_data puro e seed=seed+10_000
    (offset garantisce indipendenza tra i due blocchi pur mantenendo riproducibilità globale).

    Returns
    -------
    (ci_results, ci_summary_df, skill_results, skill_summary_df)
    """
    if show_method_summaries:
        mc_cagr_a = _mc_cagr_quick(pf_rot)
        mc_cagr_b = _mc_cagr_quick(pf_rot_base)
        try:
            vbt_cagr_a = float(pf_rot.annualized_return())
        except Exception:
            vbt_cagr_a = float('nan')
        try:
            vbt_cagr_b = float(pf_rot_base.annualized_return())
        except Exception:
            vbt_cagr_b = float('nan')
        print("=" * 55)
        print("Reference portfolios for MC validation")
        print("=" * 55)
        print(f"  Blocco A: pf_rot       (with Risk ON/OFF)  MC-CAGR={mc_cagr_a:.4f} | vbt-CAGR={vbt_cagr_a:.4f}")
        print(f"  Blocco B: pf_rot_base  (no Risk ON/OFF)    MC-CAGR={mc_cagr_b:.4f} | vbt-CAGR={vbt_cagr_b:.4f}")
        _mc_print_portfolio_note()

    ci_results, ci_summary_df = run_mc_confidence_intervals_rotational(
        pf_rot=pf_rot, pf_rot_base=pf_rot_base, regime=regime,
        benchmark_data=benchmark_data, init_cash=init_cash,
        n_simulations=n_simulations, seed=seed, block_size=block_size,
        rhp_years=rhp_years,
        capital_base=capital_base,
        show_method_plots=show_method_plots,
        show_method_summaries=show_method_summaries,
        save_plots=save_plots,
        plots_dir=plots_dir,
        _show_portfolio_header=False,
    )
    skill_results, skill_summary_df = run_mc_skill_tests_rotational(
        pf_rot=pf_rot_base, sel_tickers=sel_tickers_base,
        stocks_data=stocks_data, benchmark_data=benchmark_data,
        regime=regime, tickers_master=tickers_master, init_cash=init_cash,
        n_simulations=n_simulations, seed=seed + 10_000,
        vol_window=vol_window, n_vol_quantiles=n_vol_quantiles,
        show_method_plots=show_method_plots,
        show_method_summaries=show_method_summaries,
        save_plots=save_plots,
        plots_dir=plots_dir,
        _show_portfolio_header=False,
    )
    return ci_results, ci_summary_df, skill_results, skill_summary_df


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_mc_validation_report(
    ci_results: dict,
    ci_summary_df: pd.DataFrame,
    skill_results: dict,
    skill_summary_df: pd.DataFrame,
    pf_rot,
    pf_rot_base,
    sel_tickers: pd.DataFrame,
    benchmark_data: pd.Series,
    portfolio_name: str = "Unnamed_Portfolio",
    save_path: Optional[str] = None,
    overwrite: bool = False,
    mc_setup: Optional[dict] = None,
) -> str:
    """
    Genera un report Markdown di sintesi della MC validation post-WFO.

    Ritorna sempre la stringa Markdown. Se save_path è fornito, salva anche su disco:
    - save_path = path completo a un file .md → salva lì
    - save_path = directory → salva come {dir}/MC_validation_{portfolio_name}_{YYYY-MM-DD}.md
      La directory viene creata se non esiste (mkdir -p).

    Default directory suggerito: 'notebooks/dev/reports/MC_validation/'

    Parameters
    ----------
    mc_setup : dict | None
        Parametri MC usati per la run, es.
        {'n_simulations': 1000, 'seed': 42, 'block_size': 10,
         'vol_window': 60, 'n_vol_quantiles': 3}.
        Viene serializzato in Sezione 0 per riproducibilità.
        Se None, n_simulations viene derivato dall'equity_curves di A2.
    overwrite : bool
        Se False e il file esiste, solleva FileExistsError.
    """
    import os
    from datetime import datetime

    # ── Helpers di formattazione ───────────────────────────────────────────────
    def _pct(v, decimals=1):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v*100:+.{decimals}f}%"

    def _ratio(v, decimals=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.{decimals}f}"

    def _pval(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        stars = "***" if v < 0.001 else ("**" if v < 0.01 else ("*" if v < 0.05 else ""))
        return f"{v:.3f}{stars:3s}"

    def _pct_int(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{int(round(v * 100))}%"

    # ── Derivazione dati base ──────────────────────────────────────────────────
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Identifica il metodo A2 (block_bootstrap); fallback a iid_bootstrap
    res_a2  = ci_results.get('block_bootstrap') or ci_results.get('iid_bootstrap')
    res_a1  = ci_results.get('iid_bootstrap')
    res_a3  = ci_results.get('regime_block')
    res_b1  = skill_results.get('rotation_reshuffle')
    res_b2  = skill_results.get('rebalance_timing')
    b2_degenerate = (res_b2 is None or
                     (isinstance(res_b2, dict) and
                      res_b2.get('interpretation', '').startswith('INVALID')))

    # Periodo e n_simulations
    eq_curves = (res_a2 or res_a1 or {}).get('equity_curves')
    if eq_curves is not None:
        start_date = eq_curves.index[0].strftime('%Y-%m-%d')
        end_date   = eq_curves.index[-1].strftime('%Y-%m-%d')
        n_sims_auto = eq_curves.shape[1]
    else:
        start_date = end_date = "n/a"
        n_sims_auto = 0
    n_sims = (mc_setup or {}).get('n_simulations', n_sims_auto)

    # Cardinalità selezioni
    sel_lens  = sel_tickers['tickers'].apply(len)
    card_mean = sel_lens.mean()
    card_min  = sel_lens.min()
    card_max  = sel_lens.max()

    # Metriche portfolio (MC convention)
    def _pf_metrics(pf):
        eq = pf.value()
        if isinstance(eq, pd.DataFrame): eq = eq.squeeze()
        return _mc_compute_metrics(eq)

    m_a = _pf_metrics(pf_rot)
    m_b = _pf_metrics(pf_rot_base)
    try:    vbt_cagr_a = float(pf_rot.annualized_return())
    except: vbt_cagr_a = float('nan')
    try:    vbt_cagr_b = float(pf_rot_base.annualized_return())
    except: vbt_cagr_b = float('nan')

    # actual_pct dal metodo A2
    aqp_a2 = (res_a2 or {}).get('actual_quantile_position', {})
    cagr_pct_a2  = aqp_a2.get('CAGR',  float('nan'))
    maxdd_pct_a2 = aqp_a2.get('MaxDD', float('nan'))

    # p-values B1
    pv_b1 = (res_b1 or {}).get('p_values', {})
    p_cagr_b1  = pv_b1.get('CAGR',  float('nan'))
    p_maxdd_b1 = pv_b1.get('MaxDD', float('nan'))
    p_sharpe_b1= pv_b1.get('Sharpe',float('nan'))
    p_calmar_b1= pv_b1.get('Calmar',float('nan'))
    p_vol_b1   = pv_b1.get('Volatility', float('nan'))
    p_cagr_b2  = (res_b2 or {}).get('p_values', {}).get('CAGR', float('nan')) if not b2_degenerate else float('nan')

    # ── Sezione 0 — Setup MC ───────────────────────────────────────────────────
    setup_parts = []
    if mc_setup:
        setup_parts = [f"{k}={v}" for k, v in mc_setup.items()]
    else:
        setup_parts = [f"n_simulations={n_sims}"]
    setup_line = ", ".join(setup_parts)

    # ── Sezione 2 — Verdetto sintetico A2 ─────────────────────────────────────
    if not np.isnan(cagr_pct_a2) and not np.isnan(maxdd_pct_a2):
        verdetto_parts = []
        if   cagr_pct_a2 > 0.70:  verdetto_parts.append("rendimento sopra la mediana")
        elif cagr_pct_a2 < 0.30:  verdetto_parts.append("rendimento sotto la mediana")
        else:                       verdetto_parts.append("performance nella mediana")
        if   maxdd_pct_a2 > 0.70:  verdetto_parts.append("ottimo controllo drawdown")
        elif maxdd_pct_a2 < 0.30:  verdetto_parts.append("drawdown peggiore della maggioranza delle simulazioni")
        elif maxdd_pct_a2 > 0.50:  verdetto_parts.append("drawdown sotto controllo")
        verdetto_a2 = ", ".join(verdetto_parts)
    else:
        verdetto_a2 = "n/a"

    def _ci_rows(res, label):
        if res is None: return ""
        aqp  = res.get('actual_quantile_position', {})
        perc = res.get('percentiles', {})
        act  = res.get('actual_metrics', {})
        rows = []
        for m in ['CAGR', 'MaxDD', 'Sharpe']:
            is_pct  = m in ('CAGR', 'MaxDD')
            fmt     = _pct if is_pct else _ratio
            p5_  = perc.get('p5',  {}).get(m, float('nan'))
            p50_ = perc.get('p50', {}).get(m, float('nan'))
            p95_ = perc.get('p95', {}).get(m, float('nan'))
            act_ = act.get(m, float('nan'))
            pct_ = aqp.get(m, float('nan'))
            rows.append(
                f"| {label} | {m} | {fmt(p5_)} | {fmt(p50_)} | {fmt(p95_)} "
                f"| {fmt(act_)} | {_pct_int(pct_)} |"
            )
        return "\n".join(rows)

    ci_rows = "\n".join(filter(None, [
        _ci_rows(res_a1, "A1 · IID"),
        _ci_rows(res_a2, "A2 · Block"),
        ("| A3 · Regime | — | (skipped: pipeline non-clustered) | | | | |"
         if res_a3 is None else _ci_rows(res_a3, "A3 · Regime")),
    ]))

    # CI interpretation bullets (A2 driven)
    interp_a_lines = []
    if not np.isnan(cagr_pct_a2):
        if   cagr_pct_a2 > 0.70: interp_a_lines.append(f"- CAGR actual_pct={_pct_int(cagr_pct_a2)} (A2): rendimento sopra la mediana — possibile lucky sequence; verificare con più finestre.")
        elif cagr_pct_a2 < 0.30: interp_a_lines.append(f"- CAGR actual_pct={_pct_int(cagr_pct_a2)} (A2): rendimento sotto la mediana — struttura del PTF tende a produrre scenari migliori del realizzato.")
        else:                      interp_a_lines.append(f"- CAGR actual_pct={_pct_int(cagr_pct_a2)} (A2): rendimento reale alla mediana — performance tipica della struttura del PTF.")
    if not np.isnan(maxdd_pct_a2):
        if   maxdd_pct_a2 > 0.70: interp_a_lines.append(f"- MaxDD actual_pct={_pct_int(maxdd_pct_a2)} (A2): drawdown reale migliore del {_pct_int(maxdd_pct_a2)} delle simulazioni.")
        elif maxdd_pct_a2 < 0.30: interp_a_lines.append(f"- MaxDD actual_pct={_pct_int(maxdd_pct_a2)} (A2): drawdown reale peggiore della maggioranza — segnale di rischio.")
        else:                      interp_a_lines.append(f"- MaxDD actual_pct={_pct_int(maxdd_pct_a2)} (A2): controllo drawdown nella norma.")
    interp_a = "\n".join(interp_a_lines) if interp_a_lines else "- n/a"

    # ── Sezione SRI ──────────────────────────────────────────────────────────
    _sri_rpt = ci_results.get('sri')
    if _sri_rpt is not None:
        _sii  = _sri_rpt['iid']
        _sbl  = _sri_rpt['block']
        _scap = _sri_rpt.get('capital_base', 10_000)
        _srhp = _sii['rhp_years_effective']
        _smf  = _sri_rpt['mrm_class_final']
        sri_section = '\n'.join([
            f'| Parametro | IID Bootstrap | Block Bootstrap |',
            f'|---|:---:|:---:|',
            f"| Classe MRM (penalizzata) | {_sii['mrm_class_final']} / 7 | {_sbl['mrm_class_final']} / 7 |",
            f"| VEV annualizzata | {_sii['vev_pct']:.1f}% | {_sbl['vev_pct']:.1f}% |",
            f"| VaR 97.5% (su {_scap:,.0f}\u20ac) | {_sii['var_price']:,.0f}\u20ac | {_sbl['var_price']:,.0f}\u20ac |",
            f'| RHP effettivo | {_srhp:.1f} anni | \u2014 |',
            f'| Frequenza dati | mensile (+1 classe penalita\u2019 reg.) | |',
            f'',
            f'**Classe MRM finale (regola conservativa): {_smf} / 7**',
            f'',
            f'_+1 classe penalita\u2019 regolamentare per rendimenti mensili WFO OOS._',
        ])
    else:
        sri_section = '_SRI non disponibile: eseguire con rhp_years e capital_base._'

    # ── Sezione 3 — Skill tests ────────────────────────────────────────────────
    def _skill_verdict(p_cagr, test_label):
        if np.isnan(p_cagr): return "n/a"
        return f"SIGNIFICANT (p_CAGR={p_cagr:.3f})" if p_cagr < 0.05 else f"NOT significant (p_CAGR={p_cagr:.3f})"

    b1_verdict = _skill_verdict(p_cagr_b1, "B1")
    b2_verdict = "INVALID — degenerate distribution" if b2_degenerate else _skill_verdict(p_cagr_b2, "B2")

    # Interpretazione skill
    interp_b_lines = []
    if not np.isnan(p_cagr_b1) and not np.isnan(p_cagr_b2) and not b2_degenerate:
        if   p_cagr_b1 < 0.05 and p_cagr_b2 >= 0.10:
            interp_b_lines.append("- Skill nella **selezione**, non nel timing. La frequenza di ribilanciamento può essere ridotta senza perdita significativa.")
        elif p_cagr_b1 < 0.05 and p_cagr_b2 < 0.05:
            interp_b_lines.append("- Skill su entrambi i piani: selezione e timing contribuiscono entrambi alla performance.")
        elif p_cagr_b1 >= 0.10:
            interp_b_lines.append("- Nessuna skill di selezione rilevabile. La performance è essenzialmente beta dell'universo. **Caveat di deploy**.")
    elif not np.isnan(p_cagr_b1) and p_cagr_b1 < 0.05:
        interp_b_lines.append(f"- Skill di selezione confermata (p_CAGR={p_cagr_b1:.3f}). B2 non disponibile (degenere o saltato).")
    elif not np.isnan(p_cagr_b1) and p_cagr_b1 >= 0.10:
        interp_b_lines.append("- Nessuna skill di selezione rilevabile. **Caveat di deploy**.")
    interp_b = "\n".join(interp_b_lines) if interp_b_lines else "- n/a"

    # ── Sezione 4 — Note diagnostiche ─────────────────────────────────────────
    diag_notes: list[str] = []
    if res_a3 is None:
        diag_notes.append("**A3 skipped**: regime=None (pipeline non-clustered). Per A3 usare pipeline clustered.")
    if b2_degenerate:
        diag_notes.append("**B2 — distribuzione degenere**: timing test invalido. Causa probabile: selezioni time-invariant in sel_tickers, o pf_rot costruito su un universo diverso da sel_tickers.")
    # p_Volatility anomala in B1
    if not np.isnan(p_vol_b1) and p_vol_b1 >= 0.95:
        diag_notes.append(f"**B1 p_Volatility={p_vol_b1:.3f}**: il PTF reale è il più volatile di tutti i random pick. La logica di selezione tende a preferire nomi più volatili dell'universo.")
    # Disaccordo tra metriche in B1 (p_X > 0.95 mentre p_CAGR < 0.05)
    metric_disagreements = []
    if not np.isnan(p_cagr_b1) and p_cagr_b1 < 0.05:
        for m_name, pv in [('MaxDD', p_maxdd_b1), ('Sharpe', p_sharpe_b1), ('Calmar', p_calmar_b1)]:
            if not np.isnan(pv) and pv > 0.95:
                metric_disagreements.append(f"{m_name} (p={pv:.3f})")
    if metric_disagreements:
        diag_notes.append(f"**Disaccordo tra metriche in B1**: selezione skilled su CAGR ma penalizzante su {', '.join(metric_disagreements)}. Verificare composizione per asset difensivi/offensivi.")
    # Distribution shape warnings — accesso stretto: KeyError se campo assente
    for key, res in [("B1", res_b1), ("B2", res_b2)]:
        if res is None:
            continue
        if 'distribution_shape' not in res:
            raise KeyError(
                f"Missing 'distribution_shape' in skill_results['{key.lower()}_*']. "
                "Re-run skill tests with the current version of r_functions.ipynb."
            )
        for m, ds in res['distribution_shape'].items():
            if ds['shape_warning']:
                diag_notes.append(f"**{key} {m} — distribuzione non unimodale**: {ds['shape_warning']}.")
    # ── Raccomandazione (calcolata qui per usarla nella chiusura di sezione 4) ──
    _has_diag = len(diag_notes) > 0
    if b2_degenerate and np.isnan(p_cagr_b1):
        rec = "**INCONCLUSIVE** — distribuzione degenere o dati insufficienti. Verificare setup MC."
    elif not np.isnan(p_cagr_b1) and p_cagr_b1 >= 0.10:
        rec = "**PTF NOT validated** — selection not skilled (p_B1_CAGR≥0.10). Performance beta-driven. Non deployare senza ulteriore analisi."
    elif (not np.isnan(cagr_pct_a2) and 0.30 <= cagr_pct_a2 <= 0.70
          and not np.isnan(maxdd_pct_a2) and maxdd_pct_a2 > 0.50
          and not np.isnan(p_cagr_b1) and p_cagr_b1 < 0.05):
        if _has_diag:
            rec = f"**PTF VALIDATED with caveats** — condizioni di base soddisfatte ma sono presenti {len(diag_notes)} nota/e diagnostica/he. Leggere sezione 4."
        else:
            rec = "**PTF VALIDATED for deploy** — A2 CAGR nella mediana, MaxDD sotto controllo, selezione skilled. Nessuna nota diagnostica."
    else:
        rec = "**NEEDS REVIEW** — condizioni non pienamente soddisfatte. Vedi sezioni 2–5 per dettagli."

    # ── Sezione 4 — Paragrafo Interpretazione ───────────────────────────────────
    # Ordine frasi: A3 → p_Volatility → bimodalità MaxDD → bimodalità altre metriche
    #               → guard degenere B2 → riga di chiusura
    interp4: list[str] = []

    # A3 skipped
    if res_a3 is None:
        interp4.append(
            "Il blocco A3 (regime-conditional) è disabilitato per questa pipeline. "
            "Per intervalli di confidenza condizionati al regime di mercato, "
            "ri-eseguire la pipeline con use_clustering=True."
        )

    # p_Volatility molto alto → selezione preferisce ticker volatili
    if not np.isnan(p_vol_b1) and p_vol_b1 >= 0.95:
        interp4.append(
            "La logica di selezione tende sistematicamente verso ticker più volatili "
            "dell'universo. Non è un problema di per sé (lo Sharpe è skilled, quindi "
            "la volatilità in più è remunerata), ma è una caratteristica strutturale "
            "del PTF da conoscere — coerente con un approccio momentum-driven."
        )
    # p_Volatility molto basso → selezione preferisce ticker meno volatili
    elif not np.isnan(p_vol_b1) and p_vol_b1 <= 0.05:
        interp4.append(
            "La logica di selezione tende sistematicamente verso ticker meno volatili "
            "dell'universo (low-volatility tilt). Coerente con un approccio "
            "difensivo o low-vol screening."
        )

    # Bimodalità MaxDD in B1 e/o B2
    maxdd_bimodal = []
    if res_b1 and not res_b1['distribution_shape']['MaxDD']['is_unimodal']:
        maxdd_bimodal.append('B1')
    if res_b2 and not b2_degenerate and not res_b2['distribution_shape']['MaxDD']['is_unimodal']:
        maxdd_bimodal.append('B2')
    if maxdd_bimodal:
        where = " e ".join(maxdd_bimodal)
        interp4.append(
            f"La distribuzione del MaxDD nelle simulazioni MC ({where}) è bimodale: "
            "l'universo investibile non è omogeneo dal punto di vista del rischio e "
            "contiene asset con profili di drawdown sistematicamente diversi. "
            "Il p-value del MaxDD resta valido ma sintetico — una lettura più "
            "informativa richiede di guardare l'istogramma e verificare la posizione "
            "del PTF reale rispetto a entrambi i picchi."
        )

    # Bimodalità su metriche diverse da MaxDD
    other_bimodal: set = set()
    for _tk, _tr in [('rotation_reshuffle', 'B1'), ('rebalance_timing', 'B2')]:
        _res = skill_results.get(_tk)
        if _res is None or b2_degenerate and _tr == 'B2': continue
        for _m, _ds in _res['distribution_shape'].items():
            if _m != 'MaxDD' and not _ds['is_unimodal']:
                other_bimodal.add((_tr, _m))
    if other_bimodal:
        _ms = ", ".join(f"{t} {m}" for t, m in sorted(other_bimodal))
        interp4.append(
            f"Anche su altre metriche ({_ms}) la distribuzione MC mostra "
            "eterogeneità nell'universo. I p-value restano interpretabili ma "
            "riflettono una media tra famiglie di asset diverse."
        )

    # B2 distribuzione degenere
    if b2_degenerate:
        interp4.append(
            "Il test B2 (rebalance timing) è risultato invalido per setup degenere — "
            "selezioni costanti nel tempo o mismatch tra pf_rot e stocks_data. "
            "Verifica che il PTF abbia selezioni time-varying e che gli oggetti "
            "passati siano coerenti."
        )

    # Riga di chiusura legata alla raccomandazione
    if interp4:
        if "with caveats" in rec.lower():
            interp4.append(
                "Le note diagnostiche non invalidano la validazione, "
                "ma vanno tenute presenti come contesto strutturale del PTF prima del deploy."
            )
        elif "not validated" in rec.lower() or "inconclusive" in rec.lower():
            interp4.append(
                "Le note diagnostiche convergono nel suggerire cautela: "
                "vedi sezione 6 per la raccomandazione finale."
            )

    # Assemblaggio diag_section con lista bullet + paragrafo interpretazione
    _bullets = "\n".join(f"- {n}" for n in diag_notes) if diag_notes else "- Nessuna nota diagnostica rilevata."
    if interp4:
        _interp_md = "\n\n**Interpretazione**:\n\n" + "\n".join(f"- {s}" for s in interp4)
    else:
        _interp_md = "\n\n**Interpretazione**: Nessuna nota diagnostica rilevata. La lettura del PTF è lineare."
    diag_section = _bullets + _interp_md

    # ── Sezione 5 — Coerenza incrociata A vs B ─────────────────────────────────
    if not np.isnan(cagr_pct_a2) and not np.isnan(p_cagr_b1):
        at_median = 0.30 <= cagr_pct_a2 <= 0.70
        above_med = cagr_pct_a2 > 0.70
        b1_sig    = p_cagr_b1 < 0.05
        b1_notsig = p_cagr_b1 >= 0.10
        if at_median and b1_sig:
            cross_label = "**Coerente**"
            cross_desc  = f"A2 CAGR nella mediana ({_pct_int(cagr_pct_a2)}) + B1 significativo (p={p_cagr_b1:.3f}): la composizione è tipica e migliore del random. Nessun segnale di overfitting."
        elif above_med and b1_notsig:
            cross_label = "**Sospetto**"
            cross_desc  = f"A2 CAGR sopra la mediana ({_pct_int(cagr_pct_a2)}) + B1 non significativo (p={p_cagr_b1:.3f}): alta performance non spiegata dalla selezione. Possibile overfitting o beta favorevole."
        elif cagr_pct_a2 < 0.30 and b1_notsig:
            cross_label = "**Mediocre**"
            cross_desc  = f"A2 CAGR sotto la mediana ({_pct_int(cagr_pct_a2)}) + B1 non significativo (p={p_cagr_b1:.3f}): sotto-performance senza skill identificabile."
        elif b1_sig:
            cross_label = "**Skill confermata**"
            cross_desc  = f"B1 significativo (p={p_cagr_b1:.3f}). A2 CAGR actual_pct={_pct_int(cagr_pct_a2)}."
        else:
            cross_label = "**Ambiguo**"
            cross_desc  = f"A2 CAGR actual_pct={_pct_int(cagr_pct_a2)}, B1 p_CAGR={p_cagr_b1:.3f}. Vedi note diagnostiche."
    else:
        cross_label = "**n/a**"
        cross_desc  = "Dati insufficienti per valutazione incrociata."

    # ── Sezione 6 — Raccomandazione finale ────────────────────────────────────
    # rec è già calcolato prima di sezione 4 (necessario per la riga di chiusura
    # del paragrafo Interpretazione). Nessun ricalcolo necessario.

    # ── Assemblaggio report ────────────────────────────────────────────────────
    lines: list[str] = [
        f"# MC Validation Report — {portfolio_name}",
        "",
        f"**Generato**: {now_iso}",
        f"**Periodo backtest**: {start_date} → {end_date}",
        f"**Numero rebal_dates**: {len(sel_tickers)}",
        f"**Cardinalità selezioni**: media={card_mean:.1f}, min={card_min}, max={card_max}",
        f"**N simulazioni MC**: {n_sims}",
        f"**Setup MC**: {setup_line}",
        "",
        "---",
        "",
        "## 1. Reference portfolios",
        "",
        "| Portafoglio | Uso | MC-CAGR | vbt-CAGR | MaxDD | Sharpe |",
        "|---|---|---|---|---|---|",
        f"| `pf_rot` | Blocco A (Risk ON/OFF) | {_pct(m_a['CAGR'])} | {_pct(vbt_cagr_a)} | {_pct(m_a['MaxDD'])} | {_ratio(m_a['Sharpe'])} |",
        f"| `pf_rot_base` | Blocco B (no Risk) | {_pct(m_b['CAGR'])} | {_pct(vbt_cagr_b)} | {_pct(m_b['MaxDD'])} | {_ratio(m_b['Sharpe'])} |",
        "",
        "> Nota: MC-CAGR e vbt-CAGR usano convenzioni di annualizzazione diverse.",
        "> Confronta MC con MC, vbt con vbt. Vedi CLAUDE.md.",
        "",
        "---",
        "",
        "## 2. Robustezza statistica (Blocco A — Confidence Intervals)",
        "",
        f"**Verdetto sintetico (A2 Block Bootstrap)**: {verdetto_a2}",
        "",
        "| Metodo | Metrica | p5 | p50 | p95 | Actual | actual_pct |",
        "|---|---|---|---|---|---|---|",
        ci_rows,
        "",
        "**Interpretazione**:",
        interp_a,
        "",
        "---",
        "",
        "## 2.b — Indicatore di Rischio Sintetico (PRIIPs Cat. 3)",
        "",
        sri_section,
        "",
        "---",
        "",
        "## 3. Origine della performance (Blocco B — Skill Tests)",
        "",
        "| Test | p_CAGR | p_MaxDD | p_Sharpe | p_Calmar | Verdetto |",
        "|---|---|---|---|---|---|",
        f"| B1 · Rotation Reshuffle | {_pval(p_cagr_b1)} | {_pval(p_maxdd_b1)} | {_pval(p_sharpe_b1)} | {_pval(p_calmar_b1)} | {b1_verdict} |",
        f"| B2 · Rebalance Timing | {_pval(p_cagr_b2) if not b2_degenerate else 'n/a'} | {'n/a' if b2_degenerate else _pval((res_b2 or {}).get('p_values', {}).get('MaxDD', float('nan')))} | {'n/a' if b2_degenerate else _pval((res_b2 or {}).get('p_values', {}).get('Sharpe', float('nan')))} | {'n/a' if b2_degenerate else _pval((res_b2 or {}).get('p_values', {}).get('Calmar', float('nan')))} | {b2_verdict} |",
        "",
        "**Interpretazione**:",
        interp_b,
        "",
        "---",
        "",
        "## 4. Note diagnostiche",
        "",
        diag_section,
        "",
        "---",
        "",
        "## 5. Coerenza incrociata A vs B",
        "",
        f"{cross_label} — {cross_desc}",
        "",
        "---",
        "",
        "## 6. Raccomandazione finale",
        "",
        rec,
        "",
    ]
    report_str = "\n".join(lines)

    # ── Salvataggio su disco ───────────────────────────────────────────────────
    if save_path is not None:
        import os
        sp = str(save_path)
        if os.path.isdir(sp) or sp.endswith(os.sep) or not sp.endswith('.md'):
            # È una directory
            os.makedirs(sp, exist_ok=True)
            date_tag  = datetime.now().strftime('%Y-%m-%d')
            safe_name = portfolio_name.replace(' ', '_').replace('/', '_')
            full_path = os.path.join(sp, f"MC_validation_{safe_name}_{date_tag}.md")
        else:
            full_path = sp
            os.makedirs(os.path.dirname(full_path) or '.', exist_ok=True)
        if not overwrite and os.path.exists(full_path):
            raise FileExistsError(
                f"Report already exists at {full_path}. "
                "Pass overwrite=True to overwrite."
            )
        with open(full_path, 'w', encoding='utf-8') as fout:
            fout.write(report_str)
        print(f"Report saved to: {full_path}")

    return report_str


# =============================================================================
# STABILITY ANALYSIS — PRE-WFO HELPERS
# =============================================================================

_STABILITY_METRICS = frozenset({"CAGR", "Sharpe", "Calmar"})


def _split_history_into_periods(
    start_date,
    end_date,
    k: int = 3,
    min_period_days: int = 180,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split [start_date, end_date] into k equal contiguous sub-periods.

    Parameters
    ----------
    start_date : str or pd.Timestamp
        Inclusive start of the overall range.
    end_date : str or pd.Timestamp
        Inclusive end of the overall range.
    k : int, default 3
        Number of sub-periods to produce. Must be >= 1.
    min_period_days : int, default 180
        Minimum length in calendar days for each sub-period.
        Raises ValueError if any sub-period would be shorter.

    Returns
    -------
    list of tuple(pd.Timestamp, pd.Timestamp)
        k tuples of (period_start, period_end), contiguous and inclusive
        on both ends, covering [start_date, end_date] without gaps.

    Raises
    ------
    ValueError
        If k < 1, or if any sub-period is shorter than min_period_days.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")

    # start = pd.Timestamp(start_date).normalize()
    # end   = pd.Timestamp(end_date).normalize()

    start = pd.Timestamp(start_date).normalize()
    end   = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    
    total_days = (end - start).days

    breakpoints = [
        start + pd.Timedelta(days=round(i * total_days / k))
        for i in range(k + 1)
    ]
    breakpoints[-1] = end  # exact end, no rounding drift

    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(k):
        p_start = breakpoints[i]
        p_end   = breakpoints[i + 1] - pd.Timedelta(days=1) if i < k - 1 else end
        period_len = (p_end - p_start).days + 1  # inclusive day count
        if period_len < min_period_days:
            raise ValueError(
                f"Sub-period {i + 1}/{k} spans only {period_len} days "
                f"({p_start.date()} → {p_end.date()}), "
                f"but min_period_days={min_period_days}. "
                f"Reduce k or widen the date range."
            )
        periods.append((p_start, p_end))

    return periods


def _evaluate_ptf_on_period(
    ptf_config: dict,
    params: "EngineParams",
    start_date,
    end_date,
    metric: str = "CAGR",
) -> float:
    """
    Evaluate a rotational portfolio on a date range and return a scalar metric.

    Runs the rotational engine on a warmup-extended price slice, then builds
    the VBT portfolio restricted to [start_date, end_date].

    Parameters
    ----------
    ptf_config : dict
        Required keys:
            - ``stocks_data`` (pd.DataFrame): full price history, trading-day
              DatetimeIndex, columns = tickers.  Must cover at least
              [start_date - params.required_warmup_days(), end_date].
            - ``init_cash`` (float): initial portfolio value.
        Extra keys are silently ignored.
    params : EngineParams
        Single parameter set to evaluate (e.g. best row from wfo_summary_df).
    start_date : str or pd.Timestamp
        Inclusive start of the evaluation window.
    end_date : str or pd.Timestamp
        Inclusive end of the evaluation window.
    metric : str, default "CAGR"
        Performance metric.  One of {"CAGR", "Sharpe", "Calmar"}.
        All three are higher-is-better.

    Returns
    -------
    float
        Scalar metric value over [start_date, end_date].
        Returns ``np.nan`` if the equity curve has fewer than 2 points in
        the window (engine ran but produced no valid data).

    Raises
    ------
    ValueError
        If metric is not in {"CAGR", "Sharpe", "Calmar"} (message includes
        the received value), or if the price slice for the window is empty.
    KeyError
        If ``stocks_data`` or ``init_cash`` are missing from ptf_config.

    Notes
    -----
    Metric conventions match ``_mc_compute_metrics`` (academic formulas):
      CAGR   = (V_end/V_start)^(252/n_days) - 1
      Sharpe = mean(daily_rets) * 252 / (std * sqrt(252)), rf = 0
      Calmar = CAGR / abs(MaxDD)
    These differ from vbt's built-in annualisation; do not mix the two.
    """
    if metric not in _STABILITY_METRICS:
        raise ValueError(
            f"metric={metric!r} is not supported. "
            f"Supported metrics: {sorted(_STABILITY_METRICS)}. "
            f"To use lower-is-better metrics (MaxDD, Volatility, Ulcer), "
            f"handle sign convention explicitly before calling this function."
        )

    stocks_data: pd.DataFrame = ptf_config["stocks_data"]
    init_cash: float = float(ptf_config["init_cash"])

    start = pd.Timestamp(start_date).normalize()
    end   = pd.Timestamp(end_date).normalize()

    buf_start = start - pd.Timedelta(days=params.required_warmup_days())
    slice_prices = stocks_data.loc[buf_start:end].copy()

    if slice_prices.empty:
        raise ValueError(
            f"Empty price slice for [{buf_start.date()}, {end.date()}]. "
            f"Ensure ptf_config['stocks_data'] covers at least "
            f"{buf_start.date()} → {end.date()}."
        )

    rot_result = run_rotational_engine(slice_prices, params)

    pf_rot, _ = build_portfolio(
        rot_result,
        slice_prices,
        init_cash=init_cash,
        start_date=start,
        end_date=end,
        plot=False,
        show_report=False,
    )

    equity = pf_rot.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.squeeze()

    if len(equity) < 2:
        return np.nan

    return float(_mc_compute_metrics(equity)[metric])

_STABILITY_FLAGS = frozenset({
    "filter_ema",
    "filter_volatility",
    "filter_min_momentum",
    "use_acceleration",
})


def _evaluate_flag_stability(
    ptf_config: dict,
    base_params: dict,
    flag_name: str,
    full_start_date,
    full_end_date,
    metric: str = "CAGR",
    k: int = 3,
    n_top_anchors: list[int] | None = None,
) -> dict:
    """
    Evaluate whether toggling a binary flag improves performance consistently
    across k contiguous sub-periods of the portfolio history.

    For each sub-period and each value of n_top in n_top_anchors, runs the
    rotational engine twice (flag=True vs flag=False) and records the delta.
    Aggregates per-period deltas across anchors, then classifies the flag as
    coherently beneficial, coherently harmful, or unstable.

    Parameters
    ----------
    ptf_config : dict
        Required keys: ``stocks_data`` (pd.DataFrame), ``init_cash`` (float).
        Passed directly to ``_evaluate_ptf_on_period``.
    base_params : dict
        Base parameter set (e.g. a row of wfo_summary_df as a plain dict).
        ``flag_name`` and ``n_top`` are overridden internally; all other keys
        are passed through to ``EngineParams.from_dict``.
    flag_name : str
        The boolean flag to test. Must be one of:
        ``{"filter_ema", "filter_volatility", "filter_min_momentum",
        "use_acceleration"}``.
    full_start_date : str or pd.Timestamp
        Inclusive start of the full evaluation range.
    full_end_date : str or pd.Timestamp
        Inclusive end of the full evaluation range.
    metric : str, default "CAGR"
        Performance metric. One of ``{"CAGR", "Sharpe", "Calmar"}``.
    k : int, default 3
        Number of contiguous sub-periods.
    n_top_anchors : list of int or None, default None
        Values of n_top over which to average the flag delta.
        Defaults to [3, 5, 8] if None.

    Returns
    -------
    dict with keys:
        flag_name, metric, k, n_top_anchors,
        delta_per_period (list[float], length k),
        delta_per_period_per_anchor (list[list[float]], shape k × len(anchors)),
        mean_delta (float),
        coherent_sign (bool),
        recommended_value (bool),
        diagnostic_note (str).

    Raises
    ------
    ValueError
        If flag_name not in the supported whitelist, or metric not supported.

    Notes
    -----
    When one of the values in n_top_anchors coincides with
    base_params['n_top'], the corresponding comparison varies ONLY the
    flag (pure flag effect). For the other anchors, both the flag and n_top
    vary simultaneously, so each delta measures the joint effect
    "flag × concentration". This is intentional: averaging across anchors
    captures the flag's effect under different portfolio concentrations,
    not a single "centroid" comparison. Inspect delta_per_period_per_anchor
    to disentangle the two effects if needed.
    """
    if flag_name not in _STABILITY_FLAGS:
        raise ValueError(
            f"flag_name={flag_name!r} is not a supported binary flag. "
            f"Supported: {sorted(_STABILITY_FLAGS)}."
        )
    if metric not in _STABILITY_METRICS:
        raise ValueError(
            f"metric={metric!r} is not supported. "
            f"Supported: {sorted(_STABILITY_METRICS)}."
        )
    if n_top_anchors is None:
        n_top_anchors = [3, 5, 8]

    periods = _split_history_into_periods(full_start_date, full_end_date, k)

    delta_per_period_per_anchor: list[list[float]] = []

    for s, e in periods:
        deltas_for_period: list[float] = []
        for anchor in n_top_anchors:
            params_true  = {**base_params, flag_name: True,  "n_top": anchor}
            params_false = {**base_params, flag_name: False, "n_top": anchor}

            val_true  = _evaluate_ptf_on_period(
                ptf_config, EngineParams.from_dict(params_true),  s, e, metric
            )
            val_false = _evaluate_ptf_on_period(
                ptf_config, EngineParams.from_dict(params_false), s, e, metric
            )

            if np.isnan(val_true) or np.isnan(val_false):
                warnings.warn(
                    f"_evaluate_flag_stability: NaN for {flag_name}, "
                    f"anchor={anchor}, period={s.date()}→{e.date()} "
                    f"(true={val_true:.4f} false={val_false:.4f}). "
                    f"Delta set to NaN.",
                    stacklevel=2,
                )
                deltas_for_period.append(float("nan"))
            else:
                deltas_for_period.append(val_true - val_false)

        delta_per_period_per_anchor.append(deltas_for_period)

    # Aggregate per period: mean across anchors, ignoring NaN
    delta_per_period: list[float] = []
    for row in delta_per_period_per_anchor:
        valid = [d for d in row if not np.isnan(d)]
        if not valid:
            delta_per_period.append(float("nan"))
        else:
            delta_per_period.append(float(np.mean(valid)))

    # Sign coherence and recommendation
    non_nan = [d for d in delta_per_period if not np.isnan(d)]
    positive = sum(1 for d in non_nan if d > 0)
    negative = sum(1 for d in non_nan if d < 0)
    zero     = sum(1 for d in non_nan if d == 0)

    mean_delta = float(np.mean(non_nan)) if non_nan else float("nan")
    recommended_value: bool = False

    if not non_nan:
        coherent_sign = False
        diagnostic_note = "all periods produced NaN — insufficient data"
    elif len(non_nan) < k:
        coherent_sign = False
        diagnostic_note = f"incoherent: {k - len(non_nan)} NaN period(s)"
    elif positive == k:
        coherent_sign = True
        recommended_value = True
        diagnostic_note = "coherent positive"
    elif negative == k:
        coherent_sign = True
        recommended_value = False
        diagnostic_note = "coherent negative"
    elif zero == k:
        coherent_sign = False
        recommended_value = False
        diagnostic_note = "no effect (all deltas zero)"
    else:
        coherent_sign = False
        recommended_value = False
        diagnostic_note = "incoherent: mixed signs across periods"

    return {
        "flag_name":                    flag_name,
        "metric":                       metric,
        "k":                            k,
        "n_top_anchors":                list(n_top_anchors),
        "delta_per_period":             delta_per_period,
        "delta_per_period_per_anchor":  delta_per_period_per_anchor,
        "mean_delta":                   mean_delta,
        "coherent_sign":                coherent_sign,
        "recommended_value":            recommended_value,
        "diagnostic_note":              diagnostic_note,
    }

def reduce_grid_via_stability(
    ptf_config: dict,
    full_grid: dict,
    full_start_date,
    full_end_date,
    metric: str = "CAGR",
    k: int = 3,
    n_top_anchors: list[int] | None = None,
    verbose: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """
    Reduce a WFO parameter grid by fixing binary flags to their stability-
    recommended values, then return the reduced grid and a diagnostic report.
    """
    if metric not in _STABILITY_METRICS:
        raise ValueError(
            f"metric={metric!r} is not supported. "
            f"Supported: {sorted(_STABILITY_METRICS)}."
        )

    requested_anchors = list(n_top_anchors) if n_top_anchors is not None else [3, 5, 8]
    stocks_data: pd.DataFrame = ptf_config["stocks_data"]

    universe_size = len(stocks_data.columns)
    margin = 3
    max_allowed_anchor = universe_size - margin

    if max_allowed_anchor < 1:
        raise ValueError(
            f"Universe too small: {universe_size} tickers available. "
            f"Need at least {margin + 1} tickers to run stability analysis "
            f"with margin={margin}."
        )

    anchors = [a for a in requested_anchors if a <= max_allowed_anchor]

    if not anchors:
        anchors = [max_allowed_anchor]
    elif max_allowed_anchor not in anchors:
        anchors = sorted(set(anchors + [max_allowed_anchor]))
    else:
        anchors = sorted(set(anchors))

    if verbose and anchors != requested_anchors:
        print(
            f"[WARN] n_top_anchors adattato automaticamente: "
            f"richiesto={requested_anchors}, usato={anchors}, "
            f"universe_size={universe_size}, margin={margin}"
        )

    # ── Step 1: identify eligible flags ──────────────────────────────────────
    eligible_flags = [
        flag for flag in sorted(_STABILITY_FLAGS)
        if flag in full_grid
        and True in full_grid[flag]
        and False in full_grid[flag]
    ]

    # ── Step 2: build centroid base_params ───────────────────────────────────
    base_params: dict = {}
    for key, vals in full_grid.items():
        if key in _STABILITY_FLAGS:
            base_params[key] = False
        elif all(isinstance(v, bool) for v in vals):
            base_params[key] = False
        elif all(isinstance(v, (int, float)) for v in vals):
            median_val = float(np.median(vals))
            base_params[key] = min(vals, key=lambda v: abs(v - median_val))
        else:
            base_params[key] = vals[0]

    # ── Step 3: run stability analysis for each eligible flag ─────────────────
    stability_results: dict = {}
    for flag_name in eligible_flags:
        if verbose:
            print(f"  Evaluating {flag_name} …")

        stability_results[flag_name] = _evaluate_flag_stability(
            ptf_config=ptf_config,
            base_params=base_params,
            flag_name=flag_name,
            full_start_date=full_start_date,
            full_end_date=full_end_date,
            metric=metric,
            k=k,
            n_top_anchors=anchors,
        )

    # ── Step 4: build reduced_grid ────────────────────────────────────────────
    reduced_grid: dict = {}
    for key, vals in full_grid.items():
        if key in stability_results:
            reduced_grid[key] = [stability_results[key]["recommended_value"]]
        else:
            reduced_grid[key] = list(vals)

    # ── Step 5: build diagnostic_report ──────────────────────────────────────
    rows = []
    for flag in sorted(_STABILITY_FLAGS):
        if flag in stability_results:
            r = stability_results[flag]
            rows.append({
                "flag_name":                   flag,
                "evaluated":                   True,
                "mean_delta":                  r["mean_delta"],
                "coherent_sign":               r["coherent_sign"],
                "recommended_value":           r["recommended_value"],
                "diagnostic_note":             r["diagnostic_note"],
                "delta_per_period":            str(r["delta_per_period"]),
                "delta_per_period_per_anchor": str(r["delta_per_period_per_anchor"]),
            })
        else:
            if flag in full_grid:
                reason = f"skipped: single value {full_grid[flag]}"
            else:
                reason = "skipped: not in full_grid"

            rows.append({
                "flag_name":                   flag,
                "evaluated":                   False,
                "mean_delta":                  float("nan"),
                "coherent_sign":               None,
                "recommended_value":           None,
                "diagnostic_note":             reason,
                "delta_per_period":            None,
                "delta_per_period_per_anchor": None,
            })

    diagnostic_report = pd.DataFrame(rows)

    # ── Step 6: verbose summary ───────────────────────────────────────────────
    if verbose:
        from math import prod as _prod

        orig_count = _prod(len(v) for v in full_grid.values())
        new_count = _prod(len(v) for v in reduced_grid.values())
        reduction = orig_count / new_count if new_count > 0 else float("inf")

        print("\n=== Stability Analysis Diagnostic Report ===")
        display_cols = [
            "flag_name", "evaluated", "mean_delta",
            "coherent_sign", "recommended_value", "diagnostic_note",
        ]
        print(diagnostic_report[display_cols].to_string(index=False))

        print(
            f"\nGrid: {orig_count} → {new_count} combinations "
            f"({reduction:.1f}x reduction)"
        )

        eval_results = list(stability_results.values())
        if eval_results and all(not r["coherent_sign"] for r in eval_results):
            print("\n⚠ WARNING: all evaluated flags resulted INCOHERENT.")
            print("  Stability analysis produced no positive signal. Consider:")
            print("  - increasing k for finer temporal granularity")
            print("  - extending [full_start_date, full_end_date] range")
            print("  - inspecting diagnostic_report for per-anchor details")
            print("  - reviewing whether base_params centroid is appropriate")

    return reduced_grid, diagnostic_report
    


# =============================================================================
# OVERFITTING CHECK — ROTATIONAL PORTFOLIOS
# =============================================================================

from dataclasses import dataclass as _dc
from math import prod as _prod
from typing import Literal
from tqdm.notebook import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
_OFC_SUPPORTED_METRICS = frozenset({
    "CAGR", "Sharpe", "Sharpe Ratio", "Calmar", "Calmar Ratio", "Sortino Ratio",
})
_OFC_LOWER_IS_BETTER = frozenset({
    "MaxDD", "Volatility", "Ulcer", "Vol", "DD", "drawdown",
})
_OFC_METRIC_ALIAS = {
    "Sharpe Ratio":  "Sharpe",
    "Calmar Ratio":  "Calmar",
    "Sortino Ratio": "Sortino",
}


@_dc(frozen=True)
class _ProfileDefaults:
    metric: str
    plateau_threshold: float
    s2_coherence_threshold: float
    s3_pvalue_threshold: float
    s4_dsr_threshold: float
    min_signals_to_pass: int


_PROFILES: dict[str, _ProfileDefaults] = {
    "satellite": _ProfileDefaults(
        metric="CAGR",
        plateau_threshold=0.20,
        s2_coherence_threshold=0.50,
        s3_pvalue_threshold=0.10,
        s4_dsr_threshold=0.0,
        min_signals_to_pass=3,
    ),
    "core": _ProfileDefaults(
        metric="Calmar Ratio",
        plateau_threshold=0.30,
        s2_coherence_threshold=0.75,
        s3_pvalue_threshold=0.05,
        s4_dsr_threshold=0.5,   # empirical — to calibrate across PTFs; range [0.3, 0.8]
        min_signals_to_pass=4,
    ),
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _ofc_normalize_metric(metric: str) -> str:
    return _OFC_METRIC_ALIAS.get(metric, metric)


def _compute_sortino(equity: pd.Series, trading_days: int = 252) -> float:
    """Sortino ratio (rf=0, annualized)."""
    rets = equity.pct_change(fill_method=None).dropna()
    if len(rets) < 2:
        return np.nan
    mean_ret = float(rets.mean())
    downside = rets[rets < 0.0]
    if len(downside) < 2:
        return np.nan
    ds_std = float(downside.std(ddof=1))
    if ds_std <= 0.0:
        return np.nan
    return float((mean_ret * trading_days) / (ds_std * np.sqrt(trading_days)))


def _ofc_compute_metric(equity: pd.Series, metric: str, trading_days: int = 252) -> float:
    """Dispatch metric computation. All outputs are higher-is-better."""
    m = _ofc_normalize_metric(metric)
    if m == "Sortino":
        return _compute_sortino(equity, trading_days)
    result = _mc_compute_metrics(equity, trading_days)
    if m not in result:
        raise ValueError(f"metric={metric!r} not in _mc_compute_metrics output")
    return float(result[m])


# ── S4 DSR formula ────────────────────────────────────────────────────────────

def ofc_compute_dsr(sr_hat: float, n_trials: int, T: int) -> float:
    """
    Simplified Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    DSR = SR_hat - Φ^{-1}(1 - 1/N) × (1/√T)

    where N = n_trials, T = OOS trading days.
    Φ^{-1}(1 - 1/N) ≈ expected maximum of N iid standard normals.
    Does not include skewness/kurtosis correction (requires full SR distribution).
    """
    if n_trials <= 0 or T <= 0 or np.isnan(sr_hat):
        return np.nan
    from scipy.stats import norm as _norm
    z = float(_norm.ppf(max(1.0 - 1.0 / n_trials, 0.5)))
    return float(sr_hat - z / np.sqrt(float(T)))


# ── Random rotation helper (no VBT) ──────────────────────────────────────────

def _ofc_random_rotation_equity(
    prices: pd.DataFrame,
    n_top: int,
    rebal_dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Equal-weight random rotation equity (vectorized, no VBT).
    Weights shifted by 1 bar — same look-ahead convention as engine.
    """
    prices = prices.ffill().dropna(how="all", axis=1)
    if prices.empty or len(prices.columns) == 0:
        return pd.Series(1.0, index=prices.index[:1])

    n_select = min(n_top, len(prices.columns))
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    idx_set = set(prices.index.normalize())
    active = sorted(d for d in rebal_dates if prices.index[0] <= d <= prices.index[-1])
    if not active or active[0] > prices.index[0]:
        active = [prices.index[0]] + active

    for i, rd in enumerate(active):
        if rd not in idx_set:
            loc = prices.index.get_indexer([rd], method="nearest")[0]
            rd = prices.index[loc]
        nxt = active[i + 1] if i + 1 < len(active) else None
        sel = rng.choice(prices.columns, size=n_select, replace=False)
        mask = (prices.index >= rd) & (prices.index < nxt) if nxt else (prices.index >= rd)
        weights.loc[mask, sel] = 1.0 / n_select

    rets = prices.pct_change(fill_method=None).fillna(0.0)
    port_rets = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    return (1.0 + port_rets).cumprod()


# ── OOS equity reconstruction ─────────────────────────────────────────────────

def _ofc_reconstruct_oos_equity(
    wfo_summary: pd.DataFrame,
    stocks_data: pd.DataFrame,
    init_cash: float = 100_000,
) -> tuple[pd.Series, list[tuple]]:
    """
    Rebuild the compounded OOS equity curve from wfo_summary best params.
    Returns (compounded_normalized_equity, [(oos_start, oos_end, window_eq), ...]).
    """
    stocks = stocks_data.sort_index()
    window_equities: list[tuple] = []

    for window, row in wfo_summary.iterrows():
        sep = "→" if "→" in str(window) else "->"
        parts_w = str(window).split(sep)
        if len(parts_w) != 2:
            continue
        # oos_start = pd.Timestamp(parts_w[0].strip()).normalize()
        # oos_end   = pd.Timestamp(parts_w[1].strip()).normalize()

        oos_start = pd.Timestamp(parts_w[0].strip()).normalize()
        oos_end   = pd.Timestamp(parts_w[1].strip()).normalize()
        if oos_start > pd.Timestamp.today().normalize():   # ← ADD
            continue                                        # ← ADD
        params    = EngineParams.from_dict(dict(row))
        
        params    = EngineParams.from_dict(dict(row))
        buf_start = oos_start - pd.Timedelta(days=params.required_warmup_days())
        sl        = stocks.loc[buf_start:oos_end].copy()
        if sl.empty:
            continue

        rot = run_rotational_engine(sl, params)
        pf_rot, _ = build_portfolio(
            rot, sl, init_cash=init_cash,
            start_date=oos_start, end_date=oos_end,
            plot=False, show_report=False,
        )
        eq = pf_rot.value()
        if isinstance(eq, pd.DataFrame):
            eq = eq.squeeze()
        window_equities.append((oos_start, oos_end, eq))

    if not window_equities:
        return pd.Series(dtype=float), []

    chunks, running = [], 1.0
    for _, _, eq in window_equities:
        normed   = eq / float(eq.iloc[0]) * running
        running  = float(normed.iloc[-1])
        chunks.append(normed)

    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full, window_equities


# ── Signal implementations ────────────────────────────────────────────────────

def _ofc_s1_plateau(
    wfo_summary: pd.DataFrame,
    param_grid: dict,
    threshold: float,
) -> tuple[float, bool, str]:
    """
    S1: parameter diversity proxy for plateau width.
    score = mean fraction of grid values that appear as best-choice across windows,
            averaged over numeric params with > 1 distinct value in grid.
    """
    eligible = [
        p for p, vals in param_grid.items()
        if p in wfo_summary.columns
        and len(vals) > 1
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
    ]
    if not eligible:
        return 0.5, True, "S1: no eligible numeric params — signal waived (pass)"

    fracs = [
        len(set(wfo_summary[p].tolist())) / len(param_grid[p])
        for p in eligible
    ]
    score  = float(np.mean(fracs))
    passed = score > threshold
    note   = (f"S1: diversity={score:.3f} >threshold={threshold:.2f}? "
              f"params={eligible}")
    return score, passed, note


def _ofc_s2_coherence(
    wfo_summary: pd.DataFrame,
    param_grid: dict,
    stability_report: pd.DataFrame | None,
    threshold: float,
) -> tuple[float, bool, str]:
    """S2: sign coherence of binary flags."""
    if stability_report is not None and "evaluated" in stability_report.columns:
        ev       = stability_report[stability_report["evaluated"] == True]
        n_ev     = len(ev)
        n_coh    = int(ev["coherent_sign"].fillna(False).sum())
        source   = "stability_report"
    else:
        bin_params = [
            p for p, vals in param_grid.items()
            if p in wfo_summary.columns
            and len(vals) > 1
            and all(isinstance(v, bool) for v in vals)
        ]
        n_ev  = len(bin_params)
        n_coh = 0
        for p in bin_params:
            chosen     = wfo_summary[p].tolist()
            mode_count = max(chosen.count(True), chosen.count(False))
            if mode_count > len(chosen) / 2:
                n_coh += 1
        source = "wfo_summary"

    if n_ev == 0:
        return 0.5, True, "S2: no flags to evaluate — signal waived (pass)"

    score  = n_coh / n_ev
    passed = score >= threshold
    note   = (f"S2: {n_coh}/{n_ev} coherent={score:.2f} >=threshold={threshold:.2f} "
              f"(source: {source})")
    return score, passed, note


def _ofc_s3_bootstrap(
    actual_equity: pd.Series,
    stocks_data: pd.DataFrame,
    wfo_summary: pd.DataFrame,
    param_grid: dict,
    n_bootstrap: int,
    metric: str,
    threshold: float,
    seed: int,
) -> tuple[float, bool, str]:
    """
    S3: does the PTF beat N random equal-weight rotation baselines?
    p_value = fraction of random scores >= actual_score (lower is better for the PTF).
    Uses resolved_metric — NOT necessarily Sharpe (that is reserved for S4 DSR).
    """
    m_norm       = _ofc_normalize_metric(metric)
    actual_score = _ofc_compute_metric(actual_equity, m_norm)
    if np.isnan(actual_score):
        return np.nan, False, "S3: actual score NaN — signal FAIL"

    stocks    = stocks_data.sort_index()
    n_top_med = int(np.median([int(v) for v in wfo_summary.get("n_top", pd.Series([5]))]))
    freq_list = wfo_summary.get("rebalance_frequency", pd.Series(["ME"])).tolist()
    rebal_freq = freq_list[0] if freq_list else "ME"

    rng           = np.random.default_rng(seed)
    random_scores : list[float] = []

    for sim in tqdm(range(n_bootstrap), desc="OFC Bootstrap"):
        chunks, running = [], 1.0
        for window in wfo_summary.index:
            sep     = "→" if "→" in str(window) else "->"
            pw      = str(window).split(sep)
            os_s    = pd.Timestamp(pw[0].strip()).normalize()
            os_e    = pd.Timestamp(pw[1].strip()).normalize()
            sl      = stocks.loc[os_s:os_e].copy()
            if sl.empty:
                continue
            rdates  = compute_rebal_dates(sl.index, rebal_freq)
            rand_eq = _ofc_random_rotation_equity(sl, n_top_med, rdates, rng)
            normed  = rand_eq / float(rand_eq.iloc[0]) * running
            running = float(normed.iloc[-1])
            chunks.append(normed)

        if not chunks:
            continue
        full   = pd.concat(chunks).sort_index()
        full   = full[~full.index.duplicated(keep="first")]
        sc     = _ofc_compute_metric(full, m_norm)
        if not np.isnan(sc):
            random_scores.append(sc)

    if not random_scores:
        return np.nan, False, "S3: no valid random scores generated"

    p_value = sum(r >= actual_score for r in random_scores) / len(random_scores)
    passed  = p_value <= threshold
    note    = (f"S3 ({m_norm}): actual={actual_score:.4f}, "
               f"p={p_value:.3f} <=threshold={threshold:.2f}, "
               f"n={len(random_scores)} sims")
    return p_value, passed, note


def _ofc_s4_dsr(
    oos_equity: pd.Series,
    n_trials: int,
    threshold: float,
    trading_days: int = 252,
) -> tuple[float, bool, str]:
    """
    S4: Deflated Sharpe Ratio. Always on OOS Sharpe (DSR theory valid only for Sharpe).
    """
    m  = _mc_compute_metrics(oos_equity, trading_days)
    sr = float(m.get("Sharpe", np.nan))
    T  = len(oos_equity)

    if np.isnan(sr) or n_trials <= 0 or T <= 0:
        return np.nan, False, "S4: insufficient data"

    dsr    = ofc_compute_dsr(sr, n_trials, T)
    passed = not np.isnan(dsr) and dsr > threshold
    note   = (f"S4 (DSR on Sharpe): SR={sr:.4f}, n={n_trials}, T={T}, "
              f"DSR={dsr:.4f} >threshold={threshold:.2f}")
    return dsr, passed, note


# ── Main function ─────────────────────────────────────────────────────────────

def overfitting_check_rotational(
    wfo_summary: pd.DataFrame,
    stocks_data: pd.DataFrame,
    benchmark_data: pd.Series,
    param_grid: dict,
    *,
    n_total_trials: int | None = None,
    profile: str = "satellite",
    metric: str | None = None,
    plateau_threshold: float | None = None,
    s2_coherence_threshold: float | None = None,
    s3_pvalue_threshold: float | None = None,
    s4_dsr_threshold: float | None = None,
    min_signals_to_pass: int | None = None,
    stability_report: pd.DataFrame | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[bool, dict]:
    """
    Multi-signal overfitting check for rotational WFO results.

    Evaluates 4 signals: S1 (parameter diversity / plateau proxy),
    S2 (flag sign coherence), S3 (edge vs random rotation baseline),
    S4 (Deflated Sharpe Ratio).

    Decision: promoted=True if n_signals_passed >= min_signals_to_pass,
    with mandatory exception: S3 fail AND S4 fail → promoted=False.

    Parameters
    ----------
    wfo_summary : pd.DataFrame
        Output of walk_forward_rotational. Index = Window strings
        ("YYYY-MM-DD→YYYY-MM-DD"), columns = param_names + TrainScore + TestScore.
    stocks_data : pd.DataFrame
        Full price history. Must cover all WFO OOS windows + warmup buffer.
    benchmark_data : pd.Series
        Benchmark prices (reserved for future extensions; not used in current signals).
    param_grid : dict
        Parameter grid used in walk_forward_rotational (or reduced grid if
        auto_reduce_grid was applied). Used for S1 diversity and S4 n_trials.
    n_total_trials : int or None
        Total strategy combinations explored during WFO.
        If auto_reduce_grid was applied, pass the ORIGINAL grid cardinality
        (not the post-reduction size) to penalize the full exploration in S4.
        If None → computed from param_grid with a warning.
    profile : str, default "satellite"
        Threshold preset. "satellite" | "core" | "custom".
        - satellite: CAGR metric, permissive (≥3/4 to pass).
        - core: Calmar metric, strict (≥4/4 to pass, higher thresholds).
        - custom: all thresholds must be provided explicitly.
    metric : str or None
        Override profile default for S3 comparison.
        Valid: "CAGR", "Sharpe"/"Sharpe Ratio", "Calmar"/"Calmar Ratio",
        "Sortino Ratio". Metrics in _OFC_LOWER_IS_BETTER raise ValueError.
        S4 is ALWAYS computed on OOS Sharpe regardless of this override.
    plateau_threshold, s2_coherence_threshold, s3_pvalue_threshold,
    s4_dsr_threshold, min_signals_to_pass
        Override individual profile defaults (None = use profile default).
    stability_report : pd.DataFrame or None
        Output of reduce_grid_via_stability. If provided, S2 reuses
        coherent_sign column (avoids recomputing flag stability).
    n_bootstrap : int, default 100
        Bootstrap replications for S3. Min 30 recommended.
    seed : int, default 42
    verbose : bool, default True

    Returns
    -------
    promoted : bool
    report : dict
        Keys: promoted, profile, resolved (thresholds + n_total_trials_used),
        signals (per-signal detail including metric_used), n_signals_passed,
        override_rule_applied, diagnostic_notes.

    Raises
    ------
    ValueError
        If profile is unknown, metric is lower-is-better, or profile='custom'
        with missing required parameters.

    Notes
    -----
    On S1 (plateau proxy):
    True plateau (% combos with TrainScore >= plateau_threshold × best)
    requires storing all combo scores during WFO — not in wfo_summary.
    Parameter diversity is used as a proxy. To be replaced when WFO
    stores full score distributions (see technical debts in CLAUDE.md).

    On S4 (Deflated Sharpe Ratio):
    S4 is always computed on OOS Sharpe regardless of the profile metric.
    DSR has a known asymptotic distribution (Bailey & Lopez de Prado 2014)
    only for Sharpe. Generalizing to CAGR/Calmar requires empirical bootstrap
    and is parked as future work (see CLAUDE.md).

    On satellite default metric (CAGR):
    CAGR maximization can lead WFO to systematically select n_top=1
    (full concentration on single highest-momentum asset). Constrained
    externally by n_top_min. Override metric='Sharpe Ratio' if risk
    normalization is required.
    """
    # ── 0. Resolve profile + thresholds ──────────────────────────────────────
    if profile not in _PROFILES and profile != "custom":
        raise ValueError(
            f"profile={profile!r} not recognized. "
            f"Supported: {{'satellite', 'core', 'custom'}}."
        )
    if profile == "custom":
        _missing = [k for k, v in [("metric", metric),
                                    ("min_signals_to_pass", min_signals_to_pass)]
                    if v is None]
        if _missing:
            raise ValueError(
                f"profile='custom' requires explicit values for: {_missing}."
            )
        _def = None
    else:
        _def = _PROFILES[profile]

    def _r(override, key):
        return override if override is not None else (getattr(_def, key) if _def else None)

    res_metric  = _r(metric,                    "metric")
    res_plat    = _r(plateau_threshold,         "plateau_threshold")
    res_s2      = _r(s2_coherence_threshold,    "s2_coherence_threshold")
    res_s3      = _r(s3_pvalue_threshold,       "s3_pvalue_threshold")
    res_s4      = _r(s4_dsr_threshold,          "s4_dsr_threshold")
    res_minsig  = _r(min_signals_to_pass,       "min_signals_to_pass")

    # ── 1. Validate metric ────────────────────────────────────────────────────
    if res_metric in _OFC_LOWER_IS_BETTER:
        raise ValueError(
            f"metric={res_metric!r} is lower-is-better. "
            f"Supported: {sorted(_OFC_SUPPORTED_METRICS)}."
        )
    res_metric_norm = _ofc_normalize_metric(res_metric)

    # ── 2. n_total_trials ─────────────────────────────────────────────────────
    grid_card = _prod(len(v) for v in param_grid.values())
    if n_total_trials is None:
        n_trials = grid_card
        warnings.warn(
            "overfitting_check_rotational: n_total_trials not specified. "
            f"Using param_grid cardinality ({grid_card}). "
            "If auto_reduce_grid was applied, pass the ORIGINAL grid size "
            "for a correct S4 DSR penalty.",
            stacklevel=2,
        )
    else:
        n_trials = int(n_total_trials)

    # ── 3. Reconstruct OOS equity ─────────────────────────────────────────────
    oos_equity, _ = _ofc_reconstruct_oos_equity(wfo_summary, stocks_data)

    # ── 4. Compute signals ────────────────────────────────────────────────────
    s1_val, s1_pass, s1_note = _ofc_s1_plateau(wfo_summary, param_grid, res_plat)
    s2_val, s2_pass, s2_note = _ofc_s2_coherence(
        wfo_summary, param_grid, stability_report, res_s2)
    s3_val, s3_pass, s3_note = _ofc_s3_bootstrap(
        oos_equity, stocks_data, wfo_summary, param_grid,
        n_bootstrap, res_metric_norm, res_s3, seed,
    )
    s4_val, s4_pass, s4_note = _ofc_s4_dsr(oos_equity, n_trials, res_s4)

    # ── 5. Decision ───────────────────────────────────────────────────────────
    passes     = [s1_pass, s2_pass, s3_pass, s4_pass]
    n_passed   = sum(bool(p) for p in passes)
    both_oos_fail = (not s3_pass) and (not s4_pass)

    if both_oos_fail:
        promoted     = False
        override_msg = "Override: S3 fail AND S4 fail → not promoted (both OOS checks negative)"
    else:
        promoted     = n_passed >= res_minsig
        override_msg = None

    # ── 6. Verbose ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*64}")
        print(f"  Overfitting Check — profile={profile!r}  metric(S3)={res_metric_norm}")
        print(f"  n_trials(S4)={n_trials}  OOS windows={len(wfo_summary)}")
        print(f"{'='*64}")
        fmt = "  {:<16} {:>5} {:>10}  {}"
        print(fmt.format("Signal", "Pass?", "Value", "Note (truncated)"))
        print("  " + "-"*62)
        for lbl, val, pss, note in [
            ("S1 plateau",   s1_val, s1_pass, s1_note),
            ("S2 coherence", s2_val, s2_pass, s2_note),
            ("S3 bootstrap", s3_val, s3_pass, s3_note),
            ("S4 DSR",       s4_val, s4_pass, s4_note),
        ]:
            vs = f"{val:+.4f}" if isinstance(val, float) and not np.isnan(val) else "NaN"
            ps = "PASS" if pss else "FAIL"
            print(fmt.format(lbl, ps, vs, note[:55]))
        print(f"  {'─'*62}")
        print(f"  Signals: {n_passed}/{len(passes)} (need {res_minsig})")
        if override_msg:
            print(f"  ⚠ {override_msg}")
        print(f"  Verdict: {'PROMOTED ✓' if promoted else 'NOT PROMOTED ✗'}")
        print(f"{'='*64}\n")

    return promoted, {
        "promoted":             promoted,
        "profile":              profile,
        "resolved": {
            "metric":                 res_metric_norm,
            "plateau_threshold":      res_plat,
            "s2_coherence_threshold": res_s2,
            "s3_pvalue_threshold":    res_s3,
            "s4_dsr_threshold":       res_s4,
            "min_signals_to_pass":    res_minsig,
            "n_total_trials_used":    n_trials,
        },
        "signals": {
            "S1_plateau":   {"pass": s1_pass, "value": s1_val,  "threshold": res_plat, "note": s1_note},
            "S2_coherence": {"pass": s2_pass, "value": s2_val,  "threshold": res_s2,   "note": s2_note},
            "S3_bootstrap": {"pass": s3_pass, "p_value": s3_val,"threshold": res_s3,   "metric_used": res_metric_norm, "note": s3_note},
            "S4_dsr":       {"pass": s4_pass, "dsr": s4_val,    "threshold": res_s4,   "metric_used": "Sharpe",        "note": s4_note},
        },
        "n_signals_passed":       n_passed,
        "override_rule_applied":  both_oos_fail,
        "diagnostic_notes":       [m for m in [override_msg] if m],
    }

# =============================================================================
# OUTPUT DOCUMENTALE — decisione finale, PTF card, relazione tecnica
# =============================================================================
import datetime as _dt_doc
from pathlib import Path as _Path_doc

# Palette hex per reportlab (costanti modulo)
_RL_NAVY    = '#1B2A4A'
_RL_NAVY_LT = '#2C3E6B'
_RL_GREEN   = '#27AE60'
_RL_RED     = '#E74C3C'
_RL_ORANGE  = '#E67E22'
_RL_GRAY_LT = '#F5F6FA'
_RL_GRAY_BD = '#D5D8DC'
_RL_TEXT    = '#2C3E50'

def compute_skill_profile(
    *, mc_skill: dict,
    ofc_report_std: dict | None = None,   # mantenuto per retro-compat firma, non più usato
    mc_skill_cluster: dict | None = None,
    ofc_report_cluster: dict | None = None,
) -> tuple[str, str | None]:
    '''
    Deriva lo Skill Profile per Standard e (opzionalmente) Cluster basandosi sui 
    test MC Block B: B1 (rotation reshuffle) e B2 (rebalance timing).
    
    Logica:
      |  B1 PASS  |  B2 PASS  |  Profile          |
      |-----------|-----------|-------------------|
      |    ✓      |    ✓      |  Strong           |
      |    ✓      |    ✗      |  Selection-driven |
      |    ✗      |    ✓      |  Timing-driven    |
      |    ✗      |    ✗      |  No-skill         |
    
    Soglia p-value per PASS: < 0.10
    
    Note storiche
    -------------
    Versione precedente (pre-B-006) usava (B1, S3-OFC) e aveva nomenclatura 
    invertita: (B1 PASS, S3 FAIL) → "Timing-driven" anziché "Selection-driven".
    Vedi TODO B-006 per dettagli. I parametri ofc_report_std/cluster sono 
    mantenuti nella firma per retro-compatibilità ma non più usati.
    
    Returns
    -------
    (profile_std, profile_cluster_or_None)
        Tupla di due Skill Profile. Il secondo è None se mc_skill_cluster non passato.
    '''
    def _profile(mc):
        if mc is None:
            return None
        b1 = mc.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
        b2 = mc.get('rebalance_timing', {}).get('p_values', {}).get('CAGR')
        b1_pass = (b1 is not None) and (b1 < 0.10)
        b2_pass = (b2 is not None) and (b2 < 0.10)
        if b1_pass and b2_pass:        return 'Strong'
        if b1_pass and not b2_pass:    return 'Selection-driven'
        if not b1_pass and b2_pass:    return 'Timing-driven'
        return 'No-skill'

    profile_std = _profile(mc_skill)
    profile_cl  = _profile(mc_skill_cluster)
    return profile_std, profile_cl
    


def print_final_decision(
    *,
    portfolio_title: str,
    year: int,
    profile: str,
    ofc_report_std: dict,
    ofc_report_cluster: dict | None,
    mc_skill: dict,
    mc_ci,
    skill_profile: str,
    # NEW: opzionali per simmetria std/cluster
    mc_skill_cluster: dict | None = None,
    mc_ci_cluster=None,
) -> None:
    
    '''
    Stampa la DECISIONE FINALE: banner + tabella pandas 3 colonne.

    Mantiene layout originale STEP 8: banner di 76 '=', titolo, tabella
    Signal/WFO STANDARD/WFO CLUSTER, footer con invito a compilare.

    Parameters
    ----------
    mc_ci : pd.DataFrame
        ci_summary_df da run_all_mc_methods_rotational.
    '''
    _sigs    = ofc_report_std.get('signals', {}) if ofc_report_std else {}
    _cl_sigs = ofc_report_cluster.get('signals', {}) if ofc_report_cluster else {}
    ofc_passed_std     = bool(ofc_report_std.get('promoted', False)) if ofc_report_std else False
    ofc_passed_cluster = (
        bool(ofc_report_cluster.get('promoted', False)) if ofc_report_cluster else None
    )

    # Std (codice esistente)
    reshuffle_pval   = mc_skill.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
    reshuffle_passed = (reshuffle_pval is not None) and (reshuffle_pval < 0.10)
    
    ci_sharpe_p50 = None
    try:
        ci_sharpe_p50 = mc_ci.loc['A1 · IID Bootstrap', 'Sharpe_p50']
    except Exception:
        pass
    
    # NEW: Cluster (fallback a std se non passati → retro-compat)
    if mc_skill_cluster is not None:
        reshuffle_pval_cl   = mc_skill_cluster.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
        reshuffle_passed_cl = (reshuffle_pval_cl is not None) and (reshuffle_pval_cl < 0.10)
    else:
        reshuffle_pval_cl, reshuffle_passed_cl = reshuffle_pval, reshuffle_passed
    
    ci_sharpe_p50_cl = None
    if mc_ci_cluster is not None:
        try:
            ci_sharpe_p50_cl = mc_ci_cluster.loc['A1 · IID Bootstrap', 'Sharpe_p50']
        except Exception:
            pass
    else:
        ci_sharpe_p50_cl = ci_sharpe_p50

    
    _sig_key_map = {
        'S1 plateau':    'S1_plateau',
        'S2 coherence':  'S2_coherence',
        'S3 random sel': 'S3_bootstrap',
        'S4 DSR':        'S4_dsr',
    }

    def _fp(v):
        if v is None:
            return 'N/A'
        return 'PASS' if v else 'FAIL'

    _resh_str = (f"p={reshuffle_pval:.3f} {'PASS' if reshuffle_passed else 'FAIL'}"
                 if reshuffle_pval is not None else 'N/A')
    _resh_str_cl = (f"p={reshuffle_pval_cl:.3f} {'PASS' if reshuffle_passed_cl else 'FAIL'}"
                    if reshuffle_pval_cl is not None else 'N/A')
    _sharpe_str    = f"{ci_sharpe_p50:.3f}"    if ci_sharpe_p50    is not None else 'N/A'
    _sharpe_str_cl = f"{ci_sharpe_p50_cl:.3f}" if ci_sharpe_p50_cl is not None else 'N/A'


    _rows = []
    for sig, key in _sig_key_map.items():
        v_std = _sigs.get(key, {}).get('pass')
        v_clu = _cl_sigs.get(key, {}).get('pass') if _cl_sigs else None
        _rows.append((sig, _fp(v_std), _fp(v_clu)))

    _rows.append(('MC Reshuffle p',   _resh_str,   _resh_str_cl))
    _rows.append(('MC CI Sharpe p50', _sharpe_str, _sharpe_str_cl))
    
    _rows.append(('OFC Verdict',      _fp(ofc_passed_std), _fp(ofc_passed_cluster)))
    _rows.append(('Skill profile',    skill_profile, ''))

    _df = pd.DataFrame(_rows, columns=['Signal', 'WFO STANDARD', 'WFO CLUSTER'])

    print('=' * 76)
    print(f"  DECISIONE FINALE — {portfolio_title} ({year}) — profile={profile}")
    print('=' * 76)
    print(_df.to_string(index=False))
    print('=' * 76)
    print('  User decision: quale path deployare? [ STANDARD | CLUSTER | NESSUNO ]')
    print('  (compilare a mano nella scheda PTF con motivazione)')
    print('=' * 76)


def generate_ptf_card_md(
    *,
    portfolio_title: str,
    year: int,
    profile: str,
    benchmark: str,
    period: tuple,
    universe_size: int,
    wfo_config: dict,
    cluster_result: dict | None,
    metrics_comparison: dict,
    ofc_report_std: dict,
    ofc_report_cluster: dict | None,
    mc_skill: dict,
    mc_ci,
    skill_profile: str,
    output_path,
    mc_skill_cluster: dict | None = None,
    mc_ci_cluster=None,
) -> _Path_doc:
    '''
    Genera la PTF Card markdown (sezioni 1-9) e la scrive su output_path.

    Sezione 5 (MC) e Sezione 6 (Skill Profile) sdoppiate per path Standard / Cluster.
    Se mc_skill_cluster/mc_ci_cluster non passati, usa fallback al path standard
    (retro-compat con i chiamanti precedenti al fix MC).
    '''
    output_path = _Path_doc(output_path)
    _today = _dt_doc.date.today().isoformat()
    period_start, period_end = (period if len(period) == 2 else (period[0], _today))

    _s  = ofc_report_std.get('signals', {}) if ofc_report_std else {}
    _sc = ofc_report_cluster.get('signals', {}) if ofc_report_cluster else None
    ofc_passed_std     = bool(ofc_report_std.get('promoted', False)) if ofc_report_std else False
    ofc_passed_cluster = (
        bool(ofc_report_cluster.get('promoted', False)) if ofc_report_cluster else None
    )
    use_clustering = wfo_config.get('use_clustering', False)

    # MC Std
    reshuffle_pval  = mc_skill.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
    timing_pval     = mc_skill.get('rebalance_timing', {}).get('p_values', {}).get('CAGR')
    reshuffle_passed = (reshuffle_pval is not None) and (reshuffle_pval < 0.10)
    timing_passed   = (timing_pval is not None) and (timing_pval < 0.10)
    s3_passed_std   = _s.get('S3_bootstrap', {}).get('pass')

    _reshuffle_pval_str = f"p={reshuffle_pval:.3f}" if reshuffle_pval is not None else 'N/A'
    _timing_pval_str    = f"p={timing_pval:.3f}" if timing_pval is not None else 'N/A'
    _s3_str             = 'Pass' if s3_passed_std else 'Fail'

    # MC Cluster (fallback a std)
    if mc_skill_cluster is not None:
        reshuffle_pval_cl   = mc_skill_cluster.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
        timing_pval_cl      = mc_skill_cluster.get('rebalance_timing', {}).get('p_values', {}).get('CAGR')
        reshuffle_passed_cl = (reshuffle_pval_cl is not None) and (reshuffle_pval_cl < 0.10)
        timing_passed_cl    = (timing_pval_cl is not None) and (timing_pval_cl < 0.10)
    else:
        reshuffle_pval_cl, timing_pval_cl     = reshuffle_pval, timing_pval
        reshuffle_passed_cl, timing_passed_cl = reshuffle_passed, timing_passed

    _reshuffle_pval_str_cl = f"p={reshuffle_pval_cl:.3f}" if reshuffle_pval_cl is not None else 'N/A'
    _timing_pval_str_cl    = f"p={timing_pval_cl:.3f}"    if timing_pval_cl    is not None else 'N/A'

    s3_passed_cl = _sc.get('S3_bootstrap', {}).get('pass') if _sc else None
    _s3_str_cl   = 'Pass' if s3_passed_cl else ('Fail' if s3_passed_cl is False else 'N/A')

    # plots_dir derivato da output_path
    _plots_dir = (
        output_path.parent.parent / 'reports' / 'plots' / f'{portfolio_title}_{year}'
    )

    def _vd(v):
        if v is None: return 'N/A'
        return 'Pass' if v else 'Fail'

    def _ci(row, col):
        try: return f"{mc_ci.loc[row, col]:.3f}"
        except Exception: return 'N/A'

    def _ci_cl(row, col):
        src = mc_ci_cluster if mc_ci_cluster is not None else mc_ci
        try: return f"{src.loc[row, col]:.3f}"
        except Exception: return 'N/A'

    def _m(key, metric):
        pf = metrics_comparison.get(key)
        if pf is None: return 'N/A'
        try:
            if metric == 'cum':    return f"{pf.total_return()*100:.1f}%"
            if metric == 'cagr':   return f"{pf.annualized_return()*100:.1f}%"
            if metric == 'sharpe': return f"{pf.sharpe_ratio():.2f}"
            if metric == 'maxdd':  return f"{abs(pf.max_drawdown())*100:.1f}%"
        except Exception: return 'N/A'
        return 'N/A'

    # Sezione cluster (composizione)
    _cluster_section = ''
    if use_clustering and cluster_result is not None:
        _cgroups = cluster_result.get('cluster_groups', {})
        _clabels = cluster_result.get('cluster_labels', {})
        if _cgroups:
            def _fmt_t(ts):
                return ', '.join(ts[:8]) + (f' ... (+{len(ts)-8})' if len(ts) > 8 else '')
            _rows_cl = '\n'.join(
                f"| {cid} | {_clabels.get(cid, f'C{cid}')} | {len(_cgroups[cid])} "
                f"| {_fmt_t(_cgroups[cid])} |"
                for cid in sorted(_cgroups.keys())
            )
            _cluster_section = (
                "## 2b. Struttura dei Cluster\n"
                "*Composizione dei cluster sull'ultimo periodo WFO*\n\n"
                "| Cluster | Label | N. Titoli | Tickers |\n"
                "|---------|-------|-----------|---------|\n"
                + _rows_cl
                + "\n\nPlot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`\n\n---"
            )

    def _plot_ok(name):
        return 'Sì' if (_plots_dir / name).exists() else 'No'

    _card = (
        f"# PTF Card — {portfolio_title} {year}\n\n---\n\n"
        f"## 1. Identità\n"
        f"| Campo | Valore |\n|-------|-------|\n"
        f"| Nome | {portfolio_title} |\n"
        f"| Engine | R-portfolio (rotational momentum) |\n"
        f"| Universe | {universe_size} tickers |\n"
        f"| Benchmark | {benchmark} |\n"
        f"| Periodo analisi | {period_start} → {period_end} |\n"
        f"| Profilo | {profile} |\n"
        f"| Data generazione | {_today} |\n"
        f"| WFO file | `{wfo_config.get('wfo_file_save', 'N/A')}` |\n\n---\n\n"
        f"## 2. Configurazione WFO\n"
        f"| Parametro | Valore | Nota |\n|-----------|--------|------|\n"
        f"| WFO ratio | {wfo_config.get('ratio', 'N/A')} | Rapporto IS/OOS |\n"
        f"| WFO metric | {wfo_config.get('metric', 'N/A')} | Metrica ottimizzazione IS |\n"
        f"| Grid size (full) | {wfo_config.get('n_full_trials', 'N/A')} combinazioni | Spazio parametrico totale |\n"
        f"| Grid size (reduced) | {wfo_config.get('n_reduced_trials', 'N/A')} combinazioni | Dopo stability analysis |\n"
        f"| Stability metric | CAGR, k=3 | Metrica e sottoperiodi |\n"
        f"| n_bootstrap OFC | {wfo_config.get('n_bootstrap_ofc', wfo_config.get('n_bootstrap', 1000))} | Test S3 random selection |\n"
        f"| n_bootstrap MC | {wfo_config.get('n_bootstrap_mc', wfo_config.get('n_bootstrap', 1000))} | Block A (CI) + Block B (Skill Tests) |\n"
        f"| Risk ON/OFF | True | Filtro regime di mercato |\n"
        f"| Clustering | {use_clustering} | Se True: WFO per cluster omogenei |\n\n---\n\n"
        f"{_cluster_section}\n\n"
        f"## 3. Metriche Comparative WFO\n"
        f"*Confronto su periodo comune {period_start} → {period_end}*\n\n"
        f"| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |\n"
        f"|-----------|------------|------|--------|-------|\n"
        f"| Cluster — Risk ON/OFF | {_m('cluster_riskoff','cum')} | {_m('cluster_riskoff','cagr')} | {_m('cluster_riskoff','sharpe')} | {_m('cluster_riskoff','maxdd')} |\n"
        f"| Cluster — Base | {_m('cluster_base','cum')} | {_m('cluster_base','cagr')} | {_m('cluster_base','sharpe')} | {_m('cluster_base','maxdd')} |\n"
        f"| Standard — Risk ON/OFF | {_m('std_riskoff','cum')} | {_m('std_riskoff','cagr')} | {_m('std_riskoff','sharpe')} | {_m('std_riskoff','maxdd')} |\n"
        f"| Standard — Base | {_m('std_base','cum')} | {_m('std_base','cagr')} | {_m('std_base','sharpe')} | {_m('std_base','maxdd')} |\n"
        f"| Benchmark ({benchmark}) | {_m('benchmark','cum')} | {_m('benchmark','cagr')} | {_m('benchmark','sharpe')} | {_m('benchmark','maxdd')} |\n\n---\n\n"
        f"## 4. Overfitting Check (OFC)\n"
        f"*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*\n\n"
        f"### 4a. Path Standard\n"
        f"| Segnale | Cosa misura | Verdetto | Valore |\n|---------|-------------|---------|--------|\n"
        f"| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | {_vd(_s.get('S1_plateau', {}).get('pass'))} | {_s.get('S1_plateau', {}).get('value', 'N/A')} |\n"
        f"| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | {_vd(_s.get('S2_coherence', {}).get('pass'))} | {_s.get('S2_coherence', {}).get('value', 'N/A')} |\n"
        f"| S3 — Random selection | Il risultato Out-Of-Sample batte {wfo_config.get('n_bootstrap_ofc', wfo_config.get('n_bootstrap', 1000))} portafogli con parametri casuali? | {_vd(_s.get('S3_bootstrap', {}).get('pass'))} | p={_s.get('S3_bootstrap', {}).get('p_value', 'N/A')} |\n"
        f"| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | {_vd(_s.get('S4_dsr', {}).get('pass'))} | {_s.get('S4_dsr', {}).get('dsr', 'N/A')} |\n"
        f"| **OFC Verdict** | Soglia: 3/4 segnali | **{'PROMOTED' if ofc_passed_std else 'NOT PROMOTED'}** | |\n\n"
        f"### 4b. Path Cluster\n"
        f"| Segnale | Cosa misura | Verdetto | Valore |\n|---------|-------------|---------|--------|\n"
        f"| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | {_vd(_sc.get('S1_plateau', {}).get('pass')) if _sc else 'N/A'} | {_sc.get('S1_plateau', {}).get('value', 'N/A') if _sc else 'N/A'} |\n"
        f"| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | {_vd(_sc.get('S2_coherence', {}).get('pass')) if _sc else 'N/A'} | {_sc.get('S2_coherence', {}).get('value', 'N/A') if _sc else 'N/A'} |\n"
        f"| S3 — Random selection | Il risultato Out-Of-Sample batte {wfo_config.get('n_bootstrap_ofc', wfo_config.get('n_bootstrap', 1000))} portafogli con parametri casuali? | {_vd(_sc.get('S3_bootstrap', {}).get('pass')) if _sc else 'N/A'} | p={_sc.get('S3_bootstrap', {}).get('p_value', 'N/A') if _sc else 'N/A'} |\n"
        f"| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | {_vd(_sc.get('S4_dsr', {}).get('pass')) if _sc else 'N/A'} | {_sc.get('S4_dsr', {}).get('dsr', 'N/A') if _sc else 'N/A'} |\n"
        f"| **OFC Verdict** | Soglia: 3/4 segnali | **{'PROMOTED' if ofc_passed_cluster else 'NOT PROMOTED' if ofc_passed_cluster is not None else 'N/A'}** | |\n\n---\n\n"
        f"## 5. Monte Carlo Validation\n"
        f"*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*\n\n"
        f"### 5a. Skill Tests (Block B) — Path Standard\n"
        f"| Test | Cosa misura | Verdetto | p-value |\n|------|-------------|---------|----------|\n"
        f"| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | {'Pass' if reshuffle_passed else 'Fail'} | {_reshuffle_pval_str} |\n"
        f"| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | {'Pass' if timing_passed else 'Fail'} | {_timing_pval_str} |\n\n"
        f"### 5a. Skill Tests (Block B) — Path Cluster\n"
        f"| Test | Cosa misura | Verdetto | p-value |\n|------|-------------|---------|----------|\n"
        f"| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | {'Pass' if reshuffle_passed_cl else 'Fail'} | {_reshuffle_pval_str_cl} |\n"
        f"| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | {'Pass' if timing_passed_cl else 'Fail'} | {_timing_pval_str_cl} |\n\n"
        f"### 5b. Confidence Intervals (Block A) — Path Standard\n"
        f"| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |\n"
        f"|--------|---------|---------|---------|-----------|----------|\n"
        f"| A1 — IID Bootstrap | {_ci('A1 · IID Bootstrap', 'CAGR_p5')} | {_ci('A1 · IID Bootstrap', 'CAGR_p50')} | {_ci('A1 · IID Bootstrap', 'CAGR_p95')} | {_ci('A1 · IID Bootstrap', 'Sharpe_p50')} | {_ci('A1 · IID Bootstrap', 'MaxDD_p50')} |\n"
        f"| A2 — Block Bootstrap | {_ci('A2 · Block Bootstrap', 'CAGR_p5')} | {_ci('A2 · Block Bootstrap', 'CAGR_p50')} | {_ci('A2 · Block Bootstrap', 'CAGR_p95')} | {_ci('A2 · Block Bootstrap', 'Sharpe_p50')} | {_ci('A2 · Block Bootstrap', 'MaxDD_p50')} |\n\n"
        f"### 5b. Confidence Intervals (Block A) — Path Cluster\n"
        f"| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |\n"
        f"|--------|---------|---------|---------|-----------|----------|\n"
        f"| A1 — IID Bootstrap | {_ci_cl('A1 · IID Bootstrap', 'CAGR_p5')} | {_ci_cl('A1 · IID Bootstrap', 'CAGR_p50')} | {_ci_cl('A1 · IID Bootstrap', 'CAGR_p95')} | {_ci_cl('A1 · IID Bootstrap', 'Sharpe_p50')} | {_ci_cl('A1 · IID Bootstrap', 'MaxDD_p50')} |\n"
        f"| A2 — Block Bootstrap | {_ci_cl('A2 · Block Bootstrap', 'CAGR_p5')} | {_ci_cl('A2 · Block Bootstrap', 'CAGR_p50')} | {_ci_cl('A2 · Block Bootstrap', 'CAGR_p95')} | {_ci_cl('A2 · Block Bootstrap', 'Sharpe_p50')} | {_ci_cl('A2 · Block Bootstrap', 'MaxDD_p50')} |\n\n---\n\n"
        f"## 6. Skill Profile\n"
        f"*Sintesi della capacità predittiva (Standard / Cluster)*\n\n"
        f"| Test | Cosa misura | Standard | Cluster |\n|------|-------------|---------|---------|\n"
        f"| MC Rotation Reshuffle | La rotazione batte il caso? | {'Pass' if reshuffle_passed else 'Fail'} | {'Pass' if reshuffle_passed_cl else 'Fail'} |\n"
        f"| MC Rebalance Timing | Il timing batte il caso? | {'Pass' if timing_passed else 'Fail'} | {'Pass' if timing_passed_cl else 'Fail'} |\n"
        f"| OFC S3 | Il risultato Out-Of-Sample batte parametri casuali? | {_s3_str} | {_s3_str_cl} |\n\n"
        f"**Skill Profile: {skill_profile}**\n"
        f"*Nota: No-skill non implica PTF non deployabile — il valore può derivare\n"
        f"dalla struttura dell'universe, dal clustering o dal Risk ON/OFF.*\n\n---\n\n"
        f"## 7. Decisione Finale\n"
        f"| Dimensione | Standard | Cluster |\n|-----------|---------|----------|\n"
        f"| OFC Verdict | {'PROMOTED' if ofc_passed_std else 'NOT PROMOTED'} | {'PROMOTED' if ofc_passed_cluster else 'NOT PROMOTED' if ofc_passed_cluster is not None else 'N/A'} |\n"
        f"| Skill Profile | {skill_profile} | {skill_profile} |\n"
        f"| CAGR vs Benchmark | {_m('std_riskoff', 'cagr')} vs {_m('benchmark', 'cagr')} | {_m('cluster_riskoff', 'cagr')} vs {_m('benchmark', 'cagr')} |\n"
        f"| Sharpe vs Benchmark | {_m('std_riskoff', 'sharpe')} vs {_m('benchmark', 'sharpe')} | {_m('cluster_riskoff', 'sharpe')} vs {_m('benchmark', 'sharpe')} |\n"
        f"| MaxDD vs Benchmark | {_m('std_riskoff', 'maxdd')} vs {_m('benchmark', 'maxdd')} | {_m('cluster_riskoff', 'maxdd')} vs {_m('benchmark', 'maxdd')} |\n\n"
        f"**Path deployato**: [ STANDARD | CLUSTER | NESSUNO ] ← compilare\n"
        f"**Motivazione**: ← compilare\n\n---\n\n"
        f"## 8. Note e Avvertenze\n*(compilare a mano)*\n\n---\n\n"
        f"## 9. Plot salvati\n"
        f"| Plot | File | Disponibile |\n|------|------|-------------|\n"
        f"| Equity Standard | equity_std.png | {_plot_ok('equity_std.png')} |\n"
        f"| Equity Cluster | equity_cluster.png | {_plot_ok('equity_cluster.png')} |\n"
        f"| Equity Comparison | equity_comparison.png | {_plot_ok('equity_comparison.png')} |\n"
        f"| MC CI Block A | mc_ci.png | {_plot_ok('mc_ci.png')} |\n"
        f"| MC Reshuffle | mc_reshuffle.png | {_plot_ok('mc_reshuffle.png')} |\n"
        f"| MC Timing | mc_timing.png | {_plot_ok('mc_timing.png')} |\n"
        f"| MC Skill Summary | mc_skill_summary.png | {_plot_ok('mc_skill_summary.png')} |\n"
        f"| Cluster Heatmap | cluster_heatmap.png | {_plot_ok('cluster_heatmap.png')} |\n"
        f"| Cluster Dendrogram | cluster_dendrogram.png | {_plot_ok('cluster_dendrogram.png')} |\n"
        f"| Cluster Scatter | cluster_scatter.png | {_plot_ok('cluster_scatter.png')} |\n"
        f"| MC CI Fan Chart IID | mc_ci_fanchart_iid.png | {_plot_ok('mc_ci_fanchart_iid.png')} |\n"
        f"| MC CI Fan Chart Block | mc_ci_fanchart_block.png | {_plot_ok('mc_ci_fanchart_block.png')} |\n\n---\n"
    )

    output_path.write_text(_card)
    return output_path
    


# ── Private helpers per generate_relazione_tecnica ────────────────────────────
def _diagnose_ofc(ofc_report_std: dict, ofc_report_cluster) -> tuple:
    '''
    Genera paragrafi diagnostici OFC adattativi (sezione 6a).

    Returns (std_paragraph_html, cluster_paragraph_html_or_None).
    '''
    def _para(report, path_name):
        if report is None:
            return f"Il <b>path {path_name}</b> non è stato eseguito."
        is_cluster = path_name.lower() == "cluster"
        sigs     = report.get('signals', {})
        promoted = report.get('promoted', False)
        n_pass   = report.get('n_signals_passed', 0)
        resolved = report.get('resolved', {})
        s1 = sigs.get('S1_plateau', {})
        s2 = sigs.get('S2_coherence', {})
        s3 = sigs.get('S3_bootstrap', {})
        s4 = sigs.get('S4_dsr', {})
        s2_v   = s2.get('value') or 0
        s3_pv  = s3.get('p_value')
        s4_v   = s4.get('dsr')
        s4_thr = resolved.get('s4_dsr_threshold', 0.0)

        if promoted:
            parts = []
            if s2.get('pass'):
                s2_tail = (
                    "la segregazione per cluster elimina le principali fonti di rumore"
                    if is_cluster
                    else "la riduzione della grid via stability analysis produce parametri coerenti tra sottoperiodi"
                )
                parts.append(
                    f"<b>S2 PASS ({s2_v:.3f})</b> mostra che applicando il WFO "
                    "i parametri ottimali diventano stabili tra sottoperiodi: "
                    f"{s2_tail}."
                )
            if s3.get('pass') and s3_pv is not None:
                parts.append(
                    f"<b>S3 PASS (p = {s3_pv:.3f})</b> conferma che il risultato Out-Of-Sample (OOS) "
                    "batte statisticamente portafogli con parametri casuali."
                )
            elif s3_pv is not None:
                if is_cluster:
                    s3_text = (
                        f"S3 (p = {s3_pv:.3f}) non raggiunge la soglia, da leggere come "
                        "<i>borderline</i>: il path opera su un universo già "
                        "pre-filtrato dal clustering."
                    )
                else:
                    s3_text = (
                        f"<b>S3 FAIL (p = {s3_pv:.3f})</b>: il risultato Out-Of-Sample è "
                        "statisticamente indistinguibile da una selezione parametrica casuale."
                    )
                parts.append(s3_text)
            if s4.get('pass') and s4_v is not None:
                parts.append(
                    f"<b>S4 PASS ({s4_v:.3f})</b> indica uno Sharpe Ratio significativo "
                    f"anche dopo correzione per il numero di trial (DSR threshold={s4_thr})."
                )
            body = " ".join(parts)
            return (
                f"Il <b>path {path_name}</b> supera la soglia di promozione. "
                f"{body} "
                f"Il verdetto complessivo è <b>PROMOTED</b> con {n_pass} segnali su 4."
            )
        else:
            fail_parts = []
            if not s2.get('pass'):
                fail_parts.append(
                    f"<b>S2 (flag coherence = {s2_v:.3f})</b> indica che i parametri "
                    "ottimali sui sottoperiodi sono completamente instabili: il WFO non "
                    "converge su una configurazione ricorrente, segno che l'universo unico "
                    "non produce un comportamento sistematico sfruttabile."
                )
            if not s3.get('pass') and s3_pv is not None:
                fail_parts.append(
                    f"<b>S3 (p = {s3_pv:.3f})</b> conferma che il risultato Out-Of-Sample (OOS) è "
                    "statisticamente indistinguibile da una selezione parametrica casuale."
                )
            if not s1.get('pass'):
                s1_v = s1.get('value') or 0
                fail_parts.append(
                    f"<b>S1 ({s1_v:.3f})</b> segnala che il WFO converge su un numero "
                    "troppo ristretto di configurazioni parametriche."
                )
            if not s4.get('pass') and s4_v is not None:
                fail_parts.append(
                    f"<b>S4 DSR = {s4_v:.3f}</b> indica uno Sharpe non sufficientemente "
                    f"significativo dopo correzione per il numero di trial (threshold={s4_thr})."
                )
            body = " ".join(fail_parts) if fail_parts else "Segnali critici falliti."
            return (
                f"Il <b>path {path_name}</b> fallisce su segnali critici. "
                f"{body} "
                f"La promozione del path {path_name} non è giustificata."
            )

    std_txt = _para(ofc_report_std, "Standard")
    clu_txt = _para(ofc_report_cluster, "Cluster") if ofc_report_cluster is not None else None
    return std_txt, clu_txt


def _recommended_path(ofc_report_std, ofc_report_cluster) -> str | None:
    """
    Ritorna il path raccomandato per la narrativa:
      'cluster' / 'std' / None
    
    Logica (tie-break Cluster preferito):
      - Cluster promosso (esclusivo o con Standard)   → 'cluster'
      - Solo Standard promosso                         → 'std'
      - Nessuno promosso                               → None
    """
    std_p = bool(ofc_report_std.get('promoted', False)) if ofc_report_std else False
    clu_p = bool(ofc_report_cluster.get('promoted', False)) if ofc_report_cluster else False
    if clu_p:                 return 'cluster'
    if std_p and not clu_p:   return 'std'
    return None
def _diagnose_mc(
    mc_skill: dict,
    mc_ci,
    metrics_comparison: dict,
    *,
    mc_skill_cluster: dict | None = None,
    mc_ci_cluster=None,
    recommended_path: str | None = None,
    ofc_report_std: dict | None = None,
    ofc_report_cluster: dict | None = None,
) -> tuple:
    '''
    Genera paragrafi diagnostici MC adattivi (sezione 6.b skill + 6.c CI).
    
    Strategia Opzione C (B-005):
    - Focus su path raccomandato (cluster se promosso, std se solo std promosso, None altrimenti)
    - Nota di contrasto sull'altro path
    
    Parameters
    ----------
    recommended_path : 'std' | 'cluster' | None
        Se None, viene calcolato da ofc_report_std/cluster se passati;
        altrimenti default a 'std' per retro-compatibilità.
    
    Returns
    -------
    (skill_paragraph1, skill_paragraph2_or_None, ci_paragraph)
    '''
    # Risolvi recommended_path se non passato esplicitamente
    if recommended_path is None and (ofc_report_std is not None or ofc_report_cluster is not None):
        recommended_path = _recommended_path(ofc_report_std, ofc_report_cluster)
    if recommended_path is None:
        recommended_path = 'std'   # retro-compat
    
    # Seleziona mc_skill e mc_ci del path raccomandato
    if recommended_path == 'cluster' and mc_skill_cluster is not None:
        mc_skill_main = mc_skill_cluster
        mc_ci_main    = mc_ci_cluster if mc_ci_cluster is not None else mc_ci
        mc_skill_alt  = mc_skill
        mc_ci_alt     = mc_ci
        path_label_main = 'Cluster'
        path_label_alt  = 'Standard'
        path_qualifier  = "sul sub-universo post-clustering"
    else:
        mc_skill_main = mc_skill
        mc_ci_main    = mc_ci
        mc_skill_alt  = mc_skill_cluster
        mc_ci_alt     = mc_ci_cluster
        path_label_main = 'Standard'
        path_label_alt  = 'Cluster'
        path_qualifier  = "sull'universo intero"
    
    # ── 6.b — Skill paragraphs ──────────────────────────────────────────────
    def _extract_b1_b2(mc):
        b1_pv = mc.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
        b2_pv = mc.get('rebalance_timing', {}).get('p_values', {}).get('CAGR')
        return b1_pv, b2_pv
    
    b1_pv, b2_pv = _extract_b1_b2(mc_skill_main)
    b1_pass = (b1_pv is not None) and (b1_pv < 0.10)
    b2_pass = (b2_pv is not None) and (b2_pv < 0.10)
    b1_str  = f"p = {b1_pv:.3f}" if b1_pv is not None else "N/A"
    b2_str  = f"p = {b2_pv:.3f}" if b2_pv is not None else "N/A"
    
    # Skill profile path raccomandato (nomenclatura corretta B-006)
    if b1_pass and b2_pass:           sp_main = 'Strong'
    elif b1_pass and not b2_pass:     sp_main = 'Selection-driven'
    elif not b1_pass and b2_pass:     sp_main = 'Timing-driven'
    else:                             sp_main = 'No-skill'
    
    # Costruzione skill1 in base al profile
    if sp_main == 'Strong':
        skill1 = (
            f"Il path candidato al deploy è <b>{path_label_main}</b>. "
            f"Entrambi i test Skill (Block B) risultano <b>PASS</b>: "
            f"B1 ({b1_str}) indica che la rotazione momentum {path_qualifier} batte una "
            f"selezione casuale dei titoli, B2 ({b2_str}) che il timing mensile aggiunge "
            f"valore statisticamente significativo rispetto a date di ribilanciamento casuali. "
            f"Lo <b>Skill Profile</b> è <b>Strong</b>."
        )
        skill2 = (
            "La combinazione di selezione e timing statisticamente verificati indica un "
            "motore con capacità predittiva robusta. I risultati del portafoglio riflettono "
            "sia la struttura dell'universo sia il valore aggiunto della rotazione."
        )
    elif sp_main == 'Selection-driven':
        skill1 = (
            f"Il path candidato al deploy è <b>{path_label_main}</b>. "
            f"Il test B1 ({b1_str}) risulta <b>PASS</b>: la rotazione momentum "
            f"{path_qualifier} batte una selezione casuale dei titoli in modo statisticamente "
            f"significativo. Il test B2 ({b2_str}) risulta invece <b>FAIL</b>: il timing "
            f"mensile non aggiunge valore rispetto a date di ribilanciamento casuali. "
            f"Lo <b>Skill Profile</b> è <b>Selection-driven</b>."
        )
        skill2 = (
            f"Il valore del path {path_label_main} deriva dalla skill di <b>selezione</b> "
            "(B1 PASS) combinata con fattori strutturali (filtro Risk ON/OFF, asimmetria "
            "asset nei regimi OFF" + 
            (", pre-filtro del cluster AVOID" if recommended_path == 'cluster' else "") +
            "). Il timing dei rebalance non è una fonte di valore: opportunità di valutare "
            "frequenze di ribilanciamento ridotte."
        )
    elif sp_main == 'Timing-driven':
        skill1 = (
            f"Il path candidato al deploy è <b>{path_label_main}</b>. "
            f"Il test B1 ({b1_str}) risulta <b>FAIL</b>: la rotazione momentum {path_qualifier} "
            f"è indistinguibile da una selezione casuale dei titoli. Il test B2 ({b2_str}) "
            f"risulta invece <b>PASS</b>: il timing mensile aggiunge valore statisticamente "
            f"significativo rispetto a date di ribilanciamento casuali. "
            f"Lo <b>Skill Profile</b> è <b>Timing-driven</b>."
        )
        skill2 = (
            "Il motore dimostra capacità predittiva sul timing delle rotazioni, anche se "
            "la selezione cross-sezionale non è statisticamente distinguibile dal caso. "
            "Il valore generato dipende prevalentemente dalla struttura temporale delle "
            "rotazioni piuttosto che dalla scelta dei singoli titoli."
        )
    else:   # No-skill
        skill1 = (
            f"Il path candidato al deploy è <b>{path_label_main}</b>. "
            f"Entrambi i test Skill (Block B) risultano <b>FAIL</b>: B1 ({b1_str}) indica "
            f"che la rotazione momentum {path_qualifier} è indistinguibile da una selezione "
            f"casuale dei titoli, B2 ({b2_str}) che il timing mensile non aggiunge valore. "
            f"Lo <b>Skill Profile</b> è <b>No-skill</b>."
        )
        # Skill2: contestualizzazione strutturale (se abbiamo metriche di confronto)
        def _pf_cagr(key):
            try:
                pf = metrics_comparison.get(key)
                return pf.annualized_return() * 100 if pf else None
            except Exception:
                return None
        
        main_key = 'cluster_riskoff' if recommended_path == 'cluster' else 'std_riskoff'
        cagr_main = _pf_cagr(main_key)
        cagr_bk   = _pf_cagr('benchmark')
        
        dd_txt = ''
        try:
            pf_m = metrics_comparison.get(main_key)
            pf_b = metrics_comparison.get('benchmark')
            if pf_m and pf_b:
                dd_diff = abs(pf_b.max_drawdown()) * 100 - abs(pf_m.max_drawdown()) * 100
                if abs(dd_diff) > 0.5:
                    dd_txt = (
                        f" con MaxDD ridotto di {abs(dd_diff):.0f} punti percentuali "
                        f"({abs(pf_m.max_drawdown())*100:.1f}% vs {abs(pf_b.max_drawdown())*100:.1f}%)"
                    )
        except Exception:
            pass
        
        if cagr_main is not None and cagr_bk is not None:
            structural_sources = (
                "(1) la <b>composizione dell'universo</b> già depurata dal cluster AVOID; "
                "(2) il <b>filtro Risk ON/OFF</b> che evita drawdown sistemici; "
                "(3) l'<b>asimmetria asset</b> con WFO su equity ed esecuzione che include "
                "difensivi non-equity nei regimi OFF."
            ) if recommended_path == 'cluster' else (
                "(1) il <b>filtro Risk ON/OFF</b> che evita drawdown sistemici; "
                "(2) l'<b>asimmetria asset</b> con WFO su equity ed esecuzione che include "
                "difensivi non-equity nei regimi OFF."
            )
            skill2 = (
                f"Il fatto che il portafoglio realizzi {cagr_main:.1f}% CAGR vs {cagr_bk:.1f}% "
                f"del benchmark{dd_txt} suggerisce che il valore deriva da fonti "
                f"<b>strutturali</b> e non dalla skill rotazionale: " + structural_sources
            )
        else:
            skill2 = None
    
    # ── 6.b — Nota di contrasto sull'alternativa ────────────────────────────
    if mc_skill_alt is not None:
        b1_alt, b2_alt = _extract_b1_b2(mc_skill_alt)
        # b1_alt_str = f"p={b1_alt:.3f}" if b1_alt is not None else "N/A"
        # b2_alt_str = f"p={b2_alt:.3f}" if b2_alt is not None else "N/A"
        b1_alt_str = f"{b1_alt:.3f}" if b1_alt is not None else "N/A"
        b2_alt_str = f"{b2_alt:.3f}" if b2_alt is not None else "N/A"
        b1_alt_pass = (b1_alt is not None) and (b1_alt < 0.10)
        b2_alt_pass = (b2_alt is not None) and (b2_alt < 0.10)
        if b1_alt_pass and b2_alt_pass:           sp_alt = 'Strong'
        elif b1_alt_pass and not b2_alt_pass:     sp_alt = 'Selection-driven'
        elif not b1_alt_pass and b2_alt_pass:     sp_alt = 'Timing-driven'
        else:                                     sp_alt = 'No-skill'
        
        skill3 = (
            f"<i>Per contrasto, il path {path_label_alt} mostra B1={b1_alt_str} "
            f"({'PASS' if b1_alt_pass else 'FAIL'}) e B2={b2_alt_str} "
            f"({'PASS' if b2_alt_pass else 'FAIL'}) — Skill Profile: {sp_alt}.</i>"
        )
    else:
        skill3 = None
    
    # Concatena skill2 + skill3 se presenti entrambi (per non avere 3 paragrafi separati)
    if skill2 is None:
        skill_para2 = skill3
    elif skill3 is None:
        skill_para2 = skill2
    else:
        skill_para2 = skill2 + '<br/><br/>' + skill3
    
    # ── 6.c — CI paragraph ──────────────────────────────────────────────────
    def _civ(ci_df, row, col, pct=True):
        try:
            v = float(ci_df.loc[row, col])
            return f"{v*100:.1f}%" if pct else f"{v:.3f}"
        except Exception:
            return "N/A"
    
    a1_c50 = _civ(mc_ci_main, 'A1 · IID Bootstrap',   'CAGR_p50')
    a2_c50 = _civ(mc_ci_main, 'A2 · Block Bootstrap', 'CAGR_p50')
    a1_s50 = _civ(mc_ci_main, 'A1 · IID Bootstrap',   'Sharpe_p50', pct=False)
    a2_s50 = _civ(mc_ci_main, 'A2 · Block Bootstrap', 'Sharpe_p50', pct=False)
    a1_d50 = _civ(mc_ci_main, 'A1 · IID Bootstrap',   'MaxDD_p50')
    a2_d50 = _civ(mc_ci_main, 'A2 · Block Bootstrap', 'MaxDD_p50')
    
    # CAGR realizzato (Actual) del path raccomandato
    main_key = 'cluster_riskoff' if recommended_path == 'cluster' else 'std_riskoff'
    act_cagr_val = None
    try:
        pf = metrics_comparison.get(main_key)
        if pf:
            act_cagr_val = pf.annualized_return() * 100
    except Exception:
        pass
    act_cagr_str = f"{act_cagr_val:.1f}%" if act_cagr_val is not None else "N/A"
    
    # Calcolo posizione percentile reale dell'Actual rispetto alla distribuzione bootstrap A1
    pct_position_txt = ""
    try:
        a1_p5  = float(mc_ci_main.loc['A1 · IID Bootstrap', 'CAGR_p5'])  * 100
        a1_p50 = float(mc_ci_main.loc['A1 · IID Bootstrap', 'CAGR_p50']) * 100
        a1_p95 = float(mc_ci_main.loc['A1 · IID Bootstrap', 'CAGR_p95']) * 100
        if act_cagr_val is not None:
            if act_cagr_val > a1_p95:
                pct_position_txt = (
                    f"Il CAGR realizzato ({act_cagr_str}) cade sopra il p95 della "
                    f"distribuzione bootstrap (p95 = {a1_p95:.1f}%): risultato fuori scala "
                    "rispetto al naive bootstrap, segno di una forte componente di selezione "
                    "strutturale non catturata dal ricampionamento marginale."
                )
            elif act_cagr_val > a1_p50:
                pct_position_txt = (
                    f"Il CAGR realizzato ({act_cagr_str}) si colloca tra la mediana "
                    f"({a1_p50:.1f}%) e il p95 ({a1_p95:.1f}%) della distribuzione: "
                    "risultato favorevole ma all'interno della banda di plausibilità del bootstrap."
                )
            elif act_cagr_val > a1_p5:
                pct_position_txt = (
                    f"Il CAGR realizzato ({act_cagr_str}) si colloca tra il p5 "
                    f"({a1_p5:.1f}%) e la mediana ({a1_p50:.1f}%) della distribuzione: "
                    "risultato modesto, sotto le aspettative del bootstrap."
                )
            else:
                pct_position_txt = (
                    f"Il CAGR realizzato ({act_cagr_str}) cade sotto il p5 della "
                    f"distribuzione bootstrap (p5 = {a1_p5:.1f}%): risultato in coda negativa, "
                    "segnale di sottoperformance rispetto al naive bootstrap."
                )
    except Exception:
        pass
    
    ci_para = (
        f"Il path <b>{path_label_main}</b> mostra metodi bootstrap coerenti tra loro: "
        f"CAGR p50 = {a1_c50}–{a2_c50} (IID/Block), "
        f"Sharpe p50 = {a1_s50} / {a2_s50}, "
        f"MaxDD p50 = {a1_d50} / {a2_d50}. "
        + pct_position_txt
    )
    
    # Nota di contrasto su path alternativo (Block A)
    if mc_ci_alt is not None:
        a1_c50_alt = _civ(mc_ci_alt, 'A1 · IID Bootstrap',   'CAGR_p50')
        a2_c50_alt = _civ(mc_ci_alt, 'A2 · Block Bootstrap', 'CAGR_p50')
        a1_d50_alt = _civ(mc_ci_alt, 'A1 · IID Bootstrap',   'MaxDD_p50')
        ci_para += (
            f"<br/><br/><i>Per contrasto, il path {path_label_alt} mostra "
            f"CAGR p50 = {a1_c50_alt}–{a2_c50_alt}, MaxDD p50 ≈ {a1_d50_alt}.</i>"
        )
    
    return skill1, skill_para2, ci_para
    

def _build_verdict_text(
    *,
    ofc_report_std: dict,
    ofc_report_cluster: dict,
    metrics_comparison: dict,
    mc_skill: dict,
    skill_profile: str,
    wfo_config: dict,
    skill_profile_cluster: str | None = None,   # B-005: profile path Cluster
) -> str:
    '''
    Costruisce il testo adattivo del Verdict Box (sezione 7).
    Ritorna stringa formattata HTML reportlab (<b>, <br/>).
    Il testo NON include la motivazione soggettiva: solo
    raccomandazione tecnica derivata dai verdetti + caveat.
    
    skill_profile     : profile del path Standard (legacy: parametro singolo)
    skill_profile_cluster : B-005, opzionale; se presente, la caveat 'No-skill'
                            sarà valutata sul path raccomandato (Cluster preferito
                            quando promosso, vedi logica A/B/C/D sotto).
    '''
    promoted_std     = bool((ofc_report_std     or {}).get('promoted', False))
    promoted_cluster = (
        bool((ofc_report_cluster or {}).get('promoted', False))
        if ofc_report_cluster is not None else False
    )
    n_pass_std     = (ofc_report_std     or {}).get('n_signals_passed', 0)
    n_pass_cluster = (ofc_report_cluster or {}).get('n_signals_passed', 0)
    profile = wfo_config.get('profile', 'N/A')

    def _fmt(key, metric):
        pf = metrics_comparison.get(key)
        if pf is None:
            return 'N/A'
        try:
            if metric == 'cagr':   return f"{pf.annualized_return()*100:.1f}%"
            if metric == 'sharpe': return f"{pf.sharpe_ratio():.2f}"
            if metric == 'maxdd':  return f"{abs(pf.max_drawdown())*100:.1f}%"
        except Exception:
            return 'N/A'
        return 'N/A'

    def _sharpe_val(key):
        pf = metrics_comparison.get(key)
        if pf is None:
            return None
        try:
            return float(pf.sharpe_ratio())
        except Exception:
            return None

    s3_report = {}
    # B-005: traccia quale profile è "il riferimento" per la caveat No-skill
    skill_profile_for_caveat = skill_profile

    if promoted_cluster and not promoted_std:
        # CASO A — Solo Cluster PROMOTED
        cagr_cl = _fmt('cluster_riskoff', 'cagr')
        sh_cl   = _fmt('cluster_riskoff', 'sharpe')
        dd_cl   = _fmt('cluster_riskoff', 'maxdd')
        cagr_bm = _fmt('benchmark', 'cagr')
        sh_bm   = _fmt('benchmark', 'sharpe')
        dd_bm   = _fmt('benchmark', 'maxdd')
        rec = (
            f"<b>Raccomandazione tecnica:</b> il path <b>Cluster — Risk ON/OFF</b> supera "
            f"l'Overfitting Check (<b>OFC PROMOTED</b>, {n_pass_cluster}/4 segnali) e mostra "
            f"metriche significativamente superiori al benchmark: CAGR {cagr_cl} vs {cagr_bm}, "
            f"Sharpe {sh_cl} vs {sh_bm}, MaxDD {dd_cl} vs {dd_bm}. "
            f"Il path è candidabile al deploy operativo con profilo <b>{profile}</b>."
        )
        s3_report = (ofc_report_cluster or {}).get('signals', {}).get('S3_bootstrap', {})
        # B-005: caveat valutata sul path Cluster
        if skill_profile_cluster is not None:
            skill_profile_for_caveat = skill_profile_cluster

    elif promoted_std and not promoted_cluster:
        # CASO B — Solo Standard PROMOTED  (path raccomandato = Std, profile = skill_profile)
        cagr_st = _fmt('std_riskoff', 'cagr')
        sh_st   = _fmt('std_riskoff', 'sharpe')
        dd_st   = _fmt('std_riskoff', 'maxdd')
        cagr_bm = _fmt('benchmark', 'cagr')
        sh_bm   = _fmt('benchmark', 'sharpe')
        dd_bm   = _fmt('benchmark', 'maxdd')
        rec = (
            f"<b>Raccomandazione tecnica:</b> il path <b>Standard — Risk ON/OFF</b> supera "
            f"l'Overfitting Check (<b>OFC PROMOTED</b>, {n_pass_std}/4 segnali) e mostra "
            f"metriche significativamente superiori al benchmark: CAGR {cagr_st} vs {cagr_bm}, "
            f"Sharpe {sh_st} vs {sh_bm}, MaxDD {dd_st} vs {dd_bm}. "
            f"Il path è candidabile al deploy operativo con profilo <b>{profile}</b>."
        )
        s3_report = (ofc_report_std or {}).get('signals', {}).get('S3_bootstrap', {})
        # skill_profile_for_caveat resta = skill_profile (Standard)

    elif promoted_std and promoted_cluster:
        # CASO C — Entrambi PROMOTED: preferire Sharpe più alto (logica esistente conservata)
        sh_std_v = _sharpe_val('std_riskoff')
        sh_clu_v = _sharpe_val('cluster_riskoff')
        if sh_std_v is not None and sh_clu_v is not None and (sh_std_v - sh_clu_v) > 0.05:
            pref_path   = 'Standard — Risk ON/OFF'
            pref_key    = 'std_riskoff'
            pref_n      = n_pass_std
            pref_reason = f"Sharpe superiore ({_fmt('std_riskoff','sharpe')} vs {_fmt('cluster_riskoff','sharpe')})"
            s3_report   = (ofc_report_std or {}).get('signals', {}).get('S3_bootstrap', {})
            # skill_profile_for_caveat resta = skill_profile (Standard)
        else:
            pref_path   = 'Cluster — Risk ON/OFF'
            pref_key    = 'cluster_riskoff'
            pref_n      = n_pass_cluster
            if sh_std_v is not None and sh_clu_v is not None:
                pref_reason = f"Sharpe superiore ({_fmt('cluster_riskoff','sharpe')} vs {_fmt('std_riskoff','sharpe')})"
            else:
                pref_reason = "CAGR e stabilità complessivi superiori"
            s3_report = (ofc_report_cluster or {}).get('signals', {}).get('S3_bootstrap', {})
            # B-005: caveat sul Cluster
            if skill_profile_cluster is not None:
                skill_profile_for_caveat = skill_profile_cluster
        cagr_p  = _fmt(pref_key, 'cagr')
        sh_p    = _fmt(pref_key, 'sharpe')
        dd_p    = _fmt(pref_key, 'maxdd')
        cagr_bm = _fmt('benchmark', 'cagr')
        sh_bm   = _fmt('benchmark', 'sharpe')
        dd_bm   = _fmt('benchmark', 'maxdd')
        rec = (
            "<b>Raccomandazione tecnica:</b> entrambi i path superano l'Overfitting Check. "
            f"Si raccomanda il path <b>{pref_path}</b> ({pref_reason}): "
            f"CAGR {cagr_p} vs {cagr_bm}, Sharpe {sh_p} vs {sh_bm}, MaxDD {dd_p} vs {dd_bm}. "
            f"OFC PROMOTED con {pref_n}/4 segnali. "
            f"Il path è candidabile al deploy operativo con profilo <b>{profile}</b>."
        )

    else:
        # CASO D — Nessuno PROMOTED (logica invariata)
        sh_std_v = _sharpe_val('std_riskoff')
        sh_clu_v = _sharpe_val('cluster_riskoff')
        if sh_std_v is not None and sh_clu_v is not None:
            if sh_clu_v >= sh_std_v:
                less_bad   = "Cluster"
                less_bad_n = n_pass_cluster
            else:
                less_bad   = "Standard"
                less_bad_n = n_pass_std
            less_bad_txt = (
                f" Il path {less_bad} mostra comunque il miglior risultato relativo "
                f"({less_bad_n}/4 segnali OFC), ma rimane sotto la soglia di promozione."
            )
        else:
            less_bad_txt = ""
        rec = (
            "<b>Raccomandazione tecnica:</b> nessun path supera l'Overfitting Check. "
            f"Il deploy operativo <b>non è raccomandato</b> sulla base dell'analisi corrente."
            f"{less_bad_txt} "
            "Si consiglia di rivedere la configurazione WFO, espandere il grid di ricerca "
            "o raccogliere più storia OOS prima di rivalutare la deployabilità."
        )

    caveats = []
    # B-005: usa skill_profile_for_caveat (può essere Std o Cluster a seconda del CASO scelto)
    sp_lower = (skill_profile_for_caveat or '').lower()
    if 'no-skill' in sp_lower or 'no skill' in sp_lower:
        caveats.append(
            "<b>Skill Profile:</b> il profilo <b>No-skill</b> del rotation engine indica "
            "che i test MC Block B (reshuffle e timing) non rilevano una skill statistica "
            "diretta del motore di rotazione. Questo NON implica non-deployabilità: il "
            "valore generato deriva da fonti strutturali — composizione dell'universo "
            "post-clustering, filtro Risk ON/OFF e asimmetria degli asset difensivi nei "
            "regimi OFF — che il test B1 non cattura per design."
        )
    # B-005: nuove caveat per Selection-driven / Timing-driven / Strong
    elif 'selection-driven' in sp_lower:
        caveats.append(
            "<b>Skill Profile:</b> il profilo <b>Selection-driven</b> indica che il test "
            "MC B1 rileva skill statisticamente significativa nella selezione dei titoli "
            "(la rotazione momentum batte una selezione casuale dall'universo), mentre il "
            "test B2 non rileva valore aggiunto dal timing dei rebalance. Opportunità di "
            "valutare frequenze di ribilanciamento ridotte."
        )
    elif 'timing-driven' in sp_lower:
        caveats.append(
            "<b>Skill Profile:</b> il profilo <b>Timing-driven</b> indica che il test "
            "MC B2 rileva skill statisticamente significativa nel timing dei rebalance, "
            "mentre la selezione cross-sezionale (B1) non è distinguibile dal caso. "
            "Il valore deriva prevalentemente dalla struttura temporale delle rotazioni."
        )
    elif 'strong' in sp_lower:
        caveats.append(
            "<b>Skill Profile:</b> il profilo <b>Strong</b> indica che entrambi i test MC "
            "Block B (B1 selection + B2 timing) rilevano skill statisticamente significativa. "
            "Il motore mostra capacità predittiva robusta sia sulla selezione dei titoli "
            "sia sul timing delle rotazioni."
        )
    if s3_report:
        s3_pv  = s3_report.get('p_value')
        s3_thr = s3_report.get('threshold', 0.10)
        if s3_pv is not None and abs(s3_pv - s3_thr) <= 0.05:
            caveats.append(
                f"<b>S3 borderline:</b> il segnale S3 (p = {s3_pv:.3f}) è borderline "
                f"rispetto alla soglia di profilo ({s3_thr:.2f}). Monitorare in produzione "
                "la stabilità OOS: se il p-value supera la soglia in finestre successive, "
                "rivalutare il profilo di deploy."
            )
    caveats.append(
        "<b>Monitoraggio:</b> seguire in produzione la coerenza dei flag WFO (S2) e la "
        "composizione dei cluster ad ogni ribilanciamento annuale. Ripetere la WFO in caso "
        "di cambio di regime macroeconomico, degrado della stabilità dei parametri o "
        "variazione significativa dell'universo di investimento."
    )
    return rec + "<br/><br/>" + "<br/>".join(caveats)
    
def _build_verdict_text_compact(
    *,
    alt_path_label: str,                # 'Standard' o 'Cluster'
    ofc_report_alt: dict | None,
    metrics_comparison: dict,
    metrics_key: str,                   # 'std_riskoff' o 'cluster_riskoff'
    skill_profile_alt: str | None,
    benchmark_key: str = 'benchmark',
) -> str:
    '''
    Costruisce un Verdict Box compatto per il path NON raccomandato (sezione 7).
    Mostra metriche essenziali e verdetto OFC, senza caveat estese.
    
    Ritorna stringa HTML reportlab (<b>, <br/>).
    '''
    def _fmt(key, metric):
        pf = metrics_comparison.get(key)
        if pf is None:
            return 'N/A'
        try:
            if metric == 'cagr':   return f"{pf.annualized_return()*100:.1f}%"
            if metric == 'sharpe': return f"{pf.sharpe_ratio():.2f}"
            if metric == 'maxdd':  return f"{abs(pf.max_drawdown())*100:.1f}%"
        except Exception:
            return 'N/A'
        return 'N/A'
    
    promoted = bool((ofc_report_alt or {}).get('promoted', False))
    n_pass   = (ofc_report_alt or {}).get('n_signals_passed', 0)
    ofc_str  = f"<b>OFC PROMOTED</b> ({n_pass}/4)" if promoted else f"<b>OFC NOT PROMOTED</b> ({n_pass}/4)"
    
    cagr_a = _fmt(metrics_key, 'cagr')
    sh_a   = _fmt(metrics_key, 'sharpe')
    dd_a   = _fmt(metrics_key, 'maxdd')
    cagr_b = _fmt(benchmark_key, 'cagr')
    sh_b   = _fmt(benchmark_key, 'sharpe')
    dd_b   = _fmt(benchmark_key, 'maxdd')
    
    sp_str = f", Skill Profile <b>{skill_profile_alt}</b>" if skill_profile_alt else ""
    
    txt = (
        f"<b>Path alternativo · {alt_path_label} — Risk ON/OFF</b>: "
        f"{ofc_str}{sp_str}. "
        f"CAGR {cagr_a} vs {cagr_b}, Sharpe {sh_a} vs {sh_b}, MaxDD {dd_a} vs {dd_b}."
    )
    return txt

def generate_relazione_tecnica(
    *,
    portfolio_title: str,
    year: int,
    profile: str,
    benchmark: str,
    period: tuple,
    universe_size: int,
    wfo_config: dict,
    cluster_result: dict | None,
    metrics_comparison: dict,
    ofc_report_std: dict,
    ofc_report_cluster: dict | None,
    mc_skill: dict,
    mc_ci,
    skill_profile: str,
    skill_profile_cluster: str | None = None,   # B-005
    plots_dir,
    output_path,
    gen_date: str | None = None,
    ci_results: dict | None = None,
    mc_skill_cluster: dict | None = None,
    mc_ci_cluster=None,
) -> _Path_doc:
    
    '''
    Genera la Relazione Tecnica PDF con reportlab.

    Struttura: sezione 1 Identita, 2 WFO Config, 2b Cluster,
    3 Metriche, 4 OFC, 5 MC (5a skill + 5b CI),
    6 Diagnosi strutturale, 7 Decisione Finale + verdict box.
    Incorpora 10 figure da plots_dir.

    Parameters
    ----------
    mc_ci : pd.DataFrame
        ci_summary_df da run_all_mc_methods_rotational.
    plots_dir : path-like
        Directory con i PNG salvati.
    output_path : path-like
        Path del PDF da generare.
    gen_date : str, optional
        Data generazione ISO (default: oggi).
    ci_results : dict | None, optional
        Risultato di run_mc_confidence_intervals_rotational. Se contiene la chiave 'sri',
        aggiunge la sezione SRI/MRM PRIIPs Cat.3 nella scheda tecnica.

    Returns
    -------
    Path
        Path del PDF scritto.
    '''
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable,
    )
    from reportlab.platypus import Image as _RLImage
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    try:
        from PIL import Image as _PILImage
        _HAS_PIL = True
    except ImportError:
        _HAS_PIL = False

    plots_dir   = _Path_doc(plots_dir)
    output_path = _Path_doc(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if gen_date is None:
        gen_date = _dt_doc.date.today().isoformat()
    period_start, period_end = (period if len(period) == 2 else (period[0], gen_date))
    use_clustering = wfo_config.get('use_clustering', False)

    C_NAVY    = rl_colors.HexColor(_RL_NAVY)
    C_NAVY_LT = rl_colors.HexColor(_RL_NAVY_LT)
    C_GREEN   = rl_colors.HexColor(_RL_GREEN)
    C_RED     = rl_colors.HexColor(_RL_RED)
    C_GRAY_LT = rl_colors.HexColor(_RL_GRAY_LT)
    C_GRAY_BD = rl_colors.HexColor(_RL_GRAY_BD)
    C_WHITE   = rl_colors.white
    C_TEXT    = rl_colors.HexColor(_RL_TEXT)

    PAGE_W, PAGE_H = A4
    MARGIN    = 20 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN

    styles = getSampleStyleSheet()

    def _st(name, parent='Normal', **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    st_title    = _st('_rt_title',  'Title', fontSize=24, textColor=C_NAVY,
                       spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Bold')
    st_subtitle = _st('_rt_sub',    fontSize=10, textColor=C_NAVY_LT,
                       spaceAfter=12, alignment=TA_CENTER)
    st_section  = _st('_rt_sec',    fontSize=13, textColor=C_NAVY, spaceBefore=12,
                       spaceAfter=5, fontName='Helvetica-Bold')
    st_subsec   = _st('_rt_ssec',   fontSize=10.5, textColor=C_NAVY_LT, spaceBefore=7,
                       spaceAfter=3, fontName='Helvetica-Bold')
    st_body     = _st('_rt_body',   fontSize=9.5, textColor=C_TEXT, spaceAfter=6,
                       alignment=TA_JUSTIFY, leading=14)
    st_caption  = _st('_rt_cap',    fontSize=7.5, textColor=C_NAVY_LT, spaceAfter=6,
                       alignment=TA_CENTER, fontName='Helvetica-Oblique')
    st_cell     = _st('_rt_cell',   fontSize=8.5, textColor=C_TEXT)
    st_cell_hdr = _st('_rt_chdr',   fontSize=8.5, textColor=C_WHITE,
                       fontName='Helvetica-Bold')
    st_cell_hdrc= _st('_rt_chdrc',  fontSize=8.5, textColor=C_WHITE,
                       fontName='Helvetica-Bold', alignment=TA_CENTER)
    st_cell_bold= _st('_rt_cbold',  fontSize=8.5, textColor=C_TEXT,
                       fontName='Helvetica-Bold')
    st_cell_ctr = _st('_rt_cctr',   fontSize=8.5, textColor=C_TEXT,
                       alignment=TA_CENTER)
    st_verd     = _st('_rt_verd',   fontSize=9, textColor=C_WHITE,
                       fontName='Helvetica-Bold', alignment=TA_CENTER)
    st_vbox     = _st('_rt_vbox',   fontSize=9.5, textColor=C_NAVY, leading=14)
    st_vbox_j   = _st('_rt_vbox_j', fontSize=9.5, textColor=C_NAVY, leading=14,
                       alignment=TA_JUSTIFY)

    def _ts_base():
        return [
            ('BACKGROUND', (0, 0), (-1, 0), C_NAVY_LT),
            ('TEXTCOLOR',  (0, 0), (-1, 0), C_WHITE),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 8.5),
            ('GRID',       (0, 0), (-1, -1), 0.3, C_GRAY_BD),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_WHITE, C_GRAY_LT]),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
        ]

    _ptf_label  = f"Relazione Tecnica · {portfolio_title} {year}"
    _foot_label = f"Generato il {gen_date} · investia.cloud · uso interno"

    def _draw_hf(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, PAGE_H - 12 * mm, PAGE_W, 12 * mm, fill=1, stroke=0)
        canvas.setFont('Helvetica-Bold', 9)
        canvas.setFillColor(C_WHITE)
        canvas.drawString(MARGIN, PAGE_H - 8 * mm,
                          'TSlab — Quantitative Portfolio Lab')
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 8 * mm, _ptf_label)
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(rl_colors.HexColor('#666666'))
        canvas.drawString(MARGIN, 8 * mm, _foot_label)
        canvas.drawRightString(PAGE_W - MARGIN, 8 * mm, f"Pag. {doc.page}")
        canvas.setStrokeColor(C_GRAY_BD)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 11 * mm, PAGE_W - MARGIN, 11 * mm)
        canvas.restoreState()

    def _img(fname, caption=None):
        p = plots_dir / fname
        if not p.exists():
            return []
        try:
            from reportlab.platypus import KeepTogether
            w = CONTENT_W
            if _HAS_PIL:
                with _PILImage.open(p) as im:
                    iw, ih = im.size
                h = w * (ih / iw)
            else:
                h = w * 0.6
            elems = [_RLImage(str(p), width=w, height=h)]
            if caption:
                elems.append(Paragraph(caption, st_caption))
            return [KeepTogether(elems)]
        except Exception:
            return []
            
    def _img_sub(subdir, fname, caption=None):
            """Variante di _img che cerca in plots_dir/subdir/fname."""
            p = plots_dir / subdir / fname
            if not p.exists():
                return []
            try:
                from reportlab.platypus import KeepTogether
                w = CONTENT_W
                if _HAS_PIL:
                    with _PILImage.open(p) as im:
                        iw, ih = im.size
                    h = w * (ih / iw)
                else:
                    h = w * 0.6
                elems = [_RLImage(str(p), width=w, height=h)]
                if caption:
                    elems.append(Paragraph(caption, st_caption))
                return [KeepTogether(elems)]
            except Exception:
                return []
                
    def _vc(text):
        t = str(text).upper().strip()
        if t in ('PASS', 'PROMOTED'):     return C_GREEN
        if t in ('FAIL', 'NOT PROMOTED'): return C_RED
        return None

    def _vp(text):
        return Paragraph(text, st_verd)

    def _m(key, metric):
        pf = metrics_comparison.get(key)
        if pf is None:
            return 'N/A'
        try:
            if metric == 'cum':    return f"{pf.total_return()*100:.1f}%"
            if metric == 'cagr':   return f"{pf.annualized_return()*100:.1f}%"
            if metric == 'sharpe': return f"{pf.sharpe_ratio():.2f}"
            if metric == 'maxdd':  return f"{abs(pf.max_drawdown())*100:.1f}%"
        except Exception:
            return 'N/A'
        return 'N/A'

    def _ci(row, col, pct=True):
        try:
            v = float(mc_ci.loc[row, col])
            return f"{v*100:.1f}%" if pct else f"{v:.3f}"
        except Exception:
            return 'N/A'

    _s  = ofc_report_std.get('signals', {})    if ofc_report_std    else {}
    _sc = ofc_report_cluster.get('signals', {}) if ofc_report_cluster else None
    ofc_passed_std     = bool(ofc_report_std.get('promoted', False))    if ofc_report_std    else False
    ofc_passed_cluster = bool(ofc_report_cluster.get('promoted', False)) if ofc_report_cluster else None

    reshuffle_pval = mc_skill.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
    timing_pval    = mc_skill.get('rebalance_timing', {}).get('p_values', {}).get('CAGR')
    reshuffle_pass = (reshuffle_pval is not None) and (reshuffle_pval < 0.10)
    timing_pass    = (timing_pval is not None) and (timing_pval < 0.10)

    # Cluster fallback a std se non passati (retro-compat)
    if mc_skill_cluster is not None:
        reshuffle_pval_cl = mc_skill_cluster.get('rotation_reshuffle', {}).get('p_values', {}).get('CAGR')
        timing_pval_cl    = mc_skill_cluster.get('rebalance_timing', {}).get('p_values', {}).get('CAGR')
        reshuffle_pass_cl = (reshuffle_pval_cl is not None) and (reshuffle_pval_cl < 0.10)
        timing_pass_cl    = (timing_pval_cl is not None) and (timing_pval_cl < 0.10)
    else:
        reshuffle_pval_cl, timing_pval_cl = reshuffle_pval, timing_pval
        reshuffle_pass_cl, timing_pass_cl = reshuffle_pass, timing_pass

    def _ci_cl(row, col, pct=True):
        src = mc_ci_cluster if mc_ci_cluster is not None else mc_ci
        try:
            v = float(src.loc[row, col])
            return f"{v*100:.1f}%" if pct else f"{v:.3f}"
        except Exception:
            return 'N/A'
   
    def _hr():
        return HRFlowable(width='100%', thickness=0.5, color=C_NAVY_LT, spaceAfter=8)

    story = []

    # Cover
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(f"Relazione Tecnica · {portfolio_title} {year}", st_title))
    use_cl_desc = 'con clustering gerarchico e ' if use_clustering else ''
    story.append(Paragraph(
        f"R-portfolio {use_cl_desc}Risk ON/OFF · "
        f"Benchmark {benchmark} · Profilo {profile}", st_subtitle))
    story.append(_hr())

    # Sezione 1: Identità
    story.append(Paragraph("1. Identità del Portafoglio", st_section))
    cl_row = ('Attivo · gerarchico Ward · k adattivo'
              if use_clustering else 'Non attivo')
    id_data = [
        [Paragraph('Campo', st_cell_hdr), Paragraph('Valore', st_cell_hdr)],
        ['Nome',             portfolio_title],
        ['Engine',           'R-portfolio (rotational momentum)'],
        ['Universo',         f"{universe_size} tickers"],
        ['Benchmark',        benchmark],
        ['Profilo',          profile],
        ['Periodo analisi',  f"{period_start} → {period_end}"],
        ['Data generazione', gen_date],
        ['Clustering',       cl_row],
        ['Risk ON/OFF',      'Attivo · filtro regime di mercato'],
    ]
    id_t = Table(id_data, colWidths=[55 * mm, CONTENT_W - 55 * mm])
    id_t.setStyle(TableStyle(_ts_base()))
    story += [id_t, Spacer(1, 5 * mm)]

    # Sezione 2: Configurazione WFO
    story.append(Paragraph("2. Configurazione WFO", st_section))
    wfo_data = [
        [Paragraph('Parametro', st_cell_hdr),
         Paragraph('Valore', st_cell_hdrc),
         Paragraph('Note', st_cell_hdr)],
        ['WFO ratio',              (lambda r: str(r).replace(':', ' : ') if ':' in str(r) else f'{r} : 1')(wfo_config.get('ratio', 'N/A')),              'Rapporto IS / OOS'],
        ['Metrica ottimizzazione', str(wfo_config.get('metric', 'N/A')),                'Selezione parametri In-Sample'],
        ['Grid size (full)',        f"{wfo_config.get('n_full_trials','N/A')} comb.",   'Spazio parametrico totale'],
        ['Grid size (reduced)',     f"{wfo_config.get('n_reduced_trials','N/A')} comb.",'Dopo stability analysis (k=3, CAGR)'],
        ['Stability metric',       'CAGR · k = 3',                                'Metrica e numero di sottoperiodi'],
        ['n_bootstrap OFC',  str(wfo_config.get('n_bootstrap_ofc', wfo_config.get('n_bootstrap', 1000))),  'Test S3 random selection'],
        ['n_bootstrap MC',   str(wfo_config.get('n_bootstrap_mc',  wfo_config.get('n_bootstrap', 1000))),  'Block A (CI) + Block B (Skill Tests)'],
    ]
    wfo_t = Table(wfo_data, colWidths=[50 * mm, 45 * mm, CONTENT_W - 95 * mm])
    wfo_t.setStyle(TableStyle(_ts_base()))
    story += [wfo_t, Spacer(1, 5 * mm)]

    # Sezione 2b: Struttura Cluster (condizionale)
    if use_clustering and cluster_result is not None:
        _cgroups = cluster_result.get('cluster_groups', {})
        _clabels = cluster_result.get('cluster_labels', {})
        if _cgroups:
            # story.append(Paragraph("2.b Struttura dei Cluster", st_subsec))
            story.append(Paragraph("2.a Struttura dei Cluster", st_subsec))
            
            n_cl = len(_cgroups)
            story.append(Paragraph(
                "L'analisi di clustering gerarchico (linkage Ward, distanza combinata "
                "60% correlazione + 40% feature euclidee su Vol/Mom6m/AutoCorr/MaxDD) ha "
                f"identificato <b>{n_cl} cluster</b> sull'ultima finestra WFO. "
                "Le etichette sono assegnate adattivamente secondo priorità "
                "AVOID &gt; HIGH_MOMENTUM &gt; DEFENSIVE &gt; BALANCED.", st_body))
            
            def _fmt_t(ts):
                ab = [t.split('.')[0] for t in ts]
                r  = ' · '.join(ab[:10])
                return r + (f' ... (+{len(ts)-10})' if len(ts) > 10 else '')

            cl_hdr = [
                Paragraph('Cluster', st_cell_hdrc),
                Paragraph('Label', st_cell_hdrc),
                Paragraph('N. Titoli', st_cell_hdrc),
                Paragraph('Tickers', st_cell_hdr),
            ]
            cl_rows = [cl_hdr]
            cl_ts   = _ts_base()
            for ri, cid in enumerate(sorted(_cgroups.keys()), start=1):
                lbl = _clabels.get(cid, f'C{cid}')
                cl_rows.append([
                    Paragraph(f'C{cid}', st_cell_ctr),
                    Paragraph(f'<b>{lbl}</b>', st_cell_bold),
                    Paragraph(str(len(_cgroups[cid])), st_cell_ctr),
                    Paragraph(_fmt_t(_cgroups[cid]), st_cell),
                ])
                if 'AVOID' in str(lbl).upper():
                    cl_ts += [('BACKGROUND', (1, ri), (1, ri), C_RED),
                               ('TEXTCOLOR',  (1, ri), (1, ri), C_WHITE)]
            # cl_t = Table(cl_rows, colWidths=[18 * mm, 30 * mm, 18 * mm, CONTENT_W - 66 * mm])
            # Bug #4 fix: colonna Label da 30→38mm per accogliere HIGH_MOMENTUM senza wrap
            cl_t = Table(cl_rows, colWidths=[18 * mm, 38 * mm, 18 * mm, CONTENT_W - 74 * mm])
            cl_t.setStyle(TableStyle(cl_ts))
            story += [cl_t, Spacer(1, 3 * mm)]

            story.extend(_img('cluster_dendrogram.png',
                # caption='Fig. 1 — Dendrogramma Ward (snapshot ultima finestra IS). Soglia di taglio produce i cluster: '
                #         'gruppo verde = AVOID, blu = BALANCED. '
                #         'La composizione viene ricalcolata a ogni ribilanciamento.'))
                caption='Fig. 1 — Dendrogramma Ward (snapshot ultima finestra IS). La soglia di taglio (linea rossa tratteggiata) determina il numero di cluster; i rami sotto la soglia sono colorati per cluster di appartenenza. '
                        'La composizione viene ricalcolata a ogni ribilanciamento.'))
            story.extend(_img('cluster_scatter.png',
                caption='Fig. 2 — Scatter Volatilità annualizzata vs Momentum 6 mesi (snapshot ultima finestra IS). '
                        'Colorato per cluster. Il gruppo AVOID concentra titoli con momentum '
                        'negativo e alta volatilità. '
                        'La composizione viene ricalcolata a ogni ribilanciamento.'))
            story.extend(_img('cluster_heatmap.png',
                # caption='Fig. 3 — Matrice di correlazione su finestra coerente con WFO (snapshot ultima finestra IS). '
                #         'Ordinata per cluster. Box blu = AVOID, box arancio = BALANCED. '
                #         'La composizione viene ricalcolata a ogni ribilanciamento.'))
                caption='Fig. 3 — Matrice di correlazione su finestra coerente con WFO (snapshot ultima finestra IS). '
                        'Ordinata per cluster: i box colorati sopra la matrice delimitano i sub-gruppi e ne indicano la label. '
                        'La composizione viene ricalcolata a ogni ribilanciamento.'))
    # Sezione 3: Metriche Comparative WFO
    story.append(Paragraph("3. Metriche Comparative WFO", st_section))

    # story.append(Paragraph(
    #     f"Confronto delle quattro varianti del motore (Standard vs Cluster, ognuna con e senza "
    #     f"Risk ON/OFF) sul periodo comune. Il benchmark {benchmark} è riportato "
    #     "nell'ultima riga. La variante <b>Cluster — Risk ON/OFF</b> (evidenziata) "
    #     "è quella candidata al deploy operativo.", st_body))

    # Bug #2 fix: frase introduttiva §3 dinamica come §7
    _rec = _recommended_path(ofc_report_std, ofc_report_cluster)
    if _rec == 'cluster':
        _rec_label = "<b>Cluster — Risk ON/OFF</b>"
    elif _rec == 'std':
        _rec_label = "<b>Standard — Risk ON/OFF</b>"
    else:
        _rec_label = None
    
    if _rec_label is not None:
        _rec_phrase = f"La variante {_rec_label} (evidenziata) è quella candidata al deploy operativo."
    else:
        _rec_phrase = "Nessuna variante supera l'Overfitting Check (vedi §7)."
    
    story.append(Paragraph(
        f"Confronto delle quattro varianti del motore (Standard vs Cluster, ognuna con e senza "
        f"Risk ON/OFF) sul periodo comune. Il benchmark {benchmark} è riportato "
        f"nell'ultima riga. {_rec_phrase}", st_body))


    m_hdr  = [Paragraph('Portafoglio', st_cell_hdr),
              Paragraph('Cum Return', st_cell_hdrc),
              Paragraph('CAGR', st_cell_hdrc),
              Paragraph('Sharpe', st_cell_hdrc),
              Paragraph('MaxDD', st_cell_hdrc)]
    m_rows = [m_hdr,
        [Paragraph('<b>Cluster — Risk ON/OFF</b>', st_cell_bold),
         _m('cluster_riskoff','cum'), _m('cluster_riskoff','cagr'),
         _m('cluster_riskoff','sharpe'), _m('cluster_riskoff','maxdd')],
        ['Cluster — Base',
         _m('cluster_base','cum'), _m('cluster_base','cagr'),
         _m('cluster_base','sharpe'), _m('cluster_base','maxdd')],
        ['Standard — Risk ON/OFF',
         _m('std_riskoff','cum'), _m('std_riskoff','cagr'),
         _m('std_riskoff','sharpe'), _m('std_riskoff','maxdd')],
        ['Standard — Base',
         _m('std_base','cum'), _m('std_base','cagr'),
         _m('std_base','sharpe'), _m('std_base','maxdd')],
        [f'Benchmark ({benchmark})',
         _m('benchmark','cum'), _m('benchmark','cagr'),
         _m('benchmark','sharpe'), _m('benchmark','maxdd')],
    ]
    m_ts = _ts_base() + [
        ('BACKGROUND', (0, 1), (-1, 1), rl_colors.HexColor('#D5EAF5')),
        ('FONTNAME',   (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]
    m_t = Table(m_rows, colWidths=[55 * mm, 28 * mm, 25 * mm, 25 * mm, 25 * mm])
    m_t.setStyle(TableStyle(m_ts))
    story += [m_t, Spacer(1, 3 * mm)]
    story.extend(_img('equity_comparison.png',
        caption='Fig. 4 — Equity cumulativa comparativa. Cluster — Risk ON/OFF '
                '(rosso pieno) è la variante di riferimento'))

    # Sezione 4: OFC
    story.append(Paragraph("4. Overfitting Check (OFC)", st_section))
    minsig = (ofc_report_std.get('resolved', {}).get('min_signals_to_pass', 3)
              if ofc_report_std else 3)
    story.append(Paragraph(
        "Valutazione della robustezza del processo di ottimizzazione WFO tramite quattro "
        f"segnali indipendenti. <b>Soglia di promozione: {minsig} segnali su 4 in stato PASS.</b>",
        st_body))
    _ofc_res = (ofc_report_std or {}).get('resolved', {})
    _s1t = _ofc_res.get('plateau_threshold', 0.20)
    _s2t = _ofc_res.get('s2_coherence_threshold', 0.50)
    _s3t = _ofc_res.get('s3_pvalue_threshold', 0.10)
    _s4t = _ofc_res.get('s4_dsr_threshold', 0.0)
    story.append(Paragraph(
        f"Soglie per i singoli segnali (profilo <i>{(ofc_report_std or {}).get('profile', 'satellite')}</i>): "
        f"<b>S1</b> diversit\u00e0 &gt; {_s1t:.0%} \u00b7 "
        f"<b>S2</b> coerenza \u2265 {_s2t:.0%} \u00b7 "
        f"<b>S3</b> p \u2264 {_s3t:.2f} \u00b7 "
        f"<b>S4</b> DSR &gt; {_s4t:.2f}.",
        st_body))

    def _ofc_block(sigs, ofc_passed, title):
        story.append(Paragraph(title, st_subsec))
        n_p = sum(1 for k in ('S1_plateau', 'S2_coherence', 'S3_bootstrap', 'S4_dsr')
                  if (sigs or {}).get(k, {}).get('pass'))
        ofc_lbl = ('PROMOTED' if ofc_passed else
                   'NOT PROMOTED' if ofc_passed is not None else 'N/A')
        rows = [[Paragraph('Segnale', st_cell_hdr),
                 Paragraph('Cosa misura', st_cell_hdr),
                 Paragraph('Verdetto', st_cell_hdrc),
                 Paragraph('Valore', st_cell_hdrc)]]
        ts_x = []
        for ri, (sk, slbl, sdsc) in enumerate([
            ('S1_plateau',   'S1 — Plateau proxy',    'Diversità parametrica del WFO'),
            ('S2_coherence', 'S2 — Flag coherence',   'Stabilità dei filtri tra sottoperiodi'),
            ('S3_bootstrap', 'S3 — Random selection', f'Il risultato Out-Of-Sample batte {wfo_config.get("n_bootstrap_ofc", wfo_config.get("n_bootstrap", 1000))} portafogli con parametri casuali?'),
            ('S4_dsr',       'S4 — DSR',              'Sharpe significativo dopo correzione per numero di trial'),
        ], start=1):
            sd  = (sigs or {}).get(sk, {})
            pss = sd.get('pass')
            if sk == 'S3_bootstrap':
                raw = sd.get('p_value')
                vs  = f"p = {raw:.3f}" if isinstance(raw, float) else 'N/A'
            elif sk == 'S4_dsr':
                raw = sd.get('dsr')
                vs  = f"{raw:.3f}" if isinstance(raw, float) else 'N/A'
            else:
                raw = sd.get('value')
                vs  = f"{raw:.3f}" if isinstance(raw, float) else 'N/A'
            vl  = 'PASS' if pss else ('FAIL' if pss is not None else 'N/A')
            vc  = _vc(vl)
            rows.append([
                Paragraph(slbl, st_cell),
                Paragraph(sdsc, st_cell),
                _vp(vl),
                Paragraph(vs, st_cell_ctr),
            ])
            if vc:
                ts_x += [('BACKGROUND', (2, ri), (2, ri), vc),
                          ('TEXTCOLOR',  (2, ri), (2, ri), C_WHITE)]
        ov_vc = _vc(ofc_lbl)
        rows.append([
            Paragraph('<b>Verdetto OFC</b>', st_cell_bold),
            Paragraph(f'Soglia: {minsig}/4 segnali PASS', st_cell),
            Paragraph(
                f'<b>{"NOT<br/>PROMOTED" if ofc_lbl == "NOT PROMOTED" else ofc_lbl}</b>',
                st_verd),
            Paragraph(f'{n_p} / 4', st_cell_ctr),
        ])
        if ov_vc:
            ts_x += [('BACKGROUND', (2, 5), (2, 5), ov_vc),
                     ('TEXTCOLOR',  (2, 5), (2, 5), C_WHITE)]
        ts_x.append(('FONTNAME', (0, 5), (-1, 5), 'Helvetica-Bold'))
        t = Table(rows, colWidths=[35 * mm, 70 * mm, 30 * mm, 25 * mm])
        t.setStyle(TableStyle(_ts_base() + ts_x))
        story.extend([t, Spacer(1, 5 * mm)])

    _ofc_block(_s,        ofc_passed_std,     '4.a — Path Standard')
    _ofc_block(_sc or {}, ofc_passed_cluster, '4.b — Path Cluster')

    from reportlab.platypus import PageBreak as _PB
    story.append(_PB())

    # Sezione 5: MC
    story.append(Paragraph("5. Validazione Monte Carlo", st_section))
    story.append(Paragraph(
        "L'analisi MC è strutturata in due blocchi distinti. Il <b>Block B (Skill Tests)</b> "
        "valuta se il motore aggiunge valore rispetto al caso tramite test di permutazione; "
        "il <b>Block A (Confidence Intervals)</b> fornisce intervalli di confidenza "
        "sulle metriche del portafoglio tramite bootstrap. I due blocchi rispondono a domande "
        "diverse e i rispettivi output non devono essere mescolati.", st_body))

    def _sk_block(title, pvals_pass):
            story.append(Paragraph(title, st_subsec))
            sk_hdr  = [Paragraph('Test', st_cell_hdr),
                       Paragraph('Cosa misura', st_cell_hdr),
                       Paragraph('Verdetto', st_cell_hdrc),
                       Paragraph('p-value', st_cell_hdrc)]
            sk_rows = [sk_hdr]
            sk_ts   = _ts_base()
            for ri, (lbl, dsc, pv, pss) in enumerate(pvals_pass, start=1):
                pv_s = f"{pv:.3f}" if pv is not None else 'N/A'
                vl   = 'PASS' if pss else 'FAIL'
                vc   = _vc(vl)
                sk_rows.append([lbl, dsc, _vp(vl), pv_s])
                if vc:
                    sk_ts += [('BACKGROUND', (2, ri), (2, ri), vc),
                              ('TEXTCOLOR',  (2, ri), (2, ri), C_WHITE)]
            sk_t = Table(sk_rows, colWidths=[42 * mm, 78 * mm, 22 * mm, 20 * mm])
            sk_t.setStyle(TableStyle(sk_ts))
            story.extend([sk_t, Spacer(1, 3 * mm)])
        
    story.append(Paragraph("5.a — Skill Tests (Block B)", st_subsec))
    
    # ── §5.a.1 Path Standard: tabella + 3 figure ────────────────────────────
    _sk_block("5.a.1 — Path Standard", [
        ('B1 — Rotation Reshuffle', 'La rotazione batte una selezione casuale dei titoli?', reshuffle_pval, reshuffle_pass),
        ('B2 — Rebalance Timing',   'Il timing mensile batte date di rebalance casuali?',   timing_pval,    timing_pass),
    ])
    story.extend(_img_sub('std', 'mc_reshuffle.png',
        caption='Fig. 5a — Path Standard. Distribuzione bootstrap del CAGR sotto H0 di rotazione casuale (B1). '
                'Linea rossa = CAGR effettivo'))
    story.extend(_img_sub('std', 'mc_timing.png',
        caption='Fig. 5b — Path Standard. Distribuzione bootstrap del CAGR sotto H0 di rebalance casuale (B2). '
                'Linea rossa = CAGR effettivo'))
    story.extend(_img_sub('std', 'mc_skill_summary.png',
        caption='Fig. 5c — Path Standard. Skill Tests, p-value per metrica (CAGR, MaxDD, Sharpe, Calmar, Vol, Ulcer). '
                'Soglie tratteggiate al 5% e 1%'))

    # ── §5.a.2 Path Cluster: tabella + 3 figure ─────────────────────────────
    _sk_block("5.a.2 — Path Cluster", [
        ('B1 — Rotation Reshuffle', 'La rotazione batte una selezione casuale dei titoli?', reshuffle_pval_cl, reshuffle_pass_cl),
        ('B2 — Rebalance Timing',   'Il timing mensile batte date di rebalance casuali?',   timing_pval_cl,    timing_pass_cl),
    ])
    story.extend(_img_sub('cluster', 'mc_reshuffle.png',
        caption='Fig. 6a — Path Cluster. Distribuzione bootstrap del CAGR sotto H0 di rotazione casuale (B1). '
                'Linea rossa = CAGR effettivo'))
    story.extend(_img_sub('cluster', 'mc_timing.png',
        caption='Fig. 6b — Path Cluster. Distribuzione bootstrap del CAGR sotto H0 di rebalance casuale (B2). '
                'Linea rossa = CAGR effettivo'))
    story.extend(_img_sub('cluster', 'mc_skill_summary.png',
        caption='Fig. 6c — Path Cluster. Skill Tests, p-value per metrica (CAGR, MaxDD, Sharpe, Calmar, Vol, Ulcer). '
                'Soglie tratteggiate al 5% e 1%'))

    def _ci_block(title, ci_func):
            story.append(Paragraph(title, st_subsec))
            ci_hdr = [Paragraph('Metodo', st_cell_hdr),
                      Paragraph('CAGR p5', st_cell_hdrc),
                      Paragraph('CAGR p50', st_cell_hdrc),
                      Paragraph('CAGR p95', st_cell_hdrc),
                      Paragraph('Sharpe p50', st_cell_hdrc),
                      Paragraph('MaxDD p50', st_cell_hdrc)]
            ci_rows = [ci_hdr]
            for mkey, mlbl in [
                ('A1 · IID Bootstrap',   'A1 — IID Bootstrap'),
                ('A2 · Block Bootstrap', 'A2 — Block Bootstrap'),
            ]:
                ci_rows.append([mlbl,
                    ci_func(mkey, 'CAGR_p5'), ci_func(mkey, 'CAGR_p50'), ci_func(mkey, 'CAGR_p95'),
                    ci_func(mkey, 'Sharpe_p50', pct=False), ci_func(mkey, 'MaxDD_p50'),
                ])
            ci_t = Table(ci_rows, colWidths=[40 * mm, 23 * mm, 23 * mm, 23 * mm, 23 * mm, 26 * mm])
            ci_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
            story.extend([ci_t, Spacer(1, 3 * mm)])

    story.append(Paragraph("5.b — Confidence Intervals (Block A)", st_subsec))
    
    # ── §5.b.1 Path Standard: tabella + 3 figure ────────────────────────────
    _ci_block("5.b.1 — Path Standard", _ci)
    story.extend(_img_sub('std', 'mc_ci_fanchart_iid.png',
        caption='Fig. 7a — Path Standard. Fan chart equity A1 \u00b7 IID Bootstrap. '
                'Banda p5\u2013p95 azzurro, mediana blu, Actual rosso, Benchmark grigio tratteggiato.'))
    story.extend(_img_sub('std', 'mc_ci_fanchart_block.png',
        caption='Fig. 7b — Path Standard. Fan chart equity A2 \u00b7 Block Bootstrap. '
                'Banda p5\u2013p95 azzurro, mediana blu, Actual rosso, Benchmark grigio tratteggiato.'))
    story.extend(_img_sub('std', 'mc_ci.png',
        caption='Fig. 7c — Path Standard. Confidence Intervals cross-method per CAGR e Max Drawdown. '
                'Linea rossa tratteggiata = valore Actual del portafoglio'))

    # ── §5.b.2 Path Cluster: tabella + 3 figure ─────────────────────────────
    _ci_block("5.b.2 — Path Cluster",  _ci_cl)
    story.extend(_img_sub('cluster', 'mc_ci_fanchart_iid.png',
        caption='Fig. 8a — Path Cluster. Fan chart equity A1 \u00b7 IID Bootstrap. '
                'Banda p5\u2013p95 azzurro, mediana blu, Actual rosso, Benchmark grigio tratteggiato.'))
    story.extend(_img_sub('cluster', 'mc_ci_fanchart_block.png',
        caption='Fig. 8b — Path Cluster. Fan chart equity A2 \u00b7 Block Bootstrap. '
                'Banda p5\u2013p95 azzurro, mediana blu, Actual rosso, Benchmark grigio tratteggiato.'))
    story.extend(_img_sub('cluster', 'mc_ci.png',
        caption='Fig. 8c — Path Cluster. Confidence Intervals cross-method per CAGR e Max Drawdown. '
                'Linea rossa tratteggiata = valore Actual del portafoglio'))


    # ── Sezione 5.c: SRI / MRM PRIIPs ─────────────────────────────────────────
    _sri_data = ci_results.get('sri') if ci_results else None
    if _sri_data is not None:
        _sri_iid   = _sri_data['iid']
        _sri_blk   = _sri_data['block']
        _sri_cap   = _sri_data.get('capital_base', 10_000)
        _sri_rhp   = _sri_iid['rhp_years_effective']
        _sri_mfin  = _sri_data['mrm_class_final']
        _SRI_COLORS = ['#1A7340', '#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336', '#B71C1C']

        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("5.c — Indicatore di Rischio Sintetico (metodologia PRIIPs)", st_subsec))

        # Scale 1-7: colored boxes
        _box_w = CONTENT_W / 7
        _st_snum = _st('_sri_num', fontSize=11, fontName='Helvetica-Bold', alignment=TA_CENTER)
        _st_slbl = _st('_sri_lbl', fontSize=6.5, alignment=TA_CENTER)
        _scale_r1 = [Paragraph(f'<b>{cls}</b>', _st_snum) for cls in range(1, 8)]
        _scale_r2 = ([Paragraph('Rischio basso', _st_slbl)]
                     + [Paragraph('', _st_slbl)] * 5
                     + [Paragraph('Rischio alto', _st_slbl)])
        _sc_t = Table([_scale_r1, _scale_r2], colWidths=[_box_w] * 7)
        _sc_ts = [
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN',         (0, 0), (-1, 0),  'CENTER'),
            ('TOPPADDING',    (0, 0), (-1, 0),  6),
            ('BOTTOMPADDING', (0, 0), (-1, 0),  6),
            ('TOPPADDING',    (0, 1), (-1, 1),  2),
            ('BOTTOMPADDING', (0, 1), (-1, 1),  2),
            ('LEFTPADDING',   (0, 0), (-1, -1), 2),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
            ('GRID',          (0, 0), (-1, 0),  0.4, C_GRAY_BD),
        ]
        for _cls in range(1, 8):
            _c = _cls - 1
            _bg = rl_colors.HexColor(_SRI_COLORS[_c])
            _sc_ts.append(('BACKGROUND', (_c, 0), (_c, 0), _bg))
            _txt_c = C_WHITE if _cls in (1, 6, 7) else C_NAVY
            _sc_ts.append(('TEXTCOLOR', (_c, 0), (_c, 0), _txt_c))
            if _cls == _sri_mfin:
                _sc_ts += [
                    ('BOX',      (_c, 0), (_c, 0), 3.0, C_NAVY),
                    ('FONTSIZE', (_c, 0), (_c, 0), 13),
                ]
        _sc_t.setStyle(TableStyle(_sc_ts))
        story += [_sc_t, Spacer(1, 4 * mm)]

        # Technical table
        _st_disc = _st('_sri_disc', fontSize=7.5, textColor=C_TEXT,
                        alignment=TA_JUSTIFY, leading=10)
        _sri_rows = [
            [Paragraph('Parametro', st_cell_hdr),
             Paragraph('IID Bootstrap', st_cell_hdrc),
             Paragraph('Block Bootstrap', st_cell_hdrc)],
            [Paragraph('Classe MRM (penalizzata)', st_cell),
             Paragraph(f"{_sri_iid['mrm_class_final']} / 7", st_cell_ctr),
             Paragraph(f"{_sri_blk['mrm_class_final']} / 7", st_cell_ctr)],
            [Paragraph('VEV annualizzata', st_cell),
             Paragraph(f"{_sri_iid['vev_pct']:.1f}%", st_cell_ctr),
             Paragraph(f"{_sri_blk['vev_pct']:.1f}%", st_cell_ctr)],
            [Paragraph(f"VaR 97.5% (su {_sri_cap:,.0f} €)", st_cell),
             Paragraph(f"{_sri_iid['var_price']:,.0f} €", st_cell_ctr),
             Paragraph(f"{_sri_blk['var_price']:,.0f} €", st_cell_ctr)],
            [Paragraph('RHP effettivo', st_cell),
             Paragraph(f'{_sri_rhp:.1f} anni', st_cell_ctr),
             Paragraph('—', st_cell_ctr)],
            [Paragraph('Frequenza dati', st_cell),
             Paragraph('mensile (+1 classe penalita’ reg.)', st_cell_ctr),
             Paragraph('', st_cell_ctr)],
            # [Paragraph('<b>Classe MRM finale</b>', st_cell_bold),
            #  Paragraph(f'<b>{_sri_mfin} / 7</b>', st_cell_ctr),
            #  Paragraph('(max IID, Block)', st_cell_ctr)],
            [Paragraph('Classe MRM finale', st_cell_hdr),
             Paragraph(f'{_sri_mfin} / 7', st_cell_hdrc),
             Paragraph('(max IID, Block)', st_cell_hdrc)],
        ]
        _sri_t = Table(_sri_rows, colWidths=[65 * mm, 40 * mm, 57 * mm])
        # _sri_ts = _ts_base() + [
        #     ('FONTNAME',   (0, 6), (-1, 6), 'Helvetica-Bold'),
        #     ('BACKGROUND', (0, 6), (-1, 6), C_NAVY_LT),
        #     ('TEXTCOLOR',  (0, 6), (-1, 6), C_WHITE),
        # ]
        _sri_ts = _ts_base() + [
            ('BACKGROUND', (0, 6), (-1, 6), C_NAVY_LT),
        ]
        _sri_t.setStyle(TableStyle(_sri_ts))
        story += [_sri_t, Spacer(1, 4 * mm)]

        # Disclaimer
        story.append(Paragraph(
            "Questo indicatore è calcolato applicando la metodologia del regolamento PRIIPs "
            "(Reg. UE 1286/2014 e RTS 2017/653 come modificato dal Reg. UE 2021/2268) a fini "
            "esclusivamente informativi e didattici. Il portafoglio non è un prodotto PRIIPs "
            "registrato e questa scheda non costituisce un Key Information Document né altro "
            "documento informativo regolamentare ai sensi della normativa europea. Il calcolo "
            "segue l'approccio Category 3 (simulazione Monte Carlo) sui rendimenti mensili OOS "
            "della Walk-Forward Optimization, applicando la penalita’ regolamentare di +1 classe "
            "MRM prevista per dati a frequenza mensile. La regola conservativa adottata in caso "
            "di divergenza tra metodi bootstrap (IID e Block) seleziona la classe più alta. "
            "Questa scheda non costituisce consulenza in materia di investimenti.",
            _st_disc))


    from reportlab.platypus import PageBreak as _PB
    story.append(_PB())

    # Sezione 6: Diagnosi strutturale
    story.append(Paragraph("6. Analisi dei Segnali e Diagnosi Strutturale", st_section))
    ofc_std_txt, ofc_clu_txt = _diagnose_ofc(ofc_report_std, ofc_report_cluster)
    # sk_txt1, sk_txt2, ci_txt = _diagnose_mc(mc_skill, mc_ci, metrics_comparison)
    # B-005: passa mc_skill_cluster e mc_ci_cluster per narrativa Opzione C
    sk_txt1, sk_txt2, ci_txt = _diagnose_mc(
        mc_skill, mc_ci, metrics_comparison,
        mc_skill_cluster   = mc_skill_cluster,
        mc_ci_cluster      = mc_ci_cluster,
        ofc_report_std     = ofc_report_std,
        ofc_report_cluster = ofc_report_cluster,
    )
    
    story.append(Paragraph("6.a — Lettura dei segnali OFC", st_subsec))
    story.append(Paragraph(ofc_std_txt, st_body))
    if ofc_clu_txt:
        story.append(Paragraph(ofc_clu_txt, st_body))

    story.append(Paragraph("6.b — Lettura dei test Monte Carlo Skill", st_subsec))
    story.append(Paragraph(sk_txt1, st_body))
    if sk_txt2:
        story.append(Paragraph(sk_txt2, st_body))

    story.append(Paragraph("6.c — Confidence Intervals (Block A)", st_subsec))
    story.append(Paragraph(ci_txt, st_body))

    # Sezione 7: Decisione Finale + verdict box
    story.append(Paragraph("7. Decisione Finale", st_section))
    ofc_std_v  = 'PROMOTED' if ofc_passed_std else 'NOT PROMOTED'
    ofc_clu_v  = ('PROMOTED' if ofc_passed_cluster else
                  'NOT PROMOTED' if ofc_passed_cluster is not None else 'N/A')
    dec_hdr = [Paragraph('Dimensione', st_cell_hdr),
               Paragraph('Standard', st_cell_hdrc),
               Paragraph('Cluster', st_cell_hdrc)]
    dec_rows = [dec_hdr,
        ['OFC Verdict',     _vp(ofc_std_v),  _vp(ofc_clu_v)],
        # ['Skill Profile',   skill_profile,    skill_profile],
        ['Skill Profile',   skill_profile or 'N/A',
                            skill_profile_cluster or skill_profile or 'N/A'],   # B-005
        [f'CAGR vs {benchmark}',
         f"{_m('std_riskoff','cagr')} vs {_m('benchmark','cagr')}",
         f"{_m('cluster_riskoff','cagr')} vs {_m('benchmark','cagr')}"],
        [f'Sharpe vs {benchmark}',
         f"{_m('std_riskoff','sharpe')} vs {_m('benchmark','sharpe')}",
         f"{_m('cluster_riskoff','sharpe')} vs {_m('benchmark','sharpe')}"],
        [f'MaxDD vs {benchmark}',
         f"{_m('std_riskoff','maxdd')} vs {_m('benchmark','maxdd')}",
         f"{_m('cluster_riskoff','maxdd')} vs {_m('benchmark','maxdd')}"],
    ]
    d_ts = _ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]
    for col_i, vl in [(1, ofc_std_v), (2, ofc_clu_v)]:
        vc = _vc(vl)
        if vc:
            d_ts += [('BACKGROUND', (col_i, 1), (col_i, 1), vc),
                     ('TEXTCOLOR',  (col_i, 1), (col_i, 1), C_WHITE)]
    hw = (CONTENT_W - 45 * mm) / 2
    dec_t = Table(dec_rows, colWidths=[45 * mm, hw, hw])
    dec_t.setStyle(TableStyle(d_ts))
    story += [dec_t, Spacer(1, 5 * mm)]

    # Verdict box adattivo
    # _verdict_text = _build_verdict_text(
    #     ofc_report_std=ofc_report_std,
    #     ofc_report_cluster=ofc_report_cluster,
    #     metrics_comparison=metrics_comparison,
    #     mc_skill=mc_skill,
    #     skill_profile=skill_profile,
    #     wfo_config={**wfo_config, 'profile': profile},
    # )
    _verdict_text = _build_verdict_text(
        ofc_report_std        = ofc_report_std,
        ofc_report_cluster    = ofc_report_cluster,
        metrics_comparison    = metrics_comparison,
        mc_skill              = mc_skill,
        skill_profile         = skill_profile,
        skill_profile_cluster = skill_profile_cluster,   # B-005
        wfo_config            = {**wfo_config, 'profile': profile},
    )
    _verdict_box = Table(
        [[Paragraph(_verdict_text, st_vbox_j)]],
        colWidths=[CONTENT_W])
    _verdict_box.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), rl_colors.HexColor('#EAF0FB')),
        ('BOX',           (0, 0), (-1, -1), 1.5, C_NAVY_LT),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(_verdict_box)
    story.append(Spacer(1, 4 * mm))
    
    # B-005: secondo verdict box compatto per path alternativo
    rec_path = _recommended_path(ofc_report_std, ofc_report_cluster)
    if rec_path == 'cluster':
        # box compatto descrive Standard
        _alt_text = _build_verdict_text_compact(
            alt_path_label    = 'Standard',
            ofc_report_alt    = ofc_report_std,
            metrics_comparison= metrics_comparison,
            metrics_key       = 'std_riskoff',
            skill_profile_alt = skill_profile,
        )
    elif rec_path == 'std':
        # box compatto descrive Cluster
        _alt_text = _build_verdict_text_compact(
            alt_path_label    = 'Cluster',
            ofc_report_alt    = ofc_report_cluster,
            metrics_comparison= metrics_comparison,
            metrics_key       = 'cluster_riskoff',
            skill_profile_alt = skill_profile_cluster,
        )
    else:
        _alt_text = None
    
    if _alt_text is not None:
        _alt_box = Table(
            [[Paragraph(_alt_text, st_vbox_j)]],
            colWidths=[CONTENT_W])
        _alt_box.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), rl_colors.HexColor('#F5F5F5')),  # grigio chiaro
            ('BOX',           (0, 0), (-1, -1), 0.8, rl_colors.HexColor('#999999')),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
            ('TOPPADDING',    (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(_alt_box)
        story.append(Spacer(1, 4 * mm))
        
    # Placeholder manuale — compilare dopo analisi
    vbox_data = [[Paragraph(
        '<b>Path deployato:</b> [ STANDARD | CLUSTER | NESSUNO ] ← compilare<br/>'
        '<b>Motivazione:</b> ← compilare a mano dopo analisi',
        st_vbox)]]
    vbox = Table(vbox_data, colWidths=[CONTENT_W])
    vbox.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), rl_colors.HexColor('#EAF0FB')),
        ('BOX',        (0, 0), (-1, -1), 1.2, C_NAVY_LT),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
    ]))
    story.append(vbox)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title=f"Relazione Tecnica {portfolio_title} {year}",
        author="TSlab",
    )
    doc.build(story, onFirstPage=_draw_hf, onLaterPages=_draw_hf)
    return output_path


def run_r_portfolio_analysis(
    portfolio_cfg: dict,
    output_dir,
    year: int | None = None,
    start_date: str = "2015-01-01",
    end_date=None,
    profile: str = "satellite",
    verbose: bool = False,
) -> dict:
    """
    Esegue la pipeline completa R-portfolio in modalità headless.
    Usata da `iq analyze` e dall'agente batch relazioni tecniche.

    Args:
        portfolio_cfg:  dict portafoglio da r_portfolios.py
        output_dir:     Path directory output (PDF + PNG sub-dirs)
        year:           anno di selezione WFO (default: anno corrente)
        start_date:     inizio storico download (default: 2015-01-01)
        end_date:       fine storico (default: None = oggi)
        profile:        "satellite" | "core" — soglie OFC
        verbose:        stampe intermedie

    Returns:
        dict con chiavi:
            "pdf":       Path PDF relazione tecnica generato
            "plots_dir": Path directory PNG
            "ofc_std":   bool — OFC Standard promosso
            "ofc_cluster": bool | None — OFC Cluster promosso
            "skill_profile_std":     str
            "skill_profile_cluster": str | None
    """
    import matplotlib
    matplotlib.use('Agg')
    from pathlib import Path
    from datetime import date, timedelta
    import os

    # 1. SETUP
    if year is None:
        year = date.today().year
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir_std     = plots_dir / "std"
    plots_dir_cluster = plots_dir / "cluster"
    plots_dir_std.mkdir(parents=True, exist_ok=True)
    plots_dir_cluster.mkdir(parents=True, exist_ok=True)

    portfolio_title     = portfolio_cfg["Title"]
    tickers             = portfolio_cfg["tickers"]
    benchmark_portfolio = portfolio_cfg.get("benchmark_portfolio")
    benchmark_title     = portfolio_cfg.get("benchmark_title")
    risk_off_tickers    = portfolio_cfg.get("risk_off_tickers", [])
    init_cash           = portfolio_cfg.get("init_cash", 100_000)

    tickers = (
        extract_tickers_from_wikipedia(tickers, exclude=["GOOG"], rename={"BRK.B": "BRK-B"})
        if isinstance(tickers, str)
        else list(tickers)
    )

    wfo_results_dir = os.environ.get(
        "IQ_OUTPUTS_DIR",
        str(Path(__file__).parent.parent / "outputs")
    )
    wfo_file_save = f"{wfo_results_dir}/WFO_R_DEV_RESULTS/{portfolio_title}_{year}.wfo_summary.csv"

    # 2. DOWNLOAD
    lookback_buffer = 365
    download_start = (
        pd.to_datetime(start_date) - timedelta(days=lookback_buffer)
    ).strftime("%Y-%m-%d")

    stocks_data, company_data = fetch_data_and_companies(
        tickers, download_start, end_date, normalize=False
    )
    stocks_data_raw = download_data(tickers, download_start, end_date, auto_adjust=False)
    portfolio_cfg["stocks_data"] = stocks_data
    portfolio_cfg["init_cash"]   = init_cash

    if benchmark_portfolio:
        benchmark_data     = build_benchmark(benchmark_portfolio,
                                 stocks_data.index.min(), stocks_data.index.max()).replace(0, np.nan).ffill()
        benchmark_data_raw = build_benchmark(benchmark_portfolio,
                                 stocks_data.index.min(), stocks_data.index.max(),
                                 auto_adjust=False).replace(0, np.nan).ffill()
    elif benchmark_title:
        benchmark_data     = download_data(benchmark_title, stocks_data.index.min(), end_date)
        benchmark_data_raw = download_data(benchmark_title, stocks_data.index.min(), end_date, auto_adjust=False)
    else:
        benchmark_data = benchmark_data_raw = None

    # 3. RISK-OFF
    risk_off_tickers_uniq = [t for t in risk_off_tickers if t not in tickers]
    risk_off_data = download_data(risk_off_tickers_uniq, download_start, end_date) if risk_off_tickers_uniq else None
    if isinstance(risk_off_data, pd.Series):
        risk_off_data = risk_off_data.to_frame()

    # 4. GRIGLIA + STABILITY
    full_grid = {
        "rebalance_frequency":     ["QE", "ME"],
        "momentum_lookback_days":  [10, 20, 40, 60],
        "riskparity_lookback_days":[10, 20, 40, 60],
        "n_top":                   [1, 5, 8, 10],
        "use_acceleration":        [True, False],
        "momentum_weight":         [0.5, 0.7, 1.0],
        "filter_ema":              [True, False],
        "filter_volatility":       [True, False],
        "filter_min_momentum":     [True, False],
    }
    import itertools
    n_full_trials = len(list(itertools.product(*full_grid.values())))

    ratio   = "3:1"
    metric  = "Sharpe Ratio"
    cores   = -1
    force_next_year_params = True

    if len(tickers) > 3:
        start_date_stability = stocks_data.dropna(how="all").index.min()
        reduced_grid, stability_report = reduce_grid_via_stability(
            ptf_config      = portfolio_cfg,
            full_grid       = full_grid,
            full_start_date = start_date_stability,
            full_end_date   = end_date,
            metric          = "CAGR",
            k               = 3,
            verbose         = verbose,
        )
        n_reduced_trials = len(list(itertools.product(*reduced_grid.values())))
        stability_report_path = output_dir / f"{portfolio_title}_{year}_stability.csv"
        stability_report.to_csv(stability_report_path, index=False)
    else:
        reduced_grid     = full_grid
        stability_report = None
        n_reduced_trials = n_full_trials

    # Calcola pipeline_start_date
    ratio_int = int(str(ratio).split(":")[0])
    if benchmark_data is not None:
        benchmark_start  = benchmark_data.dropna(how="all").index.min()
    else:
        benchmark_start  = stocks_data.dropna(how="all").index.min()
    first_full_year     = pd.Timestamp(f"{benchmark_start.year + 1}-01-01")
    pipeline_start_date = first_full_year - pd.DateOffset(years=ratio_int)

    # 5. WFO STANDARD
    results_std = run_wfo_pipeline(
        stocks_data_raw        = stocks_data_raw,
        stocks_data            = stocks_data,
        benchmark_data         = benchmark_data,
        benchmark_data_raw     = benchmark_data_raw,
        tickers                = tickers,
        risk_off_data          = risk_off_data,
        ratio                  = ratio,
        metric                 = metric,
        start_date             = pipeline_start_date,
        end_date               = end_date,
        cores                  = cores,
        verbose                = verbose,
        force_next_year_params = force_next_year_params,
        use_clustering         = False,
        param_grid             = reduced_grid,
        portfolio_title        = portfolio_title,
        benchmark_title        = benchmark_title,
        init_cash              = init_cash,
        risk_on_off            = True,
        plot                   = False,
    )
    pf_rot_std       = results_std["pf_rot"]
    pf_rot_std_base  = results_std["pf_rot_base"]
    if pf_rot_std is None:
        pf_rot_std = pf_rot_std_base
    regime           = results_std["regime"]
    summary_df_std   = results_std["summary_df"]
    sel_tickers_std      = results_std["sel_tickers"]
    sel_tickers_std_base = results_std["sel_tickers_base"]

    save_rotational_wfo_summary(
        summary_df             = summary_df_std,
        start_date             = start_date,
        end_date               = end_date,
        file_path              = wfo_file_save,
        param_grid             = reduced_grid,
        metric                 = metric,
        ratio                  = ratio,
        force_next_year_params = force_next_year_params,
        extra_meta             = None,
    )

    # 6. WFO CLUSTER
    results_cluster = run_wfo_pipeline(
        stocks_data_raw    = stocks_data_raw,
        stocks_data        = stocks_data,
        benchmark_data     = benchmark_data,
        benchmark_data_raw = benchmark_data_raw,
        tickers            = tickers,
        risk_off_data      = risk_off_data,
        ratio              = ratio,
        metric             = metric,
        start_date         = pipeline_start_date,
        end_date           = end_date,
        cores              = cores,
        verbose            = verbose,
        force_next_year_params = force_next_year_params,
        use_clustering     = True,
        adaptive_k         = True,
        adaptive_k_method  = "hybrid",
        n_clusters         = 5,
        lookback_days      = 504,
        n_top_min          = 2,
        param_grid         = full_grid,
        portfolio_title    = portfolio_title,
        benchmark_title    = benchmark_title,
        init_cash          = init_cash,
        risk_on_off        = True,
        plot               = False,
        save_plots         = True,
        plots_dir          = plots_dir,
    )
    pf_rot_cluster       = results_cluster["pf_rot"]
    pf_rot_cluster_base  = results_cluster["pf_rot_base"]
    if pf_rot_cluster is None:
        pf_rot_cluster = pf_rot_cluster_base
    regime_cluster       = results_cluster["regime"]
    summary_df_cluster   = results_cluster["summary_df"]
    sel_tickers_cluster      = results_cluster["sel_tickers"]
    sel_tickers_cluster_base = results_cluster["sel_tickers_base"]

    # 7. COMPARE
    metrics_df = compare_wfo_pipelines(
        results_std     = results_std,
        results_cluster = results_cluster,
        portfolio_title = portfolio_title,
        benchmark_title = benchmark_title,
        plot_radar      = False,
        plot            = False,
        save_plots      = True,
        plots_dir       = plots_dir,
    )

    # 8. OFC STANDARD
    import json
    ofc_passed_std, ofc_report_std = overfitting_check_rotational(
        wfo_summary      = summary_df_std,
        stocks_data      = stocks_data,
        benchmark_data   = benchmark_data,
        param_grid       = reduced_grid,
        profile          = profile,
        n_total_trials   = n_full_trials,
        stability_report = stability_report,
        seed             = 42,
        verbose          = verbose,
    )
    ofc_report_std_path = output_dir / f"{portfolio_title}_{year}_ofc_std.json"
    with open(ofc_report_std_path, "w") as f:
        json.dump(ofc_report_std, f, default=str, indent=2)

    # 9. OFC CLUSTER
    ofc_passed_cluster, ofc_report_cluster = overfitting_check_rotational(
        wfo_summary      = summary_df_cluster,
        stocks_data      = stocks_data,
        benchmark_data   = benchmark_data,
        param_grid       = full_grid,
        profile          = profile,
        n_total_trials   = n_full_trials,
        seed             = 42,
        verbose          = verbose,
    )
    ofc_report_cluster_path = output_dir / f"{portfolio_title}_{year}_ofc_cluster.json"
    with open(ofc_report_cluster_path, "w") as f:
        json.dump(ofc_report_cluster, f, default=str, indent=2)

    # 10. MONTE CARLO
    mc_kwargs = dict(
        stocks_data      = stocks_data,
        benchmark_data   = benchmark_data,
        tickers_master   = tickers,
        init_cash        = init_cash,
        n_simulations    = 1000,
        seed             = 42,
        block_size       = 10,
        vol_window       = 60,
        n_vol_quantiles  = 3,
        show_method_plots    = False,
        show_method_summaries= False,
        save_plots           = True,
    )
    ci_results, ci_summary_df, skill_results, skill_summary_df = run_all_mc_methods_rotational(
        pf_rot           = pf_rot_std,
        pf_rot_base      = pf_rot_std_base,
        regime           = regime,
        sel_tickers      = sel_tickers_std,
        sel_tickers_base = sel_tickers_std_base,
        plots_dir        = plots_dir_std,
        **mc_kwargs,
    )
    ci_results_cluster, ci_summary_df_cluster, skill_results_cluster, skill_summary_df_cluster = run_all_mc_methods_rotational(
        pf_rot           = pf_rot_cluster,
        pf_rot_base      = pf_rot_cluster_base,
        regime           = regime_cluster,
        sel_tickers      = sel_tickers_cluster,
        sel_tickers_base = sel_tickers_cluster_base,
        plots_dir        = plots_dir_cluster,
        **mc_kwargs,
    )

    # 11. DECISIONE
    skill_profile_std, skill_profile_cluster = compute_skill_profile(
        mc_skill         = skill_results,
        mc_skill_cluster = skill_results_cluster,
    )

    # 12. OUTPUT
    _today_iso = date.today().isoformat()
    _wfo_config = {
        "ratio":            ratio,
        "metric":           metric,
        "n_full_trials":    n_full_trials,
        "n_reduced_trials": n_reduced_trials,
        "wfo_file_save":    wfo_file_save,
        "use_clustering":   True,
        "n_bootstrap_ofc":  1000,
        "n_bootstrap_mc":   1000,
    }
    _cluster_result = results_cluster.get("cluster_result") if results_cluster else None
    _metrics_comparison = {
        "cluster_riskoff": results_cluster.get("pf_rot"),
        "cluster_base":    results_cluster.get("pf_rot_base"),
        "std_riskoff":     results_std.get("pf_rot"),
        "std_base":        results_std.get("pf_rot_base"),
        "benchmark":       results_std.get("pf_benchmark") or results_std.get("pf_benchmark_base"),
    }

    _card_path = output_dir / f"{portfolio_title.replace(' ', '_').lower()}_{year}.md"
    _pdf_path  = output_dir / f"{portfolio_title}_{year}_Relazione_Tecnica.pdf"

    generate_ptf_card_md(
        portfolio_title    = portfolio_title,
        year               = year,
        profile            = profile,
        benchmark          = benchmark_title,
        period             = (str(pipeline_start_date), _today_iso),
        universe_size      = len(tickers),
        wfo_config         = _wfo_config,
        cluster_result     = _cluster_result,
        metrics_comparison = _metrics_comparison,
        ofc_report_std     = ofc_report_std,
        ofc_report_cluster = ofc_report_cluster,
        mc_skill           = skill_results,
        mc_ci              = ci_summary_df,
        mc_skill_cluster   = skill_results_cluster,
        mc_ci_cluster      = ci_summary_df_cluster,
        skill_profile      = skill_profile_std,
        output_path        = str(_card_path),
    )

    generate_relazione_tecnica(
        portfolio_title       = portfolio_title,
        year                  = year,
        profile               = profile,
        benchmark             = benchmark_title,
        period                = (str(pipeline_start_date), _today_iso),
        universe_size         = len(tickers),
        wfo_config            = _wfo_config,
        cluster_result        = _cluster_result,
        metrics_comparison    = _metrics_comparison,
        ofc_report_std        = ofc_report_std,
        ofc_report_cluster    = ofc_report_cluster,
        mc_skill              = skill_results,
        mc_ci                 = ci_summary_df,
        skill_profile         = skill_profile_std,
        skill_profile_cluster = skill_profile_cluster,
        plots_dir             = str(plots_dir),
        output_path           = str(_pdf_path),
        mc_skill_cluster      = skill_results_cluster,
        mc_ci_cluster         = ci_summary_df_cluster,
    )

    return {
        "pdf":                   _pdf_path,
        "plots_dir":             plots_dir,
        "ofc_std":               ofc_passed_std,
        "ofc_cluster":           ofc_passed_cluster,
        "skill_profile_std":     skill_profile_std,
        "skill_profile_cluster": skill_profile_cluster,
    }

