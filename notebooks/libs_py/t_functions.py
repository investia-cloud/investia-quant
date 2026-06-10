"""
t_functions.py — Refactored from notebooks/libs/t_functions.ipynb
"""

# =================================
# Universe Selections Functions (top performing)
# =================================
from typing import Optional, Iterable, Dict, Tuple, List, Union, Callable, Any
import pandas as pd
import numpy as np
from itertools import product
from u_functions import fetch_data_and_companies, my_display, RESET, BOLD
from tqdm.auto import tqdm
from tqdm.auto import tqdm

# Importo le funzioni di utilita' generali
# %run u_functions.ipynb




# 
# Selezione dei titoli per performance
#


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def _top_cut(row: pd.Series, top: Optional[Union[int, float]]) -> List[str]:
    """Ritorna lista tickers top per riga; top può essere int (k) o float (0..1)."""
    if top is None:
        return list(row.index[row.notna()])
    n = row.shape[0]
    if isinstance(top, float):
        k = max(1, int(round(n * top)))
    else:
        k = max(1, int(top))
    # prendi le colonne con rank più basso (1 = migliore)
    order = row.sort_values(ascending=True).index[:k]
    return list(order)

def _percent_valid(s: pd.Series) -> float:
    return s.notna().mean() if len(s) else 0.0

def _first_last_ret(frame: pd.DataFrame) -> pd.DataFrame:
    """Rendimento (last/first - 1) per ciascun periodo di un resample."""
    return (frame.last() / frame.first()) - 1

def _monthly_total_return(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """Rendimento totale su finestra rolling (n mesi), su dati mensili (ultimo prezzo del mese)."""
    m = df.resample("ME").last()
    return (m / m.shift(months)) - 1

# ---------------------------------------------------------------------
# Funzione estesa
# ---------------------------------------------------------------------


def select_top_performing_stocks(
    data: pd.DataFrame,
    top_percentage: Optional[Union[float, int]] = None,
    *,
    # ---- Selezione primaria ----
    primary: str = "Y",                 # "Y"=annuale, "Q"=trimestrale, "M"=mensile, "R"=rolling mesi
    primary_lookback_months: int = 12,  # usato solo se primary == "R"
    # shift della selezione al periodo successivo
    shift_forward: bool = True,         # True = assegna i top del periodo T al periodo T+1
    # qualità dati
    min_valid_ratio: float = 0.6,       # quota minima di valori non-NaN nel periodo per includere il ticker
    # ---- Selezione secondaria (opzionale) ----
    secondary: Optional[Dict[str, Any]] = None,
    # es. secondary = {"lookback_months": 6, "top": 0.3}
    # ---- NUOVO: calcolo portafoglio/metriche ----
    compute_metrics: bool = False,
    weighting: str = "equal",           # "equal" (per ora)
    fee_bps: float = 0.0,               # costo di ribilanciamento per periodo (in basis points, es. 5 = 0.05%)
    rf: float = 0.0                     # tasso risk-free ANNUO (decimale) per Sharpe/Sortino
) -> Union[pd.Series, Tuple[pd.Series, Dict[str, float], pd.DataFrame]]:
    """
    Seleziona i top performer su base annuale, trimestrale, mensile o rolling n-mesi.
    Supporta un secondo filtro opzionale (rolling breve) tra i candidati del filtro primario.
    (NUOVO) Se compute_metrics=True, costruisce il portafoglio EW per periodo di applicazione e calcola metriche.

    Parameters
    ----------
    data : pd.DataFrame
        Prezzi (Close adjusted) in formato wide (index DateTime, colonne ticker).
    top_percentage : float in (0,1], int >=1, o None
        - float: percentuale dei migliori per periodo
        - int  : numero fisso (k) di migliori per periodo
        - None : non filtra (restituisce tutti i candidati non-NaN per periodo)
    primary : {"Y","Q","M","R"}
        Periodicità primaria: yearly, quarterly, monthly, o rolling n mesi.
    primary_lookback_months : int
        Lookback per primary="R".
    shift_forward : bool
        Se True, i top di T vengono *applicati* nel periodo T+1.
    min_valid_ratio : float
        Quota minima di dati validi nel periodo per includere un ticker.
    secondary : dict o None
        Filtro secondario opzionale (rolling mesi) tra i candidati primari.
        Attesi:
            - "lookback_months": int   (obbligatorio)
            - "top": float|int|None    (percentuale o k; None = non riduce)
    compute_metrics : bool
        Se True, calcola i rendimenti per periodo del portafoglio EW e le metriche (CAGR, vol, Sharpe, ecc.).
    weighting : str
        “equal” = equal weight (unico supportato al momento).
    fee_bps : float
        Costo di ribilanciamento per periodo (in basis points, ad es. 5 = 0.05% per periodo).
    rf : float
        Tasso risk-free annuale per Sharpe/Sortino (decimale, es. 0.02 = 2%).

    Returns
    -------
    - Se compute_metrics=False:
        pd.Series  -> indice: periodo di *applicazione* (shiftato se richiesto), valori: lista di tickers selezionati.
    - Se compute_metrics=True:
        (prim_lists, pf_metrics, pf_frame)
        prim_lists : Series {periodo_applicazione -> [tickers]}
        pf_metrics : dict con metriche
        pf_frame   : DataFrame con colonne ["period_return","equity"] indicizzate per periodo applicazione
    """

    # ---------------------------
    # Helper interni
    # ---------------------------
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        return df

    def _normalize_close(df: pd.DataFrame) -> pd.DataFrame:
        # Gestione MultiIndex con livello "Close"
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                df = df.xs("Close", level=0, axis=1)
            elif "Close" in df.columns.get_level_values(-1):
                df = df.xs("Close", level=-1, axis=1)
            else:
                raise ValueError("Impossibile trovare 'Close' nelle colonne MultiIndex.")
        return df

    def _percent_valid(s: pd.Series) -> float:
        n = len(s)
        return float(s.notna().sum()) / float(n) if n > 0 else 0.0

    def _resample_returns_first_last(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Rendimento per periodo: last/first - 1 su resample."""
        g_first = df.resample(rule).first()
        g_last  = df.resample(rule).last()
        out = (g_last / g_first) - 1.0
        return out

    def _apply_valid_ratio_mask(df: pd.DataFrame, rule: str, ratio: float) -> pd.DataFrame:
        """Maschera colonne periodo-per-periodo se non rispettano min_valid_ratio nel periodo."""
        masked = []
        for end_ts, frame in df.resample(rule):
            # frame: sub-df del periodo
            keep = frame.apply(_percent_valid, axis=0) >= float(ratio)
            # calcolo return su first/last ma tengo solo colonne “keep”
            if keep.any():
                r = _resample_returns_first_last(frame.loc[:, keep], rule).iloc[-1:]
                masked.append(r)
            else:
                masked.append(pd.DataFrame(index=[end_ts], columns=df.columns))
        res = pd.concat(masked).sort_index()
        return res

    def _rolling_monthly_total_return(df: pd.DataFrame, months: int) -> pd.DataFrame:
        """Total return su base mensile: (P_t / P_{t-months}) - 1 calcolato ai month-end."""
        m = df.resample("ME").last()
        return (m / m.shift(months)) - 1.0

    def _top_k_from_rank_row(rank_row: pd.Series, top) -> list:
        """Dato un rank (1=best) seleziona i top. top può essere float in (0,1] o int."""
        rr = rank_row.dropna()
        if rr.empty:
            return []
        if top is None:
            return list(rr.index)
        if isinstance(top, float):
            k = max(1, int(np.floor(len(rr) * top)))
        else:
            k = int(top)
        k = max(1, min(k, len(rr)))
        # i migliori hanno rank più basso
        return list(rr.sort_values(ascending=True).index[:k])

    def _ann_factor_from_primary(primary: str) -> int:
        return 1 if primary == "Y" else 4 if primary == "Q" else 12

    def _period_bounds_from_label(label, primary: str):
        """Restituisce (start_ts, end_ts) inclusivi per periodo di APPLICAZIONE."""
        if primary == "Y":
            y = int(label)
            return pd.Timestamp(year=y, month=1, day=1), pd.Timestamp(year=y, month=12, day=31)
        elif primary == "Q":
            p = pd.Period(label, freq="Q")
            return p.start_time.normalize(), p.end_time.normalize()
        else:  # "M" o "R"
            p = pd.Period(label, freq="M")
            return p.start_time.normalize(), p.end_time.normalize()

    def _first_last_price_in_window(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
        s = prices.loc[start:end].dropna()
        if s.empty:
            return np.nan, np.nan
        return float(s.iloc[0]), float(s.iloc[-1])

    def _max_drawdown(equity: pd.Series) -> float:
        roll_max = equity.cummax()
        dd = equity / roll_max - 1.0
        return float(dd.min()) if not dd.empty else 0.0

    # ---------------------------
    # Normalizzazione input
    # ---------------------------
    df = _ensure_datetime_index(data)
    df = _normalize_close(df)
    df = df.sort_index()

    # ---------------------------
    # 1) Calcolo dei ritorni primari per periodo
    # ---------------------------
    if primary == "Y":
        primary_returns = _apply_valid_ratio_mask(df, "YE", min_valid_ratio)
        # Etichette periodo di SELEZIONE (prima di eventuale shift)
        selection_index = primary_returns.index.year.astype(int)
    elif primary == "Q":
        primary_returns = _apply_valid_ratio_mask(df, "QE", min_valid_ratio)
        selection_index = primary_returns.index.to_period("Q").astype(str)
    elif primary == "M":
        primary_returns = _apply_valid_ratio_mask(df, "ME", min_valid_ratio)
        selection_index = primary_returns.index.to_period("M").astype(str)
    elif primary == "R":
        # rolling n mesi (ai month-end). Applico un controllo validità sul numero di mesi non-NaN
        rets = _rolling_monthly_total_return(df, primary_lookback_months)
        # validità rolling: almeno ceil(n * min_valid_ratio) mesi con dati
        m_last = df.resample("ME").last()
        valid_count = m_last.notna().astype(int).rolling(primary_lookback_months, min_periods=1).sum()
        need = int(np.ceil(primary_lookback_months * min_valid_ratio))
        primary_returns = rets.where(valid_count >= need, np.nan).dropna(how="all")
        selection_index = primary_returns.index.to_period("M").astype(str)
    else:
        raise ValueError("primary deve essere uno tra {'Y','Q','M','R'}")

    if primary_returns.empty:
        return pd.Series(dtype=object)

    # Ranking (1=best)
    ranked = primary_returns.rank(axis=1, ascending=False, method="first")

    # ---------------------------
    # 2) Selezione primaria (percentuale o k)
    # ---------------------------
    if top_percentage is None:
        prim_lists_sel = primary_returns.apply(lambda row: list(row.index[row.notna()]), axis=1)
    else:
        prim_lists_sel = ranked.apply(lambda row: _top_k_from_rank_row(row, top_percentage), axis=1)

    prim_lists_sel.index = selection_index  # periodi di SELEZIONE
    
    # ---------------------------
    # 3) Selezione secondaria (rolling breve) tra i candidati primari
    # ---------------------------
    if secondary is not None:
        sec_n = int(secondary.get("lookback_months", 0) or 0)
        sec_top = secondary.get("top", None)
        if sec_n <= 0:
            raise ValueError("secondary.lookback_months deve essere un intero > 0")

        short_rets = _rolling_monthly_total_return(df, sec_n).sort_index()  # ai month-end (ME)

        def _closest_me_le(idx: pd.DatetimeIndex, ref: pd.Timestamp) -> pd.Timestamp | None:
            """Ritorna l’ultimo month-end <= ref disponibile in idx (o None se non esiste)."""
            pos = idx.searchsorted(ref, side="right") - 1
            return None if pos < 0 else idx[pos]

        refined = []
        for idx_lab, candidates in prim_lists_sel.items():
            if not candidates:
                refined.append([])
                continue

            # Timestamp di riferimento = fine periodo di SELEZIONE
            if primary == "Y":
                # idx_lab = anno (es. 2025): uso il 31/12/anno -> month-end corrispondente
                ref_end = pd.Timestamp(year=int(idx_lab), month=12, day=31)
                ref_me = ref_end.to_period("M").to_timestamp("M")
            elif primary == "Q":
                ref_me = pd.Period(idx_lab, freq="Q").end_time.to_period("M").to_timestamp("M")
            else:  # "M" o "R"
                ref_me = pd.Period(idx_lab, freq="M").to_timestamp("M")

            # Usa l’ultimo month-end disponibile <= ref_me
            use_me = _closest_me_le(short_rets.index, ref_me)
            if use_me is None:
                # nessun dato mensile disponibile: impossibile affinare
                refined.append(candidates)
                continue

            # Interseca i candidati con le colonne effettivamente presenti
            cands = [t for t in candidates if t in short_rets.columns]
            if not cands:
                refined.append([])
                continue

            row_short = short_rets.loc[use_me, cands].dropna()
            if row_short.empty:
                refined.append([])
                continue

            # Rank decrescente per rendimento (return alto = migliore)
            rank_short = (-row_short).rank(ascending=True, method="first")

            # Selezione top secondario (float in (0,1] o int, oppure None=non riduce)
            refined.append(_top_k_from_rank_row(rank_short, sec_top))

        prim_lists_sel = pd.Series(refined, index=prim_lists_sel.index, dtype=object)


    # ---------------------------
    # 4) Shift della selezione al periodo successivo (periodo di APPLICAZIONE)
    # ---------------------------
    if shift_forward:
        if primary == "Y":
            new_index = (pd.Series(prim_lists_sel.index, dtype=int) + 1).astype(int)
            prim_lists = pd.Series(prim_lists_sel.values, index=new_index, dtype=object)
        elif primary == "Q":
            next_idx = (pd.PeriodIndex(prim_lists_sel.index, freq="Q") + 1).astype(str)
            prim_lists = pd.Series(prim_lists_sel.values, index=next_idx, dtype=object)
        else:  # "M" o "R"
            next_idx = (pd.PeriodIndex(prim_lists_sel.index, freq="M") + 1).astype(str)
            prim_lists = pd.Series(prim_lists_sel.values, index=next_idx, dtype=object)
    else:
        prim_lists = prim_lists_sel.copy()

    prim_lists.name = "Top_Tickers"

    # ---------------------------
    # 5) (NUOVO) Calcolo portafoglio e metriche (opzionale)
    # ---------------------------
    if not compute_metrics:
        return prim_lists

    if weighting.lower() != "equal":
        raise NotImplementedError("Al momento è supportato soltanto weighting='equal'.")

    ann_factor = _ann_factor_from_primary(primary)

    # Confini per PERIODO DI APPLICAZIONE (indici già shiftati se richiesto)
    period_bounds = []
    for lab in prim_lists.index:
        start, end = _period_bounds_from_label(lab, primary)
        period_bounds.append((lab, start, end))
    period_bounds = pd.DataFrame(period_bounds, columns=["label", "start", "end"]).set_index("label")

    prices = df  # Close adjusted

    # Rendimento equal-weight per periodo
    period_rets = []
    for lab, row in period_bounds.iterrows():
        sel = prim_lists.loc[lab]
        if not sel:
            period_rets.append(np.nan)
            continue
        sel = [t for t in sel if t in prices.columns]
        if not sel:
            period_rets.append(np.nan)
            continue

        start_ts, end_ts = row["start"], row["end"]

        r_list = []
        for t in sel:
            p0, p1 = _first_last_price_in_window(prices[t], start_ts, end_ts)
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0.0:
                continue
            r_list.append((p1 / p0) - 1.0)

        if len(r_list) == 0:
            period_rets.append(np.nan)
            continue

        r_eq = float(np.mean(r_list))

        # costo di ribilanciamento per PERIODO (fee_bps basis points)
        if fee_bps and fee_bps != 0.0:
            r_eq -= float(fee_bps) / 1e4

        period_rets.append(r_eq)

    pf_frame = pd.DataFrame(
        {"period_return": pd.Series(period_rets, index=prim_lists.index, dtype=float)}
    ).dropna()

    if pf_frame.empty:
        pf_metrics = {
            "n_periods": 0, "total_return": np.nan, "CAGR": np.nan, "ann_vol": np.nan,
            "Sharpe": np.nan, "Sortino": np.nan, "max_dd": np.nan, "Calmar": np.nan,
            "hit_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
            "ann_factor": ann_factor, "rf": rf, "fee_bps": fee_bps, "weighting": weighting
        }
        return prim_lists, pf_metrics, pf_frame

    # Equity e metriche
    pf_frame["equity"] = (1.0 + pf_frame["period_return"]).cumprod()

    n_periods = len(pf_frame)
    total_ret = float(pf_frame["equity"].iloc[-1] - 1.0)

    years = n_periods / float(ann_factor)
    CAGR = (1.0 + total_ret)**(1.0 / years) - 1.0 if years > 0 else np.nan

    pr = pf_frame["period_return"].astype(float)
    vol_p = float(pr.std(ddof=1)) if n_periods > 1 else np.nan
    ann_vol = vol_p * np.sqrt(ann_factor) if not np.isnan(vol_p) else np.nan

    # Sharpe (usa CAGR come rendimento annuo composto vs rf annuo)
    Sharpe = (CAGR - rf) / ann_vol if (ann_vol and not np.isnan(ann_vol) and ann_vol > 0) else np.nan

    # Sortino (downside deviation su periodi → annualizzata)
    neg = pr[pr < 0]
    dd_p = float(np.sqrt((neg.pow(2)).mean())) if len(neg) > 0 else np.nan
    ann_dd = dd_p * np.sqrt(ann_factor) if not np.isnan(dd_p) else np.nan
    Sortino = (CAGR - rf) / ann_dd if (ann_dd and not np.isnan(ann_dd) and ann_dd > 0) else np.nan

    max_dd = _max_drawdown(pf_frame["equity"])
    Calmar = (CAGR / abs(max_dd)) if (max_dd is not None and max_dd < 0) else np.nan

    wins = pr[pr > 0]
    losses = pr[pr < 0]
    hit_rate = float(len(wins)) / float(len(pr)) if len(pr) > 0 else np.nan
    avg_win = float(wins.mean()) if len(wins) > 0 else np.nan
    avg_loss = float(losses.mean()) if len(losses) > 0 else np.nan

    # --- costruzione metriche (raw) ---
    pf_metrics_raw = {
        "n_periods": n_periods,
        "total_return": total_ret,   # % totale (decimale)
        "CAGR": CAGR,                # % annua (decimale)
        "ann_vol": ann_vol,          # % annua (decimale)
        "Sharpe": Sharpe,            # ratio
        "Sortino": Sortino,          # ratio
        "max_dd": max_dd,            # drawdown minimo (negativo, decimale)
        "Calmar": Calmar,            # ratio
        "hit_rate": hit_rate,        # frazione periodi + (decimale)
        "avg_win": avg_win,          # media rendimenti + (decimale)
        "avg_loss": avg_loss,        # media rendimenti - (decimale)
        "ann_factor": ann_factor,
        "rf": rf,                    # tasso annuo (decimale)
        "fee_bps": fee_bps,
        "weighting": weighting
    }

    # --- formattazione per output (percentuali a 2 decimali) ---
    def _fmt_pct(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"
    def _fmt_ratio(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"

    percent_keys = [
        "total_return", "CAGR", "ann_vol",
        "max_dd", "hit_rate", "avg_win", "avg_loss", "rf"
    ]
    ratio_keys = ["Sharpe", "Sortino", "Calmar"]

    pf_metrics = {}
    for k, v in pf_metrics_raw.items():
        if k in percent_keys:
            pf_metrics[k] = _fmt_pct(v)
        elif k in ratio_keys:
            pf_metrics[k] = _fmt_ratio(v)
        else:
            pf_metrics[k] = v  # interi/stringhe/bps senza formattazione

    # restituisci anche le raw se ti servono per ulteriori calcoli
    return prim_lists, pf_metrics, pf_frame


'''

# 1) Top annuali (come ora), assegnati all’anno successivo:
top_yr = select_top_performing_stocks(data, top_percentage=0.2, primary="Y", shift_forward=True)
# -> indice: 2016, 2017, ..., valori: liste di ticker


# 2) Top trimestrali (top 5 fissi), assegnati al trimestre successivo:
top_q = select_top_performing_stocks(data, top_percentage=5, primary="Q", shift_forward=True)
# -> indice: "2024Q2", "2024Q3", ...

# 3) Rolling 12 mesi (tipo momentum), selezione mensile, poi filtro secondario 3 mesi:
top_roll = select_top_performing_stocks(
    data,
    top_percentage=0.3,        # primi 30% per rolling 12m
    primary="R",
    primary_lookback_months=12,
    shift_forward=True,
    secondary={"lookback_months": 3, "top": 0.5}  # metà migliori su 3m tra i candidati 12m
)
# -> indice: "YYYY-MM" del mese SUCCESSIVO


# 4) Universe completo per periodo (nessun filtro), ma poi riduci coi 6 mesi:

full_then_6m = select_top_performing_stocks(
    data,
    top_percentage=None,             # nessun taglio primario
    primary="Y",
    shift_forward=True,
    secondary={"lookback_months": 6, "top": 10}  # tieni i migliori 10 su 6 mesi
)


'''
#
# Selettore top tickers da dizionario di tickers
#
def build_top_momentum_universe(
    *,
    tickers_dict: Dict[str, List[str]],
    exclude_tickers: Optional[List[str]] = None,
    base_n_select: int = 10,
    select_top_performers: bool = True,
    year: Optional[int] = None,
    primary_lookback_months: int = 12,
    top_percentage: int = 40,
    secondary_lookback_months: int = 3,
    secondary_n_top: int = 12,
    shift_forward: bool = True,
    verbose: bool = True,
) -> Tuple[List[str], pd.DataFrame, dict]:
    """
    Costruisce un universo di titoli basato su selezione Dual Momentum
    a partire da più liste di ticker già preparate esternamente.

    Logica:
      1) Per ogni lista in tickers_dict:
         - download dati storici (start/end calcolati da year)
         - selezione primary momentum (rolling 12 mesi)
         - selezione secondary momentum (rolling 3 mesi)
      2) Per ogni lista seleziona:
         base_n_select + len(exclude_tickers) titoli
         poi rimuove gli exclude_tickers
      3) Ritorna:
         - lista finale tickers aggregata
         - company_data concatenato
         - dizionario extra con info di debug

    Parametro year:
      - None          → usa l’anno corrente
      - ≤ anno corrente → end_date = 31/12/(year-1)
      - == anno prox   → end_date = None (dati fino ad oggi)
      - > anno prox    → ValueError

    NOTE:
    - Le funzioni seguenti devono essere già disponibili nell’ambiente:
        * fetch_data_and_companies
        * select_top_performing_stocks
        * my_display
    """

    from datetime import date as _date
    _today        = _date.today()
    _current_year = _today.year
    _next_year    = _current_year + 1

    if year is None:
        year = _current_year

    if year > _next_year:
        raise ValueError(
            f"year={year} non valido: massimo consentito è {_next_year}."
        )

    start_date = f"{year - 5}-01-01"
    end_date   = f"{year - 1}-12-31" if year <= _current_year else None

    if verbose:
        print(f"  year={year}  start_date={start_date}  end_date={end_date or 'oggi'}")

    exclude_tickers = list(exclude_tickers or [])
    exclude_set = set(exclude_tickers)

    # Numero di titoli da richiedere inizialmente (robusto alle esclusioni)
    target_n = int(base_n_select) + len(exclude_tickers)

    all_selected: List[str] = []
    company_data_list: List[pd.DataFrame] = []
    per_list_selected: Dict[str, List[str]] = {}

    if select_top_performers:
        for nome_lista, tickers_lista in tickers_dict.items():
            if verbose:
                print(f"\n>>> Elaboro lista: {nome_lista}  ({len(tickers_lista)} ticker)")
                print(tickers_lista)

            # --- Download dati ---
            stocks_data, list_company_data = fetch_data_and_companies(
                tickers_lista, start_date, end_date
            )
            company_data_list.append(list_company_data)

            # --- Dual Momentum ---
            top_roll = select_top_performing_stocks(
                stocks_data,
                top_percentage=top_percentage,
                primary="Y",
                primary_lookback_months=primary_lookback_months,
                shift_forward=shift_forward,
                secondary={
                    "lookback_months": secondary_lookback_months,
                    "top": secondary_n_top,
                },
            )

            # Display (come nel codice originale)
            my_display(top_roll, title=f"{secondary_n_top} dual momentum")

            # --- Candidati per l'anno selezionato ---
            candidates = list(top_roll.loc[year])

            # --- Selezione robusta con esclusioni ---
            pre = candidates[:target_n]
            post = [t for t in pre if t not in exclude_set]

            # Se dopo la rimozione siamo sotto base_n_select,
            # pescare oltre target_n
            if len(post) < base_n_select:
                extra_pool = [t for t in candidates[target_n:] if t not in exclude_set]
                need = base_n_select - len(post)
                post.extend(extra_pool[:need])

            # Taglio finale
            final_for_list = post[:base_n_select]

            per_list_selected[nome_lista] = final_for_list
            all_selected.extend(final_for_list)

            if verbose:
                print(
                    f"{BOLD}{len(final_for_list)}{RESET} "
                    f"tickers per la lista {BOLD}{nome_lista}{RESET}:\n"
                    f"{final_for_list}"
                )

        company_data = (
            pd.concat(company_data_list, ignore_index=False)
            if company_data_list
            else pd.DataFrame()
        )

        if verbose:
            print("")
            print(
                f"{BOLD}{len(all_selected)}{RESET} "
                f"tickers per l'anno {BOLD}{year}{RESET}:\n{all_selected}"
            )

    else:
        # Nessuna selezione: ritorna semplicemente l'unione delle liste
        all_selected = [
            t for sublist in tickers_dict.values() for t in sublist
        ]
        company_data = pd.DataFrame()

        if verbose:
            print(f"\n{BOLD}{len(all_selected)}{RESET} tickers:\n\n{all_selected}")

    extra = {
        "per_list_selected": per_list_selected,
        "exclude_tickers": exclude_tickers,
        "target_n_requested_per_list": target_n,
    }

    return all_selected, company_data, extra

# Esempi d'uso
# tickers, company_data, extra = build_top_momentum_universe(
#     tickers_dict=tickers_dict,
#     exclude_tickers=["GOOG", "CSCO"],
#     base_n_select=10,
#     year=2026,
# )

# tickers, company_data, extra = build_top_momentum_universe(
#     tickers_dict={"nasdaq100": tickers_nasdaq100},
#     base_n_select=8,
#     year=2026,
# )

def build_soft_weights_for_year(
    data: pd.DataFrame,
    year: int,
    *,
    k_target: int = 25,
    primary_top: Union[float, int] = 0.3,
    secondary: Dict[str, Any] = {"lookback_months": 3, "top": None},
    alpha: float = 0.5,     # peso base per chi passa solo il primario
    beta: float  = 1.0,     # peso pieno per chi passa anche il secondario
    min_valid_ratio: float = 0.6,
) -> pd.Series:
    """
    Costruisce pesi 'soft' mantenendo esattamente k_target titoli:
      - Universo = selezione primaria (12m) → score = alpha
      - Confermati dal secondario (es. 3–6m) → score = beta
      - Ordina per (score desc, 12m desc) → prendi i primi k_target
      - Pesi normalizzati a somma=1 (proporzionali allo score; equal se tutti uguali)
    """
    # --- Normalizza indice tempo ---
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.copy()
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    # --- FIX: normalizza input colonne (accetta data o data.Close) ---
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data.xs("Close", level=0, axis=1)
        elif "Close" in data.columns.get_level_values(-1):
            close = data.xs("Close", level=-1, axis=1)
        else:
            raise ValueError("Impossibile trovare 'Close' nelle colonne MultiIndex.")
    else:
        # data è già wide (colonne=ticker) oppure è data.Close
        close = data

    # --- Selezioni per anno di APPLICAZIONE (shift_forward=True) ---
    prim = select_top_performing_stocks(
        close,
        primary="Y",
        top_percentage=primary_top,
        shift_forward=True,
        min_valid_ratio=min_valid_ratio,
        secondary=None
    )
    sec = select_top_performing_stocks(
        close,
        primary="Y",
        top_percentage=primary_top,
        shift_forward=True,
        min_valid_ratio=min_valid_ratio,
        secondary=secondary
    )

    if year not in prim.index:
        raise ValueError(f"Nessuna selezione disponibile per l'anno {year} (controlla dati/periodi).")

    prim_list = list(prim.loc[year] or [])
    sec_list  = list(sec.loc[year] or [])

    if not prim_list:
        return pd.Series(dtype=float, name=f"W_{year}")

    # --- Score 'soft': alpha per (solo primario), beta per (primario ∩ secondario) ---
    universe = list(dict.fromkeys(prim_list))  # mantiene ordine
    score = pd.Series(alpha, index=universe, dtype=float)
    inters = list(set(universe) & set(sec_list))
    if inters:
        score.loc[inters] = beta

    # --- Tie-break: ritorno 12m dell'anno precedente (ex-post) ---
    m = close.resample("M").last()
    ret12 = m / m.shift(12) - 1
    ref_me = pd.Timestamp(year - 1, 12, 31).to_period("M").to_timestamp("M")
    if ref_me in ret12.index:
        tie = ret12.loc[ref_me, universe].replace([np.inf, -np.inf], np.nan)
    else:
        tie = pd.Series(index=universe, dtype=float)

    # --- Ordina per (score desc, 12m desc) e scegli k_target ---
    order = pd.DataFrame({"score": score, "ret12": tie}).sort_values(
        ["score", "ret12"], ascending=False
    )
    chosen = order.index.tolist()[:min(k_target, len(order))]

    # --- Pesi proporzionali allo score (fallback equal se tutti uguali) ---
    base = score.loc[chosen]
    if base.nunique() == 1:
        w = pd.Series(1.0 / len(chosen), index=chosen, name=f"W_{year}")
    else:
        w = base.clip(lower=0)
        s = w.sum()
        w = (w / s) if s > 0 else pd.Series(1.0 / len(chosen), index=chosen)
        w.name = f"W_{year}"

    return list(w)



# -------------------------------------------------
# Helper metriche e ranking
# -------------------------------------------------
def _ann_factor(primary: str) -> int:
    """Fattore di annualizzazione in base al primario."""
    return 1 if primary == "Y" else 4 if primary == "Q" else 12  # "M" e "R" → 12

def _max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown robusto:
    - NaN se la serie è completamente vuota
    - 0.0 se c'è un solo punto valido (nessun drawdown osservabile)
    """
    if equity is None:
        return np.nan
    e = pd.Series(equity).dropna()
    if e.empty:
        return np.nan
    if len(e) == 1:
        return 0.0
    dd = e / e.cummax() - 1.0
    return float(dd.min()) if not dd.empty else np.nan

def _metrics_from_pf_frame(pf_frame: pd.DataFrame, primary: str, rf: float = 0.0) -> Dict[str, float]:
    """
    Ricalcola metriche OOS usando pf_frame['period_return'] e ['equity'] (serie per periodo).
    Restituisce valori grezzi (decimali), non formattati.
    """
    out = { "n_periods": 0, "total_return": np.nan, "CAGR": np.nan, "ann_vol": np.nan,
            "Sharpe": np.nan, "Sortino": np.nan, "max_dd": np.nan, "Calmar": np.nan }
    if pf_frame is None or pf_frame.empty or "period_return" not in pf_frame:
        return out

    pr = pd.Series(pf_frame["period_return"]).astype(float).dropna()
    if pr.empty:
        return out

    ann = _ann_factor(primary)
    eq = (1.0 + pr).cumprod()

    total_ret = float(eq.iloc[-1] - 1.0)
    years = len(pr) / float(ann) if ann > 0 else np.nan
    CAGR = (1.0 + total_ret)**(1.0 / years) - 1.0 if years and years > 0 else np.nan

    vol_p = float(pr.std(ddof=1)) if len(pr) > 1 else np.nan
    ann_vol = vol_p * np.sqrt(ann) if not np.isnan(vol_p) else np.nan

    Sharpe = (CAGR - rf) / ann_vol if (ann_vol and not np.isnan(ann_vol) and ann_vol > 0) else np.nan
    neg = pr[pr < 0]
    dd_p = float(np.sqrt((neg.pow(2)).mean())) if len(neg) > 0 else np.nan
    ann_dd = dd_p * np.sqrt(ann) if not np.isnan(dd_p) else np.nan
    Sortino = (CAGR - rf) / ann_dd if (ann_dd and not np.isnan(ann_dd) and ann_dd > 0) else np.nan

    max_dd = _max_drawdown(eq)
    Calmar = (CAGR / abs(max_dd)) if (max_dd is not None and not np.isnan(max_dd) and max_dd < 0) else np.nan

    out.update({ "n_periods": len(pr), "total_return": total_ret, "CAGR": CAGR, "ann_vol": ann_vol,
                 "Sharpe": Sharpe, "Sortino": Sortino, "max_dd": max_dd, "Calmar": Calmar })
    return out

def _composite_rank(df: pd.DataFrame,
                    weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Ranking multi-metrico robusto:
    - Massimizzare: CAGR, Sharpe, Sortino, Calmar → rank decrescente
    - Minimizzare: ann_vol, |max_dd|              → rank crescente
    Score = media pesata dei rank (↓ = migliore).
    """
    if weights is None:
        weights = {
            "CAGR": 1.0, "Sharpe": 0.8, "Sortino": 0.6, "Calmar": 0.6,
            "ann_vol": 0.4,  # ↓
            "max_dd": 0.6    # ↓ (usiamo ampiezza)
        }

    M = df.copy()
    if "max_dd" in M.columns:
        M["_dd_abs"] = M["max_dd"].abs()

    up_cols   = [c for c in ["CAGR","Sharpe","Sortino","Calmar"] if c in M.columns]
    down_cols = [c for c in ["ann_vol","_dd_abs"] if c in M.columns]

    ranks = pd.DataFrame(index=M.index)
    for c in up_cols:   ranks[c] = M[c].rank(ascending=False, method="average")
    for c in down_cols: ranks[c] = M[c].rank(ascending=True,  method="average")

    eff_w = {}
    for k, w in weights.items():
        if k == "max_dd" and "_dd_abs" in ranks.columns:
            eff_w["_dd_abs"] = w
        elif k in ranks.columns:
            eff_w[k] = w

    ws = pd.Series(eff_w, dtype=float)
    ws = ws / ws.sum() if ws.sum() > 0 else ws

    score = (ranks[ws.index] * ws).sum(axis=1)
    return score

def _ensure_triplet(res):
    """Normalizza l'output di select_top_performing_stocks in (prim_lists, pf_metrics, pf_frame)."""
    if isinstance(res, tuple) and len(res) == 3:
        return res
    empty_series  = pd.Series(dtype=object, name="Top_Tickers")
    empty_metrics = {"n_periods":0,"total_return":np.nan,"CAGR":np.nan,"ann_vol":np.nan,
                     "Sharpe":np.nan,"Sortino":np.nan,"max_dd":np.nan,"Calmar":np.nan}
    empty_frame   = pd.DataFrame(columns=["period_return","equity"])
    return empty_series, empty_metrics, empty_frame

# -------------------------------------------------
# Prebuffer per TEST, trimming pf_frame, normalizzazione indici equity
# -------------------------------------------------
def _compute_test_prebuffer_start(test_start: pd.Timestamp, primary: str, primary_lookback_months: Optional[int]) -> pd.Timestamp:
    """Calcola la data di inizio finestra estesa per il TEST, per generare segnali applicabili dentro il test con shift_forward=True."""
    if primary == "Y":
        return (test_start - pd.DateOffset(years=1)).normalize()
    if primary == "Q":
        return (test_start - pd.DateOffset(months=3)).normalize()
    if primary == "M":
        return (test_start - pd.DateOffset(months=1)).normalize()
    # "R" → lookback + 1 mese di margine
    lb = int(primary_lookback_months or 12)
    return (test_start - pd.DateOffset(months=lb + 1)).normalize()

def _trim_pf_frame_to_test_window(pf_frame: pd.DataFrame, primary: str,
                                  test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    """Ritaglia pf_frame ai soli periodi di APPLICAZIONE che cadono dentro [test_start, test_end]."""
    if pf_frame is None or pf_frame.empty:
        return pf_frame

    if primary == "Y":
        years = pd.Index([int(x) for x in pf_frame.index])
        mask = (years >= test_start.year) & (years <= test_end.year)
        return pf_frame.loc[mask]
    elif primary == "Q":
        per = pd.PeriodIndex(pf_frame.index, freq="Q")
        mask = (per.start_time >= test_start) & (per.end_time <= test_end)
        return pf_frame.loc[mask]
    else:  # "M" o "R"
        per = pd.PeriodIndex(pf_frame.index, freq="M")
        mask = (per.start_time >= test_start) & (per.end_time <= test_end)
        return pf_frame.loc[mask]

def _to_app_period_end(obj: Union[pd.Series, pd.DataFrame], primary: str) -> Union[pd.Series, pd.DataFrame]:
    """
    Converte l'indice (period label) nella data di FINE del periodo di applicazione (DatetimeIndex).
    - Y: 31/12 dell'anno
    - Q: end_time del trimestre
    - M/R: end_time del mese
    """
    out = obj.copy()
    if primary == "Y":
        years = pd.Index([int(x) for x in out.index])
        out.index = pd.DatetimeIndex([pd.Timestamp(year=y, month=12, day=31) for y in years])
    elif primary == "Q":
        per = pd.PeriodIndex(out.index, freq="Q")
        out.index = per.end_time.normalize()
    else:  # "M" o "R"
        per = pd.PeriodIndex(out.index, freq="M")
        out.index = per.end_time.normalize()
    return out

def _ensure_full_periods(pf_frame_test: pd.DataFrame,
                         primary: str,
                         test_start: pd.Timestamp,
                         test_end: pd.Timestamp,
                         fill_mode: str = "cash") -> pd.DataFrame:
    """
    Garantisce che nel TEST ogni periodo di applicazione sia presente.
    - 'cash': i buchi diventano ritorno 0.0; equity ricalcolata come cumprod(1+ret).
    """
    if pf_frame_test is None:
        return pf_frame_test

    # Porta l'indice a date di fine periodo
    F = _to_app_period_end(pf_frame_test, primary)
    # Costruisci l'indice target
    if primary == "Y":
        target = pd.date_range(pd.Timestamp(year=test_start.year, month=12, day=31),
                               pd.Timestamp(year=test_end.year, month=12, day=31),
                               freq="YE-DEC")
    elif primary == "Q":
        target = pd.period_range(test_start, test_end, freq="Q").to_timestamp("Q").to_period("Q").end_time.normalize()
    else:  # "M" o "R"
        target = pd.date_range(test_start.to_period("M").to_timestamp("M"),
                               test_end.to_period("M").to_timestamp("M"),
                               freq="ME")
    # Reindicizza i period_return e riempi
    pr = pd.Series(F.get("period_return", pd.Series(dtype=float)), index=F.index).reindex(target)
    if fill_mode == "cash":
        pr = pr.fillna(0.0)
    # Ricalcola equity coerente
    eq = (1.0 + pr).cumprod()
    out = pd.DataFrame({"period_return": pr, "equity": eq}, index=pr.index)
    return out

# -------------------------------------------------
# WFO principale per select_top_performing_stocks
# -------------------------------------------------
def wfo_optimize_selection(
    data: pd.DataFrame,
    *,
    # Finestra WFO
    train_years: int = 3,
    test_years: int = 1,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    # Parametri selezione (possono essere fissi o inclusi in param_grid)
    primary: str = "R",
    param_grid: Optional[Dict[str, Iterable]] = None,
    min_valid_ratio: float = 0.6,
    shift_forward: bool = True,
    # Performance
    compute_metrics: bool = True,       # deve restare True in WFO
    weighting: str = "equal",
    fee_bps: float = 0.0,
    rf: float = 0.0,
    # Ranking/scoring
    selection_rule: Optional[Union[str, Callable[[pd.DataFrame], pd.Series]]] = "composite",
    selection_weights: Optional[Dict[str, float]] = None,
    # Log & qualità OOS
    min_periods_oos: Optional[int] = None,   # se None: auto (6 per M/R, 1 per Y/Q)
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[Dict[str, Any]]]:
    """
    Walk-Forward Optimization per 'select_top_performing_stocks'.

    - Supporta 'primary' nella griglia (Y/Q/M/R). 'primary_lookback_months' è usato SOLO se 'primary' == 'R'.
    - Per ogni blocco: usa un prebuffer per il TEST (dipendente da primary/combo), trimma al solo test e
      riempie i buchi come 'cash' (ritorno 0) evitando metriche NaN.
    - Filtra le combo con copertura OOS insufficiente (min_periods_oos).
    - Ritorna:
        summary_df   : vincitore per blocco con parametri e metriche OOS
        results_df   : TUTTE le combinazioni per TUTTI i blocchi (combo-blocco)
        oos_equity   : equity OOS concatenata dei vincitori (DatetimeIndex coerente)
        winners_params : lista parametri vincenti (uno per blocco)
    """

    assert primary in {"Y","Q","M","R"}, "primary deve essere in {'Y','Q','M','R'}"
    assert compute_metrics, "In WFO compute_metrics deve restare True (servono pf_frame)."

    # Griglia di default (se non fornita)
    if param_grid is None:
        param_grid = {
            # "primary": ["Y", "M", "R"],  # opzionale se vuoi ottimizzare anche la frequenza
            "top_percentage": [5, 10, 15],
            "primary_lookback_months": [6, 9, 12],   # usato SOLO con primary='R'
            "secondary_lookback_months": [3, 4, 6],
            "secondary_top": [3, 5, 8]
        }

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    if verbose:
        print(f"[WFO] Combinazioni totali: {len(combos)}")

    # Normalizza dati
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    first_year, last_year = df.index.min().year, df.index.max().year
    if start_year is None: start_year = first_year + train_years
    if end_year   is None: end_year   = last_year

    # Blocchi train/test
    blocks = []
    y = start_year
    while y <= end_year:
        tr_s = pd.Timestamp(year=y-train_years, month=1, day=1)
        tr_e = pd.Timestamp(year=y-1,          month=12, day=31)
        te_s = pd.Timestamp(year=y,            month=1,  day=1)
        te_e = pd.Timestamp(year=y,            month=12, day=31)
        blocks.append((y, tr_s, tr_e, te_s, te_e))
        y += test_years
    if verbose:
        print(f"[WFO] Blocchi: {len(blocks)} ({train_years}y train + {test_years}y test)")

    summary_rows: List[Dict[str, Any]] = []
    all_results: List[pd.DataFrame] = []
    winners_params: List[Dict[str, Any]] = []
    oos_equity_parts: List[pd.Series] = []

    for (label_year, tr_s, tr_e, te_s, te_e) in tqdm(blocks, disable=not verbose):
        df_train = df.loc[tr_s:tr_e]
        if df_train.empty:
            if verbose:
                print(f"[WFO][{label_year}] TRAIN vuoto ({tr_s.date()}→{tr_e.date()}). Skip.")
            continue

        block_results: List[Dict[str, Any]] = []

        for vals in combos:
            params = dict(zip(keys, vals))

            # ---- primary per-combo ----
            primary_mode = params.get("primary", primary)  # se non in griglia, usa quello passato alla WFO
            assert primary_mode in {"Y","Q","M","R"}, f"primary non valido: {primary_mode}"

            # lookback effettivo (solo se rolling)
            lb = params.get("primary_lookback_months", 12)
            primary_lookback_effective = int(lb) if primary_mode == "R" else None

            # secondario (opzionale)
            sec_lb  = params.get("secondary_lookback_months", None)
            sec_top = params.get("secondary_top", None)
            secondary = None
            if sec_lb is not None and sec_top is not None:
                secondary = {"lookback_months": int(sec_lb), "top": sec_top}

            # --------------- TRAIN (solo diagnostica/omogeneità) ---------------
            try:
                res_train = select_top_performing_stocks(
                    data=df_train,
                    top_percentage=params.get("top_percentage"),
                    primary=primary_mode,
                    primary_lookback_months=primary_lookback_effective or 12,  # ignorato se non 'R'
                    shift_forward=shift_forward,
                    min_valid_ratio=min_valid_ratio,
                    secondary=secondary,
                    compute_metrics=True,
                    weighting=weighting,
                    fee_bps=fee_bps,
                    rf=rf
                )
                prim_lists_train, _, _ = _ensure_triplet(res_train)
                if prim_lists_train.empty and verbose:
                    print(f"[WFO][{label_year}] TRAIN vuoto per combo {params}.")
            except Exception as e:
                if verbose: print(f"[WFO][{label_year}] Errore TRAIN combo {params}: {e}")
                continue

            # --------------- TEST con PREBUFFER per-combo ---------------
            pre_start = _compute_test_prebuffer_start(te_s, primary_mode, primary_lookback_effective)
            df_test_ext = df.loc[pre_start:te_e]
            if df_test_ext.empty:
                if verbose:
                    print(f"[WFO][{label_year}] TEST esteso vuoto ({pre_start.date()}→{te_e.date()}).")
                continue

            try:
                res_test_ext = select_top_performing_stocks(
                    data=df_test_ext,
                    top_percentage=params.get("top_percentage"),
                    primary=primary_mode,
                    primary_lookback_months=primary_lookback_effective or 12,
                    shift_forward=shift_forward,
                    min_valid_ratio=min_valid_ratio,
                    secondary=secondary,
                    compute_metrics=True,
                    weighting=weighting,
                    fee_bps=fee_bps,
                    rf=rf
                )
                _, _, pf_frame_ext = _ensure_triplet(res_test_ext)
                pf_frame_test_raw = _trim_pf_frame_to_test_window(pf_frame_ext, primary_mode, te_s, te_e)
                # NEW: riempiamo i buchi con cash (ritorno 0) e ricalcoliamo equity
                pf_frame_test = _ensure_full_periods(pf_frame_test_raw, primary_mode, te_s, te_e, fill_mode="cash")
                if (pf_frame_test is None or pf_frame_test.empty) and verbose:
                    print(f"[WFO][{label_year}] Nessun periodo di applicazione nel TEST per combo {params}.")
            except Exception as e:
                if verbose: print(f"[WFO][{label_year}] Errore TEST combo {params}: {e}")
                continue

            met = _metrics_from_pf_frame(pf_frame_test, primary=primary_mode, rf=rf)
            # Filtro qualità OOS: almeno N periodi utili
            needed = (6 if primary_mode in ("M", "R") else 1) if min_periods_oos is None else int(min_periods_oos)
            if met["n_periods"] < needed:
                if verbose:
                    print(f"[WFO][{label_year}] Combo scartata (n_periods={met['n_periods']} < {needed}): {params}")
                continue

            met.update({
                "year_block": label_year,
                "train_start": tr_s, "train_end": tr_e,
                "test_start": te_s,  "test_end": te_e,
                "primary": primary_mode,
                "top_percentage": params.get("top_percentage"),
                "primary_lookback_months": primary_lookback_effective,  # None se non 'R'
                "secondary_lookback_months": int(sec_lb) if sec_lb is not None else None,
                "secondary_top": sec_top if sec_top is not None else None
            })
            block_results.append(met)

        if len(block_results) == 0:
            if verbose: print(f"[WFO][{label_year}] Nessun risultato valido.")
            continue

        block_df = pd.DataFrame(block_results)
        block_df["block"] = label_year
        all_results.append(block_df.copy())

        # ------- Winner per blocco (robusto, senza warning) -------
        rank_keys = ["primary", "top_percentage", "primary_lookback_months",
                     "secondary_lookback_months", "secondary_top"]

        # (opzionale) deduplica combo identiche su rank_keys
        block_df = block_df.drop_duplicates(subset=rank_keys, keep="first")

        # indice per ranking coeso (ordinato per evitare warning "lexsort depth")
        block_idx = block_df.set_index(rank_keys, drop=False).sort_index()

        # calcolo score
        if callable(selection_rule):
            score = selection_rule(block_idx)
            if not isinstance(score, pd.Series) or not score.index.equals(block_idx.index):
                raise ValueError("selection_rule custom deve restituire una Series indicizzata come block_df[rank_keys].")
        elif selection_rule == "composite":
            score = _composite_rank(block_idx, weights=selection_weights)
        else:
            metric = selection_rule
            asc = True if metric in ("ann_vol","max_dd") else False
            if metric not in block_idx.columns: raise ValueError(f"Metrica '{metric}' non presente.")
            score = block_idx[metric].rank(ascending=asc, method="average")

        # winner (stabile anche con pari merito)
        winner_key = score.sort_values(kind="mergesort").index[0]
        winner_row_df = block_idx.loc[[winner_key]]     # DataFrame 1+ righe in caso di duplicati
        winner_row = winner_row_df.iloc[0]              # prendi la prima (stabile)

        winners_params.append({k: winner_row[k] for k in rank_keys})

        # valori effettivi per la combo vincente
        win_primary = winner_row["primary"]
        win_lb_eff = None
        if pd.notna(winner_row["primary_lookback_months"]):
            try:
                win_lb_eff = int(winner_row["primary_lookback_months"])
            except Exception:
                win_lb_eff = None

        win_secondary = None
        if pd.notna(winner_row.get("secondary_lookback_months", np.nan)) and pd.notna(winner_row.get("secondary_top", np.nan)):
            win_secondary = {
                "lookback_months": int(winner_row["secondary_lookback_months"]),
                "top": winner_row["secondary_top"]
            }

        # Ricalcolo equity OOS del vincitore (ext + trim + fill cash) e normalizzo l'indice alla fine periodo
        res_test_win = select_top_performing_stocks(
            data=df.loc[_compute_test_prebuffer_start(te_s, win_primary, win_lb_eff): te_e],
            top_percentage=winner_row["top_percentage"],
            primary=win_primary,
            primary_lookback_months=win_lb_eff or 12,   # ignorato se non 'R'
            shift_forward=shift_forward,
            min_valid_ratio=min_valid_ratio,
            secondary=win_secondary,
            compute_metrics=True,
            weighting=weighting,
            fee_bps=fee_bps,
            rf=rf
        )
        _, _, pf_frame_test_win_ext = _ensure_triplet(res_test_win)
        pf_frame_test_win_raw = _trim_pf_frame_to_test_window(pf_frame_test_win_ext, win_primary, te_s, te_e)
        pf_frame_test_win = _ensure_full_periods(pf_frame_test_win_raw, win_primary, te_s, te_e, fill_mode="cash")
        if pf_frame_test_win is not None and not pf_frame_test_win.empty:
            eq_win = pf_frame_test_win["equity"]  # già DatetimeIndex (fine periodo)
            oos_equity_parts.append(eq_win)

        summary_rows.append({
            "year_block": label_year,
            "train": f"{tr_s.date()}→{tr_e.date()}",
            "test":  f"{te_s.date()}→{te_e.date()}",
            **{k: winner_row[k] for k in rank_keys},
            **{m: winner_row.get(m, np.nan) for m in ["n_periods","total_return","CAGR","ann_vol","Sharpe","Sortino","max_dd","Calmar"]}
        })

    # -------- Output finali --------
    summary_df = pd.DataFrame(summary_rows).sort_values("year_block").reset_index(drop=True)
    results_df = pd.concat(all_results, ignore_index=True) if len(all_results) > 0 else pd.DataFrame()

    if len(oos_equity_parts) > 0:
        oos_equity = pd.concat(oos_equity_parts).sort_index()
        oos_equity.name = "equity"
    else:
        oos_equity = pd.Series(dtype=float, name="equity")

    return summary_df, results_df, oos_equity, winners_params

wfo_universe_selector_momentum = wfo_optimize_selection 


