"""
s_functions.py — Refactored from notebooks/libs/s_functions.ipynb
"""

# =================================
# Rotational Strategies Functions
# =================================


import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from tqdm.auto import tqdm

# Dipendenze: u_functions deve essere importato nel contesto chiamante

# --- costanti & mapping frequenze ---
ANN = 252
ALLOWED_FREQS = {"BH","ME","QE","YE"}
FREQ_TO_LOOKBACK = {"BH":126, "ME":126, "QE":189, "YE":252}

# -------------------------
# utilità base / caricamento
# -------------------------
def _safe_sort_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep="last")].sort_index()

def load_prices_fixed_universe(
    tickers: List[str],
    start: str, end: Optional[str],
    min_valid_ratio: float = 0.9
) -> pd.DataFrame:
    """
    Universo fisso: scarica prezzi su tutto l'intervallo, filtra i ticker con copertura insufficiente,
    dropna finale per rimuovere qualsiasi buco residuo.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(
        tickers, start=start, end=end,
        progress=False, multi_level_index=False
    )["Close"]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = _safe_sort_index(df).dropna(how="all")

    n = len(df)
    min_days = int(n * min_valid_ratio)
    valid = df.notna().sum()
    keep  = valid[valid >= min_days].index.tolist()
    drop  = sorted(set(df.columns) - set(keep))
    if drop:
        print(f"⚠️ Esclusi {len(drop)} ticker per copertura insufficiente: {drop}")

    out = df[keep].dropna()
    print(f"✅ Universo fisso: {len(out.columns)} ticker")
    return out

# --------------
# metriche base
# --------------
def evaluate(returns: pd.Series, rf: float = 0.0) -> Dict[str,float]:
    r = returns.dropna()
    if r.empty:
        return dict(CAGR=0, AnnualReturn=0, MaxDD=0, Sharpe=0, Sortino=0, Calmar=0)
    eq    = (1+r).cumprod()
    years = len(r)/ANN
    cagr  = eq.iloc[-1]**(1/years)-1 if years>0 else 0
    # usare 'YE' (non 'Y') per evitare FutureWarning
    ar    = (1+r).resample("YE").prod().sub(1).mean()
    peak  = eq.cummax()
    dd    = ((eq-peak)/peak).min()
    ex    = r - rf/ANN
    shar  = np.sqrt(ANN)*ex.mean()/ex.std() if ex.std()!=0 else 0
    down  = r[r<0].std()*np.sqrt(ANN)
    sort  = ((r.mean()*ANN - rf)/down) if down!=0 else 0
    calm  = (-cagr/dd) if dd!=0 else 0
    return dict(CAGR=float(cagr), AnnualReturn=float(ar), MaxDD=float(dd),
                Sharpe=float(shar), Sortino=float(sort), Calmar=float(calm))

# ---------------------------------------
# motore segmentato (NO buchi, NO duplicati)
# ---------------------------------------
def make_rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "BH":
        return pd.Index([idx[0]])
    dates = idx.to_series().resample(freq).last().index
    if len(dates)==0 or dates[0] != idx[0]:
        dates = pd.Index([idx[0]]).append(dates)
    return dates

def segment_returns(price_df: pd.DataFrame,
                    freq: str,
                    lookback: int,
                    weight_func,
                    rf: float,
                    tc: float) -> Tuple[pd.Series, pd.Series]:
    """
    Applica una funzione di pesatura su finestre scorrevoli, con rebalance a freq.
    Ritorna: (rendimenti portafoglio, ultimi pesi stimati).
    """
    price_df = _safe_sort_index(price_df)
    rets = price_df.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float), pd.Series(0, index=price_df.columns, dtype=float)

    dates = make_rebalance_dates(rets.index, freq)
    w_prev = pd.Series(0, index=price_df.columns, dtype=float)
    out = []

    for i, dt in enumerate(dates):
        win_price = price_df.loc[:dt].iloc[-(lookback+1):]
        win_rets  = win_price.pct_change().dropna()

        if win_rets.shape[0] < 2:
            w = w_prev.copy()
        else:
            try:
                w = weight_func(price_win=win_price, ret_win=win_rets, rf=rf)
            except Exception:
                w = w_prev.copy()

        # normalizza e riallinea
        s = w.sum()
        if s>0: w = w/s
        w = w.reindex(price_df.columns).fillna(0).clip(lower=0)

        nxt = dates[i+1] if i+1<len(dates) else None
        seg = rets.loc[dt:] if nxt is None else rets.loc[dt:nxt].iloc[:-1]

        if not seg.empty:
            trn = (w - w_prev).abs().sum()
            seg = seg.mul(w, axis=1).sum(axis=1)
            seg.iloc[0] -= trn * tc
            out.append(seg)

        w_prev = w

    port = pd.concat(out) if out else pd.Series(dtype=float)
    port = port[~port.index.duplicated(keep="last")].sort_index()
    return port, w_prev

# ---------------------------------------
# Metodi di pesatura (robusti)
# ---------------------------------------
def equal_weight(returns: pd.DataFrame, **_) -> pd.Series:
    n = returns.shape[1]
    if n == 0: return pd.Series(dtype=float)
    return pd.Series(1/n, index=returns.columns)

def risk_parity_weights(returns: pd.DataFrame, **_) -> pd.Series:
    vol = returns.std()
    inv = 1/vol.replace(0, np.nan)
    inv = inv.fillna(0)
    if inv.sum() == 0:
        return equal_weight(returns)
    return inv/inv.sum()

def hierarchical_clustering_weights(returns: pd.DataFrame, n_clusters: int = 3, **_) -> pd.Series:
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        corr = returns.corr()
        dist = np.sqrt(0.5*(1-corr))
        Z = linkage(squareform(dist), method="ward")
        clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
        tickers = returns.columns
        cl_map = pd.Series(clusters, index=tickers)
        weights = pd.Series(0.0, index=tickers)
        for cl in sorted(cl_map.unique()):
            members = cl_map[cl_map==cl].index
            if len(members)==0: continue
            w_cluster = 1.0/len(cl_map.unique())
            weights.loc[members] = w_cluster/len(members)
        s = weights.sum()
        if s>0: weights = weights/s
        return weights.fillna(0)
    except Exception:
        return equal_weight(returns)

def hrp_allocation_weights(returns: pd.DataFrame, **_) -> pd.Series:
    try:
        from pypfopt.hierarchical_portfolio import HRPOpt
        hrp = HRPOpt(returns)
        w = hrp.optimize()
        w = pd.Series(hrp.clean_weights(), index=returns.columns)
        s = w.sum()
        if s>0: w=w/s
        return w.fillna(0).clip(lower=0)
    except Exception:
        return equal_weight(returns)

def momentum_weights(returns: pd.DataFrame, **_) -> pd.Series:
    mom = (1+returns).prod()-1
    mom = mom.clip(lower=0).fillna(0)
    if mom.sum() == 0:
        return equal_weight(returns)
    return mom/mom.sum()

def risk_adjusted_momentum_weights(returns: pd.DataFrame, rf: float=0.0, **_) -> pd.Series:
    mean_ann = returns.mean()*ANN
    down_std = returns[returns<0].std()*np.sqrt(ANN)
    sortino  = (mean_ann - rf)/down_std.replace(0, np.nan)
    sortino  = sortino.clip(lower=0).fillna(0)
    if sortino.sum() == 0:
        return equal_weight(returns)
    return sortino/sortino.sum()

def minimum_correlation_weights(returns: pd.DataFrame, **_) -> pd.Series:
    corr = returns.corr()
    avg_corr = corr.apply(lambda s: s.drop(s.name).mean(), axis=0)
    inv = 1/avg_corr.replace(0, np.nan)
    inv = inv.clip(lower=0).fillna(0)
    if inv.sum() == 0:
        return equal_weight(returns)
    return inv/inv.sum()

def equal_risk_contribution_weights(returns: pd.DataFrame, **_) -> pd.Series:
    try:
        import scipy.optimize as sco
        cov = returns.cov().values
        n = cov.shape[0]
        if n == 0: return pd.Series(dtype=float, index=returns.columns)

        def port_var(w, C): return w @ C @ w
        def mrc(w, C): return C @ w
        def risk_contrib(w, C):
            tv = np.sqrt(port_var(w,C))
            return w * mrc(w,C) / (tv + 1e-12)

        def obj(w, C):
            rc = risk_contrib(w,C)
            return np.sum((rc - rc.mean())**2)

        cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
        bnds = [(0,1)]*n
        w0   = np.ones(n)/n
        res  = sco.minimize(obj, w0, args=(cov,), bounds=bnds, constraints=cons, tol=1e-9)
        if not res.success:
            return equal_weight(returns)
        w = pd.Series(res.x, index=returns.columns)
        return w.fillna(0).clip(lower=0)
    except Exception:
        return equal_weight(returns)

# ---------------------------------------
# ADAPTER: metodi “grandi” -> motore segment_returns
# ---------------------------------------
def _method_adapter(fn):
    """Adatta funzioni che ricevono 'returns' (e opz mu/cov/rf) alla firma (price_win, ret_win, rf)."""
    from pypfopt import expected_returns, risk_models
    def wrapper(price_win: pd.DataFrame, ret_win: pd.DataFrame, rf: float=0.0):
        try:
            mu  = expected_returns.mean_historical_return(price_win)
            cov = risk_models.CovarianceShrinkage(price_win).ledoit_wolf()
            w = fn(returns=ret_win, mu=mu, cov=cov, rf=rf)
        except TypeError:
            w = fn(returns=ret_win, rf=rf)
        # normalizza/realinea
        w = pd.Series(w, index=price_win.columns) if not isinstance(w, pd.Series) else w.reindex(price_win.columns)
        w = w.fillna(0).clip(lower=0)
        s = w.sum()
        if s>0: w = w/s
        return w
    return wrapper

def build_methods_dict() -> dict:
    """Metodi richiesti (niente MaxSharpe/MinVar/BL)."""
    return {
        "Equal":                   _method_adapter(equal_weight),
        "RiskParity":              _method_adapter(risk_parity_weights),
        "HierarchicalClustering":  _method_adapter(hierarchical_clustering_weights),
        "HRP":                     _method_adapter(hrp_allocation_weights),
        "Momentum":                _method_adapter(momentum_weights),
        "SortinoMomentum":         _method_adapter(risk_adjusted_momentum_weights),
        "MinCorr":                 _method_adapter(minimum_correlation_weights),
        "ERC":                     _method_adapter(equal_risk_contribution_weights),
    }

# ---------------------------------------
# selezione in-sample
# ---------------------------------------
def select_best_method(price_train: pd.DataFrame,
                       rf: float, tc: float,
                       freqs_to_test: List[str],
                       METHODS: dict) -> Tuple[str,str,Dict[str,Dict[str,float]]]:
    """Ritorna (best_freq, best_method, metrics_dict). Scoring: CAGR."""
    metrics = {}
    for freq in freqs_to_test:
        look = FREQ_TO_LOOKBACK[freq]
        for name, fn in METHODS.items():
            port, _ = segment_returns(price_train, freq, look, fn, rf, tc)
            met = evaluate(port, rf)
            metrics[f"{name}@{freq}"] = met
    scores = pd.Series({k:v["CAGR"] for k,v in metrics.items()})
    best_key = scores.sort_values(ascending=False).index[0]
    best_method, best_freq = best_key.split("@")
    return best_freq, best_method, metrics

# ---------------------------------------
# WFO (verbosa + fix anti-buchi)
# ---------------------------------------

def s_walk_forward(
    tickers: List[str],
    start: str, end: Optional[str],
    train_years:int=3, test_years:int=1,
    rf: float=0.0, tc: float=0.001,
    rebalance_freqs: Optional[List[str]] = None,
    min_valid_ratio: float=0.9,
    methods_dict: Optional[dict] = None,
    verbose: bool = False
):
    # Frequenze
    if rebalance_freqs is None:
        freqs_to_test = ["ME","QE","YE","BH"]
    else:
        freqs_to_test = [f for f in map(str.upper, rebalance_freqs) if f in ALLOWED_FREQS]
        if not freqs_to_test:
            freqs_to_test = ["ME"]

    # Metodi
    METHODS = methods_dict if methods_dict is not None else build_methods_dict()

    # Prezzi universo fisso
    prices = load_prices_fixed_universe(tickers, start, end, min_valid_ratio=min_valid_ratio)
    dates  = pd.date_range(start=start, end=end or datetime.today(), freq="YS")

    all_rets, rows, all_w = [], [], []

    if verbose:
        print(f"\n🔧 Frequenze da testare: {freqs_to_test}")
        print(f"🔧 Metodi considerati: {list(METHODS.keys())}")

    # Numero reale di walk (ultimi 'train_years' non hanno train completo)
    total_walks = max(len(dates) - train_years, 0)

    # Loop con tqdm sul numero reale di walk
    for i in tqdm(range(total_walks), total=total_walks, unit="walk", desc="Walk-Forward Progress", leave=True):
        tr_s, tr_e = dates[i], dates[i+train_years] - pd.Timedelta(days=1)
        te_s_i, te_e_i = i+train_years, i+train_years+test_years
        te_s = dates[te_s_i] if te_s_i < len(dates) else None
        te_e = dates[te_e_i] - pd.Timedelta(days=1) if te_e_i < len(dates) else None

        # Aggiorna descrizione dinamica (anno→anno) senza rompere la barra
        # NB: tqdm.write per messaggi verbosi fuori dalla barra
        if verbose:
            tqdm.write(f"\n⏱️ Walk {i+1}/{total_walks}: Train {tr_s.date()} → {tr_e.date()} | "
                       f"Test {te_s.date() if te_s else 'n/a'} → {te_e.date() if te_e else 'n/a'}")

        price_train = prices.loc[tr_s:tr_e]
        price_test  = prices.loc[te_s:te_e] if (te_s and te_e) else pd.DataFrame(index=[])

        # --- selezione in-sample ---
        best_freq, best_method, metrics_dict = select_best_method(price_train, rf, tc, freqs_to_test, METHODS)
        look = FREQ_TO_LOOKBACK[best_freq]
        fn   = METHODS[best_method]

        if verbose:
            tqdm.write(f"   → Winner in-sample: {best_method}@{best_freq} (lookback={look})")
            tqdm.write("   → Metriche in-sample (CAGR, Sharpe, Calmar):")
            for k,v in metrics_dict.items():
                tqdm.write(f"     {k}: CAGR={v['CAGR']:.2%}, Sharpe={v['Sharpe']:.2f}, Calmar={v['Calmar']:.2f}")

        # --- applicazione OOS ---
        if not price_test.empty:
            tail_train = price_train.iloc[-(look+1):] if len(price_train)>0 else pd.DataFrame(index=[])
            price_ext  = _safe_sort_index(pd.concat([tail_train, price_test], axis=0))
            port_ext, w_last = segment_returns(price_ext, best_freq, look, fn, rf, tc)
            port = port_ext.loc[price_test.index.intersection(port_ext.index)]
            if not port.empty:
                all_rets.append(port)
            if verbose:
                sharpe = (port.mean()/port.std()*np.sqrt(252)) if port.std() not in (0, np.nan) else np.nan
                tqdm.write(f"   → Test OOS: {len(port)} giorni. "
                           f"Rendimento medio {port.mean():.4f}, Sharpe ~{sharpe:.2f}")
        else:
            _, w_last = segment_returns(price_train.iloc[-(look+1):], best_freq, look, fn, rf, tc)
            if verbose:
                tqdm.write("   Nessun periodo di test, salvati pesi dal train.")

        # log blocco
        rows.append({
            "TrainStart": tr_s.date(), "TrainEnd": tr_e.date(),
            "TestStart":  te_s.date() if te_s else None,
            "TestEnd":    te_e.date() if te_e else None,
            "Method": best_method, "Freq": best_freq
        })

        # salva pesi
        colname = f"WFO_{te_s.year}" if te_s else "Forecast"
        w_last.name = colname
        all_w.append(w_last)

    wf_rets = pd.concat(all_rets).sort_index() if all_rets else pd.Series(dtype=float)
    summary = pd.DataFrame(rows)
    weights = pd.DataFrame(all_w).T.fillna(0)

    if verbose:
        print(f"\n✅ Walk-forward completata: {len(summary)} blocchi, {len(wf_rets)} rendimenti OOS.")
    return wf_rets, summary, weights

wfo_method_rotation_allocation = s_walk_forward

    
def print_walkforward_summary(rets, summary, weights, title="📈 Performance aggregata (walk-forward)"):

    # Performance complessive
    perf = evaluate(rets)

    print(title)

    df = pd.DataFrame({
        "Metriche": ["CAGR", "AnnualReturn", "MaxDD", "Sharpe", "Sortino", "Calmar"],
        "Valore":   [f"{perf['CAGR']:.2%}", f"{perf['AnnualReturn']:.2%}", f"{perf['MaxDD']:.2%}",
                     f"{perf['Sharpe']:.3f}", f"{perf['Sortino']:.3f}", f"{perf['Calmar']:.3f}"]
    })
    display(df)


    print("\n📊 Riepilogo WFO:")
    display(summary)

    print("\n📦 Pesi per ciascun periodo (fonte unica del piano):")
    display(
        weights.style.format("{:.2%}")
                 .bar(axis=0, color="lightblue")
                 .set_caption("Pesi allocati nei test walk-forward")
    )

# -------------------------
# Piano operativo annuale
# -------------------------
def generate_portfolio_plan(df_summary: pd.DataFrame,
                            df_weights: pd.DataFrame,
                            capitale: float = 10_000,
                            max_titoli: int = None,
                            valuta: str = "€") -> pd.DataFrame:
    """
    Piano operativo: usa *solo* i pesi salvati WFO_YYYY (coerente con la WFO).
    Se max_titoli=None, include tutti i titoli con peso > 0.
    """
    plans = []
    for col in df_weights.columns:
        yr = col.split("_")[1] if col.startswith("WFO_") else None
        row = (df_summary[df_summary["TestStart"].astype(str).str.startswith(str(yr))].iloc[0]
               if yr else None)
        method = row["Method"] if row is not None else "n/a"
        freq   = row["Freq"] if row is not None else "n/a"

        w = df_weights[col].clip(lower=0)      # evita pesi negativi
        w = w[w > 0]                           # scarta titoli con peso 0%
        s = w.sum()
        if s > 0:
            w = w / s
        if max_titoli is not None:
            w = w.sort_values(ascending=False).head(max_titoli)
        else:
            w = w.sort_values(ascending=False)

        lines = []
        for t, val in w.items():
            euro = val * capitale
            lines.append(f"{t}: {val*100:.2f}% — {valuta}{euro:,.2f}")
        alloc = "<br>".join(lines)

        plans.append(dict(
            Anno = int(yr) if yr else "Forecast",
            Strategia = method,
            Frequenza = freq,
            Capitale = f"{valuta}{capitale:,.0f}",
            Allocazione = alloc
        ))
    return pd.DataFrame(plans).sort_values("Anno")



