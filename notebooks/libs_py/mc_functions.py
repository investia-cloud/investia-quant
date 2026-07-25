"""
mc_functions.py — Refactored from notebooks/libs/mc_functions.ipynb
"""

from pypfopt import EfficientFrontier, risk_models, expected_returns
import vectorbt as vbt
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Optional, Dict, List, Union, Any, Tuple
import plotly.graph_objects as go
from u_functions import (build_and_plot_portfolio_contributions, download_data, load_ohlcv, generate_lazy_portfolio_performance, plot_cumulative_and_rolling_returns, plot_monthly_returns, plot_multiple_portfolios, my_display)
from r_functions import mc_run_iid_bootstrap, mc_run_block_bootstrap

# Fallback sicuro per display(): usa quello di IPython in Jupyter,
# altrimenti ripiega su print() quando il modulo è importato da script Python puro.
try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        for a in args:
            print(a)

def optimize_portfolio(tickers, start_date, end_date, target_metric, goal='max', num_trials=1000, init_cash=10000):
    """
    Trova i pesi ottimali per un portafoglio Buy & Hold che massimizzano/minimizzano
    una metrica (es. Sharpe Ratio, Max Drawdown, Total Return) via simulazione Monte Carlo.

    Args:
        tickers (list): Lista dei ticker da includere nel portafoglio.
        start_date (str): Data di inizio del backtest ('YYYY-MM-DD').
        end_date (str): Data di fine del backtest ('YYYY-MM-DD').
        target_metric (str): La metrica da ottimizzare. Deve essere tra quelle restituite da pf.stats().
        goal (str): 'max' per massimizzare, 'min' per minimizzare la metrica.
        num_trials (int): Numero di combinazioni di pesi casuali da testare.
        init_cash (float): Capitale iniziale del portafoglio.

    Returns:
        dict: Dizionario con i pesi ottimali, le performance e l’oggetto Portfolio ottimale.
    """
    if not isinstance(tickers, list) or len(tickers) < 2:
        raise ValueError("'tickers' deve essere una lista di almeno due simboli.")

    print(f"\nScaricamento dati per {len(tickers)} asset...")
    price_data = load_ohlcv(tickers, start=start_date, end=end_date)['Close'].dropna()
    num_assets = len(tickers)

    print(f"Esecuzione Monte Carlo con {num_trials} simulazioni...\n")
    portfolios = []
    stats_list = []

    random_weights = np.random.random(size=(num_trials, num_assets))
    normalized_weights = random_weights / np.sum(random_weights, axis=1, keepdims=True)

    first_prices = price_data.iloc[0].values

    for i in tqdm(range(num_trials), desc="Simulazioni"):
        weights_i = normalized_weights[i]
        allocated_cash = weights_i * init_cash
        sizes = allocated_cash / first_prices

        size_matrix = np.zeros_like(price_data.values)
        size_matrix[0] = sizes

        pf = vbt.Portfolio.from_orders(
            close=price_data,
            size=size_matrix,
            init_cash=init_cash,
            fees=0.001,
            freq='D'
        )

        portfolios.append(pf)
        stats_list.append(pf.stats())

    # Estrazione metrica target
    metric_values = []
    for stats in stats_list:
        if target_metric not in stats.index:
            raise ValueError(f"Metrica '{target_metric}' non trovata.")
        metric_values.append(stats.loc[target_metric])

    metric_values = np.array(metric_values)

    best_trial_idx = np.argmax(metric_values) if goal == 'max' else np.argmin(metric_values)
    best_weights_array = normalized_weights[best_trial_idx]
    best_stats = stats_list[best_trial_idx]
    optimal_weights_composition = dict(zip(tickers, best_weights_array))

    # Esegui backtest finale
    optimal_portfolio_object = run_bh_backtest(optimal_weights_composition, start_date, end_date, init_cash=init_cash)

    print("\n✅ Ottimizzazione completata.")
    print(f"📈 Metrica ottimizzata: {target_metric} ({'massimizzata' if goal == 'max' else 'minimizzata'})")
    print(f"🔢 Combinazioni testate: {num_trials}")
    print("\n--- ⚖️  Pesi Ottimali ---")
    for ticker, weight in optimal_weights_composition.items():
        print(f"{ticker:<10} → {weight:.2%}")

    print("\n--- 📊 Performance Portafoglio Ottimale ---")
    print(best_stats[[s for s in best_stats.index if '%' in s or 'Sharpe' in s or 'Return' in s]])

    return {
        "optimal_weights": optimal_weights_composition,
        "best_performance": best_stats,
        "optimal_portfolio_object": optimal_portfolio_object
    }
    
def _prepare_bh_data(portfolio: dict, start_date: str, end_date: str,
                      min_years: int = 5) -> tuple:
    """
    Estrae pesi, scarica i prezzi UNA VOLTA e applica tutti i guard di
    validazione (ticker mancanti, allineamento, storico minimo).
    Funzione privata condivisa da run_bh_backtest e compare_rebalance_frequencies
    per evitare download ripetuti sugli stessi ticker/range.

    Returns
    -------
    (weights, price_aligned) — entrambi None se un guard fallisce
    (diagnostica già stampata).
    """
    # Estrazione pesi: supporta sia il nuovo formato annidato sia il vecchio formato flat
    if "tickers" in portfolio:
        weights_dict = dict(portfolio["tickers"])
    else:
        weights_dict = dict(portfolio)
    for k in list(weights_dict.keys()):
        weights_dict[k.upper()] = weights_dict.pop(k)

    tickers = list(weights_dict.keys())
    weights = pd.Series(weights_dict, index=tickers, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("La somma dei pesi deve essere 1.")

    price = download_data(tickers, start_date=start_date, end_date=end_date)
    if isinstance(price, pd.Series):
        price = price.to_frame(name=tickers[0])

    fully_missing = [t for t in tickers if t in price.columns and price[t].isna().all()]
    if fully_missing:
        print(
            f"[run_bh_backtest] Ticker senza alcun dato disponibile (né yfinance né cache locale): "
            f"{fully_missing}. Procurare CSV NAV in inputs/fund_nav/ o rimuovere dal portfolio prima di procedere."
        )
        return None, None

    coverage_before_align = price.notna().sum().sort_values()
    price_aligned = price.dropna(how='any')

    if price_aligned.empty:
        print("[run_bh_backtest] Nessuna data con dati completi per tutti gli asset. Skip.")
        return None, None

    MIN_COMMON_DAYS = 252 * min_years
    if len(price_aligned) < MIN_COMMON_DAYS:
        worst_tickers = coverage_before_align[
            coverage_before_align < coverage_before_align.max() * 0.5
        ].to_dict()
        print(
            f"[run_bh_backtest] Storico insufficiente — skip.\n"
            f"  Date comuni con dati completi: {len(price_aligned)} "
            f"(minimo richiesto: {MIN_COMMON_DAYS}, ~{min_years} anni)\n"
            f"  Periodo risultante: {price_aligned.index.min()} → {price_aligned.index.max()}\n"
            f"  Ticker con copertura scarsa rispetto agli altri: {worst_tickers}\n"
            f"  Copertura completa: {coverage_before_align.to_dict()}\n"
            f"  → Procurare storico più ampio (CSV NAV in inputs/fund_nav/) per i ticker indicati, "
            f"o restringere l'universo del portfolio."
        )
        return None, None

    return weights, price_aligned


def run_bh_backtest(
    portfolio: dict,
    start_date: str,
    end_date: str,
    init_cash: float = 10_000,
    fees: float = 0.001,
    rebalance_freq: str = None,
    min_years: int = 5,
    _preloaded: tuple = None,
) -> vbt.Portfolio:
    """
    Backtest Buy & Hold robusto con VectorBT, solo su dati completamente validi.
    - I dati sono pre-allineati per evitare NaN.
    - Il ribilanciamento avviene solo in date con dati completi.
    - Perfettamente confrontabile con Pandas.
    portfolio: dict nel nuovo formato {"Title": str, "tickers": {ticker: peso}, "benchmark": str}
               oppure nel vecchio formato flat {ticker: peso} (retrocompatibile).
    min_years: storico minimo comune richiesto (in anni) tra tutti i ticker
               dopo l'allineamento. Default 5 — abbassare (es. min_years=1)
               solo per confronti esplorativi rapidi, non per analisi finali.
    _preloaded: uso interno — (weights, price_aligned) già calcolati da
               _prepare_bh_data, per evitare un nuovo download quando questa
               funzione è chiamata più volte sugli stessi dati (vedi
               compare_rebalance_frequencies). Non passare manualmente.

    Ritorna None (con stampa diagnostica) se lo storico disponibile è
    insufficiente per un backtest affidabile — nessuna eccezione sollevata.
    """
    if _preloaded is not None:
        weights, price = _preloaded
    else:
        weights, price = _prepare_bh_data(portfolio, start_date, end_date, min_years)
        if price is None:
            return None

    # 4. Costruzione size: DataFrame con target percent
    size = pd.DataFrame(np.nan, index=price.index, columns=price.columns)
    # 5. Date di ribilanciamento
    if rebalance_freq is None:
        reb_dates = pd.DatetimeIndex([price.index[0]])
    else:
        rf = str(rebalance_freq).upper()
        if rf in ["Y", "A", "YE"]:
            reb_dates = price.groupby(price.index.year).apply(lambda x: x.index[-1])
            reb_dates = pd.DatetimeIndex(reb_dates.values)
        else:
            periods = price.index.to_period(rebalance_freq)
            reb_dates = price.index[~periods.duplicated()]
        if price.index[0] not in reb_dates:
            reb_dates = reb_dates.insert(0, price.index[0])
    reb_dates = reb_dates.intersection(price.index)
    for d in reb_dates:
        size.loc[d] = weights
    # 6. Costruzione del portafoglio VectorBT
    pf = vbt.Portfolio.from_orders(
        close=price,
        size=size,
        size_type='targetpercent',
        init_cash=init_cash,
        fees=fees,
        cash_sharing=True,
        freq='D'
    )
    return pf


def _pf_order_stats(pf) -> tuple:
    """
    Estrae numero di operazioni e commissioni totali pagate da un
    vbt.Portfolio, con fallback robusto se l'API cambia struttura
    (best-effort: non blocca il resto del confronto se fallisce).
    """
    try:
        n_orders = int(np.sum(pf.orders.count()))
    except Exception:
        n_orders = np.nan
    try:
        fees_raw = pf.orders.fees.sum()
        fees_paid = float(np.sum(fees_raw)) if hasattr(fees_raw, '__iter__') else float(fees_raw)
    except Exception:
        fees_paid = np.nan
    return n_orders, fees_paid


def _pf_order_stats(pf) -> tuple:
    """
    Estrae numero di operazioni e commissioni totali pagate da un
    vbt.Portfolio, con fallback robusto se l'API cambia struttura
    (best-effort: non blocca il resto del confronto se fallisce).
    """
    try:
        n_orders = int(np.sum(pf.orders.count()))
    except Exception:
        n_orders = np.nan
    try:
        fees_raw = pf.orders.fees.sum()
        fees_paid = float(np.sum(fees_raw)) if hasattr(fees_raw, '__iter__') else float(fees_raw)
    except Exception:
        fees_paid = np.nan
    return n_orders, fees_paid


def compare_rebalance_frequencies(
    portfolio: dict,
    start_date: str,
    end_date: str,
    init_cash: float = 10_000,
    fees: float = 0.001,
    freqs: list = None,
    min_years: int = 1,
    selection_metric: str = 'Sharpe',
    tie_break_tolerance: float = 0.01,
    verbose: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Confronta le performance di un portfolio Buy&Hold su diverse frequenze
    di ribilanciamento, selezionando automaticamente quella con la metrica
    migliore (default: Sharpe massimo) — con tie-break verso minor turnover.

    I dati vengono scaricati UNA SOLA VOLTA e riusati per ogni backtest.

    La tabella include, oltre alle metriche di rendimento/rischio, il
    numero di operazioni (totale e annualizzato), le commissioni pagate,
    e il Calmar ratio (CAGR/MaxDD).

    Criterio di selezione (tie-break):
    Tra tutte le frequenze il cui `selection_metric` è entro
    `tie_break_tolerance` (relativo) dal valore massimo osservato,
    viene scelta quella con il minor numero di operazioni annualizzate
    (Ops_Anno) — non necessariamente quella con la metrica assoluta più
    alta. Motivazione: differenze di Sharpe/CAGR nell'ordine del
    millesimo tra frequenze vicine sono spesso rumore statistico più
    che un vantaggio reale; a parità sostanziale, meno operazioni
    significa meno costi di transazione non modellati (spread,
    slippage) e minore complessità operativa.

    Parameters
    ----------
    [... invariati ...]
    tie_break_tolerance : float
        Tolleranza relativa (frazione, es. 0.01 = 1%) sotto la quale
        due frequenze sono considerate "sostanzialmente equivalenti"
        su selection_metric. Tra le equivalenti, vince quella con
        Ops_Anno minore. tolerance=0 disabilita il tie-break (torna
        al puro idxmax()).

    Returns
    -------
    (freq_df, best_freq) : tuple
        freq_df colonne: [Freq, Sharpe, Calmar, CAGR%, TotalReturn%,
        MaxDD%, N_Ops, Ops_Anno, Fees_Paid, Fees_pct]
        best_freq selezionata col criterio tie-break sopra descritto.
    """
    if freqs is None:
        freqs = ['W', 'M', 'Q', 'Y', None]

    weights, price = _prepare_bh_data(portfolio, start_date, end_date, min_years)
    if price is None:
        if verbose:
            print("⚠️  Impossibile procedere — storico insufficiente per questo portfolio (vedi dettagli sopra).")
        return None, None

    rows = []
    for freq in freqs:
        pf = run_bh_backtest(portfolio, start_date, end_date,
                             init_cash, fees, freq, min_years=min_years,
                             _preloaded=(weights, price))
        if pf is None:
            break

        eq = pf.value()
        if isinstance(eq, pd.DataFrame):
            eq = eq.iloc[:, 0]
        eq = eq.dropna()

        yrs = len(eq) / 252
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
        max_dd = abs(float(pf.max_drawdown()))
        calmar = (cagr / max_dd) if max_dd > 0 else np.nan

        n_ops, fees_paid = _pf_order_stats(pf)
        ops_per_year = round(n_ops / yrs, 1) if (not np.isnan(n_ops) and yrs > 0) else np.nan
        fees_pct = round(fees_paid / init_cash * 100, 3) if not np.isnan(fees_paid) else np.nan

        rows.append({
            'Freq'        : freq if freq is not None else 'BH',
            'Sharpe'      : float(pf.sharpe_ratio()),
            'Calmar'      : round(calmar, 2) if not np.isnan(calmar) else np.nan,
            'CAGR%'       : round(cagr * 100, 2),
            'TotalReturn%': round(float(pf.total_return()) * 100, 2),
            'MaxDD%'      : round(max_dd * 100, 2),
            'N_Ops'       : n_ops,
            'Ops_Anno'    : ops_per_year,
            'Fees_Paid'   : round(fees_paid, 2) if not np.isnan(fees_paid) else np.nan,
            'Fees_pct'    : fees_pct,
        })

    if not rows:
        if verbose:
            print("⚠️  Impossibile procedere — storico insufficiente per questo portfolio (vedi dettagli sopra).")
        return None, None

    freq_df = pd.DataFrame(rows)
    if verbose:
        my_display(freq_df, title="Confronto frequenze di ribilanciamento")

    # --- Selezione con tie-break verso minor turnover ---
    metric_vals = freq_df[selection_metric]
    best_val = metric_vals.max()
    if tie_break_tolerance > 0 and best_val != 0:
        within_tolerance = freq_df[
            metric_vals >= best_val * (1 - tie_break_tolerance)
        ]
    else:
        within_tolerance = freq_df.loc[[metric_vals.idxmax()]]

    if 'Ops_Anno' in within_tolerance.columns and within_tolerance['Ops_Anno'].notna().any():
        best_row = within_tolerance.loc[within_tolerance['Ops_Anno'].idxmin()]
    else:
        best_row = freq_df.loc[metric_vals.idxmax()]

    best_label = best_row['Freq']
    best_freq  = None if best_label == 'BH' else best_label

    if verbose:
        n_candidates = len(within_tolerance)
        if n_candidates > 1:
            print(f"\n✅ Frequenza ottimale ({selection_metric}, tie-break su Ops_Anno tra "
                  f"{n_candidates} candidate entro {tie_break_tolerance:.1%}): {best_label}")
        else:
            print(f"\n✅ Frequenza ottimale ({selection_metric}): {best_label}")

    return freq_df, best_freq
    
# def run_bh_backtest(
#     weights_dict: dict,
#     start_date: str,
#     end_date: str,
#     init_cash: float = 10_000,
#     fees: float = 0.001,
#     rebalance_freq: str = None
# ) -> vbt.Portfolio:
#     """
#     Backtest Buy & Hold robusto con VectorBT, solo su dati completamente validi.

#     - I dati sono pre-allineati per evitare NaN.
#     - Il ribilanciamento avviene solo in date con dati completi.
#     - Perfettamente confrontabile con Pandas.
#     """

#     # Set Uppercase
#     for k in list(weights_dict.keys()):
#         weights_dict[k.upper()] = weights_dict.pop(k)

#     # 1. Validazione pesi
#     tickers = list(weights_dict.keys())

#     weights = pd.Series(weights_dict, index=tickers, dtype=float)
    
#     if not np.isclose(weights.sum(), 1.0):
#         raise ValueError("La somma dei pesi deve essere 1.")

#     # 2. Scarica i dati da yfinance
#     # data = yf.download(tickers, start=start_date, end=end_date, progress=False)
#     # # display(data)
#     # # price = data["Close"][tickers]
#     # price = data["Close"]
#     # price.columns.name = None
    
#     price=download_data(tickers, start_date=start_date, end_date=end_date)
    
#     # 3. Allinea: elimina ogni giorno con dati mancanti
#     price = price.dropna(how='any')

#     # display(price.head(),price.tail())
    
#     if price.empty:
#         raise ValueError("Nessuna data con dati completi per tutti gli asset.")

#     # 4. Costruzione size: DataFrame con target percent
#     size = pd.DataFrame(np.nan, index=price.index, columns=price.columns)

#     # 5. Date di ribilanciamento
#     if rebalance_freq is None:
#         reb_dates = pd.DatetimeIndex([price.index[0]])
#     else:
#         rf = str(rebalance_freq).upper()
    
#         if rf in ["Y", "A", "YE"]:
#             # ultimo trading day di ogni anno (robusto)
#             reb_dates = price.groupby(price.index.year).apply(lambda x: x.index[-1])
#             reb_dates = pd.DatetimeIndex(reb_dates.values)
#         else:
#             periods = price.index.to_period(rebalance_freq)
#             reb_dates = price.index[~periods.duplicated()]
    
#         # assicura inclusione start
#         if price.index[0] not in reb_dates:
#             reb_dates = reb_dates.insert(0, price.index[0])
    
#     # IMPORTANTISSIMO: garantisci che tutte le reb_dates siano nel calendario prezzi
#     reb_dates = reb_dates.intersection(price.index)  

#     # if rebalance_freq is None:
#     #     reb_dates = [price.index[0]]
#     # else:
#     #     periods = price.index.to_period(rebalance_freq)
#     #     reb_dates = price.index[~periods.duplicated()]
#     #     if price.index[0] not in reb_dates:
#     #         reb_dates = reb_dates.insert(0, price.index[0])

#     for d in reb_dates:
#         size.loc[d] = weights

#     # 6. Costruzione del portafoglio VectorBT
#     pf = vbt.Portfolio.from_orders(
#         close=price,
#         size=size,
#         size_type='targetpercent',
#         init_cash=init_cash,
#         fees=fees,
#         cash_sharing=True,
#         freq='D'
#     )

#     return pf

def compute_portfolio_returns_pandas(
    weights_dict: dict,
    start_date: str,
    end_date: str
) -> tuple[pd.Series, float, pd.Series]:
    """
    Calcola i rendimenti totali per singolo asset e del portafoglio (Buy & Hold)
    con allineamento temporale rigoroso.

    Args:
        weights_dict (dict): Dizionario {ticker: peso}, la somma deve essere 1.
        start_date (str): Data inizio (formato 'YYYY-MM-DD').
        end_date (str): Data fine (formato 'YYYY-MM-DD').

    Returns:
        asset_returns (pd.Series): Rendimento totale per asset (end / start - 1)
        portfolio_return (float): Rendimento cumulato del portafoglio
        portfolio_curve (pd.Series): Valore cumulato del portafoglio (base 1)
    """
    tickers = list(weights_dict.keys())
    weights = pd.Series(weights_dict)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("I pesi devono sommare a 1.")

    # 1. Scarica i prezzi adjusted
    data = load_ohlcv(tickers, start=start_date, end=end_date)
    price = data["Close"]  # già adjusted in versioni recenti di yfinance

    # 2. Allinea: solo le date comuni a tutti gli asset
    price = price[tickers].dropna(how='any')

    # 3. Calcola i rendimenti totali per asset (dal primo alultimo giorno della propria serie valida)
    asset_returns = price.apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)

    # 4. Normalizza i prezzi (base 1) dalla prima data comune
    price_norm = price / price.iloc[0]

    # 5. Calcola curva del portafoglio (Buy & Hold pesato)
    portfolio_curve = (price_norm * weights).sum(axis=1)
    portfolio_return = portfolio_curve.iloc[-1] - 1

    return asset_returns, portfolio_return, portfolio_curve
    
# from decimal import Decimal
def efficient_frontier_pypfopt(
    tickers: list,
    years: int = 10,
    n_points: int = 50,
    weight_bounds: tuple = (0, 1),
    show_plot: bool = True,
    interactive: bool = True,
    print_weights: bool = True,
    weights=None,
    fig_width: int = 1200,
    fig_height: int = 600,
    compute_real_annual_return: bool = True,
    start_date=None,
    end_date=None,
    _preloaded_price=None,
) -> dict:

    if len(tickers) < 2:
        print(
            "[efficient_frontier_pypfopt] Analisi non applicabile con un solo ticker "
            "— con un singolo asset non esiste una frontiera efficiente (nessuna "
            "combinazione di pesi possibile). Skip. Per singolo titolo/fondo usa "
            "generate_lazy_portfolio_performance."
        )
        return None, None

    if _preloaded_price is not None:
        price = _preloaded_price
    else:
        # Calcola date di inizio e fine
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = pd.to_datetime(end_date).to_pydatetime()
        if start_date is None:
            start_date = datetime(end_date.year - years, 1, 1)
        else:
            start_date = pd.to_datetime(start_date).to_pydatetime()

        price = load_ohlcv(tickers, start=start_date, end=end_date)["Close"].dropna(how='any')
    if isinstance(price, pd.Series):
        price = price.to_frame(name=tickers[0])
    mu = expected_returns.mean_historical_return(price)
    S = risk_models.sample_cov(price)

    ret_min, ret_max = mu.min(), mu.max()
    target_rs = np.linspace(ret_min, ret_max, n_points)
    frontier_vols, frontier_rets, frontier_weights = [], [], []

    for R in target_rs:
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        try:
            ef.efficient_return(R)
            r, v, _ = ef.portfolio_performance(verbose=False)
            w = ef.clean_weights()
            frontier_rets.append(r * 100)
            frontier_vols.append(v * 100)
            frontier_weights.append(w)
        except:
            continue

    frontier_df = pd.DataFrame({"Volatility": frontier_vols, "Return": frontier_rets})

    ef_mv = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w_mv = ef_mv.min_volatility()
    ret_mv, vol_mv, sharpe_mv = ef_mv.portfolio_performance(verbose=False)

    ef_ms = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w_ms = ef_ms.max_sharpe()
    ret_ms, vol_ms, sharpe_ms = ef_ms.portfolio_performance(verbose=False)

    idx = mu.idxmax()
    w_mr = {t: (1.0 if t == idx else 0.0) for t in tickers}
    ret_mr = mu[idx]
    vol_mr = np.sqrt(S.loc[idx, idx])
    sharpe_mr = ret_mr / vol_mr

    special = {
        "min_vol":    {"weights": w_mv, "Volatility": vol_mv*100, "Return": ret_mv*100, "Sharpe": sharpe_mv},
        "max_sharpe": {"weights": w_ms, "Volatility": vol_ms*100, "Return": ret_ms*100, "Sharpe": sharpe_ms},
        "max_return": {"weights": w_mr, "Volatility": vol_mr*100, "Return": ret_mr*100, "Sharpe": sharpe_mr}
    }

    if weights is not None:
        if isinstance(weights, dict):
            w_user = np.array([round(weights.get(t, 0.0), 4) for t in tickers], dtype=float)
        else:
            w_user = np.round(np.array(weights, dtype=float), 4)

        if w_user.shape[0] != len(tickers):
            raise ValueError("Lunghezza di weights non corrisponde al numero di tickers.")
        if not np.isclose(w_user.sum(), 1.0, rtol=1e-4):
            raise ValueError("I pesi in weights devono sommare a 1 (tolleranza 1e-4).")

        weights_dict = dict(zip(tickers, w_user))

        # Ricalcolo coerente solo sugli asset usati
        assets_used = [t for t, w in weights_dict.items() if w > 0.0]
        mu_user = expected_returns.mean_historical_return(price[assets_used])
        S_user = risk_models.sample_cov(price[assets_used])
        w_trimmed = np.array([weights_dict[t] for t in assets_used])

        ret_user = w_trimmed @ mu_user.values
        vol_user = np.sqrt(w_trimmed @ S_user.values @ w_trimmed)
        sharpe_user = ret_user / vol_user

        annual_return_real = volatility_real = sharpe_real = None
        if compute_real_annual_return:
            price_bh = price[assets_used].dropna(how='any')
            size_df = pd.DataFrame(np.nan, index=price_bh.index, columns=price_bh.columns)
            size_df.loc[price_bh.index[0]] = w_trimmed

            pf_bh = vbt.Portfolio.from_orders(
                close=price_bh,
                size=size_df,
                size_type="targetpercent",
                init_cash=100_000,
                cash_sharing=True,
                freq="D"
            )
            annual_return_real = pf_bh.annualized_return() * 100
            volatility_real = pf_bh.annualized_volatility() * 100
            sharpe_real = pf_bh.sharpe_ratio()

        special["my_portfolio"] = {
            "weights": weights_dict,
            "Return": ret_user * 100,
            "Volatility": vol_user * 100,
            "Sharpe": sharpe_user,
            "Real Return": round(float(annual_return_real), 2) if annual_return_real is not None else None,
            "Real Volatility": round(float(volatility_real), 2) if volatility_real is not None else None,
            "Real Sharpe": round(float(sharpe_real), 2) if sharpe_real is not None else None
        }
        
    title_date = f" (from {start_date.date()} - to {end_date.date()})" if end_date else f" (from {start_date.date()})"
    title = "Efficient Frontier & Portafogli Standard" + title_date
    # print(titile)

    fig = go.Figure()
    for i, row in frontier_df.iterrows():
        weights = frontier_weights[i]
        weight_text = "<br>".join([f"{k}: {v:.2%}" for k, v in weights.items() if v > 0.01])
        fig.add_trace(go.Scatter(
            x=[row["Volatility"]],
            y=[row["Return"]],
            mode='markers',
            marker=dict(size=6, color='gray'),
            name="Efficient Frontier",
            showlegend=(i == 0),
            hovertemplate=f"<b>Vol:</b> {row['Volatility']:.2f}%<br>"
                          f"<b>Return:</b> {row['Return']:.2f}%<br>{weight_text}<extra></extra>"
        ))

    fig.add_trace(go.Scatter(x=[special["min_vol"]["Volatility"]],
                             y=[special["min_vol"]["Return"]],
                             mode='markers', marker=dict(color='green', size=12),
                             name="Min Volatility"))
    fig.add_trace(go.Scatter(x=[special["max_sharpe"]["Volatility"]],
                             y=[special["max_sharpe"]["Return"]],
                             mode='markers', marker=dict(color='orange', symbol='star', size=16),
                             name="Max Sharpe"))
    fig.add_trace(go.Scatter(x=[special["max_return"]["Volatility"]],
                             y=[special["max_return"]["Return"]],
                             mode='markers', marker=dict(color='red', symbol='square', size=12),
                             name="Max Return"))
    if "my_portfolio" in special:
        mp = special["my_portfolio"]
        fig.add_trace(go.Scatter(x=[mp["Volatility"]], y=[mp["Return"]],
                                 mode='markers+text',
                                 marker=dict(color='blue', symbol='diamond', size=14),
                                 text=["My Portfolio"], textposition="bottom right",
                                 name="My Portfolio"))
    fig.update_layout(width=fig_width, height=fig_height,
                      title=title,
                      xaxis_title="Volatility [%]", yaxis_title="Expected Return [%]",
                      legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02))
    
    
    if show_plot: fig.show()
    
    rows, idxs = [], []
    label_map = {
        "min_vol": "Min Vol",
        "max_sharpe": "Max Sharpe",
        "max_return": "Max Return",
        "my_portfolio": "My Portfolio"
    }
    for name in special:
        info = special[name]
        row = {k: round(v, 4) for k, v in info["weights"].items()}
        row.update({
            "Return": round(info["Return"], 2),
            "Volatility": round(info["Volatility"], 2),
            "Sharpe": round(info["Sharpe"], 2)
        })
        if "Real Return" in info:
            row["Real Return"] = info["Real Return"]
            row["Real Volatility"] = info["Real Volatility"]
            row["Real Sharpe"] = info["Real Sharpe"]
        rows.append(row)
        idxs.append(label_map.get(name, name))

    df_special = pd.DataFrame(rows, index=idxs)
    
    if print_weights: display(df_special)

    # return {"frontier_df": frontier_df, "special": special}
    return fig, df_special
    
def efficient_frontier_pypfopt_RECOVERY(
    tickers: list,
    # start_date: str,
    # end_date: str,
    years: int = 10,
    n_points: int = 50,
    weight_bounds: tuple = (0, 1),
    show_plot: bool = True,
    interactive: bool = True,
    print_weights: bool = True,
    my_weights=None,
    fig_width: int = 1200,
    fig_height: int = 600,
    compute_real_annual_return: bool = True
) -> dict:

    # Calcola date di inizio e fine
    end_date = datetime.today()
    start_date = datetime(end_date.year - years, 1, 1)

    price = load_ohlcv(tickers, start=start_date, end=end_date)["Close"].dropna(how='any')
    mu = expected_returns.mean_historical_return(price)
    S = risk_models.sample_cov(price)

    ret_min, ret_max = mu.min(), mu.max()
    target_rs = np.linspace(ret_min, ret_max, n_points)
    frontier_vols, frontier_rets, frontier_weights = [], [], []

    for R in target_rs:
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        try:
            ef.efficient_return(R)
            r, v, _ = ef.portfolio_performance(verbose=False)
            w = ef.clean_weights()
            frontier_rets.append(r * 100)
            frontier_vols.append(v * 100)
            frontier_weights.append(w)
        except:
            continue

    frontier_df = pd.DataFrame({"Volatility": frontier_vols, "Return": frontier_rets})

    ef_mv = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w_mv = ef_mv.min_volatility()
    ret_mv, vol_mv, sharpe_mv = ef_mv.portfolio_performance(verbose=False)

    ef_ms = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w_ms = ef_ms.max_sharpe()
    ret_ms, vol_ms, sharpe_ms = ef_ms.portfolio_performance(verbose=False)

    idx = mu.idxmax()
    w_mr = {t: (1.0 if t == idx else 0.0) for t in tickers}
    ret_mr = mu[idx]
    vol_mr = np.sqrt(S.loc[idx, idx])
    sharpe_mr = ret_mr / vol_mr

    special = {
        "min_vol":    {"weights": w_mv, "Volatility": vol_mv*100, "Return": ret_mv*100, "Sharpe": sharpe_mv},
        "max_sharpe": {"weights": w_ms, "Volatility": vol_ms*100, "Return": ret_ms*100, "Sharpe": sharpe_ms},
        "max_return": {"weights": w_mr, "Volatility": vol_mr*100, "Return": ret_mr*100, "Sharpe": sharpe_mr}
    }

    if my_weights is not None:
        if isinstance(my_weights, dict):
            w_user = np.array([round(my_weights.get(t, 0.0), 4) for t in tickers], dtype=float)
        else:
            w_user = np.round(np.array(my_weights, dtype=float), 4)

        if w_user.shape[0] != len(tickers):
            raise ValueError("Lunghezza di my_weights non corrisponde al numero di tickers.")
        if not np.isclose(w_user.sum(), 1.0, rtol=1e-4):
            raise ValueError("I pesi in my_weights devono sommare a 1 (tolleranza 1e-4).")

        weights_dict = dict(zip(tickers, w_user))

        # Ricalcolo coerente solo sugli asset usati
        assets_used = [t for t, w in weights_dict.items() if w > 0.0]
        mu_user = expected_returns.mean_historical_return(price[assets_used])
        S_user = risk_models.sample_cov(price[assets_used])
        w_trimmed = np.array([weights_dict[t] for t in assets_used])

        ret_user = w_trimmed @ mu_user.values
        vol_user = np.sqrt(w_trimmed @ S_user.values @ w_trimmed)
        sharpe_user = ret_user / vol_user

        annual_return_real = volatility_real = sharpe_real = None
        if compute_real_annual_return:
            price_bh = price[assets_used].dropna(how='any')
            size_df = pd.DataFrame(np.nan, index=price_bh.index, columns=price_bh.columns)
            size_df.loc[price_bh.index[0]] = w_trimmed

            pf_bh = vbt.Portfolio.from_orders(
                close=price_bh,
                size=size_df,
                size_type="targetpercent",
                init_cash=100_000,
                cash_sharing=True,
                freq="D"
            )
            annual_return_real = pf_bh.annualized_return() * 100
            volatility_real = pf_bh.annualized_volatility() * 100
            sharpe_real = pf_bh.sharpe_ratio()

        special["my_portfolio"] = {
            "weights": weights_dict,
            "Return": ret_user * 100,
            "Volatility": vol_user * 100,
            "Sharpe": sharpe_user,
            "Real Return": round(float(annual_return_real), 2) if annual_return_real is not None else None,
            "Real Volatility": round(float(volatility_real), 2) if volatility_real is not None else None,
            "Real Sharpe": round(float(sharpe_real), 2) if sharpe_real is not None else None
        }
        
    if show_plot:
        title_date = f" (from {start_date.date()} - to {end_date.date()})" if end_date else f" (from {start_date.date()})"
        title = "Efficient Frontier & Portafogli Standard" + title_date
        # print(titile)

        if interactive:
            fig = go.Figure()
            for i, row in frontier_df.iterrows():
                weights = frontier_weights[i]
                weight_text = "<br>".join([f"{k}: {v:.2%}" for k, v in weights.items() if v > 0.01])
                fig.add_trace(go.Scatter(
                    x=[row["Volatility"]],
                    y=[row["Return"]],
                    mode='markers',
                    marker=dict(size=6, color='gray'),
                    name="Efficient Frontier",
                    showlegend=(i == 0),
                    hovertemplate=f"<b>Vol:</b> {row['Volatility']:.2f}%<br>"
                                  f"<b>Return:</b> {row['Return']:.2f}%<br>{weight_text}<extra></extra>"
                ))

            fig.add_trace(go.Scatter(x=[special["min_vol"]["Volatility"]],
                                     y=[special["min_vol"]["Return"]],
                                     mode='markers', marker=dict(color='green', size=12),
                                     name="Min Volatility"))
            fig.add_trace(go.Scatter(x=[special["max_sharpe"]["Volatility"]],
                                     y=[special["max_sharpe"]["Return"]],
                                     mode='markers', marker=dict(color='orange', symbol='star', size=16),
                                     name="Max Sharpe"))
            fig.add_trace(go.Scatter(x=[special["max_return"]["Volatility"]],
                                     y=[special["max_return"]["Return"]],
                                     mode='markers', marker=dict(color='red', symbol='square', size=12),
                                     name="Max Return"))
            if "my_portfolio" in special:
                mp = special["my_portfolio"]
                fig.add_trace(go.Scatter(x=[mp["Volatility"]], y=[mp["Return"]],
                                         mode='markers+text',
                                         marker=dict(color='blue', symbol='diamond', size=14),
                                         text=["My Portfolio"], textposition="bottom right",
                                         name="My Portfolio"))
            fig.update_layout(width=fig_width, height=fig_height,
                              title=title,
                              xaxis_title="Volatility [%]", yaxis_title="Expected Return [%]",
                              legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02))
            # fig.show()
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(fig_width/100, fig_height/100))
            plt.plot(frontier_df["Volatility"], frontier_df["Return"], '--', label="Efficient Frontier")
            plt.scatter(special["min_vol"]["Volatility"], special["min_vol"]["Return"],
                        c='green', s=100, label="Min Volatility")
            plt.scatter(special["max_sharpe"]["Volatility"], special["max_sharpe"]["Return"],
                        c='orange', marker='*', s=150, label="Max Sharpe")
            plt.scatter(special["max_return"]["Volatility"], special["max_return"]["Return"],
                        c='red', marker='s', s=100, label="Max Return")
            if "my_portfolio" in special:
                mp = special["my_portfolio"]
                plt.scatter(mp["Volatility"], mp["Return"],
                            c='blue', marker='D', s=120, label="My Portfolio")
                plt.text(mp["Volatility"], mp["Return"], " My Portfolio",
                         va='bottom', ha='right')
            plt.xlabel("Volatility [%]"); plt.ylabel("Expected Return [%]")
            plt.title(title)
            plt.legend(); plt.grid(True); plt.show()


    if print_weights:
        rows, idxs = [], []
        label_map = {
            "min_vol": "Min Vol",
            "max_sharpe": "Max Sharpe",
            "max_return": "Max Return",
            "my_portfolio": "My Portfolio"
        }
        for name in special:
            info = special[name]
            row = {k: round(v, 4) for k, v in info["weights"].items()}
            row.update({
                "Return": round(info["Return"], 2),
                "Volatility": round(info["Volatility"], 2),
                "Sharpe": round(info["Sharpe"], 2)
            })
            if "Real Return" in info:
                row["Real Return"] = info["Real Return"]
                row["Real Volatility"] = info["Real Volatility"]
                row["Real Sharpe"] = info["Real Sharpe"]
            rows.append(row)
            idxs.append(label_map.get(name, name))

        df_special = pd.DataFrame(rows, index=idxs)
        display(df_special)

    # return {"frontier_df": frontier_df, "special": special}
    return fig, df_special

    
    

def run_portfolio_analysis(
    portfolio: dict,
    start_date=None,
    end_date=None,
    title: str = '',
    benchmark: str = 'SPY',
    init_cash: float = 100_000,
    fees: float = 0.001,
    rebalance_freq: str = None,
    efficient_frontier: bool = True,
    vbt_plot_width: int = 800,
    run_as_app: bool = False,
    min_years: int = 5,
    _preloaded: tuple = None,
    _preloaded_bm_data=None,
):
    """
    Backtest e report multipli grafici.
    - start_date / end_date: accettano datetime o stringhe tipo 'YYYY-MM-DD'. Default:
        end_date = oggi
        start_date = 1 Gennaio (end_year - 10)  [fallback storico 10 anni]
    Se run_as_app=True, non stampa né fig.show(), ma restituisce un dizionario di Figure.
    """
    one_ticker = len(portfolio) == 1

    # --- Normalizza date ---
    if end_date is None:
        end_date_dt = datetime.today()
    else:
        end_date_dt = pd.to_datetime(end_date).to_pydatetime()

    # --- start_date: None => tutto lo storico (non forziamo alcuna data) ---
    start_date_dt = None if start_date is None else pd.to_datetime(start_date).to_pydatetime()

    # Set Uppercase
    for k in list(portfolio.keys()):
        portfolio[k.upper()] = portfolio.pop(k)

    # 1) Backtest B&H
    pf = run_bh_backtest(
        portfolio, start_date_dt, end_date_dt,
        init_cash=init_cash, fees=fees, rebalance_freq=rebalance_freq,
        min_years=min_years, _preloaded=_preloaded,
    )
    if pf is None:
        print(f"[run_portfolio_analysis] {title!r}: storico insufficiente "
              f"(min_years={min_years}) — report non generato.")
        return (None, {}, pd.DataFrame()) if run_as_app else None

    # header = "Titolo" if one_ticker else "Portfolio"
    # if not run_as_app:
    #     print(f"🔎 Analisi {header} «{title}»")

    benchmark_data = _preloaded_bm_data if _preloaded_bm_data is not None else download_data(benchmark, start_date, end_date)

    show_report=False if run_as_app else True

    figs = generate_lazy_portfolio_performance(pf=pf,
                                               portfolio_title=title,
                                                benchmark=benchmark,
                                                benchmark_data=benchmark_data,
                                                show_report=show_report)
    
    # Efficient frontier (opzionale)
    special_weights = pd.DataFrame()

    if efficient_frontier and not one_ticker:
        my_tickers = list(portfolio.keys())
        my_weights = list(portfolio.values())

        fig_frontier, special_weights = efficient_frontier_pypfopt(
            tickers=my_tickers,
            weights=portfolio,
            n_points=80,
            weight_bounds=(0, 1),
            show_plot=show_report,
            print_weights=show_report,            
        )

        if run_as_app:
            figs['efficient_frontier'] = fig_frontier
        # else:
        #     display(special_weights)
        #     fig_frontier.show()

    return (pf, figs, special_weights) if run_as_app else pf

def run_portfolio_analysis_RECOVERY(
    weights_dict: dict,
    start_date=None,
    end_date=None,
    title: str = '',
    benchmark: str = 'SPY',
    init_cash: float = 100_000,
    fees: float = 0.001,
    rebalance_freq: str = None,
    efficient_frontier: bool = True,
    vbt_plot_width: int = 800,
    run_as_app: bool = False,
    min_years: int = 5,
):
    """
    Backtest e report multipli grafici.
    - start_date / end_date: accettano datetime o stringhe tipo 'YYYY-MM-DD'. Default:
        end_date = oggi
        start_date = 1 Gennaio (end_year - 10)  [fallback storico 10 anni]
    Se run_as_app=True, non stampa né fig.show(), ma restituisce un dizionario di Figure.
    """
    one_ticker = len(weights_dict) == 1

    # --- Normalizza date ---
    if end_date is None:
        end_date_dt = datetime.today()
    else:
        end_date_dt = pd.to_datetime(end_date).to_pydatetime()

    # --- start_date: None => tutto lo storico (non forziamo alcuna data) ---
    start_date_dt = None if start_date is None else pd.to_datetime(start_date).to_pydatetime()

    # Set Uppercase
    for k in list(weights_dict.keys()):
        weights_dict[k.upper()] = weights_dict.pop(k)

    # 1) Backtest B&H
    pf = run_bh_backtest(
        weights_dict, start_date_dt, end_date_dt,
        init_cash=init_cash, fees=fees, rebalance_freq=rebalance_freq,
        min_years=min_years,
    )
    if pf is None:
        print(f"[run_portfolio_analysis_RECOVERY] {title!r}: storico insufficiente "
              f"(min_years={min_years}) — report non generato.")
        return (None, {}) if run_as_app else None

    header = "Titolo" if one_ticker else "Portfolio"
    if not run_as_app:
        print(f"🔎 Analisi {header} «{title}»")

    figs = {}

    # 2) Cumulative returns
    fig = pf.plot_cum_returns(width=vbt_plot_width)
    if run_as_app:
        figs['cum_returns'] = fig
    else:
        fig.show()

    # 3) Statistiche e summary
    if not run_as_app:
        print(pf.stats())
        print_summary(pf, alpha_analysis=False)

    # 4) Monthly heatmap
    fig = plot_monthly_returns(
        pf,
        eoy=True,
        title=f"Monthly returns «{title}»",
        width=vbt_plot_width,
        height=vbt_plot_width
    )
    if run_as_app:
        figs['monthly_returns'] = fig
    else:
        fig.show()

    # 5) Drawdowns + Underwater
    fig = pf.plot_drawdowns(width=vbt_plot_width)
    if run_as_app:
        figs['drawdowns'] = fig
    else:
        fig.show()

    fig = pf.plot_underwater(width=vbt_plot_width)
    if run_as_app:
        figs['underwater'] = fig
    else:
        fig.show()

    # 6) Confronto vs Benchmark
    returns_strat = pf.returns().dropna()
    portfolios_returns = {f"Portfolio {title} (B&H)": returns_strat}
    fig = plot_multiple_portfolios(
        portfolios_returns,
        benchmark=benchmark,
        start_date=start_date_dt,
        end_date=end_date_dt
    )
    if run_as_app:
        figs['vs_benchmark'] = fig
    else:
        fig.show()

    # 7) Cumulative + rolling
    fig = plot_cumulative_and_rolling_returns(pf)
    if run_as_app:
        figs['cum_rolling'] = fig
    else:
        fig.show()

    # 8) Return triangle
    fig, *_, t_msg = annual_return_triangle(
        pf, resample_freq="YE", run_as_app=run_as_app
    )
    if run_as_app:
        figs['triangle'] = fig
    else:
        fig.show()

    # 9) Asset contributions (solo multi-ticker)
    if not one_ticker:
        fig = build_and_plot_portfolio_contributions(
            pf, title=title, benchmark=benchmark,
            start_date=start_date_dt, end_date=end_date_dt
        )
        if run_as_app:
            figs['contrib_full'] = fig
        else:
            fig.show()

        fig = build_and_plot_portfolio_contributions(
            pf, title=title, benchmark=benchmark,
            start_date=ytd(), end_date=end_date_dt
        )
        if run_as_app:
            figs['contrib_ytd'] = fig
        else:
            fig.show()

    # 10) Efficient frontier (opzionale)
    special_weights = pd.DataFrame()

    if efficient_frontier and not one_ticker:
        my_tickers = list(weights_dict.keys())
        my_weights = list(weights_dict.values())

        # years stimati dal range date (minimo 1 per evitare edge case)
        years = max(1, int((end_date_dt - start_date_dt).days / 365.25))

        fig, special_weights = efficient_frontier_pypfopt(
            tickers=my_tickers,
            weights=my_weights,
            years=years,
            n_points=50,
            weight_bounds=(0, 1)
        )

        if run_as_app:
            figs['efficient_frontier'] = fig
        else:
            display(special_weights)
            fig.show()

    return (pf, figs, special_weights, t_msg) if run_as_app else pf

#
# Versione di controllo della funzione precedente: in questa versione i rendimenti degli asset vengono 
# calcolati ex-novo con pandas. Nella verisone precedente si utilizzano i risultati vectorbt
#
def build_and_plot_portfolio_contributions_pandas(
    weights_dict: dict,
    start_date: str,
    end_date: str,
    init_cash: float = 10_000,
    fees: float = 0.0,
    portfolio_name: str = "My Portfolio",
    benchmark: str = None,
    plot_start_date: str = None
):
    """
    Costruisce e visualizza le curve dei contributi cumulativi al portafoglio
    per ciascun asset, più il portafoglio aggregato (Pandas e VectorBT).

    Args:
        weights_dict (dict): {ticker: peso} - i pesi devono sommare a 1.
        start_date (str): Data inizio.
        end_date (str): Data fine.
        init_cash (float): Capitale iniziale per VectorBT.
        fees (float): Fees proporzionali (es. 0.001 = 0.1%).
        portfolio_name (str): Nome del portafoglio nel grafico.
        benchmark (str): Ticker benchmark opzionale.
        plot_start_date (str): Data inizio per il grafico.

    Returns:
        fig: oggetto Plotly
    """


    tickers = list(weights_dict.keys())
    weights = pd.Series(weights_dict, dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("La somma dei pesi deve essere 1.")

    # 1. Scarica dati
    data = load_ohlcv(tickers, start=start_date, end=end_date)
    price = data["Close"][tickers].dropna(how='any')

    if price.empty:
        raise ValueError("Dati insufficienti: price è vuoto dopo dropna.")

    # 2. Ritorni giornalieri grezzi
    daily_returns = price.pct_change().dropna()

    # 3. Contributi giornalieri pesati
    daily_contributions = daily_returns.mul(weights, axis=1)
    cumulative_contributions = (1 + daily_contributions).cumprod()

    # 4. Curva del portafoglio (Pandas)
    price_norm = price / price.iloc[0]
    portfolio_curve_pandas = (price_norm * weights).sum(axis=1)
    returns_portfolio_pandas = portfolio_curve_pandas.pct_change().dropna()

    # 5. Portafoglio VectorBT
    size = pd.DataFrame(np.nan, index=price.index, columns=price.columns)
    size.loc[price.index[0]] = weights

    pf = vbt.Portfolio.from_orders(
        close=price,
        size=size,
        size_type='targetpercent',
        init_cash=init_cash,
        fees=fees,
        cash_sharing=True,
        freq='D'
    )
    returns_portfolio_vbt = pf.returns()

    # 6. Dizionario da plottare
    portfolios_returns = {}

    # Curve dei contributi cumulativi (non grezzi)
    for ticker in tickers:
        portfolios_returns[f"{ticker} (contributo cumulato)"] = cumulative_contributions[ticker].pct_change().dropna()

    portfolios_returns[f"{portfolio_name} (Pandas)"] = returns_portfolio_pandas
    portfolios_returns[f"{portfolio_name} (VectorBT)"] = returns_portfolio_vbt

    # 7. Grafico finale
    fig = plot_multiple_portfolios(
        portfolios_returns,
        title=f"Contributi al portafoglio: {portfolio_name}",
        benchmark=benchmark,
        start_date=plot_start_date or start_date,
        end_date=end_date
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# run_lazy_analysis
# ─────────────────────────────────────────────────────────────────────────────

def _cagr_from_equity(pf, ann: int = 252) -> float:
    eq = pf.value()
    if isinstance(eq, pd.DataFrame):
        eq = eq.iloc[:, 0]
    eq = eq.dropna()
    if len(eq) < 2:
        return np.nan
    years_n = len(eq) / ann
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / years_n) - 1 if years_n > 0 else np.nan


def _safe_metric(pf, metric: str) -> float:
    try:
        if metric == 'sharpe':
            return float(pf.sharpe_ratio())
        elif metric == 'total_return':
            return float(pf.total_return())
        elif metric == 'max_drawdown':
            return float(pf.max_drawdown())
    except Exception:
        return np.nan
    return np.nan


def lazy_stability_weights(
    tickers: list,
    years: int = 10,
    weight_bounds: tuple = (0, 1),
    n_splits: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Stability test sui pesi ottimali (Max Sharpe) su sotto-periodi.
    Divide il periodo storico (years) in n_splits finestre disgiunte
    e ricalcola i pesi ottimali per ogni finestra.
    Returns dict:
        'df_weights':   DataFrame (n_splits righe x n_tickers colonne)
                        pesi Max Sharpe per ogni finestra
        'df_stats':     DataFrame (asset x [mean, std, cv])
        'stable':       bool — True se cv medio < cv_threshold
        'cv_mean':      float
        'cv_threshold': float (default 0.5)

    Ritorna None (con stampa diagnostica) se meno di 2 ticker sono
    forniti, o se uno o più ticker hanno storico insufficiente a
    coprire le finestre richieste.
    """
    if len(tickers) < 2:
        print(
            "[lazy_stability_weights] Analisi non applicabile con un solo ticker "
            "— con un singolo asset non esiste una frontiera efficiente (nessuna "
            "combinazione di pesi possibile). Skip. Per singolo titolo/fondo usa "
            "generate_lazy_portfolio_performance."
        )
        return None

    end_year_global = datetime.today().year
    start_year_global = end_year_global - years
    window_years = max(1, years // n_splits)

    # Guard: verifica storico minimo comune PRIMA di iterare le finestre.
    # Un ticker con storico più corto del periodo testato (years) produce
    # NaN/errori a cascata in ogni finestra che lo include (invece di un
    # errore chiaro una sola volta) — meglio fermarsi subito con diagnostica.
    price_check = load_ohlcv(tickers, start=f"{start_year_global}-01-01",
                              end=datetime.today().strftime("%Y-%m-%d"))
    if isinstance(price_check, pd.Series):
        price_check = price_check.to_frame(name=tickers[0])
    close_check = price_check['Close']
    if isinstance(close_check, pd.Series):
        close_check = close_check.to_frame(name=tickers[0])

    coverage = close_check.notna().sum()
    min_required_days = int(252 * window_years * 0.8)  # almeno 80% di una finestra
    insufficient = coverage[coverage < min_required_days].to_dict()
    if insufficient:
        print(
            f"[lazy_stability_weights] Test di stabilità non applicabile su {years} anni "
            f"({n_splits} finestre da ~{window_years} anni ciascuna): ticker con storico "
            f"insufficiente per coprire le finestre richieste (minimo {min_required_days} "
            f"giorni/finestra): {insufficient}. Copertura completa: {coverage.to_dict()}. "
            f"Ridurre 'years' al periodo comune disponibile, o escludere questi ticker dal test."
        )
        return None

    _metric_cols = {'Return', 'Volatility', 'Sharpe',
                    'Real Return', 'Real Volatility', 'Real Sharpe'}
    if window_years < 1:
        print(f"[lazy_stability_weights] window_years forzato a 1 (era {years // n_splits})")
        window_years = 1
    _empty = {
        'df_weights': pd.DataFrame(),
        'df_stats': None,
        'stable': False,
        'cv_mean': np.nan,
        'cv_threshold': 0.5,
    }
    # Finestre storiche DISGIUNTE: divido [oggi-years, oggi] in n_splits blocchi
    rows = []
    labels = []
    for i in range(n_splits):
        window_start_year = start_year_global + i * window_years
        window_end_year   = window_start_year + window_years
        win_start = f"{window_start_year}-01-01"
        win_end   = f"{window_end_year}-01-01"
        label = f'W{i+1} ({window_start_year}-{window_end_year})'
        try:
            _, df_special = efficient_frontier_pypfopt(
                tickers=tickers,
                start_date=win_start,
                end_date=win_end,
                weight_bounds=weight_bounds,
                show_plot=False,
                interactive=False,
                print_weights=False,
            )
            _weight_cols = [c for c in df_special.columns if c not in _metric_cols]
            if 'Max Sharpe' in df_special.index:
                weights_i = df_special.loc['Max Sharpe', _weight_cols].to_dict()
            else:
                weights_i = df_special.iloc[0][_weight_cols].to_dict()
                print(f"[lazy_stability_weights] {label}: 'Max Sharpe' non trovato, uso prima riga")
            rows.append(weights_i)
            labels.append(label)
        except Exception as e:
            print(f"[lazy_stability_weights] {label}: errore — {e}")
    if not rows:
        print("[lazy_stability_weights] Tutte le finestre hanno fallito — restituisco empty.")
        return _empty
    df_weights = pd.DataFrame(rows, index=labels)
    stats = []
    for asset in tickers:
        if asset not in df_weights.columns:
            stats.append({'asset': asset, 'mean': np.nan, 'std': np.nan, 'cv': np.nan})
            continue
        mean = df_weights[asset].mean()
        std  = df_weights[asset].std()
        cv   = std / mean if mean > 0.01 else np.nan
        stats.append({'asset': asset, 'mean': mean, 'std': std, 'cv': cv})
    df_stats = pd.DataFrame(stats).set_index('asset')
    cv_mean = float(df_stats['cv'].dropna().mean())
    cv_threshold = 0.5
    stable = bool(cv_mean < cv_threshold)
    if verbose:
        print(f"Stability test pesi — window={window_years}y, splits={n_splits}")
        try:
            from IPython.display import display as _disp
            _disp(df_weights)
            _disp(df_stats)
        except Exception:
            print(df_weights.to_string())
            print(df_stats.to_string())
        print(f"CV medio: {cv_mean:.3f} — {'STABILE ✅' if stable else 'INSTABILE ⚠️'}")
    return {
        'df_weights':   df_weights,
        'df_stats':     df_stats,
        'stable':       stable,
        'cv_mean':      cv_mean,
        'cv_threshold': cv_threshold,
    }
    
def lazy_mc_block_b_rebalancing(
    portfolio: dict,
    start_date,
    end_date,
    best_freq,
    n_simulations: int = 1000,
    jitter_days: int = 30,
    init_cash: float = 100_000,
    fees: float = 0.001,
    random_seed: int = 42,
    verbose: bool = False,
    min_years: int = 5,
    _preloaded_price=None,
) -> dict:
    """
    MC Block B — Skill test sul ribilanciamento.
    ...
    portfolio: dict nel nuovo formato {"Title": str, "tickers": {ticker: peso}, "benchmark": str}
               oppure nel vecchio formato flat {ticker: peso} (retrocompatibile).
    min_years: storico minimo comune richiesto (in anni), passato a
               run_bh_backtest. Default 5 — coerente col default di
               run_bh_backtest; abbassare (es. 1) per allineare Block B
               al resto della pipeline quando lo storico è limitato.
    """
    # Estrazione pesi: supporta sia il nuovo formato annidato sia il vecchio formato flat
    weights_dict = dict(portfolio["tickers"]) if "tickers" in portfolio else dict(portfolio)
    weights_dict = {k.upper(): v for k, v in weights_dict.items()}
    tickers = list(weights_dict.keys())
    weights = pd.Series(weights_dict, dtype=float)

    # Costruisce _preloaded_bh per run_bh_backtest se i prezzi sono già disponibili
    _preloaded_bh = (weights, _preloaded_price) if _preloaded_price is not None else None

    # 1. Metriche PTF reale
    # NOTA: passa il portfolio ORIGINALE (non weights_dict) — run_bh_backtest
    # fa la propria estrazione "tickers"/flat internamente, stessa logica qui.
    pf_actual = run_bh_backtest(portfolio, start_date, end_date,
                                init_cash, fees, best_freq, min_years=min_years,
                                _preloaded=_preloaded_bh)
    if pf_actual is None:
        print(f"[lazy_mc_block_b_rebalancing] Storico insufficiente — skip (vedi diagnostica sopra).")
        return None

    actual_sharpe = float(pf_actual.sharpe_ratio())
    actual_cagr   = _cagr_from_equity(pf_actual)

    # 2. Prezzi per il loop simulazioni — usa _preloaded_price se disponibile
    if _preloaded_price is not None:
        price = _preloaded_price
        if isinstance(price, pd.Series):
            price = price.to_frame(name=tickers[0])
        price.columns = [c.upper() for c in price.columns]
        # price è già dropna-aligned da _prepare_bh_data
    else:
        price = load_ohlcv(tickers, start=start_date, end=end_date,
                           multi_level_index=False)
        if 'Close' in price.columns.get_level_values(0) if isinstance(price.columns, pd.MultiIndex) else []:
            price = price['Close']
        elif 'Close' in price.columns:
            price = price[['Close'] if len(tickers) == 1 else tickers]
        if isinstance(price, pd.Series):
            price = price.to_frame(name=tickers[0])
        price.columns = [c.upper() for c in price.columns]
        price = price.dropna(how='any')

    if price.empty:
        raise ValueError("[lazy_mc_block_b_rebalancing] Nessun dato scaricato.")
    # 3. Date di ribilanciamento reali
    freq_for_sim = best_freq if best_freq is not None else 'Y'
    if best_freq is None:
        reb_dates_real = pd.DatetimeIndex([price.index[0]])
    else:
        rf = str(best_freq).upper()
        if rf in ['Y', 'A', 'YE']:
            reb_dates_real = price.groupby(price.index.year).apply(lambda x: x.index[-1])
            reb_dates_real = pd.DatetimeIndex(reb_dates_real.values)
        else:
            periods = price.index.to_period(best_freq)
            reb_dates_real = price.index[~periods.duplicated()]
        if price.index[0] not in reb_dates_real:
            reb_dates_real = reb_dates_real.insert(0, price.index[0])
        reb_dates_real = reb_dates_real.intersection(price.index)

    # 4. Loop simulazioni
    rng = np.random.default_rng(random_seed)
    sim_sharpes: list = []
    sim_cagrs:   list = []

    for _ in range(n_simulations):
        jitter = rng.integers(-jitter_days, jitter_days + 1,
                              size=len(reb_dates_real))
        reb_dates_sim = pd.DatetimeIndex([
            d + pd.Timedelta(days=int(j))
            for d, j in zip(reb_dates_real, jitter)
        ])
        reb_dates_sim = reb_dates_sim.intersection(price.index)
        if len(reb_dates_sim) == 0:
            continue

        size = pd.DataFrame(np.nan, index=price.index, columns=price.columns)
        for d in reb_dates_sim:
            size.loc[d] = weights

        try:
            pf_sim = vbt.Portfolio.from_orders(
                close=price,
                size=size,
                size_type='targetpercent',
                init_cash=init_cash,
                fees=fees,
                cash_sharing=True,
                freq='D',
            )
            sim_sharpes.append(float(pf_sim.sharpe_ratio()))
            sim_cagrs.append(_cagr_from_equity(pf_sim))
        except Exception:
            continue

    # 5. P-values
    sim_sharpes_arr = np.array(sim_sharpes)
    sim_cagrs_arr   = np.array(sim_cagrs)
    p_value_sharpe  = float((sim_sharpes_arr >= actual_sharpe).mean()) if len(sim_sharpes_arr) else np.nan
    p_value_cagr    = float((sim_cagrs_arr   >= actual_cagr).mean())   if len(sim_cagrs_arr)   else np.nan
    skill           = bool(p_value_sharpe < 0.05) if not np.isnan(p_value_sharpe) else False

    # 6. Verbose
    if verbose:
        print("MC Block B — Rebalancing Skill Test")
        print(f"  PTF reale  : Sharpe={actual_sharpe:.3f}  CAGR={actual_cagr:.2%}")
        if len(sim_sharpes_arr):
            print(f"  Sim median : Sharpe={np.median(sim_sharpes_arr):.3f}"
                  f"  CAGR={np.median(sim_cagrs_arr):.2%}")
        print(f"  p-value Sharpe={p_value_sharpe:.3f}"
              f"  p-value CAGR={p_value_cagr:.3f}")
        print(f"  Skill: {'✅ SI (p<0.05)' if skill else '⚠️ NO (p>=0.05)'}")

    return {
        'actual_sharpe':  actual_sharpe,
        'actual_cagr':    actual_cagr,
        'sim_sharpes':    sim_sharpes_arr,
        'sim_cagrs':      sim_cagrs_arr,
        'p_value_sharpe': p_value_sharpe,
        'p_value_cagr':   p_value_cagr,
        'skill':          skill,
        'n_simulations':  len(sim_sharpes_arr),
    }
    
def run_mc_diagnostics(
    pf,
    portfolio: dict,
    start_date: str,
    end_date: str,
    best_freq,
    n_simulations_a: int = 1000,
    block_size: int = 20,
    n_simulations_b: int = 500,
    jitter_days: int = 30,
    init_cash: float = 100_000,
    fees: float = 0.001,
    seed: int = 42,
    metrics: list = None,
    min_years: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Esegue la batteria diagnostica Monte Carlo completa su un portfolio
    già costruito:
      - Block A1: IID Bootstrap (baseline, sottostima il rischio reale
        perché ignora autocorrelazione/cluster di volatilità)
      - Block A2: Block Bootstrap (metodo principale per confidence
        interval, preserva struttura temporale)
      - Block B: Skill test sul ribilanciamento (best_freq confrontato
        con date di ribilanciamento randomizzate — verifica se la
        frequenza scelta aggiunge valore reale o è indistinguibile dal
        caso)

    Produce un report combinato unico: tabella A1 vs A2 (confidence
    interval per metrica) più tabella e verdetto di Block B.

    Parameters
    ----------
    pf : vbt.Portfolio
        Portfolio già costruito (es. da run_bh_backtest con best_freq),
        usato per Block A1/A2.
    portfolio : dict
        Configurazione portfolio (formato annidato o flat) — usato SOLO
        da Block B, che ricostruisce il portfolio internamente con date
        di ribilanciamento perturbate. Deve rappresentare lo stesso
        portfolio/periodo di `pf` per un confronto coerente — questa
        funzione non lo verifica automaticamente.
    start_date, end_date : str
        Periodo per Block B.
    best_freq : str o None
        Frequenza di ribilanciamento scelta (es. da
        compare_rebalance_frequencies), testata da Block B.
    n_simulations_a : int
        Simulazioni per A1 e A2.
    block_size : int
        Dimensione blocco per A2 (giorni di trading, default 20 ≈ 1 mese).
    n_simulations_b : int
        Simulazioni per Block B.
    jitter_days : int
        Ampiezza perturbazione (± giorni) sulle date di ribilanciamento
        in Block B.
    init_cash, fees : float
        Parametri di ricostruzione portfolio per Block B — devono
        coincidere con quelli usati per costruire `pf`, altrimenti il
        confronto Actual vs simulato non è comparabile.
    seed : int
        Seed base — A1, A2 e Block B usano generatori indipendenti
        derivati dallo stesso seed, per riproducibilità senza
        correlazione accidentale tra i tre test.
    metrics : list, opzionale
        Metriche nella tabella A1/A2. Default ['CAGR', 'Sharpe', 'MaxDD'].
    min_years: storico minimo comune richiesto (in anni), passato a
               run_bh_backtest. Default 5 — coerente col default di
               run_bh_backtest; abbassare (es. 1) per allineare Block B
               al resto della pipeline quando lo storico è limitato.   
    verbose : bool
        Se True, stampa tabelle e verdetto.

    Returns
    -------
    dict con chiavi:
        'a1', 'a2' : risultati completi mc_run_iid_bootstrap/mc_run_block_bootstrap
        'b'        : risultato completo lazy_mc_block_b_rebalancing,
                     None se il portfolio (dentro Block B) ha storico
                     insufficiente
        'summary_df'   : pd.DataFrame confidence interval A1 vs A2
        'skill_verdict': str — sintesi testuale del risultato Block B
    """
    if metrics is None:
        metrics = ['CAGR', 'Sharpe', 'MaxDD']

    rng_a1 = np.random.default_rng(seed)
    rng_a2 = np.random.default_rng(seed)

    if verbose:
        print("=== Monte Carlo Block A — Confidence Intervals ===")
    a1 = mc_run_iid_bootstrap(pf, n_simulations=n_simulations_a, rng=rng_a1)
    a2 = mc_run_block_bootstrap(pf, block_size=block_size,
                                 n_simulations=n_simulations_a, rng=rng_a2)

    rows = []
    for metric in metrics:
        rows.append({
            'Metric': metric,
            'Actual': a2['actual_metrics'][metric],
            'A1_P5' : a1['percentiles']['p5'][metric],
            'A1_P50': a1['percentiles']['p50'][metric],
            'A1_P95': a1['percentiles']['p95'][metric],
            'A2_P5' : a2['percentiles']['p5'][metric],
            'A2_P50': a2['percentiles']['p50'][metric],
            'A2_P95': a2['percentiles']['p95'][metric],
        })
    summary_df = pd.DataFrame(rows).set_index('Metric').round(3)

    if verbose:
        my_display(summary_df, title=f"Block A — Confidence Intervals "
                                      f"(n_sim={n_simulations_a}, block_size={block_size})")
        print("\n=== Monte Carlo Block B — Skill del ribilanciamento ===")

    b = lazy_mc_block_b_rebalancing(
        portfolio=portfolio,
        start_date=start_date,
        end_date=end_date,
        best_freq=best_freq,
        n_simulations=n_simulations_b,
        jitter_days=jitter_days,
        init_cash=init_cash,
        fees=fees,
        random_seed=seed,
        min_years=min_years,
        verbose=False,
    )

    if b is None:
        skill_verdict = "Block B non eseguito (storico insufficiente — vedi diagnostica sopra)."
        if verbose:
            print(f"⚠️  {skill_verdict}")
    else:
        b_rows = [{
            'Metric' : 'Sharpe',
            'Actual' : round(b['actual_sharpe'], 3),
            'Sim_P50': round(float(np.median(b['sim_sharpes'])), 3) if len(b['sim_sharpes']) else np.nan,
            'p_value': round(b['p_value_sharpe'], 3),
        }, {
            'Metric' : 'CAGR',
            'Actual' : round(b['actual_cagr'], 3),
            'Sim_P50': round(float(np.median(b['sim_cagrs'])), 3) if len(b['sim_cagrs']) else np.nan,
            'p_value': round(b['p_value_cagr'], 3),
        }]
        b_df = pd.DataFrame(b_rows).set_index('Metric')
        if verbose:
            my_display(b_df, title=f"Block B — Skill ribilanciamento "
                                    f"(freq={best_freq or 'BH'}, n_sim={n_simulations_b}, jitter=±{jitter_days}g)")
        skill_verdict = (
            f"{'✅ SKILL rilevata' if b['skill'] else '⚠️ NESSUNA skill'} "
            f"(p-value Sharpe={b['p_value_sharpe']:.3f}, soglia=0.05)"
        )
        if verbose:
            print(f"\n{skill_verdict}")

    return {
        'a1': a1,
        'a2': a2,
        'b': b,
        'summary_df': summary_df,
        'skill_verdict': skill_verdict,
    }

# def lazy_mc_block_b_rebalancing(
#     portfolio: dict,
#     start_date,
#     end_date,
#     best_freq,
#     n_simulations: int = 1000,
#     jitter_days: int = 30,
#     init_cash: float = 100_000,
#     fees: float = 0.001,
#     random_seed: int = 42,
#     verbose: bool = False,
# ) -> dict:
#     """
#     MC Block B — Skill test sul ribilanciamento.
#     Testa se la scelta della frequenza di ribilanciamento aggiunge
#     valore vs date di ribilanciamento randomizzate (jitter ±jitter_days).

#     Se best_freq is None (BH puro): confronta vs ribilanciamento annuale
#     randomizzato (verifica che BH non sia inferiore a qualsiasi rebalancing).

#     Returns dict:
#         'actual_sharpe':  float — Sharpe del PTF con best_freq
#         'actual_cagr':    float — CAGR del PTF con best_freq
#         'sim_sharpes':    np.ndarray — distribuzione Sharpe simulazioni
#         'sim_cagrs':      np.ndarray — distribuzione CAGR simulazioni
#         'p_value_sharpe': float — prob(sim_sharpe >= actual_sharpe)
#         'p_value_cagr':   float — prob(sim_cagr >= actual_cagr)
#         'skill':          bool — True se p_value_sharpe < 0.05
#         'n_simulations':  int
#     """
#     # NOTA: download_data non definita in mc_functions.py — fallback su yf.download
#     portfolio = {k.upper(): v for k, v in portfolio.items()}
#     tickers = list(portfolio.keys())
#     weights = pd.Series(portfolio, dtype=float)

#     # 1. Metriche PTF reale
#     pf_actual = run_bh_backtest(portfolio, start_date, end_date,
#                                 init_cash, fees, best_freq)
#     actual_sharpe = float(pf_actual.sharpe_ratio())
#     actual_cagr   = _cagr_from_equity(pf_actual)

#     # 2. Scarica prezzi una sola volta
#     price = load_ohlcv(tickers, start=start_date, end=end_date,
#                        multi_level_index=False)
#     if 'Close' in price.columns.get_level_values(0) if isinstance(price.columns, pd.MultiIndex) else []:
#         price = price['Close']
#     elif 'Close' in price.columns:
#         price = price[['Close'] if len(tickers) == 1 else tickers]
#     # Normalizza: se single-ticker yf restituisce Series, converti a DataFrame
#     if isinstance(price, pd.Series):
#         price = price.to_frame(name=tickers[0])
#     price.columns = [c.upper() for c in price.columns]
#     price = price.dropna(how='any')

#     if price.empty:
#         raise ValueError("[lazy_mc_block_b_rebalancing] Nessun dato scaricato.")

#     # 3. Date di ribilanciamento reali
#     freq_for_sim = best_freq if best_freq is not None else 'Y'
#     if best_freq is None:
#         reb_dates_real = pd.DatetimeIndex([price.index[0]])
#     else:
#         rf = str(best_freq).upper()
#         if rf in ['Y', 'A', 'YE']:
#             reb_dates_real = price.groupby(price.index.year).apply(lambda x: x.index[-1])
#             reb_dates_real = pd.DatetimeIndex(reb_dates_real.values)
#         else:
#             periods = price.index.to_period(best_freq)
#             reb_dates_real = price.index[~periods.duplicated()]
#         if price.index[0] not in reb_dates_real:
#             reb_dates_real = reb_dates_real.insert(0, price.index[0])
#         reb_dates_real = reb_dates_real.intersection(price.index)

#     # 4. Loop simulazioni
#     rng = np.random.default_rng(random_seed)
#     sim_sharpes: list = []
#     sim_cagrs:   list = []

#     for _ in range(n_simulations):
#         jitter = rng.integers(-jitter_days, jitter_days + 1,
#                               size=len(reb_dates_real))
#         reb_dates_sim = pd.DatetimeIndex([
#             d + pd.Timedelta(days=int(j))
#             for d, j in zip(reb_dates_real, jitter)
#         ])
#         reb_dates_sim = reb_dates_sim.intersection(price.index)
#         if len(reb_dates_sim) == 0:
#             continue

#         size = pd.DataFrame(np.nan, index=price.index, columns=price.columns)
#         for d in reb_dates_sim:
#             size.loc[d] = weights

#         try:
#             pf_sim = vbt.Portfolio.from_orders(
#                 close=price,
#                 size=size,
#                 size_type='targetpercent',
#                 init_cash=init_cash,
#                 fees=fees,
#                 cash_sharing=True,
#                 freq='D',
#             )
#             sim_sharpes.append(float(pf_sim.sharpe_ratio()))
#             sim_cagrs.append(_cagr_from_equity(pf_sim))
#         except Exception:
#             continue

#     # 5. P-values
#     sim_sharpes_arr = np.array(sim_sharpes)
#     sim_cagrs_arr   = np.array(sim_cagrs)
#     p_value_sharpe  = float((sim_sharpes_arr >= actual_sharpe).mean()) if len(sim_sharpes_arr) else np.nan
#     p_value_cagr    = float((sim_cagrs_arr   >= actual_cagr).mean())   if len(sim_cagrs_arr)   else np.nan
#     skill           = bool(p_value_sharpe < 0.05) if not np.isnan(p_value_sharpe) else False

#     # 6. Verbose
#     if verbose:
#         print("MC Block B — Rebalancing Skill Test")
#         print(f"  PTF reale  : Sharpe={actual_sharpe:.3f}  CAGR={actual_cagr:.2%}")
#         if len(sim_sharpes_arr):
#             print(f"  Sim median : Sharpe={np.median(sim_sharpes_arr):.3f}"
#                   f"  CAGR={np.median(sim_cagrs_arr):.2%}")
#         print(f"  p-value Sharpe={p_value_sharpe:.3f}"
#               f"  p-value CAGR={p_value_cagr:.3f}")
#         print(f"  Skill: {'✅ SI (p<0.05)' if skill else '⚠️ NO (p>=0.05)'}")

#     return {
#         'actual_sharpe':  actual_sharpe,
#         'actual_cagr':    actual_cagr,
#         'sim_sharpes':    sim_sharpes_arr,
#         'sim_cagrs':      sim_cagrs_arr,
#         'p_value_sharpe': p_value_sharpe,
#         'p_value_cagr':   p_value_cagr,
#         'skill':          skill,
#         'n_simulations':  len(sim_sharpes_arr),
#     }


def run_lazy_analysis(
    portfolio_cfg: dict,
    output_dir,
    title: str = '',
    start_date=None,
    end_date=None,
    benchmark: str = 'SPY',
    init_cash: float = 100_000,
    fees: float = 0.001,
    years: int = 10,
    plot: bool = False,
    save_png: bool = True,
    auto_freq: bool = True,
    freq_selection_metric: str = 'sharpe',
    weight_bounds: tuple = (0, 1),
    verbose: bool = False,
    min_years: int = 5,
    _preloaded_price=None,
    _preloaded_bm_data=None,
) -> dict:
    """
    Pipeline completa Lazy portfolio in modalità headless.
    Usata da iq lazy e dalla webapp.

    Returns dict con chiavi:
        'best_freq':       str | None — frequenza ottimale selezionata
        'freq_df':         DataFrame comparativo frequenze (None se auto_freq=False)
        'portfolio_pf':    vbt.Portfolio con best_freq
        'optimal_weights': dict ticker→peso (max Sharpe su frontiera)
        'frontier_result': dict con 'fig' e 'df_special' da efficient_frontier_pypfopt
        'plots_dir':       Path directory PNG
    """
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Normalizza years e start_date per coerenza tra backtest e frontiera
    from datetime import datetime as _dt
    _years = int(years) if (years is not None and years > 0) else 10
    if start_date is None:
        _start_date = _dt(_dt.today().year - _years, 1, 1).strftime('%Y-%m-%d')
    else:
        _start_date = start_date

    # Normalizza chiavi uppercase e ricava tickers/pesi
    portfolio_cfg = {k.upper(): v for k, v in portfolio_cfg.items()}
    tickers    = list(portfolio_cfg.keys())
    my_weights = list(portfolio_cfg.values())

    # Se i prezzi PTF sono stati pre-scaricati da _compute_lazy_full, costruiamo
    # il tuple (_preloaded) da passare a run_bh_backtest evitando ulteriori download.
    # NOTA: efficient_frontier_pypfopt usa un range basato su `years` (da oggi) che
    # può differire da _start_date quando start_date è esplicitamente specificato —
    # per questo la frontiera NON usa _preloaded_price (range potenzialmente diverso).
    if _preloaded_price is not None:
        _weights_for_preload = pd.Series(portfolio_cfg, dtype=float)
        _preloaded_bh = (_weights_for_preload, _preloaded_price)
    else:
        _preloaded_bh = None

    # ── 1. FREQUENCY SELECTION ───────────────────────────────────────────────
    if auto_freq:
        freqs = ['W', 'M', 'Q', 'Y', None]
        rows = []
        for freq in freqs:
            pf_f = run_bh_backtest(portfolio_cfg, _start_date, end_date,
                                   init_cash, fees, freq, min_years=min_years,
                                   _preloaded=_preloaded_bh)
            if pf_f is None:
                break
            rows.append({
                'Freq':        freq if freq is not None else 'BH',
                'Sharpe':      _safe_metric(pf_f, 'sharpe'),
                'CAGR':        _cagr_from_equity(pf_f),
                'TotalReturn': _safe_metric(pf_f, 'total_return'),
                'MaxDD':       abs(_safe_metric(pf_f, 'max_drawdown')),
            })
        if not rows:
            raise ValueError(
                f"[run_lazy_analysis] {title!r}: storico insufficiente per tutte le "
                f"frequenze (min_years={min_years}) — impossibile selezionare best_freq."
            )
        freq_df = pd.DataFrame(rows)

        metric_map = {
            'sharpe':       ('Sharpe',      'max'),
            'cagr':         ('CAGR',        'max'),
            'total_return': ('TotalReturn', 'max'),
            'max_dd':       ('MaxDD',       'min'),
        }
        col, goal = metric_map.get(freq_selection_metric, ('Sharpe', 'max'))
        if goal == 'max':
            best_label = freq_df.loc[freq_df[col].idxmax(), 'Freq']
        else:
            best_label = freq_df.loc[freq_df[col].idxmin(), 'Freq']
        best_freq = None if best_label == 'BH' else best_label

        if verbose:
            print(f"Frequenza ottimale ({freq_selection_metric}): {best_label}")
            try:
                from IPython.display import display as _disp
                _disp(freq_df)
            except Exception:
                print(freq_df.to_string())

        if save_png:
            try:
                metrics = ['Sharpe', 'CAGR', 'TotalReturn', 'MaxDD']
                fig_f, axes = plt.subplots(2, 2, figsize=(10, 7))
                fig_f.suptitle('Confronto Frequenze Ribilanciamento', fontsize=13)
                for ax, m in zip(axes.flat, metrics):
                    ax.bar(freq_df['Freq'].astype(str), freq_df[m])
                    ax.set_title(m)
                    ax.set_xlabel('Frequenza')
                plt.tight_layout()
                plt.savefig(str(plots_dir / 'freq_comparison.png'), dpi=120)
                plt.close(fig_f)
            except Exception as _e:
                if verbose:
                    print(f"[run_lazy_analysis] freq_comparison.png non salvato: {_e}")
    else:
        best_freq = None
        freq_df   = None

    # ── 2. BACKTEST CON FREQUENZA OTTIMALE ───────────────────────────────────
    portfolio_pf = run_bh_backtest(portfolio_cfg, _start_date, end_date,
                                   init_cash, fees, best_freq,
                                   min_years=min_years, _preloaded=_preloaded_bh)

    # ── 3. FRONTIERA EFFICIENTE ───────────────────────────────────────────────
    frontier_fig, df_special = efficient_frontier_pypfopt(
        tickers=tickers,
        years=_years,
        show_plot=plot,
        interactive=plot,
        print_weights=verbose,
        weights=my_weights,
        weight_bounds=weight_bounds,
    )

    # Estrai pesi max-Sharpe da df_special (DataFrame con index "Max Sharpe", ...)
    _metric_cols = {'Return', 'Volatility', 'Sharpe',
                    'Real Return', 'Real Volatility', 'Real Sharpe'}
    if 'Max Sharpe' in df_special.index:
        _row = df_special.loc['Max Sharpe']
        optimal_weights = {k: float(v) for k, v in _row.items()
                           if k not in _metric_cols}
    else:
        optimal_weights = {}

    frontier_result = {'fig': frontier_fig, 'df_special': df_special}

    if save_png:
        try:
            frontier_fig.write_image(str(plots_dir / 'frontier.png'))
        except Exception as _e:
            if verbose:
                print(f"[run_lazy_analysis] frontier.png non salvato: {_e}")

    # ── 4. REPORT ─────────────────────────────────────────────────────────────
    run_portfolio_analysis(
        portfolio=portfolio_cfg,
        start_date=_start_date,
        end_date=end_date,
        title=title,
        benchmark=benchmark,
        init_cash=init_cash,
        fees=fees,
        rebalance_freq=best_freq,
        efficient_frontier=False,
        run_as_app=not plot,
        min_years=min_years,
        _preloaded=_preloaded_bh,
        _preloaded_bm_data=_preloaded_bm_data,
    )

    return {
        'best_freq':       best_freq,
        'freq_df':         freq_df,
        'portfolio_pf':    portfolio_pf,
        'optimal_weights': optimal_weights,
        'frontier_result': frontier_result,
        'plots_dir':       plots_dir,
    }


def _compute_lazy_full(
    portfolio_cfg: dict,
    ptf_name: str,
    output_dir,
    start_date='2016-01-01',
    end_date=None,
    benchmark='SPY',
    init_cash=100_000.0,
    fees=0.001,
    years=10,
    n_simulations_mc_a=1000,
    n_simulations_mc_b=500,
    verbose=False,
    need_out: bool = False,
    min_years: int = 5,
) -> dict:
    """
    Pipeline Lazy completa per un SINGOLO PTF: analisi headless + backtest
    B&H + stability rolling + MC A1/A2 + MC B + DSR + metriche + verdetto.

    Estratta dal loop di run_lazy_batch_analysis() per essere riusabile sia
    nella classificazione batch (riga CSV) sia nella relazione tecnica PDF
    (oggetti completi). Ritorna un dict con TUTTI gli oggetti intermedi:

        'risultati'    : output di run_lazy_analysis (best_freq, freq_df,
                         portfolio_pf, optimal_weights, frontier_result, plots_dir)
        'pf_proposed'  : vbt.Portfolio del PTF reale (pesi fissi, best_freq)
        'stability'    : output di lazy_rolling_stability
        'mc_a1'        : MC iid bootstrap
        'mc_a2'        : MC block bootstrap (usato anche da project_lazy_capital)
        'mc_b'         : MC Block B (skill ribilanciamento)
        'dsr'          : Deflated Sharpe Ratio
        'sr', 'T'      : Sharpe e n. osservazioni
        'cagr','maxdd' : metriche aggregate
        'checks'       : dict dei 3 criteri
        'verdetto'     : 'PROMOSSO' | 'RIGETTATO'
        'row'          : riga di classificazione (per il CSV batch)
        'portfolio_cfg': dict originale dal registry (ticker:weight)
        'out'          : output di generate_lazy_portfolio_performance (solo se
                         need_out=True) — necessario per generate_relazione_investitore_report
    """
    from pathlib import Path
    # mc_run_iid_bootstrap / mc_run_block_bootstrap / ofc_compute_dsr sono
    # definite in r_functions.py e NON importate a livello di modulo qui:
    # import locale per evitare di toccare le import globali di mc_functions.py.
    from r_functions import (
        mc_run_iid_bootstrap,
        mc_run_block_bootstrap,
        ofc_compute_dsr,
    )

    # Supporta sia il vecchio formato flat {ticker: peso} sia il nuovo annidato
    # {"Title": str, "tickers": {ticker: peso}, "benchmark": str}.
    # Le funzioni di pipeline si aspettano sempre il flat {ticker: peso}.
    _nested_tickers = portfolio_cfg.get('tickers') if isinstance(portfolio_cfg.get('tickers'), dict) else None
    _pf_weights = _nested_tickers if _nested_tickers is not None else portfolio_cfg

    # ── PRE-FETCH UNICO: scarica prezzi PTF e benchmark UNA SOLA VOLTA ──────
    # Tutti i download successivi nella pipeline vengono eliminati passando
    # _preloaded_price / _preloaded_bh alle funzioni che lo accettano.
    _ptf_loaded_weights, _price_ptf = _prepare_bh_data(_pf_weights, start_date, end_date, min_years)
    if _price_ptf is None:
        raise ValueError(
            f"[_compute_lazy_full] {ptf_name}: storico PTF insufficiente "
            f"(min_years={min_years}) — skip (vedi diagnostica sopra)."
        )

    _bm_loaded_weights, _price_bm = _prepare_bh_data({benchmark: 1.0}, start_date, end_date, min_years)
    _bm_preloaded_arg = (_bm_loaded_weights, _price_bm) if _price_bm is not None else None
    if _price_bm is None and verbose:
        print(f"[_compute_lazy_full] {ptf_name}: pf_benchmark non disponibile "
              f"(benchmark={benchmark}, min_years={min_years}) — grafici vs benchmark disabilitati.")

    # 2. analisi lazy headless + backtest B&H con best_freq
    sub_output_dir = Path(output_dir) / ptf_name
    risultati = run_lazy_analysis(
        portfolio_cfg=_pf_weights,
        output_dir=sub_output_dir,
        title=ptf_name,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        init_cash=init_cash,
        fees=fees,
        years=years,
        plot=False,
        save_png=True,
        auto_freq=True,
        freq_selection_metric='sharpe',
        verbose=verbose,
        min_years=min_years,
        _preloaded_price=_price_ptf,
        _preloaded_bm_data=_price_bm,
    )
    pf_proposed = run_bh_backtest(_pf_weights, start_date, end_date,
                                  init_cash, fees, risultati['best_freq'],
                                  min_years=min_years,
                                  _preloaded=(_ptf_loaded_weights, _price_ptf))
    if pf_proposed is None:
        raise ValueError(
            f"[_compute_lazy_full] {ptf_name}: storico insufficiente per "
            f"run_bh_backtest (min_years={min_years}) — skip."
        )

    # 2b. benchmark B&H (singolo ticker) per il confronto §3 della relazione
    if _bm_preloaded_arg is not None:
        try:
            pf_benchmark = run_bh_backtest({benchmark: 1.0}, start_date, end_date,
                                           init_cash, fees, None,
                                           min_years=min_years,
                                           _preloaded=_bm_preloaded_arg)
        except Exception:
            pf_benchmark = None
    else:
        pf_benchmark = None

    # 3. stabilità rolling del PTF reale
    stability = lazy_rolling_stability(
        pf=pf_proposed,
        target_horizon=5,
        loss_prob_threshold=0.02,
        verbose=False,
    )

    # 4. MC A1 (iid bootstrap) e A2 (block bootstrap)
    rng_a1 = np.random.default_rng(42)
    mc_a1 = mc_run_iid_bootstrap(pf_proposed, n_simulations_mc_a, rng_a1)
    rng_a2 = np.random.default_rng(42)
    mc_a2 = mc_run_block_bootstrap(pf_proposed, 20, n_simulations_mc_a, rng_a2)

    # 5. MC B (rebalancing con jitter)
    mc_b = lazy_mc_block_b_rebalancing(
        portfolio=_pf_weights, start_date=start_date, end_date=end_date,
        best_freq=risultati['best_freq'], n_simulations=n_simulations_mc_b,
        jitter_days=30, init_cash=init_cash, fees=fees, verbose=False,
        min_years=min_years, _preloaded_price=_price_ptf,
    )

    # 6. DSR
    sr = float(pf_proposed.sharpe_ratio())
    T = int(pf_proposed.value().dropna().__len__())
    dsr = ofc_compute_dsr(sr_hat=sr, n_trials=1, T=T)

    # 7. metriche e verdetto
    cagr = _cagr_from_equity(pf_proposed)
    maxdd = abs(float(pf_proposed.max_drawdown()))

    checks = {
        'mc_a2_sharpe_p50_positive': mc_a2['percentiles']['p50']['Sharpe'] > 0,
        'mc_b_skill': mc_b['skill'],
        'dsr_positive': dsr > 0,
    }
    n_passed = sum(checks.values())
    verdetto = 'PROMOSSO' if n_passed >= 2 else 'RIGETTATO'

    # 8. riga di classificazione
    row = {
        'Nome': ptf_name,
        'BestFreq': risultati['best_freq'] or 'BH',
        'CAGR%': round(cagr * 100, 2),
        'Sharpe': round(sr, 3),
        'MaxDD%': round(maxdd * 100, 2),
        'StabilityOK': stability['stable'],
        'PLoss5y%': round(stability['p_loss_at_horizon'] * 100, 2) if not np.isnan(stability['p_loss_at_horizon']) else np.nan,
        'MinSafeHorizon': stability.get('min_safe_horizon'),
        'MC_A2_Sharpe_p50': round(mc_a2['percentiles']['p50']['Sharpe'], 3),
        'MC_B_pvalue': round(mc_b['p_value_sharpe'], 3),
        'MC_B_skill': mc_b['skill'],
        'DSR': round(dsr, 3),
        'CriteriPassati': f"{n_passed}/3",
        'Verdetto': verdetto,
    }

    # 9. performance report (solo se richiesto per PDF investitore — evita overhead
    #    su path CSV-only che non usa questi dati)
    out_perf = None
    if need_out:
        from u_functions import generate_lazy_portfolio_performance
        benchmark_data_for_out = pf_benchmark.value() if pf_benchmark is not None else None
        out_perf = generate_lazy_portfolio_performance(
            pf=pf_proposed,
            portfolio_title=ptf_name,
            benchmark=benchmark,
            benchmark_data=benchmark_data_for_out,
            show_report=False,
            show_plots=False,
        )

    return {
        'risultati': risultati,
        'pf_proposed': pf_proposed,
        'pf_benchmark': pf_benchmark,
        'benchmark': benchmark,
        'stability': stability,
        'mc_a1': mc_a1,
        'mc_a2': mc_a2,
        'mc_b': mc_b,
        'dsr': dsr,
        'sr': sr,
        'T': T,
        'cagr': cagr,
        'maxdd': maxdd,
        'checks': checks,
        'verdetto': verdetto,
        'row': row,
        'portfolio_cfg': portfolio_cfg,
        'out': out_perf,
    }


def run_lazy_batch_analysis(
    registry: dict,
    ptf_names: list,
    output_dir,
    start_date='2016-01-01',
    end_date=None,
    benchmark='SPY',
    init_cash=100_000.0,
    fees=0.001,
    years=10,
    n_simulations_mc_a=1000,
    n_simulations_mc_b=500,
    cache_dir=None,           # default stabile: <project_root>/outputs/lazy_cache/
    override: bool = False,
    verbose=False,
    details_out: dict = None,
    min_years: int = 5,
) -> "pd.DataFrame":
    """
    Esegue la pipeline Lazy completa (run_lazy_analysis + stability +
    MC A1/A2 + MC B + DSR + verdetto) su una lista di PTF dal registry.
    Salva un CSV 'classification_<timestamp>.csv' in output_dir.
    Ritorna il DataFrame classification.

    Se `details_out` è un dict, viene popolato in-place con
    {ptf_name: rich_dict} dove rich_dict è l'output di _compute_lazy_full()
    (oggetti completi: pf, stability, mc_a1/a2, mc_b, dsr, frontiera, ecc.).
    Necessario per generare la relazione tecnica PDF (iq l-analyze --pdf):
    quando richiesto, i PTF vengono ricalcolati anche se in cache, così gli
    oggetti rich sono disponibili (la cache row/mc_a2 viene comunque aggiornata).
    """
    from pathlib import Path
    from datetime import datetime as _dt
    import pickle

    if cache_dir:
        _cache_dir = Path(cache_dir)
    else:
        # Path stabile indipendente da output_dir (che può essere qualsiasi stringa).
        # mc_functions.py è in <project>/notebooks/libs_py/, quindi:
        # .parent.parent.parent = <project_root>/
        _cache_dir = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'lazy_cache'
    _cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Quando servono gli oggetti rich (relazione tecnica PDF) la cache row non
    # basta: occorre ricalcolare per riavere pf/stability/mc/frontiera.
    need_details = details_out is not None

    for ptf_name in ptf_names:
        # 1. risoluzione dal registry
        portfolio_cfg = registry.get(ptf_name)
        if portfolio_cfg is None:
            print(f"[WARN] PTF '{ptf_name}' non trovato nel registry: skip.")
            continue

        cache_file = _cache_dir / f'{ptf_name}.pkl'
        if (not override) and (not need_details) and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_row = pickle.load(f)
                rows.append(cached_row)
                if verbose:
                    print(f"[CACHE] {ptf_name}: caricato da {cache_file}")
                continue
            except Exception as e:
                if verbose:
                    print(f"[CACHE] {ptf_name}: cache corrotta ({e}), ricalcolo.")

        try:
            # 2-8. pipeline completa per il singolo PTF (oggetti rich)
            rich = _compute_lazy_full(
                portfolio_cfg=portfolio_cfg,
                ptf_name=ptf_name,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                benchmark=benchmark,
                init_cash=init_cash,
                fees=fees,
                years=years,
                n_simulations_mc_a=n_simulations_mc_a,
                n_simulations_mc_b=n_simulations_mc_b,
                verbose=verbose,
                need_out=need_details,
                min_years=min_years,
            )
            row = rich['row']
            mc_a2 = rich['mc_a2']
            rows.append(row)
            if need_details:
                # path della cache MC esposto per project_lazy_capital (§6 PDF)
                rich['cache_dir'] = str(_cache_dir)
                details_out[ptf_name] = rich
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(row, f)
            except Exception as e:
                if verbose:
                    print(f"[CACHE] {ptf_name}: impossibile salvare cache ({e}).")
            _mc_cache_file = _cache_dir / f'{ptf_name}_mc_a2.pkl'
            try:
                with open(_mc_cache_file, 'wb') as f:
                    pickle.dump(mc_a2, f)
            except Exception as e:
                if verbose:
                    print(f"[CACHE] {ptf_name}: impossibile salvare mc_a2 cache ({e}).")
        except Exception as e:
            print(f"[WARN] PTF '{ptf_name}' fallito: {type(e).__name__}: {e} — continuo col prossimo.")
            continue

    # 9. costruzione e salvataggio CSV
    df_classification = pd.DataFrame(rows)
    if df_classification.empty:
        if verbose:
            print(f"[run_lazy_batch_analysis] Nessun PTF analizzato con successo - CSV non salvato.")
        return df_classification
    ts = _dt.now().strftime('%Y%m%d_%H%M%S')
    csv_path = Path(output_dir) / f'classification_{ts}.csv'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_classification.to_csv(csv_path, index=False)
    if verbose:
        print(f"Classification salvata: {csv_path}")
        print(df_classification.to_string(index=False))
    return df_classification


def load_lazy_classifications(lazy_dir, pattern='*/classification_*.csv',
                               filter_mode='ALL', sort_by='Sharpe',
                               sort_asc=False, dedup=True):
    """Carica e consolida tutti i CSV classification Lazy trovati.

    Se dedup=True (default, classifica principale), in presenza dello stesso
    PTF in run diversi mantiene solo il run piu' recente. Usa dedup=False per
    il confronto multi-run.
    """
    import glob
    from pathlib import Path
    files = sorted(glob.glob(str(Path(lazy_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"Nessun CSV trovato in {Path(lazy_dir) / pattern}")
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if d.empty:
                continue
        except Exception as e:
            print(f"[WARN] Skip {f}: {e}")
            continue
        d['_run'] = Path(f).parent.name
        d['_file'] = f
        dfs.append(d)
    if not dfs:
        raise FileNotFoundError(f"Nessun CSV valido trovato in {Path(lazy_dir) / pattern}")
    df = pd.concat(dfs, ignore_index=True)
    if dedup:
        df = df.sort_values('_run', ascending=False)
        df = df.drop_duplicates(subset=['Nome'], keep='first')
    if filter_mode == 'PROMOTED':
        df = df[df['Verdetto'] == 'PROMOSSO']
    elif filter_mode == 'FAILED':
        df = df[df['Verdetto'] == 'RIGETTATO']
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=sort_asc)
    return df.reset_index(drop=True)


def plot_lazy_equity_curves(
    promoted_df,
    registry: dict,
    top: int = 5,
    sort_by: str = 'Sharpe',
    start_date: str = '2016-01-01',
    end_date=None,
    init_cash: float = 100_000.0,
    fees: float = 0.001,
    mode: str = 'overlay',          # 'overlay' | 'separate'
    common_period_only: bool = True,
) -> None:
    """
    Plotta le equity curve dei top N PTF promossi (per sort_by).
    mode='overlay': un solo grafico Plotly con tutte le curve
    normalizzate a base 100 (confrontabili indipendentemente dai pesi).
    mode='separate': un grafico per PTF.
    common_period_only=True (default): tutte le curve vengono tagliate
    al periodo comune prima della normalizzazione, rendendole realmente
    confrontabili quando i PTF hanno storici di lunghezza diversa.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    top_df = promoted_df.sort_values(sort_by, ascending=False).head(top)

    # Fase 1: accumula le equity series raw (non normalizzate)
    raw_curves = {}
    best_freqs = {}
    for _, row in top_df.iterrows():
        nome = row['Nome']
        portfolio_cfg = registry.get(nome)
        if portfolio_cfg is None:
            print(f'  [SKIP] {nome}: non trovato nel registry')
            continue
        best_freq = None if row['BestFreq'] == 'BH' else row['BestFreq']
        try:
            pf = run_bh_backtest(portfolio_cfg, start_date, end_date,
                                 init_cash, fees, best_freq)
        except Exception as e:
            print(f'  [SKIP] {nome}: errore backtest — {e}')
            continue
        if pf is None:
            print(f'  [SKIP] {nome}: storico insufficiente per run_bh_backtest — skip.')
            continue
        eq = pf.value()
        if isinstance(eq, pd.DataFrame):
            eq = eq.iloc[:, 0]
        raw_curves[nome] = eq
        best_freqs[nome] = row['BestFreq']

    if not raw_curves:
        print('Nessuna equity curve da plottare.')
        return

    # Fase 2: calcola il periodo comune (se richiesto)
    titolo_suffix = ''
    use_common = common_period_only and len(raw_curves) > 1
    if use_common:
        common_start = max(eq.index.min() for eq in raw_curves.values())
        common_end   = min(eq.index.max() for eq in raw_curves.values())
        if common_start >= common_end:
            print('[WARN] Nessuna sovrapposizione tra le equity curves — uso periodi individuali.')
            use_common = False
        else:
            titolo_suffix = f" — periodo comune {common_start.date()} → {common_end.date()}"

    # Fase 3: taglia al periodo comune (se attivo) e normalizza a base 100
    curves = {}
    for nome, eq in raw_curves.items():
        if use_common:
            eq = eq[(eq.index >= common_start) & (eq.index <= common_end)]
        curves[nome] = eq / eq.iloc[0] * 100

    # Fase 4: plot
    if mode == 'overlay':
        fig = go.Figure()
        for nome, serie in curves.items():
            fig.add_trace(go.Scatter(x=serie.index, y=serie.values,
                                     mode='lines', name=nome))
        fig.update_layout(
            title=f'Top {top} PTF promossi — Equity Curve (base 100){titolo_suffix}',
            height=500,
        )
        fig.show()
    else:
        for nome, eq_norm in curves.items():
            fig = px.line(
                eq_norm,
                title=f'{nome} — Equity Curve base 100 (freq={best_freqs[nome]}){titolo_suffix}',
            )
            fig.update_layout(height=350)
            fig.show()


def style_lazy_classification(df):
    """
    Applica gradiente colore a tutte le metriche numeriche della
    classification Lazy. Verde=migliore, rosso=peggiore per ogni
    colonna (Sharpe/CAGR/DSR alto=verde, MaxDD/PLoss5y%/MC_B_pvalue
    alto=rosso - direzione invertita dove 'meno è meglio').
    Ritorna un pandas Styler.
    """
    cols_higher_better = ['CAGR%', 'Sharpe', 'MC_A2_Sharpe_p50', 'DSR']
    cols_lower_better  = ['MaxDD%', 'PLoss5y%', 'MC_B_pvalue']

    styler = df.style
    for col in cols_higher_better:
        if col in df.columns:
            styler = styler.background_gradient(subset=[col], cmap='RdYlGn')
    for col in cols_lower_better:
        if col in df.columns:
            styler = styler.background_gradient(subset=[col], cmap='RdYlGn_r')
    styler = styler.format(precision=3)
    return styler


def lazy_rolling_stability(
    pf,
    horizons_years: list = None,
    annual_trading_days: int = 252,
    target_horizon: int = 5,
    loss_prob_threshold: float = 0.02,
    min_obs: int = 30,
    verbose: bool = False,
) -> dict:
    """
    Stability test basato su rolling return del PTF reale (pesi fissi).
    Calcola P(rendimento rolling a target_horizon anni < 0%) e la
    confronta con loss_prob_threshold.

    Returns dict:
        'prob_df':            DataFrame probabilità per orizzonte
        'p_loss_at_horizon':   float - P(R<0%) all'orizzonte target
        'target_horizon':      int
        'loss_prob_threshold': float
        'stable':              bool - True se p_loss_at_horizon <= threshold
        'min_safe_horizon':    int | None - primo orizzonte con P(R<0%) <= threshold
    """
    from u_functions import analyze_rolling_horizons

    if horizons_years is None:
        horizons_years = [1, 2, 3, 4, 5]

    daily_rets = pf.returns()
    if isinstance(daily_rets, pd.DataFrame):
        daily_rets = daily_rets.iloc[:, 0]
    daily_rets = daily_rets.dropna().sort_index()

    _empty = {
        'prob_df': pd.DataFrame(),
        'p_loss_at_horizon': np.nan,
        'target_horizon': target_horizon,
        'loss_prob_threshold': loss_prob_threshold,
        'stable': False,
        'min_safe_horizon': None,
    }
    if daily_rets.empty:
        return _empty

    roll_cum_dict = {}
    for y in horizons_years:
        ndays = int(annual_trading_days * y)
        label = f"{y}y"
        roll = (1 + daily_rets).rolling(ndays).apply(np.prod, raw=True) - 1
        roll_cum_dict[label] = roll
    roll_cum_df = pd.DataFrame(roll_cum_dict, index=daily_rets.index)

    result = analyze_rolling_horizons(
        roll_cum_df,
        loss_thresholds=(0.0,),
        min_obs=min_obs,
        target_prob=loss_prob_threshold,
    )
    prob_df = result['prob_df']

    if target_horizon in prob_df.index:
        p_loss = float(prob_df.loc[target_horizon, 'p_lt_0.0'])
    else:
        p_loss = np.nan

    stable = (not np.isnan(p_loss)) and (p_loss <= loss_prob_threshold)
    min_safe = result.get('min_safe_horizon')

    if verbose:
        print(f"Rolling stability — target={target_horizon}y, soglia={loss_prob_threshold:.1%}")
        print(prob_df)
        if not np.isnan(p_loss):
            print(f"P(R<0%) a {target_horizon}y: {p_loss:.2%}")
        else:
            print("P(R<0%) a {target_horizon}y: N/A")
        print(f"Stabile: {'✅' if stable else '⚠️'}  Min safe horizon: {min_safe}")

    return {
        'prob_df': prob_df,
        'p_loss_at_horizon': p_loss,
        'target_horizon': target_horizon,
        'loss_prob_threshold': loss_prob_threshold,
        'stable': stable,
        'min_safe_horizon': min_safe,
    }


def project_lazy_capital(
    ptf_names: list,
    cache_dir,
    initial_capital: float = 10_000.0,
    horizon_years: int = 10,
    percentiles: tuple = (10, 50, 90),
    plot: bool = True,
) -> dict:
    """
    Proietta l'evoluzione futura del capitale per uno o più PTF,
    usando la distribuzione di CAGR già simulata via MC Block Bootstrap
    (cache '{ptf_name}_mc_a2.pkl'). Per ogni PTF, estrae i CAGR
    simulati, calcola i percentili richiesti, e proietta:
    capitale(t) = initial_capital * (1 + cagr_percentile) ** t
    per t = 0..horizon_years.

    Returns dict con:
      {ptf_name: {percentile_label: pd.Series(anno 0..horizon_years)}, ...}
      '_fig_overview': go.Figure con P50 di tutti i PTF
      '_figs_detail':  {ptf_name: go.Figure con banda + P50}
    """
    import pickle
    from pathlib import Path

    _DISCLAIMER = (
        "Proiezione basata su distribuzione storica CAGR "
        "(Monte Carlo Block Bootstrap) — non garanzia di risultati futuri"
    )

    results = {}

    for ptf_name in ptf_names:
        mc_file = Path(cache_dir) / f'{ptf_name}_mc_a2.pkl'
        if not mc_file.exists():
            print(
                f"[WARN] {ptf_name}: cache MC non trovata ({mc_file}), skip. "
                f"Rilancia 'iq lazy-analyze --ptf {ptf_name} --override' per generarla."
            )
            continue
        try:
            with open(mc_file, 'rb') as f:
                mc_a2 = pickle.load(f)
        except Exception as e:
            print(f"[WARN] {ptf_name}: cache MC corrotta ({e}), skip.")
            continue

        cagr_sims = mc_a2['metrics_per_sim']['CAGR'].dropna().values
        if len(cagr_sims) == 0:
            print(f"[WARN] {ptf_name}: nessuna simulazione CAGR valida, skip.")
            continue

        years_arr = np.arange(0, horizon_years + 1)
        ptf_result = {}
        for p in percentiles:
            cagr_p = float(np.nanpercentile(cagr_sims, p))
            capital_path = initial_capital * (1 + cagr_p) ** years_arr
            ptf_result[f'p{p}'] = pd.Series(capital_path, index=years_arr, name=f'p{p}')
        results[ptf_name] = ptf_result

    if plot and results:
        # GRAFICO 1: overview, solo P50, tutti i PTF insieme
        fig_overview = go.Figure()
        for ptf_name in results.keys():
            p50_series = results[ptf_name].get('p50')
            if p50_series is None:
                continue
            fig_overview.add_trace(go.Scatter(
                x=p50_series.index, y=p50_series.values,
                mode='lines', name=ptf_name,
            ))
        fig_overview.update_layout(
            title=dict(
                text=(
                    f"Proiezione capitale — confronto P50 — €{initial_capital:,.0f} iniziali, "
                    f"{horizon_years} anni<br>"
                    f"<sub style='color:gray'>{_DISCLAIMER}</sub>"
                ),
            ),
            xaxis_title="Anno", yaxis_title="Capitale (€)",
        )
        fig_overview.show()

        # GRAFICI 2..N: uno per PTF, con banda di confidenza
        figs_detail = {}
        for ptf_name in results.keys():
            p_keys = sorted(results[ptf_name].keys())
            p_lo_key, p_hi_key = p_keys[0], p_keys[-1]
            p50_key = 'p50' if 'p50' in results[ptf_name] else p_keys[len(p_keys) // 2]

            fig_detail = go.Figure()
            lo = results[ptf_name][p_lo_key]
            hi = results[ptf_name][p_hi_key]
            fig_detail.add_trace(go.Scatter(
                x=hi.index, y=hi.values, mode='lines',
                line=dict(width=0.5, color='rgba(31,119,180,0.4)'),
                name=f'{p_hi_key}', showlegend=True,
                hovertemplate='Anno %{x}: €%{y:,.0f}<extra>' + p_hi_key + '</extra>',
            ))
            fig_detail.add_trace(go.Scatter(
                x=lo.index, y=lo.values, mode='lines',
                line=dict(width=0.5, color='rgba(31,119,180,0.4)'),
                fill='tonexty', fillcolor='rgba(31,119,180,0.15)',
                name=f'{p_lo_key}', showlegend=True,
                hovertemplate='Anno %{x}: €%{y:,.0f}<extra>' + p_lo_key + '</extra>',
            ))
            p50 = results[ptf_name][p50_key]
            fig_detail.add_trace(go.Scatter(
                x=p50.index, y=p50.values, mode='lines',
                name=f'{ptf_name} (P50)', line=dict(width=2),
            ))
            fig_detail.update_layout(
                title=dict(
                    text=(
                        f"{ptf_name} — Proiezione capitale (banda {p_lo_key}-{p_hi_key})<br>"
                        f"<sub style='color:gray'>{_DISCLAIMER}</sub>"
                    ),
                ),
                xaxis_title="Anno", yaxis_title="Capitale (€)",
            )
            fig_detail.show()
            figs_detail[ptf_name] = fig_detail

        results['_fig_overview'] = fig_overview
        results['_figs_detail'] = figs_detail

    return results


# ---------------------------------------------------------------------------
# Relazione Tecnica Lazy (PDF) — funzione indipendente (no riuso di
# generate_relazione_tecnica di r_functions). Stesso stack reportlab, palette
# coerente, struttura §1-§7 specifica per i Lazy portfolio.
# ---------------------------------------------------------------------------

# Palette coerente con la relazione R (replicata localmente per indipendenza)
_RL_LAZY_NAVY    = '#1B2A4A'
_RL_LAZY_NAVY_LT = '#2C3E6B'
_RL_LAZY_GREEN   = '#27AE60'
_RL_LAZY_RED     = '#E74C3C'
_RL_LAZY_GRAY_LT = '#F5F6FA'
_RL_LAZY_GRAY_BD = '#D5D8DC'
_RL_LAZY_TEXT    = '#2C3E50'


def generate_relazione_tecnica_lazy(
    *,
    portfolio_title: str,
    asset_allocation: dict,
    period: tuple,
    pf_proposed,
    stability: dict,
    dsr: float,
    mc_a: dict,
    mc_b: dict,
    verdetto: str,
    output_path,
    ptf_type: str = 'lazy',
    optimal_weights: dict = None,
    best_freq: str = None,
    pf_benchmark=None,
    benchmark: str = 'SPY',
    frontier_df=None,
    capital_projection: dict = None,
    initial_capital: float = 10_000.0,
    plots_dir=None,
    gen_date: str = None,
):
    """
    Genera la Relazione Tecnica PDF di un Lazy portfolio con reportlab.

    Funzione INDIPENDENTE da generate_relazione_tecnica() di r_functions:
    nessun import cross-funzione, solo riuso di _cagr_from_equity() (helper di
    basso livello già presente in questo modulo). Struttura:

        §1 Identità            — nome, tipo, asset allocation
        §2 Configurazione      — asset class/pesi, freq ribilanciamento, periodo
        §3 Metriche comparative— CAGR/Sharpe/MaxDD vs benchmark, frontiera
                                 (reale vs teorica)
        §4 Validazione         — lazy_rolling_stability (P(roll 5y < 0%)), DSR
        §5 Monte Carlo         — Block A (CI) + Block B (skill test) — solo
                                 risultati numerici, nessun disclaimer/caveat
        §6 Proiezione capitale — project_lazy_capital, percentili P10-P50-P90
        §7 Decisione finale    — verdetto promozione

    Parameters
    ----------
    asset_allocation : dict
        ticker -> peso del PTF reale (config).
    pf_proposed : vbt.Portfolio
        Backtest del PTF reale (pesi fissi, best_freq).
    mc_a : dict
        Output di mc_run_block_bootstrap (chiavi 'percentiles', 'actual_metrics').
    mc_b : dict
        Output di lazy_mc_block_b_rebalancing.
    capital_projection : dict | None
        {'p10': Series, 'p50': Series, 'p90': Series} da project_lazy_capital.
    output_path : path-like
        Path del PDF (es. outputs/l_analysis/<timestamp>/<ptf>_relazione_tecnica.pdf).

    Returns
    -------
    Path
        Path del PDF scritto.
    """
    from pathlib import Path
    import datetime as _dt
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.platypus import Image as _RLImage
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    try:
        from PIL import Image as _PILImage
        _HAS_PIL = True
    except ImportError:
        _HAS_PIL = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(plots_dir) if plots_dir is not None else None
    if gen_date is None:
        gen_date = _dt.date.today().isoformat()
    period_start, period_end = (period if period and len(period) == 2
                                else (str(period), gen_date))

    C_NAVY    = rl_colors.HexColor(_RL_LAZY_NAVY)
    C_NAVY_LT = rl_colors.HexColor(_RL_LAZY_NAVY_LT)
    C_GREEN   = rl_colors.HexColor(_RL_LAZY_GREEN)
    C_RED     = rl_colors.HexColor(_RL_LAZY_RED)
    C_GRAY_LT = rl_colors.HexColor(_RL_LAZY_GRAY_LT)
    C_GRAY_BD = rl_colors.HexColor(_RL_LAZY_GRAY_BD)
    C_WHITE   = rl_colors.white
    C_TEXT    = rl_colors.HexColor(_RL_LAZY_TEXT)

    PAGE_W, PAGE_H = A4
    MARGIN    = 20 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN

    styles = getSampleStyleSheet()

    def _st(name, parent='Normal', **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    st_title    = _st('_lz_title', 'Title', fontSize=24, textColor=C_NAVY,
                      spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Bold')
    st_subtitle = _st('_lz_sub', fontSize=10, textColor=C_NAVY_LT,
                      spaceAfter=12, alignment=TA_CENTER)
    st_section  = _st('_lz_sec', fontSize=13, textColor=C_NAVY, spaceBefore=12,
                      spaceAfter=5, fontName='Helvetica-Bold')
    st_subsec   = _st('_lz_ssec', fontSize=10.5, textColor=C_NAVY_LT, spaceBefore=7,
                      spaceAfter=3, fontName='Helvetica-Bold')
    st_body     = _st('_lz_body', fontSize=9.5, textColor=C_TEXT, spaceAfter=6,
                      alignment=TA_JUSTIFY, leading=14)
    st_caption  = _st('_lz_cap', fontSize=7.5, textColor=C_NAVY_LT, spaceAfter=6,
                      alignment=TA_CENTER, fontName='Helvetica-Oblique')
    st_cell_hdr = _st('_lz_chdr', fontSize=8.5, textColor=C_WHITE,
                      fontName='Helvetica-Bold')
    st_cell_hdrc= _st('_lz_chdrc', fontSize=8.5, textColor=C_WHITE,
                      fontName='Helvetica-Bold', alignment=TA_CENTER)
    st_verd     = _st('_lz_verd', fontSize=9, textColor=C_WHITE,
                      fontName='Helvetica-Bold', alignment=TA_CENTER)
    st_vbox     = _st('_lz_vbox', fontSize=9.5, textColor=C_NAVY, leading=14,
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

    _ptf_label  = f"Relazione Tecnica · {portfolio_title}"
    _foot_label = f"Generato il {gen_date} · investia.cloud · uso interno"

    def _draw_hf(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, PAGE_H - 12 * mm, PAGE_W, 12 * mm, fill=1, stroke=0)
        canvas.setFont('Helvetica-Bold', 9)
        canvas.setFillColor(C_WHITE)
        canvas.drawString(MARGIN, PAGE_H - 8 * mm,
                          'TSlab — Lazy Portfolio Lab')
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
        if plots_dir is None:
            return []
        p = plots_dir / fname
        if not p.exists():
            return []
        try:
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

    def _hr():
        return HRFlowable(width='100%', thickness=0.5, color=C_NAVY_LT, spaceAfter=8)

    def _metric(pf, which):
        if pf is None:
            return 'N/A'
        try:
            if which == 'cum':    return f"{float(pf.total_return()) * 100:.1f}%"
            if which == 'cagr':   return f"{_cagr_from_equity(pf) * 100:.1f}%"
            if which == 'sharpe': return f"{float(pf.sharpe_ratio()):.2f}"
            if which == 'maxdd':  return f"{abs(float(pf.max_drawdown())) * 100:.1f}%"
        except Exception:
            return 'N/A'
        return 'N/A'

    def _pct(x, dec=1):
        try:
            return f"{float(x) * 100:.{dec}f}%"
        except Exception:
            return 'N/A'

    def _num(x, dec=2):
        try:
            return f"{float(x):.{dec}f}"
        except Exception:
            return 'N/A'

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(f"Relazione Tecnica · {portfolio_title}", st_title))
    story.append(Paragraph(
        f"Lazy portfolio ({ptf_type}) · Benchmark {benchmark} · "
        f"Ribilanciamento {best_freq or 'Buy & Hold'}", st_subtitle))
    story.append(_hr())

    # ── §1 Identità ──────────────────────────────────────────────────────────
    story.append(Paragraph("1. Identità del Portafoglio", st_section))
    n_assets = len(asset_allocation or {})
    id_data = [
        [Paragraph('Campo', st_cell_hdr), Paragraph('Valore', st_cell_hdr)],
        ['Nome',             portfolio_title],
        ['Tipo',             ptf_type],
        ['Engine',           'Lazy portfolio (pesi fissi, ribilanciamento periodico)'],
        ['N. asset',         str(n_assets)],
        ['Asset allocation', ', '.join(f"{k} {v:.0%}" for k, v in (asset_allocation or {}).items())],
        ['Benchmark',        benchmark],
        ['Periodo analisi',  f"{period_start} → {period_end}"],
        ['Data generazione', gen_date],
    ]
    id_t = Table(id_data, colWidths=[45 * mm, CONTENT_W - 45 * mm])
    id_t.setStyle(TableStyle(_ts_base()))
    story += [id_t, Spacer(1, 5 * mm)]

    # ── §2 Configurazione ────────────────────────────────────────────────────
    story.append(Paragraph("2. Configurazione", st_section))
    cfg_rows = [[Paragraph('Asset', st_cell_hdr),
                 Paragraph('Peso reale', st_cell_hdrc),
                 Paragraph('Peso max-Sharpe (frontiera)', st_cell_hdrc)]]
    _ow = optimal_weights or {}
    for tk, w in (asset_allocation or {}).items():
        w_opt = _ow.get(tk, _ow.get(tk.upper()))
        cfg_rows.append([tk, f"{w:.1%}",
                         f"{w_opt:.1%}" if w_opt is not None else 'N/A'])
    cfg_t = Table(cfg_rows, colWidths=[CONTENT_W - 100 * mm, 50 * mm, 50 * mm])
    cfg_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
    story += [cfg_t, Spacer(1, 3 * mm)]
    cfg2 = [
        [Paragraph('Parametro', st_cell_hdr), Paragraph('Valore', st_cell_hdr)],
        ['Frequenza ribilanciamento', best_freq or 'Buy & Hold (nessun ribilanciamento)'],
        ['Periodo storico backtest',  f"{period_start} → {period_end}"],
    ]
    cfg2_t = Table(cfg2, colWidths=[60 * mm, CONTENT_W - 60 * mm])
    cfg2_t.setStyle(TableStyle(_ts_base()))
    story += [cfg2_t, Spacer(1, 5 * mm)]

    # ── §3 Metriche comparative ──────────────────────────────────────────────
    story.append(Paragraph("3. Metriche Comparative", st_section))
    cmp_rows = [
        [Paragraph('Metrica', st_cell_hdr),
         Paragraph(portfolio_title, st_cell_hdrc),
         Paragraph(f"Benchmark ({benchmark})", st_cell_hdrc)],
        ['CAGR',           _metric(pf_proposed, 'cagr'),   _metric(pf_benchmark, 'cagr')],
        ['Sharpe',         _metric(pf_proposed, 'sharpe'), _metric(pf_benchmark, 'sharpe')],
        ['Max Drawdown',   _metric(pf_proposed, 'maxdd'),  _metric(pf_benchmark, 'maxdd')],
        ['Rendimento tot.', _metric(pf_proposed, 'cum'),   _metric(pf_benchmark, 'cum')],
    ]
    _hw = (CONTENT_W - 50 * mm) / 2
    cmp_t = Table(cmp_rows, colWidths=[50 * mm, _hw, _hw])
    cmp_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
    story += [cmp_t, Spacer(1, 4 * mm)]

    # Frontiera efficiente: reale vs teorica (riga Max Sharpe di df_special)
    if frontier_df is not None:
        try:
            if 'Max Sharpe' in frontier_df.index:
                _r = frontier_df.loc['Max Sharpe']
                story.append(Paragraph("3.a Frontiera efficiente — Max Sharpe (teorico vs reale)", st_subsec))
                fr_rows = [
                    [Paragraph('Grandezza', st_cell_hdr),
                     Paragraph('Teorico', st_cell_hdrc),
                     Paragraph('Reale', st_cell_hdrc)],
                    ['Return',     _pct(_r.get('Return')),     _pct(_r.get('Real Return'))],
                    ['Volatility', _pct(_r.get('Volatility')), _pct(_r.get('Real Volatility'))],
                    ['Sharpe',     _num(_r.get('Sharpe')),     _num(_r.get('Real Sharpe'))],
                ]
                fr_t = Table(fr_rows, colWidths=[50 * mm, _hw, _hw])
                fr_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
                story += [fr_t, Spacer(1, 3 * mm)]
        except Exception:
            pass
    story += _img('frontier.png', 'Frontiera efficiente')
    story += _img('freq_comparison.png', 'Confronto frequenze di ribilanciamento')
    story.append(Spacer(1, 3 * mm))

    # ── §4 Validazione statistica ────────────────────────────────────────────
    story.append(Paragraph("4. Validazione Statistica", st_section))
    _th = stability.get('target_horizon', 5)
    _ploss = stability.get('p_loss_at_horizon')
    _msh = stability.get('min_safe_horizon')
    val_rows = [
        [Paragraph('Test', st_cell_hdr),
         Paragraph('Valore', st_cell_hdrc),
         Paragraph('Esito', st_cell_hdrc)],
        [f'P(rendimento rolling {_th}y < 0%)',
         _pct(_ploss) if _ploss is not None else 'N/A',
         'STABILE' if stability.get('stable') else 'INSTABILE'],
        ['Soglia P(loss)', _pct(stability.get('loss_prob_threshold')), '—'],
        ['Min safe horizon', f"{_msh}y" if _msh is not None else 'N/A', '—'],
        ['Deflated Sharpe Ratio (DSR)', _num(dsr, 3),
         'POSITIVO' if (dsr is not None and dsr > 0) else 'NON POSITIVO'],
    ]
    val_t = Table(val_rows, colWidths=[70 * mm, _hw - 10 * mm, _hw + 10 * mm])
    val_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
    story += [val_t, Spacer(1, 5 * mm)]

    # ── §5 Monte Carlo (solo risultati numerici, nessun caveat) ──────────────
    story.append(Paragraph("5. Monte Carlo", st_section))
    story.append(Paragraph("5.a Block A — Intervalli di confidenza (Block Bootstrap)", st_subsec))
    _perc = (mc_a or {}).get('percentiles', {})
    _act  = (mc_a or {}).get('actual_metrics', {})
    def _pv(lbl, metric, pct=True):
        try:
            v = _perc.get(lbl, {}).get(metric)
            return _pct(v) if pct else _num(v)
        except Exception:
            return 'N/A'
    def _av(metric, pct=True):
        try:
            v = _act.get(metric)
            return _pct(v) if pct else _num(v)
        except Exception:
            return 'N/A'
    mca_rows = [
        [Paragraph('Metrica', st_cell_hdr),
         Paragraph('p5', st_cell_hdrc), Paragraph('p50', st_cell_hdrc),
         Paragraph('p95', st_cell_hdrc), Paragraph('Reale', st_cell_hdrc)],
        ['CAGR',   _pv('p5', 'CAGR'),   _pv('p50', 'CAGR'),   _pv('p95', 'CAGR'),   _av('CAGR')],
        ['Sharpe', _pv('p5', 'Sharpe', False), _pv('p50', 'Sharpe', False),
                   _pv('p95', 'Sharpe', False), _av('Sharpe', False)],
        ['MaxDD',  _pv('p5', 'MaxDD'),  _pv('p50', 'MaxDD'),  _pv('p95', 'MaxDD'),  _av('MaxDD')],
    ]
    _cw = (CONTENT_W - 40 * mm) / 4
    mca_t = Table(mca_rows, colWidths=[40 * mm, _cw, _cw, _cw, _cw])
    mca_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
    story += [mca_t, Spacer(1, 3 * mm)]

    story.append(Paragraph("5.b Block B — Skill test ribilanciamento", st_subsec))
    mcb_rows = [
        [Paragraph('Grandezza', st_cell_hdr), Paragraph('Valore', st_cell_hdrc)],
        ['Sharpe reale',       _num((mc_b or {}).get('actual_sharpe'))],
        ['p-value Sharpe',     _num((mc_b or {}).get('p_value_sharpe'), 3)],
        ['CAGR reale',         _pct((mc_b or {}).get('actual_cagr'))],
        ['p-value CAGR',       _num((mc_b or {}).get('p_value_cagr'), 3)],
        ['Skill (p<0.05)',     'SI' if (mc_b or {}).get('skill') else 'NO'],
        ['N. simulazioni',     str((mc_b or {}).get('n_simulations', 'N/A'))],
    ]
    mcb_t = Table(mcb_rows, colWidths=[70 * mm, CONTENT_W - 70 * mm])
    mcb_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
    story += [mcb_t, Spacer(1, 5 * mm)]

    # ── §6 Proiezione capitale futuro ────────────────────────────────────────
    story.append(Paragraph("6. Proiezione Capitale Futuro", st_section))
    if capital_projection:
        _keys = [k for k in ('p10', 'p50', 'p90') if k in capital_projection]
        if _keys:
            _any = capital_projection[_keys[0]]
            _years = list(_any.index)
            _milestones = [y for y in (1, 3, 5, 10) if y in _years]
            if not _milestones:
                _milestones = _years[-3:] if len(_years) >= 3 else _years
            hdr = [Paragraph('Percentile', st_cell_hdr)] + \
                  [Paragraph(f"Anno {y}", st_cell_hdrc) for y in _milestones]
            proj_rows = [hdr]
            _label = {'p10': 'P10 (pessimistico)', 'p50': 'P50 (mediano)',
                      'p90': 'P90 (ottimistico)'}
            for k in _keys:
                ser = capital_projection[k]
                proj_rows.append(
                    [_label.get(k, k)] +
                    [f"€{float(ser.loc[y]):,.0f}" if y in ser.index else 'N/A'
                     for y in _milestones])
            _cwp = (CONTENT_W - 50 * mm) / max(1, len(_milestones))
            proj_t = Table(proj_rows, colWidths=[50 * mm] + [_cwp] * len(_milestones))
            proj_t.setStyle(TableStyle(_ts_base() + [('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
            story.append(Paragraph(
                f"Capitale iniziale €{initial_capital:,.0f}. Proiezione basata sulla "
                f"distribuzione CAGR simulata (Monte Carlo Block Bootstrap).", st_body))
            story += [proj_t, Spacer(1, 5 * mm)]
        else:
            story.append(Paragraph("Proiezione capitale non disponibile.", st_body))
    else:
        story.append(Paragraph("Proiezione capitale non disponibile.", st_body))

    # ── §7 Decisione finale ──────────────────────────────────────────────────
    story.append(Paragraph("7. Decisione Finale", st_section))
    _promosso = str(verdetto).upper().strip() == 'PROMOSSO'
    verd_t = Table(
        [[_p_verd] for _p_verd in [Paragraph(
            f"VERDETTO: {str(verdetto).upper()}", st_verd)]],
        colWidths=[CONTENT_W])
    verd_t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_GREEN if _promosso else C_RED),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story += [verd_t, Spacer(1, 4 * mm)]

    _vtext = (
        f"Il portafoglio <b>{portfolio_title}</b> ha ottenuto verdetto "
        f"<b>{str(verdetto).upper()}</b> sulla base dei criteri di validazione "
        f"(MC Block A p50 Sharpe, MC Block B skill, DSR). "
        f"CAGR {_metric(pf_proposed, 'cagr')}, Sharpe {_metric(pf_proposed, 'sharpe')}, "
        f"MaxDD {_metric(pf_proposed, 'maxdd')}; "
        f"P(rolling {_th}y &lt; 0%) {_pct(_ploss) if _ploss is not None else 'N/A'}; "
        f"DSR {_num(dsr, 3)}."
    )
    vbox = Table([[Paragraph(_vtext, st_vbox)]], colWidths=[CONTENT_W])
    vbox.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), rl_colors.HexColor('#EAF0FB')),
        ('BOX',           (0, 0), (-1, -1), 1.2, C_NAVY_LT),
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
        title=f"Relazione Tecnica Lazy {portfolio_title}",
        author="TSlab",
    )
    doc.build(story, onFirstPage=_draw_hf, onLaterPages=_draw_hf)
    return output_path


def run_lazy_portfolio_analysis(
    registry: dict,
    ptf_names: list,
    output_dir: str,
    start_date: str = '2016-01-01',
    end_date=None,
    benchmark: str = 'SPY',
    init_cash: float = 100_000.0,
    fees: float = 0.001,
    years: int = 10,
    n_simulations_mc_a: int = 1000,
    n_simulations_mc_b: int = 500,
    override: bool = False,
    verbose: bool = False,
    min_years: int = 5,
    generate_pdf: bool = False,
) -> dict:
    """
    Entry point unico per iq l-analyze: esegue la pipeline batch Lazy e,
    se generate_pdf=True, genera la Relazione Investitore PDF per ogni PTF
    promosso.

    Returns
    -------
    dict con chiavi:
        'df'        : pd.DataFrame di classificazione (output di run_lazy_batch_analysis)
        'pdf_paths' : dict {ptf_name: Path | None} — presente solo se generate_pdf=True,
                      None per i PTF saltati o con errori
    """
    import os
    import contextlib

    details = {} if generate_pdf else None

    with contextlib.ExitStack() as _silence:
        if not verbose:
            _devnull = _silence.enter_context(open(os.devnull, "w"))
            _silence.enter_context(contextlib.redirect_stdout(_devnull))
            _silence.enter_context(contextlib.redirect_stderr(_devnull))
        df = run_lazy_batch_analysis(
            registry=registry,
            ptf_names=ptf_names,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            init_cash=init_cash,
            fees=fees,
            years=years,
            n_simulations_mc_a=n_simulations_mc_a,
            n_simulations_mc_b=n_simulations_mc_b,
            override=override,
            verbose=verbose,
            details_out=details,
            min_years=min_years,
        )

    result = {'df': df, 'pdf_paths': {}}

    if generate_pdf and details:
        from r_functions import generate_relazione_investitore_report
        os.makedirs(output_dir, exist_ok=True)

        for ptf_name, rich in details.items():
            verdetto = rich.get("verdetto", "RIGETTATO")
            if verdetto != "PROMOSSO":
                print(f"[iq l-analyze] {ptf_name}: RIGETTATO — relazione investitore non generata.")
                result['pdf_paths'][ptf_name] = None
                continue
            try:
                portfolio_cfg = rich.get("portfolio_cfg") or registry.get(ptf_name, {})
                # Supporta sia formato flat {ticker: peso} sia annidato
                # {"Title": ..., "tickers": {ticker: peso}, "benchmark": ...}
                _nested = portfolio_cfg.get("tickers") if isinstance(portfolio_cfg.get("tickers"), dict) else None
                _flat_tickers = _nested if _nested is not None else portfolio_cfg
                _title = portfolio_cfg.get("Title") or ptf_name
                _bm = portfolio_cfg.get("benchmark") or portfolio_cfg.get("benchmark_title") or benchmark
                portfolio_for_report = {
                    "Title": _title,
                    "benchmark": _bm,
                    "tickers": _flat_tickers,
                }
                reports_dir = os.path.join(output_dir, ptf_name)
                os.makedirs(reports_dir, exist_ok=True)

                # Sommario strutturato (Importo investito, Valore finale
                # netto, CAGR, Max Drawdown, Deviazione standard, Sharpe,
                # ...) — dati già calcolati in rich["out"]["sintesi_df"] per
                # il PDF, mai persistiti prima. Scritto PRIMA del PDF e in
                # un try/except separato: un problema di serializzazione
                # non deve mai impedire la generazione del PDF stesso.
                try:
                    import json
                    sintesi_df = rich["out"].get("sintesi_df") if rich.get("out") else None
                    if sintesi_df is not None and not sintesi_df.empty:
                        summary_dict = sintesi_df.iloc[0].to_dict()
                        bm_meta = rich["out"].get("bm_meta") or {}
                        summary_dict["Benchmark"] = bm_meta.get("benchmark_name")
                        stats_json_path = os.path.join(reports_dir, "stats_summary.json")
                        with open(stats_json_path, "w", encoding="utf-8") as _f:
                            json.dump(summary_dict, _f, ensure_ascii=False, indent=2)
                except Exception as _exc:
                    print(f"[WARN] stats_summary.json '{ptf_name}' non scritto: {_exc}")

                # Blocco STATISTICHE completo (Alpha/Beta, tracking error,
                # underwater days, ecc. — Immagine 2 del 23/07) — stesso
                # dict rich["out"], chiave "stats_df" invece di "sintesi_df".
                # Try/except separato dal precedente: un problema qui non
                # deve compromettere stats_summary.json ne' il PDF.
                try:
                    import json
                    stats_df = rich["out"].get("stats_df") if rich.get("out") else None
                    if stats_df is not None and not stats_df.empty:
                        value_col = "Valore" if "Valore" in stats_df.columns else stats_df.columns[0]
                        stats_full_dict = stats_df[value_col].to_dict()
                        stats_full_path = os.path.join(reports_dir, "stats_full.json")
                        with open(stats_full_path, "w", encoding="utf-8") as _f:
                            json.dump(stats_full_dict, _f, ensure_ascii=False, indent=2)
                except Exception as _exc:
                    print(f"[WARN] stats_full.json '{ptf_name}' non scritto: {_exc}")

                pdf_result = generate_relazione_investitore_report(
                    out=rich["out"],
                    portfolio=portfolio_for_report,
                    reports_dir=reports_dir,
                    year=None,
                )
                result['pdf_paths'][ptf_name] = (pdf_result or {}).get("pdf_path")
            except Exception as exc:
                print(f"[WARN] Relazione investitore '{ptf_name}' fallita: {exc}")
                result['pdf_paths'][ptf_name] = None

    return result

