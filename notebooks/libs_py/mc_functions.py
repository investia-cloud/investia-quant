"""
mc_functions.py — Refactored from notebooks/libs/mc_functions.ipynb
"""

from pypfopt import EfficientFrontier, risk_models, expected_returns
import vectorbt as vbt
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Optional, Dict, List, Union, Any
import plotly.graph_objects as go
from u_functions import (build_and_plot_portfolio_contributions, download_data, generate_lazy_portfolio_performance, plot_cumulative_and_rolling_returns, plot_monthly_returns, plot_multiple_portfolios)

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
    price_data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
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
    
def run_bh_backtest(
    weights_dict: dict,
    start_date: str,
    end_date: str,
    init_cash: float = 10_000,
    fees: float = 0.001,
    rebalance_freq: str = None
) -> vbt.Portfolio:
    """
    Backtest Buy & Hold robusto con VectorBT, solo su dati completamente validi.

    - I dati sono pre-allineati per evitare NaN.
    - Il ribilanciamento avviene solo in date con dati completi.
    - Perfettamente confrontabile con Pandas.
    """

    # Set Uppercase
    for k in list(weights_dict.keys()):
        weights_dict[k.upper()] = weights_dict.pop(k)

    # 1. Validazione pesi
    tickers = list(weights_dict.keys())

    weights = pd.Series(weights_dict, index=tickers, dtype=float)
    
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("La somma dei pesi deve essere 1.")

    # 2. Scarica i dati da yfinance
    # data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    # # display(data)
    # # price = data["Close"][tickers]
    # price = data["Close"]
    # price.columns.name = None
    
    price=download_data(tickers, start_date=start_date, end_date=end_date)
    
    # 3. Allinea: elimina ogni giorno con dati mancanti
    price = price.dropna(how='any')

    # display(price.head(),price.tail())
    
    if price.empty:
        raise ValueError("Nessuna data con dati completi per tutti gli asset.")

    # 4. Costruzione size: DataFrame con target percent
    size = pd.DataFrame(np.nan, index=price.index, columns=price.columns)

    # 5. Date di ribilanciamento
    if rebalance_freq is None:
        reb_dates = pd.DatetimeIndex([price.index[0]])
    else:
        rf = str(rebalance_freq).upper()
    
        if rf in ["Y", "A", "YE"]:
            # ultimo trading day di ogni anno (robusto)
            reb_dates = price.groupby(price.index.year).apply(lambda x: x.index[-1])
            reb_dates = pd.DatetimeIndex(reb_dates.values)
        else:
            periods = price.index.to_period(rebalance_freq)
            reb_dates = price.index[~periods.duplicated()]
    
        # assicura inclusione start
        if price.index[0] not in reb_dates:
            reb_dates = reb_dates.insert(0, price.index[0])
    
    # IMPORTANTISSIMO: garantisci che tutte le reb_dates siano nel calendario prezzi
    reb_dates = reb_dates.intersection(price.index)  

    # if rebalance_freq is None:
    #     reb_dates = [price.index[0]]
    # else:
    #     periods = price.index.to_period(rebalance_freq)
    #     reb_dates = price.index[~periods.duplicated()]
    #     if price.index[0] not in reb_dates:
    #         reb_dates = reb_dates.insert(0, price.index[0])

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
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
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
    my_weights=None,
    fig_width: int = 1200,
    fig_height: int = 600,
    compute_real_annual_return: bool = True,
    start_date=None,
    end_date=None,
) -> dict:

    # Calcola date di inizio e fine
    if end_date is None:
        end_date = datetime.today()
    else:
        end_date = pd.to_datetime(end_date).to_pydatetime()
    if start_date is None:
        start_date = datetime(end_date.year - years, 1, 1)
    else:
        start_date = pd.to_datetime(start_date).to_pydatetime()

    price = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna(how='any')
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

    price = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna(how='any')
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
    run_as_app: bool = False
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
        init_cash=init_cash, fees=fees, rebalance_freq=rebalance_freq
    )

    # header = "Titolo" if one_ticker else "Portfolio"
    # if not run_as_app:
    #     print(f"🔎 Analisi {header} «{title}»")
        
    benchmark_data = download_data(benchmark,start_date,end_date)

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
            my_weights=portfolio,
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
    run_as_app: bool = False
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
        init_cash=init_cash, fees=fees, rebalance_freq=rebalance_freq
    )

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
            my_weights=my_weights,
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
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
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
    """
    _metric_cols = {'Return', 'Volatility', 'Sharpe',
                    'Real Return', 'Real Volatility', 'Real Sharpe'}

    window_years = max(1, years // n_splits)
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
    end_year_global = datetime.today().year
    start_year_global = end_year_global - years

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
) -> dict:
    """
    MC Block B — Skill test sul ribilanciamento.
    Testa se la scelta della frequenza di ribilanciamento aggiunge
    valore vs date di ribilanciamento randomizzate (jitter ±jitter_days).

    Se best_freq is None (BH puro): confronta vs ribilanciamento annuale
    randomizzato (verifica che BH non sia inferiore a qualsiasi rebalancing).

    Returns dict:
        'actual_sharpe':  float — Sharpe del PTF con best_freq
        'actual_cagr':    float — CAGR del PTF con best_freq
        'sim_sharpes':    np.ndarray — distribuzione Sharpe simulazioni
        'sim_cagrs':      np.ndarray — distribuzione CAGR simulazioni
        'p_value_sharpe': float — prob(sim_sharpe >= actual_sharpe)
        'p_value_cagr':   float — prob(sim_cagr >= actual_cagr)
        'skill':          bool — True se p_value_sharpe < 0.05
        'n_simulations':  int
    """
    # NOTA: download_data non definita in mc_functions.py — fallback su yf.download
    portfolio = {k.upper(): v for k, v in portfolio.items()}
    tickers = list(portfolio.keys())
    weights = pd.Series(portfolio, dtype=float)

    # 1. Metriche PTF reale
    pf_actual = run_bh_backtest(portfolio, start_date, end_date,
                                init_cash, fees, best_freq)
    actual_sharpe = float(pf_actual.sharpe_ratio())
    actual_cagr   = _cagr_from_equity(pf_actual)

    # 2. Scarica prezzi una sola volta
    price = yf.download(tickers, start=start_date, end=end_date,
                        progress=False, multi_level_index=False)
    if 'Close' in price.columns.get_level_values(0) if isinstance(price.columns, pd.MultiIndex) else []:
        price = price['Close']
    elif 'Close' in price.columns:
        price = price[['Close'] if len(tickers) == 1 else tickers]
    # Normalizza: se single-ticker yf restituisce Series, converti a DataFrame
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

    # ── 1. FREQUENCY SELECTION ───────────────────────────────────────────────
    if auto_freq:
        freqs = ['W', 'M', 'Q', 'Y', None]
        rows = []
        for freq in freqs:
            pf_f = run_bh_backtest(portfolio_cfg, _start_date, end_date,
                                   init_cash, fees, freq)
            rows.append({
                'Freq':        freq if freq is not None else 'BH',
                'Sharpe':      _safe_metric(pf_f, 'sharpe'),
                'CAGR':        _cagr_from_equity(pf_f),
                'TotalReturn': _safe_metric(pf_f, 'total_return'),
                'MaxDD':       abs(_safe_metric(pf_f, 'max_drawdown')),
            })
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
                                   init_cash, fees, best_freq)

    # ── 3. FRONTIERA EFFICIENTE ───────────────────────────────────────────────
    frontier_fig, df_special = efficient_frontier_pypfopt(
        tickers=tickers,
        years=_years,
        show_plot=plot,
        interactive=plot,
        print_weights=verbose,
        my_weights=my_weights,
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
    )

    return {
        'best_freq':       best_freq,
        'freq_df':         freq_df,
        'portfolio_pf':    portfolio_pf,
        'optimal_weights': optimal_weights,
        'frontier_result': frontier_result,
        'plots_dir':       plots_dir,
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
    verbose=False,
) -> "pd.DataFrame":
    """
    Esegue la pipeline Lazy completa (run_lazy_analysis + stability +
    MC A1/A2 + MC B + DSR + verdetto) su una lista di PTF dal registry.
    Salva un CSV 'classification_<timestamp>.csv' in output_dir.
    Ritorna il DataFrame classification.
    """
    from pathlib import Path
    from datetime import datetime as _dt
    # mc_run_iid_bootstrap / mc_run_block_bootstrap / ofc_compute_dsr sono
    # definite in r_functions.py e NON sono importate a livello di modulo qui:
    # import locale per evitare di toccare le import globali di mc_functions.py.
    from r_functions import (
        mc_run_iid_bootstrap,
        mc_run_block_bootstrap,
        ofc_compute_dsr,
    )

    rows = []

    for ptf_name in ptf_names:
        # 1. risoluzione dal registry
        portfolio_cfg = registry.get(ptf_name)
        if portfolio_cfg is None:
            print(f"[WARN] PTF '{ptf_name}' non trovato nel registry: skip.")
            continue

        try:
            # 2. analisi lazy headless + backtest B&H con best_freq
            sub_output_dir = Path(output_dir) / ptf_name
            risultati = run_lazy_analysis(
                portfolio_cfg=portfolio_cfg,
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
            )
            pf_proposed = run_bh_backtest(portfolio_cfg, start_date, end_date,
                                          init_cash, fees, risultati['best_freq'])

            # 3. stabilità dei pesi
            tickers = list(portfolio_cfg.keys())
            stability = lazy_stability_weights(
                tickers=tickers, years=years, weight_bounds=(0, 1),
                n_splits=5, verbose=False,
            )

            # 4. MC A1 (iid bootstrap) e A2 (block bootstrap)
            rng_a1 = np.random.default_rng(42)
            mc_a1 = mc_run_iid_bootstrap(pf_proposed, n_simulations_mc_a, rng_a1)
            rng_a2 = np.random.default_rng(42)
            mc_a2 = mc_run_block_bootstrap(pf_proposed, 20, n_simulations_mc_a, rng_a2)

            # 5. MC B (rebalancing con jitter)
            mc_b = lazy_mc_block_b_rebalancing(
                portfolio=portfolio_cfg, start_date=start_date, end_date=end_date,
                best_freq=risultati['best_freq'], n_simulations=n_simulations_mc_b,
                jitter_days=30, init_cash=init_cash, fees=fees, verbose=False,
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
            rows.append({
                'Nome': ptf_name,
                'BestFreq': risultati['best_freq'] or 'BH',
                'CAGR%': round(cagr * 100, 2),
                'Sharpe': round(sr, 3),
                'MaxDD%': round(maxdd * 100, 2),
                'StabilityCV': round(stability['cv_mean'], 3) if stability['cv_mean'] is not None else np.nan,
                'StabilityOK': stability['stable'],
                'MC_A2_Sharpe_p50': round(mc_a2['percentiles']['p50']['Sharpe'], 3),
                'MC_B_pvalue': round(mc_b['p_value_sharpe'], 3),
                'MC_B_skill': mc_b['skill'],
                'DSR': round(dsr, 3),
                'CriteriPassati': f"{n_passed}/3",
                'Verdetto': verdetto,
            })
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
    mode: str = 'overlay',   # 'overlay' | 'separate'
) -> None:
    """
    Plotta le equity curve dei top N PTF promossi (per sort_by).
    mode='overlay': un solo grafico Plotly con tutte le curve
    normalizzate a base 100 (confrontabili indipendentemente dai pesi).
    mode='separate': un grafico per PTF (comportamento precedente).
    """
    import plotly.express as px
    import plotly.graph_objects as go

    top_df = promoted_df.sort_values(sort_by, ascending=False).head(top)

    curves = {}
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
        eq = pf.value()
        if isinstance(eq, pd.DataFrame):
            eq = eq.iloc[:, 0]
        eq_norm = eq / eq.iloc[0] * 100

        if mode == 'overlay':
            curves[nome] = eq_norm
        else:
            fig = px.line(eq_norm, title=f'{nome} — Equity Curve base 100 (freq={row["BestFreq"]})')
            fig.update_layout(height=350)
            fig.show()

    if mode == 'overlay':
        if not curves:
            print('Nessuna equity curve da plottare.')
            return
        fig = go.Figure()
        for nome, serie in curves.items():
            fig.add_trace(go.Scatter(x=serie.index, y=serie.values,
                                     mode='lines', name=nome))
        fig.update_layout(
            title=f'Top {top} PTF promossi — Equity Curve (base 100)',
            height=500,
        )
        fig.show()


def style_lazy_classification(df):
    """
    Applica gradiente colore a tutte le metriche numeriche della
    classification Lazy. Verde=migliore, rosso=peggiore per ogni
    colonna (Sharpe/CAGR/DSR alto=verde, MaxDD/StabilityCV/MC_B_pvalue
    alto=rosso - direzione invertita dove 'meno è meglio').
    Ritorna un pandas Styler.
    """
    cols_higher_better = ['CAGR%', 'Sharpe', 'MC_A2_Sharpe_p50', 'DSR']
    cols_lower_better  = ['MaxDD%', 'StabilityCV', 'MC_B_pvalue']

    styler = df.style
    for col in cols_higher_better:
        if col in df.columns:
            styler = styler.background_gradient(subset=[col], cmap='RdYlGn')
    for col in cols_lower_better:
        if col in df.columns:
            styler = styler.background_gradient(subset=[col], cmap='RdYlGn_r')
    styler = styler.format(precision=3)
    return styler

