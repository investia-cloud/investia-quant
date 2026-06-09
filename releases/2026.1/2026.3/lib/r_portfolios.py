"""
r_portfolios.py — Rotational Portfolio Definitions
Refactored from notebooks/libs/r_portfolios.ipynb
Depends on: k_tickers (imported explicitly below)
"""
from k_tickers import (stocks_euro, settoriali, benchmark_settoriali, fattoriali, benchmark_fattoriali, multiasset_global_ucits, vanguard_etf, ai_dc_quantum_thematic, germany_plan_beneficiaries, italy_bigcap_tickers)

portfolio_alpha_euro = {
    "Title": "Alpha Euro",
    "tickers": stocks_euro,
    "benchmark_portfolio": None,
    "benchmark_title": "^STOXX50E"
}

portfolio_alpha_sect = {
    "Title": "Alpha Sect (Megatrend)",
    "tickers": settoriali,
    "benchmark_portfolio": benchmark_settoriali,
    "benchmark_title": "Indice sintetico settoriali"
}

portfolio_alpha_fact = {
    "Title": "Alpha Fact",
    "tickers": fattoriali,
    "benchmark_portfolio": benchmark_fattoriali,
    "benchmark_title": "Indice sintetico fattoriali"
}

portfolio_alpha_world = {
    "Title": "Alpha World",
    "tickers": multiasset_global_ucits,
    "benchmark_portfolio": None,
    "benchmark_title": "SWDA.MI"
}

portfolio_alpha_world_vanguard = {
    "Title": "Alpha World Vanguard",
    "tickers": vanguard_etf,
    "benchmark_portfolio": None,
    "benchmark_title": "V60A.DE"
}

portfolio_alpha_quant = {
    "Title": "Alpha Quant",
    "tickers": ai_dc_quantum_thematic,
    "benchmark_portfolio": None,
    "benchmark_title": "CSSPX.MI"
}

year = 2026

alpha_sp100_tickers_by_year = {
    2025: ['INTC', 'LLY', 'GOOGL', 'CAT', 'AVGO', 'AMD', 'IBM', 'GM', 'AMGN', 'NEE', 'JNJ', 'MDT', 'GILD', 'WMT', 'MS', 'CSCO', 'MMM', 'AXP', 'CVS', 'RTX', 'GS', 'ABBV', 'GE', 'C', 'KO'],
    2026: ['LLY', 'GM', 'AMD', 'GOOGL', 'CAT', 'AMGN', 'C', 'COF', 'CSCO', 'MS', 'INTC', 'WFC', 'AXP', 'JNJ', 'GILD', 'GS', 'AIG', 'RTX', 'WMT', 'IBM', 'BK', 'BAC', 'CVS', 'SCHW', 'AVGO']
}

portfolio_alpha_sp100 = {
    "Title": "Alpha SP100",
    "tickers": "sp100",
    "benchmark_portfolio": None,
    "benchmark_title": "CSSPX.MI"
}

alpha_nasdaq100_tickers_by_year = {
    2025: ['WBD', 'MU', 'AMAT', 'LRCX', 'INTC', 'GOOGL', 'ASML', 'BIIB', 'AVGO', 'KLAC', 'AMD', 'APP', 'ROST', 'AMGN', 'MNST', 'CRWD', 'EA', 'IDXX', 'AZN', 'CEG', 'GILD', 'XEL', 'SHOP', 'AEP', 'CSCO'],
    2026: ['MU', 'WDC', 'WBD', 'AMD', 'LRCX', 'GOOGL', 'AMAT', 'INSM', 'AZN', 'ROST', 'STX', 'AMGN', 'KLAC', 'MNST', 'CSCO', 'ADI', 'GILD', 'FER', 'INTC', 'SHOP', 'ASML', 'CEG', 'IDXX', 'AVGO', 'AEP']
}

portfolio_alpha_nasdaq100 = {
    "Title": "Alpha Nasdaq100",
    "tickers": "nasdaq100",
    "benchmark_portfolio": None,
    "benchmark_title": "CSNDX.MI"
}

portfolio_germany_plan = {
    "Title": "Germany Plan",
    "tickers": germany_plan_beneficiaries,
    "benchmark_portfolio": None,
    "benchmark_title": "^GDAXI"
}

portfolio_italy_big_cap = {
    "Title": "Italy Big Cap",
    "tickers": italy_bigcap_tickers,
    "benchmark_portfolio": None,
    "benchmark_title": "CSMIB.MI"
}

# Registry: mappa nome_stringa → oggetto portafoglio
# Usato dalla CLI per risolvere --ptf <nome>
R_PORTFOLIO_REGISTRY = {
    "alpha_euro":          portfolio_alpha_euro,
    "alpha_sect":          portfolio_alpha_sect,
    "alpha_fact":          portfolio_alpha_fact,
    "alpha_world":         portfolio_alpha_world,
    "alpha_world_vanguard": portfolio_alpha_world_vanguard,
    "alpha_quant":         portfolio_alpha_quant,
    "alpha_sp100":         portfolio_alpha_sp100,
    "alpha_nasdaq100":     portfolio_alpha_nasdaq100,
    "germany_plan":        portfolio_germany_plan,
    "italy_big_cap":       portfolio_italy_big_cap,
}
