"""
l_portfolios.py — Lazy Portfolio Definitions
Refactored from notebooks/libs/l_portfolios.ipynb
"""

my_portfolio_test = {
    'AAPL': 0.5,
    'MSFT': 0.3,
    'TSLA': 0.2
}

my_portfolio_XX = {
    'IE00B5BMR087': 0.75,
    'IE00BYXYYM63': 0.25
}

robohuman_portfolio = {
    '005380.KS': 0.35,
    '9984.T': 0.30,
    '6954.T': 0.20,
    'RBOT.MI': 0.15
}

ETF_PORT = {
    "SPY": 0.40,
    "VT":  0.27,
    "IVV": 0.09,
    "XLP": 0.06,
    "VTV": 0.03,
    "AGG": 0.12,
    "GLD": 0.03,
}

PTF_NO_OVERLAP_IEF = {
    "SPY":  0.40,
    "VXUS": 0.22,
    "USMV": 0.12,
    "VTV":  0.06,
    "IEF":  0.15,
    "GLD":  0.05,
}

PTF_NO_OVERLAP_SHY = {
    "SPY":  0.40,
    "VXUS": 0.22,
    "USMV": 0.12,
    "VTV":  0.06,
    "SHY":  0.15,
    "GLD":  0.05,
}

greta_base_spy_portfolio = {
    'SPY'  : 0.60,
    'GLD'  : 0.20,
    'LYXC.DE': 0.10,
    'MTD.PA' : 0.10
}

greta_base_spy_portfolio_etf_ita = {
    "VUAA.MI": 0.60,
    "SGLD.MI": 0.20,
    "EM710.MI": 0.10,
    "EM1015.MI": 0.10,
}

greta_alt_A = {
    "SPY": 0.38,
    "VXUS": 0.17,
    "QQQ": 0.10,
    "GLD": 0.12,
    "LYXC.DE": 0.11,
    "MTD.PA": 0.12,
}

greta_alt_B = {
    "SPY": 0.40,
    "USMV": 0.15,
    "QQQ": 0.10,
    "GLD": 0.12,
    "LYXC.DE": 0.11,
    "MTD.PA": 0.12,
}

greta_alt_C = {
    "SPY": 0.40,
    "QQQ": 0.10,
    "GLD": 0.12,
    "LYXC.DE": 0.10,
    "MTD.PA": 0.08,
    "SHY": 0.20,
}


greta_alt_emdiv = {
    "VUAA.MI": 0.60,
    "EIMI.MI": 0.15,
    "SGLD.MI": 0.15,
    "EM710.MI": 0.05,
    "EM1015.MI": 0.05,
}

greta_alt_emdiv_test = {
    "SPY": 0.60,
    "EEM": 0.15,
    "GLD": 0.15,
    "IEF": 0.05,
    "TLH": 0.05,
}


# L_PORTFOLIO_REGISTRY: costruito automaticamente da tutte le
# variabili dict {ticker: peso} definite sopra (somma pesi ~1.0).
# Qualsiasi nuovo PTF aggiunto a questo file è immediatamente
# disponibile via 'iq lazy-analyze --ptf <nome_variabile>'.
# Questo registry serve esclusivamente al workflow CLI/JN
# dell'architetto - i PTF dei gestori bancari (webapp futura)
# useranno un meccanismo runtime separato, non questo file.

L_PORTFOLIO_REGISTRY = {}
for _name, _obj in list(globals().items()):
    if _name.startswith('_') or _name == 'L_PORTFOLIO_REGISTRY':
        continue
    if isinstance(_obj, dict) and len(_obj) > 0:
        _vals = list(_obj.values())
        if all(isinstance(v, (int, float)) for v in _vals):
            _total = sum(_vals)
            if 0.95 <= _total <= 1.05:
                L_PORTFOLIO_REGISTRY[_name] = _obj
del _name, _obj