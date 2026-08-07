"""
l_portfolios.py — Lazy Portfolio Definitions
Refactored from notebooks/libs/l_portfolios.ipynb

Formato unico (annidato):
    {"Title": str, "tickers": {ticker_o_isin: peso}, "benchmark": str}

Il parser del registry mantiene il supporto al vecchio formato flat
({ticker: peso}) per retro-compatibilita', ma nessun PTF di questo file
lo usa piu'.

Convenzione benchmark (07/08/2026):
  - coerenza di valuta/mercato: PTF EUR/Borsa Italiana -> benchmark
    quotato a Milano; PTF in USD/US-listed -> benchmark USD.
  - PTF multi-asset EUR -> scala Vanguard LifeStrategy per quota
    azionaria (V20A.MI / V40A.MI / V60A.MI / V80A.MI), arrotondando
    alla quota piu' vicina; in caso di equidistanza si sceglie il
    gradino AZIONARIO SUPERIORE (benchmark piu' difficile da battere).
  - PTF 100% azionario EUR -> VWCE.MI (FTSE All-World).
  - PTF obbligazionari/liquidita' -> VAGF.MI (invariato).
"""

# ═══════════════════════════════════════
# SANDBOX — esperimenti, non PTF reali
# ═══════════════════════════════════════
sandbox_aapl_msft_tsla = {
    "Title": "Sandbox AAPL/MSFT/TSLA",
    "tickers": {
        'AAPL': 0.50,
        'MSFT': 0.30,
        'TSLA': 0.20,
    },
    "benchmark": "QQQ"  # mega-cap tech USA: Nasdaq-100 piu' aderente di SPY
}

sandbox_xx = {
    "Title": "Test fondi",
    "tickers": {
        'LU0034353002': 1.00, # DWS Floating Rate Notes LC https://funds.dws.com/it-it/comparti-fondi-obbligazionari/lu0034353002-dws-floating-rate-notes-lc/
        # 'IE00BYXYYM63': 0.01  # IE00BYXYYM63  iShares US Aggregate Bond UCITS ETF (Acc)
    },
    "benchmark": "VAGF.MI"
}

sandbox_dws_lc = {
    "Title": "DWS Floating Rate Notes LC",
    "tickers": {
        'LU0034353002': 1.00, # DWS Floating Rate Notes LC https://funds.dws.com/it-it/comparti-fondi-obbligazionari/lu0034353002-dws-floating-rate-notes-lc/
    },
    "benchmark": "VAGF.MI"
}

sandbox_multi_fondo  = {
    "Title": "Portafoglio liquidita' Multi-Fondo",
    "tickers": {
        "IE00BH04FZ00": 0.10,  # Vanguard EUR Corporate 1-3 Year Bond UCITS ETF
        "LU0034353002": 0.20,  # DWS Floating Rate Notes LC
        "LU1190417599": 0.20,  # Amundi Smart Overnight Return UCITS ETF
        "IE00BK5BQV03": 0.20,  # Vanguard FTSE Developed World UCITS ETF
        "LU2963696674": 0.30,  # DWS Invest StepIn Akkumula LC
    },
    "benchmark": "VAGF.MI"  # 
}

# ───────────────────────────────────────
# Scala di rischio a 5 gradini (fondi indice EUR)
# Universo comune ai 5 profili:
#   IE00B5456744  FTSE Developed All Cap Choice   (azionario sviluppati)
#   IE00BKV0W243  FTSE Emerging All Cap Choice    (azionario emergenti)
#   IE00BFPM9W02  Bloomberg Euro Government Float Adjusted Bond
#   IE00BJN4RG66  Bloomberg Global Aggregate Corporate
#   LU2531807738  Solactive Obbl. governative Eurozona 0-1 anno
# Quota azionaria: 100% / 70% / 50% / 30% / 15%
# ───────────────────────────────────────

sandbox_crescita = {
    "Title": "Portafoglio Crescita",
    "tickers": {
        "IE00B5456744": 0.80,  # FTSE Developed All Cap Choice
        "IE00BKV0W243": 0.20,  # FTSE Emerging All Cap Choice
    },
    "benchmark": "VWCE.MI"  # 100% azionario globale -> FTSE All-World
}

sandbox_energetico = {
    "Title": "Portafoglio Energetico",
    "tickers": {
        "IE00BFPM9W02": 0.18,  # Bloomberg Euro Government Float Adjusted Bond
        "IE00BJN4RG66": 0.12,  # Bloomberg Global Aggregate Corporate
        "IE00B5456744": 0.56,  # FTSE Developed All Cap Choice
        "IE00BKV0W243": 0.14,  # FTSE Emerging All Cap Choice
    },
    "benchmark": "V80A.MI"  # 70% equity: equidistante 60/80, tie-break verso l'alto
}

sandbox_liscio = {
    "Title": "Portafoglio Liscio",
    "tickers": {
        "IE00BFPM9W02": 0.30,  # Bloomberg Euro Government Float Adjusted Bond
        "IE00BJN4RG66": 0.20,  # Bloomberg Global Aggregate Corporate
        "IE00B5456744": 0.40,  # FTSE Developed All Cap Choice
        "IE00BKV0W243": 0.10,  # FTSE Emerging All Cap Choice
    },
    "benchmark": "V60A.MI"  # 50% equity: equidistante 40/60, tie-break verso l'alto
}

sandbox_calma = {
    "Title": "Portafoglio Calma",
    "tickers": {
        "IE00BFPM9W02": 0.325,  # Bloomberg Euro Government Float Adjusted Bond
        "LU2531807738": 0.095,  # Solactive Obbl. governative Eurozona 0-1 anno
        "IE00BJN4RG66": 0.280,  # Bloomberg Global Aggregate Corporate
        "IE00B5456744": 0.240,  # FTSE Developed All Cap Choice
        "IE00BKV0W243": 0.060,  # FTSE Emerging All Cap Choice
    },
    "benchmark": "V40A.MI"  # 30% equity: gradino piu' vicino (40) verso l'alto
}

sandbox_protezione = {
    "Title": "Portafoglio Protezione",
    "tickers": {
        "IE00BFPM9W02": 0.35,  # Bloomberg Euro Government Float Adjusted Bond
        "LU2531807738": 0.16,  # Solactive Obbl. governative Eurozona 0-1 anno
        "IE00BJN4RG66": 0.34,  # Bloomberg Global Aggregate Corporate
        "IE00B5456744": 0.12,  # FTSE Developed All Cap Choice
        "IE00BKV0W243": 0.03,  # FTSE Emerging All Cap Choice
    },
    "benchmark": "V20A.MI"  # 15% equity -> gradino 20% e' il piu' vicino
}


# ═══════════════════════════════════════
# EQUITY — portafogli azionari concentrati/tematici
# ═══════════════════════════════════════
equity_robohuman = {
    "Title": "Robo & Human",
    "tickers": {
        '005380.KS': 0.35,  # Hyundai Motor
        '9984.T':    0.30,  # SoftBank Group
        '6954.T':    0.20,  # Fanuc
        'RBOT.MI':   0.15,  # iShares Automation & Robotics UCITS ETF
    },
    "benchmark": "VWCE.MI"  # azionario globale: RBOT.MI e' una posizione, non un metro
}


# ═══════════════════════════════════════
# LAZY — allocazioni multi-asset class (oggetto del framework)
# ═══════════════════════════════════════
lazy_etf_port = {
    "Title": "Lazy ETF Port",
    "tickers": {
        "SPY": 0.40,
        "VT":  0.27,
        "IVV": 0.09,
        "XLP": 0.06,
        "VTV": 0.03,
        "AGG": 0.12,
        "GLD": 0.03,
    },
    "benchmark": "SPY"  # 85% equity a forte tilt USA, tutto US-listed
}

lazy_no_overlap_ief = {
    "Title": "Lazy No Overlap (IEF)",
    "tickers": {
        "SPY":  0.40,
        "VXUS": 0.22,
        "USMV": 0.12,
        "VTV":  0.06,
        "IEF":  0.15,
        "GLD":  0.05,
    },
    "benchmark": "SPY"
}

lazy_no_overlap_shy = {
    "Title": "Lazy No Overlap (SHY)",
    "tickers": {
        "SPY":  0.40,
        "VXUS": 0.22,
        "USMV": 0.12,
        "VTV":  0.06,
        "SHY":  0.15,
        "GLD":  0.05,
    },
    "benchmark": "SPY"
}

lazy_greta_base_spy = {
    "Title": "Greta Base (SPY)",
    "tickers": {
        'SPY':     0.60,
        'GLD':     0.20,
        'LYXC.DE': 0.10,
        'MTD.PA':  0.10,
    },
    "benchmark": "SPY"
}

lazy_greta_base_etf_ita = {
    "Title": "Greta Base ETF Italia",
    "tickers": {
        "VUAA.MI":   0.60,
        "SGLD.MI":   0.20,
        "EM710.MI":  0.10,
        "EM1015.MI": 0.10,
    },
    "benchmark": "V60A.MI"  # 60% equity, PTF interamente EUR/Milano
}

lazy_greta_alt_a = {
    "Title": "Greta Alt A",
    "tickers": {
        "SPY":     0.38,
        "VXUS":    0.17,
        "QQQ":     0.10,
        "GLD":     0.12,
        "LYXC.DE": 0.11,
        "MTD.PA":  0.12,
    },
    "benchmark": "SPY"
}

lazy_greta_alt_b = {
    "Title": "Greta Alt B",
    "tickers": {
        "SPY":     0.40,
        "USMV":    0.15,
        "QQQ":     0.10,
        "GLD":     0.12,
        "LYXC.DE": 0.11,
        "MTD.PA":  0.12,
    },
    "benchmark": "SPY"
}

lazy_greta_alt_c = {
    "Title": "Greta Alt C",
    "tickers": {
        "SPY":     0.40,
        "QQQ":     0.10,
        "GLD":     0.12,
        "LYXC.DE": 0.10,
        "MTD.PA":  0.08,
        "SHY":     0.20,
    },
    "benchmark": "SPY"
}

lazy_greta_alt_emdiv = {
    "Title": "Greta Alt EM Div",
    "tickers": {
        "VUAA.MI":   0.60,
        "EIMI.MI":   0.15,
        "SGLD.MI":   0.15,
        "EM710.MI":  0.05,
        "EM1015.MI": 0.05,
    },
    "benchmark": "V80A.MI"  # 75% equity, PTF EUR/Milano
}

lazy_greta_alt_emdiv_test = {
    "Title": "Greta Alt EM Div (proxy USD)",
    "tickers": {
        "SPY": 0.60,
        "EEM": 0.15,
        "GLD": 0.15,
        "IEF": 0.05,
        "TLH": 0.05,
    },
    "benchmark": "SPY"
}

lazy_conservative_40_30_30 = {
    "Title": "Lazy Conservative 40/30/30",
    "tickers": {
        "VUAA.MI":   0.40,  # azionario USA
        "SGLD.MI":   0.20,  # oro
        "EM710.MI":  0.20,  # govt bond 7-10y
        "EM1015.MI": 0.20,  # govt bond 10-15y
    },
    "benchmark": "V40A.MI"
}

# Composizione identica a lazy_greta_base_etf_ita, ma entry autonoma nel
# registry (era un alias allo stesso oggetto: ora ha Title proprio).
lazy_balanced_60_20_20 = {
    "Title": "Lazy Balanced 60/20/20",
    "tickers": {
        "VUAA.MI":   0.60,
        "SGLD.MI":   0.20,
        "EM710.MI":  0.10,
        "EM1015.MI": 0.10,
    },
    "benchmark": "V60A.MI"
}

lazy_aggressive_80_10_10 = {
    "Title": "Lazy Aggressive 80/10/10",
    "tickers": {
        "VUAA.MI":   0.80,
        "SGLD.MI":   0.10,
        "EM710.MI":  0.05,
        "EM1015.MI": 0.05,
    },
    "benchmark": "V80A.MI"
}

lazy_full_equity_95_5 = {
    "Title": "Lazy Full Equity 95/5",
    "tickers": {
        "VUAA.MI": 0.95,
        "SGLD.MI": 0.05,
    },
    "benchmark": "VWCE.MI"  # quasi interamente azionario -> FTSE All-World
}

# L_PORTFOLIO_REGISTRY: costruito automaticamente da tutte le variabili
# dict definite sopra che rispettano uno dei due formati L:
#   - Vecchio (flat):    {ticker: peso, ...}  con somma pesi ~1.0
#   - Nuovo (annidato):  {"Title": str, "tickers": {ticker: peso}, "benchmark": str}
#                        con somma pesi di "tickers" ~1.0
# Qualsiasi nuovo PTF aggiunto a questo file è immediatamente
# disponibile via 'iq l-analyze --ptf <nome_variabile>'.
# Questo registry serve esclusivamente al workflow CLI/JN
# dell'architetto - i PTF dei gestori bancari (webapp futura)
# useranno un meccanismo runtime separato, non questo file.

L_PORTFOLIO_REGISTRY = {}
for _name, _obj in list(globals().items()):
    if _name.startswith('_') or _name == 'L_PORTFOLIO_REGISTRY':
        continue
    if isinstance(_obj, dict) and len(_obj) > 0:
        # Vecchio formato flat: {ticker: peso}
        _vals = list(_obj.values())
        if all(isinstance(v, (int, float)) for v in _vals):
            _total = sum(_vals)
            if 0.95 <= _total <= 1.05:
                L_PORTFOLIO_REGISTRY[_name] = _obj
                continue
        # Nuovo formato annidato: {"Title": ..., "tickers": {ticker: peso}, ...}
        _tickers = _obj.get('tickers')
        if isinstance(_tickers, dict) and len(_tickers) > 0:
            _t_vals = list(_tickers.values())
            if all(isinstance(v, (int, float)) for v in _t_vals):
                _t_total = sum(_t_vals)
                if 0.95 <= _t_total <= 1.05:
                    L_PORTFOLIO_REGISTRY[_name] = _obj
del _name, _obj

# Sotto-registry per categoria (basati sul prefisso del nome)
L_PORTFOLIO_LAZY    = {k: v for k, v in L_PORTFOLIO_REGISTRY.items() if k.startswith('lazy_')}
L_PORTFOLIO_EQUITY  = {k: v for k, v in L_PORTFOLIO_REGISTRY.items() if k.startswith('equity_')}
L_PORTFOLIO_SANDBOX = {k: v for k, v in L_PORTFOLIO_REGISTRY.items() if k.startswith('sandbox_')}
