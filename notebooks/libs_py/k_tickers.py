"""
k_tickers.py — Refactored from notebooks/libs/k_tickers.ipynb
"""

# =================================
# Tematici
# =================================

tematici = [
    # 🔬 Tecnologia & Innovazione
    "AIAI.MI",   # iShares Artificial Intelligence & Big Data
    "WTAI.MI",   # WisdomTree Artificial Intelligence
    "WCLD.MI",   # WisdomTree Cloud Computing (Quantum/NextGen Tech)
    "AIAI.L",    # L&G Artificial Intelligence
    "ISPY.MI",   # L&G Cyber Security
    "LOCK.MI",   # iShares Digital Security
    "RBOT.MI",   # iShares Automation & Robotics
    "ROBO.MI",   # L&G Robotics & Automation
    "QNTM.SW",    # VanEck Quantum Computing UCITS (quotato anche su Borsa Italiana)  

    # ⚡ Energia & Transizione
    "INRG.L",    # iShares Global Clean Energy (Xetra: ICLN.MI alternativa)
    "RENW.MI",   # L&G Clean Energy
    "STUX.MI",   # SPDR MSCI Europe Utilities
    # "INFR.L",    # iShares Global Infrastructure

    # 🌍 Ambiente & Sostenibilità
    "IH2O.MI",   # iShares Global Water
    "GLUG.MI",   # L&G Clean Water
    "ELLE.L",   # Lyxor Gender Equality
    "SUSW.MI",   # iShares MSCI World SRI

    # 🧬 Healthcare & Biotech
    "HEAL.MI",   # iShares Healthcare Innovation
    # "XDWH.MI",   # Xtrackers MSCI World Health Care
    # "BTEK.L",   # iShares Nasdaq US Biotechnology

    # 📡 Digital Economy & Consumer Trends
    "HERU.MI",   # Global X Video Games & Esports
    "ECOM.MI",   # L&G Ecommerce Logistics
    "DGTL.MI"    # iShares Digitalisation
]


# =================================
# Quantum · AI · Data Center · Enablers
# =================================
# =================================
# ⚛️ Quantum Computing · Pure Plays & Strategic Players
# =================================
quantum_pureplays = [
    "IONQ",   # ⚛️🧲 IonQ – quantum computing a ioni intrappolati (pure play)
    "RGTI",   # ❄️⚛️ Rigetti – qubit superconduttivi & QPU-as-a-Service (pure play)
    "QBTS",   # ❄️🧮 D-Wave – quantum annealing systems (pure play)
    "QUBT",   # 💻⚛️ Quantum Computing Inc. – software, algoritmi e servizi quantistici (pure play)
    "ARQQ",   # 🔐🛰️ Arqit Quantum – crittografia quantum-safe e key distribution (pure play)
    "HON",    # 🏭⚛️ Honeywell – conglomerato; controllo strategico di Quantinuum (non pure play)
]
# =================================
# 🏗️💻 AI · Data Center · Digital Infrastructure
# =================================
data_center_infrastructure = [
    # 🌍 Global Infrastructure / Smart City / Digitalisation / Clean Energy
    "INFR.L",   # 🌉🌍 iShares Global Infrastructure UCITS – infrastrutture globali
    "CITY.MI",  # 🏙️📡 iShares Smart City Infrastructure UCITS – urban tech & digital cities
    "DGTL.L",   # 🌐💾 iShares Digitalisation UCITS – cloud, data, digital economy
    "INRG.L",   # ⚡🌱 iShares Global Clean Energy UCITS – energia per data center & AI
    
    # 💻🧠 Semiconductors & Advanced Compute
    "QDVE.DE",  # 🧠💻 iShares S&P 500 Information Technology UCITS – big tech & AI platforms
    "SMH.MI",   # 🧪🔌 VanEck Semiconductor UCITS – semiconduttori avanzati
    
    # ☁️🏢 Data Center REITs & Digital Infra
    "VPN.L",    # 🏢☁️ Global X Data Center REITs & Digital Infrastructure UCITS
]

# =================================
# 🧠🔬 AI · Quantum · Semiconductor ENABLERS
# (aggiunta TSMC & ASML – nodo critico della catena del valore)
# =================================
ai_quantum_enablers = [
    "TSM",    # 🏭🧠 TSMC – fonderia leader mondiale (AI, HPC, quantum-ready nodes)
    "ASML",   # 🔬⚙️ ASML – litografia EUV; collo di bottiglia tecnologico globale
]


# =================================
# ⛏️⚡ Bitcoin Miners · AI / HPC Power Reuse
# =================================
bitcoin_miners_ai = [
    "RIOT",   # ⛏️⚡ Riot Platforms – mining + data center power strategy
    "MARA",   # ⛏️🧠 Marathon Digital – HPC-ready mining infrastructure
    "CLSK",   # ⚡⛏️ CleanSpark – energy-efficient mining
    "CIFR",   # 🏭⚡ Cipher Mining – industrial-scale mining & HPC optionality
    "IREN",   # 🌱⚡ Iris Energy – green energy + AI-ready data centers
    "WULF",   # ⚡⛏️ TeraWulf – zero-carbon mining & compute infra
]
# =================================
# 🧬🚀 Quantum · AI · Data Center · Thematic Portfolio
# =================================
ai_dc_quantum_thematic = (
    quantum_pureplays
    + data_center_infrastructure
    + ai_quantum_enablers
    + bitcoin_miners_ai
)
quantum_thematic = ai_dc_quantum_thematic

# =================================
# 🌍 Terre Rare — Universe (focus 2022→oggi)
# =================================

# 1) ETF UCITS – Rare Earths & Strategic Metals
#    Suggerimento: usa REMX.MI come base; gli altri sono listing alternativi (fallback).
rare_earths_ucits = [
    "REMX.MI",  # 🧲 VanEck Rare Earth & Strategic Metals UCITS – Borsa Italiana
    # "REMX.L",   # 🧲 VanEck Rare Earth & Strategic Metals UCITS – LSE (USD)
    # "REGB.L",   # 🧲 VanEck Rare Earth & Strategic Metals UCITS – LSE (GBP)
    # "REMX.PA",  # 🧲 VanEck Rare Earth & Strategic Metals UCITS – Euronext Paris
    # "VVMX.DE",  # 🧲 VanEck Rare Earth & Strategic Metals UCITS – Xetra
]

# 2) Azioni “core” (estrazione, raffinazione, magneti)
rare_earths_core = [
    "MP",         # 🏭 MP Materials – miniera/ossidi (USA, Mountain Pass)
    "LYC.AX",     # 🇦🇺 Lynas Rare Earths – mining AU + impianto Malesia
    "NEO.TO",     # 🧲 Neo Performance Materials – leghe/magneti (Canada)
    "ILU.AX",     # ⚙️  Iluka Resources – minerali pesanti + progetto REE
    # "600111.SS",  # 🇨🇳 China Northern Rare Earth – leader cinese (A-share)
    # "600392.SS",  # 🇨🇳 Shenghe Resources – supply chain globale (A-share)
    "RBW.L",      # 🌍 Rainbow Rare Earths – progetti Africa (NdPr)
    "PRE.L",      # 🌍 Pensana – progetto Longonjo (Angola) / LSE
]

# 3) Satelliti (attivali nei blocchi più recenti: storico più corto/volatilità alta)
rare_earths_satellite = [
    "ARU.AX",  # 🧭 Arafura Rare Earths – progetto Nolans (AU)
    "HAS.AX",  # 🧭 Hastings Technology Metals – Yangibana (AU)
    "VML.AX",  # 🧭 Vital Metals – estrazione/processing (CA/AU)
    "AR3.AX",  # 🧭 Australian Rare Earths – Koppamurra (AU)
    "NTU.AX",  # 🧭 Northern Minerals – Browns Range (heavy REE)
    "UCU.V",   # 🏭 Ucore Rare Metals – raffinazione (USA/Canada)
]

# (Opzionale) Proxy US per backtest profondo, poi mappa in live a UCITS
rare_earths_proxies_us = [
    "REMX",   # 🧲 VanEck Rare Earth/Strategic Metals (US-domiciled)
]

# 4) Lista unificata (ETF UCITS + Core + Satelliti + eventuali proxy US)
#    Nota: dict.fromkeys(...) preserva l'ordine e rimuove duplicati.
# rare_earths = list(dict.fromkeys(
#     rare_earths_ucits + rare_earths_core + rare_earths_satellite + rare_earths_proxies_us
# ))
rare_earths = rare_earths_ucits + rare_earths_core + rare_earths_satellite
terrerare = rare_earths

# ================rare_earths=================
# Settoriali
# =================================

settoriali = [
    # 🏭 ETF Settoriali UCITS – S&P US Select Sector (Borsa Italiana)
    "XLCS.MI",  # 📡 Comunicazioni (Communication Services)
    "XLYS.MI",  # 🛍️ Beni di Consumo Discrezionali (Consumer Discretionary)
    "XLPS.MI",  # 🧃 Beni di Prima Necessità (Consumer Staples)
    "XLFS.MI",  # 💰 Finanziario (Financials)
    "XLVS.MI",  # 🏥 Sanità (Healthcare)
    "XLIS.MI",  # 🛠️ Industria (Industrials)
    "XLBS.MI",  # 🧱 Materiali (Materials)
    "XLKS.MI",  # 💻 Tecnologia (Technology)
    "XLUS.MI",  # 💡 Utilities (Servizi pubblici)
]
benchmark_settoriali = {
    ticker: 1.0 / len(settoriali)
    for ticker in settoriali
}


# =================================
# ETF Fattoriali UCITS – Borsa Italiana
# =================================

fattoriali = [
    # 📉 Low Volatility
    "IWVL.MI",  # 🛡️ MSCI World Minimum Volatility – Azioni globali a bassa volatilità
    # "EUNL.MI",  # ⚖️ MSCI World ESG Screened Minimum Volatility (alternativa difensiva)

    # 💎 Quality
    "IWQU.MI",  # 🏆 MSCI World Quality Factor – Aziende con ROE elevato e bilanci solidi
    # "XDEQ.MI",  # 🧠 MSCI USA Quality – Qualità focalizzata su mercato USA

    # 📈 Momentum
    "IWMO.MI",  # 🚀 MSCI World Momentum – Titoli con trend di performance persistente
    # "XDEM.MI",  # 🔄 MSCI USA Momentum – Momentum puro su azioni USA

    # 💰 Value  — TODO: identify correct MSCI World Value ETF (IWVL.MI is Low Vol, not Value)
    "XDEV.MI",  # 🧾 MSCI USA Value – Fattore Value sul mercato statunitense

    # 🧩 Multifactor
    # "IWFM.MI",  # 🧠 MSCI World Multifactor – Combina Value, Momentum, Quality, Low Vol
    # "XDEW.MI",  # 🧬 MSCI USA Multifactor – Approccio fattoriale bilanciato USA

    # 📊 Size / Small Cap Factor
    "IUSN.DE",  # 🧱 MSCI World Small Cap – Esposizione strutturale al fattore Size
]
benchmark_fattoriali = {
    ticker: 1.0 / len(fattoriali)
    for ticker in settoriali
}

# =================================
# Area Euro
# =================================

stocks_euro = [
    # 🇮🇹 Italia (9 titoli)
    "ENEL.MI",    # Enel - energia/utilities
    "ISP.MI",     # Intesa Sanpaolo - banca
    "UCG.MI",     # UniCredit - banca
    "ENI.MI",     # ENI - oil & gas
    "STLAM.MI",   # Stellantis - automotive
    "PRY.MI",     # Prysmian - cavi/energia
    "G.MI",       # Generali - assicurazioni
    "LDO.MI",     # Leonardo - difesa/aerospazio
    "PST.MI",     # Poste Italiane - servizi finanziari/postali

    # 🇩🇪 Germania (9 titoli)
    "SAP.DE",     # SAP - software/enterprise AI
    "SIE.DE",     # Siemens - tecnologia industriale
    "IFX.DE",     # Infineon - semiconduttori
    "DTE.DE",     # Deutsche Telekom - telecomunicazioni
    "BAYN.DE",    # Bayer - pharma/chimica
    "ALV.DE",     # Allianz - assicurazioni (ticker corretto)
    "MRK.DE",     # Merck KGaA - chimica/life science
    "BAS.DE",     # BASF - chimica
    "RWE.DE",     # RWE - utilities energia verde

    # 🇫🇷 Francia (9 titoli)
    "MC.PA",      # LVMH - lusso (moda, vini, profumi)
    "OR.PA",      # L'Oréal - cosmetici
    "AIR.PA",     # Airbus - aerospazio
    "SAN.PA",     # Sanofi - pharma
    "BNP.PA",     # BNP Paribas - banca
    "AI.PA",      # Air Liquide - gas industriali
    "KER.PA",     # Kering - lusso (Gucci, Balenciaga)
    "HO.PA",      # Schneider Electric - tech energetico
    "ML.PA",      # Michelin - automotive/gomma
    
    # 🇪🇸 Spagna (9 titoli)
    "ITX.MC",     # Inditex - abbigliamento (Zara)
    "IBE.MC",     # Iberdrola - energia
    "SAN.MC",     # Banco Santander - banca
    "REP.MC",     # Repsol - energia fossile
    "TEF.MC",     # Telefónica - telecomunicazioni
    "ACS.MC",     # ACS - infrastrutture, costruzioni
    "ACX.MC",     # Acerinox - acciaio/industria
    "CLNX.MC",    # Cellnex Telecom - infrastrutture telco
    "GRF.MC"      # Grifols - biotecnologie / plasma
]

# === Risk OFF ===
risk_off_tickers = [
    "XEON.MI",   # cash remunerato — floor assoluto
    "IBTS.MI",   # treasury USA — sale in risk-off
    "XAD5.MI",   # oro — decorrelato
]

# =================================
# Area US
# =================================
stocks_usa = [
    # 💻 Tecnologia e comunicazione (11)
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "GOOGL",    # Alphabet (Google)
    "AMZN",     # Amazon
    "META",     # Meta Platforms (Facebook)
    "NVDA",     # NVIDIA
    "TSLA",     # Tesla
    "AVGO",     # Broadcom
    "CRM",      # Salesforce
    "ORCL",     # Oracle
    "AMD",      # Advanced Micro Devices

    # 🏦 Finanza e assicurazioni (5)
    "JPM",      # JPMorgan Chase
    "BAC",      # Bank of America
    "GS",       # Goldman Sachs
    "MS",       # Morgan Stanley
    # "BRK.B",    # Berkshire Hathaway

    # 🛢️ Energia e utilities (3)
    "XOM",      # ExxonMobil
    "CVX",      # Chevron
    "NEE",      # NextEra Energy (rinnovabili)

    # 🛒 Consumi discrezionali e retail (4)
    "WMT",      # Walmart
    "HD",       # Home Depot
    "MCD",      # McDonald's
    "NKE",      # Nike

    # 🏥 Healthcare e farmaceutica (5)
    "JNJ",      # Johnson & Johnson
    "UNH",      # UnitedHealth Group
    "PFE",      # Pfizer
    "LLY",      # Eli Lilly
    "MRK",      # Merck & Co.

    # 🏗️ Industria e trasporti (4)
    "CAT",      # Caterpillar
    "BA",       # Boeing
    "UPS",      # UPS
    "GE",       # General Electric

    # 📦 Real estate, materiali e altri (3)
    "PLD",      # Prologis (REIT logistica)
    "LIN",      # Linde (chimica industriale)
    "DE",       # Deere & Co. (macchine agricole)
]

# =================================
# Area Asia
# =================================

stocks_asia = [
    # 🇯🇵 Giappone (7)
    "7203.T" , # Toyota Motor
    "9984.T" , # SoftBank Group
    "6758.T" , # Sony Group
    "6861.T" , # Keyence
    "8316.T" , # Mitsubishi UFJ Financial
    "9432.T" , # NTT (Nippon Telegraph)
    "8035.T" , # Tokyo Electron

    # 🇨🇳 Cina (8) – "Prominent 10" di Goldman Sachs :contentReference[oaicite:1]{index=1}
    "0700.HK",  # Tencent
    "BABA"   ,  # Alibaba
    "0981.HK",  # Xiaomi (HK)
    "002594.SZ",# BYD
    "3690.HK",  # Meituan
    "9999.HK",  # NetEase
    "300750.SZ",# CATL – debut HK recentemente :contentReference[oaicite:2]{index=2}
    "FUTU"   ,  # Futu Holdings – forte crescita :contentReference[oaicite:3]{index=3}

    # 🇭🇰 Hong Kong (5)
    "0005.HK",  # HSBC
    "1299.HK",  # AIA Group
    "0001.HK",  # CK Hutchison
    "388.HK",   # Hong Kong Exchanges
    "0941.HK",  # China Mobile

    # 🇮🇳 India (7)
    "RELIANCE.NS", # Reliance Industries
    "TCS.NS",      # Tata Consultancy Services
    "HDFCBANK.NS", # HDFC Bank
    "INFY.NS",     # Infosys
    "ICICIBANK.NS",# ICICI Bank
    "HDFC.NS",     # Housing Development Finance
    "KOTAKBANK.NS",# Kotak Mahindra Bank

    # 🇰🇷 Corea del Sud (8)
    "005930.KS",  # Samsung Electronics
    "000660.KS",  # SK Hynix
    "005380.KS",  # Hyundai Motor
    "000270.KS",  # Kia Motors
    "051910.KS",  # LG Chem
    "012330.KS",  # Hyundai Mobis
    "032830.KS",  # Samsung Biologics
    "047810.KS"   # Hanwha Aerospace – difesa in rally :contentReference[oaicite:4]{index=4}
]


# =================================
# World
# =================================


multiasset_global_ucits = [
    # 🇮🇹 Mercati emergenti & obbligazionari
    "EM710.MI",   # Emerging Markets – debito in valuta locale
    "EM57.MI",    # Mercati emergenti – equity
    "EM13.MI",    # MSCI EM index – equity
    "IBTM.MI",    # Investment-grade global bonds
    "IBTS.MI",    # US Treasury bonds
    # "EMG.MI",     # Emerging markets bonds (USD)
    "AHYE.MI",    # High yield Europa
    "IHYG.MI",    # High yield globale
    "IHYU.MI",    # High yield globale USD hedged
    "EMKTB.MI",   # EM bonds short duration
    "EIMI.MI",    # MSCI EM IMI equity
    "C50.MI",     # CSI 300 (Cina)
    "CSMIB.MI",   # FTSE MIB (Italia)
    "DAXX.MI",    # DAX 30 (Germania)
    "IWDE.MI",    # MSCI World ex‑Europe

    # 🪙 Oro & Brent
    "XAD5.MI",    # Xtrackers Physical Gold ETC – oro fisico
    "BRNT.MI",    # Brent Oil (commodity)

    # 🌍 Equity Globali
    "SP5A.MI",    # SPDR S&P 500 UCITS ETF Acc
    "EQQQ.MI",    # Invesco Nasdaq‑100 UCITS ETF
    "XMME.MI",    # Xtrackers MSCI Emerging Markets UCITS
    "TRET.MI",    # VanEck Global Real Estate UCITS ETF
    "GLRE.MI",    # SPDR Dow Jones Global Real Estate UCITS ETF
    "IPRP.MI",    # iShares European Property Yield UCITS ETF
    "XRES.MI",    # Invesco US Real Estate Sector UCITS ETF

    # ⚖️ Weighted Equity UCITS
    "MWEQ.MI",    # Invesco MSCI World Equal Weight UCITS ETF :contentReference[oaicite:1]{index=1}

    # 🏭 Settoriali UCITS S&P US Select Sector
]

# stocks_world = multiasset_global_ucits + settoriali
# ============================================================
# 🌍 World – Universo SOLO Vanguard + Commodities essenziali
# ============================================================
# Regola: 100% Vanguard UCITS dove disponibile su .MI
#         fallback .DE (Xetra) se non quotato su Borsa Italiana
#         + oro fisico e brent (Vanguard non copre commodities)
# Tutti UCITS armonizzati | Ticker verificati su it.vanguard
# ============================================================

multiasset_global_ucits_vanguard = [

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📈 EQUITY GLOBALE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VWCE.MI",   # 🌐 Vanguard FTSE All-World UCITS ETF (Acc)            | IE00BK5BQT80 | .MI ✅
    "VHYL.MI",   # 💰 Vanguard FTSE All-World High Dividend Yield (Dist) | IE00B8GKDB10 | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇺🇸 EQUITY USA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VUAA.MI",   # 🏛️  Vanguard S&P 500 UCITS ETF (Acc)                  | IE00BFMXXD54 | .MI ✅
    "VNRA.MI",   # 🇨🇦 Vanguard FTSE North America UCITS ETF (Acc)       | IE00BK5BQW10 | .MI ✅ USA + Canada

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌍 EQUITY DEVELOPED ex-USA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VEUR.MI",   # 🇪🇺 Vanguard FTSE Developed Europe UCITS ETF (Acc)    | IE00BK5BQX27 | .MI ✅ Europa broad
    "VGER.DE",   # 🇩🇪 Vanguard Germany All Cap UCITS ETF (Dist)         | IE00BG143G97 | .DE ✅ (non su .MI)
    "VJPN.MI",   # 🇯🇵 Vanguard FTSE Japan UCITS ETF (Acc)               | IE00BFMXYX26 | .MI ✅
    "VAPX.MI",   # 🌏 Vanguard FTSE Dev Asia Pacific ex-Japan UCITS ETF  | IE00B9F5YL18 | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌱 EQUITY EMERGING MARKETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VFEA.MI",   # 🌱 Vanguard FTSE Emerging Markets UCITS ETF (Acc)     | IE00BK5BR733 | .MI ✅ EM broad (incl. Cina)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏦 OBBLIGAZIONARIO GOVERNATIVO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VDTA.MI",   # 🇺🇸 Vanguard USD Treasury Bond UCITS ETF (Acc)        | IE00BGYWFS63 | .MI ✅ Treasury USA
    "VGEA.MI",   # 🇪🇺 Vanguard EUR Eurozone Govt Bond UCITS ETF (Acc)   | IE00BH04GL39 | .MI ✅ Govies Area Euro
    "VAGF.MI",   # 🌐 Vanguard Global Aggregate Bond EUR Hedged (Acc)    | IE00BG47KH54 | .MI ✅ Global IG hedged EUR

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 💼 OBBLIGAZIONARIO CORPORATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VUCE.MI",   # 🏦 Vanguard USD Corporate Bond UCITS ETF (Acc)        | IE00BGYWFK87 | .MI ✅ Corporate IG USD
    "VECP.MI",   # 🏦 Vanguard EUR Corporate Bond UCITS ETF (Acc)        | IE00BGYWT403 | .MI ✅ Corporate IG EUR

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌍 OBBLIGAZIONARIO EMERGING MARKETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VDEA.MI",   # 🌱 Vanguard USD EM Govt Bond UCITS ETF (Acc)          | IE00BGYWCB81 | .MI ✅ EM sovereign USD

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🥇 COMMODITIES – unica eccezione non-Vanguard
    #    (Vanguard non emette ETC su commodity)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "XAD5.MI",   # 🥇 Xtrackers Physical Gold ETC (EUR Hdg)              | DE000A1E0HR8 | .MI ✅ ⚠️ non Vanguard – oro fisico
    "BRNT.MI",   # 🛢️  WisdomTree Brent Crude Oil ETC                    | GB00B0CTWC01 | .MI ✅ ⚠️ non Vanguard – petrolio
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 ISIN MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISIN_MAP_VANGUARD = {
    # ✅ Vanguard – Borsa Italiana (.MI)
    "VWCE.MI" : "IE00BK5BQT80",  # FTSE All-World Acc
    "VHYL.MI" : "IE00B8GKDB10",  # FTSE All-World High Div Yield
    "VUAA.MI" : "IE00BFMXXD54",  # S&P 500 Acc
    "VNRA.MI" : "IE00BK5BQW10",  # FTSE North America Acc
    "VEUR.MI" : "IE00BK5BQX27",  # FTSE Developed Europe Acc
    "VJPN.MI" : "IE00BFMXYX26",  # FTSE Japan Acc
    "VAPX.MI" : "IE00B9F5YL18",  # FTSE Dev Asia Pac ex-Japan
    "VFEA.MI" : "IE00BK5BR733",  # FTSE Emerging Markets Acc
    "VDTA.MI" : "IE00BGYWFS63",  # USD Treasury Bond Acc
    "VGEA.MI" : "IE00BH04GL39",  # EUR Eurozone Govt Bond Acc
    "VAGF.MI" : "IE00BG47KH54",  # Global Aggregate EUR Hedged Acc
    "VUCE.MI" : "IE00BGYWFK87",  # USD Corporate Bond Acc
    "VECP.MI" : "IE00BGYWT403",  # EUR Corporate Bond Acc
    "VDEA.MI" : "IE00BGYWCB81",  # USD EM Govt Bond Acc
    # ✅ Vanguard – Xetra (.DE) – non disponibile su .MI
    "VGER.DE" : "IE00BG143G97",  # Germany All Cap Dist
    # ⚠️ Non-Vanguard – commodity (unica eccezione strutturale)
    "XAD5.MI" : "DE000A1E0HR8",  # Xtrackers Physical Gold ETC
    "BRNT.MI" : "GB00B0CTWC01",  # WisdomTree Brent Crude ETC
}

# Selezionati direttamente su https://www.it.vanguard/professional/prodotti con filtri:
# ETF 
# Azionario Obbligazionario Mercato monetario 
# EUR USD 
# Accumulazione

vanguard_etf = [
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📈 EQUITY GLOBALE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VWCE.MI",  # 🌐 Vanguard FTSE All-World UCITS ETF (Acc)       | IE00BK5BQT80 | USD | OCF 0,19% | .MI ✅
    "VHVE.MI",  # 🌐 Vanguard FTSE Developed World UCITS ETF (Acc) | IE00BK5BQV03 | USD | OCF 0,12% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 💰 EQUITY HIGH DIVIDEND
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VHYA.MI",  # 💰 Vanguard FTSE All-World High Div Yield UCITS ETF (Acc) | IE00BK5BR626 | USD | OCF 0,29% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇺🇸 EQUITY USA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VUAA.MI",  # 🇺🇸 Vanguard S&P 500 UCITS ETF (Acc)            | IE00BFMXXD54 | USD | OCF 0,07% | .MI ✅
    "VNRA.MI",  # 🇺🇸 Vanguard FTSE North America UCITS ETF (Acc) | IE00BK5BQW10 | USD | OCF 0,08% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇪🇺 EQUITY EUROPA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VWCG.MI",  # 🇪🇺 Vanguard FTSE Developed Europe UCITS ETF (Acc)       | IE00BK5BQX27 | EUR | OCF 0,10% | .MI ✅
    "VERE.MI",  # 🇪🇺 Vanguard FTSE Developed Europe ex UK UCITS ETF (Acc) | IE00BK5BQY34 | EUR | OCF 0,10% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇯🇵 EQUITY GIAPPONE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VJPA.MI",  # 🇯🇵 Vanguard FTSE Japan UCITS ETF (Acc)            | IE00BFMXYX26 | USD | OCF 0,10% | .MI ✅
    "VJPE.MI",  # 🇯🇵 Vanguard FTSE Japan UCITS ETF EUR Hedged (Acc) | IE00BFMXYY33 | EUR | OCF 0,13% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌏 EQUITY ASIA-PACIFIC
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VGEK.DE",  # 🌏 Vanguard FTSE Developed Asia Pacific ex Japan UCITS ETF (Acc) | IE00BK5BQZ41 | USD | OCF 0,15% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌍 EQUITY EMERGENTI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VFEA.MI",  # 🌍 Vanguard FTSE Emerging Markets UCITS ETF (Acc) | IE00BK5BR733 | USD | OCF 0,17% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌱 EQUITY ESG
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "V3AA.MI",  # 🌱 Vanguard ESG Global All Cap UCITS ETF (Acc)                 | IE00BNG8L278 | USD | OCF 0,24% | .MI ✅
    "V3NA.MI",  # 🌱 Vanguard ESG North America All Cap UCITS ETF (Acc)          | IE000O58J820 | USD | OCF 0,12% | .MI ✅
    "V3EA.MI",  # 🌱 Vanguard ESG Developed Europe All Cap UCITS ETF (Acc)       | IE000QUOSE01 | EUR | OCF 0,12% | .MI ✅
    "V3PA.MI",  # 🌱 Vanguard ESG Developed Asia Pacific All Cap UCITS ETF (Acc) | IE000GOJO2A3 | USD | OCF 0,17% | .MI ✅
    "V3MA.MI",  # 🌱 Vanguard ESG Emerging Markets All Cap UCITS ETF (Acc)       | IE000KPJJWM6 | USD | OCF 0,19% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏛️ BOND TREASURY USA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VDST.MI",  # 🏛️ Vanguard U.S. Treasury 0-1 Year Bond UCITS ETF (Acc)            | IE00BLRPPV00 | USD | OCF 0,05% | .MI ✅
    "VUDS.MI",  # 🏛️ Vanguard U.S. Treasury 1-3 Year Bond UCITS ETF (Acc)            | IE000H3Q3AF6 | USD | OCF 0,05% | .MI ✅
    "VUDE.MI",  # 🏛️ Vanguard U.S. Treasury 1-3 Year Bond UCITS ETF EUR Hedged (Acc) | IE000TAV7246 | EUR | OCF 0,08% | .MI ✅
    "VITS.MI",  # 🏛️ Vanguard U.S. Treasury 3-7 Year Bond UCITS ETF (Acc)            | IE000VZ8BBU9 | USD | OCF 0,05% | .MI ✅
    "VLDS.MI",  # 🏛️ Vanguard U.S. Treasury 7-10 Year Bond UCITS ETF (Acc)           | IE000UXDT343 | USD | OCF 0,05% | .MI ✅
    "VDTA.MI",  # 🏛️ Vanguard USD Treasury Bond UCITS ETF (Acc)                      | IE00BGYWFS63 | USD | OCF 0,05% | .MI ✅
    "VDTE.MI",  # 🏛️ Vanguard USD Treasury Bond UCITS ETF EUR Hedged (Acc)           | IE00BMX0B631 | EUR | OCF 0,08% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇬🇧 BOND GILT UK
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VGUE.DE",  # 🇬🇧 Vanguard U.K. Gilt UCITS ETF EUR Hedged (Acc) | IE00BMX0B524 | EUR | OCF 0,08% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇪🇺 BOND GOV EUROZONA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VGEA.MI",  # 🇪🇺 Vanguard EUR Eurozone Government Bond UCITS ETF (Acc)          | IE00BH04GL39 | EUR | OCF 0,07% | .MI ✅
    "VSGF.MI",  # 🇪🇺 Vanguard EUR Eurozone Government 1-3 Year Bond UCITS ETF (Acc) | IE00004S2680 | EUR | OCF 0,07% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌐 BOND GOV GLOBALE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VGGF.MI",  # 🌐 Vanguard Global Government Bond UCITS ETF EUR Hedged (Acc) | IE000B1A2798 | EUR | OCF 0,10% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌍 BOND GOV EMERGENTI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VDEA.MI",  # 🌍 Vanguard USD Emerging Markets Government Bond UCITS ETF (Acc) | IE00BGYWCB81 | USD | OCF 0,23% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌐 BOND AGGREGATE GLOBALE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VAGF.MI",  # 🌐 Vanguard Global Aggregate Bond UCITS ETF EUR Hedged (Acc) | IE00BG47KH54 | EUR | OCF 0,08% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏢 BOND CORPORATE EUR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VECA.MI",  # 🏢 Vanguard EUR Corporate Bond UCITS ETF (Acc)          | IE00BGYWT403 | EUR | OCF 0,07% | .MI ✅
    "VSCF.MI",  # 🏢 Vanguard EUR Corporate 1-3 Year Bond UCITS ETF (Acc) | IE00BH04FZ00 | EUR | OCF 0,09% | .MI ✅
    "V3RE.MI",  # 🌱 Vanguard ESG EUR Corporate Bond UCITS ETF (Acc)      | IE000QADMYA3 | EUR | OCF 0,09% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏢 BOND CORPORATE USD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VUCE.MI",  # 🏢 Vanguard USD Corporate Bond UCITS ETF (Acc)                     | IE00BGYWFK87 | USD | OCF 0,07% | .MI ✅
    "VDCE.MI",  # 🏢 Vanguard USD Corporate Bond UCITS ETF EUR Hedged (Acc)          | IE00BGYWFL94 | EUR | OCF 0,10% | .MI ✅
    "VDCA.MI",  # 🏢 Vanguard USD Corporate 1-3 Year Bond UCITS ETF (Acc)            | IE00BGYWSV06 | USD | OCF 0,09% | .MI ✅
    "VCDE.MI",  # 🏢 Vanguard USD Corporate 1-3 Year Bond UCITS ETF EUR Hedged (Acc) | IE00BGYWSW13 | EUR | OCF 0,12% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌱 BOND CORPORATE GLOBALE ESG
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "V3GF.MI",  # 🌱 Vanguard ESG Global Corporate Bond UCITS ETF EUR Hedged (Acc) | IE00BNDS1P30 | EUR | OCF 0,15% | .MI ✅

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 💵 MONETARIO EUR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "VCSHA.MI",  # 💵 Vanguard EUR Cash UCITS ETF (Acc) | IE000SOORXS0 | EUR | OCF 0,07% | .MI ✅
]

multiasset_global_ucits_wfo = [

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📈 EQUITY GLOBALE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # vs VWCE.MI (Vanguard FTSE All-World Acc, inception 2019)
    "SWDA.MI",   # 🌐 iShares Core MSCI World UCITS ETF (Acc)            | IE00B4L5Y983 | .MI ✅ | 📅 2009

    # vs VHYL.MI (Vanguard FTSE All-World HiDiv, inception 2013)
    "IWDP.MI",   # 💰 iShares Developed Markets Property Yield UCITS ETF | IE00B1FZS350 | .MI ✅ | 📅 2007
                 #    (proxy high-div/yield globale con dividendi reali)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇺🇸 EQUITY USA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # vs VUAA.MI (Vanguard S&P 500 Acc, inception 2019)
    "CSSPX.MI",  # 🏛️  iShares Core S&P 500 UCITS ETF (Acc)              | IE00B5BMR087 | .MI ✅ | 📅 2010

    # vs VNRA.MI (Vanguard FTSE North America, inception 2019)
    "CSSPX.MI",  # 🇨🇦 iShares Core S&P 500 UCITS ETF (Acc)              | (stesso sopra) | proxy USA+Canada
                 #    NB: non esiste un ETF North America UCITS con storico lungo → S&P500 come proxy

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌍 EQUITY DEVELOPED ex-USA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # vs VEUR.MI (Vanguard FTSE Developed Europe Acc, inception 2019)
    "SMEA.MI",   # 🇪🇺 iShares Core MSCI Europe UCITS ETF (Acc)          | IE00B4K48X80 | .MI ✅ | 📅 2009

    # vs VGER.DE (Vanguard Germany All Cap, inception 2018, non su .MI)
    "DAXX.MI",   # 🇩🇪 Amundi DAX II UCITS ETF (Acc)                     | LU0252633754 | .MI ✅ | 📅 2006

    # vs VJPN.MI (Vanguard FTSE Japan Acc, inception 2019)
    "SJPA.MI",   # 🇯🇵 iShares Core MSCI Japan IMI UCITS ETF (Acc)       | IE00B4L5YX21 | .MI ✅ | 📅 2009

    # vs VAPX.MI (Vanguard FTSE Dev Asia Pac ex-Japan, inception 2009 dist)
    "CPXJ.L",   # 🌏 iShares Core MSCI Pacific ex-Japan UCITS ETF (Acc) | IE00B52MJY50 | .MI ✅ | 📅 2010

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌱 EQUITY EMERGING MARKETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # vs VFEA.MI (Vanguard FTSE EM Acc, inception 2019)
    "IEEM.MI",   # 🌱 iShares MSCI EM UCITS ETF (Dist)                   | IE00B0M63177 | .MI ✅ | 📅 2005

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏦 OBBLIGAZIONARIO GOVERNATIVO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # vs VDTA.MI (Vanguard USD Treasury Acc, inception 2019)
    "IBTM.MI",   # 🇺🇸 iShares $ Treasury Bond 7-10yr UCITS ETF (Dist)   | IE00B1FZS798 | .MI ✅ | 📅 2006

    # vs VGEA.MI (Vanguard EUR Eurozone Govt Acc, inception 2019)
    "IBGX.MI",   # 🇪🇺 iShares Core Euro Govt Bond UCITS ETF (Acc)       | IE00B4WXJJ64 | .MI ✅ | 📅 2009

    # vs VAGF.MI (Vanguard Global Aggregate EUR Hdg Acc, inception 2019)
    "AGGH.MI",   # 🌐 iShares Core Global Aggregate Bond EUR Hdg (Acc)   | IE00BDBRDM35 | .MI ✅ | 📅 2017
                 #    ⚠️ inception 2017: migliore disponibile per Global Aggregate hedged EUR su .MI

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 💼 OBBLIGAZIONARIO CORPORATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # vs VUCE.MI (Vanguard USD Corporate Acc, inception 2019)
    "LQDE.MI",   # 🏦 iShares $ Corp Bond UCITS ETF (Dist)               | IE0032523478 | .MI ✅ | 📅 2003

    # vs VECP.MI (Vanguard EUR Corporate Acc, inception 2019)
    "IEAC.MI",   # 🏦 iShares Core Euro Corporate Bond UCITS ETF (Acc)   | IE00B3F81R35 | .MI ✅ | 📅 2009

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌍 OBBLIGAZIONARIO EMERGING MARKETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # vs VDEA.MI (Vanguard USD EM Govt Acc, inception 2019)
    "IEMB.MI",   # 🌱 iShares JPM $ EM Bond UCITS ETF (Dist)             | IE00B2NPKV68 | .MI ✅ | 📅 2008

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🥇 COMMODITIES – stesso della lista _vanguard
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "XAD5.MI",   # 🥇 Xtrackers Physical Gold ETC (EUR Hdg)              | DE000A1E0HR8 | .MI ✅ | 📅 2007
    "BRNT.MI",   # 🛢️  WisdomTree Brent Crude Oil ETC                    | GB00B0CTWC01 | .MI ✅ | 📅 2006
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 ISIN MAP + INCEPTION (fonte: BlackRock, Amundi, emittenti)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISIN_MAP_WFO = {
    "SWDA.MI"  : ("IE00B4L5Y983", 2009),  # iShares Core MSCI World Acc
    "IWDP.MI"  : ("IE00B1FZS350", 2007),  # iShares Dev Mkts Property Yield
    "CSSPX.MI" : ("IE00B5BMR087", 2010),  # iShares Core S&P 500 Acc
    "SMEA.MI"  : ("IE00B4K48X80", 2009),  # iShares Core MSCI Europe Acc
    "DAXX.MI"  : ("LU0252633754", 2006),  # Amundi DAX II Acc
    "SJPA.MI"  : ("IE00B4L5YX21", 2009),  # iShares Core MSCI Japan IMI Acc
    "CPXJ.MI"  : ("IE00B52MJY50", 2010),  # iShares Core MSCI Pacific ex-Japan
    "IEEM.MI"  : ("IE00B0M63177", 2005),  # iShares MSCI EM UCITS Dist
    "IBTM.MI"  : ("IE00B1FZS798", 2006),  # iShares $ Treasury 7-10yr Dist
    "IBGX.MI"  : ("IE00B4WXJJ64", 2009),  # iShares Core Euro Govt Bond Acc
    "AGGH.MI"  : ("IE00BDBRDM35", 2017),  # iShares Core Global Agg EUR Hdg ⚠️ solo da 2017
    "LQDE.MI"  : ("IE0032523478", 2003),  # iShares $ Corp Bond Dist
    "IEAC.MI"  : ("IE00B3F81R35", 2009),  # iShares Core Euro Corp Bond Acc
    "IEMB.MI"  : ("IE00B2NPKV68", 2008),  # iShares JPM $ EM Bond Dist
    "XAD5.MI"  : ("DE000A1E0HR8", 2007),  # Xtrackers Physical Gold ETC
    "BRNT.MI"  : ("GB00B0CTWC01", 2006),  # WisdomTree Brent Crude ETC
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔁 MAPPING Vanguard (_vanguard) → Proxy WFO (_wfo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VANGUARD_TO_WFO = {
    "VWCE.MI" : "SWDA.MI",   # FTSE All-World   → MSCI World (excl. EM small cap, ~90% overlap)
    "VHYL.MI" : "IWDP.MI",   # All-World HiDiv  → Dev Mkts Property Yield (proxy income)
    "VUAA.MI" : "CSSPX.MI",  # S&P 500          → S&P 500 iShares ✅ identico indice
    "VNRA.MI" : "CSSPX.MI",  # North America    → S&P 500 (USA ~96% del North America)
    "VEUR.MI" : "SMEA.MI",   # FTSE Dev Europe  → MSCI Europe ✅ quasi identico
    "VGER.DE" : "DAXX.MI",   # Germany All Cap  → DAX 40 (large cap DE, proxy accettabile)
    "VJPN.MI" : "SJPA.MI",   # FTSE Japan       → MSCI Japan IMI ✅ quasi identico
    "VAPX.MI" : "CPXJ.MI",   # Asia Pac ex-Jpn  → MSCI Pacific ex-Japan ✅ identico
    "VFEA.MI" : "IEEM.MI",   # FTSE EM          → MSCI EM ✅ quasi identico
    "VDTA.MI" : "IBTM.MI",   # USD Treasury All → USD Treasury 7-10yr (duration simile)
    "VGEA.MI" : "IBGX.MI",   # EUR Eurozone Gov → Euro Govt Bond ✅ identico
    "VAGF.MI" : "AGGH.MI",   # Global Agg EUR H → Global Agg EUR H ✅ identico (ma dal 2017 ⚠️)
    "VUCE.MI" : "LQDE.MI",   # USD Corp Bond    → iShares $ Corp Bond ✅ identico
    "VECP.MI" : "IEAC.MI",   # EUR Corp Bond    → Euro Corp Bond ✅ identico
    "VDEA.MI" : "IEMB.MI",   # USD EM Govt Bond → JPM $ EM Bond ✅ identico
    "XAD5.MI" : "XAD5.MI",   # Gold ETC         → identico ✅
    "BRNT.MI" : "BRNT.MI",   # Brent ETC        → identico ✅
}
# =================================
# Core 2
# =================================


# 📌 Titoli di Stato
core2_titoli_stato = [
    'EM710.MI',  # 📈 Titoli Stato Euro Lungo
    'EM57.MI',   # 📊 Titoli Stato Euro Medio
    'EM13.MI',   # 📉 Titoli Stato Euro Breve
    'IBTM.MI',   # 🇺🇸 Titoli Stato USA Lungo
    'IBTS.MI',   # 🇺🇸 Titoli Stato USA Breve
    'EMG.MI'     # 🌍 Titoli Stato Euro generici
]

# 💵 Obbligazionario
core2_obbligazionario = [
    'AHYE.MI',  # 💶 Obbligazionario Euro High Yield
    'IHYG.MI',  # 💶 Obbligazionario Euro High Yield
    'IHYU.MI',  # 💲 Obbligazionario USA High Yield
    'EMKTB.MI'  # 💲 Obbligazionario USA
]

# 📊 Azionario
core2_azionario = [
    'CSSPX.MI', # 🇺🇸 Azionario USA
    'EIMI.MI',  # 🌏 Azionario Mercati Emergenti
    'C50.MI',   # 🇪🇺 Azionario Euro
    'CSMIB.MI', # 🇮🇹 Azionario Italia
    'DAXX.MI',  # 🇩🇪 Azionario Germania
    'IWDE.MI'   # 🌍 Azionario Globale
]

# 🥇 Metalli Preziosi
core2_metalli_preziosi = [
    'SGLD.MI',  # 🟡 Oro
    'PHPT.MI',  # ⚪️ Platino
    'PHPD.MI',  # ⚫️ Palladio
    'PHAG.MI'   # 🔘 Argento
]

# ⚙️ Commodities Core – Oro, Petrolio, Gas Naturale
core2_commodities = [
    'SGLD.MI',   # 🟡 Oro fisico (Invesco Physical Gold ETC)
    'OILB.MI',   # 🛢️ Petrolio Brent (WisdomTree Brent Crude Oil ETC)
    # 'NGAS.MI',   # 🔥 Gas Naturale (WisdomTree Natural Gas ETC)
]

# Lista completa "core2"
core2 = core2_titoli_stato + core2_obbligazionario + core2_azionario + core2_commodities

# 🇩🇪 Beneficiari potenziali del maxi-piano tedesco (~€1.000 mld)
germany_plan_beneficiaries = [
    # 🛡️ Difesa
    "RHM.DE",   # 🛡️ Rheinmetall — munizioni, veicoli corazzati, sistemi difesa
    "HAG.DE",   # 🛰️ Hensoldt — sensori/radar/optronica
    "R3NK.DE",  # ⚙️ RENK Group — trasmissioni per mezzi militari e navali
    "MTX.DE",   # ✈️ MTU Aero Engines — motori aeronautici (civile & militare)

    # ⚡ Energia & Reti
    "ENR.DE",   # ⚡ Siemens Energy — HVDC, reti, generazione
    "EOAN.DE",  # 🔌 E.ON — distribuzione/distribuzione elettrica, grid capex
    "RWE.DE",   # ♻️ RWE — rinnovabili & generation
    "NDX1.DE",  # 🌬️ Nordex — eolico onshore (turbine)

    # 🚆 Ferrovie & Infrastrutture
    "SIE.DE",   # 🚆 Siemens (Mobility) — segnalamento, rolling stock
    "VOS.DE",   # 🛤️ Vossloh — armamento ferroviario (scambi, traverse)
    "KBX.DE",   # 🛑 Knorr-Bremse — sistemi frenanti per treni
    "HOT.DE",   # 🏗️ HOCHTIEF — grandi lavori infrastrutturali

    # 📡 Digitale & Fibra
    "DTE.DE",   # 📡 Deutsche Telekom — FTTH, 5G
    "1U1.DE",   # 📶 1&1 — rete mobile & fibra (Versatel)
    "ADV.DE",   # 🔌 Adtran Networks — apparati ottici/backbone

    # 🧱 Abitazioni & Costruzioni
    "HEI.DE",   # 🧱 Heidelberg Materials — cemento/materiali per infrastrutture
    "VNA.DE",   # 🏘️ Vonovia — residenziale (riqualificazioni/nuove unità)
    "LEG.DE",   # 🏘️ LEG Immobilien — residenziale NRW
    "GBF.DE",   # 🛠️ Bilfinger — ingegneria/impiantistica per progetti infra/energia

    # 🚛 Logistica & Aeroporti
    "DHL.DE",   # 📦 DHL Group — logistica e supply chain
    "FRA.DE",   # 🛫 Fraport — gestione aeroporti (capex trasporti)

    # 🔋 Semiconduttori & Manifattura avanzata
    "IFX.DE",   # 🔋 Infineon — power semis; nuova fab a Dresda (indotto locale)
    "WAF.DE",   # 🥞 Siltronic — wafer 300mm
    "AIXA.DE",  # 🧪 Aixtron — equipment GaN/SiC per power & data center

    # "BBBY",     # titolo di test: grande volailita' pessi performance
]

us_strategic_tech_beneficiaries = [
    # 🧠 Semiconduttori — produzione domestica / reshoring / AI compute
    "INTC",     # 🏭 Intel — fabs USA; beneficiario diretto della capacità produttiva domestica
    "GFS",      # 🏭 GlobalFoundries — foundry USA; produzione domestica strategica
    "MU",       # 💾 Micron — memoria; espansione produttiva negli USA
    "NVDA",     # 🧠 NVIDIA — AI compute e leadership tecnologica USA
    "AMD",      # ⚡ AMD — CPU/GPU e acceleratori AI
    "AVGO",     # 🔌 Broadcom — networking/ASIC per infrastruttura AI

    # 🛠️ Semiconductor equipment / supply chain
    "AMAT",     # 🏗️ Applied Materials — equipment per produzione chip
    "LRCX",     # 🧪 Lam Research — wafer fabrication equipment
    "KLAC",     # 🔬 KLA — process control / metrology
    "TER",      # 🧰 Teradyne — semiconductor test equipment

    # 🧲 Terre rare & minerali critici
    "MP",       # 🧲 MP Materials — mine-to-magnet USA; partnership diretta con DoD
    "USAR",     # 🇺🇸 USA Rare Earth — mine-to-magnet; Texas/Oklahoma/South Carolina
    "UUUU",     # ⚛️ Energy Fuels — rare earth separation + uranium / critical materials
    "NB",       # ⛏️ NioCorp — Elk Creek; niobio, scandio, titanio e potenziali REE

    # ⚛️ Quantum computing
    "IONQ",     # ⚛️ IonQ — trapped-ion quantum computing
    "RGTI",     # 🧊 Rigetti — superconducting quantum computers
    "QBTS",     # 🔷 D-Wave Quantum — quantum annealing
    "QUBT",     # 💡 Quantum Computing Inc. — photonic / quantum technologies

    # 🖥️ Quantum ecosystem / industrializzazione
    "IBM",      # ⚛️ IBM — quantum hardware, software e research
    "GOOGL",    # 🧪 Alphabet — Google Quantum AI
    "MSFT",     # 🔬 Microsoft — quantum computing / cloud ecosystem
]

# =============================================================================
# 🇮🇹 UNIVERSE: Italy Big Cap — R-Portfolio
# Validata con yfinance + yahooquery — tutti i ticker scaricano correttamente.
# Criteri: quotazione su Borsa Italiana, storico continuo dal 2014,
# no delisting risk, capitalizzazione > 5 Mld EUR, diversificazione settoriale.
# Selection bias mitigato: inclusione rule-based per cap + anzianità.
# =============================================================================

italy_bigcap_tickers = [

    # 🏦 BANCARIO / ASSICURATIVO
    "ISP.MI",      # IT0000072618 — Intesa Sanpaolo S.p.A.
    "UCG.MI",      # IT0005239360 — UniCredit S.p.A.
    "MB.MI",       # IT0000062957 — Mediobanca S.p.A.
    "BAMI.MI",     # IT0005218380 — Banco BPM S.p.A.
    "FBK.MI",      # IT0000072170 — FinecoBank S.p.A.
    "G.MI",        # IT0000062072 — Assicurazioni Generali S.p.A.

    # ⚡ ENERGIA / UTILITY
    "ENI.MI",      # IT0003132476 — Eni S.p.A.
    "ENEL.MI",     # IT0003128367 — Enel S.p.A.
    "SRG.MI",      # IT0003153415 — Snam S.p.A.
    "TRN.MI",      # IT0003242622 — Terna S.p.A.
    "A2A.MI",      # IT0001233417 — A2A S.p.A.

    # 🏭 INDUSTRIALE / DIFESA
    "LDO.MI",      # IT0003856405 — Leonardo S.p.A.
    "1CNHI.MI",    # NL0010545661 — CNH Industrial N.V.  ← corretto da CNHI.MI

    # 🏎️ AUTOMOTIVE / LUSSO
    "RACE.MI",     # NL0011585146 — Ferrari N.V.

    # 💻 SEMICONDUTTORI
    "STMMI.MI",    # NL0000226223 — STMicroelectronics N.V.  ← corretto da STM.PA

    # 🔩 MATERIALI / METALLURGIA
    "TEN.MI",      # LU0156801721 — Tenaris S.A.

    # 📡 INFRASTRUTTURE DIGITALI
    "INW.MI",      # IT0005090300 — INWIT S.p.A.

    # 💊 FARMACEUTICO
    "REC.MI",      # IT0003828271 — Recordati S.p.A.

    # 🍷 BEVERAGE
    "CPR.MI",      # IT0001173752 — Davide Campari-Milano S.p.A.  ← corretto da MONO.MI
]


# ETF/ETP UCITS equivalenti — mercati trattati su Directa

# 📦 zehnlabs_etf_blend_200_35
zehnlabs_etf_blend_200_35 = [
    # — Fattori / broad beta —
    "BTAL",  # ♟️ AGFiQ US Market Neutral Anti-Beta — fattore anti-beta (market neutral)
    "USMV",  # 🧊 iShares MSCI USA Min Vol — low volatility USA
    "VTI",   # 🇺🇸 Vanguard Total Stock Market — azionario USA totale
    "VYM",   # 💰 Vanguard High Dividend Yield — dividendo alto USA
    "SCHD",  # 💵 Schwab U.S. Dividend Equity — dividendi qualità (large cap)
    "XMLV",  # 🧊 Invesco S&P MidCap Low Volatility — mid cap low-vol

    # — Commodity / precious metals —
    "DBC",   # 🛢️ Invesco DB Commodity Index — commodities broad via futures
    "GLD",   # 🪙 SPDR Gold Trust — oro fisico
    "IAU",   # 🪙 iShares Gold Trust — oro fisico (costo più basso)

    # — Settori USA (SPDR) —
    "VOX",   # 📡 Communication Services
    "XLB",   # 🧱 Materials
    "XLE",   # ⛽ Energy
    "XLI",   # 🛠️ Industrials
    "XLK",   # 🖥️ Technology
    "XLP",   # 🧃 Consumer Staples
    "XLU",   # 💡 Utilities
    "XLV",   # 🩺 Health Care
    "XLY",   # 🛍️ Consumer Discretionary

    # — Real estate —
    "VNQ",   # 🏠 Vanguard Real Estate — REITs USA

    # — Tech / Nasdaq —
    "QQQ",   # 💻 Invesco QQQ — Nasdaq-100 core tech

    # — Leverage / Inverse (uso tattico, daily reset) —
    "PSQ",   # ⛔ ProShares Short QQQ — Nasdaq-100 −1x
    "QID",   # ⛔ ProShares UltraShort QQQ — Nasdaq-100 −2x
    "QLD",   # ⚡ ProShares Ultra QQQ — Nasdaq-100 +2x
    "TQQQ",  # ⚡ ProShares UltraPro QQQ — Nasdaq-100 +3x
    "SSO",   # ⚡ ProShares Ultra S&P 500 — S&P 500 +2x

    # — Volatilità (VIX) —
    "SVXY",  # 🔻 ProShares Short VIX Short-Term — short VIX (−0.5x)
    "UVXY",  # ⚠️ ProShares Ultra VIX Short-Term — long VIX (+1.5x)
    "VIXY",  # 📈 ProShares VIX Short-Term Futures — long VIX (breve termine)
    "VIXM",  # 📈 ProShares VIX Mid-Term Futures — long VIX (medio termine)

    # — Valute —
    "UUP",   # 💲 Invesco DB USD Bullish — long USD (proxy DXY)
]

# Zehnlabs · ETF Blend 301-20 — universo strumenti (dal report)
zehnlabs_etf_blend_301_20 = [
    # 💵 Cash / T-Bills (proxy USD)
    "IBTS.MI",   # 💵 iShares $ Treasury 1–3yr UCITS (BI) — cash-like short duration :contentReference[oaicite:0]{index=0}
    "IB01.L",    # 💵 iShares $ Treasury 0–1yr UCITS (LSE) — ultra-short, Directa abilita LSE :contentReference[oaicite:1]{index=1}

    # 🟡 Oro (GLD)
    "SGLD.MI",   # 🟡 Invesco Physical Gold ETC (BI) :contentReference[oaicite:2]{index=2}
    "PHAU.MI",   # 🟡 WisdomTree Physical Gold (BI) — alternativa liquida :contentReference[oaicite:3]{index=3}

    # 💻 Nasdaq-100 (QQQ)
    "CSNDX.MI",  # 💻 iShares Nasdaq-100 UCITS (BI) 
    "EQQQ.MI",   # 💻 Invesco NASDAQ-100 UCITS (BI) — alternativa 

    # 💡 Tecnologia USA (XLK / TECL proxy UCITS)
    "XLKS.MI",   # 💡 SPDR S&P U.S. Technology Select Sector UCITS (BI) 

    # 🧪 Semiconduttori (SMH / SOXL)
    "SMH",    # 🧪 VanEck Semiconductor UCITS (BI) — semis globali 
    # "3USD.MI",   # ⚙️ WisdomTree Semiconductors 3x Daily Leveraged (BI) — leva su semis :contentReference[oaicite:8]{index=8}
    "EUS3.MI",   # ⚙️ WisdomTree Long USD Short EUR 3x Daily

    # ⚙️ Nasdaq-100 con leva (QLD/TQQQ)
    "QQQ3.MI",   # ⚙️ WisdomTree NASDAQ-100 3x Daily Leveraged (BI) :contentReference[oaicite:9]{index=9}
    # "LSQQ2.MI",  # ⚙️ Leverage Shares 2x Long Nasdaq-100 (BI) — alternativa 2x :contentReference[oaicite:10]{index=10}
    # "SQQQ.MI",  # ⚙️ Leverage Shares -5x Short Nasdaq 100 ETP Securities (SQQQ.MI)
    "SQQQ",  # ⚙️ Leverage Shares -5x Short Nasdaq 100 ETP Securities (SQQQ.MI)

    # 🇺🇸 USA broad market (VTI proxy UCITS)
    "VUSA.MI",   # 🇺🇸 Vanguard S&P 500 UCITS (BI) — proxy semplice per mercato USA :contentReference[oaicite:11]{index=11}

    # 🌍 Globale (per test multi-asset / benchmark)
    "VWRL.MI",   # 🌍 Vanguard FTSE All-World UCITS (BI) :contentReference[oaicite:12]{index=12}

    # 🧭 S&P 500 leve (SPXL / inverse PSQ-like su S&P)
    "3USL.MI",   # ⬆️ WisdomTree S&P 500 3x Daily Leveraged (BI) :contentReference[oaicite:13]{index=13}
    "XT21.MI",   # ⬇️ Xtrackers S&P 500 2x Inverse Daily Swap UCITS (BI) :contentReference[oaicite:14]{index=14}
    "3USS.MI",   # ⬇️ WisdomTree S&P 500 3x Daily Short (BI) :contentReference[oaicite:15]{index=15}

    # 🌏 Emergenti con leva (EDC / EDZ)
    "3EUL.MI",   # 🌏 WisdomTree Emerging Markets 3x Daily Leveraged (BI) :contentReference[oaicite:16]{index=16}
    "3EUS.MI",   # 🌏 WisdomTree Emerging Markets 3x Daily Short (BI) :contentReference[oaicite:17]{index=17}

    # ⚡ Volatilità (UVXY / VIXY)
    "VIXL.MI",   # ⚡ WisdomTree S&P 500 VIX Short-Term Futures 2.25x Daily Lev (BI) :contentReference[oaicite:18]{index=18}
]

# Zehnlabs · ETF Blend 101_15
zehnlabs_etf_blend_101_15 = [
    "BTAL",  # ♟️ AGFiQ US Market Neutral Anti-Beta — fattore anti-beta (market neutral)
    "DBC",   # 🛢️ Invesco DB Commodity Index — commodities broad via futures
    "IAU",   # 🪙 iShares Gold Trust — oro fisico
    "PSQ",   # 🔻 ProShares Short QQQ — Nasdaq-100 inverse -1x (tattico)
    "QID",   # ⛔ ProShares UltraShort QQQ — Nasdaq-100 inverse -2x (leva, daily)
    "QLD",   # ⚡ ProShares Ultra QQQ — Nasdaq-100 +2x (leva, daily)
    "QQQ",   # 💻 Invesco QQQ — Nasdaq-100 core tech
    "SCHD",  # 💵 Schwab U.S. Dividend Equity — dividendi qualità (large cap)
    "UUP",   # 💲 Invesco DB USD Bullish — long USD (proxy DXY)
    "VTI",   # 🇺🇸 Vanguard Total Stock Market — azionario USA totale
    "XLK",   # 🖥️ Technology Select Sector SPDR — settore tecnologia USA
    "XLP",   # 🧃 Consumer Staples Select Sector SPDR — beni di prima necessità
    "XLU",   # 💡 Utilities Select Sector SPDR — utilities USA
    "XMLV",  # 🧊 Invesco S&P MidCap Low Volatility — mid cap low-vol
]

# Zehnlabs · ETF Blend 300-40
zehnlabs_etf_blend_300_40 = [
    # — Fattori / broad beta —
    "BTAL",  # ♟️ AGFiQ US Market Neutral Anti-Beta — fattore anti-beta (market neutral)
    "DVY",   # 💵 iShares Select Dividend — dividendi USA value
    "EFA",   # 🌍 iShares MSCI EAFE — sviluppati ex-USA/Canada
    "USMV",  # 🧊 iShares MSCI USA Min Vol — low volatility USA
    "VEA",   # 🌍 Vanguard FTSE Developed Markets — sviluppati ex-USA
    "VTI",   # 🇺🇸 Vanguard Total Stock Market — azionario USA totale
    "VYM",   # 💰 Vanguard High Dividend Yield — dividendo alto USA

    # — Commodity / precious metals —
    "GLD",   # 🪙 SPDR Gold Trust — oro fisico
    "SLV",   # 🥈 iShares Silver Trust — argento fisico
    "PLG",   # ⛏️ Platinum Group Metals — mineraria platino (azione)

    # — Settori USA (SPDR/Vanguard) —
    "VOX",   # 📡 Comunicazioni
    "XLB",   # 🧱 Materials
    "XLE",   # ⛽ Energy
    "XLI",   # 🛠️ Industrials
    "XLK",   # 🖥️ Technology
    "XLP",   # 🧃 Consumer Staples
    "XLU",   # 💡 Utilities
    "XLV",   # 🩺 Health Care
    "XLY",   # 🛍️ Consumer Discretionary

    # — Real estate —
    "IYR",   # 🏢 iShares U.S. Real Estate — REITs USA
    "VNQ",   # 🏠 Vanguard Real Estate — REITs USA broad

    # — Tech/Semiconductor —
    "QQQ",   # 💻 Invesco QQQ — Nasdaq-100 core tech
    "SMH",   # 🔧 VanEck Semiconductor — semiconduttori

    # — Leverage / Inverse (uso tattico, daily reset) —
    "PSQ",   # ⛔ ProShares Short QQQ — Nasdaq-100 −1x
    "QID",   # ⛔ ProShares UltraShort QQQ — Nasdaq-100 −2x
    "RWM",   # ⛔ ProShares Short Russell 2000 — −1x small cap
    "SPXU",  # ⛔ ProShares UltraPro Short S&P 500 — −3x
    "EDZ",   # ⛔ Direxion EM Bear 3x — mercati emergenti −3x

    "QLD",   # ⚡ ProShares Ultra QQQ — Nasdaq-100 +2x
    "TQQQ",  # ⚡ ProShares UltraPro QQQ — Nasdaq-100 +3x
    "SSO",   # ⚡ ProShares Ultra S&P 500 — +2x
    "SPXL",  # ⚡ Direxion Daily S&P 500 Bull — +3x
    "EDC",   # ⚡ Direxion EM Bull 3x — mercati emergenti +3x
    "SOXL",  # ⚡ Direxion Semiconductor Bull 3x — semiconduttori +3x
    "TECL",  # ⚡ Direxion Technology Bull 3x — tecnologia +3x

    # — Volatilità (VIX) —
    "SVXY",  # 🔻 ProShares Short VIX Short-Term — short VIX (−0.5x)
    "UVXY",  # ⚠️ ProShares Ultra VIX Short-Term — long VIX (+1.5x)
    "VIXY",  # 📈 ProShares VIX Short-Term Futures — long VIX (breve termine)
    "VIXM",  # 📈 ProShares VIX Mid-Term Futures — long VIX (medio termine)

    # — Low-Vol e Quality —
    "SPLV",  # 🧊 Invesco S&P 500 Low Volatility — low-vol large cap

    # — Altri (long/short equity, intl) —
    "FTLS",  # ♻️ First Trust Long/Short Equity — long/short US equity
]

# Mini portfolios
mini_commodities = [
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🥇 COMMODITIES – stesso della lista _vanguard
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "XAD5.MI",   # 🥇 Xtrackers Physical Gold ETC (EUR Hdg)              | DE000A1E0HR8 | .MI ✅ | 📅 2007
    "BRNT.MI",   # 🛢️  WisdomTree Brent Crude Oil ETC                    | GB00B0CTWC01 | .MI ✅ | 📅 2006
]


# My Curvo
greta_base_portfolio = {
    'SPY'  : 0.48,  # 🌍 Azionario globale (S&P 500 UCITS - Vanguard)
    'QQQ' : 0.17,  # 🏙️ Nasdaq 100 ESG (Lyxor)
    'GLD' : 0.15,  # 🟡 Oro fisico (GOLD.AS - Xetra Gold)
    'LYXC.DE' : 0.10,  # 📉 Titoli Stato Euro M/L termine (EM57 - Lyxor)
    'MTD.PA'  : 0.10   # 📈 Titoli Stato Euro Lungo termine (EM710 - Lyxor)
}

greta_base_tickers = list(greta_base_portfolio.keys())

greta_base_bitcoin_portfolio = {
    'SPY'  : 0.48,  # 🌍 Azionario globale (S&P 500 UCITS - Vanguard)
    'QQQ' : 0.17,  # 🏙️ Nasdaq 100 ESG (Lyxor)
    'Gld' : 0.15,  # 🟡 Oro fisico (GOLD.AS - Xetra Gold)
    'LYXC.DE' : 0.08,  # 📉 Titoli Stato Euro M/L termine (EM57 - Lyxor)
    'MTD.PA'  : 0.08,  # 📈 Titoli Stato Euro Lungo termine (EM710 - Lyxor)
    'XBTI.SW' : 0.04   # ₿ Bitcoin ETP (21Shares - XBTI.SW)
}
greta_base_bitcoin_tickers = list(greta_base_bitcoin_portfolio.keys())

greta_hy_portfolio = {
    'SPY'  : 0.48,  # 🌍 Azionario globale (S&P 500 UCITS - Vanguard)
    # 'NQSE.DE' : 0.17,  # 🏙️ Nasdaq 100 ESG (Lyxor)
    'QQQ': 0.17,     
    # 'GOLD.AS' : 0.15,  # 🟡 Oro fisico (GOLD.AS - Xetra Gold)
    'GLD' : 0.15,  # 🟡 Oro fisico (GOLD.AS - Xetra Gold)
    'LYXC.DE' : 0.10,  # 📉 Titoli Stato Euro M/L termine (EM57 - Lyxor)
    # 'EHYA.AS' : 0.10   # 💸 Obbligazioni High Yield Euro (iShares - EHYA)
    'IHYU.L' : 0.10     # 💸 Obbligazioni High Yield Euro (alternativa a EHYA con piu' storico)
}
greta_hy_tickers = list(greta_hy_portfolio.keys())

# greta_hy_portfolio_en1 = {
#     # 'VUAA.L'  : 0.35,  # 🌍 Azionario globale (S&P 500 UCITS - Vanguard)
#     'NQSE.DE' : 0.50,  # 🏙️ Nasdaq 100 ESG (Lyxor)
#     'GOLD.AS' : 0.20,  # 🟡 Oro fisico (GOLD.AS - Xetra Gold)
#     'LYXC.DE' : 0.15,  # 📉 Titoli Stato Euro M/L termine (EM57 - Lyxor)
#     'EHYA.AS' : 0.15   # 💸 Obbligazioni High Yield Euro (iShares - EHYA)
# }

greta_hy_bitcoin_portfolio = {
    # 'VUAA.L'  : 0.48,  # 🌍 Azionario globale (S&P 500 UCITS - Vanguard)
    'SPY'  : 0.48,  # 🌍 Azionario globale (S&P 500 UCITS - Vanguard)
    # 'NQSE.DE' : 0.17,  # 🏙️ Nasdaq 100 ESG (Lyxor)
    'QQQ': 0.17,     
    # 'GOLD.AS' : 0.13,  # 🟡 Oro fisico (GOLD.AS - Xetra Gold)
    'GLD' : 0.13,  # 🟡 Oro fisico (GOLD.AS - Xetra Gold)
    'LYXC.DE' : 0.08,  # 📉 Titoli Stato Euro M/L termine (EM57 - Lyxor)
    # 'EHYA.AS' : 0.10,   # 💸 Obbligazioni High Yield Euro (iShares - EHYA)
    'IHYU.L' : 0.08,     # 💸 Obbligazioni High Yield Euro (alternativa a EHYA con piu' storico)
    'XBTI.SW' : 0.06   # ₿ Bitcoin ETP (21Shares - XBTI.SW)
}
greta_hy_bitcoin_tickers = list(greta_hy_bitcoin_portfolio.keys())

global_benchmark = {
    "SP5A.MI": 0.40,  # 🌍 Azionario sviluppato (S&P 500 UCITS)
    "XMME.MI": 0.10,  # 🌏 Mercati emergenti equity
    "IBTM.MI": 0.15,  # 🏦 Obbligazionario investment grade globale (titoli di stato USA LT)
    "IHYG.MI": 0.10,  # 💸 Obbligazioni High Yield globali
    "XAD5.MI": 0.05,  # 🟡 Oro fisico
    "TRET.MI": 0.05,  # 🏠 Real estate globale
    "XLKS.MI": 0.05,  # 💻 Settore tecnologico (US Tech)
    "XLVS.MI": 0.05,  # 🏥 Settore healthcare (US Healthcare)
    "XLFS.MI": 0.05   # 🏛️ Settore finanziario (US Financials)
}

core2_benchmark = {
    "AGGH": 0.25,     # 🏦 Obbligazionario Governativo Globale (Euro)
    "HYLD.MI": 0.25,  # 💳 Obbligazionario High Yield Globale
    "SWDA.MI": 0.25,  # 🌍 Azionario Globale Paesi Sviluppati
    "SGLD.MI": 0.25,  # 🥇 Oro fisico
}



# =================================
# Indici
# =================================

#
# Definizione degli indici 
#

# Vedi funzione: extract_tickers_from_wikipedia(index)
# 'sp100','sp500','nasdaq100','ftsemib','dax','eurostoxx50','cac40','ibex35',
# 'nikkei','sse50','hangseng','nifty50','kospi200'








