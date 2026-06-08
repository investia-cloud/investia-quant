"""
medium_kstrat_agent/agent.py
────────────────────────────
Agente giornaliero che:
  1. Apre Medium con Playwright (browser reale headless) iniettando i cookie
  2. Scarica il testo degli articoli con Playwright
     - Fallback: legge PDF salvati localmente
  3. Invia il testo a Ollama (few-shot) per generare una K-Strategy
  4. Valida il codice Python prodotto
  5. Aggiunge una cella al Jupyter Notebook locale

Dipendenze:
    pip install playwright nbformat schedule beautifulsoup4 requests pypdf
    playwright install chromium

Ollama:
    ollama pull qwen2.5-coder:7b
"""

import ast
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import nbformat
import requests
import schedule
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
MEDIUM_FEED_URL  = "https://medium.com/me/following-feed/writers/74a837c27e96"
COOKIE_FILE      = Path(__file__).parent / "cookies.json"
STATE_FILE       = Path(__file__).parent / "processed_ids.json"
NOTEBOOK_FILE    = Path(__file__).parent / "strategies.ipynb"
PDF_FOLDER       = Path(__file__).parent / "pdf_articles"
# Cartella con file .py da convertire in K-Strategy
# Nomina i file come il titolo dell'articolo o con l'ID nell'URL
PY_FOLDER        = Path(__file__).parent / "py_articles"

# Anthropic API (usata quando LLM_PROVIDER = "anthropic")
ANTHROPIC_MODEL  = "claude-sonnet-4-20250514"

# Ollama (usata quando LLM_PROVIDER = "ollama")
# ── LLM Provider ──────────────────────────────────────────────────────────
# LLM_PROVIDER: "ollama" | "anthropic"
LLM_PROVIDER     = "ollama"

# Ollama
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5-coder:7b"
OLLAMA_TIMEOUT   = 3600          # 1 ora — il tempo non è un problema su CPU

# Anthropic (Claude)
_KEY_SH = Path(__file__).parent / "Claude-K-strategy_Key.sh"

def _load_anthropic_key() -> str:
    """Legge ANTHROPIC_API_KEY da env, oppure dal file .sh se non impostata."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key and _KEY_SH.exists():
        m = re.search(r"ANTHROPIC_API_KEY=['\"]?([^'\"]+)['\"]?", _KEY_SH.read_text())
        if m:
            key = m.group(1).strip()
    return key

ANTHROPIC_API_KEY = _load_anthropic_key()
ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"
ANTHROPIC_TIMEOUT = 120

class _TransientAPIError(Exception):
    """Eccezione lanciata quando l'API Anthropic è temporaneamente irraggiungibile (429/502/503/529)."""

DAILY_RUN_TIME   = "08:00"
MAX_PER_RUN      = 5
FEED_WAIT_SEC    = 6
ARTICLE_WAIT_SEC = 5

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "agent.log"),
    ],
)
log = logging.getLogger("kstrat-agent")

# ─────────────────────────────────────────────
# SYSTEM PROMPT  (con esempio few-shot)
# ─────────────────────────────────────────────
KSTRAT_FEW_SHOT_EXAMPLE = '''
############################
# Strategy dbma_matrix
############################

def ind_dbma_matrix_dbma(df: pd.DataFrame,
                         ma_period: int = 20,
                         sd_near: float = 0.5,
                         sd_far: float = 1.0,
                         compress_lookback: int = 20):
    close = df['Close']
    ma = close.rolling(ma_period, min_periods=1).mean()
    stdev = close.rolling(ma_period, min_periods=1).std(ddof=0)
    up_near = ma + sd_near * stdev
    lo_near = ma - sd_near * stdev
    up_far  = ma + sd_far  * stdev
    lo_far  = ma - sd_far  * stdev
    bw_near = (up_near - lo_near) / ma.replace(0, np.nan)
    bw_med  = bw_near.rolling(compress_lookback, min_periods=1).median()
    compress = (bw_near <= bw_med)
    return ma, up_near, lo_near, up_far, lo_far, compress


def ind_dbma_matrix_series(df: pd.DataFrame,
                           ma: pd.Series,
                           mom_period: int = 10,
                           smooth: int = 5) -> pd.Series:
    close = df['Close']
    ret = close.pct_change()
    vol = ret.rolling(mom_period, min_periods=1).std(ddof=0).replace(0, np.nan)
    mom = close.pct_change(mom_period)
    st_mom = close.rolling(mom_period, min_periods=1).std(ddof=0).replace(0, np.nan)
    zdist = (close - ma) / st_mom
    ms_raw = 0.6 * zdist + 0.4 * (mom / vol)
    ms = ms_raw.ewm(span=smooth, adjust=False, min_periods=1).mean()
    return ms


strategy_dbma_matrix_param_ranges = {
    'ma_range'                : range(10, 31, 5),
    'sd_near_range'           : range(5, 8, 1),
    'sd_far_range'            : range(10, 13, 1),
    'ms_mom_range'            : range(5, 21, 5),
    'ms_smooth_range'         : range(3, 11, 2),
    'compress_lookback_range' : range(10, 31, 10),
}


def strategy_dbma_matrix(data: pd.DataFrame, params: dict, year: int | None = None):
    ma_p        = params.get('ma_range')
    sd_near     = params.get('sd_near_range') / 10.0
    sd_far      = params.get('sd_far_range')  / 10.0
    ms_mom_p    = params.get('ms_mom_range')
    ms_smooth_p = params.get('ms_smooth_range')
    compr_lb_p  = params.get('compress_lookback_range')

    df = data.copy()

    ma, up_near, lo_near, up_far, lo_far, compress = ind_dbma_matrix_dbma(
        df, ma_period=ma_p, sd_near=sd_near, sd_far=sd_far,
        compress_lookback=compr_lb_p
    )
    ms = ind_dbma_matrix_series(df, ma=ma, mom_period=ms_mom_p, smooth=ms_smooth_p)

    df['MA']             = ma
    df['BB_upper_near']  = up_near
    df['BB_lower_near']  = lo_near
    df['BB_upper_far']   = up_far
    df['BB_lower_far']   = lo_far
    df['DBMA_Compress']  = compress
    df['MS']             = ms

    if year is not None:
        df = df[df.index.year == int(year)]

    trend_long  = df['Close'] > df['MA']
    trend_short = df['Close'] < df['MA']
    pullback_long  = (df['Low']  <= df['MA']) & (df['Low']  >= df['BB_lower_far'])
    pullback_short = (df['High'] >= df['MA']) & (df['High'] <= df['BB_upper_far'])
    comp       = df['DBMA_Compress']
    flip_up    = (df['MS'] > 0) & (df['MS'].shift(1) <= 0)
    flip_down  = (df['MS'] < 0) & (df['MS'].shift(1) >= 0)
    bull_close = df['Close'] > df['Close'].shift(1)
    bear_close = df['Close'] < df['Close'].shift(1)

    entries_long  = trend_long  & comp & pullback_long  & flip_up   & bull_close
    entries_short = trend_short & comp & pullback_short & flip_down & bear_close
    entries = entries_long | entries_short

    exits_long  = (df['Close'] >= df['BB_upper_near']) | flip_down
    exits_short = (df['Close'] <= df['BB_lower_near']) | flip_up
    exits = (exits_long & trend_long) | (exits_short & trend_short)

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits
'''

KSTRAT_SYSTEM_PROMPT = f"""\
Sei un generatore di strategie per il framework K-Strategies basato su Walk-Forward Optimization (WFO).
Il tuo compito e trasformare un articolo di trading in UNA strategia Python conforme al formato richiesto.

REGOLE ASSOLUTE — nessuna eccezione:
1. Rispondi SOLO con codice Python puro. Zero testo, zero spiegazioni, zero markdown.
2. Usa SOLO pandas e numpy. VIETATO importare qualsiasi altra libreria.
3. Il DataFrame ha sempre le colonne: Open, High, Low, Close, Volume.
4. Implementa tutti gli indicatori manualmente (no TA-Lib, no pandas_ta, no vbt).
5. I segnali devono essere booleani, shiftati di 1 barra anti look-ahead:
       shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
       shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
6. Il filtro year va applicato SOLO dopo il calcolo di tutti gli indicatori.
7. SKIP_ARTICLE e' consentito SOLO se l'articolo e' completamente generico (notizie, opinioni)
   e non menziona NESSUN indicatore tecnico o regola di trading.
   Se l'articolo descrive anche solo UN indicatore con una logica entry/exit,
   DEVI generare la strategia — anche se il codice non e' presente nell'articolo.
   In quel caso implementa gli indicatori descritti nel testo usando pandas/numpy.
   Quando usi SKIP_ARTICLE, scrivi SEMPRE una riga di motivazione:
   SKIP_ARTICLE: <motivo breve in italiano, max 120 caratteri>

REGOLE DI PERFORMANCE — obbligatorie per garantire velocita' nel WFO:
8. Mai usare rolling.apply(raw=False): usa SEMPRE raw=True con una lambda numpy pura.
   SBAGLIATO:  s.rolling(n).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
   CORRETTO:   s.rolling(n).apply(lambda x: np.sum(x <= x[-1]) / len(x), raw=True)
9. Mai creare oggetti pandas (Series, DataFrame) dentro una lambda di rolling/apply.
   Usa esclusivamente operazioni numpy sugli array (x[-1], np.sum, np.mean, np.std, ecc.).
10. Per il True Range e operazioni row-wise su piu' colonne, usa np.maximum invece di pd.concat:
    SBAGLIATO:  tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    CORRETTO:   tr = np.maximum(np.maximum(hl.values, hpc.values), lpc.values)
11. Per divisioni che possono avere denominatore zero, usa il pattern safe-denominator:
    safe_den = np.where(den != 0, den, 1.0)
    result   = np.where(den != 0, num / safe_den, 0.0)
    (evita RuntimeWarning e NaN che degradano le performance del WFO)
12. Mai usare loop Python con .iloc per calcoli ricorsivi (es. EMA, VIDYA, Wilder smoothing).
    Converti i Series in array numpy con .values e usa indici interi nel loop:
    SBAGLIATO:  for i in range(1, len(s)): out.iloc[i] = f(s.iloc[i], out.iloc[i-1])
    CORRETTO:   arr = s.values; out = np.empty(len(arr)); out[0] = arr[0]
                for i in range(1, len(arr)): out[i] = f(arr[i], out[i-1])
                result = pd.Series(out, index=s.index)
13. La griglia parametri NON deve superare 1000 combinazioni totali (prodotto dei len di tutti i range).
    Con 7 finestre WFO questo corrisponde a ~7000 step — oltre si allunga inutilmente il calcolo.
    Regole pratiche per restare sotto soglia:
      - Usa al massimo 6 parametri
      - Ogni range deve avere 2-4 valori (usa step ampi: range(10,21,5) non range(10,21,1))
      - Verifica mentalmente: se hai 6 parametri con 4 valori = 4^6 = 4096 → troppi, riduci
      - Preferisci 3 valori per parametro: 3^6 = 729, 3^5 = 243 — entrambi accettabili
    SBAGLIATO:  'period_range': range(10, 25)        # 15 valori
    CORRETTO:   'period_range': range(10, 25, 5)     # 3 valori: [10, 15, 20]

STRUTTURA OBBLIGATORIA (rispetta esattamente questa struttura, compreso il naming):

############################
# Strategy <nome_breve_lowercase>
############################

def ind_<nome>_<indicatore>(df: pd.DataFrame, ...) -> ...:
    # implementazione con pandas/numpy

strategy_<nome>_param_ranges = {{
    '<param>_range': range(...),
    ...
}}

def strategy_<nome>(data: pd.DataFrame, params: dict, year: int | None = None):
    # leggi parametri con params.get(...)
    df = data.copy()
    # calcola indicatori sull intero df
    # if year is not None: df = df[df.index.year == int(year)]
    # definisci entries e exits come pd.Series booleane
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits

ESEMPIO COMPLETO (segui esattamente questo stile):
{KSTRAT_FEW_SHOT_EXAMPLE}

OUTPUT: solo codice Python. Nient'altro.
"""


# ═══════════════════════════════════════════════════════════════
# MODULO 1 – COOKIES
# ═══════════════════════════════════════════════════════════════

def _map_samesite(value: str) -> str:
    return {"no_restriction": "None", "lax": "Lax",
            "strict": "Strict", "unspecified": "Lax"}.get(value.lower(), "Lax")


def load_cookies_for_playwright(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pw_cookies = []
    for c in raw:
        domain = c.get("domain", ".medium.com")
        if not domain.startswith(".") and not c.get("hostOnly", False):
            domain = "." + domain
        cookie: dict = {
            "name":     c["name"],
            "value":    c["value"],
            "domain":   domain,
            "path":     c.get("path", "/"),
            "secure":   c.get("secure", True),
            "httpOnly": c.get("httpOnly", False),
            "sameSite": _map_samesite(c.get("sameSite", "Lax")),
        }
        if "expirationDate" in c:
            cookie["expires"] = int(c["expirationDate"])
        pw_cookies.append(cookie)
    return pw_cookies


# ═══════════════════════════════════════════════════════════════
# MODULO 2 – BROWSER HELPER
# ═══════════════════════════════════════════════════════════════

def _make_browser(pw, pw_cookies):
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--ignore-certificate-errors",
        ],
    )
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="it-IT",
    )
    context.add_cookies(pw_cookies)
    page = context.new_page()
    page.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
    )
    return browser, context, page


# ═══════════════════════════════════════════════════════════════
# MODULO 3 – FETCH FEED
# ═══════════════════════════════════════════════════════════════

def fetch_feed_articles(cookie_path: Path, feed_url: str) -> list[dict]:
    pw_cookies = load_cookies_for_playwright(cookie_path)
    with sync_playwright() as pw:
        browser, context, page = _make_browser(pw, pw_cookies)
        log.info("Playwright: navigazione al feed...")
        try:
            page.goto(feed_url, wait_until="domcontentloaded", timeout=30_000)
        except PWTimeout:
            log.warning("Timeout feed, leggo DOM parziale")
        log.info(f"Attesa {FEED_WAIT_SEC}s lazy-load...")
        time.sleep(FEED_WAIT_SEC)
        page.evaluate("window.scrollBy(0, 1200)")
        time.sleep(2)
        html = page.content()
        browser.close()

    articles = _parse_state_json(html)
    if articles:
        log.info(f"State JSON: {len(articles)} articoli")
        return articles

    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].split("?")[0]
        if not re.search(r"-[0-9a-f]{8,}$", href):
            continue
        article_id = href.rstrip("/").split("-")[-1]
        if article_id in seen or len(article_id) < 8:
            continue
        seen.add(article_id)
        if href.startswith("/"):
            href = "https://medium.com" + href
        title_tag = a_tag.find(["h1", "h2", "h3"])
        title = title_tag.get_text(strip=True) if title_tag else a_tag.get_text(strip=True)[:80]
        if title:
            articles.append({"id": article_id, "url": href, "title": title})
    log.info(f"HTML href: {len(articles)} articoli")
    return articles


def _parse_state_json(html: str) -> list[dict]:
    articles = []
    for var in ("__APOLLO_STATE__", "__PRELOADED_STATE__"):
        match = re.search(
            rf'window\.{var}\s*=\s*(\{{.*?\}});?\s*</script>', html, re.DOTALL
        )
        if not match:
            continue
        try:
            state = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        for val in state.values():
            if not isinstance(val, dict) or val.get("__typename") != "Post":
                continue
            pid, title = val.get("id", ""), val.get("title", "")
            slug = val.get("uniqueSlug", "")
            url = val.get("mediumUrl") or (
                f"https://medium.com/p/{slug}" if slug else
                f"https://medium.com/p/{pid}"  if pid  else None
            )
            if pid and url and title:
                articles.append({"id": pid, "url": url, "title": title})
        if articles:
            break
    return articles


# ═══════════════════════════════════════════════════════════════
# MODULO 4 – FETCH TESTO ARTICOLO
# ═══════════════════════════════════════════════════════════════

def fetch_article_text(article: dict, cookie_path: Path) -> str:
    # 1. PDF locale
    text = _text_from_pdf(article["id"], article["title"])
    if text:
        log.info(f"  Testo da PDF ({len(text)} chars)")
        return text
    # 2. Playwright (salva automaticamente PDF in pdf_articles/)
    log.info("  Playwright: fetch articolo e salvataggio PDF...")
    text = _text_from_playwright(
        article["url"], cookie_path,
        article_title=article.get("title", ""),
        article_id=article.get("id", ""),
    )
    if text:
        log.info(f"  Testo da Playwright ({len(text)} chars)")
    return text


# Parole comuni da ignorare nel matching PDF (presenti in tutti gli articoli Kryptera)
_PDF_STOPWORDS = {
    "strategy", "trading", "medium", "kryptera", "backtest", "with",
    "from", "market", "price", "using", "this", "that", "when", "what",
    "how", "the", "and", "for", "into", "over", "like", "stop", "start",
    "more", "some", "than", "their", "about"
}


def _pdf_match_score(pdf_name: str, article_title: str, article_id: str) -> int:
    """
    Punteggio di corrispondenza tra nome PDF e titolo articolo.
    Ignora le stopword comuni a tutti gli articoli Kryptera.
    Richiede che le parole DISTINTIVE del titolo siano nel nome file.
    """
    name_clean  = re.sub(r"[^a-z0-9 ]", " ", pdf_name.lower())
    title_clean = re.sub(r"[^a-z0-9 ]", " ", article_title.lower())
    name_words  = set(name_clean.split())

    # ID nell'URL → match certo, nessuna soglia
    if article_id.lower() in pdf_name.lower():
        return 10000

    # Parole distintive del titolo (escluse stopword, lunghezza >= 4)
    distinctive = [
        w for w in title_clean.split()
        if len(w) >= 4 and w not in _PDF_STOPWORDS
    ]

    if not distinctive:
        return 0

    matched = sum(1 for w in distinctive if w in name_words)
    # Richiedi che ALMENO il 60% delle parole distintive siano presenti
    ratio = matched / len(distinctive)
    if ratio < 0.6:
        return 0

    # Punteggio = somma lunghezze parole matchate (parole rare/lunghe pesano di più)
    return sum(len(w) for w in distinctive if w in name_words)


def _text_from_pdf(article_id: str, article_title: str) -> str:
    """
    Cerca in pdf_articles/ il PDF il cui nome corrisponde al titolo dell'articolo.
    Salva il PDF con il nome proposto da Chrome: l'agente lo abbina automaticamente.
    Il matching richiede che almeno il 60% delle parole distintive del titolo
    siano presenti nel nome file — evita falsi positivi su parole comuni.
    """
    if not PDF_FOLDER.exists():
        return ""
    try:
        from pypdf import PdfReader
    except ImportError:
        log.debug("pypdf non installato — skip PDF")
        return ""

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        return ""

    scored = [
        (p, _pdf_match_score(p.name, article_title, article_id))
        for p in pdf_files
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_file, best_score = scored[0]

    if best_score == 0:
        return ""

    log.info(f"  PDF trovato (score={best_score}): {best_file.name}")
    try:
        reader = PdfReader(str(best_file))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        return re.sub(r"\n{3,}", "\n\n", text)[:14_000]
    except Exception as e:
        log.warning(f"  Errore lettura PDF: {e}")
        return ""


def _sanitize_filename(title: str) -> str:
    """Converte il titolo in un nome file sicuro."""
    name = re.sub(r"[^\w\s-]", "", title).strip()
    name = re.sub(r"[\s]+", "_", name)
    return name[:80]


def _text_from_py(article_id: str, article_title: str) -> str:
    """
    Cerca in py_articles/ un file .py il cui nome corrisponde all'articolo.
    Il testo Python viene preceduto da un'intestazione che spiega all'LLM
    cosa deve fare: convertire il codice nel formato K-Strategy.
    """
    if not PY_FOLDER.exists():
        return ""

    py_files = list(PY_FOLDER.glob("*.py"))
    if not py_files:
        return ""

    # Stesso sistema di scoring dei PDF
    scored = [
        (p, _pdf_match_score(p.name, article_title, article_id))
        for p in py_files
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_file, best_score = scored[0]

    if best_score == 0:
        return ""

    log.info(f"  File .py trovato (score={best_score}): {best_file.name}")
    try:
        py_code = best_file.read_text(encoding="utf-8")
    except Exception as e:
        log.warning(f"  Errore lettura .py: {e}")
        return ""

    # Prependi intestazione che guida il modello nella conversione
    header = (
        "# === CODICE PYTHON DA CONVERTIRE IN K-STRATEGY ===\n"
        "# Il codice seguente e' una strategia scritta in un altro framework.\n"
        "# Devi adattarlo al formato K-Strategy (ind_, param_ranges, strategy_).\n"
        "# Mantieni la logica di entry/exit originale, riscrivi solo la struttura.\n"
        "# ================================================\n\n"
    )
    full_text = header + py_code
    return full_text[:14_000]


def _text_from_playwright(url: str, cookie_path: Path,
                           article_title: str = "",
                           article_id: str = "") -> str:
    """
    Apre l'articolo con Playwright, lo salva come PDF in pdf_articles/
    (equivalente a Chrome: Stampa → Salva come PDF), poi estrae il testo dal PDF.
    """
    pw_cookies = load_cookies_for_playwright(cookie_path)

    # Nome file PDF: titolo_id.pdf
    safe_title = _sanitize_filename(article_title) if article_title else article_id
    pdf_filename = f"{safe_title}_{article_id}.pdf" if article_id else f"{safe_title}.pdf"
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_FOLDER / pdf_filename

    try:
        with sync_playwright() as pw:
            browser, context, page = _make_browser(pw, pw_cookies)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except PWTimeout:
                log.warning("  Timeout articolo, leggo DOM parziale")

            # Scroll progressivo per caricare tutto il contenuto lazy
            for scroll_y in [1500, 3000, 5000, 8000]:
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                time.sleep(1)
            time.sleep(ARTICLE_WAIT_SEC)

            # Salva come PDF (equivalente a Stampa → Salva come PDF)
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=False,
                margin={"top": "1cm", "bottom": "1cm",
                        "left": "1cm", "right": "1cm"},
            )
            browser.close()

        log.info(f"  PDF salvato: {pdf_filename}")

    except Exception as e:
        log.warning(f"  Playwright errore (PDF): {e}")
        # Fallback: leggi HTML senza salvare PDF
        return _text_from_playwright_html(url, cookie_path)

    # Estrai testo dal PDF appena salvato
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:14_000]
    except Exception as e:
        log.warning(f"  Errore lettura PDF salvato: {e}")
        return ""


def _text_from_playwright_html(url: str, cookie_path: Path) -> str:
    """Fallback: estrae testo dall'HTML senza salvare PDF."""
    pw_cookies = load_cookies_for_playwright(cookie_path)
    html = ""
    try:
        with sync_playwright() as pw:
            browser, context, page = _make_browser(pw, pw_cookies)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except PWTimeout:
                pass
            for scroll_y in [1500, 3000, 5000, 8000]:
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                time.sleep(1)
            time.sleep(ARTICLE_WAIT_SEC)
            html = page.content()
            browser.close()
    except Exception as e:
        log.warning(f"  Playwright HTML errore: {e}")
        return ""

    soup = BeautifulSoup(html, "html.parser")
    block = (
        soup.find("article")
        or soup.find("div", {"data-testid": "post-content"})
        or soup.find("section")
        or soup.body
    )
    if not block:
        return ""
    for tag in block.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = block.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text)[:14_000]


# ═══════════════════════════════════════════════════════════════
# MODULO 5 – STATO
# ═══════════════════════════════════════════════════════════════

def load_state(path: Path) -> set[str]:
    if path.exists():
        with open(path, "r") as f:
            return set(json.load(f))
    return set()


def save_state(path: Path, ids: set[str]) -> None:
    with open(path, "w") as f:
        json.dump(sorted(ids), f, indent=2)


# ═══════════════════════════════════════════════════════════════
# MODULO 6 – GENERAZIONE STRATEGIA (Ollama)
# ═══════════════════════════════════════════════════════════════


# Sezioni da scartare sempre (rumore nei PDF di Medium)
_NOISE_PATTERNS = [
    r"member.only story", r"more from kryptera", r"recommended from medium",
    r"follow me on", r"written by", r"see all from", r"no responses yet",
    r"this article is not investment advice", r"open in app",
    r"clicca qui", r"acquistare", r"disclaimer",
    r"\d+ followers", r"\d+ following", r"add to cart", r"add to wishlist",
]

def _extract_relevant_text(text: str, max_chars: int = 3500) -> str:
    """
    Estrae SOLO le parti utili per generare la K-Strategy:
      - Codice Python (righe con def, return, indentazione, operatori pandas)
      - Frasi con keyword entry/exit/indicator
    Scarta tutto il resto (header Medium, footer, disclaimer, pubblicità).
    Obiettivo: mandare a Ollama meno di 3500 chars densi invece di 5000 rumorosi.
    """
    import re as _re

    # Rimuovi sezioni rumorose (case-insensitive)
    lines = text.split("\n")
    cleaned = []
    skip_block = False
    for line in lines:
        low = line.lower().strip()
        # Attiva skip se troviamo una sezione rumorosa
        if any(_re.search(p, low) for p in _NOISE_PATTERNS):
            skip_block = True
        # Disattiva skip alla prossima riga di codice
        if skip_block and ("def " in line or "df[" in line or "return " in line):
            skip_block = False
        if not skip_block:
            cleaned.append(line)
    text = "\n".join(cleaned)

    # Classifica righe
    code_kw = {
        "def ", "return ", "df[", "df.", ".rolling", ".ewm", ".shift",
        ".clip", ".diff", ".sum(", ".mean(", ".std(", ".ewm(",
        "pct_change", "fillna", "astype", "append", "pd.Series",
        "np.", "range(", "params.get", "param_ranges",
        "entries", "exits", "entry", "exit",
    }
    signal_kw = {
        "entry", "exit", "signal", "oversold", "overbought", "cross",
        "above", "below", "condition", "indicator", "trigger",
        "ema", "rsi", "macd", "atr", "cmo", "stc", "bollinger",
        "momentum", "reversion", "breakout", "period",
    }

    code_lines, signal_lines = [], []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(k in line for k in code_kw):
            code_lines.append(line)
        elif any(k in line.lower() for k in signal_kw):
            signal_lines.append(line)

    # Assembla: codice prima (più importante), poi frasi con keyword
    result = "\n".join(code_lines)
    if len(result) < max_chars:
        result += "\n\n" + "\n".join(signal_lines)

    return result[:max_chars]

def generate_strategy(article_text: str, article_title: str) -> str | None:
    """Genera la K-Strategy usando il backend LLM configurato."""
    if LLM_PROVIDER == "anthropic":
        return _generate_anthropic(article_text, article_title)
    else:
        return _generate_ollama(article_text, article_title)


def _generate_anthropic(article_text: str, article_title: str) -> str | None:
    """Genera la K-Strategy tramite API Anthropic (Claude)."""
    # Anthropic ha contesto ampio: si manda il testo grezzo troncato,
    # senza il filtro aggressivo di _extract_relevant_text (pensato per Ollama).
    relevant_text = re.sub(r"\n{3,}", "\n\n", article_text).strip()[:12_000]
    log.info(f"  Testo ridotto: {len(article_text)} → {len(relevant_text)} chars")

    user_message = (
        f"Titolo: {article_title}\n\n"
        f"Testo articolo:\n{relevant_text}\n\n"
        "Se nell'articolo trovi codice Python, usalo come base adattandolo al formato K-Strategy.\n"
        "Se non c'e' codice, implementa tu gli indicatori descritti nel testo usando pandas/numpy.\n"
        "Output: SOLO codice Python, nient'altro."
    )

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "system": KSTRAT_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }

    # Retry con backoff su errori transitori (529 overloaded, 529, 503, 502)
    _TRANSIENT_CODES = {429, 502, 503, 529}
    last_exc = None
    for attempt in range(1, 4):  # max 3 tentativi
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", ""),
                    "anthropic-version": "2023-06-01",
                },
                timeout=120,
            )
            if resp.status_code in _TRANSIENT_CODES:
                wait = 10 * attempt
                log.warning(f"  API {resp.status_code} (overloaded/rate-limit), retry {attempt}/3 tra {wait}s...")
                time.sleep(wait)
                last_exc = requests.HTTPError(response=resp)
                continue
            resp.raise_for_status()
            result = resp.json()["content"][0]["text"].strip()
            break
        except requests.HTTPError as e:
            last_exc = e
            if e.response is not None and e.response.status_code not in _TRANSIENT_CODES:
                log.error(f"  Errore API Anthropic: {e}")
                return None
            wait = 10 * attempt
            log.warning(f"  Errore transitorio {e}, retry {attempt}/3 tra {wait}s...")
            time.sleep(wait)
        except Exception as e:
            log.error(f"  Errore API Anthropic: {e}")
            return None
    else:
        log.error(f"  API Anthropic irraggiungibile dopo 3 tentativi: {last_exc}")
        raise _TransientAPIError(str(last_exc))

    if result.upper().startswith("SKIP_ARTICLE"):
        log.warning(f"  Articolo saltato: {result[:200]}")
        return None

    result = re.sub(r"^```python\s*", "", result, flags=re.MULTILINE)
    result = re.sub(r"^```\s*",       "", result, flags=re.MULTILINE)
    result = re.sub(r"\s*```$",       "", result)
    return result.strip()


def _generate_ollama(article_text: str, article_title: str) -> str | None:
    """Genera la K-Strategy tramite Ollama locale."""
    relevant_text = _extract_relevant_text(article_text, max_chars=3500)
    log.info(f"  Testo ridotto: {len(article_text)} → {len(relevant_text)} chars")

    if len(relevant_text.strip()) < 50:
        log.warning(
            "  Nessun codice Python né keyword di trading trovati nell'articolo. "
            "Ollama non può procedere senza contenuto strutturato. "
            "Suggerimento: usa LLM_PROVIDER='anthropic' per articoli solo descrittivi."
        )
        return None

    user_message = (
        f"Titolo: {article_title}\n\n"
        f"Testo articolo:\n{relevant_text}\n\n"
        "Se nell'articolo trovi codice Python, usalo come base adattandolo al formato K-Strategy.\n"
        "Se non c'e' codice, implementa tu gli indicatori descritti nel testo usando pandas/numpy.\n"
        "Output: SOLO codice Python, nient'altro."
    )
    payload = {
        "model":  OLLAMA_MODEL,
        "system": KSTRAT_SYSTEM_PROMPT,
        "prompt": user_message,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048, "num_ctx": 8192},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
    except Exception as e:
        log.error(f"Errore Ollama: {e}")
        return None

    if result.upper().startswith("SKIP_ARTICLE"):
        log.warning(f"  Articolo saltato: {result[:200]}")
        return None

    result = re.sub(r"^```python\s*", "", result, flags=re.MULTILINE)
    result = re.sub(r"^```\s*",       "", result, flags=re.MULTILINE)
    result = re.sub(r"\s*```$",       "", result)
    return result.strip()



# ═══════════════════════════════════════════════════════════════
# MODULO 7 – VALIDAZIONE
# ═══════════════════════════════════════════════════════════════

def validate_python(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    checks = [
        ("def strategy_",    "Manca def strategy_<nome>"),
        ("_param_ranges",    "Manca _param_ranges"),
        ("shifted_entries",  "Manca shifted_entries"),
        ("shifted_exits",    "Manca shifted_exits"),
        ("def ind_",         "Manca almeno un indicatore ind_<nome>"),
    ]
    for token, msg in checks:
        if token not in code:
            return False, msg

    # Controlla import vietati
    forbidden = ["import vbt", "import vectorbt", "import talib",
                 "import pandas_ta", "import plotly", "import yfinance"]
    for f in forbidden:
        if f in code:
            return False, f"Import vietato: {f}"

    # Controlla naming: ogni ind_ deve avere il prefisso della strategia
    import re as _re
    strategy_names = _re.findall(r"def strategy_([a-z0-9_]+)\s*\(", code)
    ind_names      = _re.findall(r"def ind_([a-z0-9_]+)\s*\(", code)
    if strategy_names and ind_names:
        prefix = strategy_names[0]
        bad_inds = [n for n in ind_names if not n.startswith(prefix)]
        if bad_inds:
            return False, (
                f"Naming errato: ind_{bad_inds[0]} non ha il prefisso "
                f"della strategia (ind_{prefix}_...)"
            )

    return True, "OK"


# ═══════════════════════════════════════════════════════════════
# MODULO 8 – NOTEBOOK
# ═══════════════════════════════════════════════════════════════

def load_or_create_notebook(path: Path) -> nbformat.NotebookNode:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return nbformat.read(f, as_version=4)
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# K-Strategies — Generato da Medium Agent\n"
            "> Aggiornato automaticamente ogni giorno."
        ),
        nbformat.v4.new_code_cell("import pandas as pd\nimport numpy as np\n"),
    ]
    return nb


def append_strategy_to_notebook(nb, code, title, url, generated_at):
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"## {title}\n- **Fonte**: [{url}]({url})\n- **Generata**: {generated_at}"
    ))
    nb.cells.append(nbformat.v4.new_code_cell(code))
    return nb


def save_notebook(nb, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    log.info(f"Notebook salvato: {path}")


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATORE
# ═══════════════════════════════════════════════════════════════

def run_agent() -> None:
    log.info("═══ Avvio run agente ═══")

    if not COOKIE_FILE.exists():
        log.error(f"File cookies non trovato: {COOKIE_FILE}")
        return

    processed_ids = load_state(STATE_FILE)

    try:
        feed_articles = fetch_feed_articles(COOKIE_FILE, MEDIUM_FEED_URL)
    except Exception as e:
        log.error(f"Errore scraping feed: {e}")
        return

    new_articles = [a for a in feed_articles if a["id"] not in processed_ids]
    log.info(f"Articoli nuovi: {len(new_articles)} (max {MAX_PER_RUN})")
    new_articles = new_articles[:MAX_PER_RUN]

    if not new_articles:
        log.info("Nessun articolo nuovo. Fine run.")
        return

    nb = load_or_create_notebook(NOTEBOOK_FILE)

    for article in new_articles:
        log.info(f"→ {article['title'][:60]}")

        text = fetch_article_text(article, COOKIE_FILE)
        if len(text) < 200:
            log.warning(f"  Testo insufficiente ({len(text)} chars), skip.")
            processed_ids.add(article["id"])
            continue

        log.info(f"  Chiamata LLM ({LLM_PROVIDER})...")
        try:
            code = generate_strategy(text, article["title"])
        except _TransientAPIError:
            log.warning("  Errore transitorio API — articolo non marcato come processato, verrà riprovato.")
            continue
        if code is None:
            processed_ids.add(article["id"])
            continue

        ok, reason = validate_python(code)

        # Retry una volta se il problema è solo il naming
        if not ok and "Naming errato" in reason and code is not None:
            log.warning(f"  Validazione fallita ({reason}) — retry con correzione naming...")
            strategy_name_match = re.search(r"def strategy_([a-z0-9_]+)\s*\(", code)
            if strategy_name_match:
                strat_name = strategy_name_match.group(1)
                retry_hint = (
                    f"Il codice che hai generato ha indicatori con nomi generici (es. ind_ema, ind_rsi).\n"
                    f"DEVI rinominarli aggiungendo il prefisso della strategia '{strat_name}'.\n"
                    f"Esempio: ind_ema → ind_{strat_name}_ema, ind_rsi → ind_{strat_name}_rsi\n"
                    f"Riscrivi il codice completo con i nomi corretti. SOLO codice Python, nient'altro.\n\n"
                    f"Codice da correggere:\n{code}"
                )
                if LLM_PROVIDER == "anthropic":
                    payload_retry = {
                        "model":      ANTHROPIC_MODEL,
                        "max_tokens": 4096,
                        "system":     KSTRAT_SYSTEM_PROMPT,
                        "messages":   [{"role": "user", "content": retry_hint}],
                    }
                else:
                    payload_retry = {
                        "model":  OLLAMA_MODEL,
                        "system": KSTRAT_SYSTEM_PROMPT,
                        "prompt": retry_hint,
                        "stream": False,
                        "options": {"temperature": 0.05, "num_predict": 2048, "num_ctx": 8192},
                    }
                try:
                    if LLM_PROVIDER == "anthropic":
                        payload_retry["messages"] = [{"role": "user", "content": retry_hint}]
                        payload_retry.pop("prompt", None)
                        payload_retry.pop("stream", None)
                        payload_retry.pop("options", None)
                        payload_retry["model"]      = ANTHROPIC_MODEL
                        payload_retry["max_tokens"] = 4096
                        resp2 = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            json=payload_retry,
                            headers={
                                "Content-Type":      "application/json",
                                "x-api-key":         ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", ""),
                                "anthropic-version": "2023-06-01",
                            },
                            timeout=120,
                        )
                        resp2.raise_for_status()
                        code2 = resp2.json()["content"][0]["text"].strip()
                    else:
                        resp2 = requests.post(OLLAMA_URL, json=payload_retry, timeout=OLLAMA_TIMEOUT)
                        resp2.raise_for_status()
                        code2 = resp2.json().get("response", "").strip()
                    code2 = re.sub(r"^```python\s*", "", code2, flags=re.MULTILINE)
                    code2 = re.sub(r"^```\s*",       "", code2, flags=re.MULTILINE)
                    code2 = re.sub(r"\s*```$",       "", code2)
                    ok2, reason2 = validate_python(code2.strip())
                    if ok2:
                        log.info("  Retry naming: OK")
                        code = code2.strip()
                        ok, reason = ok2, reason2
                    else:
                        log.warning(f"  Retry naming fallito ancora: {reason2}")
                except Exception as e:
                    log.warning(f"  Errore retry: {e}")

        if not ok:
            log.warning(f"  Validazione fallita: {reason}")
            code = f"# VALIDAZIONE FALLITA: {reason}\n\n" + code

        strat_name_m = re.search(r"def strategy_([a-z0-9_]+)\s*\(", code)
        strat_label  = f"strategy_{strat_name_m.group(1)}" if strat_name_m else "?"
        status_label = "⚠️  SALVATA CON ERRORI" if not ok else "✅ OK"
        log.info(f"  {status_label}  →  {strat_label}")
        nb = append_strategy_to_notebook(
            nb, code, article["title"], article["url"],
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        processed_ids.add(article["id"])
        time.sleep(2)

    save_notebook(nb, NOTEBOOK_FILE)
    save_state(STATE_FILE, processed_ids)
    log.info("═══ Run completata ═══")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Parametro inline: --max N  sovrascrive MAX_PER_RUN
    if "--max" in sys.argv:
        try:
            idx = sys.argv.index("--max")
            MAX_PER_RUN = int(sys.argv[idx + 1])
            log.info(f"MAX_PER_RUN sovrascritto a {MAX_PER_RUN}")
        except (IndexError, ValueError):
            log.error("Uso: python agent.py --max <numero>")
            sys.exit(1)

    # Parametro inline: --llm ollama|anthropic  sovrascrive LLM_PROVIDER
    if "--llm" in sys.argv:
        try:
            idx = sys.argv.index("--llm")
            val = sys.argv[idx + 1].lower()
            if val not in ("ollama", "anthropic"):
                raise ValueError
            LLM_PROVIDER = val
            log.info(f"LLM_PROVIDER sovrascritto a {LLM_PROVIDER}")
        except (IndexError, ValueError):
            log.error("Uso: python agent.py --llm ollama|anthropic")
            sys.exit(1)

    # Parametro inline: --model <nome>  sovrascrive OLLAMA_MODEL o ANTHROPIC_MODEL
    if "--model" in sys.argv:
        try:
            idx = sys.argv.index("--model")
            val = sys.argv[idx + 1]
            if LLM_PROVIDER == "anthropic":
                ANTHROPIC_MODEL = val
                log.info(f"ANTHROPIC_MODEL sovrascritto a {ANTHROPIC_MODEL}")
            else:
                OLLAMA_MODEL = val
                log.info(f"OLLAMA_MODEL sovrascritto a {OLLAMA_MODEL}")
        except IndexError:
            log.error("Uso: python agent.py --model <nome_modello>")
            sys.exit(1)

    if "--now" in sys.argv:
        run_agent()
    else:
        log.info(f"Agente schedulato alle {DAILY_RUN_TIME}")
        schedule.every().day.at(DAILY_RUN_TIME).do(run_agent)
        while True:
            schedule.run_pending()
            time.sleep(60)
