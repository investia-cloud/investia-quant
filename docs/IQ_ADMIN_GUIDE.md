# investia-quant — Guida Admin CLI `iq`

**Ultimo aggiornamento**: 19 giugno 2026
**Root progetto**: `~/investia-quant`
**Entry point**: `iq` (definito in `investia_quant/cli.py`)

Guida di riferimento per l'amministratore: per ogni comando — scopo, opzioni
con default, output prodotto, esempio completo copiabile.

---

## Indice comandi

| Comando | Scopo | Output |
|---|---|---|
| `iq run` | Runtime operativo: genera e invia report segnali (R e K) | Email segnali |
| `iq report` | Report performance storica (R e K) | Email performance |
| `iq r-analyze` | Pipeline R-portfolio (WFO + OFC + MC) | Card `.md` + PDF opzionale |
| `iq l-analyze` | Pipeline Lazy (frontiera + backtest + stability + MC A/B + DSR) | CSV classification + PDF opzionale |
| `iq k-analyze` | Pipeline K-strategy (inspector 1×1 o panel N×M) | CSV classification + PNG |
| `iq k-agent` | Genera nuove K-strategy da feed RSS / PDF | Strategie generate |

> **Naming**: `r-analyze` e `l-analyze` sono simmetrici (R-portfolio e Lazy).
> Entrambi accettano `--pdf` (default `False`) per la relazione tecnica.

---

## `iq run`

**Scopo**: esecuzione runtime operativa — calcola e invia i report di segnale
(buy/sell/hold) ai destinatari, per R-portfolio e K-portfolio.

**Opzioni**

| Opzione | Default | Descrizione |
|---|---|---|
| `--ptf`, `--portfolio` | — | Nome singolo portafoglio |
| `--all`, `--ptf-all` | `False` | Tutti i portafogli da `portfolios.conf` |
| `--rotational`, `--ptf-all-r` | `False` | Solo portafogli tipo R |
| `--trading`, `--ptf-all-k` | `False` | Solo portafogli tipo T |
| `--recipient`, `--mail`, `--mailto` | conf | Destinatario: email, `me`, `managers`, `customers` |
| `--report-date` | oggi | Data fine report `YYYY-MM-DD` |
| `--dry-run` | `False` | Simula senza inviare email |
| `--no-send` | `False` | Non inviare email (solo esegui) |
| `--verbose`, `-v` | `False` | Output verboso |
| `--wfo-results-dir` | auto | Override directory risultati WFO |

**Output**: email con i segnali ai destinatari risolti (nessun file locale
salvo i log).

**Esempio**
```bash
iq run --rotational --mail managers --mail customers
```

---

## `iq report`

**Scopo**: genera e invia il report di **performance storica** (metriche su un
periodo) per R-portfolio e K-portfolio.

**Opzioni**

| Opzione | Default | Descrizione |
|---|---|---|
| `--ptf`, `--portfolio` | — | Nome singolo portafoglio |
| `--all`, `--ptf-all` | `False` | Tutti i portafogli da `portfolios.conf` |
| `--rotational`, `--ptf-all-r` | `False` | Solo portafogli tipo R |
| `--trading`, `--ptf-all-k` | `False` | Solo portafogli tipo T |
| `--recipient`, `--mail`, `--mailto` | conf | Destinatario: email, `me`, `managers`, `customers` |
| `--start-date` | `2015-01-01` (R) / YTD (K) | Inizio analisi `YYYY-MM-DD` |
| `--end-date` | oggi | Fine analisi `YYYY-MM-DD` |
| `--no-send` | `False` | Non inviare email |
| `--verbose` | `False` | Output verboso |
| `--wfo-results-dir` | auto | Override directory risultati WFO |

**Output**: email con il report di performance.

**Esempio**
```bash
iq report --rotational --start-date 2018-01-01 --mail managers
```

---

## `iq r-analyze`

**Scopo**: pipeline completa di analisi **R-portfolio** — WFO (walk-forward
optimization) + OFC (out-of-sample funnel check) + Monte Carlo. Solo
R-portfolio.

> **Comportamento PDF**: la pipeline di calcolo e la **card `.md`** vengono
> sempre prodotte. La **relazione tecnica PDF** viene generata **solo con
> `--pdf`** (default `False`). Senza `--pdf` il campo `pdf` in output è
> `None`.

**Opzioni**

| Opzione | Default | Descrizione |
|---|---|---|
| `--ptf` | — | Nome R-portfolio da registry (es. `alpha_fact`) |
| `--universe` | — | CSV con colonna `ticker` (universo ad hoc) — alternativa a `--ptf` |
| `--output-dir` | `outputs/reports/<nome>/<data>/` | Directory output PDF + PNG |
| `--profile` | `satellite` | Profilo OFC: `satellite` o `core` |
| `--year` | anno corrente | Anno selezione WFO |
| `--start-date` | `2015-01-01` | Inizio storico download |
| `--pdf` | `False` | Genera la relazione tecnica PDF |
| `--verbose` | `False` | Output verboso |

**Output**
- `<output-dir>/<titolo>_<anno>.md` — card di sintesi (sempre)
- `<output-dir>/<titolo>_<anno>_Relazione_Tecnica.pdf` — solo con `--pdf`
- `<output-dir>/plots/` — PNG intermedi

**Esempio**
```bash
iq r-analyze --ptf alpha_fact --pdf
```

---

## `iq l-analyze`

**Scopo**: pipeline completa di analisi **Lazy portfolio** — frontiera
efficiente + backtest B&H con frequenza ottimale + stability rolling + Monte
Carlo Block A (intervalli di confidenza) + Monte Carlo Block B (skill
ribilanciamento) + DSR + decisione finale. Batch o singolo.

> **Comportamento PDF**: senza `--pdf` esegue solo la pipeline (CSV +
> cache). Con `--pdf` genera, per ogni PTF, la relazione tecnica in
> `outputs/lazy_reports/<ptf>_relazione_tecnica.pdf`.

**Opzioni**

| Opzione | Default | Descrizione |
|---|---|---|
| `--ptf` | — | Nome Lazy portfolio, oppure `all` per tutti i PTF `lazy_` del registry |
| `--output-dir` | `outputs/lazy_analysis/<data>/` | Directory output |
| `--start-date` | `2016-01-01` | Inizio storico backtest |
| `--end-date` | oggi | Fine storico |
| `--benchmark` | `SPY` | Ticker benchmark per il confronto |
| `--init-cash` | `100000.0` | Capitale iniziale backtest |
| `--fees` | `0.001` | Commissioni per trade |
| `--years` | `10` | Anni per frontiera efficiente e stability test |
| `--n-simulations-mc-a` | `1000` | Simulazioni Monte Carlo Block A |
| `--n-simulations-mc-b` | `500` | Simulazioni Monte Carlo Block B |
| `--override` | `False` | Ricalcola anche i PTF già in cache |
| `--pdf` | `False` | Genera la relazione tecnica PDF per ciascun PTF |
| `--verbose`, `-v` | `False` | Output verboso |

**Output**
- `<output-dir>/classification_<timestamp>.csv` — classificazione con verdetto
- `outputs/lazy_cache/<ptf>.pkl` e `<ptf>_mc_a2.pkl` — cache row + MC
- `outputs/lazy_reports/<ptf>_relazione_tecnica.pdf` — solo con `--pdf`

**Struttura della relazione tecnica Lazy (`--pdf`)**

| Sezione | Contenuto |
|---|---|
| §1 Identità | Nome, tipo (lazy/equity/sandbox), asset allocation |
| §2 Configurazione | Asset class/pesi, frequenza ribilanciamento, periodo backtest |
| §3 Metriche comparative | CAGR, Sharpe, MaxDD vs benchmark; frontiera (reale vs teorica) |
| §4 Validazione statistica | `lazy_rolling_stability` (P(rolling 5y < 0%)), DSR |
| §5 Monte Carlo | Block A (intervalli di confidenza), Block B (skill test) |
| §6 Proiezione capitale | `project_lazy_capital`, percentili P10–P50–P90 |
| §7 Decisione finale | Verdetto promozione |

**Esempi**
```bash
iq l-analyze --ptf golden_butterfly --pdf
iq l-analyze --ptf all --override
```

---

## `iq k-analyze`

**Scopo**: analisi **K-strategy** — inspector (1 strategia × 1 ticker) o panel
(N strategie × M ticker) in base agli argomenti. Pipeline WFO + OFC + DSR + MC.

**Opzioni**

| Opzione | Default | Descrizione |
|---|---|---|
| `-s`, `--strategies` | tutte | Una o più strategie (es. `-s dbma_matrix bollinger`) |
| `-t`, `--tickers` | — | Uno o più ticker (es. `-t NVDA AAPL`) — escl. con `--ptf` |
| `--ptf` | — | Nome K-portfolio (estrae i ticker automaticamente) |
| `--output-dir` | `outputs/k_analysis/<data>/` | Directory output |
| `--start-date` | `2015-01-01` | Inizio storico download |
| `--end-date` | oggi | Fine storico |
| `--ratio` | `4:1` | Train:test ratio WFO |
| `--fees` | `0.001` | Commissioni per trade |
| `--slippage` | `0.002` | Slippage per trade |
| `--price-col` | `Open` | Colonna prezzo OHLCV |
| `--selection-metric` | `total_return` | Metrica selezione parametri WFO |
| `--init-cash` | `100000.0` | Capitale iniziale |
| `--warmup-years` | `1` | Anni warmup WFO |
| `--wfo-results-dir` | `outputs/WFO_T_DEV_RESULTS/` | Directory risultati WFO |
| `--override` | `False` | Ricalcola risultati WFO già salvati |
| `--n-simulations` | `1000` | Numero simulazioni Monte Carlo |
| `--block-size` | `10` | Block size per Block Bootstrap MC |
| `--verbose`, `-v` | `False` | Output verboso |

**Output**: CSV classification + PNG nella directory di output; coppie
`(ticker, strategia)` promosse stampate a video.

**Esempio**
```bash
iq k-analyze -s dbma_matrix bollinger -t NVDA AAPL --override
```

---

## `iq k-agent`

**Scopo**: genera nuove K-strategy leggendo articoli da feed RSS oppure da un
PDF locale, via LLM.

**Opzioni**

| Opzione | Default | Descrizione |
|---|---|---|
| `--max` | `5` | Numero massimo articoli per run (escl. con `--pdf`) |
| `--pdf` | — | Processa un PDF locale invece del feed RSS |
| `--llm` | `anthropic` | Provider LLM: `ollama` o `anthropic` |
| `--model` | auto | Modello LLM (default per provider) |
| `--verbose`, `-v` | `False` | Output verboso |

**Output**: nuove strategie generate (file/artefatti gestiti da
`K-Strategy-Agent/agent.py`).

**Esempio**
```bash
iq k-agent --pdf paper.pdf --llm anthropic
```

---

## Note operative

- **Risoluzione `--ptf`**: il nome viene cercato in `R_PORTFOLIO_REGISTRY`,
  `K_PORTFOLIO_REGISTRY` e `L_PORTFOLIO_REGISTRY`. I comandi `r-analyze`,
  `l-analyze`, `k-analyze` validano il tipo (R/L/K) e rifiutano i mismatch.
- **Cron produzione** (`tslab.investia.cloud`): usa solo `iq run` (K ore 08:00
  giornaliero, R ore 08:00 primo del mese). Nessun cron usa `r-analyze` /
  `l-analyze`.
- **Cache Lazy**: `outputs/lazy_cache/`. Con `--pdf`, `l-analyze` ricalcola i
  PTF richiesti anche se in cache (servono gli oggetti completi per il PDF);
  la cache row/MC viene comunque aggiornata.
