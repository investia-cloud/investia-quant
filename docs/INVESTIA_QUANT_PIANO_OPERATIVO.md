# investia-quant — Piano Operativo

**Ultimo aggiornamento**: 12 giugno 2026
**Root progetto**: `~/investia-quant`

---

## Stato attuale

**Branch `main`** aggiornato e pulito. Ultimi commit:
```
feat(k-agent): migra output da JN a k_strategies_agent.py (22 strategie + agent redirect)
chore: cleanup notebooks/dev/ — rimuovi artefatti refactor e dir obsolete
docs: piano operativo riscritto — struttura pulita post sessione 11/06
feat(cli): implementa iq analyze — runner headless pipeline R-portfolio
chore(notebooks): rinomina JN dev a naming coerente
```

**Working tree**: clean.
**Branch parcheggiati**: nessuno.

### Notebooks dev attivi

| File | Ruolo |
|---|---|
| `notebooks/dev/r_portfolio_analyst.ipynb` | Analisi interattiva R-portfolio — solo Luca |
| `notebooks/dev/k_strategy_panel.ipynb` | Analisi massiva multi-strategia/ticker + classificazione risultati — solo Luca |
| `notebooks/dev/k_strategy_inspector.ipynb` | Analisi approfondita singola strategia:ticker — solo Luca |
| `notebooks/dev/lazy_portfolio_analyst.ipynb` | Lazy portfolios — Luca (JN), tutti gli altri via web |
| `notebooks/dev/R_Strategies.ipynb` | Rotazionale su strategie/metodi — esplorativo, solo Luca |
| `notebooks/dev/_bootstrap_dev.ipynb` | Bootstrap import libs_py con reload automatico |
| `K-Strategy-Agent/k_strategy_agent_output.ipynb` | Archivio storico strategie agente (read-only) |

### CLI `iq` — comandi disponibili

| Comando | Scopo | Stato |
|---|---|---|
| `iq run --ptf/--rotational/--trading/--all` | Runtime operativo — segnali ai gestori | ✅ Production |
| `iq report --ptf/--rotational/--trading/--all` | Statistiche YTD PTF deployati | ✅ Production |
| `iq analyze --ptf/--universe` | Pipeline R completa — relazione tecnica PDF | ✅ Implementato 11/06 |

### VPS produzione

- `tslab.investia.cloud` — release `2026.1` deployata
- Crontab attivo: K-portfolio ore 08:00 giornaliero, R-portfolio ore 08:00 primo del mese
- Rollback: symlink `current` → `releases/2026.1/`

---

## Architettura — 4 filiere

### Filiera R-portfolio (rotazionale su titoli)

| Aspetto | Dettaglio |
|---|---|
| JN dev | `r_portfolio_analyst.ipynb` — solo Luca |
| CLI | `iq analyze` ✅, `iq run` ✅, `iq report` ✅ |
| Web | R-portfolio designer (Fase 4 roadmap) |
| Utenti | Tutti i livelli |

### Filiera K-portfolio (trading system)

| Aspetto | Dettaglio |
|---|---|
| JN dev | `k_strategy_inspector.ipynb` (singola strategia:ticker) + `k_strategy_panel.ipynb` (massivo + classificazione) — solo Luca |
| CLI | `iq k-analyze` ← da costruire |
| Web | Nessuna — dominio esclusivo architetto |
| Utenti | Solo Luca |

I due JN sono **separati per design**:
- `k_strategy_inspector` → analisi approfondita singola strategia:ticker
- `k_strategy_panel` → analisi massiva multi-strategia/ticker + classificazione risultati già testati

### Filiera Lazy portfolio

| Aspetto | Dettaglio |
|---|---|
| JN dev | `lazy_portfolio_analyst.ipynb` — solo Luca |
| CLI | `iq lazy` ← da costruire |
| Web | Già pubblicato in v1 portale, da reintegrare in investia-platform |
| Output attuale | Risultati interattivi JN + grafici statici web |
| Gap | Relazione tecnica AI-generated (come R-portfolio) — da aggiungere |
| Utenti | Tutti i livelli inclusi gestori bancari |

### Filiera R-strategies (esplorativa)

| Aspetto | Dettaglio |
|---|---|
| JN dev | `R_Strategies.ipynb` — solo Luca, fase esplorativa |
| CLI/Web | Nessuna per ora |
| Utenti | Solo Luca |
| Note | Approccio rotazionale su strategie/metodi/pesi anziché titoli. Contiene `select_top_performing_stocks_NEW` con API vectorbt 1.0.0 da aggiornare (`from_returns` → `from_holding`). Ruolo operativo da chiarire dopo comprensione profonda. |

---

## Lavori in piedi, in ordine di priorità

### Priorità alta

**1. Audit + ottimizzazione + documentazione JN K-portfolio** · filiera K

Analisi funzioni definite inline nei due JN (non ancora in `libs_py/`),
identificare duplicati e gap, documentare, migrare in `k_functions.py`.
Branch: `feature/k-jn-audit`

**2. CLI `iq k-analyze`** · filiera K

Dopo audit JN K. Analogo a `iq analyze` per R-portfolio.
- `iq k-analyze --strategy <nome> --ticker <ticker>` → inspector (singola)
- `iq k-analyze --batch <file.csv>` o `--all` → panel (massivo)

**3. Potenziamento Block B** · filiera R

Sorgenti di skill alternative al momentum. Motivazione: molti PTF non hanno
skill momentum ma battono il benchmark per altri driver (clustering Ward +
risk-off). Block B attuale misura solo momentum — le sorgenti alternative
catturano quegli altri driver.

Sorgenti candidate:
- Risk-adjusted (Sharpe/Sortino rotazionale)
- Idiosyncratic return (residuo rispetto benchmark)
- Low-volatility (skill nell'evitare drawdown)
- Quality factor
- Multi-factor composito

Da fare: design session (definizione formale sorgenti + integrazione in
`run_all_mc_methods_rotational()`) prima di toccare codice.

### Priorità media

**4. Relazione tecnica AI per Lazy portfolio** · filiera Lazy

Gap rispetto a R-portfolio: aggiungere `generate_relazione_tecnica()` AI-based.
Dipende da: struttura output lazy JN da verificare.

**5. CLI `iq lazy`** · filiera Lazy

JN già refactored e funzionante — il lifting CLI dovrebbe essere basso.

**6. Web Lazy portfolio** · filiera Lazy

Reintegrazione in investia-platform. Era già pubblicato in v1 portale.
Dipende da: avanzamento investia-platform (Fase 4 roadmap ecosistema).

### Priorità bassa

**7. Agente relazioni tecniche** · filiera R

Batch su tutti i PTF: chiama `run_r_portfolio_analysis()` in loop.
Dipende da: `iq analyze` ora stabile ✓. Da pianificare.

**8. Comprensione R_Strategies + fix API vectorbt** · filiera R-strategies

JN esplorativo. Fix `from_returns` → `from_holding` (vectorbt 1.0.0).
Ruolo operativo da chiarire prima di pianificare evoluzioni.

---

## Architettura `iq analyze`

`iq analyze` è il runner headless della pipeline R-portfolio. Stesso calcolo
di `r_portfolio_analyst.ipynb`, output diverso e destinatari diversi.

| Aspetto | `r_portfolio_analyst.ipynb` | `iq analyze` |
|---|---|---|
| Utente | Solo Luca | Gestori bancari (via webapp) |
| Grafici | Plotly interattivo | Matplotlib/seaborn statici (PNG) |
| Output | PDF + PTF card + stampe intermedie | PDF + PNG scaricabili |
| Esecuzione | Interattiva | Headless |
| Input universo | Config fissa nel JN | `--ptf` o `--universe CSV` |

### Funzione core

`run_r_portfolio_analysis()` in `r_functions.py` (~riga 13878).
`cli.py` la chiama e basta — nessuna logica di pipeline in `cli.py`.

```python
def run_r_portfolio_analysis(
    portfolio_cfg: dict,
    output_dir: str | Path,
    year: int | None = None,       # default: anno corrente
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    profile: str = "satellite",    # "satellite" | "core"
    verbose: bool = False,
) -> dict:   # {"pdf", "plots_dir", "ofc_std", "ofc_cluster", "skill_profile_std", "skill_profile_cluster"}
```

### Pipeline headless

```
1.  Setup           tickers, dirs, date, wfo_results_dir, griglia
2.  Download        fetch_data_and_companies(), download_data(), build_benchmark()
3.  Risk-off        download_data(risk_off_tickers_uniq)
4.  Stability       reduce_grid_via_stability() → reduced_grid
5.  WFO Std         run_wfo_pipeline(use_clustering=False, plot=False)
6.  WFO Cluster     run_wfo_pipeline(use_clustering=True, plot=False)
7.  Compare         compare_wfo_pipelines(plot=False)
8.  OFC Std         overfitting_check_rotational()
9.  OFC Cluster     overfitting_check_rotational()
10. MC              run_all_mc_methods_rotational() x2 (std + cluster)
11. Decisione       compute_skill_profile(), print_final_decision()
12. Output          generate_ptf_card_md(), generate_relazione_tecnica()
```

### Fix headless applicati (11/06/2026)

- `matplotlib.use('Agg')` all'inizio di `run_r_portfolio_analysis`
- `compare_wfo_pipelines`: aggiunto parametro `plot`, `fig.show()` condizionato
- `analyze_portfolio_metrics`: rimossa `display()` ridondante (riga 2974)
- `pf_rot_std` / `pf_rot_cluster`: fallback a `_base` quando `risk_off_data=None`
  (portafogli senza risk-off tickers, es. `portfolio_alpha_fact`)

### Opzioni CLI

```
iq analyze --ptf <nome>              # PTF da r_portfolios registry
iq analyze --universe <file.csv>     # universo ad hoc
           --output-dir <path>       # default: outputs/reports/<nome>/<data>/
           --profile satellite|core  # default: satellite
           --year <YYYY>             # default: anno corrente
           --start-date <YYYY-MM-DD> # default: 2015-01-01
           --verbose
```

`--ptf` e `--universe` sono mutuamente esclusivi.
CSV: colonna `ticker` obbligatoria; cfg sintetica: no benchmark, no risk-off.

---

## Convenzioni operative

- **Patch chirurgiche**: modifiche mirate, zero scope creep. Se emerge Y mentre
  si lavora su X → documenta Y nel piano e apri branch separato.
- **Il rerun lo fa sempre l'architetto**: Code non esegue rerun, non legge PDF/PNG.
- **Commit solo dopo validazione visiva del PDF rigenerato**.
- **Branch separati per ogni scope**: `fix/`, `feature/`, `chore/` — mai su main diretto
  (eccezione: patch doc e fix < 5 righe già validati).
- **Notebook `.ipynb`**: non committare per soli output celle. `git add` esplicito.
- **Effort esplicito nei prompt Code**: `EFFORT: minimal / standard / verbose`.
- **Decisioni di design vanno esplicitate** e validate prima di procedere.
- **STOP SE nei prompt Code**: condizioni esplicite in cui Code si ferma e segnala.

### Template prompt Code

```
Branch: [nome-branch]
EFFORT: minimal | standard | verbose

MODALITÀ: solo modifiche codice. NON eseguire rerun. NON leggere
output generati. La verifica la fa l'architetto.

CONTESTO:
File: [nome_file]
Funzione: [nome_funzione] (~riga [N])

PROBLEMA:
[Descrizione precisa con esempio concreto]

OBIETTIVO:
[Comportamento atteso dopo il fix]

PATCH:
[Logica del fix o pseudocodice]

VERIFICA:
Stampa righe toccate con numero di riga.
AST OK su [file].

STOP SE:
- [condizione 1]
- [condizione 2]
Non risolvere autonomamente. Segnala e attendi istruzioni.
```

---

## Storia bug (issue tracker)

Stato: OPEN / RESOLVED / CLOSED (chiuso senza fix).

### B-001 — `duplicate labels` su universi ampi · RESOLVED

`ValueError: cannot reindex on an axis with duplicate labels` in `run_rotational_engine`.
Causa: ticker in `tickers` e `risk_off_tickers` duplicati a valle.
Fix: `risk_off_tickers_uniq = [t for t in risk_off_tickers if t not in tickers]`.
Commit: `fix/r-mc-cluster-symmetry`.

### B-002 — `reduce_grid_via_stability` rifiuta universo piccolo · RESOLVED

`ValueError: Universe too small` su `portfolio_alpha_sect` (9 ticker, servivano 11).
Fix: graceful fallback in `reduce_grid_via_stability` per universi sotto soglia.
Commit: `fix/r-mc-cluster-symmetry`.

### B-003 — `compare_selection_columns` fallisce con `float not iterable` · RESOLVED

`TypeError` in `compare_selection_columns` su `portfolio_germany_plan`.
Causa: stessa di B-001. Fix: stesso fix di B-001.
Commit: `fix/r-mc-cluster-symmetry`.

### B-004 — Presunto swing p-value B1 MC · CLOSED (non era un bug)

Lo swing apparente (p=0.980 → p=0.001 in 3 giorni) era confronto tra path Standard
e path Cluster, non instabilità della MC. Il mismatch CAGR_MC vs CAGR_vbt è by design
(convenzione accademica vs interna vbt) e non inquina il p-value. Nessun fix.

### B-005 — Narrativa 6.b/6.c sempre sul path Standard · RESOLVED

Le sezioni narrative citavano numeri del path Standard anche quando raccomandato era
Cluster. Fix: strategia Opzione C — narrativa focalizzata sul path raccomandato +
nota di contrasto. Estesi `_diagnose_mc`, `_build_verdict_text`, aggiunto
`_build_verdict_text_compact`. CI §6.c: percentile reale calcolato, non hardcoded.
Commit: `fix/mc-narrative-per-path` (sessione 24/05).

### B-006 — Nomenclatura invertita in `compute_skill_profile` · RESOLVED

Mappa (B1 PASS, S3 FAIL) → 'Timing-driven' invece di 'Selection-driven'. B2 ignorato.
Fix: nuova mappa basata su (B1, B2): Strong / Selection-driven / Timing-driven / No-skill.
Commit: `fix/mc-narrative-per-path` (prerequisito di B-005).

---

## Storia sessioni

### Sessione 24/05/2026 — Fix MC reporting

Branch `fix/mc-narrative-per-path` → main.
Fix #1: sub-directory `std/` e `cluster/` per plot MC (evita sovrascrittura PNG).
Fix #2: etichettatura figure (assorbito in #1).
Fix #3: narrativa per-path + B-005 + B-006.
Verifica su Alpha World Vanguard: tutte le sezioni corrette ✓

### Sessione 08/06/2026 — pyproject.toml + CLI iq scheletro

Branch `refactor/libs-py`.
Task 1.2: pyproject.toml + venv `~/.venvs/investia-quant/`.
Task 1.3: CLI `iq` scheletro — `iq run` R e K validati, mail inviata ✓.
libs_py/: 11 librerie convertite da .ipynb a .py, AST OK.

### Sessione 09/06/2026 — Fix iq report K + Release versionata

Branch `refactor/libs-py`.
Fix `iq report K` periodo YTD (3 bug: import mancante, start_date, analisys_start_date).
Release `2026.1` deployata su VPS con venv isolato e symlink `current`.
CLI potenziata: `--rotational/--trading/--all`, alias, shortcut mail.

### Sessione 10/06/2026 — Chiusura refactor/libs-py + bootstrap dev

Branch `refactor/libs-py` → mergiato su main.
Eliminata `notebooks/runtime/` (16 file, 65.410 righe).
Bootstrap dev: `%run libs/*.ipynb` → `import` da `libs_py/` con reload automatico.
Fix import mancanti in r/u/t/mc_functions.py.
Fix: lxml, kaleido 0.2.1 + plotly 5.24.1, pct_change FutureWarning.
Verifica tutti i JN attivi ✓. Crontab VPS installato.

### Sessione 11/06/2026 — Rename JN + iq analyze + cleanup + k-agent

Branch `chore/rename-notebooks` + `feature/iq-analyze` + `feature/k-agent-output-to-py` → mergiati su main.
Cleanup `notebooks/dev/` diretto su main (file non tracciati).

Rename JN dev: R_Asset_v2 → r_portfolio_analyst, WFO_Framwork → k_strategy_inspector,
WFO_Strategy_Panel → k_strategy_panel, MyCurvo → lazy_portfolio_analyst,
K-Strategy-Agent/strategies → k_strategy_agent_output. ✓

iq analyze implementato: `run_r_portfolio_analysis()` in `r_functions.py`,
command `analyze` in `cli.py`. Fix headless (matplotlib Agg, display rimossa,
plot condizionato, fallback pf_rot). Test su `portfolio_alpha_fact` ✓

Cleanup `notebooks/dev/`: rimossi artefatti refactor, PNG, dir obsolete
(mc_results/, reports/, snapshots/, _workbench/, data/, ptf_cards/, OLD/). ✓

K-Agent migrato: `k_strategy_agent_output.ipynb` → `k_strategies_agent.py` (22 strategie).
`agent.py` modificato: output redirect da JN a `notebooks/libs_py/k_strategies_agent.py`.
Funzioni notebook (`load_or_create_notebook`, `save_notebook`) conservate ma non più chiamate. ✓

Definizione architettura `iq analyze` e distinzione comandi CLI approvate.
Convenzione `EFFORT` nei prompt Code introdotta.

### Sessione 12/06/2026 — Architettura 4 filiere + piano operativo

Decisioni architetturali approvate:
- Architettura definitiva a 4 filiere: R-portfolio, K-portfolio, Lazy portfolio, R-strategies.
- K-portfolio: JN separati per design (inspector vs panel) — no unificazione.
- Lazy portfolio: già refactored e funzionante; gap = relazione tecnica AI + CLI + web.
- R-strategies: fase esplorativa, solo JN architetto, nessuna pipeline per ora.
- Potenziamento Block B: indipendente dai risultati PTF; motivazione = driver non-momentum
  (clustering Ward + risk-off) non catturati dal Block B attuale.
- Piano operativo riscritto con struttura a filiere.
