# investia-quant — Piano Operativo

**Ultimo aggiornamento**: 12 giugno 2026
**Root progetto**: `~/investia-quant`

---

## Stato attuale

**Branch `main`** aggiornato e pulito. Ultimi commit:
```
feat(k): cache MC su disco, classification.csv, metriche complete, load WFO da disco
refactor(cli): unifica k-analyze e k-test, elimina test_strategies.py, implementa run_k_strategy_analysis()
feat(k): run_k_strategy_analysis() pipeline completa WFO+DSR+OFC+MC, fix headless plots
feat(cli): aggiungi k-analyze (inspector/panel) e k-test + fix build_namespace libs_py
refactor(k): migra funzioni inline in k_functions.py, pulisci JN dev
docs: pipeline valutazione K-strategy e R-portfolio
docs: piano operativo — architettura 4 filiere + priorità aggiornate (12/06)
```

**Working tree**: clean (rimuovere `None/` se presente: `rm -rf None/`).
**Branch parcheggiati**: nessuno.

### Notebooks dev attivi

| File | Ruolo |
|---|---|
| `notebooks/dev/r_portfolio_analyst.ipynb` | Analisi interattiva R-portfolio — solo Luca |
| `notebooks/dev/k_strategy_panel.ipynb` | Viewer interattivo classification.csv — da riscrivere |
| `notebooks/dev/k_strategy_inspector.ipynb` | Da eliminare — assorbito da `iq k-analyze` |
| `notebooks/dev/lazy_portfolio_analyst.ipynb` | Lazy portfolios — Luca (JN), tutti via web |
| `notebooks/dev/R_Strategies.ipynb` | Rotazionale su strategie/metodi — esplorativo, solo Luca |
| `notebooks/dev/_bootstrap_dev.ipynb` | Bootstrap import libs_py con reload automatico |
| `K-Strategy-Agent/k_strategy_agent_output.ipynb` | Archivio storico strategie agente (read-only) |

### CLI `iq` — comandi disponibili

| Comando | Scopo | Stato |
|---|---|---|
| `iq run --ptf/--rotational/--trading/--all` | Runtime operativo — segnali ai gestori | ✅ Production |
| `iq report --ptf/--rotational/--trading/--all` | Statistiche YTD PTF deployati | ✅ Production |
| `iq analyze --ptf/--universe` | Pipeline R completa — relazione tecnica PDF | ✅ Production |
| `iq k-analyze -s/-t/--ptf` | Pipeline K completa — WFO+OFC+DSR+MC | ✅ Production |

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
| JN dev | `k_strategy_panel.ipynb` — viewer classification.csv (da riscrivere) |
| CLI | `iq k-analyze` ✅ |
| Web | Nessuna — dominio esclusivo architetto |
| Utenti | Solo Luca |

**Architettura `iq k-analyze`:**
- `-s <strategie> -t <tickers>` → dispatch automatico inspector (1×1) o panel (N×M)
- `--ptf <nome>` → estrae tickers dal K-portfolio registry
- Pipeline: WFO → OFC (da precheck) → DSR → MC → verdetto
- Cache su disco: `_results.pkl`, `_precheck.pkl`, `_mc_results.pkl`
- Output: `classification_<data>.csv` in `outputs/WFO_T_DEV_RESULTS/`
- Secondo run senza `--override`: carica tutto da disco (~5 secondi)

**Flusso operativo K:**
```
agent.py → genera strategie → k_strategies_agent.py
iq k-analyze --ptf <nome> → pipeline completa → classification.csv
k_strategy_panel.ipynb → viewer interattivo classification.csv
cron → iq k-analyze --ptf <nome> (nuove strategie)
```

### Filiera Lazy portfolio

| Aspetto | Dettaglio |
|---|---|
| JN dev | `lazy_portfolio_analyst.ipynb` — solo Luca |
| CLI | `iq lazy` ← da costruire |
| Web | Già pubblicato in v1 portale, da reintegrare |
| Gap | Relazione tecnica AI-generated — da aggiungere |
| Utenti | Tutti i livelli inclusi gestori bancari |

### Filiera R-strategies (esplorativa)

| Aspetto | Dettaglio |
|---|---|
| JN dev | `R_Strategies.ipynb` — solo Luca, fase esplorativa |
| CLI/Web | Nessuna per ora |
| Note | Fix API vectorbt: `from_returns` → `from_holding`. Ruolo operativo da chiarire. |

---

## Lavori in piedi, in ordine di priorità

### Priorità alta

**1. Riscrivi `k_strategy_panel.ipynb` come viewer** · filiera K

Il JN non calcola più nulla — legge `classification_*.csv` da
`outputs/WFO_T_DEV_RESULTS/` e visualizza con Plotly interattivo.
Funzionalità: carica CSV, filtra per Promoted/OFC/MC/DSR, ordina per
metrica, confronta run diversi, grafici equity dei promossi.

**2. Elimina `k_strategy_inspector.ipynb`** · filiera K

Assorbito completamente da `iq k-analyze -s <s> -t <t>`.
Da fare dopo validazione completa CLI.

**3. Potenziamento Block B** · filiera R

Sorgenti di skill alternative al momentum. Motivazione: molti PTF non hanno
skill momentum ma battono il benchmark per altri driver (clustering Ward +
risk-off). Block B attuale misura solo momentum.

Sorgenti candidate:
- Risk-adjusted (Sharpe/Sortino rotazionale)
- Idiosyncratic return (residuo rispetto benchmark)
- Low-volatility (skill nell'evitare drawdown)
- Quality factor
- Multi-factor composito

Da fare: design session prima di toccare codice.

### Priorità media

**4. Relazione tecnica AI per Lazy portfolio** · filiera Lazy

Gap rispetto a R-portfolio: aggiungere `generate_relazione_tecnica()`.

**5. CLI `iq lazy`** · filiera Lazy

JN già refactored — lifting CLI dovrebbe essere basso.

**6. Web Lazy portfolio** · filiera Lazy

Reintegrazione in investia-platform (Fase 4 roadmap ecosistema).

### Priorità bassa

**7. Agente relazioni tecniche** · filiera R

Batch su tutti i PTF: chiama `run_r_portfolio_analysis()` in loop.

**8. Comprensione R_Strategies + fix API vectorbt** · filiera R-strategies

Fix `from_returns` → `from_holding`. Ruolo operativo da chiarire.

---

## Tech debt

- `None/` directory creata da bug wfo_results_dir=None (ora fixato) — `rm -rf None/`
- Output MC verboso ancora presente senza `--verbose` — da silenziare
- `k_strategy_inspector.ipynb` da eliminare dopo validazione CLI

---

## Architettura `iq analyze` (R-portfolio)

`iq analyze` è il runner headless della pipeline R-portfolio.

| Aspetto | `r_portfolio_analyst.ipynb` | `iq analyze` |
|---|---|---|
| Utente | Solo Luca | Gestori bancari (via webapp) |
| Grafici | Plotly interattivo | Matplotlib/seaborn statici (PNG) |
| Output | PDF + PTF card + stampe intermedie | PDF + PNG scaricabili |
| Esecuzione | Interattiva | Headless |
| Input universo | Config fissa nel JN | `--ptf` o `--universe CSV` |

### Funzione core

`run_r_portfolio_analysis()` in `r_functions.py` (~riga 13878).

```python
def run_r_portfolio_analysis(
    portfolio_cfg: dict,
    output_dir: str | Path,
    year: int | None = None,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    profile: str = "satellite",
    verbose: bool = False,
) -> dict:
```

### Pipeline headless R

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

---

## Convenzioni operative

- **Patch chirurgiche**: modifiche mirate, zero scope creep.
- **Il rerun lo fa sempre l'architetto**: Code non esegue rerun, non legge PDF/PNG.
- **Commit solo dopo validazione visiva del PDF rigenerato**.
- **Branch separati per ogni scope**: `fix/`, `feature/`, `chore/`.
- **Notebook `.ipynb`**: non committare per soli output celle. `git add` esplicito.
- **Effort esplicito nei prompt Code**: `EFFORT: minimal / standard / verbose`.
- **AUTONOMIA nei prompt Code**: sempre inclusa per evitare interruzioni.
- **Decisioni di design vanno esplicitate** e validate prima di procedere.
- **Verifiche funzionali**: mai delegare a Code — le fa l'architetto.
- **Code solo per**: modifiche codice complesse, multi-file. Tutto il resto: orchestratore o architetto.

### Template prompt Code

```
Branch: [nome-branch]
EFFORT: minimal | standard | verbose
AUTONOMIA: completa tutti i task in sequenza senza chiedere conferma
intermedia. Segnala solo se colpisci una condizione STOP SE.
Alla fine stampa un riepilogo di tutto ciò che è stato fatto.

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
`ValueError` in `run_rotational_engine`. Fix: `risk_off_tickers_uniq`.

### B-002 — `reduce_grid_via_stability` rifiuta universo piccolo · RESOLVED
Graceful fallback per universi sotto soglia.

### B-003 — `compare_selection_columns` fallisce con `float not iterable` · RESOLVED
Stessa causa di B-001.

### B-004 — Presunto swing p-value B1 MC · CLOSED (non era un bug)
Confronto tra path Standard e Cluster, non instabilità.

### B-005 — Narrativa 6.b/6.c sempre sul path Standard · RESOLVED
Narrativa focalizzata sul path raccomandato. Commit: `fix/mc-narrative-per-path`.

### B-006 — Nomenclatura invertita in `compute_skill_profile` · RESOLVED
Nuova mappa basata su (B1, B2). Commit: `fix/mc-narrative-per-path`.

### B-007 — `boxplot()` keyword `labels` deprecata · RESOLVED
Fix: `labels` → `tick_labels` righe 6672 e 6697 in `k_functions.py`.

---

## Storia sessioni

### Sessione 24/05/2026 — Fix MC reporting
Branch `fix/mc-narrative-per-path` → main.

### Sessione 08/06/2026 — pyproject.toml + CLI iq scheletro
Branch `refactor/libs-py`.

### Sessione 09/06/2026 — Fix iq report K + Release versionata
Branch `refactor/libs-py`.

### Sessione 10/06/2026 — Chiusura refactor/libs-py + bootstrap dev
Branch `refactor/libs-py` → main.

### Sessione 11/06/2026 — Rename JN + iq analyze + cleanup + k-agent
Branch `chore/rename-notebooks` + `feature/iq-analyze` + `feature/k-agent-output-to-py` → main.

### Sessione 12/06/2026 — Architettura K-portfolio + CLI k-analyze

Decisioni architetturali approvate:
- Architettura definitiva a 4 filiere: R-portfolio, K-portfolio, Lazy, R-strategies.
- K-portfolio: `iq k-analyze` unico comando (inspector + panel + batch PTF).
- `test_strategies.py` eliminato — sostituito da `iq k-analyze`.
- `k_strategy_inspector.ipynb` da eliminare — assorbito dalla CLI.
- `k_strategy_panel.ipynb` → viewer interattivo classification.csv.
- Pipeline K: WFO → OFC (precheck) → DSR → MC → verdetto.
- Cache su disco: `_results.pkl`, `_precheck.pkl`, `_mc_results.pkl`.
- Output: `classification_<data>.csv` in `outputs/WFO_T_DEV_RESULTS/`.
- Secondo run senza `--override`: ~5 secondi (tutto da cache).
- Documenti pipeline: `docs/K_STRATEGY_PIPELINE.md`, `docs/R_PORTFOLIO_PIPELINE.md`.
- Analisi librerie: tutte pulite (0 funzioni orfane su 8 file libs_py).
