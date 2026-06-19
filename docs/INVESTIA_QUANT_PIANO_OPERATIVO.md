# investia-quant — Piano Operativo

**Ultimo aggiornamento**: 19 giugno 2026
**Root progetto**: `~/investia-quant`

---

## Stato attuale

**Branch `main`** aggiornato e pulito. Ultimi commit:
```
fix(r-portfolio): risk_off_tickers per-PTF con default da k_tickers (era sempre vuoto, pf_rot None silenzioso)
docs: aggiorna piano operativo con rename r-analyze/l-analyze e flag --pdf
feat(cli): rename analyze→r-analyze, lazy-analyze→l-analyze, flag --pdf simmetrico, relazione tecnica AI Lazy, IQ_ADMIN_GUIDE
fix(mc-plot): serializzazione Timestamp in write_image dei fan chart MC (kaleido/orjson)
fix(r-portfolio): salva su wfo_file_save solo il path scelto in §8, non sempre Standard
```

**Working tree**: clean.
**Branch parcheggiati**: nessuno.

⚠️ **Regola operativa fissata il 19/06** (violata e recuperata a caro prezzo in
questa stessa sessione): **un branch alla volta, commit + merge in `main`
prima di aprirne un altro**. Niente nuovi branch finché quello corrente non
è chiuso. Niente stash dimenticati.

### Notebooks dev attivi

| File | Ruolo |
|---|---|
| `notebooks/dev/r_portfolio_analyst.ipynb` | Analisi interattiva R-portfolio — solo Luca |
| `notebooks/dev/k_strategy_panel.ipynb` | Viewer CSV classification + §7 Trading System Analysis (load, exposure, timing, comparativa promoted) |
| `notebooks/dev/lazy_portfolio_analyst.ipynb` | Lazy portfolios — Luca (JN), tutti via web |
| `notebooks/dev/R_Strategies.ipynb` | Rotazionale su strategie/metodi — esplorativo, solo Luca |
| `notebooks/dev/_bootstrap_dev.ipynb` | Bootstrap import libs_py con reload automatico |
| `K-Strategy-Agent/k_strategy_agent_output.ipynb` | Archivio storico strategie agente (read-only) |

### CLI `iq` — comandi disponibili

| Comando | Scopo | Stato |
|---|---|---|
| `iq run --ptf/--rotational/--trading/--all` | Runtime operativo — segnali ai gestori | ✅ Production |
| `iq report --ptf/--rotational/--trading/--all` | Statistiche YTD PTF deployati | ✅ Production |
| `iq r-analyze --ptf/--universe [--pdf]` | Pipeline R completa — card .md sempre, relazione tecnica PDF con `--pdf` | ✅ Production |
| `iq l-analyze --ptf [--pdf]` | Pipeline Lazy — frontiera+stability+MC A/B+DSR; relazione tecnica PDF con `--pdf` | ✅ Production |
| `iq k-analyze -s/-t/--ptf` | Pipeline K completa — WFO+OFC+DSR+MC | ✅ Production |
| `iq k-agent --max/--llm/--model/--pdf` | Genera K-strategy da articoli Medium; --pdf per PDF locali | ✅ Production |

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
| CLI | `iq r-analyze` ✅, `iq run` ✅, `iq report` ✅ |
| Web | R-portfolio designer (Fase 4 roadmap) |
| Utenti | Tutti i livelli |

### Filiera K-portfolio (trading system)

| Aspetto | Dettaglio |
|---|---|
| JN dev | `k_strategy_panel.ipynb` — viewer classification.csv ✅ |
| CLI | `iq k-analyze` ✅, `iq k-agent` ✅ |
| Web | Nessuna — dominio esclusivo architetto |
| Utenti | Solo Luca |

**Architettura `iq k-analyze`:**
- `-s <strategie> -t <tickers>` → dispatch automatico inspector (1×1) o panel (N×M)
- `--ptf <nome>` → estrae tickers dal K-portfolio registry
- Pipeline: WFO → OFC (da precheck) → DSR → MC → BH_Beat_Ret → BH_Beat_DD → verdetto
- Cache su disco: `_results.pkl`, `_precheck.pkl`, `_mc_results.pkl`
- Output: `classification_<data>.csv` in `outputs/WFO_T_DEV_RESULTS/`
- Secondo run senza `--override`: ~5 secondi (tutto da cache)

**CSV classification — colonne:**
```
Ticker, Strategy, Sharpe, CAGR%, TotRet%, BH_TotRet%, MaxDD%, BH_DD%,
DSR, OFC, MC, BH_Beat_Ret, BH_Beat_DD, Promoted
```

**Criteri di promozione (tutti devono passare):**
```
OFC pass_gate = True       (griglia parametri robusta)
AND DSR >= 0.95            (Sharpe statisticamente significativo)
AND MC >= 2/3 pass         (performance robusta)
AND TotRet% > BH_TotRet%  (batte il B&H in return)
AND MaxDD% < BH_DD%        (drawdown inferiore al B&H)
```

**Flusso operativo K:**
```
iq k-agent --max 15 --llm anthropic   → genera strategie → k_strategies_agent.py
iq k-analyze --ptf <nome>             → pipeline completa → classification.csv
k_strategy_panel.ipynb                → viewer: classifica, scatter, equity, export
outputs/k_panel_exports/              → promoted_<ts>.csv + trading_systems_<ts>.py
cron irina (locale) ore 02:00         → iq k-agent --max 15 --llm anthropic
cron irina (locale) ore 03:00         → iq k-analyze --ptf <PTF_DA_DEFINIRE>
```

**`k_strategy_panel.ipynb` — sezioni:**
- §1 Configurazione (WFO_DIR, filtri, sort)
- §2 Classifica (tabella colorata con DeltaTotRet%, DeltaDD%)
- §3 Scatter Sharpe vs TotRet%
- §4 Equity curve promossi (con B&H)
- §5 Confronto run multipli
- §6 Export CSV + snippet trading_systems per k_portfolios.py

**Funzioni panel in `k_functions.py`:**
- `load_k_classifications()` — carica e consolida CSV
- `style_k_classification()` — Styler colorato con formato decimali
- `load_k_equity()` — carica pkl vectorbt
- `plot_k_equity()` — equity curve normalizzata Plotly

### Filiera Lazy portfolio

| Aspetto | Dettaglio |
|---|---|
| JN dev | `lazy_portfolio_analyst.ipynb` — solo Luca |
| CLI | `iq l-analyze --ptf [--pdf]` ✅ (rinominato da `lazy-analyze` il 19/06) |
| Web | Da reintegrare in investia-platform — design sospeso (vedi priorità alta) |
| Relazione tecnica AI | ✅ `generate_relazione_tecnica_lazy()`, §1-§7, completata il 19/06 |
| Utenti | Tutti i livelli inclusi gestori bancari |

### Filiera R-strategies (esplorativa)

| Aspetto | Dettaglio |
|---|---|
| JN dev | `R_Strategies.ipynb` — solo Luca, fase esplorativa |
| CLI/Web | Nessuna per ora |
| Note | Fix API vectorbt: `from_returns` → `from_holding`. Ruolo operativo da chiarire. |

---

## Lavori in piedi, in ordine di priorità

### Priorità MASSIMA

**0. Risk ON/OFF non applicato dal runtime di produzione** · filiera R · ⚠️ gap critico

Scoperto e in parte risolto il 19/06, ma resta un pezzo scoperto e ora urgente.

Cosa è stato risolto il 19/06: `run_r_portfolio_analysis()` (pipeline di
analisi, usata da `iq r-analyze`) leggeva `risk_off_tickers` da una chiave
(`portfolio_cfg.get("risk_off_tickers", [])`) che non è mai esistita in
nessun dict di `r_portfolios.py` → sempre `[]` → `risk_off_data=None` →
`pf_rot` (variante Risk ON/OFF) sempre `None` per entrambi i path, sostituito
silenziosamente da `pf_rot_base` → relazione tecnica mostrava sempre N/A
per le colonne "— Risk ON/OFF". **Fix**: default ora preso da
`k_tickers.risk_off_tickers` (stessa lista globale già usata dal JN), con
possibilità di override per-PTF in `r_portfolios.py` (non ancora popolato
per nessun PTF specifico — usa tutti il default). Verificato sul motore di
selezione: il meccanismo è lo stesso identico per Standard e Cluster
(overlay applicato dopo la selezione, righe ~8109-8126 di `run_wfo_pipeline`)
— non c'è logica diversa da capire per Cluster, timore iniziale infondato.

**Cosa resta scoperto, ed è il motivo della priorità massima**: il runtime
di produzione (`r_run_portfolio` in `r_functions.py`, ~riga 3311, eseguito
da `iq run`) **non ha alcun parametro per Risk ON/OFF** — nessun
`risk_off_tickers`, nessuna logica di overlay. Estrae solo
`rebalance_frequency` e `n_top` dal summary (via
`extract_operational_params_from_summary`). L'overlay Risk ON/OFF esiste
SOLO nella pipeline di analisi, mai nel runtime.

Perché è ora massima priorità e non più "da investigare quando serve":
il 19/06, testando il fix su `portfolio_germany_plan`, la raccomandazione
AI della relazione tecnica è **Cluster — Risk ON/OFF** (Sharpe 1.04 vs 0.27,
CAGR 31.3% vs 9.8%). Se questo PTF viene deployato seguendo la
raccomandazione, il runtime eseguirebbe comunque la selezione nuda, **senza
la protezione difensiva che ha motivato la scelta** — un disallineamento
silenzioso tra cosa l'architetto pensa di aver deployato e cosa gira
davvero in produzione.

Da fare: progettare come `r_run_portfolio`/`iq run` ottiene
`risk_off_tickers` (probabilmente da `r_portfolios.py`, stesso posto del
fix di analisi) e applica l'overlay (stesso overlay "universo allargato"
già verificato, non serve logica nuova — solo portarla nel runtime).

### Priorità alta

**1. Web Lazy portfolio** · filiera Lazy

Reintegrazione in investia-platform. Discussione di design avviata il 19/06,
poi sospesa: emerso che `investia-platform` oggi non ha alcuna separazione
modulare (`webapp/routers/{analysis,dashboard,portfolio}.py` è già e solo
cert-monitor, flat — non un modulo montato accanto ad altri). Serve prima
vedere `webapp/main.py` e `webapp/auth.py` per capire come sono cablati
routing e auth, poi disegnare lo schema di modularizzazione (cert-monitor
come primo modulo + Lazy come secondo), e solo dopo scegliere un nuovo nome
per cert-monitor (il modulo è cresciuto oltre il monitoraggio: analisi
evoluta, gestione portafoglio certificati, export smart via LLM). Rename
solo a livello UI/docs — repo GitHub invariato.

### Priorità media

**2. Agente relazioni tecniche** · filiera R

Batch su tutti i PTF: chiama `run_r_portfolio_analysis()` in loop.

**3. Comprensione R_Strategies + fix API vectorbt** · filiera R-strategies

Fix `from_returns` → `from_holding`. Ruolo operativo da chiarire.

### Priorità bassa

**4. PTF K per crontab** · filiera K · ⏳ BLOCCATO

Crontab `iq k-agent` attivo su `irina` (ore 02:00, --max 15, anthropic).
Crontab `iq k-analyze` in attesa: l'universo ticker per i trading system
2027 dipende da `select_top_performing_stocks` calcolato su base annuale.
L'universo 2026 sarà disponibile solo a fine 2026.

Azione: definire PTF K e attivare `iq k-analyze` in crontab a fine 2026,
in parallelo alla certificazione PTF per la release 2027.


**5. Potenziamento Block B** · filiera R

Sorgenti di skill alternative al momentum. Motivazione: molti PTF non hanno
skill momentum ma battono il benchmark per altri driver (clustering Ward + risk-off).

Sorgenti candidate:
- Risk-adjusted (Sharpe/Sortino rotazionale)
- Idiosyncratic return (residuo rispetto benchmark)
- Low-volatility (skill nell'evitare drawdown)
- Quality factor
- Multi-factor composito

Da fare: design session prima di toccare codice.

**6. Save/load completo per ri-analisi decisionale R-portfolio** · filiera R

Accantonato il 19/06: il caso "riprendere un PTF già promosso" è coperto
da §10 Load (summary_df del path scelto, ora corretto dal fix del 19/06).
Resta scoperto solo "rivedere il confronto tra i 4 path senza rifare
WFO+OFC+MC" — accettato rifare manualmente finché i volumi restano bassi
(non vale il costo di salvare `ci_summary_df`/`skill_results` per tutti
i path + gestire la ricostruzione di `compare_wfo_pipelines`).

---

## Tech debt

- Crontab locale da attivare dopo definizione PTF target

---

## Filiera progettazione nuovi PTF di trading

```
Fase 1  iq k-agent              → genera strategie da articoli
Fase 2  k_portfolios.py         → definisce universo tickers PTF
Fase 3  iq k-analyze --ptf      → pipeline WFO+OFC+DSR+MC → classification.csv
Fase 4  k_strategy_panel.ipynb  → viewer: seleziona trading system vincenti
Fase 5  k_portfolios.py         → configura PTF finale con trading system scelti
Fase 6  release annuale         → deploy VPS
```

---

## Architettura `iq r-analyze` (R-portfolio)

### Funzione core

`run_r_portfolio_analysis()` in `r_functions.py` (~riga 13878).

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
- **Commit solo dopo validazione visiva**.
- **Branch separati per ogni scope**: `fix/`, `feature/`, `chore/`.
- **Notebook `.ipynb`**: non committare per soli output celle. `git add` esplicito.
- **AUTONOMIA nei prompt Code**: sempre inclusa per evitare interruzioni.
- **Verifiche funzionali**: mai delegare a Code — le fa l'architetto.
- **Code solo per**: modifiche codice complesse, multi-file.

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

### B-001 — `duplicate labels` su universi ampi · RESOLVED
### B-002 — `reduce_grid_via_stability` rifiuta universo piccolo · RESOLVED
### B-003 — `compare_selection_columns` fallisce con `float not iterable` · RESOLVED
### B-004 — Presunto swing p-value B1 MC · CLOSED (non era un bug)
### B-005 — Narrativa 6.b/6.c sempre sul path Standard · RESOLVED
### B-006 — Nomenclatura invertita in `compute_skill_profile` · RESOLVED
### B-007 — `boxplot()` keyword `labels` deprecata · RESOLVED
### B-008 — `wfo_file_save` scriveva sempre il path Standard, mai il path scelto in §8 · RESOLVED (19/06)
### B-009 — `write_image` fan chart MC crash su Timestamp non serializzabile (kaleido/orjson) · RESOLVED (19/06)
### B-010 — `risk_off_tickers` sempre vuota in `run_r_portfolio_analysis` (chiave mai esistita in r_portfolios.py) → pf_rot Risk ON/OFF sempre None, fallback silenzioso su Base · RESOLVED (19/06, solo lato analisi — vedi gap aperto §0 priorità massima per il runtime)

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
Branch vari → main.

### Sessione 12/06/2026 — Architettura K-portfolio + CLI k-analyze
- Architettura 4 filiere approvata
- `iq k-analyze` unico comando inspector+panel
- `test_strategies.py` eliminato
- `k_strategy_inspector.ipynb` eliminato
- Pipeline K: WFO → OFC → DSR → MC → verdetto
- Cache su disco: `_results.pkl`, `_precheck.pkl`, `_mc_results.pkl`
- `iq k-agent` aggiunto alla CLI

### Sessione 14/06/2026 — Panel viewer + criteri promozione B&H

- `k_strategy_panel.ipynb` riscritto come viewer CSV puro
- Funzioni panel migrate in `k_functions.py`:
  `load_k_classifications`, `style_k_classification`, `load_k_equity`, `plot_k_equity`
- Criteri promozione completati: aggiunto `BH_Beat_Ret` e `BH_Beat_DD`
  (TotRet% > BH_TotRet% AND MaxDD% < BH_DD%)
- CSV arricchito: `TotRet%`, `BH_TotRet%`, `DeltaTotRet%`, `DeltaDD%`
- `k_strategy_inspector.ipynb` eliminato — assorbito da `iq k-analyze`
- `iq k-agent` testato end-to-end: genera `strategy_amd_momentum_rsi` ✓
- Crontab locale progettato (PTF target da definire)
- Documenti: `docs/K_STRATEGY_PIPELINE.md`, `docs/R_PORTFOLIO_PIPELINE.md`

### Sessione 15/06/2026 — Fix iq k-agent
- Default `--llm` cambiato da `ollama` ad `anthropic`
- Aggiunta opzione `--pdf PATH` per processare PDF locali (stesso
  path di generazione RSS)
- Deduplicazione PDF allineata a `processed_ids.json` (skip pre-LLM)
- Guardia su nome strategia in `k_strategies_agent.py` come secondo
  livello di sicurezza
  
### Sessione 15/06/2026 — K-Agent PDF + Trading System Analysis

- `iq k-agent`: aggiunta opzione `--pdf PATH` per processare PDF locali
  (stesso path di generazione RSS); default `--llm` cambiato a `anthropic`
- `iq k-agent`: deduplicazione PDF allineata a `processed_ids.json`
  (skip pre-LLM); guardia su nome strategia in `k_strategies_agent.py`
  come secondo livello di sicurezza
- `k_functions.py`: fix FutureWarning `fillna(method=)` → `ffill()`
  in `analyze_exposure_regime` e `analyze_timing_efficiency`;
  fix RuntimeWarning divide-by-zero in `analyze_timing_efficiency`
- `k_functions.py`: aggiunta `analyze_promoted_ts()` — analisi comparativa
  di tutti i trading system promossi, raggruppati per ticker; metriche:
  CAGR, Sharpe, MaxDD, PctInvested, CAGR_per_InvestedYear, InvestedCAGR,
  FlatCAGR, BH_CAGR, ExcessCAGR; legenda metriche inclusa
- `k_strategy_panel.ipynb`: aggiunta sezione §7 Trading System Analysis
  con load_ts, analyze_exposure_regime, analyze_timing_efficiency,
  generate_portfolio_performance, analyze_promoted_ts

### Sessione 16/06/2026 — Filiera Lazy portfolio + fix k-agent

**k-agent:**
- Fix model string deprecato: `claude-sonnet-4-20250514` → `claude-sonnet-4-6`
- Crontab `iq k-agent` attivato su `irina` (ore 02:00, --max 15, anthropic)
- Crontab `iq k-analyze` in attesa: universo ticker dipende da
  `select_top_performing_stocks` — bloccato fino a fine 2026

**Filiera Lazy — Fase A completata:**
- `mc_functions.py`: aggiunta pipeline headless `run_lazy_analysis()`
  con frequency selection automatica, frontiera efficiente, backtest;
  helpers `_cagr_from_equity()`, `_safe_metric()`
- `lazy_portfolio_analyst.ipynb`: riscritto con flusso pulito:
  §1 Configurazione → §2 Frequency Selection → §3 Frontiera efficiente
  (con legenda Real vs teorico) → §4 Backtest PTF proposto →
  §5 Backtest PTF ottimizzato → §6 Placeholder validazione statistica
- Separazione netta JN interattivo (Plotly) vs CLI headless (`run_lazy_analysis`)

**Prossimo (Fase B — validazione statistica Lazy):**
- Stability test pesi ottimali su sotto-periodi
- MC Block A: confidence intervals CAGR/Sharpe/DD
- MC Block B: skill ribilanciamento vs BH puro
- DSR: Deflated Sharpe Ratio
- Decisione finale + relazione tecnica PDF
### Sessione 18/06/2026 — Filiera Lazy Fase B completa + fix produzione

**Fix urgente produzione:**
- `iq report --ptf portfolio_alpha_sp100/nasdaq100` crashava: tickers
  simbolici ("sp100") non risolti. Fix in cli.py (risoluzione via
  alpha_sp100_tickers_by_year[anno]). Deployato su tslab.investia.cloud
  (release 2026.1, via deploy.sh).
- Documentata procedura deploy VPS (make_release.sh + deploy.sh) nel
  piano operativo — script esistenti non erano documentati.

**Filiera Lazy — Fase B completata + infrastruttura CLI/panel:**
- Naming coerente in l_portfolios.py: prefissi lazy_/equity_/sandbox_
  per distinguere PTF core multi-asset da portafogli azionari
  concentrati e sandbox/test. Sotto-registry L_PORTFOLIO_LAZY/EQUITY/
  SANDBOX oltre al generale L_PORTFOLIO_REGISTRY (auto-discovery).
- Criterio di stabilità ridisegnato: lazy_stability_weights (pesi
  Max Sharpe della frontiera su sotto-periodi) giudicato inadatto —
  misura overfit di un'allocazione teorica mai eseguita. Sostituito
  nel verdetto da lazy_rolling_stability: P(rendimento rolling a 5
  anni < 0%) sul PTF reale, riusando analyze_rolling_horizons già
  presente in u_functions.py per il triangolo dei rendimenti.
- iq lazy-analyze: comando CLI completo, --ptf <nome|all>, --override,
  cache pkl stabile in outputs/lazy_cache/ (per PTF + risultati MC).
- lazy_panel.ipynb: viewer completo, §1-§7 (config, classifica colorata,
  scatter, equity curve top-N su periodo comune, confronto, export,
  proiezione capitale futuro via percentili MC Block Bootstrap).
- project_lazy_capital: proietta capitale futuro (overview P50 +
  dettaglio per-PTF con banda P10-P90), riusa simulazioni MC già
  calcolate, nessun nuovo modello.
- Aggiunte 4 varianti strutturali (conservative 40/30/30, balanced
  60/20/20 esistente, aggressive 80/10/10, full_equity 95/5) per
  esplorare il vero spazio di scelta (quota equity), distinto dalle
  10 micro-varianti satellite esistenti che si sono rivelate
  sostanzialmente equivalenti (stessa shape equity curve).

**Osservazioni aperte, da affrontare in futuro:**
- MC_B_skill (skill ribilanciamento) risulta sempre False su tutti
  i PTF testati — o è un fatto vero (lazy non ha skill di timing,
  coerente con letteratura) o il test manca di potere statistico
  (jitter ±30gg, 500 sim). Non ancora investigato a fondo.
- Le proiezioni di capitale a 30 anni mostrano bande di incertezza
  enormi — corretto matematicamente (capitalizzazione composta) ma
  da comunicare con cautela: usano la distribuzione di un solo
  storico decennale (2016-2026, favorevole sia a equity che a oro).

### Sessione 19/06/2026 — CLI consistency, relazione tecnica Lazy, 3 bug critici R-portfolio

**CLI — naming e relazione tecnica Lazy:**
- Rename `analyze` → `r-analyze`, `lazy-analyze` → `l-analyze` (simmetria)
- Flag `--pdf` aggiunto a entrambi, default `False` (era sempre generata
  per R, mai opzionale): senza `--pdf` la pipeline + card `.md` girano
  comunque, senza generare il PDF
- `generate_relazione_tecnica_lazy()` nuova in `mc_functions.py`: §1-§7
  dedicati (Identità, Configurazione, Metriche comparative, Validazione
  statistica, Monte Carlo, Proiezione capitale, Decisione finale).
  Estratto `_compute_lazy_full()` da `run_lazy_batch_analysis` per esporre
  i risultati ricchi per-PTF necessari al PDF (stability, MC A/B, DSR,
  proiezione) che il batch scartava
- Help in linea uniforme per tutti i comandi `iq` + nuovo `docs/IQ_ADMIN_GUIDE.md`

**Tre bug trovati e risolti in cascata sulla pipeline R (vedi B-008/009/010):**
1. Save §8 scriveva sempre Standard su `wfo_file_save`, mai il path
   effettivamente scelto dall'architetto — fix: nuova cella dopo la
   decisione che salva il path scelto
2. Crash `write_image` su Timestamp non serializzabile (fan chart MC,
   kaleido/orjson) — risolto, causa esatta non confermata via traceback
   ma il fix ha eliminato il crash
3. `risk_off_tickers` sempre vuota lato analisi (B-010) — risolto;
   **scoperto nel farlo che il runtime di produzione non supporta Risk
   ON/OFF in alcuna forma** → nuovo item priorità MASSIMA

**Verificato e chiuso**: il meccanismo Risk ON/OFF (overlay "universo
allargato" post-selezione) è identico per Standard e Cluster — timore
iniziale di un meccanismo Cluster diverso/ignoto, infondato.

**Incidente operativo**: violata la regola "un branch alla volta" (4 branch
+ 1 stash aperti in parallelo durante la sessione) → richiesto un recovery
manuale (reset --soft, separazione commit, pulizia notebook sporchi).
Regola fissata esplicitamente per il futuro (vedi Stato attuale).

**Accantonato**: panel comparativo R-portfolio e save/load completo per
ri-analisi decisionale (item 6, priorità bassa) — volumi PTF troppo bassi
oggi per giustificare il lavoro.


## Deploy VPS — procedura standard

**Script disponibili in `scripts/`:**
- `make_release.sh [VERSION]` — crea release versionata da `notebooks/libs_py/` + `investia_quant/`. Senza argomento auto-incrementa (2026.1 → 2026.2).
- `deploy.sh VERSION VPS_HOST INSTALL_DIR` — rsync della release locale (`releases/VERSION/`) sulla VPS, reinstalla venv, aggiorna symlink `current`. Legge SOLO da `releases/VERSION/` in locale — NON fa fetch/pull da git, NON legge dalla root del repo.

**Conseguenza critica**: un fix in `investia_quant/cli.py` (root del repo) NON è automaticamente nella release. Va prima copiato/rigenerato dentro `releases/<versione>/` prima di deployare.

### Procedura hotfix su release esistente (senza bump versione)
```bash
cd ~/investia-quant
# 1. Copia il file corretto nella release attiva
cp investia_quant/cli.py releases/2026.1/investia_quant/cli.py
# (ripetere per ogni file modificato)

# 2. Deploy
./scripts/deploy.sh 2026.1 tslab.investia.cloud /home/luca
```

### Procedura release completa nuova (bump versione)
```bash
cd ~/investia-quant
./scripts/make_release.sh          # crea 2026.2 da root del repo
./scripts/deploy.sh 2026.2 tslab.investia.cloud /home/luca
```

**VPS**: `INSTALL_DIR=/home/luca`, `VPS_HOST=tslab.investia.cloud`. Rollback: `ssh tslab.investia.cloud "ln -sfn <versione_precedente> /home/luca/investia-quant/releases/current"`.