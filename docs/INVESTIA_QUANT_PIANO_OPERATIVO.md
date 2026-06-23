# investia-quant — Piano Operativo

**Ultimo aggiornamento**: 23 giugno 2026
**Root progetto**: `~/investia-quant`

---

## Stato attuale

**Branch `main`** aggiornato e pulito. Ultimi commit (sessione 22/06):
```
feat(r-cluster): pool di selezione titoli profile+regime-aware (AVOID eleggibile per satellite, mai per core) — non altera n_top
feat(r-cluster): tratta AVOID a pari merito con HIGH_MOMENTUM per profilo satellite (allocazione, selezione dominante per-finestra, proxy regime)
fix(r-portfolio): n_top=1 escluso per asset_type=stock anche su path Standard (era solo su Cluster) — risolve concentrazione estrema (es. SNDK 60x)
fix(report): risolvi universo sp100/nasdaq100 via Wikipedia (era alpha_sp100_tickers_by_year, lista K-portfolio sbagliata)
fix(r-portfolio): risk_off_tickers per-PTF con default da k_tickers (era sempre vuoto, pf_rot None silenzioso)
fix(mc-plot): serializzazione Timestamp in write_image dei fan chart MC (kaleido/orjson)
fix(r-portfolio): salva su wfo_file_save solo il path scelto in §8, non sempre Standard
```

**Working tree**: clean.
**Branch parcheggiati**: nessuno.

⚠️ **Regola operativa** (fissata il 19/06, rispettata il 22/06): **un branch
alla volta, commit + merge in `main` prima di aprirne un altro**.

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

**0.c Generazione narrativa della relazione tecnica via LLM** · filiera R · proposta di redesign, da valutare con calma

Emerso il 23/06 inseguendo un secondo bug di incoerenza nella relazione
(`_diagnose_mc` forzava `recommended_path=None` a `'std'` per
"retro-compat", scrivendo "path candidato al deploy" anche quando nessun
path supera l'OFC — stessa famiglia del bug del 22/06, sintomo di un
problema più ampio): la relazione tecnica oggi è generata da una
combinatoria di `if/elif` scritta a mano per ogni sezione (§3, §6.b, §7),
una per ciascuna combinazione di promosso/non-promosso × Standard/Cluster
× profilo × skill profile. Ogni nuovo parametro (oggi: `profile`) aumenta
la combinatoria e introduce rischio di sezioni che si disallineano tra
loro — già successo due volte in due giorni consecutivi.

Proposta valutata: separare nettamente **calcolo** (resta deterministico
Python — OFC, Sharpe/MaxDD/CAGR, `_select_path_by_profile`, tutto
invariato) da **narrazione** (LLM, una sola chiamata che riceve un
pacchetto fatti finale e immutabile — es. JSON con tutti i verdetti/numeri
già calcolati — e scrive §3/§6.b/§7 in un colpo, garantendo coerenza
interna per costruzione invece che per disciplina di chi scrive il
codice).

Coerente con "Agente generazione automatica relazioni tecniche (in
sviluppo)" già presente nel piano da prima, e con l'uso LLM già attivo
altrove nel progetto (agente K-strategy).

Rischi da non sottovalutare prima di procedere: hallucination numerica
(serve validare ogni cifra nel testo generato contro il pacchetto fatti,
non fidarsi alla cieca), riproducibilità (temperature=0 + salvataggio
dell'output come parte dell'audit trail), nessun problema reale di
costo/latenza per un report a PTF.

Non task immediato — cambiamento di architettura della relazione, non un
fix puntuale. Da discutere con calma in una sessione dedicata, non
implementare di getto.

**0.b Clustering ricalcolato per finestra IS — mai implementato** · filiera R · gap di design metodologico

Verificato il 22/06: l'intento di progetto era un clustering realmente
adattivo al mercato — ricalcolato per ogni finestra In-Sample (solo dati
fino a quel punto) e testato Out-Of-Sample sulla finestra successiva,
coerente col principio WFO stesso. Non è mai stato implementato:
`run_clustered_wfo` riceve oggi una partizione cluster già calcolata
(STEP 1, una sola volta, su dati recenti fino a "oggi" — lookback_days=504)
e la applica identica a tutte le 13 finestre storiche, comprese quelle di
10 anni fa. Bias look-ahead sulla composizione dei cluster (la struttura
di mercato di oggi decide come si raggruppavano i titoli nel 2015),
distinto dal bias già noto sull'universo simbolico (Wikipedia oggi
applicato a tutto lo storico, mai risolto, resta accettato come limite
noto del motore).

Idea valutata e giudicata metodologicamente corretta, ma non banale:
- Costo computazionale: 13x il costo della fase di clustering (oggi 1
  esecuzione, diventerebbero 13)
- Rischio rumore: finestre iniziali (2012-2014) con poco storico
  potrebbero produrre partizioni instabili — rischio di sostituire un
  bias sistematico con rumore casuale, da verificare con dati reali
- Coerenza col pool eleggibile profile+regime appena costruito (22/06):
  se la partizione cambia per finestra, un titolo può cambiare label nel
  tempo → turnover più alto, e il motore oggi non modella commissioni
  (Total Fees Paid: 0.0 in tutti i log) — diventerebbe più rilevante

Non task immediato (i fix del 22/06 hanno già dato miglioramento concreto
senza questo), ma il prossimo grande tema metodologico per la filiera R —
tocca la validità di fondo dei numeri presentati in relazione tecnica.
Richiede design session dedicata prima di assegnare a Code.

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
### B-011 — `iq report` risolveva l'universo sp100/nasdaq100 con `alpha_sp100_tickers_by_year` (lista K-portfolio, top performer annuali) invece che dinamicamente via Wikipedia come `iq run`/`iq r-analyze` — selezioni e statistiche storiche non coincidenti con l'esecuzione reale, specialmente a cavallo di cambio anno · RESOLVED (22/06)
### B-012 — `n_top` in `build_cluster_grids` calcolato come percentuale della dimensione cluster (fino a 14+ per cluster grandi) invece che range assoluto — diluiva la concentrazione del cluster HIGH_MOMENTUM, vanificandone lo scopo · RESOLVED (22/06, introdotta lookup `resolve_n_top(asset_type, profile)`)
### B-013 — stesso problema di B-012 ma sul `full_grid` del path Standard: `n_top=1` non escluso per `asset_type="stock"` → concentrazione totale su singolo titolo (caso reale: SNDK, rally ~60x in un anno, catturato al 100% da `n_top=1` nella finestra 2026) · RESOLVED (22/06, stessa `resolve_n_top` riusata)
### B-014 — `_recommended_path()` (tie-break fisso "Cluster vince se promosso") disallineata da `_build_verdict_text` CASO C (Sharpe con soglia 0.05) quando entrambi i path PROMOTED — §3 e §7 potevano mostrare raccomandazioni opposte nello stesso documento; criterio inoltre cieco al profilo (satellite/core) · RESOLVED (23/06, unificato in `_select_path_by_profile`: satellite=Sharpe soglia 0.05 invariato, core=MaxDD con eccezione se CAGR < benchmark)
### B-015 — `_diagnose_mc` forzava `recommended_path=None` a `'std'` (fallback dead-code) quando nessun path supera l'OFC, dichiarando "Standard candidato al deploy" in §6.b in contraddizione con §7 ("deploy non raccomandato") · RESOLVED (23/06, early return per caso genuino None — §6.b/§3/Fig.4 ora mostrano entrambi i path senza endorsement)

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

### Sessione 22/06/2026 — Coerenza profile/asset_type su tutta la filiera R, 3 bug critici, 1 redesign cluster

**Trigger**: verifica visiva del PDF `Alpha Nasdaq100` ha rivelato numeri
assurdi (Cum Return path Standard: 75774%) — non accettato come "normale",
indagine a fondo fino alla causa reale invece di patch superficiali.

**Bug trovati e risolti in cascata (vedi B-011/012/013):**
1. `iq report` usava l'universo K-portfolio (lista curata 25 ticker) per
   PTF R-portfolio a universo simbolico (sp100/nasdaq100) — selezioni e
   statistiche non coincidenti con l'esecuzione reale. Causa scoperta
   grazie a un confronto con calcolo esterno indipendente (Luca+Emanuele),
   prima dell'invio ai gestori — nessun danno verso clienti, solo verso
   le statistiche interne in fase di verifica.
2. `n_top` proporzionale alla dimensione cluster (fino a 14+) in
   `build_cluster_grids` — diluiva la concentrazione di HIGH_MOMENTUM.
   Fix: lookup `resolve_n_top(asset_type, profile)`, con tabella esplicita
   (stock/etf × satellite/core), 1 escluso per singoli titoli (rischio di
   concentrazione totale su un solo nome).
3. Stesso problema riscontrato anche sul `full_grid` Standard (gap della
   patch precedente, non propagata lì) — causa diretta di un episodio
   reale: `n_top=1` ha concentrato l'intero PTF su SNDK (rally ~60x/anno,
   verificato NON essere dato corrotto — variazioni giornaliere max 28%,
   nessun salto anomalo) per la finestra 2026.

**Redesign cluster AVOID/HIGH_MOMENTUM (4 fix in cascata, stesso filo):**
- Scoperta che `aggregate_cluster_portfolios` (meccanismo a pesi fissi)
  è dead code — mai chiamata da nessun punto della pipeline reale
- Il meccanismo vivo è `merge_cluster_summary_dfs`: winner-take-all per
  finestra, con fallback arbitrario (`available[0]`, ordine di dict) se
  la label richiesta (default `HIGH_MOMENTUM`) non esiste nella partizione
- Fix 1: selezione del cluster dominante per-finestra basata su confronto
  diretto di TestScore tra label ammesse (satellite: AVOID vs HIGH_MOMENTUM
  a pari merito; core: mai AVOID), non più nome fisso né fallback arbitrario
- Fix 2: proxy di calcolo regime (`equity_tickers` per `compute_market_regime`)
  esteso per coerenza — satellite include AVOID∪HIGH_MOMENTUM, core solo
  HIGH_MOMENTUM (prima: solo HIGH_MOMENTUM sempre, fallback su tutto
  l'universo se assente)
- Fix 3 (il più rilevante): il pool di titoli eleggibili per la selezione
  finale (non solo i parametri) ora dipende da profilo×regime — satellite
  ON pesca da AVOID∪HIGH_MOMENTUM, non solo dal cluster vincente. `n_top`
  resta invariato (nessuna alterazione del numero di posizioni). Risultato
  su Alpha Nasdaq100: Cluster CAGR 26.5%→46.1%, Sharpe 0.97→1.46, Skill
  Profile No-skill→Selection-driven (B1 ora PASS)

**Gap di design scoperto e valutato, non risolto oggi**: la partizione
cluster è calcolata una volta su dati recenti e applicata identica a
tutte le finestre storiche — bias look-ahead distinto da quello
sull'universo. Vedi priorità alta, item 0.b.

**Confermato indipendentemente, nessuna azione**: il meccanismo Risk
ON/OFF (overlay "universo allargato") è identico per Standard e Cluster —
timore iniziale di un meccanismo Cluster diverso, infondato.

**Metodo di sessione**: ogni fix preceduto da lettura diretta del codice
reale (mai supposizioni accettate senza verifica grep/sed), inclusi alcuni
errori di percorso recuperati esplicitamente (es. proposta iniziale di
blend di rendimenti scartata perché concettualmente sbagliata — si stanno
selezionando titoli, non simulando scenari MC).

### Sessione 23/06/2026 — Incoerenza Standard/Cluster nella relazione (continuazione), profile-awareness raccomandazione

**Bug 1 — incoerenza §3 vs §7 (criterio di raccomandazione disallineato)**:
`_recommended_path()` (usata da §3 e dal box alternativo §7) usava tie-break
fisso "Cluster vince se promosso"; `_build_verdict_text` CASO C (verdetto
principale §7) usava invece confronto Sharpe con soglia 0.05. Le due
potevano disaccordare quando entrambi i path erano PROMOTED — verificato
su Alpha Nasdaq100 satellite: §3 evidenziava Cluster, §7 raccomandava
Standard. RESOLVED: criterio unificato in `_select_path_by_profile()`,
ora **profile-aware** (non lo era nemmeno nella versione precedente):
satellite → Sharpe con soglia 0.05 (comportamento storico, invariato);
core → MaxDD più basso, con eccezione se il CAGR di quel path è inferiore
al benchmark (capital preservation non ha senso se non batte nemmeno il
benchmark). Test sintetici 3/3 confermati prima del rerun reale.

**Intervento minimo collaterale**: nome PDF/card `.md` non includeva
`profile` — due run stesso giorno con profili diversi si sovrascrivevano.
Fix diretto (un comando sed, no Code): aggiunto `{profile}` al nome file.

**Bug 2 — trovato testando il caso "nessuno promosso" (profile=core su
Alpha Nasdaq100)**: `_diagnose_mc` forzava `recommended_path=None` a
`'std'` ("retro-compat", verificato essere dead code — l'unico chiamante
passa sempre entrambi gli ofc_report), scrivendo "Il path candidato al
deploy è Standard" in §6.b anche quando l'OFC non promuove nessun path —
mentre §7 correttamente concludeva "deploy non raccomandato". RESOLVED:
early return per `recommended_path is None` — §6.b ora mostra B1/B2/Skill
Profile per entrambi i path senza dichiarare alcun candidato; tabella §3
e caption Fig.4 anch'esse senza highlight/riferimento fisso quando nessuno
è promosso. Verificato sul PDF rigenerato: documento internamente
coerente da §3 a §7, nessuna contraddizione residua.

**Proposta emersa, non implementata**: generazione narrativa via LLM
invece di combinatoria if/elif scritta a mano — vedi priorità alta, item
0.c. Motivata direttamente da questi due bug consecutivi, stessa famiglia
di causa (sezioni diverse, logiche scritte a mano separatamente, rischio
di disallineamento ad ogni nuovo parametro).

**Incidente recuperato**: confusione tra macchine (`irina`/`adriana`) —
`origin/main` su `irina` sembrava non avere i commit del 22/06; causa
reale: `irina` non aveva fatto `git pull` recente, non un problema di
lavoro perso. Verificato che il lavoro era già su `adriana`, committato e
pushato. Nessuna perdita, solo allarme eccessivo per non aver controllato
con un semplice fetch prima di concludere il peggio.


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