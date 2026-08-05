# investia-quant — Piano Operativo

**Ultimo aggiornamento**: 4 agosto 2026 (famiglie rotazionali, criteri nuovi engine, regola di verifica)
**Root progetto**: `~/investia-quant`

> **Nota consolidamento 22/07**: le voci "0.b" e "0.d-bis" sono state
> unificate in un'unica voce "Cluster ridisegnato come pre-filtro di
> selezione universo annuale". La voce "0.c" (narrativa relazione
> tecnica via LLM) è stata scartata. L'item "5. Potenziamento Block B"
> è stato unificato con la discussione sui nuovi engine di rotazione
> del 21/07. Le voci "OFC S2 coherence" e "dubbio meta-overfitting
> sistema OFC" (discusse il 30/06 in altra sede, mai entrate in questo
> file) sono state scartate e non compaiono più tra i task aperti.

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

### Risolti

**0. Risk ON/OFF non applicato dal runtime di produzione** · filiera R · ✅ RISOLTO 28/06

Scoperto il 19/06, chiuso definitivamente il 28/06.

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

**Risoluzione (28/06)**: §8 del notebook ora salva anche
`deployed_variant` ("BASE" | "RISK_ON_OFF") in `extra_meta`;
`r_run_portfolio` legge il metadata via `read_wfo_metadata` e applica
`apply_risk_off_overlay` (funzione estratta e condivisa con
`run_wfo_pipeline`) prima della selezione, solo per path STANDARD. Path
CLUSTER esplicitamente bloccato con errore parlante (vedi item 0.d).
Verificato end-to-end due volte su `portfolio_germany_plan`
(`iq promote` + `iq run -v` →
`[INFO] Risk ON/OFF overlay applicato: [...]`).

Incidente operativo nella stessa sessione: il fix è stato inizialmente
committato sul branch sbagliato (`fix/report-path-and-sections-parity`,
branch vecchio con lavoro non revisionato accumulato da settimane —
vedi nota tech debt in fondo al documento), poi recuperato con
cherry-pick mirato (`d05fa14` + `2aa57dd`, incluso il recupero di
`iq promote`, scoperto mancante solo in fase di verifica finale) su
branch dedicato e correttamente mergiato in `main`. Nessuna perdita di
lavoro, ma lezione operativa: verificare sempre `git branch` prima di
un commit quando si torna da un `checkout` precedente.

### Priorità MASSIMA

**0.d Path Cluster — diagnosi precisa: look-ahead nella selezione, non solo gap di persistenza** · filiera R · ⚠️ aggiornato 28/06 sera

**Aggiornamento rispetto alla prima diagnosi (28/06 pomeriggio):** la
causa profonda del path Cluster non è "solo" un problema di
persistenza/runtime — è un **look-ahead metodologico nella selezione
stessa**, identificato analizzando le performance anomale (CAGR
19.8%, Sharpe 0.97, Cluster–Risk ON/OFF) emerse da un run reale.

**Diagnosi precisa**: `merge_cluster_summary_dfs` decide, per ogni
finestra storica, quale gruppo (label cluster, es. HIGH_MOMENTUM vs
AVOID) è "vincente" confrontando il `TestScore` — cioè il punteggio
**Out-of-Sample della finestra stessa**. Una WFO disciplinata seleziona
i parametri guardando solo l'In-Sample (TrainScore); qui invece la
selezione del gruppo usa il risultato del test che dovrebbe solo
*misurare*, non *scegliere*. È un leakage a livello meta: si riporta
come "skill del clustering" il fatto che, scelto a posteriori il
migliore tra N sotto-universi, quello batte la media — vero quasi per
costruzione. Coerente con l'evidenza già raccolta in relazione tecnica:
Skill Profile "No-skill" ai test Monte Carlo Block B anche quando
CAGR/Sharpe Cluster erano superiori — i test di permutazione non
trovano skill perché il vantaggio non viene dalla rotazione, viene
dalla selezione retrospettiva del sotto-universo.

**Tentativo di refactor (rimozione completa del codice Cluster) —
ANNULLATO il 28/06 sera**: si è tentato di rimuovere interamente il
path Cluster da `run_wfo_pipeline`/`run_r_portfolio_analysis` (branch
`refactor/remove-cluster-path`). Il tentativo ha causato una catena di
incidenti operativi (3 funzioni generali del motore — `resolve_n_top`,
`run_wfo_pipeline` stessa, `apply_risk_off_overlay` — cancellate per
errore insieme al blocco Cluster, perché fisicamente intercalate nello
stesso range di righe; notebook con autosave del browser che ha
vanificato più `git restore`) — **branch annullato**, si è tornati a
`main` con il path Cluster **ancora presente e funzionante nel
codice**.

**Decisione operativa attuale (diversa da quella di metà giornata)**:
il path Cluster **non viene rimosso dal codice**. Resta disponibile
per chi volesse eseguirlo consapevolmente, ma:
- non è eseguito di default (vedi item 0.d-ter, flag CLI in arrivo)
- la relazione tecnica/PTF card/stampa decisione ora gestiscono
  correttamente il caso "Cluster non eseguito" (vedi 0.d-quater sotto)
  invece di mostrare dati fuorvianti

**Principio per il futuro, confermato e invariato**: qualsiasi
redesign futuro (Cluster v2, o altre varianti WFO) deve produrre un
artefatto runtime-applicabile per costruzione E non usare mai, nella
selezione, un dato che esiste solo a posteriori (TestScore o
equivalente) — non solo per ragioni di eseguibilità a runtime, ma per
validità statistica della misura stessa.

---

**0.d-quater Relazione tecnica, PTF card, stampa decisione — fallback fuorvianti su Cluster=None** · filiera R · ✅ RISOLTO 28/06 sera

Scoperto generando un PDF reale (Germany Plan) con Cluster non
eseguito: tre funzioni (`generate_relazione_tecnica`,
`generate_ptf_card_md`, `print_final_decision`) avevano fallback
impliciti che **copiavano i valori del path Standard** quando i
corrispondenti dati Cluster erano `None` — risultato: sezioni "Path
Cluster" nella relazione tecnica mostravano tabelle MC Skill Tests e
Confidence Intervals **identiche, alla cifra decimale**, a quelle
Standard, sotto etichetta Cluster — fuorviante, sembrava una
validazione indipendente quando era una copia. Causa: pattern
`else: usa_valore_standard` ripetuto in tre punti indipendenti invece
di propagare `None` → "N/A".

Risolto in tutte e tre le funzioni: ogni sezione/riga/cella relativa a
Cluster mostra ora "N/A" in modo coerente quando il dato è `None`, mai
un valore copiato da Standard, mai una cella vuota (anche questo
verificato: la cella "OFC Verdict" mostrava vuoto invece di N/A,
corretto separatamente). Verificato su PDF reale rigenerato — §3 testo
intro, §4.b, §5.a.2, §5.b.2, §7, stampa a schermo: tutti coerenti.

**Nota collaterale**: `compare_wfo_pipelines` (funzione di confronto
usata in sviluppo JN §5c) richiedeva `results_cluster` come parametro
obbligatorio — patchato per renderlo opzionale (default `None`), con
omissione pulita di tabelle/plot Cluster quando assente. File pronto,
da incollare in `r_functions.py` (non ancora applicato/commesso).

---

**0.d-ter Flag CLI per disabilitare Cluster di default** · filiera R · ✅ RISOLTO 28/06 sera

`run_r_portfolio_analysis` ora accetta `run_cluster: bool = False` —
di default esegue solo il path Standard, saltando interamente la
pipeline Cluster (WFO, download dati specifici, OFC, MC). `iq r-analyze`
esegue Cluster solo se richiesto esplicitamente. Committato e pushato
in `main` (28/06 sera).

**Nota per quando 0.d-bis (filtro Cluster v2) arriva in Fase 3
(validazione storica) o oltre**: rinominare il flag CLI attuale da
`--cluster` a `--cluster-legacy`, per evitare ambiguità tra il vecchio
meccanismo (look-ahead noto, solo per studio/confronto, mai per
certificazione) e il nuovo filtro v2 (pre-selezione d'universo
annuale, senza look-ahead). Riservare `--cluster` per il filtro v2
quando pronto — non riusare lo stesso nome per due logiche diverse,
per non confondere risultati storici già generati con quel flag.
Nessuna azione richiesta ora: rinominare solo quando si arriva
effettivamente a implementare la Fase 3 di 0.d-bis.



---

**Audit completo del motore rotazionale (28/06)** — eseguito per
contestualizzare il problema Cluster, ha rivelato altri gap minori non
ancora in questo piano. Aggiunti come item separati: 0.e (deriva
universo dinamico sp100/nasdaq100), 0.f (parametri motore non nel
param_grid, rischio futuro), 0.g (nessun guard su file WFO errato/anno
sbagliato), 0.h (risk_off_tickers identici per tutti i PTF inclusi
quelli US).

### Priorità media (da audit 28/06)

**0.e Deriva universo dinamico (sp100/nasdaq100)** · filiera R

I parametri WFO sono calibrati in analisi su un universo risolto "oggi"
da Wikipedia; a runtime `iq run` lo risolve di nuovo, in un momento
diverso — se la composizione dell'indice cambia tra analisi ed
esecuzione mensile, l'universo runtime può differire da quello
validato. Zero errori, zero warning. PTF impattati: `alpha_sp100`,
`alpha_nasdaq100`.

**0.f Parametri motore (`ema_span`, `volatility_quantile`,
`min_momentum_threshold`) assenti dal param_grid** · filiera R

Non salvati come colonne CSV — runtime usa default fissi
(`EngineParams.from_dict`, r_functions.py:184-205) che oggi coincidono
con quelli di analisi. Nessun gap oggi, ma se questi parametri venissero
resi variabili in analisi senza aggiornare il lato runtime, la
divergenza sarebbe silenziosa.

### Priorità bassa (da audit 28/06)

**0.g Nessun guard su TrainScore/file WFO errato** · filiera R

`extract_operational_params_from_summary` valida solo il mismatch di
anno — un file con anno corretto nell'header ma contenuto/path
sbagliato passerebbe senza errori.

**0.h risk_off_tickers identici per tutti i PTF, anche quelli US** ·
filiera R · informativo

Tutti i 10 PTF usano lo stesso default (`XEON.MI`, `IBTS.MI`,
`XAD5.MI` — ETF europei) come ticker difensivi, inclusi PTF azionari US
(`alpha_sp100`, `alpha_nasdaq100`). Calendario/liquidità non verificati
per coerenza con l'orario di esecuzione di `iq run`.

### Architettura di estensione del motore rotazionale — due punti di aggancio (28/06)

Principio emerso ridisegnando il filtro Cluster per il 2027 (vedi 0.d):
il motore di rotazione (`run_rotational_engine`/`walk_forward_rotational`,
ora semplicemente **WFO**, senza più distinzione Standard/Cluster come
due pipeline parallele) resta **unico e immutato**. Ogni evoluzione
futura si aggancia in uno solo di due punti, entrambi runtime-sicuri per
costruzione:

**Punto 1 — Filtri di pre-selezione (a monte della WFO, sull'universo)**
Restringono *chi* è candidato, non toccano il motore. Calcolati una
volta all'anno con dati disponibili fino a quel momento (mai
informazione futura), producono una lista di ticker, persistita nel
summary — il runtime la legge, nessun ricalcolo. Esempio: il filtro
Cluster ridisegnato (0.d). Altri esempi plausibili stesso pattern:
liquidità, settore, qualità fondamentale, ESG.

**Punto 2 — Nuovi criteri di rotazione (dentro la WFO, nel ranking/
selezione)**
Nuovi `EngineParams` (come oggi `momentum_weight`, `use_acceleration`,
`filter_ema`) — già runtime-sound per costruzione attuale: ogni nuovo
parametro del motore, se segue la convenzione esistente
(`EngineParams.from_dict`, colonna nel summary, default coerente), è
letto a runtime senza lavoro aggiuntivo. Coerente con l'item 5
"Potenziamento Block B" (fonti di skill alternative al momentum) già
presente nel piano.

**Perché conta**: il problema del Cluster nasceva dal mescolare le due
categorie nella stessa pipeline (filtro d'universo + logica di
selezione dinamica per-finestra, codificata insieme). Separandole, il
motore resta semplice qualunque cosa si aggiunga dopo — nessuna nuova
pipeline parallela, solo nuovi filtri a monte o nuovi parametri dentro
lo stesso motore.

---

### In corso — Design 2027

**Cluster ridisegnato come pre-filtro di selezione universo annuale** · filiera R · design 28/06, unifica le precedenti voci 0.b e 0.d-bis — consolidato 22/07

Sostituisce il vecchio path Cluster (accantonato, vedi sopra) con un
filtro a monte della WFO standard, secondo il principio "Punto 1" sopra.
Assorbe anche l'esigenza di un clustering realmente adattivo al mercato
(intento di progetto originario, mai implementato prima d'ora): il pool
deve riflettere il regime corrente, non una partizione statica applicata
identica a tutte le finestre storiche.

```
1. CALCOLO FILTRO (una volta/anno, su dati fino a oggi)
   pool = analyze_and_cluster_universe(universo_pieno, lookback_days=504)
        + compute_market_regime(...)
   → lista ticker eleggibili per l'anno (no PTF senza filtro: pool = universo pieno)

2. WFO (motore unico, sempre walk_forward_rotational, nessuna variante)
   gira su `pool` — nessuna logica path-aware nel motore

3. SALVATAGGIO — pool persistito nel summary (non più solo in memoria)

4. RUNTIME (iq run) — legge summary_df + pool, scarica dati solo per
   pool, selezione runtime = selezione validata per costruzione
```

Elimina entrambi i difetti del path Cluster vecchio: nessun confronto
`TestScore` retrospettivo (il pool usa regime corrente, non performance
storica per-finestra), pool sempre persistito (non più solo
`summary_df_std`).

**Da fare prima dell'implementazione**: la validazione storica
(Sharpe/CAGR/Skill Profile, es. Alpha Nasdaq100 26.5%→46.1% CAGR,
0.97→1.46 Sharpe) era su un motore diverso (`TestScore` per finestra)
— va rifatta da capo con questa nuova definizione di pool, non
assunta valida. Nessuna garanzia che i numeri reggano identici, anche
se il principio di fondo (separare titoli per comportamento) resta
plausibile.

**Step 0 (priorità immediata, prima del design 2027)**: epurazione del
codice — rimuovere il vecchio path Cluster (pipeline parallela,
`use_clustering=True`, `merge_cluster_summary_dfs`, selezione dominante
per-finestra via TestScore) lasciando solo la WFO standard (rinominata
semplicemente **WFO**, non più "path Standard" in opposizione a
Cluster). Verificare che la WFO unica funzioni identica a se stessa
prima e dopo la rimozione (nessuna regressione). Il filtro Cluster
ridisegnato si aggiungerà SOPRA questa base pulita, non sopra il codice
vecchio.

**Punti aperti ereditati dalla vecchia voce 0.b (clustering per finestra IS)**,
da risolvere in fase di design del pre-filtro, non scartati:
- Costo computazionale: 13x il costo della fase di clustering se si
  ricalcola per ogni finestra storica invece che una volta/anno
- Rischio rumore: finestre iniziali (2012-2014) con poco storico
  potrebbero produrre partizioni instabili
- Turnover più alto se la partizione cambia nel tempo, e il motore oggi
  non modella commissioni (Total Fees Paid: 0.0 in tutti i log) —
  diventerebbe più rilevante

Richiede design session dedicata prima di assegnare a Code. Tocca la
validità di fondo dei numeri presentati in relazione tecnica.

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


**5. Nuove fonti di skill per la rotazione (potenziamento Block B)** · filiera R · unifica "Potenziamento Block B" e la discussione "nuovi engine di rotazione" (21/07) — consolidato 22/07

Motivazione: la MC (Block B, skill tests B1/B2) non mostra skill
significativo per i motori attuali (momentum, multifactor) in diversi
PTF/anni — molti PTF battono comunque il benchmark, ma per altri driver
(clustering Ward + risk-off), non per skill di selezione/rotazione.

Prima domanda da chiarire in design session, prima di scegliere una
direzione: la MC boccia *tutti* i PTF/anni o solo alcuni? Il problema è
nel motore di ranking o nella logica di rebalance timing? La risposta
cambia quale famiglia di alternative ha senso esplorare per prima.

Sorgenti/famiglie candidate:
- Risk-adjusted (Sharpe/Sortino rotazionale)
- Idiosyncratic return (residuo rispetto benchmark)
- Low-volatility (skill nell'evitare drawdown)
- Quality factor
- Multi-factor composito
- Cross-sectional mean reversion / dispersion trading (ranking per
  deviazione da un equilibrio relativo, non momentum assoluto)
- Correlation regime rotation (rotazione su cambi di regime di
  correlazione tra asset, non sui rendimenti individuali)
- Clustering dinamico come motore di rotazione stesso (rotare tra
  cluster, non solo usarlo per la selezione del path)
- Volatility-targeting / risk-parity rotation
- Trend strength filtering (qualità del trend, es. R²/efficiency ratio
  di Kaufman, non solo direzione/ampiezza)
- Regime-switching con HMM
- Ensemble/voting tra fattori deboli (skill emergente da combinazione,
  anche se i singoli fattori non passano MC da soli)

Da fare: design session prima di toccare codice — inclusa la domanda
di cui sopra sull'estensione reale del problema.

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
- `fix/report-path-and-sections-parity`: branch vivo dall'8/06, ~9
  commit non mergiati in main (incluso lavoro su k-agent, l-analyze,
  accesso remoto Adriana). Da mappare e integrare con calma, non in
  fretta — causa dell'incidente di commit-sul-branch-sbagliato del
  28/06. Contiene anche `get_analysis_output_dir` (recuperata
  singolarmente il 28/06) e probabilmente altro lavoro di
  normalizzazione path ancora da valutare.
- **Lezione operativa (28/06)**: l'autosave di Jupyter può sovrascrivere
  un file appena ripulito con `git restore` se il kernel/la tab del
  notebook restano aperti — prima di un `git restore` su un `.ipynb`,
  chiudere/fare shutdown del kernel; verificare con `git status` dopo,
  non assumere che il restore abbia tenuto.
- **Lezione operativa (28/06)**: rimuovere codice "per intervallo di
  righe" (range di un diff/refactor ampio) rischia di cancellare
  funzioni generali fisicamente intercalate con quelle da rimuovere,
  anche se logicamente indipendenti — verificare sempre per confine di
  funzione, non per range di righe, specialmente su file con funzioni
  storicamente intrecciate (es. `r_functions.py`).

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
  Sono verifiche funzionali: il numero è corretto? il PDF è leggibile?
  il comportamento è quello atteso? Richiedono giudizio sul dominio.
- **Verifiche fattuali statiche**: sempre delegare a Code, in sola
  lettura. Sono verifiche fattuali: cosa contiene il param_grid, che
  firma ha la funzione, dove viene scritto un campo, quali branch hanno
  commit non mergiati. Non richiedono giudizio, richiedono di aprire il
  file. **Questa voce corregge la precedente formulazione "Code solo
  per: modifiche codice complesse, multi-file"**, che di fatto
  scoraggiava l'uso dello strumento proprio dove costa meno e serve di
  più.

### ⚠️ Regola di verifica (fissata 04/08/2026)

> **Nessuna affermazione sullo stato di codice, file, repo o documento
> senza averlo aperto nella sessione corrente.** Ogni affermazione va
> citata con `file:riga` o con l'artefatto da cui è letta. Un'affermazione
> priva di citazione è **non verificata** e va marcata come tale, non
> presentata come fatto.

**Come si verifica**, in ordine di preferenza:
1. Aprire il file (`view`, `sed -n`)
2. Code in sola lettura, con un prompt mirato
3. Se nessuna delle due è possibile: **dichiarare che non è verificato**

**Cosa NON è verifica:**
- dedurre da questo piano — il 03-04/08 tre voci su tre verificate sono
  risultate stantie (`compare_wfo_pipelines`,
  `fix/report-path-and-sections-parity`, `feature/ranking-multifactor-v2`)
- dedurre da un'affermazione di seconda mano — è così che
  `ECOSISTEMA_INVESTIA.md` v2.6 è stato scambiato per v1.3, con un piano
  costruito sopra che cancellava undici sessioni di lavoro
- dedurre da un comportamento osservato — il fallimento di B2 su Germany
  Plan è stato preso come prova che `rebalance_frequency` fosse imposto,
  mentre è un `EngineParams` cercato dalla WFO (`r_functions.py:2088-2089`)

**Perché la regola è severa**: questi sono sistemi di investimento, non
prototipi. Il costo di un'inferenza sbagliata non è un bug visibile ma
un documento di progetto che diventa la base di decisioni successive —
e nessuno rilegge un'affermazione plausibile. È la stessa classe di
errore dei fallback silenziosi corretti in `cert-monitor` il 29/07: la
posizione più esposta del portafoglio mostrata come la più sicura,
perché un valore mancante era stato sostituito da una stima plausibile
invece che da un errore parlante.

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
invece di combinatoria if/elif scritta a mano, motivata direttamente da
questi due bug consecutivi (sezioni diverse, logiche scritte a mano
separatamente, rischio di disallineamento ad ogni nuovo parametro).
Proposta valutata e poi **scartata il 22/07** — non più in coda.

**Incidente recuperato**: confusione tra macchine (`irina`/`adriana`) —
`origin/main` su `irina` sembrava non avere i commit del 22/06; causa
reale: `irina` non aveva fatto `git pull` recente, non un problema di
lavoro perso. Verificato che il lavoro era già su `adriana`, committato e
pushato. Nessuna perdita, solo allarme eccessivo per non aver controllato
con un semplice fetch prima di concludere il peggio.

### Sessione 03/08/2026 — Relazione investitore: anagrafica portafoglio, ptf_type/thesis verificato

**Allineamento macchine**: `adriana` era indietro di 38 commit su
`cert-monitor` e 45 su `investia-platform` pur dichiarando
`## main...origin/main` senza divergenze — `verifica_progetti.sh` non fa
`fetch`, quindi mostra lo stato dell'ultimo contatto col remoto, non
quello reale. Stessa dinamica dell'incidente del 23/06, a parti
invertite. Aggiunti `session_start_all.sh` / `session_end_all.sh`:
wrapper multi-repo sugli script esistenti, che lavorano su tutti e tre i
progetti invece che sul solo repo corrente — lavorando su
investia-platform si toccano per definizione più repo, e chiudere la
sessione da una sola directory lascia gli altri indietro.

**Stash**: 6 stash accumulati dal 29/06, invisibili sia a `git status`
sia a `git branch` — nessuno degli script di verifica li elencava. Ora
inclusi nel report di `session_start_all.sh`. Eliminati dopo verifica.

**`k_strategies_agent.py` committato dal cron**: il file è l'output
giornaliero dell'agente, riscritto ogni notte alle 02:00. Essendo
tracciato, lasciava il working tree sporco tutti i giorni e bloccava
ogni allineamento. Il cron ora committa e pusha in coda al run:

```
00 02 * * *  .../iq k-agent --max 15 --llm anthropic >> .../k_agent.log 2>&1 && cd /home/luca/investia-quant && git add notebooks/libs_py/k_strategies_agent.py && git commit -q -m "chore(k-agent): strategie $(date +\%F)" && git push -q
```

Nota: se il push fallisce (repo indietro) o l'agente non produce nulla
di nuovo, la catena `&&` si interrompe in silenzio. Il lavoro resta
committato in locale, ma può accumularsi per giorni senza segnale.

#### `ptf_type` / `thesis` — verificato, nessun difetto

Sospetto iniziale: i campi `ptf_type: "thematic"` e `thesis` introdotti
in `r_portfolios.py` non comparivano nella relazione tecnica di Germany
Plan. Verificato: alimentano `generate_relazione_investitore_llm`
(`r_functions.py:13119`), non la relazione tecnica. La relazione tecnica
è un referto statistico neutro; la tesi d'investimento inquadra i numeri
per chi legge da investitore. Entrambi i rami del prompt condizionale
(riga 13191) funzionano come previsto:

- **thematic** (Germany Plan): alpha rolling in accelerazione descritto
  come conferma attesa della tesi
- **systematic** (default, Portafoglio Multi-Fondo): stabilità descritta
  come pattern da monitorare, non come vantaggio strutturale

⚠️ **Default silenzioso** a riga 13143: `ptf_def.get("ptf_type",
"systematic")`. Se `ptf_def` non è il dict giusto, o il campo manca, la
relazione esce con l'interpretazione sbagliata senza alcun segnale.
Stesso pattern del fallback di `resolve_n_top` (`.get(..., [3,5,8])`),
dove un `asset_type` scritto male ricade sul range stock togliendo
`n_top=1` senza avviso. Nessuno dei due è un problema attuale — i valori
in `r_portfolios.py` sono corretti — ma sono difetti che si scoprono
tardi.

#### Fix `_build_portfolio_anagrafica_table` · ✅ branch `fix/anagrafica-tickers-lista`

`ValueError: dictionary update sequence element #0 has length 6; 2 is
required` alla generazione della relazione investitore su Germany Plan.

Causa: riga 13392, `dict(portfolio["tickers"])`. La funzione era stata
estesa ai Lazy portfolio, che hanno i pesi come mapping ticker→peso; per
un R-portfolio `tickers` è una **lista** (es. `germany_plan_beneficiaries`
in `k_tickers.py:805`) e `dict()` tenta di spacchettare ogni stringa in
una coppia.

Tre modifiche, tutte in quella funzione:

1. **Tre forme supportate**: `tickers` come lista → pesi derivati `1/N`
   (equipesatura implicita, con guardia sulla lista vuota); `tickers`
   come dict → pesi letti; `portfolio` flat → retrocompatibile.
2. **Universo vs composizione**: per un R-portfolio la tabella non è la
   composizione — il motore ruota e detiene `n_top` titoli alla volta.
   Chiamarla "Composizione del Portafoglio" contraddiceva il testo della
   stessa relazione, che dichiara 1,6 operazioni al mese. Ora titolo
   "Universo di Selezione", colonna "Peso teorico" (con colonna allargata
   a 26 mm, l'intestazione non entrava nei 20 mm), e nota esplicativa
   sotto la tabella. Per i Lazy resta "Composizione del Portafoglio".
3. **Nome non risolto**: yfinance restituisce il proprio codice interno
   come `Company` per alcuni fondi (`0P0001ULK1.F` per `LU2963696674`).
   In un documento destinato all'investitore va trattato come mancante.
   Guardia su pattern `^0P[0-9A-Z]{8,}(\.[A-Z]+)?$`. Nota: il confronto
   ingenuo `nome != ticker` non funziona — `company_df` è indicizzato per
   Ticker con colonna ISIN separata (`u_functions.py:566`), e per i Lazy
   l'indice contiene ISIN, quindi le due stringhe differiscono sempre.

Verificato su entrambi i portafogli. Strada alternativa per il nome
reale del fondo: un override analogo a `ticker_isin_overrides.csv`, ma è
funzionalità nuova, non inclusa.

#### Numeri relazione investitore Germany Plan — verificato, nessun difetto

CumRet 229,89% / CAGR 11,01% / MaxDD 69,98% coincidevano con la riga
"Multifactor — Base" della relazione tecnica, su periodi dichiarati
diversi (2015 vs 2012). Verificato: `out` arriva dal chiamante, e nel JN
era impostato `variant_scelta = "BASE"` con `engine_scelto =
"Multifactor"`. Il codice ha fatto quanto richiesto; la coincidenza era
la stessa serie.

Scelta deliberata e coerente col profilo tematico: l'overlay Risk ON/OFF
taglia l'esposizione nei regimi avversi, comportamento desiderabile su un
sistematico ma non su un posizionamento tematico, dove l'esposizione
continua è il punto. Da tenere presente che la §7 della relazione tecnica
raccomandava Momentum, e che Multifactor Base ha il MaxDD peggiore delle
quattro varianti (69,98% contro 62,02%).

#### Aperto — prompt tesi e p-value complessivo

La relazione investitore di Germany Plan cita il p-value medio degli
ultimi 90 giorni (4,74%) come "significatività che inizia a essere
rilevante", senza menzionare il p-value sull'intero periodo (0,238, non
significativo). Le due misure sono diverse e la lettura è difendibile per
un tematico — l'orizzonte completo include nove anni precedenti alla
tesi, in cui il posizionamento non aveva ragione d'essere — ma quel
ragionamento non compare nel documento: un lettore esterno vede solo il
numero favorevole.

Patch proposta, non ancora applicata — ramo `thematic` di
`_thesis_instruction` (`r_functions.py:13191`), da fare su branch
`fix/thesis-prompt-pvalue`:

```
"Poiche' ptf_type='thematic', un alpha rolling in aumento nella finestra "
"recente e' la conferma attesa che la tesi d'investimento (vedi campo "
"'thesis') si sta confermando — descrivilo in questi termini. "
"OBBLIGATORIO: cita anche il p-value dell'alpha sull'intero periodo, "
"anche quando non e' significativo, e spiega in una frase perche' su un "
"tematico non e' la metrica dirimente — l'orizzonte completo include gli "
"anni precedenti alla tesi, in cui il posizionamento non aveva ancora "
"una ragione d'essere. Non presentare il solo p-value recente come se "
"fosse la significativita' complessiva."
```

Da validare su rerun: che il modello non trasformi la spiegazione in una
excusatio lunga. Se accade, accorciare a una frase.

#### Nota di metodo

Il piano piattaforma `ECOSISTEMA_INVESTIA.md` è stato quasi sostituito da
un documento costruito senza aprire il file reale (v2.6, 83 KB): la
versione era stata dedotta da una citazione di seconda mano. Il
risultato cancellava D7–D18 e l'intero sottosistema KID. Branch
eliminato, nessuna perdita.

> **Regola**: nessuna affermazione sullo stato di un file senza averlo
> aperto nella sessione corrente, e ogni affermazione citata con file e
> riga. Un'affermazione senza citazione è da trattare come non
> verificata.

---


### Sessione 04/08/2026 — Famiglie rotazionali, vincolo di equipesatura, criteri per nuovi engine

Discussione di progetto, nessuna modifica al codice. Messa per iscritto
perché discussioni analoghe si sono ripetute più volte nell'ultimo anno
senza lasciare traccia.

> **Convenzione di questa sezione**: ogni affermazione sullo stato del
> codice è seguita dalla fonte (file e riga, o l'artefatto da cui è
> letta). Le affermazioni prive di fonte sono marcate **[da
> verificare]** e vanno trattate come non verificate finché qualcuno non
> le controlla. Serve a non ripetere l'errore ricorrente: dedurre da una
> descrizione plausibile invece di leggere il codice.

#### Vincolo non negoziabile — R-portfolio equipesati

**Gli R-portfolio sono equipesati, senza eccezioni.** `1/n_top` sui
titoli selezionati, sempre. Non è un default né un parametro: è una
proprietà della famiglia.

Conseguenza operativa: un nuovo engine può modificare **solo il
ranking**, cioè quali titoli entrano. Mai l'allocazione. Qualunque
proposta che assegni pesi differenziati — volatility targeting, risk
parity, inverse-vol weighting — è **fuori dalla famiglia R-portfolio**
per costruzione, non per parametrizzazione, e non va valutata come
engine candidato.

I portafogli non equipesati avranno una famiglia propria, basata sulla
rotazione delle strategie (vedi sotto).

#### Le famiglie rotazionali, distinte

| | **R-portfolio** | **R-Strategies** (famiglia futura) |
|---|---|---|
| Cosa ruota | Titoli dentro un universo | Metodi di allocazione |
| Pesi | `1/n_top`, sempre | Differenziati, prodotti dal metodo |
| Engine | Momentum, Multifactor | Metodi di allocazione confrontati per blocco |
| Pipeline | WFO + OFC + MC + DSR | Oggi solo WFO propria |
| Stato | Produzione | Esplorativo, `R_Strategies.ipynb` |

#### Parametri reali dei due engine R (verificato)

Fonte: relazione tecnica Germany Plan 2026-08-03, §6.a.1 e §6.a.2, campo
`params` del segnale S1 — output reale del sistema, non documentazione.

```
Momentum     ['momentum_lookback_days', 'riskparity_lookback_days',
              'n_top', 'momentum_weight', 'ivol_weight']
Multifactor  ['momentum_lookback_days', 'riskparity_lookback_days',
              'n_top', 'momentum_weight', 'ivol_weight',
              'sortino_weight', 'idio_weight']
```

`rebalance_frequency` è anch'esso un `EngineParams` cercato dalla WFO:
presente nel param_grid (`r_functions.py:2088-2089`), letto dal motore
(righe 525 e 17799), con esempio `'QE'` alla riga 5237. Non compare
nella lista S1 sopra perché quella riporta i soli parametri su cui è
calcolata la diversità del plateau.

Spazi parametrici su Germany Plan (relazione tecnica §2): Momentum 2.304
combinazioni, Multifactor 62.208.

#### Cosa contiene oggi `R_Strategies.ipynb` (verificato 04/08 leggendo il notebook)

Il notebook contiene **due blocchi indipendenti**, spesso confusi come
un unico esperimento.

**1. `wfo_universe_selector_momentum` (§ Universe momentum WFO, cella 19)**
— seleziona *quali titoli* compongono l'universo, con WFO propria
(`train_years=3, test_years=1`), `param_grid` su `primary` /
`top_percentage` / `primary_lookback_months` / `secondary_lookback_months`
/ `secondary_top`, `shift_forward=True` contro il look-ahead,
`weighting="equal"`, `selection_rule="composite"`, `fee_bps=0.01`.

⚠️ **Non è un engine di rotazione: è un pre-filtro dell'universo con
ottimizzazione propria** — cioè lo stesso slot architetturale del
**Punto 1** (filtri di pre-selezione), dove è previsto Cluster v2.
Esiste, funziona, e non era mai stato collegato a quel ragionamento. Da
tenere presente quando si affronterà il redesign Cluster: potrebbe
esserci sovrapposizione, o materiale riusabile.

**2. `wfo_method_rotation_allocation` (§ Strategy WFO, cella 23)** —
ruota tra metodi di allocazione, restituisce `weights`, con
`plot_strategy_pie` a mostrare la distribuzione delle strategie
selezionate e `generate_portfolio_plan` a produrre il piano operativo.
È l'abbozzo della famiglia non equipesata.

Parametri della chiamata: `train_years=3, test_years=1`, `rf=0.0`,
**`tc=0.001`**, `rebalance_freqs=["ME","QE","YE","BH"]`,
`min_valid_ratio=0.9`.

**Non è una differenza `rebalance_freqs`**: come sopra,
`rebalance_frequency` è già cercato dalla WFO del motore R. Una versione
precedente di questa nota affermava il contrario, dedotto dal
fallimento di B2 su Germany Plan — inferenza priva di fondamento, B2
confronta il timing con date casuali e non dice nulla sul contenuto del
param_grid.

Manca invece tutto il resto della pipeline: nessun OFC, nessuna MC,
nessun DSR. `R_Strategies` ha una WFO propria, non quella del motore R.

#### Analisi articolo esterno — "Quick 5 ETF Rotational Strategy" (Paper to Profit, 04/08)

Strategia: universo di 5 ETF (VTI, AGG, VNQ, DBC, GLD), ranking per
momentum medio multi-orizzonte, top 3 equipesati, ribilanciamento
mensile.

```
M_mix = (R_1 + R_3 + R_6 + R_9 + R_12) / 5      R_k = P_t / P_{t-k} - 1
```

**Cosa è interessante**: invece di ottimizzare `momentum_lookback_days`,
media cinque orizzonti fissi. Inversione metodologica — l'orizzonte è
trattato come ignoto da diversificare, non come parametro da scegliere.
Effetto collaterale positivo: un grado di libertà in meno nel
param_grid, quindi meno penalizzazione DSR a parità di Sharpe.

**Perché non è un candidato naturale per la nostra pipeline**: `M_mix`
ha **zero parametri**. La pipeline è costruita per esplorare uno spazio
parametrico e giudicarne la robustezza; con un engine senza gradi di
libertà, S1 (diversità parametrica) e S4 (DSR su n trial) diventano
quasi vacui. Ci passa dentro ma non la usa.

Nota sulla formula: la media è semplice ma `R_1` è contenuto anche in
`R_3`, `R_6`, `R_9`, `R_12` — il mese più recente pesa cinque volte,
l'undicesimo una sola. Tilt implicito sul breve, non dichiarato
nell'articolo.

Altre osservazioni, non trasferibili ma utili:
- La riduzione di drawdown (25,77% contro 50,84% di VTI, dal tearsheet
  dell'articolo) viene dalla **rotazione cross-asset**: obbligazionario
  e oro sono *dentro* l'universo di rotazione. È l'equivalente continuo
  dell'overlay Risk ON/OFF, che invece è binario e agisce fuori dalla
  selezione. Su un universo monoclasse la rotazione non può ridurre il
  drawdown: non c'è nulla su cui rifugiarsi. Germany Plan ha MaxDD
  69,98% (relazione investitore 2026-08-03).
- **Nessuna validazione**: il basket viene da una ricerca su 126.144
  configurazioni, senza walk-forward né correzione per numero di trial.
  I numeri pubblicati sono in-sample. È esattamente ciò che S3 e S4
  esistono per catturare.

#### Criteri per un engine candidato (fissati 04/08)

Un nuovo engine di rotazione R-portfolio deve avere **tutte e tre**:

1. **Modifica il ranking, mai l'allocazione** — vincolo di equipesatura
2. **Ha parametri veri da ottimizzare** — altrimenti la pipeline non
   può giudicarlo
3. **Porta un'ipotesi economica distinta** — non un caso particolare del
   Multifactor

Sul terzo criterio: il Multifactor combina momentum, ivol, sortino e
idio tramite `momentum_weight`, `ivol_weight`, `sortino_weight`,
`idio_weight` (verificato sopra). **[Da verificare]**: se rotazione
risk-adjusted, low-volatility e idiosyncratic return siano davvero casi
particolari del Multifactor con pesi degenerati. È plausibile ma non
dimostrato — un `sortino_weight` come componente di un ranking composito
non è necessariamente equivalente a un ranking puro per Sortino rolling,
e la differenza dipende da come i pesi entrano nel punteggio. Da
controllare nel codice del ranking prima di scartare quelle tre
famiglie.

Restano in gioco, come criteri di **ordinamento** genuinamente diversi:

| Candidato | Ipotesi | Perché sembra ortogonale |
|---|---|---|
| **Trend strength** (R² / efficiency ratio di Kaufman) | Conta la *qualità* del trend, non l'ampiezza | Nessuno dei quattro pesi misura la linearità del percorso |
| **Cross-sectional mean reversion** | Segno opposto: compra i peggiori | Unica ipotesi economica davvero contraria. Test diagnostico: se la pipeline promuove sia momentum sia il suo contrario sullo stesso universo, il problema è nella pipeline |
| **Correlation regime rotation** | Ruota in base al regime di correlazione dell'universo | Proprietà dell'universo, non del singolo titolo |
| **Ensemble / voting** | Combina fattori deboli per voto anziché per pesi | Aggregazione diversa, non pesi diversi |

**Esclusi dal vincolo di equipesatura**: volatility targeting, risk
parity, inverse-vol weighting. Materiale per la famiglia R-Strategies.

#### Prima di implementare qualunque engine — la diagnosi manca ancora

L'item 5 resta una **design session**, non un task di implementazione, e
la prima domanda è invariata dal 21/07: *la MC boccia tutti i PTF e
tutti gli anni, o solo alcuni? Il problema è nel ranking o nel rebalance
timing?*

Su Germany Plan **B1 e B2 falliscono entrambi** — rotation reshuffle
p=0,877 e rebalance timing p=0,811 su Momentum; p=0,712 e p=0,746 su
Multifactor (relazione tecnica §5.a). Né la selezione né il timing
dimostrano skill. Aggiungere un engine senza sapere quale dei due sia il
collo di bottiglia rischia di essere lavoro sprecato: se il problema è
il timing, un ranking migliore non lo risolve.

Da tenere presente che `rebalance_frequency` è già un parametro cercato,
quindi il fallimento di B2 non si spiega con una frequenza imposta.

#### Voci di piano risultate stantie (verificate 03-04/08)

Tre voci descrivevano problemi non più esistenti. Il piano ha undici
settimane di sedimentazione: **prima di riprendere il lavoro sul motore
R, rileggere "Lavori in piedi" e "Tech debt" verificando ogni voce
contro il codice**, non fidandosi della descrizione.

- **`compare_wfo_pipelines` — patch mai applicata**: falso. La firma
  attuale (`r_functions.py:18951`) non ha alcun parametro
  `results_cluster`; prende un dict generico `{nome: risultati}` ed è
  già agnostica per costruzione.
- **`fix/report-path-and-sections-parity` — ~9 commit non mergiati**:
  falso. `git log origin/main..origin/fix/report-path-and-sections-parity`
  restituisce vuoto, tutto il contenuto è in `main`. Branch cancellato.
- **`feature/ranking-multifactor-v2` — lavoro sospeso**: idem, tutto in
  `main`.

Ripulito anche il remoto: 21 branch mergiati cancellati, resta solo
`main`.

#### Nota operativa — `git fetch --prune`

`session_start.sh` (passo 1/5) fa `git fetch origin` senza `--prune`:
i riferimenti a branch remoti cancellati altrove restano nella lista
locale e compaiono come "non mergiati". Da aggiungere.

---

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