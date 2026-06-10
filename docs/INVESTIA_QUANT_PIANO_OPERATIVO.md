# TSlab — Piano Operativo

**Ultimo aggiornamento**: 24 maggio 2026
**Root progetto**: `/home/luca/TSlab_project`

---

## Stato attuale

**Branch `main`** aggiornato e pulito. Ultimi commit storici:
34e2479  data: aggiornamento sel_tickers per relazioni tecniche
f710533  chore(tooling): scripts wrapper + direnv
0538d7e  feat(report): tabella holdings con ISIN + localizzazione italiana
7519138  Merge fix/r-mc-cluster-symmetry
7a1a193  fix(r-portfolio): MC §7/§8 sdoppiamento std/cluster + cluster fragmentation
d2b6c71  chore(snapshot): baseline pre-fix MC su 3 portfolio

**Sessione 24/05** (questo documento): branch `fix/mc-narrative-per-path` con tre fix
concatenati di reporting MC. In commit.

**Working tree**: clean (esclusi deliberatamente `notebooks/dev/R_Asset_v2.ipynb` per
output celle e `wget-log` per log temporanei).

**Branch parcheggiati**:
- `refactor/runtime-revamp-2026` — refactor lib `.ipynb` → `.py`, parcheggiato a sotto-fase 1.4

---

## Lavori chiusi nella sessione 24/05

Tre fix di reporting MC, applicati sullo stesso branch `fix/mc-narrative-per-path`:

### Fix #1 — Plot MC per-path (sub-directory std/ e cluster/)

**Problema**: in `R_Asset_v2.ipynb` §7 le due chiamate consecutive a 
`run_all_mc_methods_rotational` (Standard + Cluster) condividevano lo stesso 
`plots_dir`. La seconda sovrascriveva i PNG della prima. Conseguenza: la relazione 
PDF mostrava sempre figure del path Cluster anche nelle sezioni Standard, 
producendo incoerenze numeriche visibili (es. Actual 8.3% nella tabella Standard 
con CAGR vbt 4.0%).

**Fix**: sub-directory `plots_dir/std/` e `plots_dir/cluster/`, salvati ai 
rispettivi percorsi; `_img_sub(subdir, fname)` in `generate_relazione_tecnica` per 
risolvere il path corretto; rinumerazione figure §5: Fig. 5a-c (Std B1/B2/skill), 
Fig. 6a-c (Cluster B1/B2/skill), Fig. 7a-c (Std A1/A2/CI), Fig. 8a-c (Cluster 
A1/A2/CI). Totale: 12 figure MC al posto di 6.

PTF card markdown aggiornata di conseguenza: ogni file MC viene cercato in 
`std/...` e `cluster/...`.

### Fix #2 — Etichettatura figure

Assorbito in Fix #1 (le caption ora dichiarano esplicitamente "Path Standard" o 
"Path Cluster"). Nessun lavoro aggiuntivo richiesto.

### Fix #3 — Narrativa MC per-path (B-005 + B-006)

**B-005**: le sezioni 6.b/6.c citavano sempre numeri del path Standard anche 
quando il path raccomandato era Cluster. **B-006**: `compute_skill_profile` 
mappava (B1 PASS, S3 FAIL) → 'Timing-driven' anziché 'Selection-driven' 
(nomenclatura invertita), e ignorava B2.

Fix integrato:

- Nuova `compute_skill_profile` basata su (B1, B2):
  - B1 PASS + B2 PASS → 'Strong'
  - B1 PASS + B2 FAIL → 'Selection-driven'
  - B1 FAIL + B2 PASS → 'Timing-driven'
  - B1 FAIL + B2 FAIL → 'No-skill'
  
  Ritorna tupla `(profile_std, profile_cluster)`.

- Nuovo helper `_recommended_path(ofc_std, ofc_cluster)` → 'cluster' / 'std' / None
  (tie-break: Cluster preferito quando entrambi promossi).

- `_diagnose_mc` esteso: nuovi argomenti `mc_skill_cluster`, `mc_ci_cluster`, 
  `recommended_path`. Strategia narrativa **Opzione C**: focus sul path 
  raccomandato + nota di contrasto sull'alternativo. Posizione percentile reale 
  dell'Actual rispetto alla distribuzione bootstrap A1 (non più frase hardcoded 
  "ben sopra il p95", che era anche fattualmente sbagliata).

- `_build_verdict_text` esteso con `skill_profile_cluster` opzionale e caveat 
  per-profile (testi separati per Strong / Selection-driven / Timing-driven / 
  No-skill).

- Nuova `_build_verdict_text_compact` per box alternativo (sfondo grigio chiaro) 
  che mostra metriche essenziali del path non raccomandato.

- `generate_relazione_tecnica`: nuovo parametro `skill_profile_cluster`, riga 
  "Skill Profile" della tabella §7 sdoppiata Standard/Cluster, secondo verdict 
  box compatto aggiunto dopo il principale.

**Risolto in passing**: anche il campo `Periodo analisi` di pag. 1 (che mostrava 
"2015-01-01 → None") è stato corretto aggiornando la chiamata 
`generate_relazione_tecnica` in `R_Asset_v2.ipynb` con `period = 
(str(pipeline_start_date), _today_iso)`.

### Verifica post-fix su Alpha World Vanguard

Relazione 24/05 rigenerata:

| Sezione | Pre-fix | Post-fix |
|---|---|---|
| Fig. 5a/b (Standard B1/B2) | mostrava Actual Cluster 8.3% | Actual Std 2.7% ✓ |
| Fig. 6a/b (Cluster B1/B2) | identica a 5a/b | Actual Cl 8.3%, p=0.001 ✓ |
| Tabella §7 Skill Profile | "No-skill / No-skill" | "No-skill / Selection-driven" ✓ |
| §6.b paragrafo iniziale | "Entrambi FAIL... No-skill" | "Path Cluster, B1 PASS, Selection-driven" ✓ |
| §6.c posizione Actual | "ben sopra il p95" (hardcoded, falso) | "tra mediana (9.2%) e p95 (17.2%)" ✓ |
| §7 verdict box | uno solo | principale (Cluster) + compatto alternativo (Std) ✓ |

Tutti i test post-fix passati. Stabilità tra esecuzioni 19/05 → 22/05 → 24/05 
confermata: nessuno "swing" del p-value B1 (era diagnosi errata, vedi B-004).

---

## Lavori in piedi, in ordine di priorità

### 1. Potenziamento Block B — sorgenti di skill alternative al momentum (priorità alta)

**Problema reale emerso**: dei 4 PTF analizzati, solo Alpha World Cluster ha B1 PASS 
in modo robusto (p=0.001). Pattern dell'inventario 22/05:

| PTF | Universo | B1 cluster | OFC cluster |
|---|---|---|---|
| Italy Big Cap | 19 | 0.349 borderline | PROMOTED |
| Alpha Sect Megatrend | 9 | **0.015 PASS** | PROMOTED |
| Alpha World Vanguard | 38 | 0.001 PASS (confermato 24/05) | PROMOTED |
| Alpha Euro | 36 | 0.218 borderline | PROMOTED |

Pattern: universi piccoli/settoriali (Alpha Sect) e universi grandi pre-filtrati 
da clustering (Alpha World) mostrano skill rotazionale rilevabile dal Block B. 
Gli universi medi (Italy, Euro) restano borderline. **B2 (timing) è sempre FAIL.**

**Direzioni di lavoro discusse**:
- Risk-adjusted momentum (Sharpe ranking)
- Idiosyncratic momentum (residui post-neutralizzazione fattori sistematici)
- Low-volatility factor
- Quality factor (ROE, ROIC, debt/equity)
- Mean reversion (orizzonti corti/lunghi)
- Multi-factor compositi (momentum + low-vol + quality)
- Cross-sectional z-score ranking

**Prossimo passo**: prima della creazione di nuovi modelli, rilanciare R_Asset_v2 
sugli altri 3 PTF (Italy Big Cap, Alpha Sect, Alpha Euro) per riconfermare i 
numeri con la nuova narrativa corretta. Solo dopo questa baseline aggiornata 
decideremo dove intervenire.

### 2. Refactor `runtime-revamp-2026` (lavoro grosso, atteso)

**Cosa**: conversione delle librerie principali da `.ipynb` a `.py`. La cartella 
`notebooks/libs_py/` contiene già parti del refactor in corso.

**Perché serve**:
- I notebook si dirtano continuamente per output celle (es. R_Asset_v2)
- Test automatizzati impraticabili su `.ipynb`
- Import tra notebook fragile vs import Python standard
- Diff Git leggibile su `.py`, illeggibile su `.ipynb`

**Stato branch**: parcheggiato a sotto-fase 1.4 interrotta. Prima di riprenderlo:
```bash
git log refactor/runtime-revamp-2026 --oneline | head -20
git diff main..refactor/runtime-revamp-2026 --stat
```

**Note pre-refactor**: B-004 chiuso senza fix (diagnosi errata), B-005 e B-006 
risolti nella sessione 24/05. Il refactor parte da base MC stabile.

### 3. Agente automatico generazione relazioni tecniche

**Cosa**: agente che legge i PTF disponibili e lancia R_Asset_v2 per generare/
aggiornare schede e relazioni tecniche per tutti i portafogli.

**Perché serve**: attualmente 4 PTF analizzati. Con 15-20 PTF il pattern "skill 
rotazionale per universo piccolo/clustered" diventerebbe statisticamente robusto. 
È anche il presupposto per validare nuovi modelli di selezione (vedi punto 1).

**Dipendenza**: conviene farlo **dopo** il refactor `.py` per avere libreria 
testabile e CLI stabili.

**Stato**: solo accennato, nessun lavoro iniziato.

### 4. Cose minori in coda

- `wget-log` — rimuovere fisicamente o aggiungere a `.gitignore`
- `notebooks/libs/Untitled.ipynb` — se ricompare, non committarlo mai (scratch Jupyter)
- Caveat "S3 borderline" in `_build_verdict_text`: la soglia attuale 
  `abs(s3_pv - s3_thr) <= 0.05` non scatta per S3=0.182 vs soglia 0.10 (distanza 
  0.082). Da rivedere se si vuole catturare meglio i casi borderline reali, ma 
  non bloccante.

---

## Sequenza operativa consigliata

1. **Commit della sessione 24/05** — branch `fix/mc-narrative-per-path` su main
2. **Rilancio 3 PTF rimanenti** (Italy Big Cap, Alpha Sect, Alpha Euro) per 
   baseline aggiornata con narrativa corretta
3. **Discussione "sorgenti di skill alternative"** sulla base dei numeri 
   aggiornati dei 4 PTF
4. **Refactor `runtime-revamp-2026`** — lavoro grosso, ma con reporting MC ora 
   solido è meno rischioso
5. **Agente automatico relazioni tecniche** — sopra infrastruttura refactored

---

## Storia bug (issue tracker)

Sezione tipo issue tracker: tutti i bug noti del progetto con sintomo, diagnosi, 
fix. Stato: OPEN (in corso) / RESOLVED (chiuso) / CLOSED (chiuso senza fix 
perché diagnosi errata o non applicabile).

### B-001 — `run_rotational_engine` fallisce con `duplicate labels` su universi ampi · RESOLVED

**Sintomo**: `ValueError: cannot reindex on an axis with duplicate labels` 
nella costruzione di `rankings = pd.DataFrame(rank_records).T` (riga ~559 
di `r_functions`).

**Affliggeva**: `portfolio_alpha_world` (universo con ~20 ETF `.MI`).

**Causa effettiva**: ticker presenti sia in `tickers` (universo principale) 
sia in `risk_off_tickers` venivano duplicati a valle, propagandosi fino a 
generare index non univoco in `rank_records`. NON era un problema dei dati 
o di storia parziale come inizialmente ipotizzato.

**Fix**: in `R_Asset_v2.ipynb`, prima del download di `risk_off_data`, 
deduplica `risk_off_tickers` rispetto a `tickers` con variabile dedicata 
idempotente:
```python
risk_off_tickers_uniq = [t for t in risk_off_tickers if t not in tickers]
risk_off_data = download_data(risk_off_tickers_uniq, ...) if risk_off_tickers else None
```

**Fix commit**: branch `fix/r-mc-cluster-symmetry`, "fix(r-portfolio): 
risolve duplicate labels + Universe too small".

### B-002 — `reduce_grid_via_stability` rifiuta portfolio con universo piccolo · RESOLVED

**Sintomo**: `ValueError: Universe too small: N tickers available, need >= M 
(max(n_top_anchors)=K + 3 margin)`.

**Affliggeva**: `portfolio_alpha_sect` (9 ticker SPDR settoriali, servivano 11).

**Fix**: graceful fallback in `reduce_grid_via_stability` di `r_functions` 
per universi sotto soglia minima (fix applicato direttamente 
dall'architetto).

**Fix commit**: branch `fix/r-mc-cluster-symmetry`, stesso commit di B-001.

### B-003 — `compare_selection_columns` fallisce con `'float' object is not iterable` · RESOLVED

**Sintomo**: `TypeError: 'float' object is not iterable` in 
`compare_selection_columns` (riga ~2910 di `r_functions`) durante il 
confronto delle selezioni Risk ON/OFF vs Base.

**Affliggeva**: `portfolio_germany_plan`.

**Causa effettiva**: stessa di B-001 (ticker duplicati tra `tickers` e 
`risk_off_tickers` propagavano NaN a livello di righe nei dataframe di 
selezione).

**Fix**: stesso fix di B-001 (deduplica risk_off_tickers).

**Fix commit**: branch `fix/r-mc-cluster-symmetry`, stesso commit di B-001.

### B-004 — Presunto bug CAGR warmup in MC functions · CLOSED (non era un bug)

**Sintomo originale**: si era osservato uno swing del p-value B1 di Alpha World da
p=0.980 FAIL (19/05) a p=0.001 PASS (22/05) in 3 giorni di calendario, attribuito
a un presunto bug nel calcolo CAGR delle MC functions.

**Diagnosi reale (ispezione 24/05)**: i due numeri NON si riferiscono allo stesso
test. Tra il 19/05 e il 22/05 è stata aggiunta in §7 del notebook la chiamata MC
sul path Cluster (`pf_rot_cluster` / `sel_tickers_cluster`), che prima non veniva
validata. Confronto puntuale tra le tre relazioni:
- 19/05 §5.a: tabella unica, B1 = 0.980, Actual 2.7% → path **Standard**
- 22/05 §5.a.1 Standard: B1 = 0.983, Actual ~2.7% → **stesso test, stabile**
- 22/05 §5.a.2 Cluster: B1 = 0.001, Actual 8.2% → **nuovo test, non esisteva**
- 24/05 (rerun): B1 Std = 0.985, B1 Cl = 0.001 → conferma stabilità

Lo "swing" era confronto tra due PTF/path diversi, non instabilità della MC.

**Mismatch CAGR_MC vs CAGR_vbt** (~12.19% vs ~8.26% sull'esempio Alpha World): 
esiste ed è documentato by design in `_mc_print_portfolio_note()`:
"MC uses (V_end/V_start)^(252/n)-1 (academic convention). vbt uses an internal
annualization that differs systematically. Compare MC-Actual vs MC distributions
only; do NOT mix vbt and MC values."

Il mismatch è simmetrico tra Actual e Sim (stessa formula applicata a entrambi)
quindi NON inquina il p-value B1/B2. Resta solo una differenza di display tra
header (`_mc_cagr_quick`) e tabella WFO/vbt, accettata by design.

**Azione**: nessun fix necessario. Voce chiusa senza fix.

### B-005 — Sezioni narrative 6.b/6.c di Relazione Tecnica · RESOLVED

**Sintomo**: le sezioni 6.b/6.c citavano sempre i numeri del path Standard anche 
quando il path raccomandato era Cluster, producendo testi incoerenti con le 
tabelle e le figure circostanti.

**Esempio Alpha World 22/05**: §6.c diceva "CAGR p50 = 8.8%–9.1%" che erano i 
percentili del path Standard, mentre il path raccomandato (Cluster) aveva CAGR 
p50 = 12.1%–12.2%.

**Causa**: `_diagnose_mc` e `_build_verdict_text` ricevevano solo `mc_skill` e 
`mc_ci` (path standard), non `mc_skill_cluster` / `mc_ci_cluster`.

**Fix**: strategia **Opzione C** — narrativa focalizzata sul path raccomandato + 
nota di contrasto sull'alternativo. Implementata:
- nuovo helper `_recommended_path` (Cluster preferito se promosso)
- `_diagnose_mc` esteso con `mc_skill_cluster`, `mc_ci_cluster`, 
  `recommended_path`
- `_build_verdict_text` esteso con `skill_profile_cluster`, caveat per-profile 
  (Selection-driven, Timing-driven, Strong, No-skill)
- nuovo `_build_verdict_text_compact` per box alternativo
- riga "Skill Profile" tabella §7 sdoppiata Standard/Cluster
- CI §6.c: posizione percentile reale dell'Actual (calcolato), non più frase 
  hardcoded "ben sopra il p95"

**Dipendenza**: ha richiesto fix B-006 (nomenclatura corretta) come prerequisito.

**Fix commit**: branch `fix/mc-narrative-per-path` (sessione 24/05).

### B-006 — Nomenclatura invertita in compute_skill_profile · RESOLVED

**Sintomo**: `compute_skill_profile` mappava (B1 PASS, S3 FAIL) → 'Timing-driven' 
e (B1 FAIL, S3 PASS) → 'Selection-driven'. Nomenclatura invertita: B1 misura 
skill di selezione (rotation reshuffle), quindi (B1 PASS, S3 FAIL) dovrebbe 
essere 'Selection-driven'. Inoltre la logica ignorava completamente B2 
(rebalance timing).

**Affliggeva**: tutti i PTF. Per Alpha World 24/05 path Cluster (B1=0.001 PASS, 
B2=0.606 FAIL, S3=0.182 FAIL) il profilo riportato era 'No-skill' invece del 
corretto 'Selection-driven'.

**Fix**: nuova `compute_skill_profile` basata su mappa (B1, B2):

| B1 PASS | B2 PASS | Profile           |
|---------|---------|-------------------|
|   ✓     |   ✓     | Strong            |
|   ✓     |   ✗     | Selection-driven  |
|   ✗     |   ✓     | Timing-driven     |
|   ✗     |   ✗     | No-skill          |

S3 NON è più asse principale (rimane in `ofc_report` come metadata).

**Fix commit**: branch `fix/mc-narrative-per-path`, stesso branch di B-005 
(B-005 dipendeva da B-006).

---

## Discussioni evolutive aperte (chat parallela)

Spostate dall'utente in chat dedicata il 23/05 perché concettuali, non operative:

- **Lettura corretta di B1**: il WFO ottimizza configurazione multi-parametro 
  (momentum + volatilità + filtri + pesi), non solo momentum puro. La critica 
  "momentum arbitragiato via" applicata al framework era imprecisa.

- **Framing onesto del framework per presentazione cliente**: il framework 
  documenta quando un componente funziona e quando no via MC validation rigorosa. 
  È valore, non difetto. La frase "framework rotazionale che non rotaziona" era 
  imprecisa e va sostituita con "framework che riconosce la natura dell'universo 
  e adatta la fonte del valore". Confermata su Alpha World 24/05: il path Cluster 
  è Selection-driven (skill di selezione PASS), il Standard è No-skill — la 
  differenza è il clustering, non un problema del motore.

- **B2 FAIL sistematico**: tutti i 4 PTF hanno B2 (rebalance timing) FAIL. 
  Conferma: il timing mensile non aggiunge valore rispetto a timing random nei 
  bucket di volatilità. Da considerare ribilanciamento meno frequente 
  (trimestrale? semestrale?) per ridurre turnover senza perdere performance. 
  Da validare empiricamente.

---

## Convenzioni di lavoro (lezioni dal recente passato)

- **Branch separati per ogni scope**, niente scope creep
- **Disciplina del fix**: se in un branch dedicato a "X" emerge la necessità di 
  fare anche "Y", documentare Y nel TODO e aprire branch separato, non estendere 
  lo scope. Eccezione: dipendenze hard (es. B-005 richiedeva B-006 come 
  prerequisito → unificati nello stesso branch ma commit message esplicito).
- **Notebook .ipynb in dev** (es. `R_Asset_v2.ipynb`) si dirtano per output 
  celle: non committarli a meno di modifiche reali al codice. Per skipparli 
  durante commit usare `git add` esplicito sui file (non `git add .`)
- **Patches manuali OK per fix piccoli e ben definiti**: l'utente preferisce 
  risparmiare crediti Code quando le modifiche sono chirurgiche e copiabili a 
  mano. Solo per fix più complessi o multi-file delegare a Code.
- **Decisioni di design importanti vanno esplicitate**: anche scelte tecniche 
  dell'assistente vanno nominate per validazione utente. Esempio (sessione 
  24/05): la scelta di tie-break "Cluster preferito quando entrambi promossi" 
  in `_recommended_path` è stata esplicitata e validata.
- **Verifica visuale dei report PDF dopo ogni fix**: i bug più insidiosi (es. 
  PNG sovrascritti, profile invertito) si vedono solo confrontando la relazione 
  generata con i numeri attesi. La checklist puntuale sezione-per-sezione vale 
  più di test automatici.

---

## File chiave per riprendere il lavoro

- **`TSlab_PIANO_OPERATIVO.md`** (questo file) — piano operativo + storia bug
- **`notebooks/libs/r_functions.ipynb`** — libreria principale R-portfolio
- **`notebooks/libs/u_functions.ipynb`** — utility functions (modificata per ISIN)
- **`notebooks/dev/R_Asset_v2.ipynb`** — notebook principale R-portfolio, contiene 
  cella §7 (MC Std + Cluster con sub-directory plot)
- **`notebooks/libs_py/`** — libreria refactored parziale (branch refactor)
- **`notebooks/dev/snapshots/pre_mc_fix/`** — baseline pre-fix MC su 3 portfolio
- **`notebooks/dev/ptf_cards/`** — PTF Card markdown output
- **`notebooks/dev/reports/runtime_revamp/`** — report del refactor
- **`cache/ticker_isin_overrides.csv`** — mapping manuale Ticker→ISIN
- **`scripts/run_portfolios_v2.sh`** — workflow lancio portafogli (target `manager`)

---

## Note finali

Questa chat ha portato a termine il blocco di fix di reporting MC (3 fix 
concatenati, B-005 + B-006 risolti, B-004 chiuso senza fix come diagnosi errata) 
e ha consolidato la documentazione fondendo `TODO.md` in questo file.

Per riprendere in nuova chat è sufficiente allegare questo file e dire da quale 
punto vuoi ripartire — il punto naturale ora è il rilancio dei 3 PTF rimanenti, 
oppure direttamente la discussione sulle sorgenti di skill alternative.


### Sessione 08/06/2026 — pyproject.toml + venv (Task 1.2 ✓)

- Root progetto: ~/investia-quant (già rinominato)
- Python: 3.12.3
- Venv: ~/.venvs/investia-quant/
- pyproject.toml compilato con dipendenze runtime + [dev]
- pip install -e ".[dev]" completato senza errori
- requirements.lock generato
- iq registrato nel PATH (ModuleNotFoundError atteso: Task 1.3 pending)

Fix durante compilazione:
- build-backend: setuptools.backends.legacy:build → setuptools.build_meta
- pypfopt → PyPortfolioOpt (nome PyPI corretto)

Prossimo: Task 1.3 — scheletro CLI iq (investia_quant/cli.py)


---

## Sessione 08/06/2026 — Refactor libs_py + CLI iq

**Branch**: `refactor/libs-py`
**Root progetto**: `~/investia-quant` (rinominato da TSlab_project)

### Completato

**Task 1.2 ✓** — pyproject.toml + venv
- Python 3.12.3, venv `~/.venvs/investia-quant/`
- `pip install -e ".[dev]"` OK
- Fix: build-backend e pypfopt → PyPortfolioOpt

**Task 1.3 ✓** — CLI `iq` scheletro + implementazione completa
- `iq run` R e K — validato su ciclo reale, mail inviata e ricevuta
- `iq report` R — validato, figure complete, periodo corretto
- `iq report` K — funzionante, periodo YTD by design (maggio-giugno perché load_trading_systems_batch scarica solo ultimo periodo runtime)
- `iq analyze` — placeholder (Fase 2)

**libs_py/ completo** — 11 librerie convertite da .ipynb a .py, tutte AST OK:
u_functions, r_functions, k_functions, s_functions, t_functions, mc_functions,
k_tickers, k_strategies, k_portfolios, r_portfolios, l_portfolios

**Fix applicati durante sessione:**
- `r_portfolios.py`: aggiunto `from k_tickers import *` per risolvere dipendenze
- `k_functions.py`: aggiunto `from k_strategies import *` per _resolve_strategy via globals()
- `k_functions.py`: aggiunto `analisys_start_date/end_date` a `run_ts_portfolio_performance` e `load_trading_systems_batch`
- `u_functions.py`: aggiunto patch con funzioni grafiche mancanti (plot_cumulative_and_rolling_returns, plot_annual_performance, plot_year_returns_histogram, plot_ticker_frequencies, plot_total_return_per_ticker) e versione completa di `_generate_portfolio_performance_core_refactored`
- `cli.py`: default start_date differenziato (R: 2015-01-01, K: YTD)

### Pendente

1. **`iq report` K — periodo YTD** — `load_trading_systems_batch` hardcoda `start_date = datetime(end_date.year - 2, 1, 1)`. Il periodo risultante è sempre ~1 mese. Da confrontare con comportamento notebook per capire se era già così o il notebook gestiva diversamente.

2. **Task 3 — Release versionata** — struttura `releases/2026.x/` + symlink `current`

3. **Cleanup JN runtime** — dopo validazione completa CLI:
   - `notebooks/runtime/R_Run_Portfolio.ipynb` → dismissione
   - `notebooks/runtime/K_Run_Portfolio.ipynb` → dismissione
   - `notebooks/runtime/_bootstrap_runtime.ipynb` → dismissione
   - `notebooks/runtime/libs/` → verificare, dismissione se duplicato
   - Regola finale: in production gira solo `libs_py/` + `investia_quant/` + `scripts/`


## Sessione 09/06/2026 — Fix iq report K + Release versionata + CLI potenziata

**Branch**: `refactor/libs-py`

### Completato

**Fix `iq report K` periodo YTD** ✓
Tre bug risolti che causavano periodo `2026-05-08 → 2026-06-08` invece di `2026-01-02 → 2026-06-08`:
- `k_functions.py`: aggiunto `from u_functions import *` (load_ohlcv non trovata)
- `k_functions.py`: ramo `holding` in `load_trading_systems_batch` non passava `start_date` a `get_clean_financial_data`
- `cli.py`: `analisys_start_date=None` per K (YTD gestito da `crea_portafoglio_combinato` con comportamento legacy)

**Task 3 — Release versionata** ✓
- `scripts/make_release.sh`: crea `releases/YYYY.N/` con codice frozen, dati WFO, cache, config, secrets
- `scripts/deploy.sh`: rsync + install.sh + symlink `current` sulla VPS (parametri obbligatori: VERSION, VPS_HOST, INSTALL_DIR)
- `releases/2026.1/` deployata su `tslab.investia.cloud:/home/luca/investia-quant/releases/2026.1/`
- Venv per release (`releases/2026.1/.venv/`) — isolamento completo, rollback garantito
- `lib/` e `investia_quant/` registrati via `.pth` (no `pip install -e`, no git clone sulla VPS)
- direnv installato sulla VPS

**CLI `iq` potenziata** ✓
- `--rotational/--trading/--all`: esegue portafogli per tipo da `portfolios.conf`
- Alias `--portfolio` per `--ptf`, `--mail/--mailto` per `--recipient`
- Shortcut `--mail me/managers/customers`
- Validazione: obbligatorio `--ptf` oppure uno tra `--rotational/--trading/--all`
- Stampa destinatari risolti anche con `--no-send/--dry-run`

### Pendenti

1. **Cleanup JN runtime** — dopo validazione completa CLI:
   - `notebooks/runtime/R_Run_Portfolio.ipynb` → dismissione
   - `notebooks/runtime/K_Run_Portfolio.ipynb` → dismissione
   - `notebooks/runtime/_bootstrap_runtime.ipynb` → dismissione
   - `notebooks/runtime/libs/` → verifica + dismissione se duplicato

2. **Task 1.4** — smoke test `k_run_portfolio()`: un FAIL rimasto, target 10/10

3. **Crontab VPS** — configurare cron su `tslab.investia.cloud` usando `releases/current/scripts/crontab.txt` come riferimento
