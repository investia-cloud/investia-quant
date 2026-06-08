# Framework Glossary — r-portfolios

Vocabolario condiviso del framework. Questa è la fonte normativa di 
riferimento per i termini tecnici usati nel codice, nei commit, nelle 
schede PTF e nelle interazioni con Claude Code.

Per aggiornamenti: aggiungere voci in ordine alfabetico all'interno 
della categoria pertinente. Quando un termine viene aggiunto al codice 
o alla documentazione, deve essere referenziabile qui.

---

## Process & development terms

**Smoke test**
Test minimale che verifica solo che il codice non esploda quando 
eseguito sui dati reali. Non valida la correttezza dei risultati, 
valida solo che il flusso non si rompa. Origine: nei circuiti 
elettrici, il "test del fumo" era accendere il dispositivo e vedere 
se usciva fumo. Esempio: dopo aver implementato 
`reduce_grid_via_stability`, lanciarla su Alpha Euro e verificare 
che produca un dict + DataFrame senza crash. Non testa che le 
raccomandazioni siano "giuste" — testa che la funzione lavori.

**Plan-first**
Pattern di interazione con Claude Code: prima di scrivere codice, 
Claude Code propone un piano (signature, pseudocodice, decisioni di 
design, edge case) e attende approvazione. Solo dopo l'approvazione 
scrive codice. I bug di design si scoprono sulla carta, non sul 
codice già scritto.

**Workbench notebook**
Notebook di sviluppo dove Claude Code lavora durante l'implementazione 
di una nuova feature. Vive in `notebooks/dev/_workbench/`. Distinto 
dai notebook di produzione (che il modellista usa) e di libreria 
(che contengono le funzioni del framework). Vedi sezione "Notebook 
workflow protocol" in `CLAUDE.md`.

**Audit trail**
Sequenza tracciabile di artefatti (CSV, log, schede PTF, commit git) 
che documenta come una decisione è stata presa. Esempio: il 
`diagnostic_report` della stability analysis salvato come CSV è parte 
dell'audit trail della scelta dei flag binari per quel PTF.

**Milestone**
Unità di lavoro architetturale del framework. Non è "quanto lavoro 
fa Claude Code in una sessione", è "una capability del framework 
progettata e portata a chiusura come unità coerente". Milestone 1 = 
stability analysis. Milestone 2 = overfitting check. Milestone 3 
candidata = profile unification across stack.

---

## Statistical analysis terms

**WFO — Walk-Forward Optimization**
Simulazione che ricostruisce, anno per anno, le scelte di parametri 
che il modellista avrebbe fatto se avesse gestito il PTF storicamente. 
In ogni finestra di train (es. 3 anni) sceglie i migliori parametri 
secondo una metrica, e li applica nella finestra di test successiva 
(es. 1 anno). Il risultato cumulato delle finestre di test è la 
performance OOS.

**OOS — Out-of-sample**
Dati che il modello non ha visto durante la sua selezione. Performance 
OOS = performance reale che il modello avrebbe ottenuto senza conoscere 
il futuro. Opposto di in-sample.

**In-sample (IS) / Train**
Porzione dei dati storici usata per scegliere i parametri ottimali. Le 
metriche IS sono sempre artificialmente migliori di quelle OOS perché 
il modello è stato adattato a quei dati.

**Overfitting**
Adattamento eccessivo del modello al passato, che produce ottime 
metriche IS ma cattive metriche OOS. Il modello memorizza il rumore 
dei dati storici invece di catturare la struttura sottostante. Il 
framework è progettato per misurarlo e contenerlo.

**Stability analysis**
Metodologia introdotta nella Milestone 1: per ciascun parametro 
binario della griglia, valuta se il suo valore vincente è coerente in 
più sotto-periodi storici. Un flag positivo in 3 sotto-periodi su 3 è 
"stabile". Uno che oscilla è "instabile" e viene fissato a False per 
parsimonia.

**Coherent sign / Coherence**
Proprietà di un flag binario nella stability analysis: il delta della 
metrica (True vs False) ha lo stesso segno in tutti i sotto-periodi. 
Coherent positive = flag aiuta in tutti i periodi. Coherent negative = 
flag danneggia in tutti i periodi. Incoherent = segno misto.

**Plateau locale**
Concetto del segnale S1 dell'overfitting check: percentuale di 
combinazioni della griglia che raggiungono almeno l'80% della metrica 
del best. Un plateau alto significa che il best non è un picco isolato 
(overfitted) ma è immerso in una zona di configurazioni simili che 
funzionano. Indicativo di robustezza.

**Edge**
Vantaggio statistico misurabile della strategia rispetto a un benchmark 
di riferimento. Avere edge significa produrre rendimenti aggiustati per 
il rischio sistematicamente superiori a quelli attesi da puro caso. Il 
framework usa più test per misurarlo (Reshuffle, S3, DSR).

**Centroide**
Nella stability analysis, è il punto base nello spazio dei parametri 
rispetto al quale viene valutato il singolo flag. Si è scoperto (Step 
1.3) che la scelta del centroide influenza fortemente le conclusioni. 
Il framework usa "isolation centroid" (tutti gli altri flag a False) 
per default.

---

## Monte Carlo terms

**MC — Monte Carlo**
Famiglia di tecniche che genera molteplici scenari alternativi tramite 
ricampionamento dei dati storici, per valutare la robustezza di un 
risultato. Distingue tra "il PTF ha funzionato perché ha edge" e "il 
PTF ha funzionato per fortuna".

**Bootstrap**
Tecnica di ricampionamento: si estraggono campioni con reinserimento 
dai dati osservati per costruire una distribuzione empirica di una 
statistica. Esempio: bootstrap dei rendimenti giornalieri per ottenere 
intervalli di confidenza sul CAGR.

**Block Bootstrap**
Variante del bootstrap che estrae blocchi contigui di osservazioni 
invece di osservazioni singole. Preserva l'autocorrelazione dei 
rendimenti, importante perché i rendimenti finanziari non sono 
indipendenti.

**Rotation Reshuffle**
Test di skill specifico del framework r-portfolios: permuta le 
selezioni storiche del motore tra le date di ribilanciamento. Misura 
se il timing delle selezioni del motore è informativo. Parte del 
Block B della MC validation.

**Confidence Interval (CI)**
Intervallo di valori entro cui ci si aspetta che cada una metrica con 
una certa probabilità (es. 95%). Output del Block A della MC: "il CAGR 
atteso è 8.2% con CI 95% [5.1%, 11.4%]". Misura la robustezza 
distributiva.

**P-value**
Probabilità di osservare un risultato uguale o più estremo di quello 
osservato sotto un'ipotesi nulla. Nel contesto S3: "p=0.82 significa 
che l'82% dei portafogli random hanno performato meglio o uguale al 
PTF". P-value bassi sono "buoni" (PTF batte il baseline).

---

## Performance metrics

**CAGR — Compound Annual Growth Rate**
Tasso di crescita annuale composto. Misura il rendimento medio 
annualizzato. Higher is better.

**Sharpe Ratio**
Rendimento in eccesso sul risk-free, normalizzato sulla deviazione 
standard totale. Penalizza simmetricamente volatilità positiva e 
negativa. Higher is better.

**Sortino Ratio**
Variante di Sharpe che penalizza solo la deviazione downside 
(volatilità delle perdite). Concettualmente più adatta per profili 
avversi al rischio negativo. Higher is better.

**Calmar Ratio**
CAGR diviso per Max Drawdown. Mette in relazione esplicita rendimento 
e perdita massima. Adatto a profili che danno priorità alla protezione. 
Higher is better.

**Max Drawdown (MaxDD)**
Massima perdita percentuale registrata dal picco al minimo successivo 
nella curva di equity. Lower is better.

**Ulcer Index**
Misura combinata di durata e profondità dei drawdown. Penalizza non 
solo la perdita ma anche quanto a lungo si resta sotto acqua. Lower 
is better.

**DSR — Deflated Sharpe Ratio**
Sharpe Ratio penalizzato per il numero di trial esplorati durante la 
selezione del modello (Bailey & Lopez de Prado 2014). Corregge il bias 
da selezione multipla: più strategie si testano, più alto è lo Sharpe 
massimo atteso anche su rumore puro. DSR > 0 = c'è edge anche dopo la 
correzione.

---

## Portfolio design terms

**Profile (satellite / core)**
Categoria di destinazione del PTF nel portafoglio del cliente. 
Satellite = quota tattica, cerca rendimento aggiustato per il rischio 
totale. Core = quota di base, prioritizza capital preservation. Il 
framework parametrizza soglie e metriche secondo il profilo.

**PTF categories (A / B / C)**

Cross-framework taxonomy classifying PTFs by universe construction
logic. Each PTF (whether k or r) belongs to one category, which
determines the methodological lens for validation and bias
considerations.

- **Category A — Event-driven / thematic universe**: universe defined
  by a specific recent event or thesis (e.g. "Germany Plan" tied to a
  strategic plan, "AI revolution" post-ChatGPT, "energy transition"
  post-IRA). Relevant analysis horizon is structurally short.
  Survivorship bias near zero by construction. Selection bias of the
  modellista is justified and explicit by the thesis itself.

- **Category B — Big cap structural universe**: universe composed of
  tickers with effectively zero historical mortality (e.g. "Top 4 EU
  large caps", "Big Tech USA", "Italy Big Cap"). Survivorship bias
  near zero. Selection bias mitigated structurally if the universe is
  defined rule-based (e.g. "top N capitalization of market X at date
  Y") rather than discretionally.

- **Category C — Open universe with historical turnover**: universe
  defined by membership in an index with significant historical
  changes (e.g. "Russell 3000 components", "S&P 500 stocks over 10
  years"). Survivorship bias is the central methodological issue.
  Requires point-in-time membership lists for valid backtests.

The category is a property of the PTF and must be documented in the
PTF card. Validation thresholds and interpretation of overfitting
signals (especially S3 in r-portfolios) may need to be category-aware:
e.g. for category B PTFs, S3 fail (PTF not beating random rotation on
curated universe) is structurally expected and not necessarily a
negative signal.

Examples in current state:
- Alpha Euro: category B (top large caps from 4 EU countries)
- Germany Plan: category A (event-driven on German strategic plan)
- US Trading / Euro Trading: k1 workflow, see "Validation flow" entry;
  category does not directly apply since universe is dynamic year by year.
- Hypothetical "Italy Big Cap" k2: category B

**Universo / Universe**
Lista di ticker tra cui il motore rotazionale può selezionare. Definito 
a priori dal modellista nel `portfolio` object.

**Asymmetric universe (Risk ON / Risk OFF)**
Pattern di design dove la fase di selezione del motore include asset 
difensivi (es. XEON.MI, IBTS.MI) che la WFO standard non vede. Permette 
al motore di passare a defensive mode quando il segnale Risk ON/OFF 
cambia, senza modificare il motore stesso.

**n_top**
Numero di asset selezionati dal motore in ogni ribilanciamento. 
n_top=5 = il motore mantiene sempre 5 ticker in portafoglio.

**n_top_min**
Vincolo di portfolio design (non parametro WFO) che impone un minimo 
al numero di asset in portafoglio. Tipicamente 2 per evitare 
concentrazione totale su singolo ticker quando la WFO ottimizza CAGR.

**Rebalance frequency**
Cadenza con cui il motore rivede le selezioni: ME (month-end), QE 
(quarter-end), YE (year-end), ecc.

**Lookback (momentum / riskparity)**
Numero di giorni di storico usati dal motore per calcolare le metriche 
di selezione. `momentum_lookback_days=60` = il motore guarda gli ultimi 
60 giorni di prezzo per calcolare il momentum.

**Risk ON / Risk OFF regime**
Stato di mercato classificato dal framework tramite segnale (EMA50/
EMA200 cross + soglia volatility rolling sui ticker ad alta beta). 
Determina l'allocazione tra asset risk-on e defensive.

**Survivorship bias**
Distorsione causata dall'analizzare solo i titoli sopravvissuti fino a 
oggi (escludendo quelli falliti, delistati, fusi). Produce backtest 
artificialmente ottimistici. Il framework r-portfolios elimina questo 
bias usando liste storiche corrette per ogni epoca.

**Selection bias**
Distorsione causata dallo scegliere a posteriori la configurazione che 
ha funzionato meglio. La WFO walk-forward è il presidio classico contro 
questo bias.

---

## Workflow-specific terms

**`auto_reduce_grid`**
Flag in `walk_forward_rotational` che, se True, lancia automaticamente 
la stability analysis prima della WFO e usa la griglia ridotta. Default 
False.

**Skill profile (typology)**
Classificazione di un PTF basata sui due test di skill (Reshuffle MC 
+ S3 overfitting). Quattro profili: Strong (entrambi pass), 
Timing-driven (Reshuffle pass / S3 fail), Selection-driven (S3 pass / 
Reshuffle fail), No-skill (entrambi fail). Alpha Euro 2026 è 
classificato Timing-driven.

**Promotion / Promotion verdict**
Decisione finale del framework: il PTF passa al deploy o no. Output di 
`overfitting_check_rotational` come `promoted: bool`. La decisione è 
automatica multi-criteri ma documentata in audit trail.

**PTF card / Scheda PTF**
Documento markdown in `notebooks/dev/ptf_cards/<ptf_name>_<year>.md` 
che riassume identity, skill profile, diagnostic results, promotion 
status e methodology configuration di un PTF. Inaugurato con Alpha 
Euro 2026.

---

## Cross-framework architecture notes

**Validation flow: k-portfolios vs r-portfolios**

The two frameworks share the same methodological philosophy — validate 
before trusting — but position validation checks at different points in 
the workflow because they measure structurally different things.

### k-portfolios: overfitting check is PRE-WFO (two sub-workflows)

The k-portfolios framework supports two structurally different
workflows that share the same validation gate (`overfitting_optimization`
+ WFO + MC) but differ in how the universe is constructed.

**k1 — Yearly top performers (rule-based dynamic universe)**

Examples: existing "US Trading" and "Euro Trading" portfolios.

For each year N, select the top ~10 tickers from a reference index
according to a rule-based criterion evaluated at N-1 (e.g. capitalization,
liquidity, or momentum). Apply the full k validation pipeline
(overfitting + WFO + MC) on each selected ticker independently to find
strategies that pass all three gates.

Bias profile:
- Survivorship bias of tickers: structurally absent. Selection happens
  contemporaneously each year with information available at N-1, no
  retroactive exclusion of delisted tickers.
- Selection bias of the modellista: structurally absent. Universe
  selection is rule-based and reproducible.
- Selection bias of strategies: addressed by design via the
  overfitting/WFO/MC triple gate.

**k2 — Fixed universe (curated static universe)** [planned, not yet implemented]

Examples in roadmap: "Italy Big Cap", "USA Techno Big Cap".

A fixed set of tickers is selected upfront and held for multiple years.
The framework searches for strategies validated on each ticker
individually. The universe does not change year to year.

Bias profile:
- Survivorship bias of tickers: depends on how the static list was
  composed. Big cap stable lists (category B per "PTF categories"
  glossary entry) have negligible exposure. Curated lists composed by
  the modellista may have residual selection bias.
- Selection bias of the modellista: present, mitigated by structural
  constraints (e.g. "all big cap of country X at date Y") if defined
  rule-based.
- Cumulative trial inflation: the same N tickers are tested year after
  year, increasing the cumulative number of strategies tested. May
  require Deflated-style correction analogous to r-portfolios DSR.
  Design consideration for k2 implementation.

The validation flow (`overfitting_optimization` PRE-WFO) is identical
in both k1 and k2; the difference lies entirely in universe construction
and its methodological implications.

### r-portfolios: stability is PRE-WFO, overfitting check is POST-WFO

In r-portfolios, the validation is split across two distinct checks at 
two distinct points:

**Pre-WFO**: `reduce_grid_via_stability` (Milestone 1)
- Filters binary flags whose effect is incoherent across historical 
  sub-periods
- Reduces grid cardinality (typically 16x on `param_grid_rotational_v3`)
- Output: reduced grid + diagnostic report
- Conceptually analogous to k's pre-WFO filter, but acts on temporal 
  stability of flags rather than on full-grid distribution

**Post-WFO**: `overfitting_check_rotational` (Milestone 2)
- Four signals (S1 plateau, S2 sign coherence, S3 edge vs random, 
  S4 DSR), see GLOSSARY entries above
- Three of four signals (S1, S3, S4) require WFO output by construction:
  - S1 plateau: computed on WFO TrainScores per window
  - S3 edge vs random: compares WFO OOS performance against random 
    baselines on the same universe
  - S4 DSR: computes Deflated Sharpe on the WFO OOS equity
- Only S2 (sign coherence of binary flags) is independent of WFO — it 
  derives from the stability report
- Decision: binary — does the WFO-promoted PTF have OOS edge after 
  multiple-testing correction?

### Why the asymmetry

The two strategy types have different overfitting signatures:

- A k-portfolios strategy is **static**: fixed rules applied to a price 
  series. Overfitting manifests as "parameters fit to noise of that 
  ticker", visible by inspecting the grid distribution alone.
- An r-portfolios PTF is **dynamic**: composition and parameters change 
  over time via WFO. Overfitting manifests in two modes that require 
  separate checks:
  - **Structural**: selection rules unstable in time → caught by 
    stability (pre-WFO)
  - **Operational**: WFO produces good numbers but they are beatable by 
    random rotation on the same universe → caught by overfitting check 
    (post-WFO)

### Practical consequence

When working on k-portfolios: overfitting check gates entry to WFO. 
When working on r-portfolios: stability gates entry to WFO, overfitting 
check gates entry to deploy.

Both frameworks document the analysis trail at each step (CSV reports, 
JSON dumps, PTF cards) — the structure is consistent, the position 
of the checks differs by design.
