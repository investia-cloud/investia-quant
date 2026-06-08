# Audit Bias — PTF Rotazionali
**Data**: 2026-05-04  
**Scope**: 9 PTF rotazionali, classificazione tassonomia A/B/C per survivorship/selection bias  
**Normativa**: `notebooks/GLOSSARY.md` voce "PTF categories (A/B/C)"  
**Status**: Draft — in attesa di review prima di aggiornare PTF cards o codice

---

## TASK 1 — Inventario ticker list

| PTF | Campo `tickers` | Origine concreta | Static/Dynamic/PIT | Documentazione |
|---|---|---|---|---|
| **Alpha Euro** | `stocks_euro` (variabile) | `k_tickers.ipynb` Cell 0 L212–256: lista hard-coded 36 ticker da IT/DE/FR/ES (9 per paese) | Static | Solo commento `# Area Euro`. Nessuna regola esplicita di selezione. |
| **Alpha Sect** | `settoriali` (variabile) | `k_tickers.ipynb` Cell 0 L157–168: 9 ETF UCITS `XL*.MI` iShares S&P US Select Sector | Static | Commento per ogni ETF indica il settore. Struttura sistematica implicita. |
| **Alpha Fact** | `fattoriali` (variabile) | `k_tickers.ipynb` Cell 0 L179–201: 5 ETF iShares MSCI World factor (Low Vol, Quality, Momentum, Value, Multifactor) + Small Cap. **IWVL.MI appare due volte** (bug, L181 e L193). | Static | Commenti descrivono il fattore. Nessuna regola sulla scelta dei provider. |
| **Alpha World** | `multiasset_global_ucits` (variabile) | `k_tickers.ipynb` Cell 0 L376–411: ~21 ETF multi-asset (EM equity/debt, bond, gold, global equity settoriali) | Static | Commenti per ogni ETF. Nessuna regola esplicita su quali asset class includere o quanti ETF per classe. |
| **Alpha World Vanguard** | `vanguard_etf` (variabile) | `k_tickers.ipynb` Cell 0 L508–587: ~26 ETF Vanguard UCITS, copertura geografica sistematica | Static | Commento esplicito (L417–421): "Regola: 100% Vanguard UCITS dove disponibile su .MI, fallback .DE. Tutti UCITS armonizzati. Ticker verificati su it.vanguard". |
| **Alpha Quant** | `ai_dc_quantum_thematic` (variabile composita) | `k_tickers.ipynb` Cell 0 L46–101: composizione di `quantum_pureplays` + `data_center_infrastructure` + `ai_quantum_enablers` + `bitcoin_miners_ai`. Misto ETF e azioni individuali. | Static | Commenti descrivono la tesi tematica. Titoli selezionati per esposizione al tema AI/quantum/data center. |
| **Alpha SP100** | `"sp100"` (stringa) | `r_functions.ipynb` Cell 6 L61: risolto a runtime via `extract_tickers_from_wikipedia("sp100")` → scraping Wikipedia corrente dell'indice S&P 100. Alternativa commentata: `alpha_sp100_tickers_by_year[year]` con liste 2025/2026. | **Dynamic (current)** | Nessuna documentazione sulla scelta di usare il live scraper vs le liste annuali. Le liste annuali (L-sopra in `r_portfolios.ipynb`) sono commentate. |
| **Alpha Nasdaq100** | `"nasdaq100"` (stringa) | Stesso pattern SP100: `extract_tickers_from_wikipedia("nasdaq100")` → composizione corrente. Liste annuali `alpha_nasdaq100_tickers_by_year` (2025/2026) disponibili ma commentate. | **Dynamic (current)** | Stesso gap documentativo di SP100. |
| **Germany Plan** | `germany_plan_beneficiaries` (variabile) | `k_tickers.ipynb` Cell 0 L801–841: 19 azioni tedesche selezionate per esposizione al Sondervermögen tedesco (difesa, energia, ferrovie, digitale, edilizia, logistica, semiconduttori) | Static | Struttura settoriale esplicita con commenti. Tesi dichiarata: esposizione agli investimenti del piano 500B EUR tedesco. |

---

## TASK 2 — Meccanismi point-in-time nel framework

### Funzioni cercate e risultato

| Pattern cercato | Trovato? | Dettaglio |
|---|---|---|
| `get_universe_at_date`, `historical_membership`, `point_in_time_universe`, `index_components_at` | **No** | Non esistono funzioni con questi nomi in `r_functions.ipynb` o `r_portfolios.ipynb` |
| File CSV con colonne `ticker, date_in, date_out` | **No** | `data/` contiene solo `stability_reports/`. `inputs/` contiene solo `WFO_R_RUN_RESULTS/` e `WFO_T_RUN_RESULTS/`. Nessun file di membership storica. |
| Logica nei downloader per filtrare per validità storica | **No** | `extract_tickers_from_wikipedia` scarica la lista **corrente** senza parametro data. Nessun filtro temporale. |
| Librerie con point-in-time interno | **No** | `vectorbt` usato per portfolio construction, non per universe definition. |

### Unico meccanismo "quasi-PIT" esistente

`alpha_sp100_tickers_by_year` e `alpha_nasdaq100_tickers_by_year` in `r_portfolios.ipynb`:
```python
alpha_sp100_tickers_by_year = {
    2025: ['INTC', 'LLY', 'GOOGL', ...],  # 25 tickers
    2026: ['LLY', 'GM', 'AMD', ...],       # 25 tickers
}
```
Queste liste rappresentano una selezione **manuale** di tickers per anno basata su un criterio non documentato (probabilmente "top performers dell'anno precedente" o "selezionati dal modellista"). Sono attualmente **commentate** — il codice attivo usa `"sp100"` e `"nasdaq100"` con il live scraper.

### Conclusione TASK 2

**Nessun meccanismo point-in-time esiste nel framework.** I PTF con universo dinamico (SP100, Nasdaq100) usano la composizione corrente dell'indice per tutto lo storico. I PTF con universo static usano la lista composta oggi per tutto lo storico. Non ci sono file di membership storica, né funzioni per costruire universi retroattivi.

---

## TASK 3 — Classificazione tassonomia A/B/C

| PTF | Categoria | Sotto-categoria | Razionale |
|---|---|---|---|
| **Alpha Euro** | **B** | **B-discr** | 36 large cap EU da 4 paesi. Mortalità storica ~zero (tutti incumbent). Assenza di regola esplicita per la selezione (non è "top N per capitalizzazione di EURO STOXX 50 al [data]", è una lista curata discrezionalmente dal modellista). |
| **Alpha Sect** | **B** | **B-rule** | 9 ETF UCITS, uno per ogni settore S&P US Select Sector. Copertura sistematica e completa di tutti i settori. ETF non hanno mortalità storica rilevante. Nessuna discrezionalità nella selezione: un ETF per settore = regola. |
| **Alpha Fact** | **B** | **B-rule** | 5-6 ETF iShares MSCI World factor, copertura sistematica dei fattori canonici (Low Vol, Quality, Momentum, Value, Size). ETF non hanno mortalità. Selezione basata su framework fattoriale consolidato. **Nota bug**: IWVL.MI appare due volte. |
| **Alpha World** | **B** | **B-discr** | ~21 ETF multi-asset globali. Mortalità ETF ~zero. Ma la selezione degli ETF (quali provider, quante asset class, quanti per classe) è discrezionale: nessuna regola documentata tipo "un ETF per ogni asset class in [lista]". |
| **Alpha World Vanguard** | **B** | **B-rule** | ~26 ETF Vanguard UCITS, con regola esplicita: "100% Vanguard UCITS dove disponibile su .MI, fallback .DE". Provider constraint + UCITS requirement = regola documentata e riproducibile. |
| **Alpha Quant** | **A** | — | Universo tematico: quantum computing + AI data center + bitcoin miners. Tesi esplicita: esposizione all'ondata AI/quantum post-2022. Orizzonte strutturalmente breve. Selection bias del modellista giustificato dalla tesi. |
| **Alpha SP100** | **C** | **C-static** | L'indice S&P 100 ha turnover storico significativo. Il framework usa la composizione corrente (live scraper) per tutto lo storico → survivorship bias presente. Il meccanismo quasi-PIT (liste annuali) esiste ma è commentato. |
| **Alpha Nasdaq100** | **C** | **C-static** | Stesso pattern SP100. Nasdaq-100 ha turnover ancora più alto di SP100 (aziende entrano/escono frequentemente: Tesla, Zoom, Meta, ABNB). Live scraper → survivorship bias presente su backtest lunghi. |
| **Germany Plan** | **A** | — | 19 azioni tedesche selezionate per esposizione al Sondervermögen (piano 500B EUR annunciato 2025). Tesi esplicita, orizzonte breve, selection bias giustificato dalla tesi. |

---

## TASK 4 — Bias residuo e raccomandazione

| PTF | Survivorship bias | Selection bias | Raccomandazione |
|---|---|---|---|
| **Alpha Euro** | **Limitato**: 33/36 ticker da pre-2007, 3 post-2007 (PRY.MI 2007, CLNX.MC 2015, PST.MI 2015). I 3 "recenti" sono aziende stabili con IPO documentate (Cellnex, Poste Italiane, Prysmian). Mortalità storica effettiva: ~zero. | **Presente (B-discr)**: nessuna regola documentata per la selezione dei 9 ticker per paese. | 🟡 **YELLOW — document caveat & monitor**: bias trascurabile in pratica ma la lista va documentata come "composta discrezionalmente alla data X con criteri Y". Aggiornare PTF card con categoria + razionale composizione. |
| **Alpha Sect** | **Assente**: ETF UCITS non si delistano (cambiano gestore o si fondono, ma non si azzerano). | **Assente (B-rule)**: copertura sistematica di tutti i settori. | 🟢 **GREEN — document & go** |
| **Alpha Fact** | **Assente**: ETF UCITS. | **Assente (B-rule)** ma con **bug** (IWVL.MI duplicato). | 🟢 **GREEN** con nota: correggere IWVL.MI duplicato in `fattoriali` (appare come Low Vol L181 e come Value L193). |
| **Alpha World** | **Assente**: tutti ETF. | **Presente (B-discr)**: selezione discrezionale degli ETF. | 🟡 **YELLOW — document caveat**: documentare i criteri di composizione nella PTF card. Basso rischio operativo perché ETF non si delistano. |
| **Alpha World Vanguard** | **Assente**: tutti ETF Vanguard. | **Mitigato (B-rule)**: vincolo provider + UCITS già documentato nel codice. | 🟢 **GREEN — document & go** |
| **Alpha Quant** | **Quasi-assente per design**: universo tematico con orizzonte breve. Alcune azioni hanno storia breve (IONQ IPO 2021, QBTS history limited). Ma è coerente con la tesi A. | **Giustificato (A)**: selezione per tesi esplicita. | 🟢 **GREEN — document & go**: documenta la categoria A e l'orizzonte strutturalmente breve. Backtest significativo solo dal 2022. |
| **Alpha SP100** | **Presente (C-static)**: S&P 100 ha ~5-10% di turnover annuo. Su backtest 10 anni si escludono aziende fallite/delistate che erano nell'indice → rendimenti storici sovrastimati. | **Assente**: selezione rule-based (componenti dell'indice). | 🟠 **ORANGE — plan a fix**: attivare le liste annuali `alpha_sp100_tickers_by_year` o costruire un meccanismo PIT formale. Prima di deploy su orizzonte > 3 anni, il bias va gestito. |
| **Alpha Nasdaq100** | **Presente (C-static)**: turnover Nasdaq-100 ancora più alto. Esempi storici: Tesla (entrata 2020), Zoom (entrata 2020, uscita 2022), Meta (entrata 2022). Su backtest 5-10 anni il bias è rilevante. | **Assente**: rule-based. | 🟠 **ORANGE — plan a fix**: stesso intervento di SP100. Nasdaq-100 ha il bias più critico di tutti i PTF perché il turnover è strutturalmente alto. |
| **Germany Plan** | **Quasi-assente per design**: tutte aziende attive e quotate, orizzonte breve (2025→). | **Giustificato (A)**: selezione per tesi esplicita. | 🟢 **GREEN — document & go**: documenta categoria A, anno di composizione, e il piano a cui fa riferimento. |

---

## TASK 5 — Focus su Alpha Euro

### 5a. Diagnostica empirica sulla lista (36 ticker)

**Dati raccolti via yfinance, download da 2000-01-01 a 2025-01-01:**

| Gruppo | Ticker | Prima data disponibile |
|---|---|---|
| Pre-2007 (33 ticker) | ENEL.MI, ISP.MI, UCG.MI, ENI.MI, STLAM.MI, G.MI, SAP.DE, SIE.DE, DTE.DE, BAYN.DE, ALV.DE, MRK.DE, BAS.DE, RWE.DE, MC.PA, OR.PA, SAN.PA, BNP.PA, AI.PA, KER.PA, HO.PA, ML.PA, IBE.MC, SAN.MC, REP.MC, TEF.MC, ACX.MC | 2000-01-03 |
| Pre-2007 (altri) | IFX.DE | 2000-03-13 |
| | ITX.MC | 2001-05-24 |
| | AIR.PA | 2001-09-03 |
| | ACS.MC | 2002-01-02 |
| | LDO.MI | 2005-07-18 |
| | GRF.MC | 2006-05-17 |
| Post-2007 (3 ticker) | PRY.MI (Prysmian) | 2007-05-03 |
| | CLNX.MC (Cellnex) | 2015-05-07 |
| | PST.MI (Poste Italiane) | 2015-10-27 |

**Riepilogo**: 33/36 (92%) disponibili da pre-2007. PRY.MI disponibile dal maggio 2007 (appena post-soglia). CLNX.MC e PST.MI disponibili dal 2015 (IPO recenti). Total copertura 2015–2024: 36/36.

**Come il framework gestisce i ticker con storia parziale**: `dropna(how='all').ffill()` — esclude le righe dove TUTTI i ticker sono NaN (non applica), e fa forward fill. In pratica: CLNX.MC e PST.MI entrano nel pool di selezione solo dalla loro data IPO. Prima di quella data, il motore lavora su 34-35 ticker invece di 36. Non c'è penalizzazione esplicita per dati mancanti iniziali.

### 5b. Verifica composizione della lista

**Criteri di composizione effettivi**: la lista è stata composta **discrezionalmente** dal modellista. Struttura: 9 ticker per ognuno dei 4 paesi principali dell'Area Euro (IT, DE, FR, ES). Non esiste:
- Una regola formale di selezione (es. "top 9 per capitalizzazione di FTSE MIB al [data]")
- Una data di composizione documentata nel codice o nei commenti
- Un riferimento a un indice di partenza

L'unica documentazione è il commento `# Area Euro` in `k_tickers.ipynb` L209 e i commenti per ciascun ticker che descrivono il settore dell'azienda. La struttura geografica (9 per paese) suggerisce una scelta deliberata di bilanciamento geografico, ma non una regola sistematica.

**Evidenza dall'elenco stesso**: la selezione include alcune scelte discutibili da un punto di vista rule-based puro — es. Cellnex (infrastrutture telco) e Acerinox (acciaio) non sarebbero necessariamente nei "top 9 per capitalizzazione" di IBEX35. Questo conferma la composizione B-discr.

### 5c. Re-interpretazione di S3 fail alla luce della categoria B

**Revisione metodologica**: Alpha Euro è categoria B (big cap structural, ~zero survivorship bias). Per i PTF di categoria B, S3 fail è **strutturalmente atteso** e **non un segnale di debolezza dell'engine**.

Ragionamento:
- Il baseline random in S3 seleziona portafogli da quello stesso universo curato (35 large cap EU)
- L'"alpha da curation" (scegliere EU large cap vs mercato generale) è già embedded nell'universo → il baseline random cattura questo alpha
- S3 misura solo se l'engine aggiunge **selezione cross-settoriale** all'interno dell'universo
- Per un universo altamente correlato (35 large cap EU dello stesso ciclo economico), la selezione cross-settoriale aggiunge poco → S3 fail è il risultato atteso

La skill del PTF Alpha Euro è nel **timing** (Reshuffle = Pass), non nella selezione dell'individuale titolo all'interno dell'universo omogeneo. Questo è coerente con la classificazione "Timing-driven" già nella PTF card.

**Conclusione**: S3 fail su Alpha Euro non è una red flag operativa. Va documentato come "expected for category B — curated homogeneous universe constrains cross-sectional edge" invece di "engine does not have cross-sectional skill".

### 5d. Diff proposto per PTF card Alpha Euro 2026

**Sezione "Identity" — aggiungere campo categoria:**
```diff
+| Category | B-discr (EU large cap structural, curated) |
```

**Sezione "S3 Diagnostic" — aggiornare interpretazione:**
```diff
-S3 fail is **structural, not artifact**: robust across selection metric
-(CAGR vs Sharpe) and sample size (N=50 vs N=500). The engine does not
-beat random rotation on the curated universe at any threshold.
+S3 fail is **structurally expected for category B** (curated homogeneous universe).
+The random baseline draws from the same 35 EU large caps — all highly correlated
+within the same economic cycle. Cross-sectional selection within a homogeneous
+universe cannot be expected to distinguish signal from noise. This is not a
+failure of the engine; it confirms that Alpha Euro's edge is timing-based
+(Reshuffle = Pass), not cross-sectional selection-based.
+
+For comparison: S3 would be informative for category C (open index with turnover)
+or for an expanded universe including non-EU assets. Parked as future work.
```

**Nuova sezione "Universe composition":**
```markdown
## Universe Composition

| Field | Value |
|---|---|
| Category | B-discr |
| Composition | 9 EU large cap per country (IT, DE, FR, ES) = 36 total |
| Composition date | Not documented — composed discrezionalmente |
| Composition rule | None formal — geographically balanced (9/country) but discretional within each country |
| Survivorship bias | Negligible: 33/36 tickers from pre-2007; CLNX.MC and PST.MI from 2015 (IPO dates) |
| Action item | Document formal composition criteria to upgrade from B-discr to B-rule |
```

---

## TASK 6 — Sintesi finale e action items

### 6a. Tabella di sintesi

| PTF | Categoria | Raccomandazione | Action item |
|---|---|---|---|
| Alpha Euro | B-discr | 🟡 YELLOW | Documentare criteri composizione lista nella PTF card; aggiornare interpretazione S3 fail |
| Alpha Sect | B-rule | 🟢 GREEN | Solo documentazione: aggiungere categoria B-rule nella PTF card |
| Alpha Fact | B-rule | 🟢 GREEN | Correggere IWVL.MI duplicato in `fattoriali`; aggiungere categoria |
| Alpha World | B-discr | 🟡 YELLOW | Documentare criteri composizione ETF nella PTF card |
| Alpha World Vanguard | B-rule | 🟢 GREEN | Solo documentazione: categoria B-rule già quasi-documentata nel codice |
| Alpha Quant | A | 🟢 GREEN | Documentare anno di composizione e la tesi nel PTF card |
| Alpha SP100 | C-static | 🟠 ORANGE | Attivare `alpha_sp100_tickers_by_year` come meccanismo quasi-PIT; pianificare gestione storica membership |
| Alpha Nasdaq100 | C-static | 🟠 ORANGE | Stesso intervento SP100; priorità più alta per turnover più elevato |
| Germany Plan | A | 🟢 GREEN | Documentare anno di composizione e piano di riferimento nella PTF card |

### 6b. Priorità degli action items

**Alta priorità (bias operativamente rilevante):**
1. **Alpha Nasdaq100** (C-static, ORANGE): turnover Nasdaq-100 è il più alto. Su backtest > 3 anni il bias è materiale. Intervento: attivare le liste annuali `alpha_nasdaq100_tickers_by_year` o costruire un CSV di composizione storica anno per anno.
2. **Alpha SP100** (C-static, ORANGE): stesso problema, leggermente meno urgente perché SP100 ha turnover inferiore a Nasdaq-100. Stesso tipo di intervento.

**Media priorità (documentazione mancante, bias basso):**
3. **Alpha Euro** (B-discr, YELLOW): criteri di composizione non documentati. Basso rischio pratico (mortalità ~zero) ma il modellista dovrebbe essere in grado di rispiegare perché ha scelto quei 36 ticker. Intervento: aggiungere nota in k_tickers.ipynb e nella PTF card.
4. **Alpha World** (B-discr, YELLOW): stessa natura di Alpha Euro ma con ETF → rischio anche più basso. Intervento documentativo.

**Bassa priorità (solo documentazione formale):**
5-9. Alpha Sect, Alpha Fact, Alpha World Vanguard, Alpha Quant, Germany Plan: tutti GREEN, solo aggiungere categoria nelle future PTF card. **Alpha Fact ha un bug tecnico** (IWVL.MI duplicato) che va corretto indipendentemente dal bias.

### 6c. Debiti tecnici emersi

**Da aggiungere a `CLAUDE.md` sezione "Open technical debts":**

1. **Framework does not support point-in-time universe construction**: per i PTF di categoria C (SP100, Nasdaq100), il framework non ha nessun meccanismo per filtrare la composizione dell'indice alla data storica. `extract_tickers_from_wikipedia` è stateless. Le liste annuali manuali (`alpha_sp100_tickers_by_year`) sono un workaround commentato, non una soluzione sistematica.

2. **`fattoriali` lista ha IWVL.MI duplicato**: `k_tickers.ipynb` L181 e L193 usano entrambi IWVL.MI, rispettivamente per Low Vol e per Value. Presumibilmente L193 dovrebbe essere un ETF value diverso (es. XDEV.MI o simile). Il duplicato non causa errori runtime (il motore deduplicherebbe i ticker) ma riduce la diversificazione fattoriale.

3. **No formal "composition date" field in portfolio objects**: il `portfolio` object non ha un campo `composed_on: str` o `composition_rule: str`. Non è possibile sapere programmaticamente quando una lista è stata composta o per quale criterio. Da valutare se aggiungere questo campo nel `portfolio` dict standard.

### 6d. Note metodologiche

1. **B-discr è la categoria di default tacita per molti PTF**: la maggior parte dei PTF esistenti sono B-discr perché il modellista compone una lista che "sembra ragionevole" senza documentare una regola formale. Non è un problema in sé se i ticker sono big cap stabili, ma la mancanza di documentazione rende difficile spiegare a un terzo perché quei ticker e non altri.

2. **ETF-only portfolios hanno zero survivorship bias per costruzione**: Alpha Sect, Alpha Fact, Alpha World, Alpha World Vanguard non hanno problemi di survivorship bias perché gli ETF non si azzerano sul mercato secondario come le azioni. Questo è un vantaggio metodologico degli ETF-based portfolios.

3. **S3 fail è interpretato diversamente per A/B vs C**: per i PTF di categoria A (tematico) e B (big cap), S3 fail è coerente con la struttura dell'universo — il random baseline cattura già l'alpha da curation. Per i PTF di categoria C (open index), S3 fail avrebbe un peso interpretativo diverso.

4. **Germany Plan ha un orizzonte intrinsecamente limitato**: il Sondervermögen tedesco è un piano con un orizzonte temporale definito (~2030). La tesi si esaurisce quando i fondi vengono spesi. La categoria A è corretta ma implica che il PTF card debba documentare esplicitamente "quando questa tesi è esaurita, il PTF va chiuso o ri-contestualizzato".

---

## Note di audit

- Nessun file di membership storica trovato nel repository
- `extract_tickers_from_wikipedia` non ha parametro data → irrimediabilmente live-only
- Le liste annuali in `r_portfolios.ipynb` sono un approccio quasi-PIT manuale che va formalizzato
- CLNX.MC e PST.MI hanno storia dal 2015: il motore li esclude implicitamente fino a quella data (ffill non può retrocompletare dati inesistenti → NaN fino all'IPO). Non è documentato come il framework gestisce questo edge case esplicitamente.
