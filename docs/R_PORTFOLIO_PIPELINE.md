# Pipeline di Valutazione — R-Portfolio (rotazionale)

**Progetto**: investia-quant
**Ultimo aggiornamento**: 12 giugno 2026

---

## Ordine di esecuzione

```
1. WFO Std          (path Standard — senza clustering)
2. WFO Cluster      (path Cluster — con clustering gerarchico Ward)
3. Compare          (confronto i due path)
4. OFC Std          (overfitting check sul path Standard)
5. OFC Cluster      (overfitting check sul path Cluster)
6. MC Std           (Monte Carlo sul path Standard)
7. MC Cluster       (Monte Carlo sul path Cluster)
8. Decisione finale (skill profile + path raccomandato)
9. Output           (relazione tecnica PDF + PTF card)
```

---

## Step 1-2 — WFO (Walk-Forward Optimization)

**Cosa fa**: ottimizza i parametri del motore rotazionale (lookback, top-N,
risk-off trigger) su finestre train/test. Produce due path paralleli:

- **Path Standard**: selezione top-N asset per momentum semplice
- **Path Cluster**: selezione top-N asset con clustering gerarchico Ward
  (raggruppa asset correlati, seleziona il migliore per cluster — diversificazione
  strutturale)

**Funzione**: `run_wfo_pipeline()` in `r_functions.py`

**Stabilità griglia**: prima del WFO, `reduce_grid_via_stability()` riduce la
griglia parametri eliminando combinazioni instabili — evita overfitting sulla
griglia stessa.

---

## Step 3 — Compare

**Cosa fa**: confronta le metriche dei due path (Standard vs Cluster) su CAGR,
Sharpe, Max Drawdown, Calmar. Determina quale path è raccomandato per la
decisione finale.

**Funzione**: `compare_wfo_pipelines()` in `r_functions.py`

---

## Step 4-5 — OFC (Overfitting Check)

L'OFC è composto da 4 segnali indipendenti, applicati separatamente su ciascun
path (Standard e Cluster).

### S1 — Plateau Proxy
**Cosa misura**: stabilità della performance attorno ai parametri ottimali.
Se la superficie parametri ha un plateau ampio attorno al massimo, la strategia
è robusta. Se il massimo è un picco isolato, è overfitting.

### S2 — Flag Coherence
**Cosa misura**: coerenza dei segnali di ribilanciamento nel tempo. Se il
portfolio ruota in modo erratico (flag incoerenti tra finestre WFO consecutive),
è segnale di instabilità.

### S3 — Random Selection
**Cosa misura**: confronto con selezione casuale degli asset. Se la strategia
ottimizzata non batte significativamente la selezione random, non c'è skill
reale.

### S4 — DSR (Deflated Sharpe Ratio)
**Cosa misura**: stessa logica del DSR K-strategy — corregge lo Sharpe per
multiple testing sull'intera griglia parametri WFO.

**Soglia promozione OFC**: ≥ 3 segnali su 4 devono passare.

**Funzione**: `overfitting_check_rotational()` in `r_functions.py`

---

## Step 6-7 — MC (Monte Carlo)

Il MC è strutturato in due blocchi con scopi distinti, applicati su ciascun
path.

### Block B — Skill Tests (testa la fonte della performance)

**B1 — Rotation Reshuffle**
Mischia casualmente le selezioni degli asset tra le finestre WFO mantenendo
i pesi. Testa se la skill di *selezione* degli asset è reale o casuale.
- p-value basso → la selezione specifica degli asset conta (skill di selezione)
- p-value alto → la performance viene dal timing, non dalla selezione

**B2 — Rebalance Timing**
Sposta casualmente le date di ribilanciamento. Testa se il *timing* del
ribilanciamento contribuisce alla performance.
- p-value basso → il timing conta (skill di timing)
- p-value alto → la performance viene dalla selezione, non dal timing

### Block A — Confidence Intervals (stima la distribuzione della performance)

**A1 — IID Bootstrap**
Ricampiona i returns giornalieri con reinserimento. Produce intervalli di
confidenza sulla performance futura assumendo indipendenza dei returns.

**A2 — Block Bootstrap**
Ricampiona blocchi contigui di returns. Produce intervalli di confidenza
più conservativi rispetto ad A1, preservando l'autocorrelazione.

**Funzione**: `run_all_mc_methods_rotational()` in `mc_functions.py`

---

## Step 8 — Skill Profile e Decisione Finale

### Skill Profile
Basato sui risultati B1 e B2, classifica il portfolio in 4 categorie:

| B1 (Selezione) | B2 (Timing) | Skill Profile |
|---|---|---|
| PASS | PASS | Strong — skill reale su entrambe le dimensioni |
| PASS | FAIL | Selection-driven — skill di selezione, timing marginale |
| FAIL | PASS | Timing-driven — skill di timing, selezione marginale |
| FAIL | FAIL | No-skill — performance non giustificata da skill |

**Funzione**: `compute_skill_profile()` in `r_functions.py`

### Verdetto per path

```
OFC ≥ 3/4 segnali pass
AND MC Block B: almeno B1 o B2 pass (skill profile ≠ No-skill)
AND MC Block A: CAGR atteso > 0 (CI lower bound positivo)
→ PROMOSSO
```

### Path raccomandato
Il path con OFC migliore e skill profile più forte viene raccomandato.
Se entrambi i path sono promossi, viene preferito Cluster per la
diversificazione strutturale. La narrativa del report si focalizza sul
path raccomandato con nota di contrasto sull'altro.

**Soglia promozione globale**: 3/4 segnali OFC — soglia deliberatamente
conservativa per evitare falsi positivi su universi piccoli.

---

## Relazione tra i filtri

| Filtro | Cosa rileva | Costo computazionale |
|---|---|---|
| WFO Std + Cluster | Performance reale OOS su due approcci | Alto |
| OFC S1-S4 | Overfitting sulla griglia parametri | Medio |
| MC Block B | Fonte della performance (selezione vs timing) | Medio |
| MC Block A | Distribuzione performance futura attesa | Medio |

A differenza della K-strategy, nell'R-portfolio non c'è un gate economico
preliminare — il WFO gira sempre su entrambi i path. Il costo è giustificato
dal fatto che l'R-portfolio è una decisione annuale, non un'analisi massiva
quotidiana.

---

## Confronto K-strategy vs R-portfolio

| Aspetto | K-strategy | R-portfolio |
|---|---|---|
| Oggetto | Coppia strategia:ticker | Universo di asset rotazionale |
| Gate economico | OFC precheck (evita WFO inutili) | Nessuno (WFO sempre) |
| Path paralleli | No | Sì (Standard + Cluster) |
| OFC | pass_gate da precheck | 4 segnali S1-S4, soglia 3/4 |
| MC Block B | 3 metodi bootstrap | B1 rotation + B2 timing |
| MC Block A | Incluso in Block B | A1 IID + A2 block bootstrap |
| DSR | Step esplicito post-WFO | S4 dentro OFC |
| Skill profile | No | Sì (Strong/Selection/Timing/No-skill) |
| Frequenza | Quotidiana (trading) | Mensile (rotazione) |
| Decisione | Promuovi coppia per PTF trading | Promuovi PTF per deploy annuale |

---

## Output

### `iq r-analyze` (headless)
- PTF card markdown (sempre)
- PNG grafici statici (matplotlib/seaborn)
- PDF relazione tecnica **solo con `--pdf`** (default off) con sezioni:
  §1 Identità, §2 Config WFO+Clustering, §3 Metriche comparative, §4 OFC,
  §5 Monte Carlo, §6 Diagnosi strutturale, §7 Decisione finale

### `r_portfolio_analyst.ipynb` (interattivo, solo Luca)
- Stessa pipeline con grafici Plotly interattivi
- Stampe intermedie per analisi esplorative
- Decisione sui 4 path WFO

---

## Implementazione

| Componente | File | Funzione/Comando |
|---|---|---|
| Pipeline headless | `notebooks/libs_py/r_functions.py` | `run_r_portfolio_analysis()` |
| CLI | `investia_quant/cli.py` | `iq r-analyze` |
| WFO | `notebooks/libs_py/r_functions.py` | `run_wfo_pipeline()` |
| Stabilità griglia | `notebooks/libs_py/r_functions.py` | `reduce_grid_via_stability()` |
| Compare path | `notebooks/libs_py/r_functions.py` | `compare_wfo_pipelines()` |
| OFC | `notebooks/libs_py/r_functions.py` | `overfitting_check_rotational()` |
| MC | `notebooks/libs_py/mc_functions.py` | `run_all_mc_methods_rotational()` |
| Skill profile | `notebooks/libs_py/r_functions.py` | `compute_skill_profile()` |
| Report PDF | `notebooks/libs_py/r_functions.py` | `generate_relazione_tecnica()` |
