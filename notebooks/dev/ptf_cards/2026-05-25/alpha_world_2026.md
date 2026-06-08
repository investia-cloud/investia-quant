# PTF Card — Alpha World 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Alpha World |
| Engine | R-portfolio (rotational momentum) |
| Universe | 24 tickers |
| Benchmark | SWDA.MI |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-25 |
| Profilo | satellite |
| Data generazione | 2026-05-25 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Alpha World_2026.wfo_summary.csv` |

---

## 2. Configurazione WFO
| Parametro | Valore | Nota |
|-----------|--------|------|
| WFO ratio | 3:1 | Rapporto IS/OOS |
| WFO metric | Sharpe Ratio | Metrica ottimizzazione IS |
| Grid size (full) | 4608 combinazioni | Spazio parametrico totale |
| Grid size (reduced) | 288 combinazioni | Dopo stability analysis |
| Stability metric | CAGR, k=3 | Metrica e sottoperiodi |
| n_bootstrap OFC | 1000 | Test S3 random selection |
| n_bootstrap MC | 1000 | Block A (CI) + Block B (Skill Tests) |
| Risk ON/OFF | True | Filtro regime di mercato |
| Clustering | True | Se True: WFO per cluster omogenei |

---

## 2b. Struttura dei Cluster
*Composizione dei cluster sull'ultimo periodo WFO*

| Cluster | Label | N. Titoli | Tickers |
|---------|-------|-----------|---------|
| 1 | DEFENSIVE | 9 | EM710.MI, EM57.MI, EM13.MI, IBTM.MI, IBTS.MI, AHYE.MI, IHYG.MI, IHYU.MI ... (+1) |
| 2 | HIGH_MOMENTUM | 14 | EIMI.MI, C50.MI, CSMIB.MI, DAXX.MI, IWDE.MI, XAD5.MI, BRNT.MI, SP5A.MI ... (+6) |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-25*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 106.5% | 9.7% | 0.74 | 15.6% |
| Cluster — Base | 108.0% | 9.7% | 0.73 | 15.6% |
| Standard — Risk ON/OFF | -2.7% | -0.4% | 0.02 | 34.9% |
| Standard — Base | 10.1% | 1.2% | 0.17 | 34.9% |
| Benchmark (SWDA.MI) | 214.4% | 15.9% | 0.86 | 33.6% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 1.0 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Fail | 0.25 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.999 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.3169579782573835 |
| **OFC Verdict** | Soglia: 3/4 segnali | **NOT PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.9375 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.463 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.4885810284218654 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=1.000 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.971 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.130 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.475 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | -0.048 | -0.001 | 0.043 | 0.032 | -0.290 |
| A2 — Block Bootstrap | -0.056 | -0.002 | 0.041 | 0.018 | -0.309 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.006 | 0.064 | 0.125 | 0.608 | -0.227 |
| A2 — Block Bootstrap | 0.016 | 0.065 | 0.115 | 0.608 | -0.203 |

---

## 6. Skill Profile
*Sintesi della capacità predittiva (Standard / Cluster)*

| Test | Cosa misura | Standard | Cluster |
|------|-------------|---------|---------|
| MC Rotation Reshuffle | La rotazione batte il caso? | Fail | Fail |
| MC Rebalance Timing | Il timing batte il caso? | Fail | Fail |
| OFC S3 | Il risultato Out-Of-Sample batte parametri casuali? | Fail | Fail |

**Skill Profile: No-skill**
*Nota: No-skill non implica PTF non deployabile — il valore può derivare
dalla struttura dell'universe, dal clustering o dal Risk ON/OFF.*

---

## 7. Decisione Finale
| Dimensione | Standard | Cluster |
|-----------|---------|----------|
| OFC Verdict | NOT PROMOTED | PROMOTED |
| Skill Profile | No-skill | No-skill |
| CAGR vs Benchmark | -0.4% vs 15.9% | 9.7% vs 15.9% |
| Sharpe vs Benchmark | 0.02 vs 0.86 | 0.74 vs 0.86 |
| MaxDD vs Benchmark | 34.9% vs 33.6% | 15.6% vs 33.6% |

**Path deployato**: [ STANDARD | CLUSTER | NESSUNO ] ← compilare
**Motivazione**: ← compilare

---

## 8. Note e Avvertenze
*(compilare a mano)*

---

## 9. Plot salvati
| Plot | File | Disponibile |
|------|------|-------------|
| Equity Standard | equity_std.png | No |
| Equity Cluster | equity_cluster.png | No |
| Equity Comparison | equity_comparison.png | No |
| MC CI Block A | mc_ci.png | No |
| MC Reshuffle | mc_reshuffle.png | No |
| MC Timing | mc_timing.png | No |
| MC Skill Summary | mc_skill_summary.png | No |
| Cluster Heatmap | cluster_heatmap.png | No |
| Cluster Dendrogram | cluster_dendrogram.png | No |
| Cluster Scatter | cluster_scatter.png | No |
| MC CI Fan Chart IID | mc_ci_fanchart_iid.png | No |
| MC CI Fan Chart Block | mc_ci_fanchart_block.png | No |

---
