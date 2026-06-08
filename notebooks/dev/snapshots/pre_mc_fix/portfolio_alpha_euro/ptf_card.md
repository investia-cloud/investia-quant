# PTF Card — Alpha Euro 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Alpha Euro |
| Engine | R-portfolio (rotational momentum) |
| Universe | 36 tickers |
| Benchmark | ^STOXX50E |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-19 |
| Profilo | satellite |
| Data generazione | 2026-05-19 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Alpha Euro_2026.wfo_summary.csv` |

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
| 1 | AVOID | 5 | STLAM.MI, SAP.DE, MC.PA, KER.PA, GRF.MC |
| 2 | BALANCED | 31 | ENEL.MI, ISP.MI, UCG.MI, ENI.MI, PRY.MI, G.MI, LDO.MI, PST.MI ... (+23) |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-19*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 255.1% | 17.3% | 0.81 | 26.4% |
| Cluster — Base | 316.6% | 19.7% | 0.84 | 34.9% |
| Standard — Risk ON/OFF | 51.2% | 5.4% | 0.36 | 43.0% |
| Standard — Base | 91.3% | 8.7% | 0.48 | 43.0% |
| Benchmark (^STOXX50E) | 58.5% | 6.1% | 0.37 | 38.3% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.9375 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Fail | 0.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.999 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.3169495581552509 |
| **OFC Verdict** | Soglia: 3/4 segnali | **NOT PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.7291666666666666 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.124 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.7811621062491507 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso*

### 5a. Skill Tests (Block B)
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.982 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.960 |

### 5b. Confidence Intervals (Block A)
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | -0.048 | 0.037 | 0.129 | 0.298 | -0.409 |
| A2 — Block Bootstrap | -0.051 | 0.034 | 0.128 | 0.279 | -0.440 |

---

## 6. Skill Profile
*Sintesi della capacità predittiva del motore di selezione rotazionale*

| Test | Cosa misura | Verdetto |
|------|-------------|----------|
| MC Rotation Reshuffle | La rotazione batte il caso? | Fail |
| MC Rebalance Timing | Il timing batte il caso? | Fail |
| OFC S3 | Il risultato Out-Of-Sample batte parametri casuali? | Fail |

**Skill Profile: No-skill**
*Nota: No-skill non implica PTF non deployabile — il valore può derivare
dalla struttura dell'universe, dal clustering o dal Risk ON/OFF.*

---

## 7. Decisione Finale
| Dimensione | Standard | Cluster |
|-----------|---------|----------|
| OFC Verdict | NOT PROMOTED | PROMOTED |
| Skill Profile | No-skill | No-skill |
| CAGR vs Benchmark | 5.4% vs 6.1% | 17.3% vs 6.1% |
| Sharpe vs Benchmark | 0.36 vs 0.37 | 0.81 vs 0.37 |
| MaxDD vs Benchmark | 43.0% vs 38.3% | 26.4% vs 38.3% |

**Path deployato**: [ STANDARD | CLUSTER | NESSUNO ] ← compilare
**Motivazione**: ← compilare

---

## 8. Note e Avvertenze
*(compilare a mano)*

---

## 9. Plot salvati
| Plot | File | Disponibile |
|------|------|-------------|
| Equity Standard | equity_std.png | Sì |
| Equity Cluster | equity_cluster.png | Sì |
| Equity Comparison | equity_comparison.png | Sì |
| MC CI Block A | mc_ci.png | Sì |
| MC Reshuffle | mc_reshuffle.png | Sì |
| MC Timing | mc_timing.png | Sì |
| MC Skill Summary | mc_skill_summary.png | Sì |
| Cluster Heatmap | cluster_heatmap.png | Sì |
| Cluster Dendrogram | cluster_dendrogram.png | Sì |
| Cluster Scatter | cluster_scatter.png | Sì |
| MC CI Fan Chart IID | mc_ci_fanchart_iid.png | Sì |
| MC CI Fan Chart Block | mc_ci_fanchart_block.png | Sì |

---
