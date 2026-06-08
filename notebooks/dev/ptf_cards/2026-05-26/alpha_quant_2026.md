# PTF Card — Alpha Quant 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Alpha Quant |
| Engine | R-portfolio (rotational momentum) |
| Universe | 21 tickers |
| Benchmark | CSSPX.MI |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-26 |
| Profilo | satellite |
| Data generazione | 2026-05-26 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Alpha Quant_2026.wfo_summary.csv` |

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
| 1 | DEFENSIVE | 4 | HON, INFR.L, CITY.MI, INRG.L |
| 2 | DEFENSIVE | 6 | DGTL.L, QDVE.DE, SMH.MI, VPN.L, TSM, ASML |
| 3 | AVOID | 7 | IONQ, RIOT, MARA, CLSK, CIFR, IREN, WULF |
| 4 | AVOID | 4 | RGTI, QBTS, QUBT, ARQQ |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-26*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 1048.8% | 35.7% | 1.01 | 50.6% |
| Cluster — Base | 2765.1% | 52.1% | 1.13 | 60.6% |
| Standard — Risk ON/OFF | 2025.9% | 47.3% | 1.06 | 57.1% |
| Standard — Base | 17732.8% | 92.9% | 1.01 | 60.8% |
| Benchmark (CSSPX.MI) | 296.9% | 19.1% | 0.97 | 33.6% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.9166666666666666 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Fail | 0.25 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.522 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.7390030002328266 |
| **OFC Verdict** | Soglia: 3/4 segnali | **NOT PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.7291666666666666 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.504 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.8822982279435817 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.749 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Pass | p=0.030 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Pass | p=0.013 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Pass | p=0.056 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.097 | 0.296 | 0.568 | 0.862 | -0.523 |
| A2 — Block Bootstrap | 0.062 | 0.309 | 0.625 | 0.887 | -0.573 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.076 | 0.235 | 0.428 | 0.843 | -0.454 |
| A2 — Block Bootstrap | 0.056 | 0.231 | 0.444 | 0.833 | -0.471 |

---

## 6. Skill Profile
*Sintesi della capacità predittiva (Standard / Cluster)*

| Test | Cosa misura | Standard | Cluster |
|------|-------------|---------|---------|
| MC Rotation Reshuffle | La rotazione batte il caso? | Fail | Pass |
| MC Rebalance Timing | Il timing batte il caso? | Pass | Pass |
| OFC S3 | Il risultato Out-Of-Sample batte parametri casuali? | Fail | Fail |

**Skill Profile: Timing-driven**
*Nota: No-skill non implica PTF non deployabile — il valore può derivare
dalla struttura dell'universe, dal clustering o dal Risk ON/OFF.*

---

## 7. Decisione Finale
| Dimensione | Standard | Cluster |
|-----------|---------|----------|
| OFC Verdict | NOT PROMOTED | PROMOTED |
| Skill Profile | Timing-driven | Timing-driven |
| CAGR vs Benchmark | 47.3% vs 19.1% | 35.7% vs 19.1% |
| Sharpe vs Benchmark | 1.06 vs 0.97 | 1.01 vs 0.97 |
| MaxDD vs Benchmark | 57.1% vs 33.6% | 50.6% vs 33.6% |

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
