# PTF Card — Italy Big Cap 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Italy Big Cap |
| Engine | R-portfolio (rotational momentum) |
| Universe | 19 tickers |
| Benchmark | CSMIB.MI  |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-22 |
| Profilo | satellite |
| Data generazione | 2026-05-22 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Italy Big Cap_2026.wfo_summary.csv` |

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
| 1 | BALANCED | 18 | ISP.MI, UCG.MI, MB.MI, BAMI.MI, FBK.MI, G.MI, ENI.MI, ENEL.MI ... (+10) |
| 2 | AVOID | 1 | STMMI.MI |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-22*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 268.6% | 18.0% | 0.81 | 42.3% |
| Cluster — Base | 368.6% | 21.7% | 0.84 | 48.1% |
| Standard — Risk ON/OFF | 162.0% | 13.2% | 0.74 | 46.1% |
| Standard — Base | 251.8% | 17.6% | 0.81 | 46.1% |
| Benchmark (CSMIB.MI ) | 214.3% | 15.9% | 0.71 | 41.1% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.9375 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Fail | 0.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.689 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.7513058159171493 |
| **OFC Verdict** | Soglia: 3/4 segnali | **NOT PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.7291666666666666 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.181 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.8038765878365235 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.856 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.776 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.349 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.624 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.007 | 0.088 | 0.175 | 0.609 | -0.317 |
| A2 — Block Bootstrap | -0.005 | 0.091 | 0.178 | 0.632 | -0.374 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.022 | 0.121 | 0.231 | 0.667 | -0.372 |
| A2 — Block Bootstrap | 0.018 | 0.122 | 0.220 | 0.673 | -0.352 |

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
| CAGR vs Benchmark | 13.2% vs 15.9% | 18.0% vs 15.9% |
| Sharpe vs Benchmark | 0.74 vs 0.71 | 0.81 vs 0.71 |
| MaxDD vs Benchmark | 46.1% vs 41.1% | 42.3% vs 41.1% |

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
| Cluster Scatter | cluster_scatter.png | No |
| MC CI Fan Chart IID | mc_ci_fanchart_iid.png | Sì |
| MC CI Fan Chart Block | mc_ci_fanchart_block.png | Sì |

---
