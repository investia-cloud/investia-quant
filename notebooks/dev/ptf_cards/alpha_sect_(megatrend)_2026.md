# PTF Card — Alpha Sect (Megatrend) 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Alpha Sect (Megatrend) |
| Engine | R-portfolio (rotational momentum) |
| Universe | 9 tickers |
| Benchmark | Indice sintetico settoriali |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-22 |
| Profilo | satellite |
| Data generazione | 2026-05-22 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Alpha Sect (Megatrend)_2026.wfo_summary.csv` |

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
| 1 | HIGH_MOMENTUM | 2 | XLYS.MI, XLKS.MI |
| 2 | DEFENSIVE | 7 | XLCS.MI, XLPS.MI, XLFS.MI, XLVS.MI, XLIS.MI, XLBS.MI, XLUS.MI |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-22*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 142.0% | 11.9% | 0.57 | 26.8% |
| Cluster — Base | 254.4% | 17.5% | 0.67 | 35.7% |
| Standard — Risk ON/OFF | 158.5% | 13.0% | 0.74 | 23.2% |
| Standard — Base | 325.1% | 20.5% | 0.84 | 32.5% |
| Benchmark (Indice sintetico settoriali) | 255.1% | 17.8% | 0.75 | 67.2% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.9375 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 0.5 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=1.0 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.6567252093618017 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.8333333333333333 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.698 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.5255541764976626 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=1.000 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.999 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Pass | p=0.015 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.734 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.008 | 0.088 | 0.176 | 0.615 | -0.302 |
| A2 — Block Bootstrap | 0.028 | 0.087 | 0.151 | 0.606 | -0.226 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | -0.021 | 0.080 | 0.191 | 0.472 | -0.415 |
| A2 — Block Bootstrap | 0.006 | 0.079 | 0.154 | 0.475 | -0.293 |

---

## 6. Skill Profile
*Sintesi della capacità predittiva (Standard / Cluster)*

| Test | Cosa misura | Standard | Cluster |
|------|-------------|---------|---------|
| MC Rotation Reshuffle | La rotazione batte il caso? | Fail | Pass |
| MC Rebalance Timing | Il timing batte il caso? | Fail | Fail |
| OFC S3 | Il risultato Out-Of-Sample batte parametri casuali? | Fail | Fail |

**Skill Profile: No-skill**
*Nota: No-skill non implica PTF non deployabile — il valore può derivare
dalla struttura dell'universe, dal clustering o dal Risk ON/OFF.*

---

## 7. Decisione Finale
| Dimensione | Standard | Cluster |
|-----------|---------|----------|
| OFC Verdict | PROMOTED | PROMOTED |
| Skill Profile | No-skill | No-skill |
| CAGR vs Benchmark | 13.0% vs 17.8% | 11.9% vs 17.8% |
| Sharpe vs Benchmark | 0.74 vs 0.75 | 0.57 vs 0.75 |
| MaxDD vs Benchmark | 23.2% vs 67.2% | 26.8% vs 67.2% |

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
