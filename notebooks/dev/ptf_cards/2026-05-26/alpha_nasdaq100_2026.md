# PTF Card — Alpha Nasdaq100 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Alpha Nasdaq100 |
| Engine | R-portfolio (rotational momentum) |
| Universe | 100 tickers |
| Benchmark | CSNDX.MI |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-26 |
| Profilo | satellite |
| Data generazione | 2026-05-26 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Alpha Nasdaq100_2026.wfo_summary.csv` |

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
| 1 | DEFENSIVE | 43 | GOOGL, AMZN, AEP, AMGN, AAPL, ADSK, BKR, BKNG ... (+35) |
| 2 | AVOID | 56 | ADBE, AMD, ABNB, ALNY, ADI, AMAT, APP, ARM ... (+48) |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-26*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 1093.2% | 36.3% | 1.32 | 24.0% |
| Cluster — Base | 787.3% | 32.3% | 1.16 | 26.8% |
| Standard — Risk ON/OFF | 6600.9% | 70.4% | 1.70 | 41.7% |
| Standard — Base | 4672.2% | 65.4% | 1.59 | 38.5% |
| Benchmark (CSNDX.MI) | 577.6% | 27.5% | 1.10 | 31.2% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.875 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Fail | 0.25 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Pass | p=0.034 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 1.0453294383860192 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.6666666666666666 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.622 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.9581954262468659 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Pass | p=0.002 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Pass | p=0.019 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Pass | p=0.098 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.769 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.243 | 0.448 | 0.649 | 1.422 | -0.356 |
| A2 — Block Bootstrap | 0.261 | 0.444 | 0.659 | 1.411 | -0.364 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.123 | 0.240 | 0.385 | 1.101 | -0.315 |
| A2 — Block Bootstrap | 0.129 | 0.239 | 0.357 | 1.097 | -0.288 |

---

## 6. Skill Profile
*Sintesi della capacità predittiva (Standard / Cluster)*

| Test | Cosa misura | Standard | Cluster |
|------|-------------|---------|---------|
| MC Rotation Reshuffle | La rotazione batte il caso? | Pass | Pass |
| MC Rebalance Timing | Il timing batte il caso? | Pass | Fail |
| OFC S3 | Il risultato Out-Of-Sample batte parametri casuali? | Pass | Fail |

**Skill Profile: Strong**
*Nota: No-skill non implica PTF non deployabile — il valore può derivare
dalla struttura dell'universe, dal clustering o dal Risk ON/OFF.*

---

## 7. Decisione Finale
| Dimensione | Standard | Cluster |
|-----------|---------|----------|
| OFC Verdict | PROMOTED | PROMOTED |
| Skill Profile | Strong | Strong |
| CAGR vs Benchmark | 70.4% vs 27.5% | 36.3% vs 27.5% |
| Sharpe vs Benchmark | 1.70 vs 1.10 | 1.32 vs 1.10 |
| MaxDD vs Benchmark | 41.7% vs 31.2% | 24.0% vs 31.2% |

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
