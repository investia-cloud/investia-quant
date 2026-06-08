# PTF Card — Alpha Nasdaq100 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Alpha Nasdaq100 |
| Engine | R-portfolio (rotational momentum) |
| Universe | 100 tickers |
| Benchmark | CSNDX.MI |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-25 |
| Profilo | satellite |
| Data generazione | 2026-05-25 |
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
| 1 | DEFENSIVE | 32 | GOOGL, AMZN, AEP, AMGN, AAPL, BKR, CTAS, CSCO ... (+24) |
| 2 | AVOID | 67 | ADBE, AMD, ABNB, ALNY, ADI, AMAT, APP, ARM ... (+59) |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-25*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 912.4% | 33.6% | 1.29 | 26.6% |
| Cluster — Base | 966.6% | 35.5% | 1.29 | 28.2% |
| Standard — Risk ON/OFF | 6512.8% | 70.2% | 1.69 | 41.7% |
| Standard — Base | 4608.9% | 65.1% | 1.58 | 38.5% |
| Benchmark (CSNDX.MI) | 578.5% | 27.5% | 1.10 | 31.2% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.875 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Fail | 0.25 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Pass | p=0.035 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 1.0417189242213631 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.8124999999999999 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.692 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.9780624023930152 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Pass | p=0.003 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Pass | p=0.019 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Pass | p=0.042 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.548 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.248 | 0.439 | 0.654 | 1.401 | -0.365 |
| A2 — Block Bootstrap | 0.260 | 0.439 | 0.656 | 1.398 | -0.368 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.108 | 0.217 | 0.350 | 1.054 | -0.307 |
| A2 — Block Bootstrap | 0.120 | 0.224 | 0.330 | 1.085 | -0.286 |

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
| CAGR vs Benchmark | 70.2% vs 27.5% | 33.6% vs 27.5% |
| Sharpe vs Benchmark | 1.69 vs 1.10 | 1.29 vs 1.10 |
| MaxDD vs Benchmark | 41.7% vs 31.2% | 26.6% vs 31.2% |

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
