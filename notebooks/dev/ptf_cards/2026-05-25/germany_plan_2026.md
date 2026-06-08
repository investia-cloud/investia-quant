# PTF Card — Germany Plan 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Germany Plan |
| Engine | R-portfolio (rotational momentum) |
| Universe | 24 tickers |
| Benchmark | ^GDAXI |
| Periodo analisi | 2012-01-01 00:00:00 → 2026-05-25 |
| Profilo | satellite |
| Data generazione | 2026-05-25 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Germany Plan_2026.wfo_summary.csv` |

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
| 1 | DEFENSIVE | 9 | EOAN.DE, RWE.DE, DTE.DE, 1U1.DE, ADV.DE, VNA.DE, LEG.DE, DHL.DE ... (+1) |
| 2 | BALANCED | 10 | MTX.DE, ENR.DE, NDX1.DE, SIE.DE, VOS.DE, KBX.DE, HOT.DE, HEI.DE ... (+2) |
| 3 | AVOID | 5 | RHM.DE, HAG.DE, R3NK.DE, WAF.DE, AIXA.DE |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2012-01-01 00:00:00 → 2026-05-25*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 1300.8% | 39.7% | 1.35 | 32.8% |
| Cluster — Base | 1480.8% | 42.0% | 1.32 | 39.1% |
| Standard — Risk ON/OFF | 101.4% | 9.4% | 0.53 | 55.3% |
| Standard — Base | 198.6% | 15.1% | 0.70 | 51.0% |
| Benchmark (^GDAXI) | 111.6% | 10.1% | 0.54 | 38.8% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.875 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 0.5 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.892 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.650373731271559 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.7916666666666666 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Pass | p=0.027 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 1.0229307894271245 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.922 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.881 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.253 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Pass | p=0.072 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | -0.031 | 0.065 | 0.157 | 0.444 | -0.381 |
| A2 — Block Bootstrap | -0.039 | 0.065 | 0.161 | 0.442 | -0.415 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.129 | 0.256 | 0.401 | 1.108 | -0.324 |
| A2 — Block Bootstrap | 0.130 | 0.259 | 0.410 | 1.118 | -0.304 |

---

## 6. Skill Profile
*Sintesi della capacità predittiva (Standard / Cluster)*

| Test | Cosa misura | Standard | Cluster |
|------|-------------|---------|---------|
| MC Rotation Reshuffle | La rotazione batte il caso? | Fail | Fail |
| MC Rebalance Timing | Il timing batte il caso? | Fail | Pass |
| OFC S3 | Il risultato Out-Of-Sample batte parametri casuali? | Fail | Pass |

**Skill Profile: No-skill**
*Nota: No-skill non implica PTF non deployabile — il valore può derivare
dalla struttura dell'universe, dal clustering o dal Risk ON/OFF.*

---

## 7. Decisione Finale
| Dimensione | Standard | Cluster |
|-----------|---------|----------|
| OFC Verdict | PROMOTED | PROMOTED |
| Skill Profile | No-skill | No-skill |
| CAGR vs Benchmark | 9.4% vs 10.1% | 39.7% vs 10.1% |
| Sharpe vs Benchmark | 0.53 vs 0.54 | 1.35 vs 0.54 |
| MaxDD vs Benchmark | 55.3% vs 38.8% | 32.8% vs 38.8% |

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
