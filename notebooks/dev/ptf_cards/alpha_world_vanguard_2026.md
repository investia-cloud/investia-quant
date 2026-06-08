# PTF Card — Alpha World Vanguard 2026

---

## 1. Identità
| Campo | Valore |
|-------|-------|
| Nome | Alpha World Vanguard |
| Engine | R-portfolio (rotational momentum) |
| Universe | 38 tickers |
| Benchmark | V60A.DE |
| Periodo analisi | 2018-01-01 00:00:00 → 2026-05-24 |
| Profilo | satellite |
| Data generazione | 2026-05-24 |
| WFO file | `../../outputs/WFO_R_DEV_RESULTS/Alpha World Vanguard_2026.wfo_summary.csv` |

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
| 1 | HIGH_MOMENTUM | 15 | VWCE.MI, VHVE.MI, VUAA.MI, VNRA.MI, VWCG.MI, VERE.MI, VJPA.MI, VJPE.MI ... (+7) |
| 2 | BALANCED | 5 | VDST.MI, VDTA.MI, VDEA.MI, VUCE.MI, VDCA.MI |
| 3 | DEFENSIVE | 9 | VDTE.MI, VGUE.DE, VGEA.MI, VAGF.MI, VECA.MI, V3RE.MI, VDCE.MI, VCDE.MI ... (+1) |

Plot salvati: `cluster_heatmap.png`, `cluster_dendrogram.png`

---

## 3. Metriche Comparative WFO
*Confronto su periodo comune 2018-01-01 00:00:00 → 2026-05-24*

| Portfolio | Cum Return | CAGR | Sharpe | MaxDD |
|-----------|------------|------|--------|-------|
| Cluster — Risk ON/OFF | 62.2% | 13.9% | 1.19 | 10.0% |
| Cluster — Base | 53.2% | 12.2% | 1.05 | 14.1% |
| Standard — Risk ON/OFF | 20.0% | 5.2% | 0.63 | 8.9% |
| Standard — Base | 15.1% | 4.0% | 0.49 | 14.7% |
| Benchmark (V60A.DE) | 37.7% | 9.3% | 0.91 | 15.3% |

---

## 4. Overfitting Check (OFC)
*Valuta la robustezza del processo di ottimizzazione WFO. Soglia promozione: 3/4 segnali.*

### 4a. Path Standard
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.875 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Fail | 0.25 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.898 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.49676046788287587 |
| **OFC Verdict** | Soglia: 3/4 segnali | **NOT PROMOTED** | |

### 4b. Path Cluster
| Segnale | Cosa misura | Verdetto | Valore |
|---------|-------------|---------|--------|
| S1 — Plateau proxy | Diversità parametrica: il WFO converge su un unico punto? | Pass | 0.875 |
| S2 — Flag coherence | I filtri sono stabili tra sottoperiodi? | Pass | 1.0 |
| S3 — Random selection | Il risultato Out-Of-Sample batte 1000 portafogli con parametri casuali? | Fail | p=0.182 |
| S4 — DSR | Lo Sharpe è significativo dopo correzione per n. trial? | Pass | 0.7108209638332587 |
| **OFC Verdict** | Soglia: 3/4 segnali | **PROMOTED** | |

---

## 5. Monte Carlo Validation
*Valuta se il motore aggiunge valore rispetto al caso (analisi indipendente per path)*

### 5a. Skill Tests (Block B) — Path Standard
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Fail | p=0.985 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.912 |

### 5a. Skill Tests (Block B) — Path Cluster
| Test | Cosa misura | Verdetto | p-value |
|------|-------------|---------|----------|
| B1 — Rotation Reshuffle | La rotazione batte una selezione casuale dei titoli? | Pass | p=0.001 |
| B2 — Rebalance Timing | Il timing mensile batte date di ribilanciamento casuali? | Fail | p=0.606 |

### 5b. Confidence Intervals (Block A) — Path Standard
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | -0.015 | 0.037 | 0.093 | 0.540 | -0.119 |
| A2 — Block Bootstrap | -0.013 | 0.035 | 0.084 | 0.513 | -0.114 |

### 5b. Confidence Intervals (Block A) — Path Cluster
| Metodo | CAGR p5 | CAGR p50 | CAGR p95 | Sharpe p50 | MaxDD p50 |
|--------|---------|---------|---------|-----------|----------|
| A1 — IID Bootstrap | 0.021 | 0.092 | 0.172 | 0.966 | -0.128 |
| A2 — Block Bootstrap | 0.025 | 0.094 | 0.167 | 0.979 | -0.120 |

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
| OFC Verdict | NOT PROMOTED | PROMOTED |
| Skill Profile | No-skill | No-skill |
| CAGR vs Benchmark | 5.2% vs 9.3% | 13.9% vs 9.3% |
| Sharpe vs Benchmark | 0.63 vs 0.91 | 1.19 vs 0.91 |
| MaxDD vs Benchmark | 8.9% vs 15.3% | 10.0% vs 15.3% |

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
