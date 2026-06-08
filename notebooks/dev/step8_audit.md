# STEP 8 Audit — Pre-refactor

Branch di lavoro: `feature/step8-canonico`
Data audit: 2026-05-12

---

## 1. Celle metacodice identificate

### Cell 26 — Decisione finale (61 righe)
**Tipo output**: stampa a video (banner + tabella pandas)
**File scritto**: nessuno

**Variabili in scope al momento dell'esecuzione:**
- Da Cell 2: `portfolio_title`, `year`, `profile`, `tickers`, `benchmark_title`, `plots_dir`
- Da Cell 7: `ratio`, `metric`, `n_full_trials`
- Da Cell 8: `n_reduced_trials`
- Da Cell 11: `pipeline_start_date`, `results_std`
- Da Cell 15: `use_clustering`, `results_cluster`
- Da Cell 19: `ofc_passed_std`, `ofc_report_std`
- Da Cell 20: `ofc_passed_cluster`, `ofc_report_cluster`
- Da Cell 23: `ci_results`, `ci_summary_df`, `skill_results`, `skill_summary_df`

**Logica di assemblaggio:**
1. Deriva `reshuffle_pval`, `reshuffle_passed`, `s3_passed_std` da `skill_results` e `ofc_report_std`
2. Computa `skill_profile` (Strong / Timing-driven / Selection-driven / No-skill)
3. Costruisce dizionari `signals_std` e lookup da `ofc_report_cluster`
4. Assembla lista `_rows` con 8 righe (S1-S4, Reshuffle, CI Sharpe, OFC Verdict, Skill Profile)
5. Crea `_decision_df` DataFrame 3 colonne
6. Stampa banner + tabella + footer

**Variabili prodotte che rimangono in scope per Cell 27:**
`reshuffle_pval`, `reshuffle_passed`, `s3_passed_std`, `skill_profile`

---

### Cell 27 — Generazione PTF Card (217 righe)
**Tipo output**: file markdown scritto + print path
**File scritto**: `ptf_cards/{portfolio_title}_{year}.md`

**Variabili in scope** (eredita tutto da Cell 26, aggiunge):
- Usa `reshuffle_pval`, `reshuffle_passed`, `skill_profile` (da Cell 26)
- `results_std`, `results_cluster`, `wfo_file_save`, `ratio`, `metric`
- `n_full_trials`, `n_reduced_trials`, `pipeline_start_date`, `benchmark_title`
- `tickers`, `profile`, `use_clustering`

**Logica di assemblaggio:**
1. Costruisce path `ptf_cards/`
2. Definisce helpers locali `_vd()`, `_ci()`, `_m()` (formattatori inline)
3. Recupera struttura cluster da `results_cluster.get('cluster_result')`
4. Assembla template f-string ~200 righe (sezioni 1-9 PTF Card)
5. Chiama `_ptf_card_path.write_text(_card)`

---

## 2. Struttura OFC report (da lettura Cell 19 di r_functions.ipynb)

```python
ofc_report_std = {
    "promoted": bool,
    "profile": str,
    "resolved": {
        "metric": str,
        "plateau_threshold": float,
        "s2_coherence_threshold": float,
        "s3_pvalue_threshold": float,
        "s4_dsr_threshold": float,
        "min_signals_to_pass": int,
        "n_total_trials_used": int,
    },
    "signals": {
        "S1_plateau":   {"pass": bool, "value": float, "threshold": float, "note": str},
        "S2_coherence": {"pass": bool, "value": float, "threshold": float, "note": str},
        "S3_bootstrap": {"pass": bool, "p_value": float, "threshold": float, "metric_used": str, "note": str},
        "S4_dsr":       {"pass": bool, "dsr": float, "threshold": float, "metric_used": str, "note": str},
    },
    "n_signals_passed": int,
    "override_rule_applied": bool,
    "diagnostic_notes": [str],
}
```

## 3. Struttura ci_summary_df (DataFrame)

Index: `['A1 · IID Bootstrap', 'A2 · Block Bootstrap']`
Columns: `CAGR_p5`, `CAGR_p25`, `CAGR_p50`, `CAGR_p75`, `CAGR_p95`, `Sharpe_p50`, `MaxDD_p50`, `Actual_CAGR`, ...

## 4. Struttura skill_results (dict)

```python
skill_results = {
    'rotation_reshuffle': {
        'p_values': {'CAGR': float, 'MaxDD': float, 'Sharpe': float, ...},
        'distribution_shape': {...},
        ...
    },
    'rebalance_timing': {
        'p_values': {'CAGR': float, ...},
        ...
    },
}
```

## 5. Script di riferimento PDF

Il file `notebooks/dev/reference/generate_relazione_tecnica_v1_reference.py` NON è presente
nel filesystem. Il PDF v1 (`Alpha_Euro_2026_Relazione_Tecnica.pdf`, 10 pagine) è stato letto
visivamente e fornisce la struttura completa per la reimplementazione.

**Struttura PDF v1 (10 pagine):**
- Pag 1: Cover + §1 Identità + §2 WFO Config + §2b Cluster table
- Pag 2: Fig 1 — Dendrogramma
- Pag 3: Fig 2 — Scatter Vol/Momentum + Fig 3 — Heatmap correlazione
- Pag 4: §3 Metriche Comparative + Fig 4 — Equity comparison
- Pag 5: Fig 5 — Equity Cluster + Fig 6 — Equity Standard
- Pag 6: §4 OFC con tabelle 4a/4b colorate
- Pag 7: §5 MC + 5a Skill table + Fig 7 — Reshuffle + Fig 8 — Timing
- Pag 8: Fig 9 — MC Skill summary + 5b CI table
- Pag 9: Fig 10 — CI cross-method
- Pag 10: §6 Diagnosi strutturale (6a OFC, 6b MC Skill, 6c CI)

**Sezione 7 (Decisione Finale)** — non presente in v1, da aggiungere
come verdict box nella funzione canonica (richiesto dalla checklist di validazione).

**Palette colori:**
- Header/navbar: #1B2A4A (dark navy)
- Section title: #1B2A4A
- Table header bg: #2C3E6B, text: WHITE
- PASS: background #27AE60, text white
- FAIL: background #E74C3C, text white
- PROMOTED: background #27AE60
- NOT PROMOTED: background #E67E22 (orange)

---

## 6. Decisione di design per parametro mc_ci

Il parametro `mc_ci` accetta `ci_summary_df` (pandas DataFrame) anziché
`ci_results` (dict raw) perché:
- `ci_summary_df` è già in scope come struttura aggregata pronta all'uso
- evita di dover ricostruire la struttura interna di `ci_results` (non documentata)
- è coerente con la regola "no ricalcoli" (la summary è già calcolata)
Documentato anche in step8_integration_report.md §6 Anomalie.
