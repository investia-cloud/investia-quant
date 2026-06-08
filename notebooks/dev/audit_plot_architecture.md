# Audit Architettura Plot/Save — R_Asset_v2.ipynb

**Data audit**: 2026-05-08  
**Scope**: `notebooks/dev/R_Asset_v2.ipynb` + `notebooks/libs/r_functions.ipynb`  
**Metodo**: analisi statica, read-only, nessuna esecuzione

---

## Sezione 1 — Funzioni canoniche di plot

### 1.1 Funzioni in `r_functions.ipynb`

| # | Funzione | Cell lib | Tipo di plot | Chiamata da | save_plots/plots_dir? | Salva su disco? | plt.show()? |
|---|----------|----------|--------------|-------------|----------------------|----------------|-------------|
| F1 | `plot_dendrogram_colored(Z, labels, n_clusters, ax, ...)` | Cell 11 | Dendrogramma colorato per cluster + linea di taglio (helper, disegna su `ax` passato) | Solo da `analyze_and_cluster_universe` | No | No | No (helper puro, nessuna figura indipendente) |
| F2 | `analyze_and_cluster_universe(prices, n_clusters, lookback_days, plot, ...)` | Cell 11 | Fig 1×2: sinistra = dendrogramma via F1; destra = scatter Vol/Momentum con etichette ticker per cluster | `run_wfo_pipeline` (linea 1075) via `plot=True` | No | No | Sì — `plt.show()` linea 243 |
| F3 | `aggregate_cluster_portfolios(wfo_results, stocks_data, ..., plot)` | Cell 11 | Fig 2×1: sopra = equity cumulata per cluster + aggregato; sotto = regime Risk ON/OFF | Non chiamata direttamente dal notebook (solo da codice legacy o manuale) | No | No | Sì — `plt.show()` linea 526 |
| F4 | `_plot_results(pf_rot, pf_bh, pf_mom, pf_rp, init_cash, portfolio_name, width)` | Cell 4 | Plotly equity curve cumulata (interactive, range selector) | `build_portfolio_from_wfo_summary` (linea 628, solo se `plot=True`); **non chiamata direttamente da run_wfo_pipeline** | No | No | Sì — `fig.show()` linea 759 |
| F5 | `analyze_portfolio_metrics(..., plot_radar, ...)` | Cell 5 | Plotly radar chart normalizzato su range assoluti (opzionale, solo se `plot_radar=True` e ≤ 5 portafogli) | `run_wfo_pipeline` linee 1241 e 1279; `compare_wfo_pipelines` linea 158 | No | No | Sì — `fig.show()` linea 238 |
| F6 | `compare_wfo_pipelines(results_std, results_cluster, portfolio_title, ..., plot_radar)` | Cell 12 | Plotly equity cumulata multi-path + tabella metriche + radar via F5 | R_Asset_v2 Cell 18 | No | No | Sì — `fig.show()` linea 153 |
| F7 | `_mc_plot_ci_method(method_label, result, actual_equity, benchmark_equity, ...)` | Cell 15 | Plotly: fan chart equity curves simulate + actual + percentili (una figura per metodo A1/A2/A3) | `run_mc_confidence_intervals_rotational` se `show_method_plots=True` | No | No | No — ritorna lista di figure senza `.show()`. Il wrapper chiama `.show()` esternamente |
| F8 | `_mc_plot_ci_summary(ci_results, actual_equities)` | Cell 15 | Plotly: boxplot 1×2 (CAGR + MaxDD) cross-method, con linee actual | `run_mc_confidence_intervals_rotational` (linea 1194); R_Asset_v2 Cell 26 | No | No — ma il caller può fare `write_image` | No — ritorna figura |
| F9 | `_mc_plot_skill_test(test_label, result, actual_equity)` | Cell 15 | Plotly: istogramma p-value B1/B2 con linea actual e p-value annotato | `run_mc_skill_tests_rotational` (linea 1367); R_Asset_v2 Cell 26 | No | No — ma il caller può fare `write_image` | No — ritorna lista di figure |
| F10 | `_mc_plot_skill_summary(skill_results)` | Cell 15 | Plotly: barplot cross-test p-value con soglie 0.05/0.01 | `run_mc_skill_tests_rotational` (linea 1369) | No | No | No — ritorna figura |
| F11 | `plot_mc_distribution(mc_equity_curves, portfolio_actual, save_path, figsize)` | Cell 7 | Matplotlib: distribuzione MC equity (legacy) | Non usata nel notebook attuale | Sì — `save_path` | Sì se `save_path` | Sì |
| F12 | `plot_ranking_noise_analysis(results, save_path, figsize)` | Cell 7 | Matplotlib: Ranking Noise Analysis (legacy) | Non usata nel notebook attuale | Sì — `save_path` | Sì se `save_path` | Sì |
| F13 | `plot_wfo_mc_results(wfo_mc_df, save_path, figsize)` | Cell 7 | Matplotlib: WFO MC multi-finestra (legacy) | Non usata nel notebook attuale | Sì — `save_path` | Sì se `save_path` | Sì |

### 1.2 Funzioni chiamate da `run_wfo_pipeline` con `plot=True`

Quando `run_wfo_pipeline` viene chiamato con `plot=True` (default), attiva:
- F2 (`analyze_and_cluster_universe`) — solo se `use_clustering=True` — mostra inline durante STEP 1
- F5 (`analyze_portfolio_metrics`) × 2 — durante STEP 6 (portafoglio con e senza Risk ON/OFF)
- `build_rotational_portfolios_from_wfo_result(plot=True)` × 2 — che chiama internamente una Plotly equity display via `_plot_results`-equivalent

---

## Sezione 2 — Celle metacodice con logica plot/save

### Cell 19 — §5c "Salvataggio equity comparison"

**Posizione**: dopo Cell 18 (`compare_wfo_pipelines`) e prima di §6 (OFC)

**Cosa produce**:
- `equity_std.png` — matplotlib, 1 linea: Std Risk ON/OFF vs Benchmark
- `equity_cluster.png` — matplotlib, 1 linea: Cluster Risk ON/OFF vs Benchmark (solo se `results_cluster is not None`)
- `equity_comparison.png` — matplotlib, 4–5 linee: tutti i path (std, std_base, cluster, cluster_base, benchmark)

**Funzione canonica che produce output simile**:
- F6 (`compare_wfo_pipelines`) → stessa logica multi-path, ma renderer Plotly
- F4 (`_plot_results`) → singola equity curve Plotly

**Ricalcola o riusa?** Ricalcola da zero: legge `pf_rot_std`, `pf_rot_std_base`, `pf_rot_cluster`, `pf_rot_cluster_base` da namespace, ricava `pf.value() / pf.value().iloc[0]`

**File output**:
- `reports/plots/{portfolio_title}_{year}/equity_std.png`
- `reports/plots/{portfolio_title}_{year}/equity_cluster.png`
- `reports/plots/{portfolio_title}_{year}/equity_comparison.png`

**Nota**: questa cella definisce anche la variabile `plots_dir` (Path) usata da Cell 26 e Cell 31.

---

### Cell 26 — §7 "Salvataggio MC plot"

**Posizione**: dopo Cell 25 (run MC completo con `show_method_plots=True`)

**Cosa produce**:
- `mc_ci.png` — Plotly → PNG via `write_image`: boxplot CI cross-method (F8 `_mc_plot_ci_summary`)
- `mc_reshuffle.png` — Plotly → PNG: istogramma B1 Rotation Reshuffle (F9 `_mc_plot_skill_test`[0])
- `mc_timing.png` — Plotly → PNG: istogramma B2 Rebalance Timing (F9 `_mc_plot_skill_test`[0])

**Funzione canonica che produce output simile**:
- F8/F9 sono già chiamate con `.show()` dentro `run_all_mc_methods_rotational` (Cell 25, `show_method_plots=True`)

**Ricalcola o riusa?** Riusa `ci_results`, `skill_results` dal namespace — **NON ricalcola**, chiama le stesse funzioni di plot una seconda volta.

**File output**:
- `reports/plots/{portfolio_title}_{year}/mc_ci.png`
- `reports/plots/{portfolio_title}_{year}/mc_reshuffle.png`
- `reports/plots/{portfolio_title}_{year}/mc_timing.png`

**Nota**: `_mc_plot_skill_summary` (F10, barplot cross-test) è mostrata inline in Cell 25 ma **non salvata** in Cell 26.

---

### Cell 31 — §8 "Cluster heatmap e dendrogramma"

**Posizione**: dopo Cell 30 (scheda PTF), ultima cella eseguita normalmente

**Cosa produce**:
- `cluster_heatmap.png` — matplotlib: imshow della matrice di correlazione, ordinata per cluster, con rettangoli colorati per blocco
- `cluster_dendrogram.png` — matplotlib: dendrogramma Ward scipy standalone

**Funzione canonica che produce output simile**:
- F2 (`analyze_and_cluster_universe`) → genera dendrogramma + scatter in una figura 1×2 **con parametri di distanza e finestra diversi** (vedi §3)

**Ricalcola o riusa?** Ricalcola **parzialmente da zero**:
- legge `results_cluster['cluster_result']['cluster_groups']` e `['cluster_labels']` (struttura prodotta dalla canonica)
- ricalcola la matrice di correlazione da `stocks_data.tail(504)` **alla data di esecuzione del notebook**, non alla data WFO
- ricalcola la matrice di distanza e il linkage con parametri diversi dalla canonica

**File output**:
- `reports/plots/{portfolio_title}_{year}/cluster_heatmap.png`
- `reports/plots/{portfolio_title}_{year}/cluster_dendrogram.png`

**Dipendenza critica**: Cell 31 richiede che `plots_dir` sia definita da Cell 19. Se Cell 19 non è stata eseguita, Cell 31 fallisce con `NameError`.

---

## Sezione 3 — Divergenze identificate

### DIV-1 (CRITICA): Dendrogramma in Cell 31 vs `analyze_and_cluster_universe`

Questi due dendrogrammi rappresentano **clustering strutturalmente diverso** dello stesso universo.

| Parametro | `analyze_and_cluster_universe` (runtime WFO) | Cell 31 (metacodice, post-run) |
|-----------|---------------------------------------------|-------------------------------|
| Finestra temporale | `lookback_days` dalla call (`504` in Cell 16, `252` default) — **data fissa al momento del WFO** | `tail(504)` dalla **data di esecuzione del notebook** |
| Matrice di distanza | Combinata: `0.6 × √(0.5×(1−corr)) + 0.4 × euclidean(features)/n_feats` dove features = `[vol, mom_6m, autocorr, max_dd]` | Pura correlazione: `(1−corr)/2` |
| Linkage | Ward | Ward |
| Colorazione rami | Per cluster ID via `plot_dendrogram_colored` (palette: 5 colori custom) | `color_threshold = 0.7 × max(Z[:,2])` (scipy default, colori random) |
| Output | Subplot sinistra in figura 1×2 (con scatter a destra) | Figura standalone |
| Salvataggio | No | Sì → `cluster_dendrogram.png` |

**Conseguenza**: il `cluster_dendrogram.png` salvato su disco **non è il dendrogramma che ha guidato il WFO**. I cluster sul disco possono essere diversi per composizione e ordine rispetto a quelli usati per costruire i portafogli. Documentazione tecnica basata su questo file è fuorviante.

---

### DIV-2 (MODERATA): Scatter Vol/Momentum non salvato

`analyze_and_cluster_universe` genera a runtime uno scatter Vol/Momentum per cluster con etichette ticker. Questo plot è **visibile inline durante il WFO** (via `plt.show()`) ma **non viene mai salvato su disco** — né dalla canonica (no `savefig`) né da alcuna cella metacodice.

Il disco contiene `cluster_heatmap.png` e `cluster_dendrogram.png` (Cell 31) ma lo scatter — che è il vero "fingerprint visivo" del clustering effettuato — è irrecuperabile a posteriori.

---

### DIV-3 (MODERATA): Equity plots — renderer e normalizzazione

| Aspetto | `compare_wfo_pipelines` (Cell 18, canonica) | Cell 19 (metacodice) |
|---------|---------------------------------------------|---------------------|
| Renderer | Plotly (interattivo, range selector) | matplotlib (statico, headless Agg) |
| Normalizzazione equity | `pf.cumulative_returns() + 1` (vbt internal) | `pf.value() / pf.value().iloc[0]` |
| Plot 1 | std + std_base + cluster + cluster_base + bm | std + bm |
| Plot 2 | — | cluster + bm |
| Plot 3 | — | tutti i path (4–5 linee) |
| Save | No | Sì → 3 PNG |

La normalizzazione diversa (cumulative_returns vs value) è numericamente equivalente in assenza di cash flows intermedi, ma può divergere se vbt tiene traccia diversamente delle commissioni o dei dividendi.

---

### DIV-4 (MINORE): MC plots — doppia generazione

| Fase | Dove | Cosa viene mostrato | Cosa viene salvato |
|------|------|--------------------|--------------------|
| Cell 25 (run) | `run_all_mc_methods_rotational(..., show_method_plots=True)` | Tutti i plot: A1/A2/A3 fan chart, CI summary, B1/B2 histograms, skill summary barplot | Niente |
| Cell 26 (save) | Chiamate dirette a `_mc_plot_ci_summary`, `_mc_plot_skill_test` | — (i plot vengono mostrati inline una seconda volta via `write_image` + Jupyter re-render) | `mc_ci.png`, `mc_reshuffle.png`, `mc_timing.png` |

Ogni plot MC viene **generato due volte** (stessi dati, stesse funzioni). Il `_mc_plot_skill_summary` (barplot cross-test p-value) è mostrato in Cell 25 ma **non salvato** in Cell 26.

---

### DIV-5 (INFORMATIVA): `plots_dir` come variabile globale implicita

`plots_dir` è definita in Cell 19 ed usata da Cell 26 e Cell 31. Non è un parametro passato esplicitamente — è un effetto collaterale di namespace. Se Cell 19 non viene eseguita (es. run parziale del notebook), Cell 26 e Cell 31 falliscono con `NameError: plots_dir`.

---

### DIV-6 (INFORMATIVA): Cell 31 non viene eseguita se run è parziale

Come evidenziato dall'esecuzione Alpha Euro 2026 (cells 29, 30, 31 con `execution_count=None`), la cella dei cluster plots è **l'ultima nel flusso principale** e viene saltata se il run si interrompe dopo Cell 28. La dipendenza da Cell 19 (per `plots_dir`) e Cell 16 (per `results_cluster`) non è esplicitata.

---

## Sezione 4 — Mappa di consolidamento proposta

| Plot canonico finale | Funzione da modificare | Celle metacodice da eliminare / assorbire | Parametri da aggiungere | Note |
|----------------------|------------------------|------------------------------------------|------------------------|------|
| Dendrogramma + Scatter (runtime WFO) | `analyze_and_cluster_universe` (F2) | Cell 31 — Plot B (dendrogramma) va eliminato | `save_plots: bool = False`, `plots_dir: str \| None = None` | Il dendrogramma su disco deve essere quello prodotto dalla canonica, non il ricalcolo di Cell 31. Naming: `cluster_scatter.png` per la figura 1×2 |
| Heatmap correlazione | Nuova funzione `plot_cluster_heatmap(cluster_result, stocks_data, lookback_days, plots_dir)` — oppure Cell 31 Plot A promosso a funzione | Cell 31 — Plot A (heatmap) va incapsulato in funzione con `plots_dir` | `save_plots`, `plots_dir`, `lookback_days` | La heatmap non ha un equivalente canonico; è un contributo originale di Cell 31 ma deve usare gli **stessi dati della canonica** (finestra = `lookback_days`, non `tail(504)` fisso) |
| Equity std / cluster / comparison | `compare_wfo_pipelines` (F6) | Cell 19 — da eliminare dopo che F6 acquisisce `save_plots`/`plots_dir` | `save_plots: bool = False`, `plots_dir: str \| None = None`, `renderer: str = 'plotly'` | Alternativa: mantenere Cell 19 ma renderla una chiamata a una funzione `save_equity_plots(results_std, results_cluster, plots_dir)` incapsulata in r_functions |
| MC CI summary | `_mc_plot_ci_summary` (F8) | Cell 26 — mc_ci.png viene assorbito da F8 | `save_path: str \| None = None` | Pattern già presente in F11/F12/F13 |
| MC Skill test B1/B2 | `_mc_plot_skill_test` (F9) | Cell 26 — mc_reshuffle.png, mc_timing.png assorbiti da F9 | `save_path: str \| None = None` | |
| MC Skill summary (barplot) | `_mc_plot_skill_summary` (F10) | Nessuna cella da eliminare, attualmente non salvato | `save_path: str \| None = None` | Aggiungere salvataggio consistente con gli altri MC |

---

## Sezione 5 — Stima impatto

### Funzioni da modificare

| Funzione | Modifica | Difficoltà |
|----------|----------|------------|
| `analyze_and_cluster_universe` | + `save_plots`, `plots_dir`; `plt.savefig` prima di `plt.show` | Minimal-diff, 4 righe |
| `aggregate_cluster_portfolios` | + `save_plots`, `plots_dir`; `plt.savefig` prima di `plt.show` | Minimal-diff, 4 righe |
| `_mc_plot_ci_summary` | + `save_path`; `.write_image` prima di `return fig` | Minimal-diff, 3 righe |
| `_mc_plot_skill_test` | + `save_path`; `.write_image` per `figs[0]` prima di `return figs` | Minimal-diff, 4 righe |
| `_mc_plot_skill_summary` | + `save_path`; `.write_image` prima di `return fig` | Minimal-diff, 3 righe |
| `compare_wfo_pipelines` | + `save_plots`, `plots_dir`; matplotlib equivalents o `write_image` | Moderata (scelta renderer) |

**N. funzioni da modificare**: 6

### Celle da eliminare / consolidare

| Cella | Azione | Condizione |
|-------|--------|------------|
| Cell 19 | Sostituire con chiamata a `compare_wfo_pipelines(..., save_plots=True, plots_dir=plots_dir)` | Dopo che F6 acquisisce save support |
| Cell 31 — Plot B (dendrogramma) | Eliminare; assorbito in `analyze_and_cluster_universe` | Dopo che F2 acquisisce save support |
| Cell 31 — Plot A (heatmap) | Promuovere a funzione canonica con `lookback_days` coerente | Richiede nuova funzione |
| Cell 26 | Sostituire con parametri `save_path` nelle chiamate MC in Cell 25 | Dopo che F8/F9 acquisiscono save support |

**N. celle da consolidare**: 3 celle (19, 26, 31)

### File PNG su disco con naming attuale

```
{ptf}_{year}/
  equity_std.png        ← Cell 19 (matplotlib)
  equity_cluster.png    ← Cell 19 (matplotlib, condizionale)
  equity_comparison.png ← Cell 19 (matplotlib)
  mc_ci.png             ← Cell 26 (plotly → write_image)
  mc_reshuffle.png      ← Cell 26 (plotly → write_image)
  mc_timing.png         ← Cell 26 (plotly → write_image)
  cluster_heatmap.png   ← Cell 31 (matplotlib, ricalcolo con parametri diversi)
  cluster_dendrogram.png← Cell 31 (matplotlib, ricalcolo con parametri diversi) ⚠
```

**Totale**: 8 file PNG per PTF (7 se `use_clustering=False`)  
**PTF con tutti i file**: Alpha Euro, Alpha World Vanguard, Germany Plan, Italy Big Cap (+ Alpha Euro.SAVE)  
**Naming uniforme**: Sì — ma `cluster_scatter.png` non esiste (scatter prodotto da canonica non viene salvato)

### Pattern minimal-diff applicabile a tutte?

**Sì** per F2, F3, F8, F9, F10 — aggiunta di `save_plots`/`save_path` + `savefig`/`write_image` prima di `show()` o `return`.

**Eccezione**: `compare_wfo_pipelines` (F6) usa Plotly per il plot principale ma delega a `analyze_portfolio_metrics` per il radar. Richiederebbe `write_image` per Plotly (già usato in Cell 26) — dipendenza da `kaleido`. Non è un problema tecnico ma richiede coerenza di renderer con Cell 19 (che usa matplotlib).

**Eccezione**: la heatmap di Cell 31 non ha una funzione canonica — richiede promozione a nuova funzione, non solo aggiunta di parametri.

---

## Appendice — Dipendenze di namespace tra celle metacodice

```
Cell 19 → definisce: plots_dir
            ↓
Cell 26 → richiede: plots_dir, ci_results, skill_results
Cell 31 → richiede: plots_dir, results_cluster, stocks_data, use_clustering,
                     portfolio_title, year
```

Se Cell 19 non è eseguita → Cell 26 e Cell 31 falliscono con `NameError`.  
Se Cell 16 non è eseguita (nessun clustering) → Cell 31 stampa "Clustering non eseguito" e non produce file.
