# Proposed Diff — Bonifica architetturale plot/save

**Fase 1 — Read-only. Nessuna modifica applicata.**  
**Data**: 2026-05-08 | Riferimento: `audit_plot_architecture.md`

---

## A. Modifiche `r_functions.ipynb`

### A.1 — F2: `analyze_and_cluster_universe` (Cell 11, linea 66)

**Firma attuale → firma proposta**

```python
# PRIMA
def analyze_and_cluster_universe(
    prices            : pd.DataFrame,
    n_clusters        : int  = 3,
    lookback_days     : int  = 252,
    plot              : bool = True,
    adaptive_k        : bool = False,
    adaptive_k_method : str  = 'hybrid',
) -> dict:

# DOPO
def analyze_and_cluster_universe(
    prices            : pd.DataFrame,
    n_clusters        : int  = 3,
    lookback_days     : int  = 252,
    plot              : bool = True,
    adaptive_k        : bool = False,
    adaptive_k_method : str  = 'hybrid',
    save_plots        : bool = False,
    plots_dir                = None,   # str | Path | None
) -> dict:
```

**Codice aggiunto — inserimento PRIMA di `plt.show()` alla linea 243**

Contesto attuale (linee 241-244):
```python
        axes[1].legend()
        plt.tight_layout()
        plt.show()                       # ← riga 243
```

Diventa:
```python
        axes[1].legend()
        plt.tight_layout()
        if save_plots and plots_dir is not None:
            _pd = str(plots_dir)
            plt.savefig(f"{_pd}/cluster_scatter.png",    dpi=150, bbox_inches='tight')
            plt.savefig(f"{_pd}/cluster_dendrogram.png", dpi=150, bbox_inches='tight')
        plt.show()
```

**File PNG salvati**:
- `cluster_scatter.png` — figura 1×2 completa (dendrogramma + scatter Vol/Momentum)
- `cluster_dendrogram.png` — stessa figura: mantiene compatibilità con link esistenti in PTF cards; elimina la versione errata di Cell 31

**Nota DIV-1/DIV-2**: entrambi i file ora contengono il dendrogramma corretto (distanza combinata 60/40, `lookback_days` del WFO), non il ricalcolo di Cell 31. Il pannello scatter era irrecuperabile a posteriori — ora è salvato.

---

### A.2 — F3: `aggregate_cluster_portfolios` (Cell 11, linea 418)

**Firma attuale → firma proposta**

```python
# PRIMA
def aggregate_cluster_portfolios(
    wfo_results    : dict,
    stocks_data    : pd.DataFrame,
    benchmark_data,
    regime         : pd.Series,
    weight_on      : dict  = None,
    weight_off     : dict  = None,
    start_date     : str   = None,
    end_date       : str   = None,
    init_cash      : float = 100_000,
    plot           : bool  = True,
) -> dict:

# DOPO
def aggregate_cluster_portfolios(
    wfo_results    : dict,
    stocks_data    : pd.DataFrame,
    benchmark_data,
    regime         : pd.Series,
    weight_on      : dict  = None,
    weight_off     : dict  = None,
    start_date     : str   = None,
    end_date       : str   = None,
    init_cash      : float = 100_000,
    plot           : bool  = True,
    save_plots     : bool  = False,
    plots_dir             = None,    # str | Path | None
) -> dict:
```

**Codice aggiunto — inserimento PRIMA di `plt.show()` alla linea 526**

Contesto attuale (linee 524-527):
```python
        plt.tight_layout()
        plt.show()                       # ← riga 526
```

Diventa:
```python
        plt.tight_layout()
        if save_plots and plots_dir is not None:
            plt.savefig(f"{str(plots_dir)}/cluster_aggregate.png", dpi=150, bbox_inches='tight')
        plt.show()
```

**File PNG salvato**: `cluster_aggregate.png`

**Nota**: questa funzione non è chiamata dal notebook principale corrente (audit §1.1 F3). La modifica è preventiva per futura integrazione.

---

### A.3 — F6: `compare_wfo_pipelines` (Cell 12, linea 0)

**Firma attuale → firma proposta**

```python
# PRIMA
def compare_wfo_pipelines(
    results_std     : dict,
    results_cluster : dict,
    portfolio_title : str  = "Portfolio",
    benchmark_title : str  = "Benchmark",
    plot_radar      : bool = True,
    start_date      : str  = None,
    end_date        : str  = None,
) -> pd.DataFrame:

# DOPO
def compare_wfo_pipelines(
    results_std     : dict,
    results_cluster : dict,
    portfolio_title : str  = "Portfolio",
    benchmark_title : str  = "Benchmark",
    plot_radar      : bool = True,
    start_date      : str  = None,
    end_date        : str  = None,
    save_plots      : bool = False,
    plots_dir              = None,    # str | Path | None
) -> pd.DataFrame:
```

**Codice aggiunto — inserimento DOPO `fig.show()` alla linea 153, PRIMA della sezione "Tabella metriche"**

Contesto attuale (linee 152-156):
```python
    )
    fig.show()                           # ← riga 153

    # ------------------------------------------------------------------
    # 2. Tabella metriche + Radar (via analyze_portfolio_metrics)
```

Diventa:
```python
    )
    if save_plots and plots_dir is not None:
        from pathlib import Path as _P
        _pd = _P(str(plots_dir))
        _pd.mkdir(parents=True, exist_ok=True)
        # equity_comparison.png — tutti i path (fig già costruita sopra)
        fig.write_image(str(_pd / 'equity_comparison.png'))
        # equity_std.png — solo percorsi Standard
        _std_cols = [c for c in port_cumrets.columns
                     if c in (lbl['std_on'], lbl['std_base'])]
        if _std_cols:
            _fig_std = go.Figure()
            for _c in _std_cols:
                _fig_std.add_trace(go.Scatter(
                    x=port_cumrets.index, y=port_cumrets[_c], name=_c, mode='lines',
                    line=dict(color=COLORS.get(_c,'#333'), dash=DASH.get(_c,'solid'), width=WIDTH.get(_c,2))))
            if bm_cumret is not None:
                _ba = bm_cumret.reindex(port_cumrets.index, method='ffill')
                _fig_std.add_trace(go.Scatter(
                    x=_ba.index, y=_ba.values, name=benchmark_title, mode='lines',
                    line=dict(color=COLORS[benchmark_title], dash=DASH[benchmark_title], width=WIDTH[benchmark_title])))
            _fig_std.update_layout(title=f"Rendimenti cumulativi – Standard – {portfolio_title}",
                                   height=400, width=900, template='plotly_white',
                                   hovermode='x unified')
            _fig_std.write_image(str(_pd / 'equity_std.png'))
        # equity_cluster.png — solo percorsi Cluster (condizionale)
        _cl_cols = [c for c in port_cumrets.columns
                    if c in (lbl.get('cl_on',''), lbl.get('cl_base',''))]
        if _cl_cols:
            _fig_cl = go.Figure()
            for _c in _cl_cols:
                _fig_cl.add_trace(go.Scatter(
                    x=port_cumrets.index, y=port_cumrets[_c], name=_c, mode='lines',
                    line=dict(color=COLORS.get(_c,'#333'), dash=DASH.get(_c,'solid'), width=WIDTH.get(_c,2))))
            if bm_cumret is not None:
                _ba = bm_cumret.reindex(port_cumrets.index, method='ffill')
                _fig_cl.add_trace(go.Scatter(
                    x=_ba.index, y=_ba.values, name=benchmark_title, mode='lines',
                    line=dict(color=COLORS[benchmark_title], dash=DASH[benchmark_title], width=WIDTH[benchmark_title])))
            _fig_cl.update_layout(title=f"Rendimenti cumulativi – Cluster – {portfolio_title}",
                                   height=400, width=900, template='plotly_white',
                                   hovermode='x unified')
            _fig_cl.write_image(str(_pd / 'equity_cluster.png'))
    fig.show()

    # ------------------------------------------------------------------
    # 2. Tabella metriche + Radar (via analyze_portfolio_metrics)
```

**File PNG salvati**:
- `equity_comparison.png` — Plotly multi-path (tutti i path)
- `equity_std.png` — Plotly, solo percorsi Standard
- `equity_cluster.png` — Plotly, solo percorsi Cluster (condizionale: solo se dati cluster presenti)

**Nota DIV-3**: tutti e tre i file ora usano Plotly + `write_image` (stessa fonte dati di `fig.show()`). La divergenza di renderer e normalizzazione rispetto alla vecchia Cell 19 matplotlib è eliminata.

---

### A.4 — F8: `_mc_plot_ci_summary` (Cell 15, linea 440)

**Firma attuale → firma proposta**

```python
# PRIMA
def _mc_plot_ci_summary(ci_results: dict, actual_equities: dict) -> go.Figure:

# DOPO
def _mc_plot_ci_summary(
    ci_results      : dict,
    actual_equities : dict,
    save_path               = None,    # str | Path | None
) -> go.Figure:
```

**Codice aggiunto — inserimento PRIMA di `return fig` alla linea 489**

Contesto attuale (linee 488-490):
```python
    ))
    return fig                           # ← riga 489
```

Diventa:
```python
    ))
    if save_path is not None:
        fig.write_image(str(save_path))
    return fig
```

**File PNG salvato**: il path viene passato dal caller (es. `plots_dir / 'mc_ci.png'`).

---

### A.5 — F9: `_mc_plot_skill_test` (Cell 15, linea 492)

**Firma attuale → firma proposta**

```python
# PRIMA
def _mc_plot_skill_test(
    test_label: str,
    result: dict,
    actual_equity: pd.Series,
) -> list:

# DOPO
def _mc_plot_skill_test(
    test_label    : str,
    result        : dict,
    actual_equity : pd.Series,
    save_path             = None,    # str | Path | None — salva figs[0] (CAGR histogram)
) -> list:
```

**Codice aggiunto — inserimento PRIMA di `return figs` alla linea 540**

Contesto attuale (linee 538-541):
```python
        figs.append(fig)

    return figs                          # ← riga 540
```

Diventa:
```python
        figs.append(fig)

    if save_path is not None and figs:
        figs[0].write_image(str(save_path))
    return figs
```

**File PNG salvato**: `figs[0]` = istogramma CAGR (il più informativo per il report tecnico). Il path viene passato dal caller.

---

### A.6 — F10: `_mc_plot_skill_summary` (Cell 15, linea 543)

**Firma attuale → firma proposta**

```python
# PRIMA
def _mc_plot_skill_summary(skill_results: dict) -> go.Figure:

# DOPO
def _mc_plot_skill_summary(
    skill_results : dict,
    save_path             = None,    # str | Path | None
) -> go.Figure:
```

**Codice aggiunto — inserimento PRIMA di `return fig` alla linea 603**

Contesto attuale (linee 602-604):
```python
    ))
    return fig                           # ← riga 603
```

Diventa:
```python
    ))
    if save_path is not None:
        fig.write_image(str(save_path))
    return fig
```

**File PNG salvato**: il path viene passato dal caller (es. `plots_dir / 'mc_skill_summary.png'`).

---

### A.7 — Wrappers MC: passthrough di `save_plots`/`plots_dir`

Tre wrapper in Cell 15 devono essere estesi per propagare i parametri di salvataggio verso F8/F9/F10.

#### `run_mc_confidence_intervals_rotational` (linea 1047)

Aggiungere alla firma (dopo `show_method_summaries`):
```python
    save_plots     : bool = False,
    plots_dir             = None,
```

Modifica al blocco `show_method_plots` (linee 1188-1194) — da:
```python
    if show_method_plots:
        for key, res in ci_results.items():
            if res is None:
                continue
            for fig in _mc_plot_ci_method(method_labels[key], res, equity_actual, bm_equity):
                fig.show()
        _mc_plot_ci_summary(ci_results, {'pf_rot': equity_actual, 'pf_rot_base': equity_base}).show()
```

A:
```python
    if show_method_plots:
        for key, res in ci_results.items():
            if res is None:
                continue
            for fig in _mc_plot_ci_method(method_labels[key], res, equity_actual, bm_equity):
                fig.show()
        _save_ci = (str(_Path(plots_dir) / 'mc_ci.png')
                    if save_plots and plots_dir is not None else None)
        _mc_plot_ci_summary(
            ci_results, {'pf_rot': equity_actual, 'pf_rot_base': equity_base},
            save_path=_save_ci,
        ).show()
```

Richiede `from pathlib import Path as _Path` — già disponibile in Cell 15 (da verificare; se non presente aggiungere import locale).

#### `run_mc_skill_tests_rotational` (linea 1203)

Aggiungere alla firma (dopo `show_method_summaries`):
```python
    save_plots     : bool = False,
    plots_dir             = None,
```

Aggiungere mapping key → filename, poi modificare blocco `show_method_plots` (linee 1363-1369) — da:
```python
    if show_method_plots:
        for key, res in skill_results.items():
            if res is None:
                continue
            for fig in _mc_plot_skill_test(test_labels[key], res, equity_actual):
                fig.show()
        _mc_plot_skill_summary(skill_results).show()
```

A:
```python
    _skill_filenames = {
        'rotation_reshuffle': 'mc_reshuffle.png',
        'rebalance_timing':   'mc_timing.png',
    }
    if show_method_plots:
        for key, res in skill_results.items():
            if res is None:
                continue
            _save_sk = (str(_Path(plots_dir) / _skill_filenames[key])
                        if save_plots and plots_dir is not None else None)
            for fig in _mc_plot_skill_test(test_labels[key], res, equity_actual,
                                            save_path=_save_sk):
                fig.show()
        _save_ss = (str(_Path(plots_dir) / 'mc_skill_summary.png')
                    if save_plots and plots_dir is not None else None)
        _mc_plot_skill_summary(skill_results, save_path=_save_ss).show()
```

#### `run_all_mc_methods_rotational` (linea 1378)

Aggiungere alla firma (dopo `show_method_summaries`):
```python
    save_plots     : bool = False,
    plots_dir             = None,
```

Passare ai due blocchi interni (linee 1425-1442) — aggiungere:
```python
    ci_results, ci_summary_df = run_mc_confidence_intervals_rotational(
        ...,
        show_method_plots=show_method_plots,
        show_method_summaries=show_method_summaries,
        save_plots=save_plots,
        plots_dir=plots_dir,
        _show_portfolio_header=False,
    )
    skill_results, skill_summary_df = run_mc_skill_tests_rotational(
        ...,
        show_method_plots=show_method_plots,
        show_method_summaries=show_method_summaries,
        save_plots=save_plots,
        plots_dir=plots_dir,
        _show_portfolio_header=False,
    )
```

---

### A.8 — `run_wfo_pipeline` (Cell 11, linea 932): passthrough a F2

Aggiungere alla firma (dopo `plot: bool = True`):
```python
    save_plots     : bool = False,
    plots_dir             = None,
```

Modificare chiamata a `analyze_and_cluster_universe` (linee 1075-1082) — aggiungere i due parametri:
```python
        cluster_result = analyze_and_cluster_universe(
            prices            = prices_df,
            n_clusters        = n_clusters,
            lookback_days     = lookback_days,
            plot              = plot,
            adaptive_k        = adaptive_k,
            adaptive_k_method = adaptive_k_method,
            save_plots        = save_plots,
            plots_dir         = plots_dir,
        )
```

---

## B. Nuova funzione `plot_cluster_heatmap`

### Firma completa

```python
def plot_cluster_heatmap(
    cluster_result : dict,
    stocks_data    : pd.DataFrame,
    lookback_days  : int  = 252,
    save_path             = None,    # str | Path | None
) -> None:
    """
    Heatmap di correlazione dell'universo, ordinata per cluster, con blocchi colorati.

    Usa la stessa finestra temporale del WFO (lookback_days) per coerenza con
    il clustering effettivo. NON usa tail() fisso indipendente dal WFO.

    Parameters
    ----------
    cluster_result : output di analyze_and_cluster_universe (chiavi: cluster_groups, cluster_labels)
    stocks_data    : prezzi storici — stessa sorgente passata al WFO
    lookback_days  : finestra in giorni — DEVE corrispondere al lookback_days del run WFO
    save_path      : se fornito, salva la figura nel path indicato (dpi=150)
    """
    import matplotlib.patches as patches

    cluster_groups = cluster_result['cluster_groups']
    cluster_labels = cluster_result['cluster_labels']

    sorted_t = []
    for cid in sorted(cluster_groups.keys()):
        sorted_t.extend([t for t in cluster_groups[cid] if t in stocks_data.columns])

    if not sorted_t:
        print("plot_cluster_heatmap: nessun ticker disponibile in stocks_data — skip")
        return

    ret_sub = (stocks_data[sorted_t]
               .dropna(how='all')
               .tail(lookback_days)
               .pct_change()
               .dropna(how='all'))
    corr = ret_sub.corr()

    PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fs = max(10, len(sorted_t) * 0.38)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.85))
    im = ax.imshow(corr.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(sorted_t)))
    ax.set_xticklabels(sorted_t, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sorted_t)))
    ax.set_yticklabels(sorted_t, fontsize=7)
    pos = 0
    for i, cid in enumerate(sorted(cluster_groups.keys())):
        n = len([t for t in cluster_groups[cid] if t in sorted_t])
        rect = patches.Rectangle(
            (pos - 0.5, pos - 0.5), n, n,
            lw=2.5, edgecolor=PALETTE[i % len(PALETTE)], facecolor='none')
        ax.add_patch(rect)
        ax.text(pos + n / 2 - 0.5, -1.2,
                cluster_labels.get(cid, f'C{cid}'),
                ha='center', fontsize=8,
                color=PALETTE[i % len(PALETTE)], fontweight='bold')
        pos += n
    ax.set_title(f'Correlazione per Cluster (ultimi {lookback_days} gg)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.show()
```

### Riferimenti e scelte di design

- **Fonte dati**: `cluster_result['cluster_groups']` e `['cluster_labels']` — lo STESSO oggetto restituito da `analyze_and_cluster_universe` e presente in `results_cluster['cluster_result']`. Zero ricalcoli.
- **Finestra temporale**: `tail(lookback_days)` dove `lookback_days` = il valore usato nel WFO (Cell 16: `lookback_days=504`). Non più `tail(504)` hard-coded indipendente dal WFO.
- **Render grafico**: stessa resa visiva di Cell 31 Plot A (imshow, blocchi colorati, label cluster).
- **`import matplotlib.patches`**: importato inline nella funzione (non presente nei top-level import di Cell 11).

### Posizione di inserimento in r_functions.ipynb

Cell 11, tra la fine di `analyze_and_cluster_universe` (linea 250) e l'inizio di `build_cluster_grids` (linea 253):

```
linea 250:     )    ← fine return di analyze_and_cluster_universe
linea 251:
linea 252:           ← INSERIRE QUI plot_cluster_heatmap
linea 253: def build_cluster_grids(
```

---

## C. Modifiche `R_Asset_v2.ipynb`

### C.1 — Cell 2: definizione esplicita di `plots_dir`

Aggiungere IN FONDO alla cella (dopo la definizione di `year = 2026`):

```python
from pathlib import Path as _Path
plots_dir = _Path(f"reports/plots/{portfolio_title}_{year}")
plots_dir.mkdir(parents=True, exist_ok=True)
print(f"plots_dir: {plots_dir}")
```

`plots_dir` è ora una variabile di setup come `portfolio_title` e `year`. Non più implicita e dipendente dall'esecuzione di Cell 19.

### C.2 — Cell 16: aggiungere save_plots + plots_dir a `run_wfo_pipeline`

Aggiungere alla chiamata `run_wfo_pipeline` (DOPO `plot = True`):
```python
    # Salvataggio plot
    save_plots         = True,
    plots_dir          = plots_dir,
```

E aggiungere, in FONDO alla cella (dopo le righe di estrazione dei risultati), la chiamata alla nuova funzione:
```python
# Heatmap correlazione cluster (finestra coerente con WFO)
if results_cluster is not None and results_cluster.get('cluster_result') is not None:
    plot_cluster_heatmap(
        cluster_result = results_cluster['cluster_result'],
        stocks_data    = stocks_data,
        lookback_days  = lookback_days,
        save_path      = plots_dir / 'cluster_heatmap.png',
    )
```

### C.3 — Cell 18: aggiungere save_plots + plots_dir a `compare_wfo_pipelines`

```python
# PRIMA
if results_cluster is not None:
    metrics_df = compare_wfo_pipelines(
        results_std     = results_std,
        results_cluster = results_cluster,
        portfolio_title = portfolio_title,
        benchmark_title = benchmark_title,
        plot_radar      = True,
    )

# DOPO
if results_cluster is not None:
    metrics_df = compare_wfo_pipelines(
        results_std     = results_std,
        results_cluster = results_cluster,
        portfolio_title = portfolio_title,
        benchmark_title = benchmark_title,
        plot_radar      = True,
        save_plots      = True,
        plots_dir       = plots_dir,
    )
```

### C.4 — Cell 25: aggiungere save_plots + plots_dir a `run_all_mc_methods_rotational`

```python
# PRIMA
ci_results, ci_summary_df, skill_results, skill_summary_df = run_all_mc_methods_rotational(
    ...,
    show_method_plots     = True,
    show_method_summaries = True,
)

# DOPO
ci_results, ci_summary_df, skill_results, skill_summary_df = run_all_mc_methods_rotational(
    ...,
    show_method_plots     = True,
    show_method_summaries = True,
    save_plots            = True,
    plots_dir             = plots_dir,
)
```

### C.5 — Eliminazione Cell 19 (§5c "Salvataggio equity comparison")

**Eliminare l'intera cella.** La logica è ora in F6 (`compare_wfo_pipelines`) con `save_plots=True`.

**Effetto a cascata**: `plots_dir` non è più definita in Cell 19 — è ora definita in Cell 2 (C.1). Le celle successive (ex-Cell 26, ex-Cell 31) non hanno più dipendenza implicita.

### C.6 — Eliminazione Cell 26 (§7 "Salvataggio MC plot")

**Eliminare l'intera cella.** Il salvataggio è ora dentro `run_all_mc_methods_rotational` → `run_mc_skill_tests_rotational` → F9/F10, chiamato una sola volta in Cell 25.

### C.7 — Eliminazione Cell 31 (§8 "Cluster heatmap e dendrogramma")

**Eliminare l'intera cella.**  
- `cluster_dendrogram.png` + `cluster_scatter.png`: ora da F2, inserimento in Cell 16 via `run_wfo_pipeline(save_plots=True)`.  
- `cluster_heatmap.png`: ora da `plot_cluster_heatmap()`, chiamata in Cell 16.

---

## D. Set canonico di output finale

```
reports/plots/{portfolio_title}_{year}/
  cluster_scatter.png      ← F2  analyze_and_cluster_universe (fig 1×2: dendro+scatter)
  cluster_dendrogram.png   ← F2  stessa fig, secondo save — compatibilità PTF cards ⚑
  cluster_heatmap.png      ← NEW plot_cluster_heatmap (corr matrix, lookback_days coerente WFO)
  equity_std.png           ← F6  compare_wfo_pipelines / write_image — Plotly
  equity_cluster.png       ← F6  compare_wfo_pipelines / write_image — Plotly (condiz.)
  equity_comparison.png    ← F6  compare_wfo_pipelines / write_image — Plotly
  mc_ci.png                ← F8  _mc_plot_ci_summary / write_image
  mc_reshuffle.png         ← F9  _mc_plot_skill_test  / write_image (B1)
  mc_timing.png            ← F9  _mc_plot_skill_test  / write_image (B2)
  mc_skill_summary.png     ← F10 _mc_plot_skill_summary / write_image   ← NUOVO
```

**Totale**: 10 file per PTF con clustering; 7 senza (`cluster_*` e `equity_cluster.png` esclusi)

⚑ `cluster_dendrogram.png` = stessa figura di `cluster_scatter.png` (due `plt.savefig` dello stesso oggetto `fig`). Entrambi mostrano dendrogramma + scatter. Il nome `cluster_dendrogram.png` è mantenuto per compatibilità con i link in PTF cards. Futuro cleanup: unificare in un unico file (fuori scope di questo intervento).

---

## E. Verifica coerenza con audit

### DIV-1 (CRITICA) — RISOLTO ✓

**Prima**: `cluster_dendrogram.png` prodotto da Cell 31 con distanza `(1−corr)/2` e `tail(504)` dalla data di esecuzione del notebook → clustering strutturalmente diverso dal WFO.

**Dopo**: `cluster_dendrogram.png` prodotto da F2 con distanza combinata `0.6×corr_dist + 0.4×feat_dist` e `lookback_days` del WFO → STESSA figura mostrata inline durante STEP 1 del WFO. È letteralmente lo stesso oggetto `fig` salvato prima di `plt.show()`.

### DIV-2 (MODERATA) — RISOLTO ✓

**Prima**: scatter Vol/Momentum (da F2) visibile solo inline, irrecuperabile a posteriori.

**Dopo**: `cluster_scatter.png` salvato da F2 prima di `plt.show()`. Il file contiene il pannello scatter (+ dendrogramma) con le etichette ticker esatte usate dal WFO.

### DIV-3 (MODERATA) — RISOLTO ✓

**Prima**: equity plots su disco (Cell 19) = matplotlib headless, normalizzazione `pf.value()/pf.value().iloc[0]`. Equity plots JN (F6 `compare_wfo_pipelines`) = Plotly, `pf.cumulative_returns() + 1`.

**Dopo**: equity plots su disco = stessa `fig` Plotly di F6, salvata con `write_image` immediatamente dopo `fig.show()`. Stessa normalizzazione, stesso renderer, zero divergenza.

### DIV-4 (MINORE) — RISOLTO ✓

**Prima**: ogni plot MC generato due volte — una in Cell 25 (`run_all_mc_methods_rotational`) e una in Cell 26 (ri-chiamata esplicita).

**Dopo**: generazione unica in Cell 25. Il `save_path` viene propagato attraverso i wrapper fino a F8/F9/F10. `write_image` avviene dentro la funzione prima di `return fig`. Cell 26 eliminata.

### DIV-5 (INFORMATIVA) — RISOLTO ✓

**Prima**: `plots_dir` definita come effetto collaterale di Cell 19, dipendenza implicita per Cell 26 e Cell 31.

**Dopo**: `plots_dir` definita esplicitamente in Cell 2 (setup PTF), passata come parametro a tutte le funzioni che la usano. Nessuna variabile globale di namespace implicita.

### DIV-6 (INFORMATIVA) — RISOLTO ✓

**Prima**: Cell 31 (e Cell 26) sono celle terminali eseguite solo se il run è completo fino a Cell 28+. Dipendenze nascoste da `NameError` in caso di run parziale.

**Dopo**: tutto il salvataggio avviene direttamente nelle funzioni canoniche, chiamate nei punti naturali del flusso (Cell 16 per clustering, Cell 18 per equity, Cell 25 per MC). Non esistono più celle terminali critiche con dipendenze di namespace fragili.

---

## Note implementative

1. **kaleido**: `fig.write_image()` richiede `kaleido`. Già in uso in Cell 26 corrente — nessuna dipendenza nuova.

2. **`from pathlib import Path as _P`**: usato inline nei blocchi save. Alternativa: aggiungere `from pathlib import Path` ai top-level import dei file di libreria.

3. **`lbl` in F6**: i dizionari `lbl`, `COLORS`, `DASH`, `WIDTH` sono locali alla funzione. I filtri `_std_cols` e `_cl_cols` usano queste variabili che esistono già nello scope del blocco save aggiunto.

4. **`_Path` in wrappers MC**: aggiungere `from pathlib import Path as _Path` come import locale nel blocco save, oppure verificare che sia già importato nella cell 15 header.

5. **Ordine salvataggi vs show**: in tutti i casi il salvataggio precede lo show() — rispettando la regola 4 ("salvataggio PRIMA di plt.show()/fig.show()"). Il disco riceve la figura identica a quella mostrata nel notebook.
