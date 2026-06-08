# Refactor Validation Report — FASE 3
**Branch**: `refactor/plot-save-architecture`
**Data esecuzione**: 2026-05-11 12:01–12:05
**Configurazione PTF testata**: Alpha Euro 2026, `use_clustering=True`

---

## 1. Esito rerun

| Parametro | Valore |
|-----------|--------|
| Stato | **SUCCESS** (exit code 0) |
| Tempo esecuzione | ~3m23s (12:02:02 → 12:05:25) |
| Notebook eseguito | `notebooks/dev/R_Asset_v2.ipynb` |
| Kernel usato | Investia-3.12 (`/home/luca/PEnv/Investia-3.12/bin/jupyter`) |
| Log file | `notebooks/dev/refactor_run_log.txt` |

**Note sul kernel**: `jupyter nbconvert` di sistema (Ubuntu) usa `/usr/bin/python3` senza numpy installato → `ModuleNotFoundError` al primo tentativo. Rerun eseguito con il kernel del virtualenv Investia-3.12. Da documentare: per automazioni CLI usare sempre `/home/luca/PEnv/Investia-3.12/bin/jupyter nbconvert`.

**Warning/errori nel log**: nessuno. Output unico:
```
[NbConvertApp] Converting notebook notebooks/dev/R_Asset_v2.ipynb to notebook
[NbConvertApp] Writing 39600218 bytes to notebooks/dev/R_Asset_v2.ipynb
```

---

## 2. File PNG generati

Directory: `notebooks/dev/reports/plots/Alpha Euro_2026/`

| Filename | Size (bytes) | Dimensioni (WxH) | Stato |
|----------|-------------|------------------|-------|
| cluster_dendrogram.png | 53,265 | 1214 × 667 | OK |
| cluster_scatter.png | 71,668 | 1214 × 667 | ⚠️ STESSA DIM. DI DENDROGRAM |
| cluster_heatmap.png | 178,096 | 1896 × 1720 | OK |
| equity_std.png | 57,419 | n/a | OK |
| equity_cluster.png | 48,529 | n/a | OK |
| equity_comparison.png | 104,299 | n/a | OK |
| mc_ci.png | 46,219 | n/a | OK |
| mc_reshuffle.png | 23,559 | n/a | OK |
| mc_timing.png | 23,654 | n/a | OK |
| mc_skill_summary.png | 47,574 | n/a | OK |

**Riepilogo**: tutti 10 file presenti. Nessun file mancante, nessun file extra. Tutti >5KB. File minore: `mc_reshuffle.png` a 23,559 byte (plausibile per test con n basso).

---

## 3. Verifica strutturale notebook

**grep `plt.savefig|fig.write_image` in `R_Asset_v2.ipynb`**: **0 occorrenze** ✓

Le 3 celle metacodice sono state eliminate come previsto:
- ex-Cell 19 (equity save matplotlib) → eliminata ✓
- ex-Cell 26 (MC save metacodice) → eliminata ✓
- ex-Cell 31 (cluster heatmap/dendrogramma metacodice) → eliminata ✓

Tutto il salvataggio plot è ora delegato alle funzioni in `r_functions.ipynb`.

---

## 4. Verifica visiva cluster files

| File | Dimensioni (WxH) | Mode | Content hash (MD5) |
|------|-----------------|------|-------------------|
| cluster_dendrogram.png | 1214 × 667 | RGBA | `0ea69848c29735c48f50c89ff3272c00` |
| cluster_scatter.png | 1214 × 667 | RGBA | `99c382ba6c10ddc9acd26b0951f75603` |
| cluster_heatmap.png | 1896 × 1720 | RGBA | n/a |

**Analisi dimensioni cluster_dendrogram vs cluster_scatter**:

⚠️ **ANOMALIA DA INVESTIGARE**: Le due immagini hanno dimensioni pixel identiche (1214×667). 

La FASE 3 richiedeva dimensioni DIVERSE come conferma che `get_window_extent` estragga correttamente i due axes separati. Le dimensioni identiche possono indicare due scenari:

- **Scenario A (bug)**: `get_window_extent` restituisce lo stesso bounding box per entrambi gli axes → le due immagini riprendono la stessa area della figura.
- **Scenario B (design)**: i due axes nel layout del cluster plot sono dimensionati identicamente dall'autore → entrambi hanno la stessa area fisica → salvataggio corretto ma dimensioni inevitabilmente uguali.

**Elemento mitigante**: i due file hanno MD5 **diversi** — il contenuto è differente, quindi non si tratta di due copie dello stesso file. Questo esclude il caso peggiore (save loop che sovrascrive lo stesso axes due volte).

**Raccomandazione**: ispezione manuale dell'immagine. Aprire entrambi i file e verificare visivamente che `cluster_dendrogram.png` mostri il dendrogramma e `cluster_scatter.png` mostri lo scatter Vol/Momentum. Se entrambe mostrano contenuto corretto → Scenario B, accettabile. Se una delle due appare sbagliata → Scenario A, bug da correggere.

**cluster_heatmap.png**: 1896×1720 → aspect ratio ~1.1:1, plausibile per matrice di correlazione con label tickers ✓

---

## 5. Git diff summary

**Nota strutturale**: il branch `refactor/plot-save-architecture` non ha commit propri sopra `main` (merge-base = HEAD = `c37459f`). Tutte le modifiche sono nel working tree non committato. Il diff rilevante è `git diff HEAD`.

### File modificati (working tree vs HEAD)

| File | +lines / -lines | Ambito | Status |
|------|----------------|--------|--------|
| `notebooks/dev/R_Asset_v2.ipynb` | +715664 / -87153 | Principalmente outputs del rerun | ✓ Atteso |
| `notebooks/libs/r_functions.ipynb` | +188 / -8 | Nuove funzioni save_plots | ✓ Atteso |
| `notebooks/libs/u_functions.ipynb` | +2 / -3194 | Fuori scope — vedi sotto | ⚠️ Inatteso |
| Italy Big Cap plots (3 PNG) | binario | Outputs precedenti altri PTF | Pre-esistente |
| ofc_reports JSON (4 file) | ~8 lines ciascuno | Outputs WFO altri PTF | Pre-esistente |
| stability CSV (2 file) | ~8 lines ciascuno | Outputs stabilità altri PTF | Pre-esistente |
| ptf_cards MD (4 file) | variabile | PTF cards aggiornate | Pre-esistente |

### Conferma F3 invariata

`aggregate_cluster_portfolios` **NON presente nel diff** di `r_functions.ipynb` ✓

La funzione non è stata toccata: firma, corpo e comportamento intatti.

### Conferma scope logica

Il diff di `r_functions.ipynb` (+188/-8) riguarda esclusivamente:
- Aggiunta parametri `save_plots`, `plots_dir`, `save_path` alle funzioni di plotting
- Blocchi `if save_*:` per il salvataggio condizionale
- Nessuna modifica a logica WFO, clustering, MC ✓

### Anomalia u_functions.ipynb

`u_functions.ipynb` mostra -3194/+2 in git diff. Esaminando il diff raw: si tratta di un **cambio di serializzazione JSON** — la stessa cella è passata da formato multi-stringa (lista di righe, `["riga1\n", "riga2\n", ...]`) a singola stringa (`"riga1\nriga2\n..."`). Il contenuto sorgente appare invariato, solo la rappresentazione JSON è diversa.

Questa modifica è **pre-esistente rispetto a FASE 3** (presente già nel `git status` iniziale, prima del rerun). Il rerun di `R_Asset_v2.ipynb` non ha toccato `u_functions.ipynb`.

---

## 6. Anomalie e raccomandazioni

### Anomalia 1 — Dimensioni identiche cluster_dendrogram / cluster_scatter (PRIORITÀ ALTA)

**Cosa**: entrambi i file cluster hanno dimensione 1214×667 pixel.
**Atteso**: dimensioni diverse (i due axes del cluster plot sono di tipo diverso).
**Stato attuale**: contenuto differente (MD5 diversi) → non è un doppio save dello stesso axes.
**Azione richiesta**: aprire manualmente `notebooks/dev/reports/plots/Alpha Euro_2026/cluster_dendrogram.png` e `cluster_scatter.png` e verificare visivamente che mostrino contenuto corretto e distinto. Se ok → closing note nel report, nessun fix necessario. Se uno dei due appare errato → segnalare come bug nella save logic (`get_window_extent` o coordinate bbox).

### Anomalia 2 — u_functions.ipynb fuori scope (PRIORITÀ BASSA)

**Cosa**: `u_functions.ipynb` appare modificato nel working tree con un diff di -3194/+2.
**Valutazione**: probabile cambio di serializzazione JSON (pre-esistente, non causato dal rerun). Il contenuto sorgente non sembra alterato.
**Azione richiesta**: prima del commit, eseguire `git checkout HEAD -- notebooks/libs/u_functions.ipynb` se si vuole escluderlo dal commit del refactor, oppure verificare esplicitamente che il sorgente sia invariato aprendo il file nel notebook.

### Anomalia 3 — Kernel CLI (INFORMATIVA)

**Cosa**: `jupyter nbconvert` di sistema fallisce con `ModuleNotFoundError: No module named 'numpy'`.
**Azione richiesta**: documentare in `CLAUDE.md` o in uno script helper che per esecuzioni CLI del notebook va usato `/home/luca/PEnv/Investia-3.12/bin/jupyter`.

---

## Raccomandazione finale

Il rerun è **passato con successo**. Le funzioni di salvataggio producono tutti i 10 file attesi con dimensioni plausibili. La verifica strutturale è pulita (0 savefig nel notebook principale).

**Prima del commit**: risolvere l'Anomalia 1 (verifica visiva manuale dei due cluster PNG) e valutare l'Anomalia 2 (u_functions.ipynb in scope o da escludere).
