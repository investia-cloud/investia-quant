# Pipeline di Valutazione — K-Strategy (coppia strategia:ticker)

**Progetto**: investia-quant
**Ultimo aggiornamento**: 12 giugno 2026

---

## Ordine di esecuzione

```
1. OFC Precheck   (interno a wfo_strategy_panel)
2. WFO            (solo se precheck passa)
3. DSR            (su risultati WFO)
4. MC             (su risultati WFO, solo se DSR passa)
5. Verdetto finale
```

---

## Step 1 — OFC Precheck

**Cosa fa**: analisi statica della griglia parametri *prima* di girare il WFO
completo. È un gate economico — evita di spendere minuti su strategie che non
hanno speranza.

**Cosa misura**:
- `Gate 1`: il miglior parametro in-sample batte il B&H?
  (soglia: `best_total_return > bh_total_return`)
- `Gate 2`: quante combinazioni della griglia battono il B&H?
  (soglia default: ≥ 40% con scenario B, ≥ 50% con scenario C/E)
- `Gate 3`: `recommend_wfo` — analisi strutturale della superficie parametri
  (plateau, stabilità locale, robustezza della griglia)

**Output**: `pass_gate: True/False` + `reason` — salvato in
`results_panel[key]['precheck']`

**In `run_k_strategy_analysis()`**: letto direttamente da
`precheck['pass_gate']`, non ricalcolato.

---

## Step 2 — WFO (Walk-Forward Optimization)

**Cosa fa**: divide la storia in finestre train/test con ratio configurabile
(default 4:1). Su ogni finestra trova i parametri ottimali in-sample e li
applica out-of-sample. Assembla il portfolio OOS completo concatenando le
finestre.

**Output**: portfolio vectorbt con performance reale OOS —
`results_panel[key]['portfolio']`

**Perché dopo OFC**: il WFO è costoso (centinaia di combinazioni × N finestre
temporali). L'OFC precheck lo protegge da strategie inutili.

---

## Step 3 — DSR (Deflated Sharpe Ratio)

**Cosa fa**: corregge lo Sharpe Ratio per il problema del multiple testing.
Quando si testano N coppie, alcune avranno Sharpe alto per pura fortuna
statistica. Il DSR deflaziona lo Sharpe in base al numero di trial testati e
alla distribuzione dei returns (skewness, kurtosis).

**Formula**: `DSR = P(SR_osservato > SR_casuale)` — probabilità che lo Sharpe
sia genuino e non frutto del data snooping.

**Soglia**: DSR ≥ 0.95 (configurabile con `--dsr-threshold` in `iq k-analyze`)

**Funzione**: `compute_panel_dsr(results_panel)` in `k_functions.py`

**Perché prima di MC**: è computazionalmente leggerissimo (calcolo su returns
già disponibili dal WFO) e taglia i falsi positivi prima del MC che è costoso.

---

## Step 4 — MC (Monte Carlo)

**Cosa fa**: simula N portafogli alternativi perturbando i returns del portfolio
reale. Verifica se la performance osservata è robusta o fragile/fortunata.

**Tre metodi**:

| Metodo | Logica | Cosa testa |
|---|---|---|
| Bootstrap i.i.d. | Ricampiona i returns giornalieri con reinserimento | Performance con ordine casuale — indipendenza dei returns |
| Block Bootstrap (10d) | Ricampiona blocchi contigui di returns | Robustezza preservando l'autocorrelazione (regime persistence) |
| Regime Switching (win=20) | Simula cambi di regime bull/bear tramite matrice di transizione | Robustezza in condizioni di mercato diverse |

**Criteri di pass**: ≥ 2 metodi su 3 con p-value favorevole.

**Funzione**: `run_all_mc_methods()` in `k_functions.py`

---

## Verdetto finale

```
OFC pass_gate = True
AND DSR ≥ 0.95
AND MC ≥ 2/3 metodi pass
→ PROMOSSA
```

Se uno qualsiasi dei tre criteri fallisce → NON PROMOSSA.
Non c'è compensazione tra criteri — tutti e tre devono passare.

---

## Scenari OFC Precheck

Lo scenario modifica la sensibilità del Gate 1/2/3 nell'OFC precheck interno
a `wfo_strategy_panel`.

| Scenario | Gate 1 | Gate 2 | Gate 3 | Uso tipico |
|---|---|---|---|---|
| A | ✅ ON | ❌ OFF | ❌ OFF | Esplorazione — passa quasi tutto, WFO sempre |
| B | ✅ ON | ✅ ≥40% | ❌ OFF | Default — buon equilibrio velocità/qualità |
| C | ✅ ON | ✅ ≥50% | ❌ OFF | Qualità media — più selettivo di B |
| D | ✅ ON | ❌ OFF | ✅ ON | Solo analisi strutturale parametri |
| E | ✅ ON | ✅ ≥50% | ✅ ON | Produzione — massimamente selettivo |

**Nota importante**: in `run_k_strategy_analysis()` lo scenario è fissato
internamente ad **A** — il precheck è permissivo perché il vero filtraggio è
affidato a DSR + MC. Lo scenario è quindi un parametro di performance (evita
WFO inutili su strategie palesemente inadatte), non un criterio di promozione.

---

## Relazione tra i filtri

| Filtro | Cosa rileva | Costo computazionale |
|---|---|---|
| OFC precheck | Griglia parametri senza speranza | Basso (solo IS) |
| WFO | Performance reale OOS | Alto |
| DSR | Falsi positivi da multiple testing | Trascurabile |
| MC | Fragilità/fortuna della performance | Medio |

L'ordine non è casuale — ogni filtro è più costoso del precedente e opera
su un insieme già ridotto dal filtro precedente. Il design minimizza il tempo
complessivo di analisi mantenendo la robustezza statistica del verdetto finale.

---

## Flusso operativo

```
agent.py
  → legge articoli (Medium/web)
  → genera K-strategy nel template canonico
  → appende a notebooks/libs_py/k_strategies_agent.py

iq k-analyze -s <strategie> -t <tickers>
  → dispatch automatico: 1×1 = inspector, N×M = panel
  → pipeline completa: OFC + WFO + DSR + MC
  → risultati su disco: outputs/k_analysis/<data>/plots/classification.png
  → stampa coppie promosse

cron (opzionale)
  → iq k-analyze -s <nuove_strategie> -t <tickers_ptf_trading>
```

---

## Implementazione

| Componente | File | Funzione/Comando |
|---|---|---|
| Pipeline headless | `notebooks/libs_py/k_functions.py` | `run_k_strategy_analysis()` |
| CLI | `investia_quant/cli.py` | `iq k-analyze` |
| OFC precheck | `notebooks/libs_py/k_functions.py` | `overfitting_optimization()` |
| WFO panel | `notebooks/libs_py/k_functions.py` | `wfo_strategy_panel()` |
| DSR | `notebooks/libs_py/k_functions.py` | `compute_panel_dsr()` |
| MC | `notebooks/libs_py/k_functions.py` | `run_all_mc_methods()` |
| Generazione strategie | `K-Strategy-Agent/agent.py` | `run_agent()` |
