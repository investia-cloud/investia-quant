# PTF Card — Alpha Quant 2026

## Identity

| Field | Value |
|-------|-------|
| Name | Alpha Quant |
| Engine | R-portfolio (rotational momentum) |
| Universe | ~18 instruments — quantum pureplays + AI data center ETFs + bitcoin miners (composite thematic) |
| Category | **A** — event-driven / thematic universe |
| Thesis | Exposure to the AI/quantum computing wave post-2022: quantum computing hardware, AI data center infrastructure, bitcoin mining as compute/energy thesis |
| Composition | `ai_dc_quantum_thematic = quantum_pureplays + data_center_infrastructure + ai_quantum_enablers + bitcoin_miners_ai` in `k_tickers.ipynb` Cell 0 L46-101 |

## Bias Profile

| Bias type | Assessment | Detail |
|---|---|---|
| Survivorship bias (tickers) | **Near-absent for category A** | Universe defined by a specific thesis, not historical membership in an index. Most tickers are recent IPOs (2021+) — short but complete history. |
| Selection bias (modellista) | **Justified by thesis** | The modellista selects companies based on their fit with the AI/quantum thesis. This is explicit and expected for category A. |
| Recommendation | 🟢 **GREEN — document & go** | Typical category A: thesis-justified selection, intrinsically short horizon. Document composition date and thesis context. |

## Notes

**Important**: category A PTFs have structurally short relevant horizons.
The AI/quantum thesis is anchored to post-2022 dynamics. Backtesting before
2022 is possible but has limited relevance (quantum pureplays did not exist,
bitcoin miners had different risk profiles). The analysis window should be
restricted to 2022-present for meaningful validation.

Several tickers have very short histories:
- IONQ: IPO Oct 2021
- RGTI: IPO Oct 2021
- QBTS (D-Wave): IPO Nov 2022 via SPAC

The motore handles missing history via NaN → exclusion from rebalancing
until data is available.

## Open Items

- Document composition date and the specific thesis articulation in
  k_tickers.ipynb comments (currently only implied from the ticker descriptions).
- Review whether bitcoin mining tickers remain part of the thesis or should
  be moved to a separate "compute infrastructure" sub-universe.
- Consider restricting the WFO analysis window to 2022+ to match thesis horizon.
