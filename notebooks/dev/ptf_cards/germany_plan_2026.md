# PTF Card — Germany Plan 2026

## Identity

| Field | Value |
|-------|-------|
| Name | Germany Plan |
| Engine | R-portfolio (rotational momentum) |
| Universe | 19 German-listed stocks — beneficiaries of Germany's Sondervermögen (special investment vehicle) |
| Category | **A** — event-driven / thematic universe |
| Thesis | Germany announced a ~500B EUR Sondervermögen in early 2025 for defence, energy, rail, digital infrastructure, housing, and logistics. This PTF rotates within the direct beneficiaries of this spending plan. |
| Composition date | 2025 (composed at announcement of Sondervermögen) |
| Composition | `germany_plan_beneficiaries` in `k_tickers.ipynb` Cell 0 L801-841 |

## Bias Profile

| Bias type | Assessment | Detail |
|---|---|---|
| Survivorship bias (tickers) | **Near-absent** | All 19 are established German companies currently listed. The investment horizon matches the plan duration (~2025-2030). No expectation of delistings in this window. |
| Selection bias (modellista) | **Justified by thesis** | Companies selected for direct exposure to specific sectors funded by the Sondervermögen (defence: RHM/HAG/RENK; energy: ENR/EOAN/RWE; rail: SIE/VOS/KBX; etc.). Explicit sectoral logic documented in k_tickers.ipynb comments. |
| Recommendation | 🟢 **GREEN — document & go** | Standard category A. Document thesis and expiry condition. |

## Notes

This is a category A PTF with a natural expiry: the investment thesis is
anchored to the Sondervermögen deployment timeline. When the funds are
fully committed (estimated 2028-2030), the thesis is exhausted and the
PTF should be either closed or re-contextualized.

Sectors covered: defence (4), energy & grids (4), rail & infrastructure (4),
digital & fibre (3), housing & construction (4), logistics (2), semiconductors (3).
Total: 19 + 1 commented-out (test ticker removed).

## Open Items

- **Thesis expiry monitor**: add periodic check (annually) on Sondervermögen
  disbursement progress. When > 80% of funds committed, flag for PTF review.
- **Benchmark choice**: `^GDAXI` (DAX) is used as benchmark. A more precise
  alternative would be a basket of the Sondervermögen sectors or an EU
  infrastructure ETF. Low priority.
