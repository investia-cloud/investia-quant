# PTF Card — Alpha World 2026

## Identity

| Field | Value |
|-------|-------|
| Name | Alpha World |
| Engine | R-portfolio (rotational momentum) |
| Universe | ~21 UCITS ETFs — multi-asset global (EM equity/debt, global bonds, gold, brent, global equity) |
| Category | **B-discr** — multi-asset ETF universe, discretionally curated by asset class |
| Composition | `multiasset_global_ucits` in `k_tickers.ipynb` Cell 0 L376-411 |

## Bias Profile

| Bias type | Assessment | Detail |
|---|---|---|
| Survivorship bias (tickers) | **Absent** | All ETFs; no individual stock mortality risk. |
| Selection bias (modellista) | **Present — B-discr** | No formal rule for which ETFs or how many per asset class. Discretional multi-asset coverage. |
| Recommendation | 🟡 **YELLOW — document caveat & monitor** | Document selection criteria (e.g. "one UCITS ETF per major asset class at date X"). Low operational risk because ETFs don't fail, but composition is not reproducible by rule alone. |

## Notes

Universe mixes asset classes with different risk profiles and correlation
structures (EM equity, gold, brent, treasuries). This is deliberate design
for a multi-asset momentum PTF. The benchmark is SWDA.MI (MSCI World).

## Open Items

- Document composition date and criteria for asset class inclusion.
- Verify all tickers have adequate yfinance price history (some UCITS ETFs
  on Borsa Italiana have limited history < 5 years).
