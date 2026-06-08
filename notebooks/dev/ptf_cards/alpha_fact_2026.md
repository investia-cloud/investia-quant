# PTF Card — Alpha Fact 2026

## Identity

| Field | Value |
|-------|-------|
| Name | Alpha Fact |
| Engine | R-portfolio (rotational momentum) |
| Universe | 5 UCITS factor ETFs — iShares MSCI World (Low Vol, Quality, Momentum, Multifactor, Small Cap) |
| Category | **B-rule** — one ETF per MSCI World canonical factor, systematic coverage |
| Composition | `fattoriali` in `k_tickers.ipynb` Cell 0 L179-202 |

## Bias Profile

| Bias type | Assessment | Detail |
|---|---|---|
| Survivorship bias (tickers) | **Absent** | ETFs do not delist. |
| Selection bias (modellista) | **Absent — B-rule** | Standard academic factors (Low Vol, Quality, Momentum, Multifactor, Size). Selection rule is "one ETF per canonical factor". |
| Recommendation | 🟢 **GREEN — document & go** |  |

## Notes

**Bug fixed in post-audit cleanup (2026-05-05)**: `fattoriali` previously
contained IWVL.MI twice — once correctly as Low Volatility ETF, and once
erroneously labelled as "MSCI World Value". The erroneous duplicate was
removed. The Value factor slot is currently empty with a TODO comment.
The list now has 5 tickers instead of 6. Before adding a Value ETF,
verify the correct iShares MSCI World Value ticker (e.g. IWVL is Low Vol,
not Value — candidate: XDEV.MI = MSCI USA Value, or a proper global value ETF).

## Open Items

- Identify and add correct MSCI World Value ETF to `fattoriali` to restore
  full 6-factor coverage. The `# "XDEV.MI"` alternative (MSCI USA Value,
  US-only) in the commented code is a starting point but not ideal for a
  global factor universe.
