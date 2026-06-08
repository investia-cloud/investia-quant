# PTF Card — Alpha Sect (Megatrend) 2026

## Identity

| Field | Value |
|-------|-------|
| Name | Alpha Sect (Megatrend) |
| Engine | R-portfolio (rotational momentum) |
| Universe | 9 UCITS ETFs — iShares S&P US Select Sector (XL*.MI, Borsa Italiana) |
| Category | **B-rule** — one ETF per S&P US sector, systematic full-coverage |
| Composition | `settoriali` in `k_tickers.ipynb` Cell 0 L157-168 |

## Bias Profile

| Bias type | Assessment | Detail |
|---|---|---|
| Survivorship bias (tickers) | **Absent** | ETFs do not delist. AUM may shrink but the instrument persists or merges. |
| Selection bias (modellista) | **Absent — B-rule** | One ETF per GICS sector, systematic coverage of all 9 S&P sectors. No discretion. |
| Recommendation | 🟢 **GREEN — document & go** | Bias profile clean. Add category to PTF card metadata when portfolio object supports it. |

## Notes

Universe covers all 9 S&P US Select Sector ETFs listed on Borsa Italiana.
The iShares XL*.MI series tracks a single equity benchmark per sector —
clean, rule-based, reproducible.

Historical availability: all XL*.MI ETFs listed in 2015-2017 timeframe.
Backtest depth before 2015 is limited by the ETF inception dates.

## Open Items

- Verify inception dates of all 9 XL*.MI ETFs and document earliest
  valid backtest start date in this card.
