# PTF Card — Alpha Nasdaq100 2026

## Identity

| Field | Value |
|-------|-------|
| Name | Alpha Nasdaq100 |
| Engine | R-portfolio (rotational momentum) |
| Universe | Nasdaq-100 components — resolved at runtime via `extract_tickers_from_wikipedia("nasdaq100")` |
| Category | **C-static** — open index with high historical turnover, no PIT handling |
| Composition | Live Wikipedia scrape of current Nasdaq-100 composition at execution time |

## Bias Profile

| Bias type | Assessment | Detail |
|---|---|---|
| Survivorship bias (tickers) | **Present — highest severity among all PTFs** | Nasdaq-100 has among the highest turnover of major indices. Notable historical additions/removals: Tesla (2020), Zoom (2020-2022), Meta (2022+), ABNB (2023+), many others. Using current 2026 composition for a 15-year backtest systematically over-represents survivors. |
| Selection bias (modellista) | **Absent** | Rule-based: Nasdaq-100 index membership. |
| Estimated bias magnitude | **+3-6% CAGR overstatement** | Higher than SP100 estimate due to higher index turnover. Exact magnitude requires PIT membership data. |
| Recommendation | 🟠 **ORANGE — plan a fix** (higher urgency than SP100) | Backtest structurally optimistic. Not deployable to clients. Higher priority than SP100 due to higher turnover severity. |

## Open Items

### Status: NOT DEPLOYABLE — pending strategic decision

This PTF uses the current Wikipedia composition of the Nasdaq-100 as a
static universe for the entire historical backtest. The framework
correctly handles per-ticker price availability via NaN handling, but
does NOT handle historical index membership.

Consequence: backtest is structurally optimistic by an estimated
+3-6% CAGR over a 15-year horizon (higher than SP100 due to higher
Nasdaq-100 turnover). The PTF cannot be presented as deployable to
clients without misrepresenting expected returns.

### Why this is not a "fix it later" technical debt

Free data sources do not provide rigorous historical index membership.
The yearly lists previously prepared in `r_portfolios.ipynb`
(`alpha_nasdaq100_tickers_by_year`) were artificially derived from the
same biased universe and do not solve the problem.

Resolution requires either subscribing to a paid PIT data vendor
(Norgate, EOD Historical Data, CRSP) or building membership history
from primary sources (SEC filings, index announcements). This is a
**product decision, not a technical task**.

### Decision required from user

1. **Discontinue**: remove from active framework.
2. **Maintain as research-only**: keep with explicit "not for deploy" tag.
3. **Invest in PIT data**: subscribe to a vendor and implement
   `get_universe_at_date`. Becomes a dedicated milestone.

Until this decision is made, the PTF is frozen in current state with
this caveat documented.
