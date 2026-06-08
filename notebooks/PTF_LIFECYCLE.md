# PTF Lifecycle — Operational Guide

This document describes the end-to-end workflow for designing, validating, and promoting a new portfolio (PTF) within the framework. It is the practical operating manual for the modellista.

For terminology: see `GLOSSARY.md`.
For Claude Code interaction protocol: see `CLAUDE.md`.

---

## Overview: the five phases

A PTF moves through five phases from concept to deploy. Phases 1-2 and 4-5 are manual decisions by the modellista. Phase 3 is automated execution via R_Asset_v2.ipynb.

| Phase | Owner | Output | Time |
|-------|-------|--------|------|
| 1. Conception | Modellista | Concept note | Variable |
| 2. Portfolio definition | Modellista | Portfolio dict in r_portfolios.ipynb | 5-10 min |
| 3. Framework execution | R_Asset_v2 (modellista launches) | wfo_summary, stability_report, ofc_report, auto-generated PTF card | 30-60 min |
| 4. PTF card completion | Modellista | Complete PTF card with bias profile | 15-30 min |
| 5. Deploy decision | Modellista | Binary verdict + production setup if deployed | Variable |

---

## Phase 1 — Conception

Before writing any code, the modellista clarifies the PTF design:

- **Theme / objective**: what is the investment thesis?
- **Target profile**: satellite or core?
- **Expected category**: A, B-rule, B-discr, or C-static? See GLOSSARY entry "PTF categories".
- **Universe sketch**: rough idea of which tickers / how many / which market.

If the expected category is **C-static**, stop here. The framework currently does not support PIT data and PTFs in this category cannot be deployed without strategic decision on data investment. See `CLAUDE.md` section "No point-in-time universe handling for category C PTFs".

**Output**: a clear concept. Optional: short note in `notebooks/dev/_workbench/dev_<ptf_name>_concept.md`.

---

## Phase 2 — Portfolio definition

Add the portfolio dict to `notebooks/libs/r_portfolios.ipynb`:

```python
portfolio_<ptf_name> = {
    "Title": "<Display Title>",
    "tickers": [...],
    "benchmark_portfolio": None,
    "benchmark_title": "<benchmark ticker>",
}
```

**Ticker list construction**: prefer rule-based criteria you can document.

- Good: "components of DAX + CAC40 + FTSE MIB + IBEX 35 as of 2024-12-31, market cap > 10B EUR"
- Avoid: "tickers I think look solid"

If criteria are rule-based → B-rule. If discretional → B-discr (acceptable but with caveats).

**Open technical debt**: portfolio dict currently lacks `composed_on`, `composition_rule`, `category` fields. These live in the PTF card for now.

**Output**: portfolio variable accessible from R_Asset_v2.

---

## Phase 3 — Framework execution via R_Asset_v2

Open `notebooks/dev/R_Asset_v2.ipynb` and configure §2:

```python
portfolio = portfolio_<ptf_name>
profile = "satellite"  # or "core"
```

Execute sections sequentially:

1. §1 Bootstrap
2. §2 Portfolio configuration
3. §3 Download data — verify ticker count, no serious NaN
4. §4 Stability Analysis — `reduced_grid` ~16x smaller than full grid
5. §5a WFO Standard
6. §5b WFO Cluster (optional)
7. §5c Compare WFO (if cluster ran)
8. §6 Overfitting Check — saves JSON in `data/ofc_reports/`
9. §7 Monte Carlo
10. §8 Decision + auto PTF card — generates `<ptf_name>_<year>_auto.md` in `notebooks/dev/ptf_cards/`
11. §9 Performance
12. §10 Load WFO Results (skip on first run)

**What §8 produces**: auto PTF card with quantitative results (S1/S2/S3/S4 verdicts, MC reshuffle, MC CI, skill profile, promotion decision). Does NOT contain category, bias profile, recommendation. Those are added in Phase 4.

---

## Phase 4 — PTF card completion

Reference template: `notebooks/dev/ptf_cards/alpha_euro_2026.md`.

### Sections to add manually

**1. Identity → Category** (top of file)

```markdown
- Category: B-rule
- Universe: <description>
- Composition: <rule-based / discretional / event-driven>
- Composed on: <YYYY-MM-DD>
```

**2. Bias profile** (new section)

```markdown
## Bias profile

- Survivorship bias: <Assente | Limitato | Presente | Gestito>
- Selection bias: <Assente | Mitigato | Presente | Giustificato>
- Recommendation: <GREEN | YELLOW | ORANGE>
```

Quick reference for typical category → recommendation:

| Category | Survivorship | Selection | Recommendation |
|----------|--------------|-----------|----------------|
| A | Absent | Justified | GREEN |
| B-rule | Limited | Mitigated | GREEN |
| B-discr | Limited | Present | YELLOW |
| C-static | Present | Mitigated | ORANGE (deploy-blocked) |
| C-pit | Managed | Mitigated | GREEN (when implemented) |

**3. Notes** (if relevant): PTF-specific methodological notes, e.g. re-interpretation of S3 given category, universe quirks.

**4. Open items** (if any): action items pending. For ORANGE PTFs, must explicitly state strategic decision required (see `alpha_sp100_2026.md` template).

### Save

Save as `<ptf_name>_<year>.md` (without `_auto` suffix). Replace the auto card. Auto card can be discarded or kept.

---

## Phase 5 — Deploy decision

Combines three inputs:
1. Quantitative verdict from R_Asset_v2 §8 (promoted satellite/core?)
2. Methodological recommendation from PTF card (GREEN/YELLOW/ORANGE)
3. Strategic context (commercial roadmap, client mandate)

Decision matrix:

| Quantitative | Recommendation | Decision |
|--------------|----------------|----------|
| Promoted | GREEN | Deploy candidate. Strategic call. |
| Promoted | YELLOW | Deploy with explicit caveats |
| Promoted | ORANGE | Do not deploy until ORANGE issue resolved |
| Not promoted | Any | Do not deploy. Investigate. |

If deploy:
- Add to production cron / scheduler
- Set up monitoring
- Document in commercial materials with caveats
- Schedule annual re-validation

If not deploy:
- Park as research-only
- Document reason in PTF card Open items

---

## Annual re-validation cycle

Each deployed PTF must be re-validated annually:

1. Re-run R_Asset_v2 with updated data
2. Compare new verdict with previous year's
3. If verdict changes from PASS to FAIL, or recommendation degrades, the PTF may need to be paused or discontinued
4. Update PTF card with new findings

A framework that always re-promotes every PTF is a framework that does not validate. Willingness to de-promote when data demands it distinguishes a serious quantitative framework from a marketing exercise.

---

## Common pitfalls

**1. Skipping Phase 4 because the auto card looks complete**
Auto card lacks category and bias profile. Always do Phase 4.

**2. Assuming B-discr is fine because the universe "feels solid"**
B-discr is YELLOW for a reason. Convert to B-rule by formalizing criteria when possible.

**3. Tempted to "fix" an ORANGE PTF with a quick patch**
Free data sources do not provide rigorous PIT membership. Do not retrofit yearly lists from biased universes — they propagate the bias.

**4. Deploying with significant capital because performance looks good**
Until 3-5 years of real track record, attribution between skill, beta, and luck is statistically impossible. Use homeopathic capital for PTFs in observation phase.

---

## Quick reference checklist

For each new PTF:

- [ ] Phase 1: concept clear, expected category identified
- [ ] Phase 2: portfolio dict added, composition rule documented
- [ ] Phase 3: R_Asset_v2 executed end-to-end without errors
- [ ] Phase 4: PTF card completed with category, bias profile, recommendation
- [ ] Phase 5: deploy decision recorded
- [ ] If deployed: production setup + monitoring + annual re-validation scheduled

If any checkbox is unchecked, the PTF is not in a deployable state.
