# RSF Research — Retail Sentiment Factor (STOXX Universe)

Companion code for *Retail Sentiment, Valuation Regimes, and the Cross-Section of European Equity Returns*.

## Repository Layout

```
rsf_research/
├── config/
│   └── constants.py            # ALL named constants — no magic numbers elsewhere
├── data/
│   ├── db.py                   # DuckDB wrapper (adapted from master_db.py)
│   ├── fetch_refinitiv.py      # Incremental Refinitiv pull (adapted from pull_master_data.py)
│   └── fetch_french_factors.py # Ken French European FF5+Mom download
├── features/
│   ├── build_sat.py            # Signed Abnormal Turnover construction
│   ├── build_pe.py             # Trailing PE cleaning and winsorisation
│   ├── build_quintiles.py      # Independent PE and SAT quintile assignment
│   └── forward_returns.py      # Single source of the sort-at-t / hold-at-t+1 rule
├── portfolios/
│   └── construct_ff25.py       # FF-25 double-sort, spreads, long-short portfolio
├── factors/
│   └── build_rsf_factor.py     # RSF as an FF-style 2×3 size/SAT factor (RHS use)
├── estimation/
│   ├── time_series_alpha.py    # Factor-model alpha for all 25 portfolios (NW SEs)
│   ├── fama_macbeth.py         # Fama-MacBeth + Chow test for regime shift
│   └── factor_spanning.py      # RSF spanning tests vs FF5+UMD; FF6 vs FF6+RSF alphas
├── rsf_spec/
│   └── fpe_function.py         # Weibull f(PE) NLS fitting (parametric complement)
├── tests/
│   ├── test_build_sat.py       # SAT unit tests (AAA pattern)
│   ├── test_construct_ff25.py  # Portfolio construction unit tests
│   └── test_no_lookahead.py    # Look-ahead regression tests (zero-future-signal panels)
├── paper/
│   ├── rsf_sat_pe_corrected.tex # Corrected main paper (null predictive result + look-ahead diagnostic)
│   ├── rsf_factor_spanning.tex  # Companion paper: RSF as a complementary RHS factor
│   ├── make_tables.py           # Regenerates all LaTeX tables from DuckDB results
│   └── tables/                  # Generated table fragments (\input by the papers)
├── main.py                     # Pipeline orchestrator
└── reset_db.py                 # Drop tables for a fresh pull
```

## Setup

```bash
pip install duckdb pandas numpy statsmodels scipy requests refinitiv-data
```

Requires Python 3.12 and a valid Refinitiv Eikon session for the data pull step.

## Running the Pipeline

```bash
# Full pipeline
python main.py

# Reset data tables and re-pull
python reset_db.py
python main.py

# Reset everything
python reset_db.py --all
python main.py
```

Toggle individual steps in `main.py`:

```python
PULL_DATA        = True   # Step 1: Refinitiv incremental pull
FETCH_FACTORS    = True   # Step 2: Ken French factors
BUILD_FEATURES   = True   # Step 3: SAT, PE, quintiles
BUILD_PORTFOLIOS = True   # Step 4: FF-25 portfolios
RUN_ESTIMATION   = True   # Step 5: Alphas, FMB, Chow test
RUN_SPANNING     = True   # Step 8: RSF factor + FF spanning tests
BUILD_DAILY      = True   # Step 7: Daily portfolios and daily FMB
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Design Decisions

**Strict sort-at-t / hold-at-t+1 convention.** SAT is signed by month-t return
direction, so pairing a month-t sort with month-t returns is mechanically
profitable — pure look-ahead. All monthly portfolio returns are therefore
computed over month t+1 with month-t weights, via the single shared helper in
`features/forward_returns.py` (which also masks forward returns across listing
gaps so t+1 is always the immediately following calendar month). Portfolio
return rows are labelled with the *holding* month so they align with factor
returns. `tests/test_no_lookahead.py` locks this in: panels where the sort
variable drives only the formation-month return must produce exactly zero
portfolio returns and a zero RSF factor.

**RSF as a complementary factor, not a prediction signal.** Beyond the FF-25
sort study, RSF is packaged as a Fama-French-style tradable factor
(`factors/build_rsf_factor.py`): an independent 2×3 sort on size (median
split) and SAT (30/70 breakpoints), RSF = ½(Small/High + Big/High) −
½(Small/Low + Big/Low), value-weighted, formation at t and held over t+1. The
size legs neutralise the small-cap tilt of retail activity so RSF is not a
repackaged SMB. `estimation/factor_spanning.py` then treats RSF as a
right-hand-side regressor: a spanning regression of RSF on FF5+UMD (is RSF
redundant?), correlations with each FF factor, and a comparison of the 25
portfolio alphas under FF6 vs FF6+RSF (does RSF strip sentiment comovement
out of the cross-section?).

**SAT as retail sentiment proxy.** The BJZZ algorithm (standard US approach) exploits
subpenny price improvements specific to Regulation NMS. It has no European equivalent.
SAT — abnormal turnover signed by return direction — is grounded in Baker & Stein (2004)
and requires only daily price, volume, and shares outstanding from Refinitiv.

**Incremental data pull.** `fetch_refinitiv.py` resumes from the last committed date on
rerun. On failure it stops immediately rather than skipping, ensuring `db.lastDate()`
always reflects a contiguous block. This mirrors the gap-safe design in `pull_master_data.py`.

**Regime shift test.** The pre/post-2020 split at December 2019 is predetermined by
theory, not chosen to maximise significance. The Chow test in `fama_macbeth.py`
directly tests whether `delta_Cross` increased post-2020.

**Named constants only.** All numeric parameters live in `config/constants.py`.
No magic numbers appear in analysis code.
