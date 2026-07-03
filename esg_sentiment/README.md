# ESF Research — ESG Sentiment Factor (STOXX Universe)

Does retail sentiment amplify or distort the pricing of ESG in European
equities? This pipeline double-sorts the STOXX universe on **Refinitiv ESG
composite scores** and **Signed Abnormal Turnover (SAT)** — a retail
sentiment proxy — and tests whether the ESG–return relationship shifted
around two predetermined regime breaks:

- **2018 — EU Sustainable Finance Action Plan** (institutional ESG mainstreaming)
- **2020 — retail participation surge + SFDR implementation**

It is the ESG adaptation of the sibling
[`retail_sentiment`](../retail_sentiment) pipeline: the
valuation sort (trailing PE) is replaced by the ESG score, and the single
2020 break becomes a dual-break Chow test. Both breaks are theory-driven and
were fixed before any subsample results were examined.

## Repository Layout

```
esg_sentiment/
├── config/
│   └── constants.py            # ALL named constants — no magic numbers elsewhere
├── data/
│   ├── db.py                   # DuckDB wrapper
│   ├── fetch_refinitiv.py      # Incremental Refinitiv pull (returns, volume, market cap, TR.TRESGScore)
│   └── fetch_french_factors.py # Ken French European FF5+Mom download
├── features/
│   ├── build_sat.py            # Signed Abnormal Turnover construction
│   ├── build_esg.py            # ESG composite cleaning, winsorisation, financials exclusion
│   ├── build_quintiles.py      # Independent ESG and SAT quintile assignment
│   └── build_daily_panel.py    # Daily panel + AR(2)-residual SAT variants
├── portfolios/
│   ├── construct_ff25.py       # ESG × SAT 5×5 double-sort, spreads, ESG-SAT long-short
│   └── construct_daily_ff25.py # Daily portfolio variants (VW, 3 SAT definitions)
├── estimation/
│   ├── time_series_alpha.py    # FF6 alpha for all 25 portfolios (Newey-West SEs)
│   ├── fama_macbeth.py         # Fama-MacBeth + dual-break Chow test (2018, 2020)
│   ├── parametric_sat.py       # Parametric SAT amplification by ESG regime
│   └── daily_fmb.py            # Daily Fama-MacBeth, full sample + regime splits
├── rsf_spec/
│   └── fpe_function.py         # Weibull f(ESG) NLS fit (bounds rescaled for 0–100 scores)
├── tests/
│   ├── test_build_sat.py       # SAT unit tests (AAA pattern)
│   └── test_construct_ff25.py  # Portfolio construction unit tests
├── plots/                      # Generated figures (heatmaps, spread ladders, rolling stats)
├── main.py                     # Pipeline orchestrator (togglable steps)
└── reset_db.py                 # Drop tables for a fresh pull
```

## Setup

```bash
pip install duckdb pandas numpy statsmodels scipy requests refinitiv-data
```

Requires Python 3.12 and a valid Refinitiv Eikon session for the data pull step.

## Running the Pipeline

```bash
python main.py
```

Toggle individual steps in `main.py`:

```python
IMPORT_STATIC    = False  # Step 0: GICS sector snapshot
PULL_DATA        = False  # Step 1: Refinitiv incremental pull
FETCH_FACTORS    = False  # Step 2: Ken French factors
BUILD_FEATURES   = True   # Step 3: SAT, ESG, quintiles
BUILD_PORTFOLIOS = True   # Step 4: ESG × SAT FF-25 portfolios (+ EW and small/mid-cap robustness)
RUN_ESTIMATION   = True   # Step 5: FF6 alphas, FMB, dual-break Chow, parametric amplification
GENERATE_PLOTS   = True   # Step 6: full figure suite → plots/
BUILD_DAILY      = False  # Step 7: daily portfolios + daily FMB (compute-intensive, ~15–30 min)
```

Sample: STOXX constituents, June 2009 – December 2025, EUR total returns.
Results persist to `data/esf.duckdb`; reruns are incremental and idempotent.

## Running Tests

```bash
pytest tests/ -v
```

## Key Design Decisions

**ESG score handling.** The sort variable is `TR.TRESGScore` (0–100 composite
of E/S/G pillars). Scores update at most quarterly — often annually — so the
daily panel carries scores forward up to 252 trading days (gaps between
genuine updates are carry-forward values, not missing data). At the monthly
level, however, a missing score means genuine **non-disclosure** and is *not*
filled: it propagates as NaN so undisclosed firms simply drop out of the
quintile sort. Financials (GICS sector 40) are excluded; scores are
winsorised cross-sectionally within each month; low-coverage months
(pre-2013) are flagged rather than dropped.

**SAT as retail sentiment proxy.** The BJZZ algorithm (standard US approach)
exploits subpenny price improvements specific to Regulation NMS and has no
European equivalent. SAT — abnormal turnover signed by return direction — is
grounded in Baker & Stein (2004) and needs only daily price, volume, and
shares outstanding from Refinitiv.

**Predetermined dual regime breaks.** The Chow test examines both the
Dec-2017 and Dec-2019 breaks jointly, testing whether the sentiment–ESG
interaction (`delta_Cross`) strengthened first with institutional ESG
mainstreaming and again with the retail surge.

**Robustness by construction.** Every headline result is re-run
equal-weighted and on the small/mid-cap subsample; the daily module rebuilds
the whole analysis at daily frequency with three SAT variants (raw, 5-day,
10-day rolling) to address bid-ask-bounce concerns.

**Named constants only.** All numeric parameters live in
`config/constants.py`. No magic numbers appear in analysis code.
