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
│   └── build_quintiles.py      # Independent PE and SAT quintile assignment
├── portfolios/
│   └── construct_ff25.py       # FF-25 double-sort, spreads, long-short portfolio
├── estimation/
│   ├── time_series_alpha.py    # FF6 alpha for all 25 portfolios (Newey-West SEs)
│   └── fama_macbeth.py         # Fama-MacBeth + Chow test for regime shift
├── rsf_spec/
│   └── fpe_function.py         # Weibull f(PE) NLS fitting (parametric complement)
├── tests/
│   ├── test_build_sat.py       # SAT unit tests (AAA pattern)
│   └── test_construct_ff25.py  # Portfolio construction unit tests
├── main.py                     # Five-step pipeline orchestrator
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
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Design Decisions

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
