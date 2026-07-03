# Sentiment Factors in European Equities (STOXX)

Two companion cross-sectional asset-pricing studies built on one shared
pipeline (incremental Refinitiv pulls → DuckDB → FF-25 double sorts → FF6
alphas, Fama–MacBeth, Chow tests). Same framework, different sort variable —
kept in one repo so the shared design is visible. Each study has its own
full README:

- **[`retail_sentiment/`](retail_sentiment/)** — trailing PE × retail
  sentiment (Signed Abnormal Turnover), with a predetermined 2020 regime
  break → [retail_sentiment/README.md](retail_sentiment/README.md)
- **[`esg_sentiment/`](esg_sentiment/)** — Refinitiv ESG composite ×
  sentiment, with dual regime breaks (2018 EU Action Plan, 2020 retail
  surge) → [esg_sentiment/README.md](esg_sentiment/README.md)
