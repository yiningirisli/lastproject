# Day 2 Data-Quality Memo

The cleaned panel contains 538,888 rows and
458 symbols from 2018-01-01 through
2026-07-28. It contains
0 duplicate symbol-date rows after
validation.

## Missingness

Daily returns at listing boundaries or after unavailable prior observations are
left missing; 458 rows carry `return_missing=True`.
No backward interpolation or future observation is used. Crypto missingness is
often listing-, outage-, or liquidity-related and therefore not assumed MCAR.

## Universe

The per-date universe contains the top 50 eligible symbols ranked by
the trailing 30-day mean USDT quote volume. Volume is shifted
by 1 day before ranking, so date-t
membership uses only information known by t-1. A symbol needs more than
30 observed days before eligibility. The resulting panel has
134,559 universe rows over
3,101 dates.

## Outliers

Invalid OHLC relationships, non-positive prices, negative volume, malformed
rows, and duplicate symbol-dates are treated as data errors. Valid return tails
are retained in `return_1d`; a separate cross-sectional winsorized series caps
returns at p1/p99 per date.
A MAD-based robust z-score is also supplied without overwriting raw returns.

## Point-in-time and residual bias

All rolling volume inputs are lagged and targets are stored separately. However,
the Day 1 Binance `exchangeInfo` snapshot cannot recover coins delisted before
the pull date. Residual survivorship bias therefore remains and likely biases
historical performance upward. The current public raw source is insufficient to
claim that this bias has been eliminated.

## Validation notes

- No malformed, duplicate, or invalid OHLCV rows were detected.
