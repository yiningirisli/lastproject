# lastproject

# Crypto Cross-Section — Day 1 Raw Data Layer

This repository implements the Day 1 sourcing stage for MMF1927H Option 4:
Crypto Cross-Section. It downloads raw daily Binance spot klines for the
candidate USDT universe and preserves the vendor response without cleaning or
feature engineering.

## Research scope

- **Source:** Binance public REST API (no account or API key required)
- **Endpoints:** `/api/v3/exchangeInfo` and `/api/v3/klines`
- **Frequency:** daily (`1d`)
- **Default sample:** 2018-01-01 through the current UTC date
- **Raw universe:** every currently trading USDT spot pair that passes the
  documented exclusions below
- **Downstream universe:** rank candidates using trailing, historically
  observable Binance quote volume at each rebalance date. Ranking is not
  performed during ingestion, because doing so using today's ranking would
  introduce look-ahead bias.

The 2018 start captures the 2018 crypto drawdown, the 2020–21 bull market, the
2022 rate-driven selloff, and subsequent regimes. Many assets listed later and
therefore have shorter histories.

## Candidate-universe rule

The raw layer excludes:

- stablecoin and fiat-like base assets;
- leveraged tokens (`UP`, `DOWN`, `3L`, `3S`, `5L`, `5S`);
- wrapped or duplicate exposures such as WBTC and WETH;
- symbols that are not currently trading or do not permit spot trading.

Pulling the full candidate set instead of today's top-N preserves the data
needed to form a trailing-volume top-N universe separately at every historical
rebalance date.

## Known limitation

`exchangeInfo` exposes only symbols known to Binance at pull time. Assets
delisted before the pull are absent, so the historical candidate set remains
survivorship-biased. The likely direction is upward: failed or delisted assets
are disproportionately omitted, which can overstate historical portfolio
performance. The manifest records this limitation explicitly. A truly
point-in-time listing history would be needed to eliminate it.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

All direct dependencies are pinned exactly for reproducibility.

## Usage

Small smoke pull:

```bash
python pull_binance_raw.py --start 2024-01-01 --limit-symbols 5
```

Full Day 1 pull:

```bash
python pull_binance_raw.py --start 2018-01-01
```

Specify an exclusive end date:

```bash
python pull_binance_raw.py --start 2024-01-01 --end 2024-02-01
```

Existing symbol files are skipped. Use `--force` only when an intentional
replacement is required.

## Raw output

Each run writes to:

```text
data/raw/binance_klines_1d_YYYY-MM-DD/
├── exchange_info.json
├── universe_candidates.json
├── provenance_log.jsonl
├── run_summary.json
└── SYMBOL.json
```

`exchange_info.json` is the untouched exchange metadata response.
Each symbol file contains the untouched 12-field kline arrays plus a provenance
wrapper. `run_summary.json` records requested dates, counts, failures, and
completion status.

The raw directory is intentionally ignored by Git because a full pull is large.
Commit the lightweight validation artifacts under `validation/`, while retaining
the immutable raw files in shared project storage.

## Validation

Run the offline tests:

```bash
python -m unittest discover -s tests -v
```

After a live pull, validate its structure:

```bash
python validate_raw.py data/raw/binance_klines_1d_YYYY-MM-DD
```

The validator exits nonzero for malformed symbol payloads, duplicate or
unordered timestamps, candles outside the requested half-open interval, or a
summary that reports failures.

## Day 1 / Day 2 boundary

Day 1 stores vendor responses and provenance only. Type conversion, missing-data
handling, outlier checks, panel alignment, historical top-N construction,
returns, features, and modeling belong downstream and must not mutate this raw
layer.
