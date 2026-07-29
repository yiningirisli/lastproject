# Day 2 — Data Engineering and Cleaning

`day2_clean.py` transforms the immutable Binance JSON files produced on Day 1
into one validated, analysis-ready Parquet panel. It never edits the raw layer.

## Outputs

```text
data/clean/
├── crypto_analysis_ready.parquet
├── day2_lineage_manifest.json
└── DAY2_DATA_QUALITY_MEMO.md
```

- The Parquet file preserves exact dtypes and uses Zstandard compression.
- The lineage manifest records parameters, SHA-256 hashes for every raw input,
  the cleaning-script hash, output hash, diagnostics, and the Git commit when
  available.
- The memo documents missingness, outlier treatment, universe construction,
  exclusions, and the unresolved survivorship limitation.

## Run

```bash
python day2_clean.py data/raw/binance_klines_1d_YYYY-MM-DD
```

Explicit parameters:

```bash
python day2_clean.py RAW_DIRECTORY \
  --out data/clean \
  --top-n 50 \
  --volume-window 30 \
  --min-history 30 \
  --winsor-lower 0.01 \
  --winsor-upper 0.99
```

## Cleaning decisions

### Validation

Every symbol file must have the Day 1 provenance wrapper and 12 Binance kline
fields. The pipeline converts numeric fields explicitly, sorts dates, removes
duplicate symbol-dates, and rejects:

- missing or nonnumeric OHLCV fields;
- non-positive prices;
- negative volume;
- high below open/close or low above open/close;
- high below low.

### Missingness

Returns are not imputed. A missing return at the start of a listing or after an
unavailable prior observation is structurally related to listing and trading
history and is not safely assumed MCAR. Such rows remain `NaN` and are marked
with `return_missing`.

No backward interpolation is used. Future observations never fill earlier rows.

### Point-in-time universe

The default modeling universe contains the top 50 eligible symbols per date,
ranked by mean USDT quote volume over the preceding 30 observed days.

The volume series is shifted one day before rolling and ranking. Therefore,
membership on date t uses only information available through t-1. A symbol must
have more than 30 observed days before becoming eligible; pre-listing history is
never backfilled.

### Outliers

Raw valid returns are retained. A separate `return_1d_winsor` column applies
cross-sectional p1/p99 winsorization independently on every date. A
MAD-standardized `return_1d_robust_z` column is also supplied. Neither replaces
the original return.

### Target

`target_return_1d` is the next observed daily return for the same symbol. It is
stored separately so Day 3 can construct features without accidentally using
the target as an input.

## Panel columns

- identifiers: `date`, `symbol`;
- raw daily values: OHLC, volume, quote volume, trade count and taker-buy
  volumes;
- returns: raw arithmetic return and log return;
- missingness and robust versions: missing flag, winsorized return, robust z;
- universe fields: history length, lagged trailing quote volume, volume rank,
  eligibility, membership;
- future target: next daily return.

## What is and is not resolved

The panel is point-in-time correct with respect to its rolling calculations:
no future price or volume enters earlier dates. It also permits an unbalanced
panel and never backfills an asset before its first observed Binance candle.

However, the Day 1 `exchangeInfo` snapshot cannot recover symbols delisted
before the pull date. Consequently, residual survivorship bias remains and
likely biases historical results upward. The code records this in the lineage
manifest and memo rather than claiming the public source can eliminate it.

Crypto spot does not have equity splits, dividends, GICS classifications or
futures roll adjustments, so those lecture checks are not applicable here.

## Tests

```bash
python -m unittest discover -s tests -v
```

Day 2 tests cover schema/type conversion, duplicate removal, invalid OHLC
rejection, causal lagged universe construction, and non-imputation of returns.
