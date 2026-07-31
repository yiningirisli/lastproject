# MMF1927H — Option 4: Cryptocurrency Cross-Section

A one-week group project (Workshop in Mathematical Finance) that builds and
evaluates cross-sectional return models for cryptocurrencies using only Binance
public market data. The pipeline runs end to end: raw ingestion, cleaning and
universe construction, paper-mapped feature engineering, an eight-model suite with
leakage-safe evaluation, and a portfolio backtest over an in-sample and a forward
out-of-sample window.

**Anchor papers**

- **P1** — Liu, Tsyvinski & Wu (2022), *Common Risk Factors in Cryptocurrency*,
  Journal of Finance. Source of the meaningful cross-sectional characteristics
  (size/price, momentum, volatility, liquidity, market risk).
- **P2** — Cakici, Shahzad, Bedowska-Sojka & Zaremba (2024), *Machine Learning
  and the Cross-Section of Cryptocurrency Returns*, International Review of
  Financial Analysis. The direct machine-learning precedent and model suite.

---

## Repository layout

```
.
├── code/
│   ├── pull_binance_raw.py     # raw-layer ingestion from Binance
│   └── clean.py                # raw JSON -> analysis-ready panel
├── data/
│   ├── raw/
│   │   └── binance_klines_1d_2026-07-29/   # one JSON per symbol (immutable raw)
│   └── crypto_analysis_ready_1_.json       # analysis-ready panel (Git LFS, ~307 MB)
├── backtest/
│   ├── crypto_model.py         # features, 8 models, Rank-IC diagnostics
│   ├── model_2023_2026_portfolio_performance_summary_all8.csv
│   ├── model_2023_2026_return_curve_all8.png
│   ├── backtest_2026_onward_portfolio_performance_summary_all8.csv
│   ├── backtest_2026_onward_return_curve_all8.png
│   └── backtest_2026_onward_comb_return_curve_with_signals.png
├── README.md
└── LICENSE
```

The large analysis-ready panel is tracked with **Git LFS**. After cloning, run
`git lfs pull` to download it (otherwise `data/crypto_analysis_ready_1_.json` is a
small pointer file).

---

## Pipeline

### Raw ingestion (`code/pull_binance_raw.py`)

Pulls daily (1d) OHLCV klines for **every qualifying USDT spot pair** from the
Binance public API and stores each symbol's response verbatim as JSON under
`data/raw/binance_klines_1d_<date>/`. Two graded design decisions:

- **Pull wide, not just today's top 50.** Universe selection needs trailing
  volume for all *candidates* at each rebalance date, so ranking at pull time
  would bake in look-ahead. Pulling every pair keeps the raw layer immutable and
  pushes the universe rule downstream into the cleaning stage, where it can
  change without re-pulling.
- **Store all 12 kline fields verbatim.** No parsing or transformation at the raw
  layer, so the raw data is always reproducible and auditable.

```bash
python code/pull_binance_raw.py --start 2018-01-01 --out data/raw
```

### Cleaning and universe (`code/clean.py`)

Converts the immutable raw JSON into a single tidy, analysis-ready panel with one
row per (symbol, date). It computes leakage-safe daily returns (no imputation
across listing gaps), winsorizes returns at the 1st/99th percentiles, builds the
point-in-time investable universe, and records data-quality diagnostics. The
universe is the **top 50** names by trailing 30-day dollar volume, requiring at
least 30 days of history.

```bash
python code/clean.py data/raw/binance_klines_1d_2026-07-29 \
    --out data/clean --top-n 50 --volume-window 30 --min-history 30
```

The committed panel `data/crypto_analysis_ready_1_.json` has one row per
(symbol, date) with these fields:

| Column | Meaning |
|---|---|
| `date`, `symbol` | daily observation key |
| `open`, `high`, `low`, `close` | daily OHLC |
| `volume`, `quote_asset_volume` | base and quote (dollar) volume |
| `n_trades`, `taker_buy_base_volume`, `taker_buy_quote_volume` | trade counts and taker flow |
| `return_1d`, `log_return_1d` | leakage-safe daily returns (NaN on first day / after gaps) |
| `history_days`, `trailing_quote_volume`, `volume_rank`, `eligible_history` | universe-construction fields |
| `in_universe` | point-in-time top-50 membership flag |
| `target_return_1d` | one-day-ahead return (reference target) |

### Features and models (`backtest/crypto_model.py`)

Reads the analysis-ready panel and produces the modeling deliverables. It
re-derives its own leakage-safe fields rather than trusting upstream flags:
returns are recomputed within uninterrupted listing segments, the target is the
one-day-ahead return `close.shift(-1)/close - 1` (winsorized to `[-0.99, 3.0]`
per P2), and the universe is rebuilt from **lagged** 30-day dollar volume.

```bash
python backtest/crypto_model.py \
    --input data/crypto_analysis_ready_1_.json \
    --output-dir backtest/model_outputs
```

**Safety exclusions.** Because the raw pull sweeps up every USDT pair, non-crypto
and duplicate instruments are removed and logged before modeling:

| Category | Why excluded |
|---|---|
| Stablecoins / pegged tokens | return pinned near zero by design |
| Metal-backed (tokenized gold) | track gold, not crypto |
| Wrapped / staked tokens | duplicate exposure to an underlying coin |
| Derivative-collateral tokens | not a clean spot cross-section member |
| Tokenized equities / index ETFs (e.g. tokenized Tesla, S&P 500) | driven by the equity market, not crypto |
| Multiplier contracts (`1000...`, `1M...`) | re-denominated duplicates of the underlying |

**Features (21, paper-mapped).** Every feature is causal (known at date *t*),
cross-sectionally percentile-ranked per date, and documented with its source
paper, formula, and rationale in the generated `day3_feature_dictionary.csv`.

| Group | Features |
|---|---|
| Size / price | `log_prc`, `age` |
| Momentum / reversal | `mom_7`, `mom_21`, `mom_7_28`, `r2_1`, `r31_2`, `r30_14` |
| Price position | `close_90dh` |
| Volume / liquidity | `log_vol`, `log_prcvol`, `std_prcvol`, `amihud_illiq`, `volsh_30` |
| Volatility | `retvol_7`, `rvol_30`, `var_90` |
| Market risk (vs crypto market) | `beta_30`, `ivol_30`, `capm_alpha_30` |
| Distribution shape | `skew_90` |

Market-capitalization, turnover, and on-chain features are intentionally omitted:
Binance klines carry no circulating-supply field, and the top-50 liquid universe
suppresses the size effect (P2). This is a documented design choice, not a gap.

**Target and validation.** The target is the one-day-ahead return, cross-
sectionally standardized per date (evaluation is against the raw realized return).
Evaluation uses an expanding-window, date-based walk-forward with an embargo
between train and test to prevent overlap-driven leakage.

**Models.** Seven base models are fitted inside each fold (tree models on raw
features, the rest on standardized features), plus an eighth combined output:

| Model | Role |
|---|---|
| OLS | linear baseline (P2's best level-prediction model) |
| PLS | dimension-reduction linear model |
| LASSO | sparse penalized regression |
| Elastic Net | penalized regression (course requirement) |
| Random Forest | bagged trees (P2's highest forecast-correlation model) |
| LightGBM (HGB) | gradient-boosted trees (course-required boosted model) |
| FFNN | small feed-forward neural network |
| **COMB** | equal-weight average of the seven base forecasts (P2's combination) |

**Headline three:** OLS, Random Forest, and COMB, chosen to represent three
distinct paper outcomes. COMB is an ensemble, not a separately fitted algorithm;
fitting it requires fitting all seven base models, which is how the required
Elastic Net and LightGBM models stay in the pipeline without becoming the headline
story. `crypto_model.py` deliberately stops at ranking diagnostics (Rank IC and hit
rate); portfolio construction and costs are handled in the backtest stage.

### Backtest (`backtest/`)

The model forecasts are turned into daily portfolios and evaluated over two
windows, with results committed to `backtest/`:

- **In-sample era, 2023–2026** — `model_2023_2026_*` (658 trading days).
- **Forward out-of-sample, 2026 onward** — `backtest_2026_onward_*` (88 days),
  a genuine hold-out that the models never saw in training.

Each `*_portfolio_performance_summary_all8.csv` reports, for all eight models: IC,
number of days, mean daily return, annualized return and volatility, Sharpe ratio,
max drawdown, cumulative return, and win rate. The `*_return_curve_all8.png` files
show cumulative equity curves for all models, and
`backtest_2026_onward_comb_return_curve_with_signals.png` overlays the COMB
strategy's buy/sell signals on its forward-period equity curve.

---

## Key results

### In-sample era (2023–2026, 658 days)

| Model | IC | Ann. return | Sharpe | Max DD | Cumulative | Win rate |
|---|---|---|---|---|---|---|
| **OLS** | 0.084 | 211% | **2.12** | −31% | 4.96× | 58% |
| Elastic Net | 0.085 | 211% | 2.11 | −31% | 4.95× | 58% |
| LASSO | 0.085 | 203% | 2.06 | −34% | 4.68× | 58% |
| **COMB** | 0.091 | 196% | 1.99 | −41% | 4.40× | 59% |
| PLS | 0.103 | 205% | 1.93 | −42% | 4.49× | 58% |
| **Random Forest** | 0.097 | 125% | 1.45 | −37% | 2.25× | 57% |
| LightGBM (HGB) | 0.083 | 119% | 1.39 | −50% | 2.07× | 57% |
| FFNN | 0.043 | 60% | 0.97 | −43% | 0.89× | 55% |

### Forward out-of-sample (2026 onward, 88 days)

| Model | IC | Cumulative | Sharpe | Max DD | Win rate |
|---|---|---|---|---|---|
| LASSO | 0.105 | +80% | **3.24** | −28% | 67% |
| Elastic Net | 0.105 | +80% | 3.23 | −27% | 66% |
| **OLS** | 0.106 | +77% | 3.17 | −27% | 67% |
| PLS | 0.111 | +55% | 2.57 | −30% | 64% |
| LightGBM (HGB) | 0.095 | +48% | 2.45 | −25% | 64% |
| **Random Forest** | 0.111 | +37% | 2.03 | −26% | 60% |
| **COMB** | 0.085 | +38% | 2.02 | −27% | 55% |
| FFNN | −0.011 | −17% | −0.91 | −36% | 51% |

*(Annualized returns for the 88-day forward window are extrapolations of a short,
strong period and should be read alongside the cumulative column, not on their
own.)*

**What the results say**

- **Linear models and COMB deliver the best risk-adjusted performance** in both
  windows (Sharpe ~2.0–2.1 in-sample, ~3.2 forward for OLS/LASSO/Elastic Net).
  The tree models have competitive or higher IC but lower Sharpe and cumulative
  return under this portfolio construction.
- **The best model by IC is not the best by return.** PLS and Random Forest post
  the highest IC yet trail OLS/LASSO/Elastic Net on Sharpe, because IC scores the
  whole ranking while portfolio return depends on the traded tails.
- **FFNN fails to generalize.** It is the weakest model in-sample and goes
  outright negative out-of-sample (IC −0.01, −17% cumulative) — a clean example
  of a flexible model overfitting a low-signal, short-history cross-section.
- **The signal persists out-of-sample.** The 2026-onward hold-out confirms the
  linear-model edge survives on data the models never saw, which is the main
  robustness result.

---

## Interpretation and caveats

- **Short forward window.** The 2026-onward test is only 88 days; treat its
  annualized figures as indicative, and lean on cumulative return, Sharpe, and IC.
- **Survivorship.** The candidate history skews toward symbols that survive to the
  final date, so the universe is not fully survivorship-free; this inflates
  predictability. It is logged in the data audit.
- **Costs and turnover.** The cross-sectional signal is high-turnover (short-term
  reversal is a large component), so realized net performance is sensitive to
  transaction costs and to the exact rebalancing rule.
- **Short-borrow reality.** A long-short construction assumes the bottom names can
  be borrowed and shorted, which is often impractical in crypto; a long-versus-
  market variant is more realistic.

---

## How to run

```bash
git clone https://github.com/yiningirisli/lastproject.git
cd lastproject
git lfs pull                      # download the analysis-ready panel

# raw ingestion
python code/pull_binance_raw.py --start 2018-01-01 --out data/raw

# clean into an analysis-ready panel
python code/clean.py data/raw/binance_klines_1d_2026-07-29 \
    --out data/clean --top-n 50 --volume-window 30 --min-history 30

# features, models, Rank-IC diagnostics
python backtest/crypto_model.py \
    --input data/crypto_analysis_ready_1_.json \
    --output-dir backtest/model_outputs
```

---

## Requirements

Python 3.11+ with: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `statsmodels`,
`matplotlib`, and (for reading the panel) `pyarrow`. Git LFS is required to fetch
the analysis-ready data file.
