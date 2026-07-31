# MMF1927H — Option 4: Cryptocurrency Cross-Section

A one-week group project (Workshop in Mathematical Finance) that builds and
evaluates cross-sectional return models for cryptocurrencies using only Binance
public data. The pipeline runs end to end from raw data ingestion through
feature engineering, model fitting, out-of-sample evaluation, and a long-short
relative-return backtest.

**Anchor papers**

- **P1** — Liu, Tsyvinski & Wu (2022), *Common Risk Factors in Cryptocurrency*,
  Journal of Finance. Source of the meaningful cross-sectional characteristics
  (size/price, momentum, volatility, liquidity, market risk).
- **P2** — Cakici, Shahzad, Bedowska-Sojka & Zaremba (2024), *Machine Learning
  and the Cross-Section of Cryptocurrency Returns*, International Review of
  Financial Analysis. The direct machine-learning precedent and model suite.

---

## 1. Pipeline overview

The project is organized as an immutable, layered pipeline. Each stage reads the
previous stage's output and never edits it, so the raw and cleaned layers are
reproducible and auditable.

```
Day 1  Raw ingestion        Binance public API  ->  raw daily klines (immutable)
Day 2  Cleaning + universe  raw klines          ->  analysis-ready panel (parquet)
Day 3  Features + models    analysis-ready panel->  features, forecasts, metrics,
                                                     relative-return backtest
Day 4  Portfolios (planned) forecasts           ->  cost-aware portfolio backtest
```

### Day 1 — Data sourcing

Pulls daily OHLCV klines for the USDT spot pairs from the Binance public API and
stores them in an immutable raw layer. No transformations are applied at this
stage beyond writing the exchange response to disk, so the raw data can always be
re-derived and audited.

### Day 2 — Cleaning and universe construction

Reads the raw klines and produces a single tidy, analysis-ready panel with one
row per (symbol, date). It computes leakage-safe daily returns (no imputation
across listing gaps), attaches a point-in-time investable-universe flag, and
records data-quality diagnostics. The investable universe is the **top 50** names
by trailing 30-day median quote (dollar) volume, ranked at month-end and held for
the following month, with a 30-name floor in the early, sparse part of the sample.

The Day-2 deliverable is the analysis-ready panel used by Day 3 (for example
`crypto_analysis_ready.parquet`). Its schema:

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

### Day 3 — Features, models, evaluation, relative return

The core modeling stage (`crypto_day3_model.py`), described below.

---

## 2. Day 3 in detail

### 2.1 Panel safeguards and universe rebuild

Day 3 does not blindly trust upstream flags. It re-derives its own leakage-safe
fields from the panel:

- **Returns** are recomputed as `close.pct_change()` within uninterrupted listing
  segments (segmented so returns never bridge a gap in a coin's history).
- **Target** is the one-day-ahead return, `close.shift(-1) / close - 1`, winsorized
  to `[-0.99, 3.0]` following P2's error-control bounds.
- **Universe** is rebuilt from **lagged** 30-day dollar volume (top 50), so
  selection never uses same-day information.

### 2.2 Safety exclusions

Because the raw pull sweeps up every USDT pair, several non-crypto or duplicate
instruments are removed before modeling. Each exclusion is explicit and logged in
`day3_data_audit.csv`:

| Category | Why excluded |
|---|---|
| Stablecoins / pegged tokens | return pinned near zero by design |
| Metal-backed (e.g. tokenized gold) | track gold, not crypto |
| Wrapped / staked tokens | duplicate exposure to an underlying coin |
| Derivative-collateral tokens | not a clean spot cross-section member |
| Tokenized equities / index ETFs (e.g. tokenized Tesla, S&P 500) | driven by the equity market, not crypto |
| Multiplier contracts (`1000...`, `1M...`) | re-denominated duplicates of the underlying |

### 2.3 Features (21, paper-mapped)

Every feature is causal (known at date *t*), cross-sectionally percentile-ranked
per date, and documented with its source paper, formula, and economic rationale
in `day3_feature_dictionary.csv`.

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
suppresses the size effect (per P2). This is a documented design choice, not a gap.

### 2.4 Target and validation

- **Target:** one-day-ahead return, cross-sectionally standardized per date
  (evaluation is against the raw realized return).
- **Cross-validation:** expanding-window walk-forward with 3 outer folds, an
  initial train fraction of 0.40, and a 5-day embargo between train and test to
  prevent overlap-driven leakage.
- **Minimums:** 90 days of history before a name is modeled; at least 10 names on
  a date for that date to contribute to a metric.

### 2.5 Models

Seven base models are fitted inside each fold (tree models on raw features, the
rest on standardized features), and an eighth output combines them:

| Model | Role |
|---|---|
| OLS | linear baseline (P2's best level-prediction model) |
| PLS | dimension-reduction linear model |
| LASSO | sparse penalized regression |
| Elastic Net | penalized regression (course requirement) |
| Random Forest | bagged trees (P2's highest forecast-correlation model) |
| LightGBM | gradient-boosted trees (course-required boosted model) |
| FFNN | small feed-forward neural network |
| **COMB** | equal-weight average of the seven base forecasts (P2's combination) |

**Headline three:** OLS, Random Forest, and COMB. These are chosen because they
represent three distinct paper outcomes, not because one metric declares them
jointly best. COMB is an ensemble, not a separately fitted algorithm; fitting it
requires fitting all seven base models, which is how the required Elastic Net and
LightGBM models remain in the pipeline without becoming the headline story.

### 2.6 Evaluation metrics

- **Rank IC** — daily cross-sectional Spearman correlation between predicted and
  realized returns, averaged over all out-of-sample days.
- **ICIR** — mean Rank IC divided by its daily standard deviation.
- **HAC t-stat / p-value** — Newey-West-corrected significance of the mean IC.
- **positive-IC rate** — fraction of days with IC > 0.
- **hit rate** — fraction of individual predictions with the correct sign.
- **permutation-null IC** — IC after shuffling predictions (a no-skill baseline;
  computed for the headline models).

### 2.7 Long-short relative return (buy/sell signals)

The models output a **ranking score**, which is turned into a tradeable signal:
each day, **buy** the top 20% of the score and **sell** the bottom 20%. The
long-minus-short spread is a dollar-neutral, market-neutral return (relative to
the crypto cross-section). It is reported **gross and net** of a per-side
transaction cost (default 10 bps) applied to daily turnover, since the signal is
high-turnover.

---

## 3. Key results

Out-of-sample period: 2021-07-30 to 2026-07-27 (1,824 trading days).

### 3.1 Predictive accuracy (Rank IC)

| Model | Mean Rank IC | ICIR | Positive-IC rate | Hit rate |
|---|---|---|---|---|
| PLS | 0.094 | 0.39 | 0.66 | 0.52 |
| LASSO | 0.092 | 0.39 | 0.67 | 0.52 |
| Elastic Net | 0.092 | 0.39 | 0.67 | 0.52 |
| **OLS** | 0.092 | 0.39 | 0.67 | 0.52 |
| **Random Forest** | 0.091 | 0.38 | 0.66 | 0.50 |
| **COMB** | 0.087 | 0.36 | 0.65 | 0.52 |
| LightGBM | 0.085 | 0.37 | 0.65 | 0.50 |
| FFNN | 0.028 | 0.15 | 0.57 | 0.50 |

The linear models cluster tightly and are statistically indistinguishable; the
tree models are marginally lower on IC; the small neural net trails, as expected
given its size. HAC t-stats for the top models are ~18 (p ≈ 0), so the ranking
skill is highly significant. Statistical significance is not the same as
tradability — see the return results below.

### 3.2 Long-short relative return (top/bottom 20%, 10 bps per side)

| Model | Gross ann. return | **Net ann. return** | Gross Sharpe | **Net Sharpe** | Daily turnover | Net max drawdown |
|---|---|---|---|---|---|---|
| **Random Forest** | 145% | **90%** | 2.25 | **1.40** | 1.50 | −71% |
| LightGBM | 134% | 76% | 2.17 | 1.23 | 1.59 | −77% |
| **COMB** | 98% | 49% | 1.60 | 0.80 | 1.35 | −71% |
| **OLS** | 87% | 41% | 1.42 | 0.67 | 1.27 | −76% |
| PLS / ENet / LASSO | ~85–95% | ~38–41% | ~1.4–1.6 | ~0.63–0.67 | ~1.3–1.5 | ~−71 to −80% |
| FFNN | 41% | −23% | 0.76 | −0.43 | 1.76 | −167% |

**The best model by IC is not the best model by return.** Random Forest has
slightly lower Rank IC than OLS/PLS but the highest long-short return, because
IC scores the whole ranking while the long-short return depends only on the top
and bottom tails, where the tree models discriminate better.

### 3.3 Cost sensitivity

The signal is dominated by short-term reversal (buy yesterday's losers, sell
yesterday's winners), so the book turns over heavily and net performance is very
sensitive to trading costs:

| Cost per side | Random Forest net Sharpe | OLS net Sharpe | COMB net Sharpe |
|---|---|---|---|
| 10 bps | 1.40 | 0.67 | 0.80 |
| 25 bps | 0.12 | −0.46 | −0.40 |
| 40 bps | −1.15 | −1.60 | −1.59 |

At realistic-optimistic costs the strategy works; by 25 bps only Random Forest
survives; by 40 bps all models lose money. Random Forest is both the
highest-returning and the most cost-robust model.

---

## 4. Interpretation and caveats

- **Gross vs net is the whole story.** Report these as gross cross-sectional
  predictability, with net-of-cost performance as the explicit test. A high gross
  Sharpe from a reversal signal is expected to shrink sharply after costs.
- **Short-borrow reality.** The short leg assumes bottom-quintile altcoins can be
  borrowed and shorted, which is often impractical or expensive in crypto. A
  long-only-versus-market variant is more realistic and is a natural Day-4 step.
- **Survivorship.** The audit flags that surviving symbols reach the final date,
  so the universe is not fully survivorship-free; this inflates predictability.
- **Drawdowns.** Even the best model has a net max drawdown near −70%. This is not
  a smooth strategy.

---

## 5. How to run

```bash
python crypto_day3_model.py \
    --input  crypto_analysis_ready.parquet \
    --output-dir day3_outputs
```

The input may be Parquet, JSON, CSV, or Pickle. Key settings live in the `Config`
dataclass, including:

- `horizon` (forecast horizon in days, default 1)
- `top_n` (universe size, default 50)
- `outer_folds`, `initial_train_fraction`, `embargo_days` (walk-forward CV)
- `long_short_quantile` (top/bottom fraction, default 0.20)
- `cost_bps_per_side` (transaction cost, default 10)
- `trading_days_per_year` (annualization, default 365; set to 252 for equity-style conventions)

---

## 6. Outputs

Written to the output directory:

**Headline / presentation**
- `day3_headline_metrics.csv` — OLS, Random Forest, COMB Rank-IC metrics
- `day3_relative_return_headline.csv` — headline long-short return metrics
- `day3_ic_diagnostics.png` — cumulative out-of-sample Rank IC (headline models)
- `day3_fold_ic.png` — mean Rank IC per walk-forward fold
- `day3_relative_return_curves.png` — gross vs net cumulative long-short return
- `DAY3_MODELING_SUMMARY.md` — written narrative summary

**Full evidence**
- `day3_model_metrics.csv` — Rank-IC metrics for all eight models
- `day3_relative_return_metrics.csv` — long-short return metrics for all models
- `day3_oos_predictions.csv` — per-date, per-coin predictions and realized returns
- `day3_relative_returns_daily.csv` — daily gross/net/turnover series per model
- `day3_ic_timeseries.csv` — daily Rank IC per model
- `day3_fold_metrics.csv` — per-fold IC and hit rate

**Interpretability / audit / reproducibility**
- `day3_ols_coefficients_by_fold.csv`, `day3_random_forest_importance_by_fold.csv`
- `day3_feature_dictionary.csv` — feature source, formula, rationale
- `day3_data_audit.csv` — exclusions, universe, survivorship checks
- `day3_walk_forward_folds.csv` — fold train/test windows and embargo
- `run_config.json`, `requirements_day3.txt`

---

## 7. Requirements

Python 3.11+ with: `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`,
`statsmodels`, `matplotlib`, `tabulate`. Exact versions are pinned in
`requirements_day3.txt` after a run.

---

## 8. Day 4 (planned)

Turn the daily forecasts into properly constructed portfolios: quantile
long-short baskets with position sizing, turnover and transaction-cost modeling,
a long-only-versus-market variant to handle the short-borrow constraint, and full
performance and drawdown reporting.
