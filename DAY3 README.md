# MMF1927H Option 4 — Cryptocurrency Cross-Section, Day 3

## Objective

Use information available at date `t` to rank eligible cryptocurrencies by their
one-day-ahead relative returns.

## Three headline outputs

We focus the presentation on three outcomes motivated by Cakici et al. (2024):

1. **OLS** — the paper reports the best predictive out-of-sample R² and lowest MAE.
2. **Random Forest** — the paper reports the highest average forecast correlation.
3. **COMB** — the paper reports the strongest value-weighted risk-adjusted portfolio
   performance for the equal-weight forecast combination.

`COMB` is an ensemble rather than a separately fitted algorithm. Following the paper's
structure, the script fits seven base models inside each walk-forward fold—OLS, PLS,
LASSO, Elastic Net, Random Forest, LightGBM/GBRT, and FFNN—and averages their forecasts.
Only OLS, Random Forest, and COMB are emphasized in the presentation.

This design also retains the course-required **penalized regression** (Elastic Net) and
**LightGBM** models. Their diagnostics are included in `results/model_metrics.csv`.

## Headline out-of-sample results

| Model | Mean Rank IC | IC-IR | Positive-IC dates | Hit rate |
|---|---:|---:|---:|---:|
| OLS | 0.0916 | 0.3892 | 66.94% | 51.71% |
| Random Forest | 0.0910 | 0.3844 | 65.73% | 50.40% |
| COMB | 0.0873 | 0.3644 | 65.02% | 51.49% |

These are results from our own daily Binance-style panel and should not be expected to
match the paper's weekly, multi-exchange results exactly.

## Files

- `crypto_day3_model.py` — full feature-engineering, walk-forward modeling, COMB,
  and Day-3 evaluation pipeline
- `feature_dictionary.csv` — feature formulas, economic rationale, source-paper map,
  internal/external tag, and risk-model bucket
- `results/model_metrics.csv` — all model diagnostics, with a `headline` flag
- `results/ic_diagnostics.png` — cumulative Rank IC for OLS, Random Forest, and COMB
- `requirements.txt` — Python dependencies

The cleaned dataset is not committed because it is large. Put it locally under `data/`.

## Run

```bash
pip install -r requirements.txt

python crypto_day3_model.py \
  --input data/cleaned_crypto.json \
  --output-dir outputs
```

The script creates detailed local outputs under `outputs/`; they do not need to be
committed to GitHub.

## Day 3 scope

This repository covers feature engineering, paper-derived features, model fitting,
chronological walk-forward testing, Rank IC, hit rate, and model interpretation.
Portfolio weights, turnover, costs, neutrality, Sharpe ratio, and residual diagnostics
belong to Day 4.

## Important limitation

The available candidate-symbol history may contain survivorship bias. We disclose this
and do not describe the universe as fully survivorship-free.

## References

- Cakici, N., Shahzad, S. J. H., Będowska-Sojka, B., & Zaremba, A. (2024).
  *Machine learning and the cross-section of cryptocurrency returns*.
- Liu, Y., Tsyvinski, A., & Wu, X. (2021).
  *Common Risk Factors in Cryptocurrency*.
