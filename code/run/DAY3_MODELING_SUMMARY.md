# MMF1927H Option 4 — Day 3 Modeling Summary

## Headline model decision

The presentation focuses on three paper-motivated outputs:

1. **OLS** — the paper reports the least-negative predictive out-of-sample R² and the lowest MAE.
2. **Random Forest** — the paper reports the highest average forecast correlation.
3. **COMB** — the paper's forecast combination, defined as the equal-weight average of all seven base-model forecasts.

COMB is an ensemble, not a separately fitted third algorithm. To reproduce its structure honestly, the code fits OLS, PLS, LASSO, Elastic Net, Random Forest, LightGBM/GBRT, and FFNN inside each walk-forward fold, then averages their forecasts. Elastic Net and LightGBM are therefore still fitted and evaluated, satisfying the explicit course requirement, while OLS, Random Forest, and COMB remain the presentation's three headline outputs.

## Target and validation

- Target: **1-day-ahead close-to-close return**.
- Features: known at date *t* and cross-sectionally ranked on that date.
- Universe: top-50 eligible instruments by lagged 30-day dollar volume.
- Evaluation: 3 expanding walk-forward folds with a 5-day embargo.
- Main Day-3 metrics: daily Rank IC, IC stability, positive-IC rate, and hit rate.

## Headline out-of-sample results

| model         |   mean_rank_ic |   ic_ir |   positive_ic_rate |   hit_rate |   permutation_null_mean_ic |   n_ic_dates | oos_start   | oos_end    |
|:--------------|---------------:|--------:|-------------------:|-----------:|---------------------------:|-------------:|:------------|:-----------|
| PLS           |         0.0938 |  0.3889 |             0.6573 |     0.5188 |                     0.0034 |         1824 | 2021-07-30  | 2026-07-27 |
| LASSO         |         0.0919 |  0.3893 |             0.6727 |     0.517  |                     0.0048 |         1824 | 2021-07-30  | 2026-07-27 |
| Elastic Net   |         0.0918 |  0.3892 |             0.6716 |     0.5169 |                     0.0001 |         1824 | 2021-07-30  | 2026-07-27 |
| OLS           |         0.0916 |  0.3892 |             0.6694 |     0.5171 |                     0.0037 |         1824 | 2021-07-30  | 2026-07-27 |
| Random Forest |         0.091  |  0.3844 |             0.6573 |     0.504  |                     0.0004 |         1824 | 2021-07-30  | 2026-07-27 |
| COMB          |         0.0873 |  0.3644 |             0.6502 |     0.5149 |                     0.0005 |         1824 | 2021-07-30  | 2026-07-27 |
| LightGBM      |         0.0854 |  0.3749 |             0.6453 |     0.5042 |                     0.0019 |         1824 | 2021-07-30  | 2026-07-27 |
| FFNN          |         0.0276 |  0.1469 |             0.5647 |     0.505  |                    -0.002  |         1824 | 2021-07-30  | 2026-07-27 |

The strongest headline mean Rank IC in this run is **PLS** at **0.0938**. This is our own sample result; it does not have to reproduce the paper's exact ranking because our universe, data source, features, and one-day horizon differ from the paper's weekly design.

## Day-3 requirement status

| Requirement | Status |
|---|---|
| Documented feature set with internal/external and risk-bucket tags | Met |
| Paper-derived feature construction reproduced | Met |
| Penalized regression fitted | Met internally through Elastic Net |
| LightGBM fitted | Met internally as the GBRT constituent |
| Initial IC and hit-rate read | Met for all models; three headline rows reported |
| Git commit | Must be completed in the group repository |

## Important interpretation

- **OLS, RF, and COMB are selected because they represent three different paper outcomes**, not because one universal metric declares all three jointly best.
- The paper's Table 3 gives OLS the best R²/MAE and RF the highest average Pearson correlation; the paper's portfolio section gives COMB the best value-weighted risk-adjusted performance.
- Our Day-3 comparison uses Rank IC because our task is cross-sectional ranking.

## Remaining data limitation

**Survivorship warning:** the available candidate-symbol history appears to contain final-date survivors, so the project must not claim a fully survivorship-free universe.
