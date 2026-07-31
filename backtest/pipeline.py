"""
Runs the Day-3 paper-model pipeline (crypto_day3_model.py) on two chronological
windows of the same panel:

  Window 1 ("model" window):     2023-01-01  -> 2026-01-01   (in-sample fit + its own
                                                                internal walk-forward OOS)
  Window 2 ("backtest" window):  2026-01-01  -> end of file  (2026-07-28) treated as a
                                                                held-out forward test

Two environment substitutions were required because this container has no network
access to install packages that are not already present:

  * LightGBM is not installed -> replaced with sklearn's HistGradientBoostingRegressor
    (also a boosted-tree learner; hyperparameters mapped 1:1 where a mapping exists).
    This model's column is still called "LightGBM" / "GBRT" in outputs to preserve the
    paper's naming, but it is trained with HistGradientBoostingRegressor.
  * statsmodels is not installed -> the HAC (Newey-West) t-stat on mean Rank IC is
    computed with a small hand-rolled Newey-West estimator instead of
    statsmodels' sm.OLS(..., cov_type="HAC").

Everything else (feature engineering, panel safeguards, universe construction,
walk-forward folds, Rank IC/hit-rate diagnostics) is taken unmodified from
crypto_day3_model.py so the two runs stay faithful to that file.

On top of the original script's Rank-IC diagnostics, this file adds what the
original explicitly deferred to "Day 4": a long-short quintile portfolio return
curve and a performance summary (annualized return, annualized vol, Sharpe ratio,
max drawdown) for each of the three headline models (OLS, Random Forest, COMB).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from crypto_day3_model import (  # noqa: E402
    Config,
    FEATURES,
    prepare_analysis_panel,
    build_market_return,
    build_features,
    create_model_frame,
    make_outer_folds,
    rank_ic_series,
    hit_rate,
    permutation_null_ic,
    BASE_MODEL_COLUMNS,
    ALL_MODEL_COLUMNS,
)

HEADLINE_MODELS = ("OLS", "Random Forest", "COMB")  # paper's three headline outputs


# ---------------------------------------------------------------------------
# Substitute 1: HistGradientBoostingRegressor stands in for LightGBM (offline env)
# Substitute 2: hand-rolled Newey-West HAC t-stat stands in for statsmodels
# ---------------------------------------------------------------------------
def hac_tstat_manual(series: pd.Series, maxlags: int = 7) -> tuple[float, float]:
    y = series.dropna().to_numpy(float)
    n = len(y)
    if n < 10:
        return float("nan"), float("nan")
    mean = y.mean()
    resid = y - mean
    gamma0 = float(np.dot(resid, resid) / n)
    long_run_var = gamma0
    for lag in range(1, min(maxlags, n - 1) + 1):
        w = 1.0 - lag / (maxlags + 1)
        gamma = float(np.dot(resid[lag:], resid[:-lag]) / n)
        long_run_var += 2 * w * gamma
    long_run_var = max(long_run_var, 1e-12)
    se = np.sqrt(long_run_var / n)
    t = mean / se if se > 0 else float("nan")
    # Normal-approximation two-sided p-value (avoids a scipy/statsmodels dependency).
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


def fit_walk_forward_offline(model_df: pd.DataFrame, cfg: Config):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    outer_folds = make_outer_folds(model_df["date"], cfg)
    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    ols_coef_rows: list[dict] = []
    rf_imp_rows: list[dict] = []

    for fold_info in outer_folds:
        fold = int(fold_info["fold"])
        tr_dates = fold_info["train_dates"]
        te_dates = fold_info["test_dates"]
        train = model_df[model_df["date"].isin(tr_dates)].copy()
        test = model_df[model_df["date"].isin(te_dates)].copy()
        if len(train) < 1000 or test.empty:
            continue

        x_train = train[FEATURES].to_numpy(float)
        x_test = test[FEATURES].to_numpy(float)
        y_train = train["y"].to_numpy(float)

        scaler = StandardScaler().fit(x_train)
        x_train_s = scaler.transform(x_train)
        x_test_s = scaler.transform(x_test)

        ols = LinearRegression()
        ols.fit(x_train_s, y_train)
        pred_ols = ols.predict(x_test_s)

        pls = PLSRegression(n_components=min(cfg.pls_components, len(FEATURES)), scale=False, max_iter=500)
        pls.fit(x_train_s, y_train)
        pred_pls = np.asarray(pls.predict(x_test_s)).reshape(-1)

        lasso = Lasso(alpha=cfg.lasso_alpha, fit_intercept=True, max_iter=2500, tol=1e-3, random_state=cfg.random_state)
        lasso.fit(x_train_s, y_train)
        pred_lasso = lasso.predict(x_test_s)

        enet = ElasticNet(
            alpha=cfg.enet_alpha, l1_ratio=cfg.enet_l1_ratio,
            fit_intercept=True, max_iter=2500, tol=1e-3,
            random_state=cfg.random_state, selection="cyclic"
        )
        enet.fit(x_train_s, y_train)
        pred_enet = enet.predict(x_test_s)

        rf = RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=cfg.rf_max_features,
            bootstrap=True,
            random_state=cfg.random_state + fold,
            n_jobs=-1,
        )
        rf.fit(x_train, y_train)
        pred_rf = rf.predict(x_test)

        # LightGBM stand-in (no network access to install lightgbm in this container).
        hgb = HistGradientBoostingRegressor(
            learning_rate=cfg.lgb_learning_rate,
            max_iter=cfg.lgb_n_estimators,
            max_leaf_nodes=cfg.lgb_num_leaves,
            max_depth=cfg.lgb_max_depth,
            min_samples_leaf=cfg.lgb_min_child_samples,
            l2_regularization=cfg.lgb_reg_lambda,
            random_state=cfg.random_state + fold,
        )
        hgb.fit(x_train, y_train)
        pred_lgb = hgb.predict(x_test)

        ffnn = MLPRegressor(
            hidden_layer_sizes=(cfg.ffnn_hidden_1, cfg.ffnn_hidden_2),
            activation="relu", solver="adam", alpha=cfg.ffnn_alpha,
            batch_size=1024, learning_rate_init=1e-3,
            max_iter=cfg.ffnn_max_iter, early_stopping=True,
            validation_fraction=0.10, n_iter_no_change=3,
            random_state=cfg.random_state + fold,
        )
        ffnn.fit(x_train_s, y_train)
        pred_ffnn = ffnn.predict(x_test_s)

        out = test[["date", "symbol", "y_real", "y"]].copy()
        out["outer_fold"] = fold
        out["pred_ols"] = pred_ols
        out["pred_pls"] = pred_pls
        out["pred_lasso"] = pred_lasso
        out["pred_elastic_net"] = pred_enet
        out["pred_random_forest"] = pred_rf
        out["pred_lightgbm"] = pred_lgb
        out["pred_ffnn"] = pred_ffnn
        out["pred_comb"] = out[list(BASE_MODEL_COLUMNS.values())].mean(axis=1)
        prediction_parts.append(out)

        fold_rows.append({
            "fold": fold,
            "train_start": str(pd.Timestamp(tr_dates[0]).date()),
            "train_end": str(pd.Timestamp(tr_dates[-1]).date()),
            "test_start": str(pd.Timestamp(te_dates[0]).date()),
            "test_end": str(pd.Timestamp(te_dates[-1]).date()),
            "n_train_rows": len(train), "n_test_rows": len(test),
            "n_train_dates": len(tr_dates), "n_test_dates": len(te_dates),
            "embargo_days": cfg.embargo_days,
        })
        for feature, coef in zip(FEATURES, ols.coef_):
            ols_coef_rows.append({"fold": fold, "feature": feature,
                                   "coefficient": float(coef), "abs_coefficient": float(abs(coef))})
        for feature, importance in zip(FEATURES, rf.feature_importances_):
            rf_imp_rows.append({"fold": fold, "feature": feature, "importance": float(importance)})

    if not prediction_parts:
        raise RuntimeError("No outer folds were fitted for this window (too few dates).")

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["date", "symbol"])
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(ols_coef_rows), pd.DataFrame(rf_imp_rows)


def evaluate_offline(predictions: pd.DataFrame, cfg: Config):
    summary_rows, ic_frames, fold_rows = [], [], []
    for model, col in ALL_MODEL_COLUMNS.items():
        ic = rank_ic_series(predictions, col, cfg.minimum_names_for_ic)
        mean_ic = float(ic.mean())
        sd_ic = float(ic.std(ddof=1))
        hac_t, hac_p = hac_tstat_manual(ic)
        summary_rows.append({
            "model": model,
            "headline": model in HEADLINE_MODELS,
            "mean_rank_ic": mean_ic,
            "ic_std": sd_ic,
            "ic_ir": mean_ic / sd_ic if sd_ic > 0 else np.nan,
            "hac_t_stat": hac_t,
            "hac_p_value": hac_p,
            "positive_ic_rate": float((ic > 0).mean()),
            "hit_rate": hit_rate(predictions, col),
            "permutation_null_mean_ic": (
                permutation_null_ic(predictions, col, cfg.minimum_names_for_ic,
                                     cfg.random_state + list(ALL_MODEL_COLUMNS).index(model))
                if model in HEADLINE_MODELS else np.nan
            ),
            "n_ic_dates": int(len(ic)),
            "oos_start": str(ic.index.min().date()) if len(ic) else None,
            "oos_end": str(ic.index.max().date()) if len(ic) else None,
        })
        ic_frames.append(pd.DataFrame({"date": ic.index, "model": model, "rank_ic": ic.values}))

    for fold, fold_data in predictions.groupby("outer_fold"):
        for model, col in ALL_MODEL_COLUMNS.items():
            ic = rank_ic_series(fold_data, col, cfg.minimum_names_for_ic)
            fold_rows.append({
                "fold": int(fold), "model": model, "headline": model in HEADLINE_MODELS,
                "mean_rank_ic": float(ic.mean()), "positive_ic_rate": float((ic > 0).mean()),
                "hit_rate": hit_rate(fold_data, col), "n_dates": int(len(ic)),
            })

    metrics = pd.DataFrame(summary_rows).sort_values("mean_rank_ic", ascending=False)
    ic_ts = pd.concat(ic_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    return metrics, ic_ts, fold_metrics


# ---------------------------------------------------------------------------
# Day-4-style long-short portfolio construction (return curve, Sharpe, etc.)
# ---------------------------------------------------------------------------
def daily_long_short_returns(predictions: pd.DataFrame, pred_col: str, top_frac: float = 0.2,
                              min_leg_names: int = 3) -> pd.DataFrame:
    rows = []
    for date, g in predictions.groupby("date", sort=True):
        g = g.dropna(subset=[pred_col, "y_real"])
        n = len(g)
        leg_n = max(min_leg_names, int(round(n * top_frac)))
        if n < 2 * min_leg_names:
            continue
        g_sorted = g.sort_values(pred_col, ascending=False)
        long_leg = g_sorted.iloc[:leg_n]
        short_leg = g_sorted.iloc[-leg_n:]
        rows.append({
            "date": date,
            "n_names": n,
            "leg_n": leg_n,
            "long_ret": float(long_leg["y_real"].mean()),
            "short_ret": float(short_leg["y_real"].mean()),
            "long_short_ret": float(long_leg["y_real"].mean() - short_leg["y_real"].mean()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def performance_summary(daily_ret: pd.Series, periods_per_year: int = 365) -> dict:
    r = daily_ret.dropna()
    if len(r) < 5:
        return {"n_days": len(r), "mean_daily_ret": np.nan, "ann_return": np.nan,
                "ann_vol": np.nan, "sharpe_ratio": np.nan, "max_drawdown": np.nan,
                "cum_return": np.nan, "win_rate": np.nan}
    mean_d = float(r.mean())
    std_d = float(r.std(ddof=1))
    ann_return = (1 + mean_d) ** periods_per_year - 1
    ann_vol = std_d * np.sqrt(periods_per_year)
    sharpe = (mean_d / std_d) * np.sqrt(periods_per_year) if std_d > 0 else np.nan
    cum_curve = (1 + r).cumprod()
    running_max = cum_curve.cummax()
    drawdown = cum_curve / running_max - 1
    max_dd = float(drawdown.min())
    cum_return = float(cum_curve.iloc[-1] - 1)
    win_rate = float((r > 0).mean())
    return {"n_days": int(len(r)), "mean_daily_ret": mean_d, "ann_return": ann_return,
            "ann_vol": ann_vol, "sharpe_ratio": sharpe, "max_drawdown": max_dd,
            "cum_return": cum_return, "win_rate": win_rate}


def build_portfolio_outputs(predictions: pd.DataFrame, cfg: Config, output_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_cols = {"OLS": "pred_ols", "Random Forest": "pred_random_forest", "COMB": "pred_comb"}
    summary_rows = []
    curves = {}
    for model, col in pred_cols.items():
        ls = daily_long_short_returns(predictions, col)
        ls = ls.set_index("date")
        curves[model] = (1 + ls["long_short_ret"]).cumprod() - 1
        ls.to_csv(output_dir / f"{tag}_{model.replace(' ', '_').lower()}_daily_long_short_returns.csv")
        perf = performance_summary(ls["long_short_ret"])
        perf["model"] = model
        perf["leg"] = "long_short"
        summary_rows.append(perf)
        perf_long = performance_summary(ls["long_ret"])
        perf_long["model"] = model
        perf_long["leg"] = "long_only"
        summary_rows.append(perf_long)

    summary = pd.DataFrame(summary_rows)[
        ["model", "leg", "n_days", "mean_daily_ret", "ann_return", "ann_vol",
         "sharpe_ratio", "max_drawdown", "cum_return", "win_rate"]
    ]
    summary.to_csv(output_dir / f"{tag}_portfolio_performance_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    for model, curve in curves.items():
        ax.plot(curve.index, curve.values * 100, label=model)
    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_title(f"Long-short (top vs. bottom quintile) cumulative return — {tag}")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_return_curve.png", dpi=160)
    plt.close(fig)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_window(model_df_window: pd.DataFrame, cfg: Config, output_dir: Path, tag: str, label: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} ===")
    print(f"    rows={len(model_df_window):,}  dates={model_df_window['date'].nunique():,}  "
          f"span={model_df_window['date'].min().date()} -> {model_df_window['date'].max().date()}")

    predictions, folds, ols_coef, rf_imp = fit_walk_forward_offline(model_df_window, cfg)
    metrics, ic_ts, fold_metrics = evaluate_offline(predictions, cfg)

    predictions.to_csv(output_dir / f"{tag}_oos_predictions.csv", index=False)
    metrics.to_csv(output_dir / f"{tag}_model_metrics.csv", index=False)
    ic_ts.to_csv(output_dir / f"{tag}_ic_timeseries.csv", index=False)
    fold_metrics.to_csv(output_dir / f"{tag}_fold_metrics.csv", index=False)
    folds.to_csv(output_dir / f"{tag}_walk_forward_folds.csv", index=False)

    perf_summary = build_portfolio_outputs(predictions, cfg, output_dir, tag)

    print(f"    OOS predictions: {len(predictions):,} rows across {predictions['date'].nunique()} dates")
    print("    Headline Rank-IC metrics:")
    print(metrics[metrics["headline"]][["model", "mean_rank_ic", "ic_ir", "hit_rate", "n_ic_dates"]]
          .round(4).to_string(index=False))
    print("    Portfolio performance (long-short quintile):")
    print(perf_summary[perf_summary["leg"] == "long_short"]
          [["model", "ann_return", "ann_vol", "sharpe_ratio", "max_drawdown", "win_rate"]]
          .round(4).to_string(index=False))

    return metrics, perf_summary


def main():
    raw_path = Path("/mnt/user-data/uploads/crypto_analysis_ready_1_.json")
    out_root = Path("/home/claude/work/outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    print("[1/4] Loading raw panel...")
    raw = pd.read_json(raw_path)
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"      {len(raw):,} rows, {raw['symbol'].nunique():,} symbols, "
          f"{raw['date'].min().date()} -> {raw['date'].max().date()}")

    print("[2/4] Applying panel safeguards, universe construction, and paper-mapped features "
          "(computed once on the FULL history so later date-filtering does not lose warm-up)...")
    panel, data_audit = prepare_analysis_panel(raw, cfg)
    market = build_market_return(panel)
    featured = build_features(panel, market)
    model_df_all, model_audit = create_model_frame(featured, cfg)
    data_audit = pd.concat([data_audit, model_audit], ignore_index=True)
    data_audit.to_csv(out_root / "data_audit_full_history.csv", index=False)
    print(f"      full model frame: {len(model_df_all):,} rows, {model_df_all['date'].nunique():,} dates")

    cut1 = pd.Timestamp("2023-01-01", tz="UTC")
    cut2 = pd.Timestamp("2026-01-01", tz="UTC")

    window_model = model_df_all[(model_df_all["date"] >= cut1) & (model_df_all["date"] < cut2)].copy()
    window_backtest = model_df_all[model_df_all["date"] >= cut2].copy()

    print("[3/4] Running model window (2023-01-01 -> 2026-01-01)...")
    metrics_model, perf_model = run_window(
        window_model, cfg, out_root / "model_window_2023_2026",
        tag="model_2023_2026", label="MODEL WINDOW: 2023-01-01 to 2026-01-01"
    )

    print("[4/4] Running backtest window (2026-01-01 -> end of file, out-of-sample forward test)...")
    metrics_bt, perf_bt = run_window(
        window_backtest, cfg, out_root / "backtest_window_2026_onward",
        tag="backtest_2026_onward", label="BACKTEST WINDOW: 2026-01-01 to 2026-07-28 (held-out forward period)"
    )

    print("\nAll outputs written under", out_root)


if __name__ == "__main__":
    main()
