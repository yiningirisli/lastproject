"""
Day 4 residual diagnostics for OOS model predictions.

Important target convention:
- crypto_model.py fits models to standardized cross-sectional target `y`.
- portfolio P&L is evaluated using raw future return `y_real`.
- residual diagnostics therefore use residual = y - prediction.

Autocorrelation is evaluated on the DAILY MEAN residual so repeated cross-sectional
rows on the same date are not falsely treated as a serial time series.
Distribution and heteroskedasticity diagnostics use the pooled OOS residuals.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera

MODEL_COLUMNS = {
    "OLS": "pred_ols",
    "PLS": "pred_pls",
    "LASSO": "pred_lasso",
    "Elastic Net": "pred_elastic_net",
    "Random Forest": "pred_random_forest",
    "LightGBM": "pred_lightgbm",
    "FFNN": "pred_ffnn",
    "COMB": "pred_comb",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _safe_pvalue(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _interpret(row: dict, alpha: float = 0.05) -> str:
    flags = []
    lb = row.get("ljung_box_pvalue_lag10", np.nan)
    bp = row.get("breusch_pagan_pvalue", np.nan)
    white = row.get("white_pvalue", np.nan)
    jb = row.get("jarque_bera_pvalue", np.nan)

    if np.isfinite(lb) and lb < alpha:
        flags.append("serial dependence in daily mean residuals")
    if (np.isfinite(bp) and bp < alpha) or (np.isfinite(white) and white < alpha):
        flags.append("residual variance depends on fitted level")
    if np.isfinite(jb) and jb < alpha:
        flags.append("non-normal residual distribution")

    if not flags:
        return "No major rejection at the 5% level; still inspect plots and effect sizes."
    return "Potential misspecification: " + "; ".join(flags) + "."


def diagnose_model(
    predictions: pd.DataFrame,
    model: str,
    pred_col: str,
    output_dir: Path,
    tag: str,
) -> tuple[dict, pd.DataFrame]:
    required = {"date", "symbol", "y", pred_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns for residual diagnostics: {sorted(missing)}")

    x = predictions[["date", "symbol", "y", pred_col]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", pred_col])
    x["fitted"] = x[pred_col].astype(float)
    x["residual"] = x["y"].astype(float) - x["fitted"]

    residual = x["residual"].to_numpy(float)
    fitted = x["fitted"].to_numpy(float)

    daily = (
        x.groupby("date", sort=True)
        .agg(
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
            rmse=("residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            mae=("residual", lambda s: float(np.mean(np.abs(s)))),
            n_names=("residual", "size"),
        )
        .reset_index()
    )

    dm = daily["mean_residual"].dropna().to_numpy(float)
    dw = float(durbin_watson(dm)) if len(dm) >= 3 else np.nan

    lb_p = {1: np.nan, 5: np.nan, 10: np.nan}
    valid_lags = [lag for lag in lb_p if len(dm) > lag + 2]
    if valid_lags:
        lb = acorr_ljungbox(dm, lags=valid_lags, return_df=True)
        for lag in valid_lags:
            lb_p[lag] = float(lb.loc[lag, "lb_pvalue"])

    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residual)

    bp_stat = bp_p = bp_f = bp_fp = np.nan
    white_stat = white_p = white_f = white_fp = np.nan
    if len(residual) >= 20 and np.nanstd(fitted) > 0:
        exog_bp = sm.add_constant(fitted, has_constant="add")
        bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residual, exog_bp)

        # het_white augments the supplied [constant, fitted] with fitted^2 internally.
        try:
            white_stat, white_p, white_f, white_fp = het_white(residual, exog_bp)
        except Exception:
            pass

    row = {
        "model": model,
        "n_residual_rows": int(len(residual)),
        "n_dates": int(daily["date"].nunique()),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)) if len(residual) > 1 else np.nan,
        "residual_skew": float(stats.skew(residual, bias=False)) if len(residual) > 2 else np.nan,
        "residual_excess_kurtosis": float(stats.kurtosis(residual, fisher=True, bias=False))
        if len(residual) > 3 else np.nan,
        "durbin_watson_daily_mean": dw,
        "ljung_box_pvalue_lag1": lb_p[1],
        "ljung_box_pvalue_lag5": lb_p[5],
        "ljung_box_pvalue_lag10": lb_p[10],
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_pvalue": _safe_pvalue(bp_p),
        "white_stat": float(white_stat),
        "white_pvalue": _safe_pvalue(white_p),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "jarque_bera_skew": float(jb_skew),
        "jarque_bera_kurtosis": float(jb_kurt),
    }
    row["interpretation"] = _interpret(row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.gofplots import qqplot

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ACF on time-indexed daily mean residual.
    if len(dm) >= 5:
        plot_acf(dm, lags=min(20, max(1, len(dm) // 3)), ax=axes[0, 0], zero=False)
        axes[0, 0].set_title("ACF: daily mean residual")
    else:
        axes[0, 0].text(0.5, 0.5, "Not enough dates for ACF", ha="center", va="center")
        axes[0, 0].set_title("ACF: daily mean residual")

    # Residual vs fitted.
    sample_n = min(len(residual), 20_000)
    if sample_n > 0:
        idx = np.linspace(0, len(residual) - 1, sample_n).astype(int)
        axes[0, 1].scatter(fitted[idx], residual[idx], s=5, alpha=0.25)
    axes[0, 1].axhline(0.0, linewidth=0.8)
    axes[0, 1].set_title("Residual vs fitted")
    axes[0, 1].set_xlabel("Fitted standardized return")
    axes[0, 1].set_ylabel("Residual")

    # Q-Q and histogram.
    qqplot(residual, line="45", ax=axes[1, 0], fit=True)
    axes[1, 0].set_title("Residual Q-Q")
    axes[1, 1].hist(residual, bins=60, density=True, alpha=0.75)
    axes[1, 1].set_title("Residual histogram")
    axes[1, 1].set_xlabel("Residual")

    fig.suptitle(f"{model} OOS residual diagnostics — {tag}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_{_slug(model)}_residual_diagnostics.png", dpi=160)
    plt.close(fig)

    daily.insert(1, "model", model)
    return row, daily


def run_residual_diagnostics(
    predictions: pd.DataFrame,
    output_dir: Path,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    daily_parts = []

    for model, pred_col in MODEL_COLUMNS.items():
        if pred_col not in predictions.columns:
            continue
        row, daily = diagnose_model(predictions, model, pred_col, output_dir, tag)
        rows.append(row)
        daily_parts.append(daily)

    if not rows:
        raise RuntimeError("No recognized prediction columns were available for residual diagnostics.")

    summary = pd.DataFrame(rows)
    daily_all = pd.concat(daily_parts, ignore_index=True)
    summary.to_csv(output_dir / f"{tag}_residual_diagnostics_summary.csv", index=False)
    daily_all.to_csv(output_dir / f"{tag}_residual_daily_timeseries.csv", index=False)
    return summary, daily_all


def _prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*_oos_predictions.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-4 OOS residual diagnostics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest/pipeline_outputs/diagnostics/residuals"),
    )
    args = parser.parse_args()

    files = _prediction_files(args.predictions)
    if not files:
        raise FileNotFoundError(f"No *_oos_predictions.csv files found under {args.predictions}")

    for pred_path in files:
        predictions = pd.read_csv(pred_path)
        tag = pred_path.stem.replace("_oos_predictions", "")
        out = args.output_dir / tag
        summary, _ = run_residual_diagnostics(predictions, out, tag)
        print(f"\n{pred_path}")
        print(
            summary[
                [
                    "model",
                    "durbin_watson_daily_mean",
                    "ljung_box_pvalue_lag10",
                    "breusch_pagan_pvalue",
                    "white_pvalue",
                    "jarque_bera_pvalue",
                    "interpretation",
                ]
            ].round(4).to_string(index=False)
        )


if __name__ == "__main__":
    main()
