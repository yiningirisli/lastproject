"""
Runs the Day-3 paper-model pipeline (crypto_model.py) on two chronological
windows of the same panel:

  Window 1 ("model" window):     --start-date -> --split-date   (walk-forward OOS)
  Window 2 ("backtest" window):  --split-date -> end of file     (held-out forward test)

Feature engineering, panel safeguards, universe construction, walk-forward folds,
the LightGBM parameterization, the HAC (Newey-West) t-stat, and the Rank IC /
hit-rate diagnostics are all imported from crypto_model.py so the two files stay
faithful to one source of truth. LightGBM and statsmodels are pinned in
requirements.txt and used directly here (no stand-ins).

On top of crypto_model.py's Rank-IC diagnostics, this file adds what that file
defers to "Day 4": a long-short quintile portfolio return curve and a performance
summary (annualized return, annualized vol, Sharpe, max drawdown) for the three
headline models (OLS, Random Forest, COMB). Transaction costs, turnover, and
net-of-cost performance are added by backtest_portfolio.py, which consumes the
`*_oos_predictions.csv` this file writes.

Usage:
    python backtest/pipeline.py \
        --input data/crypto_analysis_ready.json \
        --output-dir backtest/pipeline_outputs \
        --start-date 2023-01-01 --split-date 2026-01-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from crypto_model import (  # noqa: E402
    Config,
    prepare_analysis_panel,
    build_market_return,
    build_features,
    create_model_frame,
    fit_walk_forward,
    evaluate,
)

# Models shown in the portfolio section (the presentation's three headline outputs).
PORTFOLIO_MODELS = {"OLS": "pred_ols", "Random Forest": "pred_random_forest", "COMB": "pred_comb"}


# ---------------------------------------------------------------------------
# Reproducibility helpers: hash inputs/outputs, pin the code commit and the
# exact package versions, and record the config so any run is rerunnable.
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    import sklearn, scipy, lightgbm, statsmodels
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
        "lightgbm": lightgbm.__version__, "statsmodels": statsmodels.__version__,
    }


def write_run_manifest(output_dir: Path, tag: str, cfg: Config, input_path: Path,
                       input_sha256: str, artifacts: dict[str, Path], span: dict) -> None:
    from dataclasses import asdict
    manifest = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(Path(__file__).resolve().parent),
        "pipeline_script": Path(__file__).name,
        "pipeline_script_sha256": sha256(Path(__file__).resolve()),
        "model_module_sha256": sha256(Path(__file__).resolve().parent / "crypto_model.py"),
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "seed": cfg.random_state,
        "window": span,
        "config": asdict(cfg),
        "package_versions": package_versions(),
        "artifact_sha256": {name: sha256(p) for name, p in artifacts.items() if p.exists()},
    }
    (output_dir / f"{tag}_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")



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

    summary_rows = []
    curves = {}
    for model, col in PORTFOLIO_MODELS.items():
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
def run_window(model_df_window: pd.DataFrame, cfg: Config, output_dir: Path, tag: str, label: str,
               input_path: Path, input_sha256: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} ===")
    print(f"    rows={len(model_df_window):,}  dates={model_df_window['date'].nunique():,}  "
          f"span={model_df_window['date'].min().date()} -> {model_df_window['date'].max().date()}")

    predictions, folds, ols_coef, rf_imp = fit_walk_forward(model_df_window, cfg)
    metrics, ic_ts, fold_metrics = evaluate(predictions, cfg)

    pred_path = output_dir / f"{tag}_oos_predictions.csv"
    perf_path = output_dir / f"{tag}_portfolio_performance_summary.csv"
    predictions.to_csv(pred_path, index=False)
    metrics.to_csv(output_dir / f"{tag}_model_metrics.csv", index=False)
    ic_ts.to_csv(output_dir / f"{tag}_ic_timeseries.csv", index=False)
    fold_metrics.to_csv(output_dir / f"{tag}_fold_metrics.csv", index=False)
    folds.to_csv(output_dir / f"{tag}_walk_forward_folds.csv", index=False)
    ols_coef.to_csv(output_dir / f"{tag}_ols_coefficients.csv", index=False)
    rf_imp.to_csv(output_dir / f"{tag}_rf_importances.csv", index=False)

    perf_summary = build_portfolio_outputs(predictions, cfg, output_dir, tag)

    write_run_manifest(
        output_dir, tag, cfg, input_path, input_sha256,
        artifacts={"oos_predictions": pred_path, "portfolio_performance_summary": perf_path,
                   "model_metrics": output_dir / f"{tag}_model_metrics.csv"},
        span={"start": str(model_df_window["date"].min().date()),
              "end": str(model_df_window["date"].max().date()),
              "n_dates": int(model_df_window["date"].nunique()),
              "n_rows": int(len(model_df_window))},
    )

    print(f"    OOS predictions: {len(predictions):,} rows across {predictions['date'].nunique()} dates")
    print("    Headline Rank-IC metrics:")
    print(metrics[metrics["headline"]][["model", "mean_rank_ic", "ic_ir", "hit_rate", "n_ic_dates"]]
          .round(4).to_string(index=False))
    print("    Portfolio performance (long-short quintile):")
    print(perf_summary[perf_summary["leg"] == "long_short"]
          [["model", "ann_return", "ann_vol", "sharpe_ratio", "max_drawdown", "win_rate"]]
          .round(4).to_string(index=False))

    return metrics, perf_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day-3/4 crypto pipeline: fit, predict, portfolio")
    parser.add_argument("--input", type=Path, default=Path("data/crypto_analysis_ready.json"),
                        help="analysis-ready panel (JSON or Parquet)")
    parser.add_argument("--output-dir", type=Path, default=Path("backtest/pipeline_outputs"),
                        help="root directory for all outputs")
    parser.add_argument("--start-date", default="2023-01-01", help="model-window start (inclusive)")
    parser.add_argument("--split-date", default="2026-01-01",
                        help="model/backtest boundary: model window ends here, backtest window begins")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    np.random.seed(cfg.random_state)  # any incidental numpy draws are deterministic

    # Freeze the environment used for this run, and hash the input so the exact
    # panel behind these results is verifiable.
    (out_root / "resolved_environment.json").write_text(
        json.dumps(package_versions(), indent=2), encoding="utf-8")
    input_sha256 = sha256(args.input)
    print(f"      input sha256: {input_sha256[:16]}...  seed: {cfg.random_state}")

    print("[1/4] Loading raw panel...")
    raw = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_json(args.input)
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

    cut1 = pd.Timestamp(args.start_date, tz="UTC")
    cut2 = pd.Timestamp(args.split_date, tz="UTC")
    s1, s2 = cut1.strftime("%Y%m%d"), cut2.strftime("%Y%m%d")

    window_model = model_df_all[(model_df_all["date"] >= cut1) & (model_df_all["date"] < cut2)].copy()
    window_backtest = model_df_all[model_df_all["date"] >= cut2].copy()

    print(f"[3/4] Running model window ({cut1.date()} -> {cut2.date()})...")
    metrics_model, perf_model = run_window(
        window_model, cfg, out_root / f"model_window_{s1}_{s2}",
        tag=f"model_{s1}_{s2}", label=f"MODEL WINDOW: {cut1.date()} to {cut2.date()}",
        input_path=args.input, input_sha256=input_sha256,
    )

    print(f"[4/4] Running backtest window ({cut2.date()} -> end of file, out-of-sample forward test)...")
    if window_backtest.empty:
        print("      backtest window is empty for this split date — skipping.")
    else:
        try:
            metrics_bt, perf_bt = run_window(
                window_backtest, cfg, out_root / f"backtest_window_{s2}_onward",
                tag=f"backtest_{s2}_onward",
                label=f"BACKTEST WINDOW: {cut2.date()} to end of file (held-out forward period)",
                input_path=args.input, input_sha256=input_sha256,
            )
        except RuntimeError as exc:
            print(f"      backtest window skipped: {exc}")

    print("\nAll outputs written under", out_root)


if __name__ == "__main__":
    main()
