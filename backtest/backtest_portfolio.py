"""
Day 4 portfolio evaluation: turnover, transaction costs, and gross/net returns.

Consumes the *_oos_predictions.csv files produced by backtest/pipeline.py or
accepts a predictions DataFrame directly.

Portfolio convention
--------------------
- Daily cross-sectional top/bottom quintile.
- 100% long top leg and 100% short bottom leg (gross exposure = 2.0, net = 0).
- Equal weights within each leg.
- turnover_half_l1 = 0.5 * sum_i |w_i,t - w_i,t-1|
- traded_notional = sum_i |w_i,t - w_i,t-1| = 2 * turnover_half_l1
- transaction costs are specified in ONE-WAY basis points per dollar traded:
      cost_t = traded_notional_t * cost_bps / 10,000
  Example: 10 bps one-way and traded_notional=1.5 implies a 15 bp return drag.

The first rebalance is measured from a zero portfolio, so implementation costs
are included rather than silently ignored.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

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


def _performance_summary(r: pd.Series, periods_per_year: int = 365) -> dict:
    r = pd.Series(r, dtype=float).dropna()
    n = int(len(r))
    if n == 0:
        return {
            "n_days": 0,
            "mean_daily_ret": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "cum_return": np.nan,
            "win_rate": np.nan,
            "drawdown_duration_days": np.nan,
        }

    wealth = (1.0 + r).cumprod()
    cum_return = float(wealth.iloc[-1] - 1.0)
    ann_return = float(wealth.iloc[-1] ** (periods_per_year / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else np.nan
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))
        if n > 1 and r.std(ddof=1) > 0
        else np.nan
    )
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    underwater = drawdown < 0
    max_duration = 0
    current = 0
    for flag in underwater:
        current = current + 1 if bool(flag) else 0
        max_duration = max(max_duration, current)

    return {
        "n_days": n,
        "mean_daily_ret": float(r.mean()),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "cum_return": cum_return,
        "win_rate": float((r > 0).mean()),
        "drawdown_duration_days": int(max_duration),
    }


def _weights_for_date(
    g: pd.DataFrame,
    pred_col: str,
    top_frac: float,
    min_leg_names: int,
) -> tuple[dict[str, float], dict]:
    g = g.dropna(subset=["symbol", pred_col, "y_real"]).copy()
    g = g.sort_values(pred_col, ascending=False)
    n = len(g)

    if n < 2 * min_leg_names:
        return {}, {}

    leg_n = max(min_leg_names, int(round(n * top_frac)))
    leg_n = min(leg_n, n // 2)
    if leg_n < 1:
        return {}, {}

    long_leg = g.iloc[:leg_n]
    short_leg = g.iloc[-leg_n:]

    weights: dict[str, float] = {}
    long_w = 1.0 / leg_n
    short_w = -1.0 / leg_n

    for symbol in long_leg["symbol"].astype(str):
        weights[symbol] = weights.get(symbol, 0.0) + long_w
    for symbol in short_leg["symbol"].astype(str):
        weights[symbol] = weights.get(symbol, 0.0) + short_w

    realized = dict(zip(g["symbol"].astype(str), g["y_real"].astype(float)))
    gross_ret = sum(w * realized.get(sym, 0.0) for sym, w in weights.items())
    long_contribution = sum(
        w * realized.get(sym, 0.0) for sym, w in weights.items() if w > 0
    )
    short_contribution = sum(
        w * realized.get(sym, 0.0) for sym, w in weights.items() if w < 0
    )

    info = {
        "n_names": n,
        "leg_n": leg_n,
        "gross_return": float(gross_ret),
        "long_contribution": float(long_contribution),
        "short_contribution": float(short_contribution),
        "gross_exposure": float(sum(abs(w) for w in weights.values())),
        "net_exposure": float(sum(weights.values())),
        "max_abs_weight": float(max(abs(w) for w in weights.values())),
    }
    return weights, info


def backtest_model(
    predictions: pd.DataFrame,
    pred_col: str,
    *,
    cost_bps: float = 10.0,
    top_frac: float = 0.20,
    min_leg_names: int = 3,
) -> pd.DataFrame:
    required = {"date", "symbol", "y_real", pred_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")

    x = predictions.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values(["date", "symbol"]).reset_index(drop=True)

    previous_weights: dict[str, float] = {}
    rows: list[dict] = []

    for date, g in x.groupby("date", sort=True):
        current_weights, info = _weights_for_date(
            g, pred_col=pred_col, top_frac=top_frac, min_leg_names=min_leg_names
        )
        if not info:
            continue

        all_symbols = set(previous_weights) | set(current_weights)
        traded_notional = float(
            sum(abs(current_weights.get(s, 0.0) - previous_weights.get(s, 0.0))
                for s in all_symbols)
        )
        turnover_half_l1 = 0.5 * traded_notional
        transaction_cost = traded_notional * float(cost_bps) / 10_000.0
        net_return = info["gross_return"] - transaction_cost

        rows.append({
            "date": date,
            **info,
            "turnover_half_l1": turnover_half_l1,
            "traded_notional": traded_notional,
            "cost_bps_one_way": float(cost_bps),
            "transaction_cost": transaction_cost,
            "net_return": float(net_return),
        })
        previous_weights = current_weights

    return pd.DataFrame(rows)


def build_costed_portfolio_outputs(
    predictions: pd.DataFrame,
    output_dir: Path,
    tag: str,
    *,
    cost_bps: float = 10.0,
    top_frac: float = 0.20,
    min_leg_names: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all available models and write Day-4 cost/turnover artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_parts = []
    summary_rows = []
    curve_data = {}

    for model, pred_col in MODEL_COLUMNS.items():
        if pred_col not in predictions.columns:
            continue

        daily = backtest_model(
            predictions,
            pred_col,
            cost_bps=cost_bps,
            top_frac=top_frac,
            min_leg_names=min_leg_names,
        )
        if daily.empty:
            continue

        daily.insert(1, "model", model)
        daily_parts.append(daily)

        gross_perf = _performance_summary(daily["gross_return"])
        net_perf = _performance_summary(daily["net_return"])

        common = {
            "model": model,
            "cost_bps_one_way": float(cost_bps),
            "avg_daily_turnover_half_l1": float(daily["turnover_half_l1"].mean()),
            "avg_daily_traded_notional": float(daily["traded_notional"].mean()),
            "annualized_traded_notional": float(daily["traded_notional"].mean() * 365),
            "total_transaction_cost": float(daily["transaction_cost"].sum()),
            "avg_daily_transaction_cost": float(daily["transaction_cost"].mean()),
            "avg_gross_exposure": float(daily["gross_exposure"].mean()),
            "avg_net_exposure": float(daily["net_exposure"].mean()),
            "avg_long_contribution": float(daily["long_contribution"].mean()),
            "avg_short_contribution": float(daily["short_contribution"].mean()),
        }

        for basis, perf in [("gross", gross_perf), ("net", net_perf)]:
            summary_rows.append({**common, "return_basis": basis, **perf})

        curve_data[(model, "Gross")] = (1.0 + daily.set_index("date")["gross_return"]).cumprod() - 1.0
        curve_data[(model, "Net")] = (1.0 + daily.set_index("date")["net_return"]).cumprod() - 1.0

        daily.to_csv(
            output_dir / f"{tag}_{_slug(model)}_daily_costed_returns.csv",
            index=False,
        )

    if not daily_parts:
        raise RuntimeError("No recognized prediction columns were available for portfolio evaluation.")

    daily_all = pd.concat(daily_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    daily_all.to_csv(output_dir / f"{tag}_portfolio_daily_gross_net.csv", index=False)
    summary.to_csv(output_dir / f"{tag}_portfolio_gross_net_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One figure per model keeps gross/net comparison readable.
    for model in summary["model"].drop_duplicates():
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for basis in ("Gross", "Net"):
            curve = curve_data.get((model, basis))
            if curve is not None:
                ax.plot(curve.index, curve.values * 100.0, label=basis)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(f"{model}: gross vs net cumulative return ({cost_bps:g} bps one-way)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{tag}_{_slug(model)}_gross_vs_net.png", dpi=160)
        plt.close(fig)

    # Turnover plot across all models.
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, g in daily_all.groupby("model"):
        gg = g.sort_values("date")
        rolling = gg.set_index("date")["traded_notional"].rolling(30, min_periods=5).mean()
        ax.plot(rolling.index, rolling.values, label=model)
    ax.set_title("30-day rolling average traded notional")
    ax.set_xlabel("Date")
    ax.set_ylabel("Traded notional (sum |Δw|)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_turnover_rolling30.png", dpi=160)
    plt.close(fig)

    return daily_all, summary


def _prediction_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*_oos_predictions.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day-4 transaction costs, turnover, and gross/net portfolio evaluation."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="One *_oos_predictions.csv file or a directory containing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest/pipeline_outputs/diagnostics/portfolio"),
    )
    parser.add_argument("--cost-bps", type=float, default=10.0,
                        help="One-way transaction cost in bps per dollar traded.")
    parser.add_argument("--top-frac", type=float, default=0.20)
    parser.add_argument("--min-leg-names", type=int, default=3)
    args = parser.parse_args()

    files = _prediction_files(args.predictions)
    if not files:
        raise FileNotFoundError(f"No *_oos_predictions.csv files found under {args.predictions}")

    for pred_path in files:
        predictions = pd.read_csv(pred_path)
        tag = pred_path.stem.replace("_oos_predictions", "")
        out = args.output_dir / tag
        _, summary = build_costed_portfolio_outputs(
            predictions,
            out,
            tag,
            cost_bps=args.cost_bps,
            top_frac=args.top_frac,
            min_leg_names=args.min_leg_names,
        )
        cols = [
            "model", "return_basis", "ann_return", "ann_vol", "sharpe",
            "max_drawdown", "avg_daily_traded_notional", "total_transaction_cost"
        ]
        print(f"\n{pred_path}")
        print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
