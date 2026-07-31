"""
Day 3 multicollinearity diagnostics for the exact feature matrix used by crypto_model.py.

The script intentionally diagnoses the post-construction, per-date ranked model frame
rather than raw OHLCV columns. That is the matrix actually fed to the models.

Outputs
-------
- feature_correlation.csv
- high_correlation_pairs.csv
- feature_vif.csv
- feature_correlation_heatmap.png
- feature_diagnostics_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crypto_model import (  # noqa: E402
    Config,
    FEATURES,
    load_input,
    prepare_analysis_panel,
    build_market_return,
    build_features,
    create_model_frame,
)


def _compute_vif(x: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    arr = x.to_numpy(float)
    arr = sm.add_constant(arr, has_constant="add")
    names = ["const", *list(x.columns)]

    rows = []
    for i, name in enumerate(names):
        if name == "const":
            continue
        try:
            vif = float(variance_inflation_factor(arr, i))
        except Exception:
            vif = np.inf
        rows.append({"feature": name, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def run_feature_diagnostics_from_frame(
    model_df: pd.DataFrame,
    output_dir: Path,
    *,
    max_vif_rows: int = 50_000,
    corr_threshold: float = 0.80,
    vif_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = set(FEATURES).difference(model_df.columns)
    if missing:
        raise ValueError(f"Model frame missing feature columns: {sorted(missing)}")

    x = model_df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if x.empty:
        raise RuntimeError("No complete feature rows available for diagnostics.")

    corr = x.corr(method="pearson")
    corr.to_csv(output_dir / "feature_correlation.csv")

    pairs = []
    for i, f1 in enumerate(FEATURES):
        for j in range(i + 1, len(FEATURES)):
            f2 = FEATURES[j]
            value = float(corr.loc[f1, f2])
            if abs(value) >= corr_threshold:
                pairs.append(
                    {
                        "feature_1": f1,
                        "feature_2": f2,
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )
    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        pair_df = pd.DataFrame(
            columns=["feature_1", "feature_2", "correlation", "abs_correlation"]
        )
    else:
        pair_df = pair_df.sort_values("abs_correlation", ascending=False)
    pair_df.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

    # Deterministic downsample keeps VIF tractable without changing date order at random.
    if len(x) > max_vif_rows:
        idx = np.linspace(0, len(x) - 1, max_vif_rows).astype(int)
        x_vif = x.iloc[idx].copy()
    else:
        x_vif = x

    vif = _compute_vif(x_vif)
    vif["above_threshold"] = vif["vif"] >= vif_threshold
    vif.to_csv(output_dir / "feature_vif.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_yticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURES, fontsize=8)
    ax.set_title("Feature correlation matrix (actual ranked model inputs)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    summary = {
        "n_model_rows": int(len(model_df)),
        "n_complete_feature_rows": int(len(x)),
        "n_features": int(len(FEATURES)),
        "vif_sample_rows": int(len(x_vif)),
        "corr_threshold_abs": float(corr_threshold),
        "vif_threshold": float(vif_threshold),
        "n_high_corr_pairs": int(len(pair_df)),
        "n_features_vif_above_threshold": int(vif["above_threshold"].sum()),
        "highest_vif_feature": None if vif.empty else str(vif.iloc[0]["feature"]),
        "highest_vif": None if vif.empty else float(vif.iloc[0]["vif"]),
        "interpretation_note": (
            "High correlation/VIF does not automatically require deletion for tree models, "
            "but it weakens coefficient stability and feature-attribution interpretation. "
            "If near-duplicates remain, document the economic reason or use a representative feature."
        ),
    }
    (output_dir / "feature_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return corr, pair_df, vif


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-3 correlation and VIF diagnostics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/crypto_analysis_ready_1_.json"),
        help="Analysis-ready panel used by crypto_model.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest/pipeline_outputs/diagnostics/features"),
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-vif-rows", type=int, default=50_000)
    parser.add_argument("--corr-threshold", type=float, default=0.80)
    parser.add_argument("--vif-threshold", type=float, default=5.0)
    args = parser.parse_args()

    cfg = Config()
    raw = load_input(args.input)
    panel, _ = prepare_analysis_panel(raw, cfg)
    market = build_market_return(panel)
    featured = build_features(panel, market)
    model_df, _ = create_model_frame(featured, cfg)

    if args.start_date:
        model_df = model_df[model_df["date"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        model_df = model_df[model_df["date"] < pd.Timestamp(args.end_date, tz="UTC")]

    _, pairs, vif = run_feature_diagnostics_from_frame(
        model_df,
        args.output_dir,
        max_vif_rows=args.max_vif_rows,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
    )

    print("\nHighest VIF values:")
    print(vif.head(10).round(3).to_string(index=False))
    print("\nHighest-correlation pairs:")
    print(pairs.head(10).round(3).to_string(index=False) if not pairs.empty else "None above threshold.")


if __name__ == "__main__":
    main()
