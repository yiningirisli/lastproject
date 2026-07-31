"""
MMF1927H — Option 4: Cryptocurrency Cross-Section
Day 3: Feature Engineering & Modeling

Purpose
-------
Build a defensible Day-3 submission from the Day-2 OHLCV panel:
  1) paper-mapped, documented features;
  2) three headline outputs motivated by the source paper:
       - OLS (best level-prediction error in the paper)
       - Random Forest (highest average forecast correlation in the paper)
       - COMB (equal-weight combination of all seven base-model forecasts)
  3) the course-required Elastic Net and LightGBM models are also fitted as
     COMB constituents and retained in the all-model diagnostics;
  4) leakage-safe, date-based expanding-window evaluation;
  5) initial cross-sectional Rank IC and hit-rate diagnostics.

Important scope boundary
------------------------
This file intentionally stops at Day 3. It does NOT treat gross long-short wealth,
turnover, costs, factor-neutral portfolio construction, or residual diagnostics as
final results; those are Day-4 deliverables.

Paper connection
----------------
P1: Liu, Tsyvinski & Wu, "Common Risk Factors in Cryptocurrency"
P2: Cakici et al., "Machine Learning and the Cross-Section of Cryptocurrency Returns"

Model strategy
--------------
The presentation focuses on OLS, Random Forest, and COMB. COMB is not an
independently fitted third algorithm: following the paper, it is the equal-weight
average of forecasts from seven base models (OLS, PLS, LASSO, Elastic Net, Random
Forest, LightGBM/GBRT, and FFNN). This also preserves the course-required fitted
penalized regression and LightGBM models without making them the headline story.

Usage
-----
python crypto_day3_model.py \
    --input /path/to/cleaned_panel.json \
    --output-dir /path/to/day3_outputs
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    horizon: int = 1
    top_n: int = 50
    universe_lookback: int = 30
    minimum_model_history: int = 90
    minimum_names_for_ic: int = 10
    minimum_feature_coverage: float = 0.90
    outer_folds: int = 3
    initial_train_fraction: float = 0.40
    embargo_days: int = 5
    random_state: int = 42
    xsec_transform: str = "rank"

    # Fixed, pre-declared settings. They are not selected using outer test data.
    pls_components: int = 3
    lasso_alpha: float = 1e-4
    enet_alpha: float = 1e-4
    enet_l1_ratio: float = 0.5

    rf_n_estimators: int = 40
    rf_max_depth: int = 5
    rf_min_samples_leaf: int = 200
    rf_max_features: float = 0.70

    lgb_n_estimators: int = 80
    lgb_learning_rate: float = 0.03
    lgb_num_leaves: int = 15
    lgb_max_depth: int = 5
    lgb_min_child_samples: int = 100
    lgb_subsample: float = 0.80
    lgb_colsample_bytree: float = 0.80
    lgb_reg_alpha: float = 0.10
    lgb_reg_lambda: float = 1.00

    ffnn_hidden_1: int = 8
    ffnn_hidden_2: int = 4
    ffnn_alpha: float = 1e-3
    ffnn_max_iter: int = 15


# ---------------------------------------------------------------------------
# Explicit, conservative safety exclusions.
# These are documented transformations, not hidden edits.
# The list is intentionally narrow; survivorship bias cannot be repaired here.
# ---------------------------------------------------------------------------
STABLE_OR_PEGGED = {
    "BFUSD", "EURI", "FRAX", "RLUSD", "USDE", "USDS", "XUSD"
}
METAL_BACKED = {"PAXG", "XAUT"}
WRAPPED_OR_STAKED = {"BNSOL"}
DERIVATIVE_COLLATERAL_EXAMPLE = {"SNX"}
TOKENIZED_EQUITY_OR_INDEX = {
    "AMDB", "ARMB", "AT", "AVGOB", "AXTIB", "BABAB", "CBRSB", "COINB",
    "CRCLB", "CRWVB", "EWYB", "GOOGLB", "HOODB", "IBMB", "INTCB", "INTWB",
    "LITEB", "METAB", "MRVLB", "MSFTB", "MSTRB", "MUUB", "MVLLB", "NBISB",
    "NOKB", "NVDAB", "ORCLB", "PLTRB", "QCOMB", "QQQB", "RKLBB", "SNDKB",
    "SOXLB", "SPCXB", "SPYB", "TQQQB", "TSLAB", "TSMB", "WDCB"
}


def base_asset(symbol: pd.Series) -> pd.Series:
    return symbol.str.replace(r"USDT$", "", regex=True)


def exclusion_reason(base: str) -> str | None:
    if base in STABLE_OR_PEGGED:
        return "stable_or_pegged"
    if base in METAL_BACKED:
        return "metal_backed"
    if base in WRAPPED_OR_STAKED:
        return "wrapped_or_staked"
    if base in DERIVATIVE_COLLATERAL_EXAMPLE:
        return "derivative_collateral"
    if base in TOKENIZED_EQUITY_OR_INDEX:
        return "tokenized_equity_or_index"
    if base.startswith("1000") or base.startswith("1M"):
        return "multiplier_contract"
    return None


# ---------------------------------------------------------------------------
# Feature dictionary: source, formula, rationale, and adaptation are explicit.
# ---------------------------------------------------------------------------
FEATURE_META: dict[str, dict[str, str]] = {
    "log_prc": {
        "category": "size_price", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 PRC; P2 prc",
        "formula": "log(close_t)",
        "economic_rationale": "Low nominal price may proxy for size, accessibility, and limits to arbitrage.",
        "implementation_note": "Exact market-price analogue; market capitalization unavailable."
    },
    "age": {
        "category": "size_price", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 AGE",
        "formula": "log(consecutive trading days in current uninterrupted segment)",
        "economic_rationale": "Young assets can be less established, less liquid, and harder to arbitrage.",
        "implementation_note": "Resets after calendar gaps instead of using lifetime row count."
    },
    "mom_7": {
        "category": "momentum", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 r(1,0)",
        "formula": "log(close_t / close_{t-7})",
        "economic_rationale": "Short-horizon continuation is a central crypto return predictor.",
        "implementation_note": "Daily-data equivalent of one-week momentum."
    },
    "mom_21": {
        "category": "momentum", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 r(3,0)",
        "formula": "log(close_t / close_{t-21})",
        "economic_rationale": "Three-week momentum is used in the P1 crypto momentum factor.",
        "implementation_note": "Daily-data equivalent of three-week momentum."
    },
    "mom_7_28": {
        "category": "momentum", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 r(4,1)",
        "formula": "log(close_{t-7} / close_{t-28})",
        "economic_rationale": "Skip-recent momentum separates continuation from immediate reversal.",
        "implementation_note": "Daily-data equivalent of weeks 1–4 excluding the most recent week."
    },
    "r2_1": {
        "category": "reversal", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P2 r2_1",
        "formula": "close_t / close_{t-1} - 1",
        "economic_rationale": "Captures one-day reversal or continuation after an extreme move.",
        "implementation_note": "Computed only across consecutive calendar observations."
    },
    "r31_2": {
        "category": "momentum", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P2 r31_2",
        "formula": "log(close_{t-2} / close_{t-31})",
        "economic_rationale": "P2 identifies approximately one-month momentum as an important predictor.",
        "implementation_note": "Skips the most recent day to reduce bid-ask-bounce sensitivity."
    },
    "r30_14": {
        "category": "momentum", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P2 r30_14",
        "formula": "log(close_{t-14} / close_{t-30})",
        "economic_rationale": "Intermediate momentum distinguishes persistent trends from very recent noise.",
        "implementation_note": "Direct daily-horizon analogue."
    },
    "close_90dh": {
        "category": "momentum", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P2 90dh",
        "formula": "close_t / max(high_{t-89:t})",
        "economic_rationale": "Proximity to a recent high captures trend strength and anchoring.",
        "implementation_note": "Uses a trailing 90-calendar-observation window within a continuous segment."
    },
    "log_vol": {
        "category": "liquidity", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 VOL; P2 volume",
        "formula": "log(base-asset volume_t)",
        "economic_rationale": "Trading activity proxies for attention and liquidity.",
        "implementation_note": "Binance base volume; cross-sectionally ranked each date."
    },
    "log_prcvol": {
        "category": "liquidity", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 PRCVOL; P2 volume",
        "formula": "log(quote_asset_volume_t)",
        "economic_rationale": "Dollar trading volume measures investability and limits to arbitrage.",
        "implementation_note": "Quote-asset volume is already denominated in USDT."
    },
    "std_prcvol": {
        "category": "liquidity", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 STDPRCVOL",
        "formula": "log(std(quote_asset_volume, 7 days))",
        "economic_rationale": "Unstable trading activity may indicate attention shocks and fragile liquidity.",
        "implementation_note": "Seven-day daily-data analogue of weekly price-volume volatility."
    },
    "amihud_illiq": {
        "category": "liquidity", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P1 DAMIHUD; P2 illiq",
        "formula": "log(mean(|return| / dollar_volume, 90 days))",
        "economic_rationale": "Price impact and limits to arbitrage can support return predictability.",
        "implementation_note": "Uses safe consecutive returns and quote volume."
    },
    "volsh_30": {
        "category": "liquidity", "internal_external": "internal",
        "risk_bucket": "fundamental", "source_paper": "P2 volsh_30d",
        "formula": "log(volume_t) - log(mean(volume, 30 days))",
        "economic_rationale": "Abnormal trading-volume shocks may reveal attention or information arrival.",
        "implementation_note": "Current volume relative to trailing mean."
    },
    "retvol_7": {
        "category": "risk", "internal_external": "internal",
        "risk_bucket": "statistical", "source_paper": "P1 RETVOL",
        "formula": "std(daily return, 7 days)",
        "economic_rationale": "Short-horizon volatility captures rapidly changing risk and uncertainty.",
        "implementation_note": "One-week daily-data analogue."
    },
    "rvol_30": {
        "category": "risk", "internal_external": "internal",
        "risk_bucket": "statistical", "source_paper": "P2 rvol",
        "formula": "sqrt(mean(log(high/low)^2 / (4 log 2), 30 days))",
        "economic_rationale": "OHLC range contains information beyond close-to-close returns.",
        "implementation_note": "Parkinson analogue rather than P2's Yang–Zhang estimator."
    },
    "beta_30": {
        "category": "risk", "internal_external": "internal",
        "risk_bucket": "statistical", "source_paper": "P2 beta",
        "formula": "rolling 30-day cov(asset, market) / var(market)",
        "economic_rationale": "Separates broad crypto-market exposure from asset-specific behavior.",
        "implementation_note": "Market is lagged-volume-weighted because market cap is unavailable."
    },
    "ivol_30": {
        "category": "risk", "internal_external": "internal",
        "risk_bucket": "statistical", "source_paper": "P2 ivol",
        "formula": "rolling 30-day standard deviation of market-model residual",
        "economic_rationale": "High idiosyncratic risk can proxy for limits to arbitrage.",
        "implementation_note": "Single-factor market-model analogue."
    },
    "capm_alpha_30": {
        "category": "risk", "internal_external": "internal",
        "risk_bucket": "statistical", "source_paper": "P2 alpha",
        "formula": "rolling mean(asset return) - beta * rolling mean(market return)",
        "economic_rationale": "Past market-adjusted performance is a leading P2 predictor.",
        "implementation_note": "Daily alpha estimate over 30 days."
    },
    "var_90": {
        "category": "risk", "internal_external": "internal",
        "risk_bucket": "statistical", "source_paper": "P2 var",
        "formula": "5th percentile of daily return over 90 days",
        "economic_rationale": "Downside-tail risk may be priced differently from symmetric volatility.",
        "implementation_note": "Historical empirical VaR."
    },
    "skew_90": {
        "category": "distribution", "internal_external": "internal",
        "risk_bucket": "statistical", "source_paper": "P2 skew",
        "formula": "skewness of daily returns over 90 days",
        "economic_rationale": "Lottery-like upside tails and crash risk can affect expected returns.",
        "implementation_note": "Trailing return-distribution characteristic."
    },
}
FEATURES = list(FEATURE_META)


# ---------------------------------------------------------------------------
# I/O and panel safeguards
# ---------------------------------------------------------------------------
def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        df = pd.read_json(path)
    elif suffix in {".csv", ".gz"}:
        df = pd.read_csv(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pyarrow or fastparquet. Install one, or use the JSON input."
            ) from exc
    else:
        raise ValueError(f"Unsupported input type: {suffix}")

    required = {
        "date", "symbol", "open", "high", "low", "close", "volume",
        "quote_asset_volume"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def prepare_analysis_panel(raw: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    initial_rows = len(df)
    initial_symbols = df["symbol"].nunique()

    duplicate_count = int(df.duplicated(["symbol", "date"]).sum())
    if duplicate_count:
        df = df.drop_duplicates(["symbol", "date"], keep="last")

    df["base_asset"] = base_asset(df["symbol"])
    df["exclusion_reason"] = df["base_asset"].map(exclusion_reason)
    exclusion_counts = (
        df.dropna(subset=["exclusion_reason"])
          .groupby("exclusion_reason")
          .agg(rows=("symbol", "size"), symbols=("symbol", "nunique"))
          .reset_index()
    )
    df = df[df["exclusion_reason"].isna()].copy()

    # Break each history at any non-consecutive calendar observation.
    day_diff = df.groupby("symbol", sort=False)["date"].diff().dt.days
    df["segment_start"] = day_diff.isna() | day_diff.ne(1)
    df["segment_id"] = df.groupby("symbol", sort=False)["segment_start"].cumsum().astype(int)
    gap_count = int((df["segment_start"] & day_diff.notna()).sum())

    seg_keys = [df["symbol"], df["segment_id"]]
    df["consecutive_history_days"] = df.groupby(["symbol", "segment_id"], sort=False).cumcount() + 1
    df["return_safe"] = df.groupby(["symbol", "segment_id"], sort=False)["close"].pct_change()
    next_close = df.groupby(["symbol", "segment_id"], sort=False)["close"].shift(-cfg.horizon)
    df["target_safe"] = next_close / df["close"] - 1.0

    # Lagged 30-day average dollar volume; current-day volume is not used for membership.
    shifted_dvol = df.groupby(["symbol", "segment_id"], sort=False)["quote_asset_volume"].shift(1)
    df["trailing_quote_volume_safe"] = (
        shifted_dvol.groupby(seg_keys)
        .rolling(cfg.universe_lookback, min_periods=cfg.universe_lookback)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["eligible_universe_safe"] = (
        (df["consecutive_history_days"] >= cfg.universe_lookback + 1)
        & df["trailing_quote_volume_safe"].notna()
        & (df["trailing_quote_volume_safe"] > 0)
    )
    df["volume_rank_safe"] = np.nan
    eligible_idx = df.index[df["eligible_universe_safe"]]
    df.loc[eligible_idx, "volume_rank_safe"] = (
        df.loc[eligible_idx]
        .groupby("date")["trailing_quote_volume_safe"]
        .rank(method="first", ascending=False)
    )
    df["in_universe_safe"] = df["eligible_universe_safe"] & (df["volume_rank_safe"] <= cfg.top_n)

    final_date = df["date"].max()
    symbols_reaching_final_date = int(df.loc[df["date"].eq(final_date), "symbol"].nunique())
    surviving_symbols = int(df["symbol"].nunique())

    audit_rows = [
        {"check": "input_rows", "value": initial_rows, "note": "Rows before safeguards"},
        {"check": "input_symbols", "value": initial_symbols, "note": "Symbols before safeguards"},
        {"check": "duplicate_symbol_dates_removed", "value": duplicate_count, "note": "Kept last duplicate"},
        {"check": "excluded_rows", "value": initial_rows - len(df), "note": "Explicit instrument-type exclusions"},
        {"check": "excluded_symbols", "value": initial_symbols - df["symbol"].nunique(), "note": "Explicit instrument-type exclusions"},
        {"check": "calendar_gaps_detected", "value": gap_count, "note": "Histories reset at gaps"},
        {"check": "remaining_rows", "value": len(df), "note": "Rows after safeguards"},
        {"check": "remaining_symbols", "value": df["symbol"].nunique(), "note": "Symbols after safeguards"},
        {"check": "symbols_present_on_final_date", "value": symbols_reaching_final_date, "note": "Used as survivorship-bias warning"},
        {"check": "all_remaining_symbols_reach_final_date", "value": int(symbols_reaching_final_date == surviving_symbols), "note": "1 means survivorship concern remains"},
        {"check": "first_date", "value": str(df["date"].min().date()), "note": "UTC"},
        {"check": "last_date", "value": str(final_date.date()), "note": "UTC"},
    ]
    audit = pd.concat([pd.DataFrame(audit_rows), exclusion_counts.rename(columns={"exclusion_reason": "check", "rows": "value"}).assign(note=lambda x: "Excluded symbols=" + x["symbols"].astype(str)).drop(columns="symbols")], ignore_index=True)
    return df, audit


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _rolling(df: pd.DataFrame, col: str, window: int, min_periods: int, method: str) -> pd.Series:
    gb = df.groupby(["symbol", "segment_id"], sort=False)[col]
    roll = gb.rolling(window, min_periods=min_periods)
    result = getattr(roll, method)()
    return result.reset_index(level=[0, 1], drop=True)


def build_market_return(df: pd.DataFrame) -> pd.Series:
    m = df[df["in_universe_safe"] & df["return_safe"].notna()].copy()
    m["weight"] = m["trailing_quote_volume_safe"].clip(lower=0).fillna(0)

    def weighted_mean(g: pd.DataFrame) -> float:
        w = g["weight"].to_numpy(float)
        r = g["return_safe"].to_numpy(float)
        total = w.sum()
        return float(np.dot(w, r) / total) if total > 0 else float(np.nanmean(r))

    out = m.groupby("date", sort=True).apply(weighted_mean)
    out.name = "market_return"
    return out


def build_features(df: pd.DataFrame, market_return: pd.Series) -> pd.DataFrame:
    x = df.copy()
    eps = 1e-12
    x["log_close"] = np.log(x["close"].clip(lower=eps))
    g = x.groupby(["symbol", "segment_id"], sort=False)

    x["log_prc"] = x["log_close"]
    x["age"] = np.log(x["consecutive_history_days"].clip(lower=1))
    x["mom_7"] = x["log_close"] - g["log_close"].shift(7)
    x["mom_21"] = x["log_close"] - g["log_close"].shift(21)
    x["mom_7_28"] = g["log_close"].shift(7) - g["log_close"].shift(28)
    x["r2_1"] = x["return_safe"]
    x["r31_2"] = g["log_close"].shift(2) - g["log_close"].shift(31)
    x["r30_14"] = g["log_close"].shift(14) - g["log_close"].shift(30)

    x["rolling_high_90"] = _rolling(x, "high", 90, 90, "max")
    x["close_90dh"] = x["close"] / x["rolling_high_90"].clip(lower=eps)

    x["log_vol"] = np.log(x["volume"].clip(lower=eps))
    x["log_prcvol"] = np.log(x["quote_asset_volume"].clip(lower=eps))
    x["std_prcvol_raw"] = _rolling(x, "quote_asset_volume", 7, 7, "std")
    x["std_prcvol"] = np.log(x["std_prcvol_raw"].clip(lower=eps))

    x["amihud_daily"] = x["return_safe"].abs() / x["quote_asset_volume"].clip(lower=eps)
    x["amihud_illiq"] = np.log(_rolling(x, "amihud_daily", 90, 90, "mean").clip(lower=1e-30))
    x["volume_mean_30"] = _rolling(x, "quote_asset_volume", 30, 30, "mean")
    x["volsh_30"] = x["log_prcvol"] - np.log(x["volume_mean_30"].clip(lower=eps))

    x["retvol_7"] = _rolling(x, "return_safe", 7, 7, "std")
    x["park_daily"] = np.log((x["high"] / x["low"].clip(lower=eps)).clip(lower=1.0)) ** 2 / (4 * np.log(2))
    x["rvol_30"] = np.sqrt(_rolling(x, "park_daily", 30, 30, "mean").clip(lower=0))
    x["var_90"] = (
        x.groupby(["symbol", "segment_id"], sort=False)["return_safe"]
         .rolling(90, min_periods=90)
         .quantile(0.05)
         .reset_index(level=[0, 1], drop=True)
    )
    x["skew_90"] = _rolling(x, "return_safe", 90, 90, "skew")

    # Market-model features.
    x = x.merge(market_return, left_on="date", right_index=True, how="left")
    x["ret_sq"] = x["return_safe"] ** 2
    x["mkt_sq"] = x["market_return"] ** 2
    x["ret_mkt"] = x["return_safe"] * x["market_return"]

    rm = _rolling(x, "return_safe", 30, 30, "mean")
    mm = _rolling(x, "market_return", 30, 30, "mean")
    rr = _rolling(x, "ret_sq", 30, 30, "mean")
    m2 = _rolling(x, "mkt_sq", 30, 30, "mean")
    rmkt = _rolling(x, "ret_mkt", 30, 30, "mean")
    cov = rmkt - rm * mm
    var_m = (m2 - mm * mm).clip(lower=1e-12)
    var_r = (rr - rm * rm).clip(lower=0)
    beta = cov / var_m
    x["beta_30"] = beta
    x["capm_alpha_30"] = rm - beta * mm
    x["ivol_30"] = np.sqrt((var_r - beta * cov).clip(lower=0))

    return x


def create_model_frame(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = df[
        df["in_universe_safe"]
        & (df["consecutive_history_days"] >= cfg.minimum_model_history)
        & df["target_safe"].notna()
    ].copy()

    # P2's broad error-control bounds, applied only to the model target.
    m["y_real"] = m["target_safe"].clip(lower=-0.99, upper=3.0)
    m["feature_coverage"] = m[FEATURES].notna().mean(axis=1)
    before_coverage = len(m)
    m = m[m["feature_coverage"] >= cfg.minimum_feature_coverage].copy()

    # Per-date percentile ranks are robust and match IC-based evaluation.
    for feature in FEATURES:
        m[feature] = m.groupby("date")[feature].rank(pct=True) - 0.5
    m[FEATURES] = m[FEATURES].fillna(0.0)

    # Cross-sectional standardized target; evaluation remains against raw y_real.
    y_mean = m.groupby("date")["y_real"].transform("mean")
    y_std = m.groupby("date")["y_real"].transform("std").replace(0, np.nan)
    m["y"] = ((m["y_real"] - y_mean) / y_std).clip(-4, 4)
    m = m.dropna(subset=["y"]).sort_values(["date", "symbol"]).reset_index(drop=True)

    audit = pd.DataFrame([
        {"check": "rows_before_feature_coverage_filter", "value": before_coverage, "note": "Universe, history, and target valid"},
        {"check": "rows_after_feature_coverage_filter", "value": len(m), "note": f"Coverage >= {cfg.minimum_feature_coverage:.0%}"},
        {"check": "model_dates", "value": m["date"].nunique(), "note": "Distinct modeling dates"},
        {"check": "model_symbols", "value": m["symbol"].nunique(), "note": "Distinct modeled symbols"},
        {"check": "median_names_per_date", "value": float(m.groupby("date").size().median()), "note": "After filters"},
        {"check": "minimum_names_per_date", "value": int(m.groupby("date").size().min()), "note": "After filters"},
    ])
    return m, audit


# ---------------------------------------------------------------------------
# Date-based CV
# ---------------------------------------------------------------------------
def make_outer_folds(dates: Iterable[pd.Timestamp], cfg: Config) -> list[dict[str, np.ndarray]]:
    unique_dates = np.array(sorted(pd.unique(pd.Series(dates))))
    n = len(unique_dates)
    first_test = max(int(n * cfg.initial_train_fraction), 120)
    test_edges = np.linspace(first_test, n, cfg.outer_folds + 1).astype(int)
    folds: list[dict[str, np.ndarray]] = []
    for i in range(cfg.outer_folds):
        test_start, test_end = test_edges[i], test_edges[i + 1]
        train_end = test_start - cfg.embargo_days
        if train_end <= 60 or test_end <= test_start:
            continue
        folds.append({
            "fold": i + 1,
            "train_dates": unique_dates[:train_end],
            "test_dates": unique_dates[test_start:test_end],
        })
    return folds


def make_inner_date_splits(train_dates: np.ndarray, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(train_dates)
    first_validation = max(int(n * 0.55), 60)
    edges = np.linspace(first_validation, n, cfg.inner_splits + 1).astype(int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(cfg.inner_splits):
        val_start, val_end = edges[i], edges[i + 1]
        inner_train_end = val_start - cfg.embargo_days
        if inner_train_end <= 30 or val_end <= val_start:
            continue
        splits.append((train_dates[:inner_train_end], train_dates[val_start:val_end]))
    return splits


def mean_daily_rank_ic(dates: pd.Series, realized: np.ndarray, predicted: np.ndarray, min_names: int) -> float:
    tmp = pd.DataFrame({"date": dates.to_numpy(), "realized": realized, "predicted": predicted})
    values = []
    for _, g in tmp.groupby("date", sort=False):
        if len(g) >= min_names and g["realized"].nunique() > 1 and g["predicted"].nunique() > 1:
            values.append(g["realized"].corr(g["predicted"], method="spearman"))
    return float(np.nanmean(values)) if values else float("nan")



# ---------------------------------------------------------------------------
# Paper-model suite and forecast combination
# ---------------------------------------------------------------------------
BASE_MODEL_COLUMNS = {
    "OLS": "pred_ols",
    "PLS": "pred_pls",
    "LASSO": "pred_lasso",
    "Elastic Net": "pred_elastic_net",
    "Random Forest": "pred_random_forest",
    "LightGBM": "pred_lightgbm",
    "FFNN": "pred_ffnn",
}
ALL_MODEL_COLUMNS = {**BASE_MODEL_COLUMNS, "COMB": "pred_comb"}
HEADLINE_MODELS = ("OLS", "Random Forest", "COMB")


def fit_walk_forward(model_df: pd.DataFrame, cfg: Config):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

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

        # 1. OLS
        ols = LinearRegression()
        ols.fit(x_train_s, y_train)
        pred_ols = ols.predict(x_test_s)

        # 2. PLS
        pls = PLSRegression(n_components=min(cfg.pls_components, len(FEATURES)), scale=False, max_iter=500)
        pls.fit(x_train_s, y_train)
        pred_pls = np.asarray(pls.predict(x_test_s)).reshape(-1)

        # 3. LASSO
        lasso = Lasso(alpha=cfg.lasso_alpha, fit_intercept=True, max_iter=2500, tol=1e-3, random_state=cfg.random_state)
        lasso.fit(x_train_s, y_train)
        pred_lasso = lasso.predict(x_test_s)

        # 4. Elastic Net — course-required penalized regression
        enet = ElasticNet(
            alpha=cfg.enet_alpha, l1_ratio=cfg.enet_l1_ratio,
            fit_intercept=True, max_iter=2500, tol=1e-3,
            random_state=cfg.random_state, selection="cyclic"
        )
        enet.fit(x_train_s, y_train)
        pred_enet = enet.predict(x_test_s)

        # 5. Random Forest
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

        # 6. LightGBM — paper's GBRT family and course-required boosted-tree model
        lgb_model = lgb.LGBMRegressor(
            objective="regression", n_estimators=cfg.lgb_n_estimators,
            learning_rate=cfg.lgb_learning_rate, num_leaves=cfg.lgb_num_leaves,
            max_depth=cfg.lgb_max_depth, min_child_samples=cfg.lgb_min_child_samples,
            subsample=cfg.lgb_subsample, subsample_freq=1,
            colsample_bytree=cfg.lgb_colsample_bytree,
            reg_alpha=cfg.lgb_reg_alpha, reg_lambda=cfg.lgb_reg_lambda,
            random_state=cfg.random_state + fold, n_jobs=-1, verbosity=-1
        )
        lgb_model.fit(x_train, y_train)
        pred_lgb = lgb_model.predict(x_test)

        # 7. Feed-forward neural network
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

        # Paper-faithful structure: equal-weight average of all seven base forecasts.
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
            ols_coef_rows.append({
                "fold": fold, "feature": feature,
                "coefficient": float(coef), "abs_coefficient": float(abs(coef))
            })
        for feature, importance in zip(FEATURES, rf.feature_importances_):
            rf_imp_rows.append({
                "fold": fold, "feature": feature,
                "importance": float(importance)
            })

    if not prediction_parts:
        raise RuntimeError("No outer folds were fitted.")

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["date", "symbol"])
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(ols_coef_rows), pd.DataFrame(rf_imp_rows)


# ---------------------------------------------------------------------------
# Day-3 evaluation
# ---------------------------------------------------------------------------
def rank_ic_series(predictions: pd.DataFrame, pred_col: str, min_names: int) -> pd.Series:
    def one_date(g: pd.DataFrame) -> float:
        if len(g) < min_names or g["y_real"].nunique() <= 1 or g[pred_col].nunique() <= 1:
            return np.nan
        return float(g["y_real"].corr(g[pred_col], method="spearman"))
    return predictions.groupby("date", sort=True).apply(one_date).dropna()


def hac_tstat(series: pd.Series, maxlags: int = 7) -> tuple[float, float]:
    import statsmodels.api as sm
    y = series.dropna().to_numpy(float)
    if len(y) < 10:
        return float("nan"), float("nan")
    fit = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def hit_rate(predictions: pd.DataFrame, pred_col: str) -> float:
    temp = predictions[["date", "y_real", pred_col]].copy()
    temp["realized_centered"] = temp["y_real"] - temp.groupby("date")["y_real"].transform("mean")
    temp["pred_centered"] = temp[pred_col] - temp.groupby("date")[pred_col].transform("mean")
    valid = temp["realized_centered"].ne(0) & temp["pred_centered"].ne(0)
    return float((np.sign(temp.loc[valid, "realized_centered"]) == np.sign(temp.loc[valid, "pred_centered"])).mean())


def permutation_null_ic(predictions: pd.DataFrame, pred_col: str, min_names: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    temp = predictions[["date", "y_real", pred_col]].copy()
    temp["shuffled"] = temp.groupby("date")[pred_col].transform(lambda s: rng.permutation(s.to_numpy()))
    return float(rank_ic_series(temp, "shuffled", min_names).mean())


def evaluate(predictions: pd.DataFrame, cfg: Config):
    summary_rows: list[dict] = []
    ic_frames: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for model, col in ALL_MODEL_COLUMNS.items():
        ic = rank_ic_series(predictions, col, cfg.minimum_names_for_ic)
        mean_ic = float(ic.mean())
        sd_ic = float(ic.std(ddof=1))
        hac_t, hac_p = hac_tstat(ic)
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
                permutation_null_ic(
                    predictions, col, cfg.minimum_names_for_ic,
                    cfg.random_state + list(ALL_MODEL_COLUMNS).index(model)
                ) if model in HEADLINE_MODELS else np.nan
            ),
            "n_ic_dates": int(len(ic)),
            "oos_start": str(ic.index.min().date()),
            "oos_end": str(ic.index.max().date()),
        })
        ic_frames.append(pd.DataFrame({"date": ic.index, "model": model, "rank_ic": ic.values}))

    for fold, fold_data in predictions.groupby("outer_fold"):
        for model, col in ALL_MODEL_COLUMNS.items():
            ic = rank_ic_series(fold_data, col, cfg.minimum_names_for_ic)
            fold_rows.append({
                "fold": int(fold), "model": model,
                "headline": model in HEADLINE_MODELS,
                "mean_rank_ic": float(ic.mean()),
                "positive_ic_rate": float((ic > 0).mean()),
                "hit_rate": hit_rate(fold_data, col),
                "n_dates": int(len(ic)),
            })

    metrics = pd.DataFrame(summary_rows).sort_values("mean_rank_ic", ascending=False)
    ic_ts = pd.concat(ic_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    return metrics, ic_ts, fold_metrics


def make_plots(output_dir: Path, ic_timeseries: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headline = ic_timeseries[ic_timeseries["model"].isin(HEADLINE_MODELS)]
    pivot = headline.pivot(index="date", columns="model", values="rank_ic").sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.cumsum().plot(ax=ax)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Cumulative out-of-sample Rank IC: headline models")
    ax.set_ylabel("Cumulative Rank IC")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / "day3_ic_diagnostics.png", dpi=160)
    plt.close(fig)

    fold_headline = fold_metrics[fold_metrics["headline"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    fold_headline.pivot(index="fold", columns="model", values="mean_rank_ic").plot(kind="bar", ax=ax)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Mean Rank IC by walk-forward fold")
    ax.set_ylabel("Mean Rank IC")
    ax.set_xlabel("Outer fold")
    fig.tight_layout()
    fig.savefig(output_dir / "day3_fold_ic.png", dpi=160)
    plt.close(fig)


def write_markdown_summary(output_dir: Path, cfg: Config, metrics: pd.DataFrame, data_audit: pd.DataFrame) -> None:
    headline = metrics[metrics["headline"]].copy()
    display = headline[[
        "model", "mean_rank_ic", "ic_ir", "positive_ic_rate", "hit_rate",
        "permutation_null_mean_ic", "n_ic_dates", "oos_start", "oos_end"
    ]].copy()
    num = display.select_dtypes(include=[np.number]).columns
    display[num] = display[num].round(4)
    best = headline.sort_values("mean_rank_ic", ascending=False).iloc[0]

    survivor_flag = data_audit.loc[data_audit["check"].eq("all_remaining_symbols_reach_final_date"), "value"]
    survivor_warning = bool(len(survivor_flag) and int(survivor_flag.iloc[0]) == 1)

    text = f"""# MMF1927H Option 4 — Day 3 Modeling Summary

## Headline model decision

The presentation focuses on three paper-motivated outputs:

1. **OLS** — the paper reports the least-negative predictive out-of-sample R² and the lowest MAE.
2. **Random Forest** — the paper reports the highest average forecast correlation.
3. **COMB** — the paper's forecast combination, defined as the equal-weight average of all seven base-model forecasts.

COMB is an ensemble, not a separately fitted third algorithm. To reproduce its structure honestly, the code fits OLS, PLS, LASSO, Elastic Net, Random Forest, LightGBM/GBRT, and FFNN inside each walk-forward fold, then averages their forecasts. Elastic Net and LightGBM are therefore still fitted and evaluated, satisfying the explicit course requirement, while OLS, Random Forest, and COMB remain the presentation's three headline outputs.

## Target and validation

- Target: **{cfg.horizon}-day-ahead close-to-close return**.
- Features: known at date *t* and cross-sectionally ranked on that date.
- Universe: top-{cfg.top_n} eligible instruments by lagged {cfg.universe_lookback}-day dollar volume.
- Evaluation: {cfg.outer_folds} expanding walk-forward folds with a {cfg.embargo_days}-day embargo.
- Main Day-3 metrics: daily Rank IC, IC stability, positive-IC rate, and hit rate.

## Headline out-of-sample results

{display.to_markdown(index=False)}

The strongest headline mean Rank IC in this run is **{best['model']}** at **{best['mean_rank_ic']:.4f}**. This is our own sample result; it does not have to reproduce the paper's exact ranking because our universe, data source, features, and one-day horizon differ from the paper's weekly design.

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

{'**Survivorship warning:** the available candidate-symbol history appears to contain final-date survivors, so the project must not claim a fully survivorship-free universe.' if survivor_warning else 'Historical membership still needs to be documented from the sourcing process.'}
"""
    (output_dir / "DAY3_MODELING_SUMMARY.md").write_text(text, encoding="utf-8")


def write_requirements(output_dir: Path) -> None:
    import sklearn
    import lightgbm
    import statsmodels
    import matplotlib
    requirements = [
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"lightgbm=={lightgbm.__version__}",
        f"statsmodels=={statsmodels.__version__}",
        f"matplotlib=={matplotlib.__version__}",
    ]
    (output_dir / "requirements_day3.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="MMF1927H Option 4 Day-3 paper-model pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Cleaned JSON/CSV/Pickle/Parquet panel")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for Day-3 outputs")
    args = parser.parse_args()

    cfg = Config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print("[1/8] Loading input...")
    raw = load_input(args.input)
    print(f"      {len(raw):,} rows, {raw['symbol'].nunique():,} symbols")

    print("[2/8] Applying panel safeguards and rebuilding the universe...")
    panel, data_audit = prepare_analysis_panel(raw, cfg)

    print("[3/8] Building market return...")
    market = build_market_return(panel)

    print("[4/8] Engineering paper-mapped features...")
    featured = build_features(panel, market)

    print("[5/8] Creating the model frame...")
    model_df, model_audit = create_model_frame(featured, cfg)
    data_audit = pd.concat([data_audit, model_audit], ignore_index=True)
    print(f"      {len(model_df):,} rows across {model_df['date'].nunique():,} dates")

    print("[6/8] Fitting seven base models and constructing COMB...")
    predictions, folds, ols_coef, rf_imp = fit_walk_forward(model_df, cfg)
    print(f"      {len(predictions):,} OOS predictions across {predictions['date'].nunique():,} dates")

    print("[7/8] Computing Rank-IC and hit-rate diagnostics...")
    metrics, ic_ts, fold_metrics = evaluate(predictions, cfg)

    feature_dict = pd.DataFrame([{"feature": feature, **meta} for feature, meta in FEATURE_META.items()])

    print("[8/8] Writing outputs...")
    predictions.to_csv(args.output_dir / "day3_oos_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "day3_model_metrics.csv", index=False)
    metrics[metrics["headline"]].to_csv(args.output_dir / "day3_headline_metrics.csv", index=False)
    ic_ts.to_csv(args.output_dir / "day3_ic_timeseries.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "day3_fold_metrics.csv", index=False)
    folds.to_csv(args.output_dir / "day3_walk_forward_folds.csv", index=False)
    ols_coef.to_csv(args.output_dir / "day3_ols_coefficients_by_fold.csv", index=False)
    rf_imp.to_csv(args.output_dir / "day3_random_forest_importance_by_fold.csv", index=False)
    feature_dict.to_csv(args.output_dir / "day3_feature_dictionary.csv", index=False)
    data_audit.to_csv(args.output_dir / "day3_data_audit.csv", index=False)

    make_plots(args.output_dir, ic_ts, fold_metrics)
    write_markdown_summary(args.output_dir, cfg, metrics, data_audit)
    write_requirements(args.output_dir)

    print("\n=== HEADLINE DAY-3 OOS METRICS ===")
    print(metrics[metrics["headline"]].round(4).to_string(index=False))
    print("\n=== ALL MODEL OOS METRICS ===")
    print(metrics.round(4).to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
