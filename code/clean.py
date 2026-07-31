"""Day 2: convert immutable Binance JSON pulls into an analysis-ready panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "n_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "n_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_symbol(path: Path) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("klines")
    provenance = payload.get("provenance")
    if not isinstance(rows, list) or not isinstance(provenance, dict):
        raise ValueError(f"{path.name}: invalid payload")
    if any(not isinstance(row, list) or len(row) != 12 for row in rows):
        raise ValueError(f"{path.name}: expected 12 fields per kline")

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame.insert(0, "symbol", path.stem)
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.normalize()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate = frame.duplicated(["symbol", "date"], keep=False)
    if duplicate.any():
        errors.append(f"{path.name}: removed {int(duplicate.sum())} duplicate rows")
        frame = frame.drop_duplicates(["symbol", "date"], keep="last")

    invalid = (
        frame[NUMERIC_COLUMNS].isna().any(axis=1)
        | (frame["open"] <= 0)
        | (frame["high"] <= 0)
        | (frame["low"] <= 0)
        | (frame["close"] <= 0)
        | (frame["volume"] < 0)
        | (frame["quote_asset_volume"] < 0)
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["high"] < frame["low"])
    )
    if invalid.any():
        errors.append(f"{path.name}: dropped {int(invalid.sum())} invalid OHLCV rows")
        frame = frame.loc[~invalid].copy()

    frame = frame.sort_values("date")
    if not frame["date"].is_monotonic_increasing:
        raise AssertionError(f"{path.name}: date ordering failed")
    return frame, errors


def winsorize_by_date(
    frame: pd.DataFrame, column: str, lower: float, upper: float
) -> pd.Series:
    grouped = frame.groupby("date", observed=True)[column]
    lo = grouped.transform("quantile", lower)
    hi = grouped.transform("quantile", upper)
    return frame[column].clip(lo, hi)


def robust_z_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("date", observed=True)[column]
    median = grouped.transform("median")
    absolute_deviation = (frame[column] - median).abs()
    mad = absolute_deviation.groupby(frame["date"]).transform("median")
    denominator = 1.4826 * mad
    return ((frame[column] - median) / denominator).where(denominator > 0)


def build_panel(
    raw_dir: Path,
    top_n: int = 50,
    volume_window: int = 30,
    min_history: int = 30,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted(raw_dir.glob("*USDT.json"))
    if not paths:
        raise ValueError(f"no symbol JSON files found in {raw_dir}")

    frames: list[pd.DataFrame] = []
    validation_notes: list[str] = []
    hashes: dict[str, str] = {}
    for path in paths:
        frame, notes = load_symbol(path)
        frames.append(frame)
        validation_notes.extend(notes)
        hashes[path.name] = sha256(path)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    by_symbol = panel.groupby("symbol", observed=True, sort=False)
    panel["history_days"] = by_symbol.cumcount() + 1
    panel["return_1d"] = by_symbol["close"].pct_change(fill_method=None)
    panel["log_return_1d"] = np.log(panel["close"]).groupby(panel["symbol"]).diff()

    # Membership at date t is based only on information available through t-1.
    shifted_quote_volume = by_symbol["quote_asset_volume"].shift(1)
    panel["trailing_quote_volume"] = (
        shifted_quote_volume.groupby(panel["symbol"])
        .rolling(volume_window, min_periods=volume_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    panel["eligible_history"] = panel["history_days"] > min_history
    panel["volume_rank"] = panel["trailing_quote_volume"].groupby(
        panel["date"]
    ).rank(method="first", ascending=False)
    panel["in_universe"] = (
        panel["eligible_history"]
        & panel["trailing_quote_volume"].notna()
        & (panel["volume_rank"] <= top_n)
    )

    # Returns are never imputed. Keep a flag and retain NaN at listing boundaries.
    panel["return_missing"] = panel["return_1d"].isna()
    panel["return_1d_winsor"] = winsorize_by_date(
        panel, "return_1d", winsor_lower, winsor_upper
    )
    panel["return_1d_robust_z"] = robust_z_by_date(panel, "return_1d_winsor")

    # Day 3 target: next observed daily return; last observation remains missing.
    panel["target_return_1d"] = by_symbol["return_1d"].shift(-1)

    output_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "n_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "return_1d",
        "log_return_1d",
        "return_missing",
        "return_1d_winsor",
        "return_1d_robust_z",
        "history_days",
        "trailing_quote_volume",
        "volume_rank",
        "eligible_history",
        "in_universe",
        "target_return_1d",
    ]
    panel = panel[output_columns].sort_values(["date", "symbol"]).reset_index(drop=True)

    in_universe = panel["in_universe"]
    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "start_date": panel["date"].min().isoformat(),
        "end_date": panel["date"].max().isoformat(),
        "duplicate_symbol_dates": int(panel.duplicated(["symbol", "date"]).sum()),
        "missing_returns": int(panel["return_missing"].sum()),
        "universe_rows": int(in_universe.sum()),
        "universe_dates": int(panel.loc[in_universe, "date"].nunique()),
        "max_universe_size": int(panel.groupby("date")["in_universe"].sum().max()),
        "validation_notes": validation_notes,
        "parameters": {
            "top_n": top_n,
            "volume_window": volume_window,
            "min_history": min_history,
            "winsor_lower": winsor_lower,
            "winsor_upper": winsor_upper,
            "membership_information_lag_days": 1,
        },
        "known_limitations": [
            "The Day 1 exchangeInfo snapshot cannot recover pairs delisted before "
            "the pull date; residual survivorship bias remains.",
            "Binance current-symbol metadata is not a complete point-in-time "
            "listing and delisting membership table.",
            "Missing returns are left missing rather than imputed because listing "
            "boundaries and trading interruptions are not MCAR.",
        ],
        "raw_sha256": hashes,
    }
    return panel, diagnostics


def write_memo(path: Path, diagnostics: dict[str, Any]) -> None:
    p = diagnostics["parameters"]
    notes = diagnostics["validation_notes"]
    text = f"""# Day 2 Data-Quality Memo

The cleaned panel contains {diagnostics['rows']:,} rows and
{diagnostics['symbols']} symbols from {diagnostics['start_date'][:10]} through
{diagnostics['end_date'][:10]}. It contains
{diagnostics['duplicate_symbol_dates']} duplicate symbol-date rows after
validation.

## Missingness

Daily returns at listing boundaries or after unavailable prior observations are
left missing; {diagnostics['missing_returns']:,} rows carry `return_missing=True`.
No backward interpolation or future observation is used. Crypto missingness is
often listing-, outage-, or liquidity-related and therefore not assumed MCAR.

## Universe

The per-date universe contains the top {p['top_n']} eligible symbols ranked by
the trailing {p['volume_window']}-day mean USDT quote volume. Volume is shifted
by {p['membership_information_lag_days']} day before ranking, so date-t
membership uses only information known by t-1. A symbol needs more than
{p['min_history']} observed days before eligibility. The resulting panel has
{diagnostics['universe_rows']:,} universe rows over
{diagnostics['universe_dates']:,} dates.

## Outliers

Invalid OHLC relationships, non-positive prices, negative volume, malformed
rows, and duplicate symbol-dates are treated as data errors. Valid return tails
are retained in `return_1d`; a separate cross-sectional winsorized series caps
returns at p{p['winsor_lower'] * 100:g}/p{p['winsor_upper'] * 100:g} per date.
A MAD-based robust z-score is also supplied without overwriting raw returns.

## Point-in-time and residual bias

All rolling volume inputs are lagged and targets are stored separately. However,
the Day 1 Binance `exchangeInfo` snapshot cannot recover coins delisted before
the pull date. Residual survivorship bias therefore remains and likely biases
historical performance upward. The current public raw source is insufficient to
claim that this bias has been eliminated.

## Validation notes

{chr(10).join('- ' + note for note in notes) if notes else '- No malformed, duplicate, or invalid OHLCV rows were detected.'}
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Day 2 crypto panel")
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/clean"))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--volume-window", type=int, default=30)
    parser.add_argument("--min-history", type=int, default=30)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n <= 0 or args.volume_window <= 1 or args.min_history < 1:
        raise SystemExit("top-n, volume-window, and min-history must be positive")
    if not 0 <= args.winsor_lower < args.winsor_upper <= 1:
        raise SystemExit("winsor thresholds must satisfy 0 <= lower < upper <= 1")

    panel, diagnostics = build_panel(
        args.raw_dir,
        top_n=args.top_n,
        volume_window=args.volume_window,
        min_history=args.min_history,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    panel_path = args.out / "crypto_analysis_ready.parquet"
    panel.to_parquet(panel_path, index=False, compression="zstd")

    manifest = {
        "artifact": panel_path.name,
        "artifact_sha256": sha256(panel_path),
        "cleaning_script": Path(__file__).name,
        "cleaning_script_sha256": sha256(Path(__file__)),
        "git_commit": git_commit(Path(__file__).resolve().parent),
        **diagnostics,
    }
    (args.out / "day2_lineage_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_memo(args.out / "DAY2_DATA_QUALITY_MEMO.md", diagnostics)
    print(json.dumps({k: v for k, v in diagnostics.items() if k != "raw_sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
