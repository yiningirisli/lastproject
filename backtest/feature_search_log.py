"""
Generate the Day-3 feature-search log from the feature metadata currently committed
in crypto_model.py, without inventing historical experiment results.

The repository does not preserve numerical search metrics for past rejected features.
Accordingly:
- every current production feature is recorded as `kept_current_model`;
- two alternatives explicitly documented in crypto_model.py are recorded as
  `alternative_not_implemented`;
- any real failed experiments can be appended from --manual-log.

This is deliberately auditable: blank result fields mean "not recoverable", not a
fabricated backfilled score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crypto_model import FEATURE_META  # noqa: E402


COLUMNS = [
    "feature",
    "status",
    "category",
    "source_paper",
    "formula_or_parameters",
    "result_metric",
    "result_value",
    "decision",
    "decision_reason",
    "experiment_date_utc",
    "git_commit",
    "notes",
]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def build_current_log() -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    commit = _git_commit()
    rows = []

    for feature, meta in FEATURE_META.items():
        rows.append({
            "feature": feature,
            "status": "kept_current_model",
            "category": meta.get("category", ""),
            "source_paper": meta.get("source_paper", ""),
            "formula_or_parameters": meta.get("formula", ""),
            "result_metric": "",
            "result_value": "",
            "decision": "KEEP",
            "decision_reason": (
                "Present in the current production feature set with documented economic rationale. "
                "Historical feature-search score is not preserved in the repository."
            ),
            "experiment_date_utc": now,
            "git_commit": commit,
            "notes": meta.get("implementation_note", ""),
        })

    # These alternatives are explicitly mentioned in the current feature metadata,
    # so recording them does not invent a failed experiment.
    rows.extend([
        {
            "feature": "market_cap_size",
            "status": "alternative_not_implemented",
            "category": "size_price",
            "source_paper": "P1/P2 size characteristic",
            "formula_or_parameters": "log(market capitalization)",
            "result_metric": "",
            "result_value": "",
            "decision": "DROP / SUBSTITUTE",
            "decision_reason": (
                "Market capitalization is unavailable in the Binance-only analysis panel; "
                "the current model explicitly uses log price as the documented analogue."
            ),
            "experiment_date_utc": now,
            "git_commit": commit,
            "notes": "Data-availability decision, not an empirical performance rejection.",
        },
        {
            "feature": "yang_zhang_volatility",
            "status": "alternative_not_implemented",
            "category": "risk",
            "source_paper": "P2 rvol",
            "formula_or_parameters": "Yang-Zhang realized-volatility estimator",
            "result_metric": "",
            "result_value": "",
            "decision": "DROP / SUBSTITUTE",
            "decision_reason": (
                "Current crypto_model.py explicitly implements a 30-day Parkinson range-volatility "
                "analogue rather than the paper's Yang-Zhang estimator."
            ),
            "experiment_date_utc": now,
            "git_commit": commit,
            "notes": "Methodological substitution documented in FEATURE_META.",
        },
    ])

    return pd.DataFrame(rows, columns=COLUMNS)


def write_feature_search_log(output: Path, manual_log: Path | None = None) -> pd.DataFrame:
    log = build_current_log()

    if manual_log is not None and manual_log.exists():
        manual = pd.read_csv(manual_log)
        for col in COLUMNS:
            if col not in manual.columns:
                manual[col] = ""
        log = pd.concat([log, manual[COLUMNS]], ignore_index=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(output, index=False)

    # Create a clean template for genuinely attempted/rejected experiments.
    template_path = output.with_name("feature_search_manual_template.csv")
    if not template_path.exists():
        pd.DataFrame([{
            "feature": "<actual attempted feature>",
            "status": "tested_rejected",
            "category": "",
            "source_paper": "",
            "formula_or_parameters": "",
            "result_metric": "mean_rank_ic",
            "result_value": "",
            "decision": "DROP",
            "decision_reason": "<why it was rejected>",
            "experiment_date_utc": "",
            "git_commit": "",
            "notes": "",
        }], columns=COLUMNS).to_csv(template_path, index=False)

    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate auditable Day-3 feature-search log.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backtest/pipeline_outputs/diagnostics/features/feature_search_log.csv"),
    )
    parser.add_argument(
        "--manual-log",
        type=Path,
        default=None,
        help="Optional CSV containing real historical tested/rejected feature experiments.",
    )
    args = parser.parse_args()

    log = write_feature_search_log(args.output, args.manual_log)
    print(f"Wrote {len(log)} feature-search rows to {args.output}")
    print(log[["feature", "status", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
