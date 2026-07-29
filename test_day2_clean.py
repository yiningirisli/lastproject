import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import day2_clean


def row(day, close, quote_volume=100):
    open_time = day * 86_400_000
    return [
        open_time, str(close), str(close + 1), str(close - 1), str(close),
        "10", open_time + 86_399_999, str(quote_volume), 5, "4", "40", "0",
    ]


def write_symbol(root, symbol, rows):
    payload = {"provenance": {"source": "test"}, "klines": rows}
    (root / f"{symbol}.json").write_text(json.dumps(payload), encoding="utf-8")


class Day2Tests(unittest.TestCase):
    def test_load_symbol_schema_and_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_symbol(root, "AAAUSDT", [row(0, 10), row(1, 11)])
            frame, notes = day2_clean.load_symbol(root / "AAAUSDT.json")
            self.assertFalse(notes)
            self.assertEqual(len(frame), 2)
            self.assertTrue(pd.api.types.is_numeric_dtype(frame["close"]))

    def test_duplicate_rows_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_symbol(root, "AAAUSDT", [row(0, 10), row(0, 10)])
            frame, notes = day2_clean.load_symbol(root / "AAAUSDT.json")
            self.assertEqual(len(frame), 1)
            self.assertTrue(notes)

    def test_volume_rank_is_lagged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_symbol(root, "AAAUSDT", [row(i, 10 + i, 100) for i in range(5)])
            write_symbol(root, "BBBUSDT", [row(i, 20 + i, 50) for i in range(5)])
            panel, _ = day2_clean.build_panel(
                root, top_n=1, volume_window=2, min_history=2
            )
            selected = panel.loc[panel["in_universe"], "symbol"].unique().tolist()
            self.assertEqual(selected, ["AAAUSDT"])
            first_selected_day = panel.loc[panel["in_universe"], "date"].min()
            self.assertEqual(first_selected_day, pd.Timestamp("1970-01-03", tz="UTC"))

    def test_returns_are_not_imputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_symbol(root, "AAAUSDT", [row(0, 10), row(1, 11)])
            panel, _ = day2_clean.build_panel(
                root, top_n=1, volume_window=2, min_history=1
            )
            first = panel.sort_values("date").iloc[0]
            self.assertTrue(np.isnan(first["return_1d"]))
            self.assertTrue(first["return_missing"])

    def test_invalid_ohlc_is_dropped(self):
        bad = row(0, 10)
        bad[2] = "5"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_symbol(root, "AAAUSDT", [bad, row(1, 11)])
            frame, notes = day2_clean.load_symbol(root / "AAAUSDT.json")
            self.assertEqual(len(frame), 1)
            self.assertTrue(any("invalid OHLCV" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
