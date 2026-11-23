"""Core unit tests for History Transfer (CIFEr) MVP."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from residual_source_ridge import calibrate, macro, ql  # noqa: E402


class HistoryTransferMVPTests(unittest.TestCase):
    def test_qlike_and_macro_properties(self) -> None:
        self.assertAlmostEqual(float(ql(0.0, 0.0)), 0.0, places=8)
        self.assertGreater(float(ql(0.5, 0.0)), 0.0)
        score = macro(np.array([0.2, 0.2]), np.array([0.2, 0.2]), np.array([0, 1]))
        self.assertAlmostEqual(score, 0.0, places=8)

    def test_calibrate_finds_positive_gamma_on_correlated_signal(self) -> None:
        rng = np.random.default_rng(11)
        sig = rng.normal(0.0, 0.4, size=60)
        y = 0.5 * sig + rng.normal(0.0, 0.05, size=60)
        df = pd.DataFrame({
            "year": [2023] * 60,
            "market": [i % 3 for i in range(60)],
            "y": y,
            "own": np.zeros(60),
            "signal": sig,
        })
        (best_g, best_score), grid = calibrate(df, year=2024)
        self.assertGreater(best_g, 0.0)
        self.assertLess(best_score, macro(y, np.zeros(60), df["market"].to_numpy()))
        self.assertEqual(len(grid), 41)


if __name__ == "__main__":
    unittest.main()
