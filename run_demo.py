"""Synthetic demo for History Transfer (CIFEr original core).

Exercises the exact QLIKE loss, equal-market macro score, ridge cross-market
source weighting (excluding own-market leakage), and validation-year gamma
calibration on a synthetic multi-market panel.

NOTE: Synthetic data only — NOT an Oxford-Man 31-index empirical reproduction.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from residual_source_ridge import calibrate, macro, ql  # noqa: E402


def fit_synthetic_source_ridge(errors: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression of each market's OOS error on other markets' errors."""
    n_markets = errors.shape[1]
    mu = errors.mean(axis=0)
    sd = np.maximum(errors.std(axis=0), 0.05)
    Z = (errors - mu) / sd
    B = np.zeros((n_markets, n_markets))
    for i in range(n_markets):
        src = [j for j in range(n_markets) if j != i]
        X = Z[:, src]
        y = Z[:, i]
        b = np.linalg.solve(X.T @ X + alpha * np.eye(len(src)), X.T @ y)
        B[i, src] = b
    return B


def main() -> None:
    print("=== History Transfer (CIFEr Core) MVP (synthetic demo) ===")
    rng = np.random.default_rng(42)
    n_markets = 6

    # Synthetic prequential errors with shared latent shock across markets
    n_train, n_val, n_test = 120, 40, 40
    total = n_train + n_val + n_test
    common_shock = rng.normal(0.0, 0.35, size=(total, 1))
    errors = common_shock + rng.normal(0.0, 0.20, size=(total, n_markets))

    B = fit_synthetic_source_ridge(errors[:n_train], alpha=1.0)
    assert np.allclose(np.diag(B), 0.0), "Own-market diagonal must be zero"
    print(f"[1/3] Cross-market ridge weights fitted: shape={B.shape}, diag_max={np.abs(np.diag(B)).max():.1e}")

    # Build validation (year=2023) and test (year=2024) synthetic rows
    rows = []
    for idx, year in [(slice(n_train, n_train + n_val), 2023), (slice(n_train + n_val, total), 2024)]:
        E_seg = errors[idx]
        sig_seg = E_seg @ B.T
        for t in range(len(E_seg)):
            for m in range(n_markets):
                own_pred = 0.0
                y_true = float(E_seg[t, m])
                rows.append({"year": year, "market": m, "y": y_true, "own": own_pred, "signal": float(sig_seg[t, m])})
    df = pd.DataFrame(rows)

    (best_gamma, val_qlike), _ = calibrate(df, year=2024)
    te = df[df["year"] == 2024]
    own_qlike = macro(te["y"].to_numpy(), te["own"].to_numpy(), te["market"].to_numpy())
    adj_qlike = macro(
        te["y"].to_numpy(),
        te["own"].to_numpy() + best_gamma * te["signal"].to_numpy(),
        te["market"].to_numpy(),
    )
    print(f"[2/3] Calibrated multiplier on val split: gamma={best_gamma:.2f} (val macro QLIKE={val_qlike:.4f})")
    print(f"[3/3] Synthetic held-out macro QLIKE: own_only={own_qlike:.4f} | own+source={adj_qlike:.4f} (pointwise ql(0,0)={float(ql(0.0, 0.0)):.4f})")


if __name__ == "__main__":
    main()
