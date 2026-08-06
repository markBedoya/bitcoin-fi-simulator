from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.financial_independence import build_rebased_btc_paths


def main() -> None:
    latest_actual = 100.0
    daily = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4, freq="D"),
        "row_type": ["historical_training", "projected", "projected", "projected"],
        "structural_centerline_usd": [90.0, 150.0, 165.0, 180.0],
        "fitted_or_projected_price_usd": [95.0, 200.0, 220.0, 240.0],
    })

    anchored = build_rebased_btc_paths(daily, latest_actual)

    # Canonical anchoring must be derived from the final historical fitted value,
    # not independently from the first future market or centerline point.
    common_scale = latest_actual / 95.0
    assert abs(anchored["btc_cycle_price"].iloc[0] - 200.0 * common_scale) < 1e-9
    assert abs(anchored["btc_centerline_price"].iloc[0] - 150.0 * common_scale) < 1e-9
    assert abs(anchored["btc_cycle_price"].iloc[-1] - 240.0 * common_scale) < 1e-9
    assert abs(anchored["btc_centerline_price"].iloc[-1] - 180.0 * common_scale) < 1e-9

    # Both projected lines must have exactly the same scale factor.
    cycle_scale = anchored["btc_cycle_price"].iloc[-1] / 240.0
    center_scale = anchored["btc_centerline_price"].iloc[-1] / 180.0
    assert abs(cycle_scale - center_scale) < 1e-12

    print("Projection consistency test passed.")


if __name__ == "__main__":
    main()
