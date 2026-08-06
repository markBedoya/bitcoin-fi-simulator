from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

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

    assert abs(anchored["btc_cycle_price"].iloc[0] - latest_actual) < 1e-9

    # Both lines must use the same anchor factor. The centerline is not forced
    # independently to the actual price because that would distort geometry.
    common_scale = latest_actual / 200.0
    assert abs(anchored["btc_centerline_price"].iloc[0] - 150.0 * common_scale) < 1e-9

    expected_center_last = 180.0 * common_scale
    expected_cycle_last = 240.0 * common_scale

    assert abs(anchored["btc_centerline_price"].iloc[-1] - expected_center_last) < 1e-9
    assert abs(anchored["btc_cycle_price"].iloc[-1] - expected_cycle_last) < 1e-9

    print("Projection consistency test passed.")


if __name__ == "__main__":
    main()
