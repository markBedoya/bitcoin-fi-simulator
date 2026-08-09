import numpy as np
import pandas as pd

from src.walk_forward_calibration import (
    AMPLITUDE_FACTOR_MAX,
    AMPLITUDE_FACTOR_MIN,
    CALIBRATION_FLOOR,
    _snapshot_amplitude_factor,
    _snapshot_growth_factor,
    first_fake_today_for_lookback,
    generate_fake_today_dates,
)


def main():
    # The hard Jan-2015 floor produces a true 4Y test beginning in Jan-2019 and
    # a true 8Y test beginning in Jan-2023.  Both require at least 12 months of
    # realized future data, so Aug-2026 data ends with a Jul-2025 fake today.
    dates = pd.date_range(CALIBRATION_FLOOR, "2026-08-07", freq="D")
    prices = pd.DataFrame({"date": dates, "price_usd": np.linspace(200, 60000, len(dates))})
    fake4 = generate_fake_today_dates(prices, 4)
    fake8 = generate_fake_today_dates(prices, 8)
    assert fake4[0] == pd.Timestamp("2019-01-14")
    assert fake8[0] == pd.Timestamp("2023-01-14")
    assert fake4[-1] == pd.Timestamp("2025-07-14")
    assert fake8[-1] == pd.Timestamp("2025-07-14")
    assert first_fake_today_for_lookback(4) - pd.DateOffset(years=4) == CALIBRATION_FLOOR
    assert first_fake_today_for_lookback(8) - pd.DateOffset(years=8) == CALIBRATION_FLOOR

    # Multi-horizon structural evidence should be able to move G materially
    # away from 1 when several long-horizon observations agree.
    structural = pd.DataFrame({
        "implied_growth_factor": [0.70, 0.72, 0.68, 0.71],
        "evidence_weight": [1.0, 2.0, 3.0, 4.0],
    })
    G = _snapshot_growth_factor(structural)
    assert 0.66 <= G <= 0.76, G

    # Cycle-envelope K is learned from realized turning-point amplitude around
    # the G-adjusted centerline, not from ordinary daily price observations.
    c0 = 100.0
    raw_center = 200.0
    raw_amp = np.log(1.50)
    G_true = 0.80
    K_true = 0.60
    cal_center = c0 * (raw_center / c0) ** G_true
    actual_peak = cal_center * np.exp(K_true * raw_amp)
    envelope = pd.DataFrame([
        {
            "start_centerline_usd": c0,
            "raw_centerline_usd": raw_center,
            "actual_anchor_price_usd": actual_peak,
            "expected_sign": 1.0,
            "raw_amplitude": raw_amp,
            "evidence_weight": 12.0,
        },
        {
            "start_centerline_usd": c0,
            "raw_centerline_usd": raw_center,
            "actual_anchor_price_usd": cal_center / np.exp(K_true * raw_amp),
            "expected_sign": -1.0,
            "raw_amplitude": raw_amp,
            "evidence_weight": 12.0,
        },
    ])
    K = _snapshot_amplitude_factor(envelope, G_true)
    assert AMPLITUDE_FACTOR_MIN <= K <= AMPLITUDE_FACTOR_MAX
    assert abs(K - K_true) < 0.08, (K, K_true)

    print("Cycle-aware multi-horizon walk-forward calibration checks passed.")


if __name__ == "__main__":
    main()
