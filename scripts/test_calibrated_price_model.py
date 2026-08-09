import numpy as np
import pandas as pd

from src.price_model import NEXT_TROUGH, PriceModelResult
from src.walk_forward_calibration import (
    WalkForwardCalibrationResult,
    build_calibrated_price_model,
)


def main():
    training_end = pd.Timestamp("2026-08-07")
    dates = pd.date_range(training_end, "2030-09-02", freq="D")
    row_type = np.where(dates == training_end, "historical_training", "projected")

    # Smooth raw centerline.
    years = (dates - training_end).days.to_numpy(dtype=float) / 365.25
    center = 100000 * np.exp(0.22 * years)

    peak = pd.Timestamp("2029-09-03")
    next_trough = pd.Timestamp("2030-09-02")
    trough = NEXT_TROUGH
    A = 0.55
    raw_price = center.copy()

    # Create monotone phase paths matching a symmetric raw envelope after the
    # live Oct-2026 trough. The exact interior shape is not important here.
    trough_i = int(np.where(dates == trough)[0][0])
    peak_i = int(np.where(dates == peak)[0][0])
    next_trough_i = int(np.where(dates == next_trough)[0][0])
    raw_price[:trough_i + 1] = np.linspace(65000, center[trough_i] * np.exp(-0.45), trough_i + 1)
    bull_progress = np.linspace(0, 1, peak_i - trough_i + 1)
    raw_price[trough_i:peak_i + 1] = np.exp(
        np.log(raw_price[trough_i])
        + bull_progress * (np.log(center[peak_i] * np.exp(A)) - np.log(raw_price[trough_i]))
    )
    bear_progress = np.linspace(0, 1, next_trough_i - peak_i + 1)
    raw_price[peak_i:next_trough_i + 1] = np.exp(
        np.log(center[peak_i] * np.exp(A))
        + bear_progress * (np.log(center[next_trough_i] * np.exp(-A)) - np.log(center[peak_i] * np.exp(A)))
    )

    daily = pd.DataFrame({
        "date": dates,
        "row_type": row_type,
        "actual_price_usd": np.where(dates == training_end, 65000.0, np.nan),
        "structural_centerline_usd": center,
        "fitted_or_projected_price_usd": raw_price,
    })
    anchors = pd.DataFrame([
        {"date": trough, "type": "trough", "cycle": 1, "knot_price_usd": raw_price[trough_i], "structural_centerline_usd": center[trough_i], "source": "current bear-conditioned projected trough (live-cycle exception)"},
        {"date": peak, "type": "peak", "cycle": 1, "knot_price_usd": raw_price[peak_i], "structural_centerline_usd": center[peak_i], "source": "projected symmetric cycle envelope around locked structural centerline"},
        {"date": next_trough, "type": "trough", "cycle": 2, "knot_price_usd": raw_price[next_trough_i], "structural_centerline_usd": center[next_trough_i], "source": "projected symmetric cycle envelope around locked structural centerline"},
    ])
    base = PriceModelResult(
        daily=daily,
        diagnostics={
            "training_end": training_end.date().isoformat(),
            "projection_end_date": dates[-1],
            "model_version": "synthetic-v3.12",
            "cycle_anchor_table": anchors,
            "cycle_anchor_lookahead_table": pd.DataFrame(),
        },
        cycle_overlays=pd.DataFrame(),
        cycle_template=pd.DataFrame(),
    )
    calibration = WalkForwardCalibrationResult(
        summary={
            "growth_factor": 0.80,
            "amplitude_factor": 0.70,
            "status": "PASS",
            "version": "walk-forward-calibration-v1.0",
        },
        tests=pd.DataFrame(),
        observations=pd.DataFrame(),
        fingerprint="synthetic-calibration",
    )

    out = build_calibrated_price_model(base, calibration)
    result = out.daily.set_index("date")

    # Current live bear trough remains exactly the frozen-model price.
    assert abs(result.loc[trough, "calibrated_price_usd"] / result.loc[trough, "raw_price_usd"] - 1) < 1e-12

    # Future centerline growth is re-estimated with G but remains continuous at
    # the calibration boundary.
    c0 = float(result.loc[trough, "raw_centerline_usd"])
    raw_peak_center = float(result.loc[peak, "raw_centerline_usd"])
    expected_peak_center = c0 * (raw_peak_center / c0) ** 0.80
    assert abs(result.loc[peak, "calibrated_centerline_usd"] / expected_peak_center - 1) < 1e-12

    # At a complete future peak, cycle amplitude is the frozen symmetric
    # amplitude multiplied by K.
    raw_amp = np.log(result.loc[peak, "raw_price_usd"] / result.loc[peak, "raw_centerline_usd"])
    cal_amp = np.log(result.loc[peak, "calibrated_price_usd"] / result.loc[peak, "calibrated_centerline_usd"])
    assert abs(cal_amp - 0.70 * raw_amp) < 1e-10

    assert out.diagnostics["live_cycle_preserved_through_calibration_start"] is True
    print("Calibrated centerline/amplitude transformation checks passed.")


if __name__ == "__main__":
    main()
