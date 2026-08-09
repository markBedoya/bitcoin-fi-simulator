import numpy as np
import pandas as pd

import src.walk_forward_calibration as wfc
from src.price_model import NEXT_TROUGH, PriceModelResult


def _synthetic_parent(start_date, weight, center_growth=0.20, amplitude=0.50):
    training_end = pd.Timestamp("2026-08-07")
    dates = pd.date_range(pd.Timestamp(start_date), "2030-09-02", freq="D")
    years = (dates - training_end).days.to_numpy(dtype=float) / 365.25
    center = 100000 * np.exp(center_growth * years)
    dev = np.zeros(len(dates), dtype=float)
    t0 = pd.Timestamp(NEXT_TROUGH)
    p1 = pd.Timestamp("2029-09-03")
    t1 = pd.Timestamp("2030-09-02")
    for i, d in enumerate(dates):
        if d <= t0:
            dev[i] = -0.35 * max((d - training_end).days, 0) / max((t0 - training_end).days, 1)
        elif d <= p1:
            q = (d - t0).days / max((p1 - t0).days, 1)
            dev[i] = -0.35 + q * (amplitude + 0.35)
        elif d <= t1:
            q = (d - p1).days / max((t1 - p1).days, 1)
            dev[i] = amplitude + q * (-2 * amplitude)
        else:
            dev[i] = -amplitude
    price = center * np.exp(dev)
    daily = pd.DataFrame({
        "date": dates,
        "row_type": np.where(dates <= training_end, "historical_training", "projected"),
        "actual_price_usd": np.nan,
        "structural_centerline_usd": center,
        "fitted_or_projected_price_usd": price,
    })
    anchors = pd.DataFrame([
        {"date": t0, "type": "trough", "cycle": 1, "source": "live"},
        {"date": p1, "type": "peak", "cycle": 1, "source": "future"},
        {"date": t1, "type": "trough", "cycle": 2, "source": "future"},
    ])
    model = PriceModelResult(
        daily=daily,
        diagnostics={
            "training_start": pd.Timestamp(start_date).date().isoformat(),
            "training_end": training_end.date().isoformat(),
            "projection_end_date": dates[-1],
            "projection_years": 5,
            "model_version": "synthetic-v3.12",
            "cycle_anchor_table": anchors,
            "cycle_anchor_lookahead_table": pd.DataFrame(),
        },
        cycle_overlays=pd.DataFrame(), cycle_template=pd.DataFrame(),
    )
    return pd.Timestamp(start_date), weight, model


def main():
    dates = pd.date_range("2015-01-14", "2026-08-07", freq="D")
    prices = pd.DataFrame({"date": dates, "price_usd": np.exp(np.linspace(np.log(200), np.log(65000), len(dates)))})
    starts = wfc.discover_cycle_aligned_parent_starts(prices)
    assert pd.Timestamp("2015-01-14") in starts
    assert pd.Timestamp("2018-12-15") in starts
    assert pd.Timestamp("2022-11-07") in starts

    # Trend is now explicitly cycle-index based.
    points = pd.DataFrame({
        "cycle_index": [-2, -1, 0],
        "date": pd.to_datetime(["2017-12-17", "2021-11-08", "2025-10-06"]),
        "factor": [0.90, 0.66, 0.48],
        "weight": [2.0, 2.0, 2.0],
    })
    trend = wfc._fit_amplitude_trend(points)
    assert trend["direction"] == "DECLINING", trend
    k0 = wfc._trend_factor_at_cycle(trend, 0)
    k1 = wfc._trend_factor_at_cycle(trend, 1)
    k2 = wfc._trend_factor_at_cycle(trend, 2)
    assert k2 < k1 < k0 < 1.0, (k0, k1, k2)

    parents = [
        _synthetic_parent("2015-01-14", 0.30, 0.18, 0.55),
        _synthetic_parent("2018-12-15", 0.45, 0.20, 0.50),
        _synthetic_parent("2022-11-07", 0.25, 0.16, 0.42),
    ]
    original = wfc._fit_current_parent_models
    wfc._fit_current_parent_models = lambda prices, calibration, projection_years: parents
    try:
        cal = wfc.WalkForwardCalibrationResult(
            summary={
                "effective_growth_factor": 1.0,
                "structural_blend_weight": 0.0,
                "amplitude_factor": 0.55,
                "amplitude_constant_factor": 0.60,
                "amplitude_trend_blend_weight": 0.70,
                "amplitude_mode": "BLENDED_TREND",
                "amplitude_trend_direction": "DECLINING",
                "amplitude_trend_center_cycle": 0.0,
                "amplitude_trend_center_log_factor": float(np.log(0.55)),
                "amplitude_trend_effective_log_slope_per_cycle": -0.08,
                "current_cycle_index": 0,
                "status": "PASS",
                "version": wfc.CALIBRATION_VERSION,
                "cycle_parents": [
                    {"start_date": "2015-01-14", "weight": 0.30},
                    {"start_date": "2018-12-15", "weight": 0.45},
                    {"start_date": "2022-11-07", "weight": 0.25},
                ],
            },
            tests=pd.DataFrame(), observations=pd.DataFrame(), fingerprint="synthetic",
        )
        dummy_prices = pd.DataFrame({
            "date": pd.date_range("2026-08-01", "2026-08-07", freq="D"),
            "price_usd": np.linspace(64000, 65000, 7),
        })
        base_a = parents[0][2]
        base_b = PriceModelResult(
            daily=base_a.daily.copy(),
            diagnostics={**base_a.diagnostics, "training_start": "2018-11-28"},
            cycle_overlays=pd.DataFrame(), cycle_template=pd.DataFrame(),
        )
        out_a = wfc.build_calibrated_price_model(base_a, cal, prices=dummy_prices)
        out_b = wfc.build_calibrated_price_model(base_b, cal, prices=dummy_prices)
        pa = out_a.daily.loc[out_a.daily["row_type"] == "projected", "calibrated_price_usd"].to_numpy()
        pb = out_b.daily.loc[out_b.daily["row_type"] == "projected", "calibrated_price_usd"].to_numpy()
        assert np.allclose(pa, pb, rtol=0, atol=1e-9, equal_nan=True)
        assert out_a.diagnostics["geometry_valid"] is True
    finally:
        wfc._fit_current_parent_models = original

    print("Dynamic cycle-parent ensemble and cycle-index maturity checks passed.")


if __name__ == "__main__":
    main()
