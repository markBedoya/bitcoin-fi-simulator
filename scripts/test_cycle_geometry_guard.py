import numpy as np
import pandas as pd

import src.walk_forward_calibration as wfc
from src.price_model import NEXT_TROUGH, PriceModelResult


def main():
    training_end = pd.Timestamp("2026-08-07")
    peak = pd.Timestamp("2029-09-03")
    trough2 = pd.Timestamp("2030-09-02")
    dates = pd.date_range(training_end, trough2, freq="D")
    years = (dates - training_end).days.to_numpy(dtype=float) / 365.25
    # Fast centerline growth deliberately creates the inversion problem when K is tiny.
    center = 100000 * np.exp(0.30 * years)
    dev = np.zeros(len(dates))
    t0 = pd.Timestamp(NEXT_TROUGH)
    t0i = int(np.where(dates == t0)[0][0])
    pi = int(np.where(dates == peak)[0][0])
    ti = int(np.where(dates == trough2)[0][0])
    dev[:t0i + 1] = np.linspace(0, -0.40, t0i + 1)
    dev[t0i:pi + 1] = np.linspace(-0.40, 0.40, pi - t0i + 1)
    dev[pi:ti + 1] = np.linspace(0.40, -0.40, ti - pi + 1)
    raw = center * np.exp(dev)
    daily = pd.DataFrame({
        "date": dates,
        "row_type": np.where(dates <= training_end, "historical_training", "projected"),
        "actual_price_usd": np.nan,
        "structural_centerline_usd": center,
        "fitted_or_projected_price_usd": raw,
    })
    anchors = pd.DataFrame([
        {"date": t0, "type": "trough", "cycle": 1},
        {"date": peak, "type": "peak", "cycle": 1},
        {"date": trough2, "type": "trough", "cycle": 2},
    ])
    base = PriceModelResult(daily, {
        "training_start": "2018-12-15",
        "training_end": training_end.date().isoformat(),
        "projection_end_date": trough2,
        "projection_years": 5,
        "cycle_anchor_table": anchors,
        "cycle_anchor_lookahead_table": pd.DataFrame(),
    }, pd.DataFrame(), pd.DataFrame())
    cal = wfc.WalkForwardCalibrationResult(summary={
        "effective_growth_factor": 1.0,
        "structural_blend_weight": 0.0,
        "amplitude_factor": 0.10,
        "amplitude_constant_factor": 0.10,
        "amplitude_trend_blend_weight": 0.0,
        "amplitude_mode": "CONSTANT",
        "current_cycle_index": 0,
        "status": "PASS",
        "version": wfc.CALIBRATION_VERSION,
    }, tests=pd.DataFrame(), observations=pd.DataFrame(), fingerprint="x")

    out = wfc.build_calibrated_price_model(base, cal)
    turns = out.turning_points.set_index("date")
    p = turns.loc[peak]
    t = turns.loc[trough2]
    assert p["unconstrained_amplitude_factor_K"] < p["minimum_geometric_K"]
    assert p["geometry_constrained"]
    assert t["geometry_constrained"]
    assert p["calibrated_price_usd"] > t["calibrated_price_usd"]
    assert out.diagnostics["geometry_valid"] is True
    print("Mathematical peak/trough geometry guard checks passed.")


if __name__ == "__main__":
    main()
