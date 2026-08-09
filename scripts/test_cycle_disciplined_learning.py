import numpy as np
import pandas as pd

import src.walk_forward_calibration as wfc


def main():
    # Three independent realized cycles with steadily shrinking realized envelope.
    rows = []
    for cycle, factor in [(-2, 0.95), (-1, 0.70), (0, 0.50)]:
        for ahead, typ in [(1, "peak"), (2, "trough")]:
            raw_amp = 0.60
            sign = 1.0 if typ == "peak" else -1.0
            center = 100.0
            actual = center * np.exp(sign * factor * raw_amp)
            rows.append({
                "cycle": cycle,
                "anchor_date": pd.Timestamp("2018-01-01") + pd.DateOffset(years=(cycle + 2) * 4, months=(ahead - 1) * 6),
                "anchor_type": typ,
                "fake_today": pd.Timestamp("2017-01-01") + pd.DateOffset(years=(cycle + 2) * 4),
                "raw_amplitude": raw_amp,
                "expected_sign": sign,
                "start_centerline_usd": 100.0,
                "raw_centerline_usd": center,
                "actual_anchor_price_usd": actual,
                "evidence_weight": 1.0,
                "turning_point_ahead": ahead,
                "cycle_horizon": 1,
            })
    envelope = pd.DataFrame(rows)
    points = wfc._amplitude_cycle_points(envelope)
    assert list(points["cycle_index"]) == [-2, -1, 0]
    assert len(points) == 3  # repeated forecasts are collapsed to realized cycles

    trend = wfc._fit_amplitude_trend(points)
    assert trend["direction"] == "DECLINING"
    assert wfc._trend_factor_at_cycle(trend, 1) < wfc._trend_factor_at_cycle(trend, 0)

    cv = wfc._cross_validate_amplitude_blend(points, envelope)
    assert 0.0 <= cv["blend_weight"] <= 1.0
    assert cv["calibrated"] < cv["raw"]
    assert cv["blend_weight"] > 0.0

    direct = wfc._direct_cycle_validation(cv["observation_scores"])
    assert direct and direct[0]["cycle_horizon"] == 1
    assert direct[0]["calibrated_error"] < direct[0]["raw_error"]

    # Structural correction trust is learned rather than forced.
    structural = pd.DataFrame([
        {"fake_today": pd.Timestamp(f"202{i}-01-01"), "actual_structural_log_growth": 0.8, "raw_structural_log_growth": 1.0, "implied_growth_factor": 0.8, "evidence_weight": 1.0}
        for i in range(1, 6)
    ])
    scv = wfc._cross_validate_structural_blend(structural)
    assert 0.0 <= scv["blend_weight"] <= 1.0
    assert scv["calibrated"] <= scv["raw"]

    print("Cycle-index trend, OOS blend weights, and direct cycle validation checks passed.")


if __name__ == "__main__":
    main()
