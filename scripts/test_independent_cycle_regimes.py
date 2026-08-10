import numpy as np
import pandas as pd

import src.walk_forward_calibration as wfc


def _row(fake, date, typ, cycle, start_center, raw_center, raw_price, actual, months, cycle_horizon=1):
    sign = 1.0 if typ == "peak" else -1.0
    raw_amp = max(sign * np.log(raw_price / raw_center), 0.0)
    return {
        "fake_today": pd.Timestamp(fake),
        "anchor_date": pd.Timestamp(date),
        "anchor_type": typ,
        "cycle": int(cycle),
        "start_centerline_usd": float(start_center),
        "raw_centerline_usd": float(raw_center),
        "raw_projected_anchor_price_usd": float(raw_price),
        "actual_anchor_price_usd": float(actual),
        "raw_amplitude": float(raw_amp),
        "expected_sign": sign,
        "evidence_weight": 1.0,
        "months_forward": float(months),
        "cycle_horizon": int(cycle_horizon),
        "turning_point_ahead": 1,
    }


def main():
    rows = []
    # Complete 2015->2018 regime: predicted cycle is wider than what occurred.
    for fake in ("2016-01-01", "2016-07-01"):
        rows.append(_row(fake, "2017-12-17", "peak", -2, 20, 50, 120, 100, 24))
        rows.append(_row(fake, "2018-12-15", "trough", -1, 20, 65, 25, 18, 36))
    # Complete 2018->2022 regime.
    for fake in ("2019-01-01", "2020-01-01"):
        rows.append(_row(fake, "2021-11-08", "peak", -1, 40, 130, 320, 200, 24))
        rows.append(_row(fake, "2022-11-07", "trough", 0, 40, 180, 70, 60, 36))
    # Current 2022 regime has a completed peak but no confirmed following trough.
    for fake in ("2023-01-01", "2024-01-01"):
        rows.append(_row(fake, "2025-10-06", "peak", 0, 80, 260, 600, 400, 24))

    envelope = pd.DataFrame(rows)
    regimes = wfc._cycle_regime_points(envelope)
    assert list(regimes["regime_index"]) == [-2, -1, 0]
    assert regimes["complete"].tolist() == [True, True, False]
    assert regimes.loc[regimes["regime_index"] == 0, "realized_turning_points"].iloc[0] == 1

    fit = wfc._fit_constant_k_from_regimes(regimes, conservative_one_se=True)
    # Direct peak/trough + drawdown scoring should produce a meaningful finite K,
    # while small-sample shrinkage keeps it closer to 1 than the raw optimum.
    assert wfc.AMPLITUDE_FACTOR_MIN < fit["best_factor"] < 1.0, fit
    assert fit["best_factor"] < fit["factor"] <= 1.0, fit
    assert fit["sample_confidence"] < 1.0
    assert wfc._score_cycle_regimes(regimes, fit["factor"]) < wfc._score_cycle_regimes(regimes, 1.0)

    points = wfc._regime_k_points(regimes)
    assert len(points) == 2, points
    assert set(points["cycle_index"]) == {-2, -1}

    draw = wfc._fit_drawdown_maturity(regimes)
    assert draw["count"] == 2
    assert 0.0 < draw["floor_confidence"] < 1.0
    expected, required = wfc._drawdown_requirement_at_cycle(draw, 1)
    assert expected > 0 and 0 < required < expected

    print("Independent cycle-regime K learning and partial-current-cycle evidence checks passed.")


if __name__ == "__main__":
    main()
