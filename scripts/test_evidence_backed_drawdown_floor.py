import numpy as np
import pandas as pd

import src.walk_forward_calibration as wfc


def main():
    peak = pd.Timestamp("2029-09-03")
    trough = pd.Timestamp("2030-09-02")
    dates = pd.date_range(peak, trough, freq="D")
    # Centerline rises materially during the bear year.
    center = np.exp(np.linspace(np.log(250000.0), np.log(315000.0), len(dates)))
    raw_dev = np.linspace(0.35, -0.35, len(dates))
    anchors = pd.DataFrame([
        {"date": peak, "type": "peak", "cycle": 1},
        {"date": trough, "type": "trough", "cycle": 2},
    ])
    draw_model = {
        "count": 2,
        "center_cycle": -1.5,
        "center_log_drawdown": np.log(1.4),
        "effective_slope": -0.08,
        "floor_confidence": 0.5,
    }
    floor, table = wfc._geometry_k_floor_schedule(
        dates, anchors, peak - pd.Timedelta(days=1), center, raw_dev, draw_model
    )
    assert len(table) == 1
    row = table.iloc[0]
    assert row["required_bear_drawdown_log"] > 0
    assert row["minimum_drawdown_K"] > row["minimum_geometric_K"]
    assert np.isclose(row["minimum_effective_K"], max(row["minimum_geometric_K"], row["minimum_drawdown_K"]), rtol=0, atol=1e-12)
    assert floor[0] >= row["minimum_effective_K"] - 1e-12

    k = float(row["minimum_effective_K"])
    p = center[0] * np.exp(k * max(raw_dev[0], 0.0))
    t = center[-1] * np.exp(k * min(raw_dev[-1], 0.0))
    dd = np.log(p / t)
    assert dd + 1e-8 >= float(row["required_bear_drawdown_log"])
    assert p > t
    print("Evidence-backed mature bear drawdown floor checks passed.")


if __name__ == "__main__":
    main()
