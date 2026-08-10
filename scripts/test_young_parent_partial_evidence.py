import numpy as np
import pandas as pd

import src.walk_forward_calibration as wfc


def main():
    dates = pd.date_range("2015-01-14", "2026-08-07", freq="D")
    t = (dates - dates[0]).days.to_numpy(dtype=float)
    # Smooth positive synthetic series with enough observations for every parent.
    logp = np.log(200.0) + 0.00125 * t - 0.00000005 * t * t + 0.7 * np.sin(2 * np.pi * t / 1428.0)
    prices = pd.DataFrame({"date": dates, "price_usd": np.exp(logp)})

    table, _, _ = wfc._score_cycle_aligned_parents(prices)
    row = table[pd.to_datetime(table["start_date"]) == pd.Timestamp("2022-11-07")]
    assert len(row) == 1, table
    row = row.iloc[0]
    # With data through Aug-2026 the 2022 parent has just crossed the frozen
    # model's 1000-observation minimum and owns at least one valid fake-today test.
    assert int(row["tests"]) >= 1, row
    assert row["weight_source"] == "own OOS evidence", row
    assert float(row["weight"]) > 0.0, row
    print("Young 2022 cycle parent earns partial own OOS evidence once 1000 training days exist.")


if __name__ == "__main__":
    main()
