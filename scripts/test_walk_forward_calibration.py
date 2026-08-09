import numpy as np
import pandas as pd

from src.walk_forward_calibration import (
    CALIBRATION_FLOOR,
    first_fake_today_for_lookback,
    generate_fake_today_dates,
)


def main():
    dates = pd.date_range(CALIBRATION_FLOOR, "2026-08-07", freq="D")
    prices = pd.DataFrame({"date": dates, "price_usd": np.linspace(200, 60000, len(dates))})
    fake4 = generate_fake_today_dates(prices, 4)
    fake8 = generate_fake_today_dates(prices, 8)
    assert fake4[0] == first_fake_today_for_lookback(4) == pd.Timestamp("2019-01-14")
    assert fake8[0] == first_fake_today_for_lookback(8) == pd.Timestamp("2023-01-14")
    assert fake4[-1] == pd.Timestamp("2025-07-14")
    assert fake8[-1] == pd.Timestamp("2025-07-14")
    assert all(d - pd.DateOffset(years=4) >= CALIBRATION_FLOOR for d in fake4)
    assert all(d - pd.DateOffset(years=8) >= CALIBRATION_FLOOR for d in fake8)
    print("Walk-forward Jan-2015 floor and rolling-window checks passed.")


if __name__ == "__main__":
    main()
