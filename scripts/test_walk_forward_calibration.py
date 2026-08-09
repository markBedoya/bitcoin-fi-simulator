import numpy as np
import pandas as pd

from src.walk_forward_calibration import (
    CALIBRATION_FLOOR,
    FIRST_FAKE_TODAY,
    _fit_growth_amplitude_factors,
    generate_fake_today_dates,
)


def main():
    # The solver should recover known structural-growth and amplitude corrections
    # from clean synthetic out-of-sample observations.
    rng = np.random.default_rng(7)
    G_true = 0.82
    K_true = 0.73
    xg = np.linspace(0.03, 0.45, 80)
    xc = 0.55 * np.sin(np.linspace(-2.2, 3.8, 80))
    y = G_true * xg + K_true * xc + rng.normal(0.0, 0.005, len(xg))
    obs = pd.DataFrame({
        "structural_log_growth": xg,
        "cycle_log_change": xc,
        "actual_log_return": y,
    })
    G, K = _fit_growth_amplitude_factors(obs, ridge_strength=0.1)
    assert abs(G - G_true) < 0.05, (G, G_true)
    assert abs(K - K_true) < 0.05, (K, K_true)

    # Standardized fake-today dates must not begin until an exact 8-year window
    # can remain entirely inside the Jan-2015 calibration regime.
    dates = pd.date_range(CALIBRATION_FLOOR, "2026-08-07", freq="D")
    prices = pd.DataFrame({"date": dates, "price_usd": np.linspace(200, 60000, len(dates))})
    fake = generate_fake_today_dates(prices)
    assert fake[0] == FIRST_FAKE_TODAY
    assert fake[-1] == pd.Timestamp("2025-07-14")
    assert all(d - pd.DateOffset(years=8) >= CALIBRATION_FLOOR for d in fake)

    print("Walk-forward calibration factor/floor checks passed.")


if __name__ == "__main__":
    main()
