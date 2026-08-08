import numpy as np
import pandas as pd

from src.price_model import (
    FIXED_BEAR_DAYS,
    NEXT_TROUGH,
    REFERENCE_PEAK,
    _fit_current_partial_bear,
    _interp_cycle_price_knots,
)

# Synthetic current bear market generated from a known fast-early bear template.
grid = np.linspace(0.0, 1.0, 401)
bear_template = 1.0 - (1.0 - grid) ** 2.6
training_end = pd.Timestamp("2026-08-08")
dates = pd.date_range(REFERENCE_PEAK, training_end, freq="D")
progress = np.clip(
    (dates - REFERENCE_PEAK).days.to_numpy(dtype=float) / FIXED_BEAR_DAYS,
    0.0,
    1.0,
)
peak_price = 126_000.0
true_trough = 52_000.0
total_decline = np.log(peak_price / true_trough)
shape = np.interp(progress, grid, bear_template)
prices = peak_price * np.exp(-total_decline * shape)
data = pd.DataFrame({"date": dates, "price_usd": prices})

fit = _fit_current_partial_bear(
    data=data,
    training_end=training_end,
    phase_grid=grid,
    bear_template=bear_template,
)

assert fit is not None
assert fit["phase"] == "bear"
assert 0.80 < fit["current_progress"] < 0.90
assert fit["projected_trough_price_usd"] < fit["current_price_usd"]
assert abs(fit["projected_trough_price_usd"] / true_trough - 1.0) < 0.03
assert fit["remaining_change_pct"] < 0.0
assert fit["phase_end"] == NEXT_TROUGH

print("Current partial-bear continuation checks passed.")

# The remaining partial bear path must move from the exact latest actual price
# toward the lower modeled trough without restarting the bear template.
current_date = training_end
current_price = float(data.loc[data["date"] == current_date, "price_usd"].iloc[0])
projected_trough = float(fit["projected_trough_price_usd"])
knots = pd.DataFrame({
    "date": [current_date, NEXT_TROUGH],
    "type": ["latest_actual", "trough"],
    "knot_price_usd": [current_price, projected_trough],
})
check_dates = pd.DatetimeIndex([
    current_date,
    current_date + pd.Timedelta(days=1),
    current_date + (NEXT_TROUGH - current_date) / 2,
    NEXT_TROUGH,
])
path = _interp_cycle_price_knots(
    check_dates, knots, grid, grid ** 3.0, bear_template
)
assert abs(path[0] - current_price) < 1e-6
assert path[1] <= path[0]
assert path[2] < path[0]
assert abs(path[-1] - projected_trough) < 1e-6

print("Current partial-bear path continuation checks passed.")
