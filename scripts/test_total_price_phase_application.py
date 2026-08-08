import numpy as np
import pandas as pd

from src.price_model import _interp_cycle_price_knots

# Delayed bull template: at 50% of bull time, only 12.5% of the total
# normalized log-price move has occurred. This verifies that the learned phase
# shape controls the entire price move, not merely a residual around a trend.
grid = np.linspace(0.0, 1.0, 401)
bull_template = grid ** 3
bear_template = 1.0 - (1.0 - grid) ** 3

start = pd.Timestamp('2030-01-01')
end = start + pd.Timedelta(days=1064)
knots = pd.DataFrame({
    'date': [start, end],
    'type': ['trough', 'peak'],
    'knot_price_usd': [100_000.0, 1_000_000.0],
})

dates = pd.DatetimeIndex([start, start + pd.Timedelta(days=532), end])
prices = _interp_cycle_price_knots(dates, knots, grid, bull_template, bear_template)

# Log-linear halfway price would be sqrt(100k * 1m) ~= 316k. A late-accelerating
# template must remain materially below that at 50% of bull time.
assert abs(prices[0] - 100_000.0) < 1e-6
assert abs(prices[-1] - 1_000_000.0) < 1e-6
assert prices[1] < 200_000.0

print('Total-price empirical phase application checks passed.')
