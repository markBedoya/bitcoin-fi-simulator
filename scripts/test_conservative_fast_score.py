import numpy as np
import pandas as pd

from src.price_model import fit_price_model, _score_projection_endpoint_exact

# Synthetic daily history long enough to exercise all model components.
dates = pd.date_range('2014-01-01', '2026-08-07', freq='D')
t = np.arange(len(dates), dtype=float)
# Positive, smoothly rising path with fixed-cycle-like oscillation.
log_price = 5.0 + 0.00135 * t + 0.65 * np.sin(2*np.pi*t/1428.0 - 1.0)
prices = pd.DataFrame({'date': dates, 'price_usd': np.exp(log_price)})
start = pd.Timestamp('2016-01-31')
end = pd.Timestamp('2026-08-07')
years = 6

full = fit_price_model(prices, start, end, years)
full_end = float(full.daily.loc[full.daily['row_type']=='projected', 'fitted_or_projected_price_usd'].iloc[-1])
fast_end = _score_projection_endpoint_exact(prices, start, end, years)
rel = abs(fast_end/full_end - 1.0)
assert rel < 1e-10, (full_end, fast_end, rel)
print('Exact conservative-search endpoint scorer matches full daily model.')
