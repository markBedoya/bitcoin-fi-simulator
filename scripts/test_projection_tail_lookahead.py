import numpy as np
import pandas as pd
from src.price_model import fit_price_model

# Synthetic Bitcoin-like path. Choose a horizon that ends between future
# turning points; the projected tail should still move because the model uses
# the next out-of-horizon anchor internally.
dates = pd.date_range("2015-01-01", "2026-08-07", freq="D")
t = np.arange(len(dates), dtype=float)
trend = np.exp(np.log(250.0) + 0.0012 * t)
prices = pd.DataFrame({
    "date": dates,
    "price_usd": trend * np.exp(0.45 * np.sin(2*np.pi*t/1428.0 - 1.2)),
})
result = fit_price_model(prices, pd.Timestamp("2018-11-28"), dates.max(), 10)
proj = result.daily[result.daily["row_type"] == "projected"].copy()
assert result.diagnostics["projection_tail_uses_lookahead_anchor"] is True
# Over the last 180 days, price must not be flat-filled from the last turning point.
tail = proj.tail(180)["fitted_or_projected_price_usd"].to_numpy(float)
assert np.ptp(tail) > 1e-6 * np.mean(tail)
# Public daily output still ends exactly at the requested horizon.
assert pd.Timestamp(proj["date"].iloc[-1]) == dates.max() + pd.DateOffset(years=10)
print("Projection tail look-ahead continuation checks passed.")
