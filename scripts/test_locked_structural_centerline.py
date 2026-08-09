import numpy as np
import pandas as pd
from src.price_model import fit_price_model

# Bitcoin-like synthetic series with diminishing cycle amplitude.
dates = pd.date_range("2015-01-01", "2026-08-07", freq="D")
t = np.arange(len(dates), dtype=float)
trend = np.exp(np.log(250.0) + 0.00125 * t)
amp = np.linspace(0.75, 0.35, len(dates))
prices = pd.DataFrame({
    "date": dates,
    "price_usd": trend * np.exp(amp * np.sin(2*np.pi*t/1428.0 - 1.4)),
})
result = fit_price_model(prices, pd.Timestamp("2018-11-28"), dates.max(), 12)
proj = result.daily[result.daily["row_type"] == "projected"]
assert np.allclose(
    proj["structural_centerline_usd"].to_numpy(float),
    proj["raw_structural_centerline_usd"].to_numpy(float),
    rtol=0.0,
    atol=1e-10,
)
assert result.diagnostics["bull_gain_guardrail_mode"] == "non-expansion ceiling only"
assert result.diagnostics["amplitude_training_independent_of_structural_start"] is False
print("Locked backbone architecture checks passed.")
