import numpy as np
import pandas as pd

from src.price_model import (
    SCHEDULED_CYCLE_START,
    SCHEDULED_PRE_2015_PEAK,
    _build_cycle_fit,
    _fit_centerline,
    fit_price_model,
)


def make_prices(future_peak_override=None):
    anchor_values = [
        (SCHEDULED_CYCLE_START, 0.71),
        (SCHEDULED_PRE_2015_PEAK, 850.0),
        (pd.Timestamp("2015-01-14"), 180.0),
        (pd.Timestamp("2017-12-17"), 19_000.0),
        (pd.Timestamp("2018-12-15"), 3_200.0),
        (pd.Timestamp("2021-11-08"), 67_500.0),
        (pd.Timestamp("2022-11-07"), 20_500.0),
        (pd.Timestamp("2025-10-06"), 124_828.0 if future_peak_override is None else future_peak_override),
        (pd.Timestamp("2026-08-07"), 65_000.0),
    ]
    dates = pd.date_range("2010-07-18", "2026-08-07", freq="D")
    x = np.array([d.toordinal() for d in dates], dtype=float)
    ax = np.array([d.toordinal() for d, _ in anchor_values], dtype=float)
    ay = np.log(np.array([p for _, p in anchor_values], dtype=float))
    log_price = np.interp(x, ax, ay, left=ay[0], right=ay[-1])
    return pd.DataFrame({"date": dates, "price_usd": np.exp(log_price)})


prices = make_prices()
cutoff = pd.Timestamp("2025-10-06")
start = pd.Timestamp("2015-01-14")
result = fit_price_model(prices, start, cutoff, 10)
diag = result.diagnostics

assert diag["future_endpoints_centerline_generated"] is True
assert not diag["cycle_valuation_history"].empty
assert diag["peak_valuation_model"]["observations"] >= 3
assert diag["trough_valuation_model"]["observations"] >= 3

future = diag["cycle_anchor_table"]
future = future[future["date"] > cutoff].sort_values("date")
trough_2026 = future[(future["date"] == pd.Timestamp("2026-10-05")) & (future["type"] == "trough")].iloc[0]
peak_2029 = future[(future["date"] == pd.Timestamp("2029-09-03")) & (future["type"] == "peak")].iloc[0]
trough_2030 = future[(future["date"] == pd.Timestamp("2030-09-02")) & (future["type"] == "trough")].iloc[0]
trough_2034 = future[(future["date"] == pd.Timestamp("2034-07-31")) & (future["type"] == "trough")].iloc[0]

# The screenshot failure is repaired by attaching the next trough to fair value
# through the learned trough multiple, not by keeping it near the preceding peak.
assert float(trough_2026["knot_price_usd"]) < 124_828.0
assert float(trough_2026["valuation_multiple"]) < 1.0
assert float(peak_2029["valuation_multiple"]) > 1.0
assert float(trough_2030["valuation_multiple"]) < 1.0

# Rising structural fair value + non-expanding trough discount must produce
# rising cycle lows rather than the downward staircase exposed in v3.4.
assert float(trough_2030["knot_price_usd"]) > float(trough_2026["knot_price_usd"])
assert float(trough_2034["knot_price_usd"]) > float(trough_2030["knot_price_usd"])

# Centerline is economically connected again. Scale FUTURE centerline only;
# historical valuation learning stays identical, and future endpoints move.
train = prices[(prices["date"] >= start) & (prices["date"] <= cutoff)].copy()
future_dates = pd.date_range(cutoff + pd.Timedelta(days=1), pd.Timestamp("2030-09-02"), freq="D")
centerline, _ = _fit_centerline(train, future_dates)
all_dates = pd.DatetimeIndex(train["date"].tolist() + future_dates.tolist())
scaled = centerline.copy()
scaled[len(train):] *= 1.25
fit_a = _build_cycle_fit(prices, train, all_dates, centerline, start, cutoff)
fit_b = _build_cycle_fit(prices, train, all_dates, scaled, start, cutoff)
ka = fit_a[3]
kb = fit_b[3]
a = ka[(ka["date"] == pd.Timestamp("2026-10-05")) & (ka["type"] == "trough")].iloc[0]
b = kb[(kb["date"] == pd.Timestamp("2026-10-05")) & (kb["type"] == "trough")].iloc[0]
assert float(b["knot_price_usd"]) > float(a["knot_price_usd"])
assert abs(float(b["projected_valuation_multiple_before_guardrails"]) - float(a["projected_valuation_multiple_before_guardrails"])) < 1e-12

# No future-price leakage: changing the 2025 actual price after a 2022 fake
# cutoff changes the diagnostic actual outcome, but not the forecast produced at
# the 2022 cutoff.
cutoff_2022 = pd.Timestamp("2022-11-07")
base = fit_price_model(prices, SCHEDULED_CYCLE_START, cutoff_2022, 5)
changed = fit_price_model(make_prices(future_peak_override=500_000.0), SCHEDULED_CYCLE_START, cutoff_2022, 5)
base_peak = base.diagnostics["cycle_anchor_table"]
base_peak = base_peak[(base_peak["date"] == pd.Timestamp("2025-10-06")) & (base_peak["type"] == "peak")].iloc[0]
changed_peak = changed.diagnostics["cycle_anchor_table"]
changed_peak = changed_peak[(changed_peak["date"] == pd.Timestamp("2025-10-06")) & (changed_peak["type"] == "peak")].iloc[0]
assert abs(float(base_peak["knot_price_usd"]) / float(changed_peak["knot_price_usd"]) - 1.0) < 1e-12

print("Fair-value endpoint, rising-low, centerline-linkage, and no-lookahead checks passed.")
