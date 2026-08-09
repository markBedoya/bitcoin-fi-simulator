import numpy as np
import pandas as pd

from src.price_model import fit_price_model

# Synthetic mature Bitcoin-like history chosen to trigger a future conflict
# between raw structural growth and compressed bull-run gains.
dates = pd.date_range("2014-01-01", "2026-08-05", freq="D")
days = np.arange(len(dates), dtype=float)
trend = np.exp(np.log(300.0) + 0.00135 * days)
cycle = np.exp(0.42 * np.sin(2 * np.pi * days / 1428.0 - 1.2))
prices = pd.DataFrame({"date": dates, "price_usd": trend * cycle})

result = fit_price_model(
    prices=prices,
    training_start=dates.min(),
    training_end=dates.max(),
    projection_years=10,
)
diag = result.diagnostics

assert diag["future_centerline_reconciled"] is True
scale_table = diag["future_centerline_scale_table"].sort_values("date")
scales = scale_table["scale"].to_numpy(dtype=float)
assert abs(scales[0] - 1.0) < 1e-12
assert np.all(np.diff(scales) <= 1e-12)
assert np.all(scales > 0)

# Future displayed centerline is a downward reconciliation of the raw structural
# prior, never an upward boost.
proj = result.daily[result.daily["row_type"] == "projected"]
assert (proj["structural_centerline_usd"] <= proj["raw_structural_centerline_usd"] * (1 + 1e-12)).all()

# Crucial regression: maturity guardrails must never solve a conflict by placing
# a projected peak on the centerline. Peaks retain positive deviation; troughs
# retain negative deviation against the *displayed* reconciled centerline.
anchors = diag["amplitude_anchor_table"]
future = anchors[anchors["actual_price_usd"].isna()].copy()
peaks = future[future["type"] == "peak"]
troughs = future[future["type"] == "trough"]
assert not peaks.empty and not troughs.empty
assert (peaks["log_deviation"] >= 0.03 - 1e-10).all()
assert (peaks["knot_price_usd"] > peaks["structural_centerline_usd"]).all()
assert (troughs["log_deviation"] <= -0.03 + 1e-10).all()
assert (troughs["knot_price_usd"] < troughs["structural_centerline_usd"]).all()

# The full bull-run compression rule still holds after moving the centerline.
bulls = diag["bull_gain_table"].sort_values("peak_date")
completed = bulls[bulls["source"].astype(str).str.contains("observed", case=False, na=False)]
projected = bulls[bulls["source"].astype(str).str.contains("projected", case=False, na=False)]
if not completed.empty and not projected.empty:
    prior = float(completed.iloc[-1]["bull_log_gain"])
    for gain in projected["bull_log_gain"].to_numpy(dtype=float):
        assert gain <= prior + 1e-10
        prior = float(gain)

print("Reconciled future centerline checks passed.")
