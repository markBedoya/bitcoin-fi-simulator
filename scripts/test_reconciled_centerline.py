"""Regression retained under the old filename for compatibility.

v3.1 intentionally removes future-centerline reconciliation. The structural
centerline is now immutable and cycle constraints adapt around it.
"""
import numpy as np
import pandas as pd

from src.price_model import fit_price_model

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

assert diag["structural_centerline_locked"] is True
assert diag["future_centerline_reconciled"] is False
assert diag["future_centerline_reconciliation_applied"] is False

# The displayed centerline must be exactly the raw structural model everywhere.
daily = result.daily
assert np.allclose(
    daily["structural_centerline_usd"].to_numpy(dtype=float),
    daily["raw_structural_centerline_usd"].to_numpy(dtype=float),
    rtol=0.0,
    atol=1e-10,
)

# Normal projected peaks/troughs must remain on the expected side of the locked
# centerline; a sanity ceiling is never allowed to move the centerline.
anchors = diag["amplitude_anchor_table"]
future = anchors[anchors["actual_price_usd"].isna()].copy()
peaks = future[future["type"] == "peak"]
troughs = future[future["type"] == "trough"]
assert not peaks.empty and not troughs.empty
assert (peaks["knot_price_usd"] > peaks["structural_centerline_usd"]).all()
assert (troughs["knot_price_usd"] < troughs["structural_centerline_usd"]).all()

print("Locked structural centerline checks passed.")
