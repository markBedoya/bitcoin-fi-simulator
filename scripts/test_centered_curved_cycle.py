import numpy as np
import pandas as pd

from src.price_model import fit_price_model

# Synthetic daily Bitcoin-like series: enough history to exercise the fair-value
# valuation engine without network data.
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
assert diag["future_endpoints_centerline_generated"] is True
assert diag["future_endpoint_method"] == "structural fair value × independently learned peak/trough valuation multiples"
assert diag["transition_role"] == "validation and direction guardrails only"
assert diag["phase_shape_applied_to"] == "total log-price path"
assert diag["bull_phases_used"] >= 2
assert diag["bear_phases_used"] >= 2
assert not diag["phase_shape_templates"].empty

anchors = diag["cycle_anchor_table"].copy()
future = anchors[anchors["date"] > pd.Timestamp(dates.max())].sort_values("date")
assert not future.empty
assert "valuation_multiple" in future.columns

# Peaks stay above fair value, troughs below fair value.
for row in future.itertuples(index=False):
    if row.type == "peak":
        assert float(row.valuation_multiple) >= 1.0
    elif row.type == "trough":
        assert float(row.valuation_multiple) <= 1.0

# Same-type future turning points cannot drift downward while the fair-value
# backbone rises.
for anchor_type in ["peak", "trough"]:
    subset = future[future["type"] == anchor_type].sort_values("date")
    if len(subset) >= 2:
        vals = subset["knot_price_usd"].to_numpy(dtype=float)
        assert np.all(np.diff(vals) >= 0.0)

projected = result.daily[result.daily["row_type"] == "projected"]
assert not projected.empty
assert np.isfinite(projected["fitted_or_projected_price_usd"]).all()
assert (projected["fitted_or_projected_price_usd"] > 0).all()

print("Fair-value centered empirical-cycle projection checks passed.")
