import numpy as np
import pandas as pd

from src.price_model import fit_price_model

# Synthetic daily Bitcoin-like series: enough history to exercise the sequential
# transition engine without network data.
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
assert diag["future_cycle_centered"] is False
assert diag["future_endpoints_centerline_generated"] is False
assert diag["future_endpoint_method"] == "sequential anchor-to-anchor bull gains and bear losses"
assert diag["phase_shape_applied_to"] == "total log-price path"
assert diag["bull_phases_used"] >= 2
assert diag["bear_phases_used"] >= 2
assert not diag["phase_shape_templates"].empty

# Sequential future extrema must preserve actual cycle direction even though
# they no longer have to be symmetric around the structural centerline.
anchors = diag["cycle_anchor_table"].copy()
future = anchors[
    anchors["source"].astype(str).str.contains(
        "projected sequential|current bear-conditioned sequential",
        case=False,
        regex=True,
        na=False,
    )
].sort_values("date")
assert not future.empty
for row in future.itertuples(index=False):
    if np.isfinite(float(row.phase_start_price_usd)):
        if row.type == "peak":
            assert float(row.knot_price_usd) >= float(row.phase_start_price_usd)
        elif row.type == "trough":
            assert float(row.knot_price_usd) <= float(row.phase_start_price_usd)

projected = result.daily[result.daily["row_type"] == "projected"]
assert not projected.empty
assert np.isfinite(projected["fitted_or_projected_price_usd"]).all()
assert (projected["fitted_or_projected_price_usd"] > 0).all()

print("Sequential empirical-cycle projection checks passed.")
