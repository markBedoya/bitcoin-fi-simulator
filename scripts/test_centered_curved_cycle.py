import numpy as np
import pandas as pd

from src.price_model import fit_price_model

# Synthetic daily Bitcoin-like series: enough history to exercise all known
# historical phase anchors without depending on network data.
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
assert diag["future_cycle_centered"] is True
assert diag["phase_shape_applied_to"] == "total log-price path"
assert diag["bull_phases_used"] >= 2
assert diag["bear_phases_used"] >= 2
assert not diag["phase_shape_templates"].empty

# Peaks must stay above and troughs below the structural centerline. Their
# amplitudes are now estimated independently so a deep live trough cannot force
# an equally large future upside amplitude.
anchors = diag["cycle_anchor_table"]
future = anchors[
    anchors["source"].astype(str).str.contains("projected|conditioned", case=False, regex=True)
]
peaks = future[future["type"] == "peak"].sort_values("cycle")
troughs = future[future["type"] == "trough"].sort_values("cycle")
assert (peaks["log_deviation"] >= -1e-12).all()
assert (troughs["log_deviation"] < 0).all()

# Purely projected amplitudes must not expand cycle over cycle.
if len(peaks) >= 2:
    assert np.all(np.diff(np.abs(peaks["log_deviation"].to_numpy(dtype=float))) <= 1e-12)
if len(troughs) >= 3:
    later = np.abs(troughs["log_deviation"].to_numpy(dtype=float))[1:]
    assert np.all(np.diff(later) <= 1e-12)

projected = result.daily[result.daily["row_type"] == "projected"]
assert not projected.empty
assert np.isfinite(projected["fitted_or_projected_price_usd"]).all()
assert (projected["fitted_or_projected_price_usd"] > 0).all()

print("Decaying empirical-cycle projection checks passed.")
