import numpy as np
import pandas as pd

from src.price_model import _learn_empirical_phase_templates

# Build synthetic daily data spanning the mature historical phase dates.
dates = pd.date_range("2017-12-17", "2025-10-06", freq="D")
# Positive synthetic price path is sufficient to test phase inclusion counts.
prices = 1000.0 * np.exp(np.linspace(0, 4.0, len(dates)))
data = pd.DataFrame({"date": dates, "price_usd": prices})
center = pd.Series(prices, index=dates)

_, _, _, overlays, _, _, _ = _learn_empirical_phase_templates(
    data=data,
    center_series=center,
    training_start=pd.Timestamp("2018-11-28"),
    training_end=pd.Timestamp("2025-10-06"),
)
# Structural training starts after the 2017 peak, but mature phase learning
# should still keep the 2017-2018 bear phase.
bears = overlays.loc[overlays["phase"] == "bear", "phase_id"].nunique()
bulls = overlays.loc[overlays["phase"] == "bull", "phase_id"].nunique()
assert bears == 2, bears
assert bulls == 2, bulls
print("Phase-shape training independence checks passed.")
