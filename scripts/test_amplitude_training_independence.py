"""Compatibility filename: amplitude is now aligned to structural training."""
import numpy as np
import pandas as pd
from src.price_model import _build_amplitude_anchor_history

dates = pd.date_range("2017-01-01", "2026-08-07", freq="D")
t = np.arange(len(dates), dtype=float)
prices = pd.DataFrame({"date": dates, "price_usd": np.exp(6.0 + 0.001*t)})
training_start = pd.Timestamp("2018-11-28")
training_end = pd.Timestamp("2026-08-07")
train = prices[(prices["date"] >= training_start) & (prices["date"] <= training_end)].copy()
train_days = (train["date"] - pd.Timestamp("2009-01-03")).dt.days.to_numpy(dtype=float)
center = np.exp(-20.0 + 4.0*np.log(train_days))
center_series = pd.Series(center, index=pd.DatetimeIndex(train["date"]))
history = _build_amplitude_anchor_history(prices, train, center_series, training_end)

# A 2017 point lies outside this structural fit and must not influence amplitudes.
assert pd.Timestamp("2017-12-17") not in set(history["date"]), history
assert {pd.Timestamp("2018-12-15"), pd.Timestamp("2021-11-08"), pd.Timestamp("2022-11-07"), pd.Timestamp("2025-10-06")}.issubset(set(history["date"]))
assert history["inside_structural_training"].all()
assert history["used_for_amplitude_decay"].all()
print("Amplitude anchors stay aligned with the selected structural training range.")
