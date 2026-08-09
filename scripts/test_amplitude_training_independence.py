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
assert pd.Timestamp("2017-12-17") in set(history["date"]), history
row = history.loc[history["date"] == pd.Timestamp("2017-12-17")].iloc[0]
assert bool(row["inside_structural_training"]) is False
assert bool(row["used_for_amplitude_decay"]) is True
assert {pd.Timestamp("2021-11-08"), pd.Timestamp("2025-10-06")}.issubset(set(history["date"]))
print("Mature amplitude anchors remain available outside the structural training start.")
