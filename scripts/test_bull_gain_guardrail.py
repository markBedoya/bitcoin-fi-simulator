import numpy as np
import pandas as pd

from src.price_model import _bull_gain_decay, _project_bull_log_gain

history = pd.DataFrame([
    {"cycle": -1, "peak_date": pd.Timestamp("2021-11-08"), "bull_log_gain": np.log(21.2)},
    {"cycle": 0, "peak_date": pd.Timestamp("2025-10-06"), "bull_log_gain": np.log(6.07)},
])
decay = _bull_gain_decay(history)
assert decay["retention_per_cycle"] <= 1.0
next_gain = _project_bull_log_gain(decay, 1)
assert next_gain <= np.log(6.07) + 1e-12
assert np.exp(next_gain) <= 6.07 + 1e-12
print("Bull-gain compression guardrail checks passed.")
