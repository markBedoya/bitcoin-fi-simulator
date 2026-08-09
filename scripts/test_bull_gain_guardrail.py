import numpy as np
import pandas as pd

from src.price_model import _bull_gain_decay

history = pd.DataFrame([
    {"cycle": -1, "peak_date": pd.Timestamp("2021-11-08"), "bull_log_gain": np.log(21.2)},
    {"cycle": 0, "peak_date": pd.Timestamp("2025-10-06"), "bull_log_gain": np.log(6.07)},
])
stats = _bull_gain_decay(history)

# Historical compression remains diagnostic, but v3.1 uses only the latest
# completed mature bull multiple as the next-cycle non-expansion ceiling.
assert abs(stats["latest_multiple"] - 6.07) < 1e-10
assert stats["latest_cycle"] == 0.0
assert stats["retention_per_cycle"] <= 1.0
print("Bull non-expansion ceiling basis checks passed.")
