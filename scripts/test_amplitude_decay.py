import numpy as np
import pandas as pd
from src.price_model import _anchor_amplitude_decay, _project_anchor_amplitude

history = pd.DataFrame([
    {"type":"peak","cycle":-2,"date":pd.Timestamp("2017-12-17"),"log_deviation":1.20},
    {"type":"peak","cycle":-1,"date":pd.Timestamp("2021-11-08"),"log_deviation":0.80},
    {"type":"peak","cycle":0,"date":pd.Timestamp("2025-10-06"),"log_deviation":0.45},
    {"type":"trough","cycle":-1,"date":pd.Timestamp("2018-12-15"),"log_deviation":-1.00},
    {"type":"trough","cycle":0,"date":pd.Timestamp("2022-11-07"),"log_deviation":-0.60},
])
peak = _anchor_amplitude_decay(history, "peak")
trough = _anchor_amplitude_decay(history, "trough")
assert peak["retention_per_cycle"] <= peak["robust_retention_per_cycle"] + 1e-12
assert peak["retention_per_cycle"] <= peak["recent_retention_per_cycle"] + 1e-12
assert peak["retention_per_cycle"] < 1.0
assert trough["retention_per_cycle"] < 1.0
assert _project_anchor_amplitude(peak, 1) <= 0.45 + 1e-12
assert _project_anchor_amplitude(trough, 1) <= 0.60 + 1e-12
print("Recent-aware mature peak/trough amplitude decay is monotone.")
