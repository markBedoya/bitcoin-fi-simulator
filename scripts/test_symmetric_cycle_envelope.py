import numpy as np
import pandas as pd

from src.price_model import _symmetric_cycle_amplitude_decay, _project_symmetric_cycle_amplitude

history = pd.DataFrame([
    {"cycle": -1, "type": "peak", "log_deviation": 1.10},
    {"cycle": -1, "type": "trough", "log_deviation": -0.55},
    {"cycle": 0, "type": "peak", "log_deviation": 0.22},
    {"cycle": 0, "type": "trough", "log_deviation": -0.42},
])

decay = _symmetric_cycle_amplitude_decay(history)
assert decay["observations"] == 2
assert decay["transitions"] == 1
assert 0.0 < decay["raw_retention_per_cycle"] <= decay["retention_per_cycle"] <= 1.0

amp1 = _project_symmetric_cycle_amplitude(decay, 1)
amp2 = _project_symmetric_cycle_amplitude(decay, 2)
assert amp1 > 0.05
assert amp2 <= amp1 + 1e-12

center = 250_000.0
peak = center * np.exp(amp1)
trough = center * np.exp(-amp1)
assert abs(np.log(peak / center) + np.log(trough / center)) < 1e-12
assert peak > center > trough
print("Symmetric cycle-envelope checks passed.")
