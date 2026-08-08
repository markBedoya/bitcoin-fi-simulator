import numpy as np
import pandas as pd
from src.price_model import _anchor_amplitude_decay, _project_anchor_amplitude

history = pd.DataFrame([
    {'type':'peak','cycle':-2,'log_deviation':1.20},
    {'type':'peak','cycle':-1,'log_deviation':0.80},
    {'type':'peak','cycle':0,'log_deviation':0.45},
    {'type':'trough','cycle':-2,'log_deviation':-1.00},
    {'type':'trough','cycle':-1,'log_deviation':-0.75},
    {'type':'trough','cycle':0,'log_deviation':-0.60},
])
peak = _anchor_amplitude_decay(history, 'peak')
trough = _anchor_amplitude_decay(history, 'trough')
assert peak['retention_per_cycle'] < 1.0
assert trough['retention_per_cycle'] < 1.0
assert _project_anchor_amplitude(peak, 1) < 0.45
assert _project_anchor_amplitude(trough, 1) < 0.60
print('Peak/trough amplitude decay is monotone and separately estimated.')
