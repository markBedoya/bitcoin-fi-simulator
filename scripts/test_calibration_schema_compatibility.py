from types import SimpleNamespace
import pandas as pd

from src.walk_forward_calibration import CALIBRATION_VERSION, REQUIRED_SUMMARY_KEYS, calibration_is_current
from src.price_model import PRICE_MODEL_ENGINE_VERSION

prices = pd.DataFrame({
    "date": pd.to_datetime(["2026-08-06", "2026-08-07"]),
    "price_usd": [64000.0, 65000.0],
})

old = SimpleNamespace(summary={
    "version": "walk-forward-calibration-v2.0.1-cycle-aware",
    "price_model_engine_version": PRICE_MODEL_ENGINE_VERSION,
    "latest_data_date": "2026-08-07",
})
assert not calibration_is_current(old, prices)

summary = {key: 0.0 for key in REQUIRED_SUMMARY_KEYS}
summary.update({
    "version": CALIBRATION_VERSION,
    "price_model_engine_version": PRICE_MODEL_ENGINE_VERSION,
    "latest_data_date": "2026-08-07",
    "cycle_parents": [],
})
current = SimpleNamespace(summary=summary)
assert calibration_is_current(current, prices)

missing = dict(summary)
missing.pop("amplitude_mode")
assert not calibration_is_current(SimpleNamespace(summary=missing), prices)
print("Calibration v4 schema compatibility checks passed.")
