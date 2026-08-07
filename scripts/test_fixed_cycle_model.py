from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.price_model import (
    FIXED_CYCLE_DAYS,
    FIXED_BULL_DAYS,
    FIXED_BEAR_DAYS,
    REFERENCE_TROUGH,
    REFERENCE_PEAK,
    NEXT_TROUGH,
)

assert FIXED_BULL_DAYS + FIXED_BEAR_DAYS == FIXED_CYCLE_DAYS == 1428
assert (REFERENCE_PEAK - REFERENCE_TROUGH).days == 1064
assert (NEXT_TROUGH - REFERENCE_PEAK).days == 364
assert REFERENCE_TROUGH == pd.Timestamp("2022-11-07")
assert REFERENCE_PEAK == pd.Timestamp("2025-10-06")
assert NEXT_TROUGH == pd.Timestamp("2026-10-05")

print("Fixed 1428-day cycle timing checks passed.")

import numpy as np
from src.price_model import fit_price_model

dates = pd.date_range("2015-01-01", "2026-08-05", freq="D")
days = np.arange(1, len(dates) + 1, dtype=float)
# Positive, smooth synthetic price series with enough variation for regression.
prices = 200.0 * np.power(1.0008, days) * (1.0 + 0.05 * np.sin(days / 120.0))
frame = pd.DataFrame({"date": dates, "price_usd": prices})
result = fit_price_model(
    frame,
    pd.Timestamp("2016-02-29"),
    pd.Timestamp("2026-08-05"),
    10,
)
anchors = result.diagnostics["cycle_anchor_table"]
historical = anchors[anchors["actual_price_usd"].notna()].copy()
historical["modeled"] = historical["structural_centerline_usd"] * np.exp(historical["log_deviation"])
assert np.allclose(historical["modeled"], historical["actual_price_usd"], rtol=1e-12, atol=1e-9)
assert result.diagnostics["next_modeled_trough"] == "2026-10-05"
print("Historical fixed-cycle anchor intersection checks passed.")
