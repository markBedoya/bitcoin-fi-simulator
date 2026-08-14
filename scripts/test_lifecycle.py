from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.test_price_model import synthetic_prices  # noqa: E402
from src.data_pipeline import cache_is_stale  # noqa: E402
from src.price_model import fit_bottom_anchored_model  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "prices.csv"
        cache.touch()
        modified = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
        assert not cache_is_stale(cache, now=modified + timedelta(hours=23))
        assert cache_is_stale(cache, now=modified + timedelta(hours=24))

    forming = fit_bottom_anchored_model(synthetic_prices("2027-02-21"))
    assert forming.summary["current_cycle"] == 4
    assert forming.summary["current_cycle_lifecycle_state"] == "forming"
    assert forming.summary["automatically_promoted_bottom_cycles"] == 0

    rolled = fit_bottom_anchored_model(synthetic_prices("2027-02-22"))
    assert rolled.summary["current_cycle"] == 5
    assert rolled.summary["current_cycle_lifecycle_state"] == "pre_window"
    assert rolled.summary["completed_bottom_cycles"] == 5
    assert rolled.summary["automatically_promoted_bottom_cycles"] == 1
    assert rolled.summary["current_cycle_observed_region_available"] is False
    assert rolled.summary["forming_evidence_weight"] == 0
    assert rolled.summary["linear_window_progress"] == 0
    assert np.isnan(rolled.summary["forming_bottom_region_usd"])
    assert np.isclose(
        rolled.summary["dynamic_settled_bottom_estimate_usd"],
        rolled.summary["pre_observation_bottom_forecast_usd"],
    )
    settled_cycle = rolled.bottom_regions[rolled.bottom_regions["cycle"] == 4].iloc[0]
    assert settled_cycle["lifecycle_state"] == "settled"
    assert settled_cycle["status"] == "completed"
    assert pd.Timestamp(rolled.summary["current_bottom_anchor"]) > pd.Timestamp("2029-01-01")

    peak_forming = fit_bottom_anchored_model(synthetic_prices("2029-03-01"))
    assert peak_forming.summary["active_peak_cycle"] == 4
    assert peak_forming.summary["active_peak_lifecycle_state"] == "forming"
    peak_row = peak_forming.peak_regions[peak_forming.peak_regions["cycle"] == 4].iloc[0]
    assert np.isfinite(float(peak_row["region_price_usd"]))

    later = fit_bottom_anchored_model(synthetic_prices("2029-08-22"))
    later_settled = later.bottom_regions[later.bottom_regions["cycle"] == 4].iloc[0]
    assert pd.Timestamp(later_settled["region_date"]) == pd.Timestamp(settled_cycle["region_date"])
    assert np.isclose(float(later_settled["region_price_usd"]), float(settled_cycle["region_price_usd"]))
    settled_peak = later.peak_regions[later.peak_regions["cycle"] == 4].iloc[0]
    assert settled_peak["lifecycle_state"] == "settled"
    assert settled_peak["status"] == "completed"

    second_roll = fit_bottom_anchored_model(synthetic_prices("2030-12-31"))
    assert second_roll.summary["current_cycle"] == 6
    assert second_roll.summary["automatically_promoted_bottom_cycles"] == 2
    assert second_roll.summary["completed_bottom_cycles"] == 6

    print("Rolling lifecycle time-travel checks passed.")


if __name__ == "__main__":
    main()
