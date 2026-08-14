from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.price_model import SUMMARY_SCHEMA, fit_bottom_anchored_model  # noqa: E402


def synthetic_prices() -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", "2026-08-12", freq="D")
    genesis = pd.Timestamp("2009-01-03")
    days = (dates - genesis).days.to_numpy(dtype=float)
    floor = 2.0e-17 * np.power(days, 5.75)
    progress = np.mod(np.arange(len(dates)), 1431) / 1431.0
    cycle_premium = np.exp(1.15 * np.square(np.sin(np.pi * progress)))
    price = np.maximum(floor * cycle_premium, 0.01)
    return pd.DataFrame({"date": dates, "price_usd": price})


def main() -> None:
    result = fit_bottom_anchored_model(synthetic_prices())
    assert len(result.bottom_regions) == 5
    assert len(result.peak_regions) == 4
    assert len(result.forming_prior_forecasts) == 4
    assert np.isclose(result.forming_prior_forecasts["ensemble_weight"].sum(), 1.0)
    assert result.summary["dynamic_fair_value_usd"] > 0
    assert result.summary["summary_schema"] == SUMMARY_SCHEMA
    assert result.summary["dynamic_settled_bottom_estimate_usd"] > 0
    assert 0 <= result.summary["forming_evidence_weight"] <= 1
    assert result.summary["next_bottom_core_usd"] > 0
    assert result.summary["next_bottom_core_multiple"] > 1
    assert result.summary["mature_observed_growth_path"].count("→") == 2
    assert result.summary["next_bottom_core_low_usd"] <= result.summary["next_bottom_core_usd"]
    assert result.summary["next_bottom_core_high_usd"] >= result.summary["next_bottom_core_usd"]
    assert result.summary["status"] == "RESEARCH_ONLY"
    assert result.curve["date"].max() == result.summary["next_bottom_anchor"]
    assert result.curve["bottom_foundation_usd"].gt(0).all()
    historical_curve = result.curve[result.curve["row_type"] == "historical"]
    projected_curve = result.curve[result.curve["row_type"] == "projected"]
    assert historical_curve["dynamic_fair_value_usd"].gt(0).all()
    assert projected_curve["dynamic_fair_value_usd"].isna().all()
    assert not result.walk_forward.empty
    assert len(result.mature_cycle_forecast) == 1
    assert result.mature_cycle_forecast["mature_transitions"].iloc[0] == 3
    assert not result.fair_value_methods.empty
    assert np.isclose(result.fair_value_methods["ensemble_weight"].sum(), 1.0)
    assert not result.dynamic_settling_summary.empty
    assert result.bottom_sensitivity["available"].any()
    assert result.bottom_sensitivity.loc[
        result.bottom_sensitivity["available"], "mature_cycle_next_bottom_usd"
    ].notna().all()
    assert result.summary["bottom_sensitivity_dynamic_low_usd"] <= result.summary["bottom_sensitivity_dynamic_high_usd"]
    print("Bottom-anchored price-model checks passed.")


if __name__ == "__main__":
    main()
