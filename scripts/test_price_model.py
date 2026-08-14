from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.price_model import SUMMARY_SCHEMA, fit_bottom_anchored_model  # noqa: E402


def synthetic_prices(end_date: str = "2026-08-12") -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", end_date, freq="D")
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
    assert len(result.peak_regions) == 5
    assert result.peak_regions.iloc[-1]["lifecycle_state"] == "pre_window"
    assert len(result.forming_prior_forecasts) == 4
    assert np.isclose(result.forming_prior_forecasts["ensemble_weight"].sum(), 1.0)
    assert result.summary["dynamic_fair_value_usd"] > 0
    assert result.summary["summary_schema"] == SUMMARY_SCHEMA
    assert result.summary["dynamic_settled_bottom_estimate_usd"] > 0
    assert 0 <= result.summary["linear_window_progress"] <= 1
    assert 0 <= result.summary["forming_evidence_weight"] <= 1
    assert result.summary["settling_calibration_cycles"] == 4
    assert result.summary["next_bottom_core_usd"] > 0
    assert result.summary["next_bottom_core_multiple"] > 1
    assert pd.Timestamp(result.summary["current_bottom_anchor"]) == pd.Timestamp("2026-10-25")
    assert result.summary["current_anchor_shift_from_rough_days"] == 18
    assert result.summary["expected_cycle_days"] == 1434
    assert result.summary["current_cycle"] == 4
    assert result.summary["current_cycle_lifecycle_state"] == "forming"
    assert result.summary["completed_bottom_cycles"] == 4
    assert result.summary["automatically_promoted_bottom_cycles"] == 0
    assert result.summary["current_cycle_observed_region_available"] is True
    assert result.summary["rolling_cycle_engine"].startswith("enabled")
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
    assert not result.settling_calibration.empty
    assert len(result.settling_calibration_detail["target_cycle"].unique()) == 4
    assert result.settling_calibration["window_fraction"].iloc[0] == 0
    assert result.settling_calibration["empirical_evidence_weight"].iloc[0] == 0
    assert result.settling_calibration["empirical_evidence_weight"].iloc[-1] == 1
    assert result.settling_calibration["empirical_evidence_weight"].diff().dropna().ge(0).all()
    assert not result.settling_leave_one_out.empty
    assert result.settling_leave_one_out["omitted_cycle"].nunique() == 4
    assert result.settling_leave_one_out.groupby("omitted_cycle")[
        "empirical_evidence_weight"
    ].apply(lambda values: values.diff().dropna().ge(0).all()).all()
    assert not result.settling_cycle_dependence.empty
    assert len(result.settling_cycle_dependence) == 4
    assert result.settling_cycle_dependence["remaining_cycles"].eq(3).all()
    assert result.summary["forming_evidence_weight_cycle_low"] <= result.summary[
        "forming_evidence_weight_cycle_high"
    ]
    assert result.summary["dynamic_settled_bottom_cycle_low_usd"] <= result.summary[
        "dynamic_settled_bottom_cycle_high_usd"
    ]
    assert result.summary["dynamic_fair_value_cycle_low_usd"] <= result.summary[
        "dynamic_fair_value_cycle_high_usd"
    ]
    assert result.summary["next_bottom_cycle_low_usd"] <= result.summary["next_bottom_cycle_high_usd"]
    assert not result.anchor_timing_sensitivity.empty
    assert len(result.anchor_timing_sensitivity) == 5
    assert result.anchor_timing_sensitivity["available"].all()
    assert result.anchor_timing_sensitivity["timing_role"].str.startswith("completed-cycle").sum() == 3
    empirical_timing = result.anchor_timing_sensitivity[
        result.anchor_timing_sensitivity["timing_role"].str.startswith("completed-cycle")
    ]
    marginalized_bottom = float(np.exp(np.mean(np.log(
        empirical_timing["dynamic_current_bottom_usd"].to_numpy(dtype=float)
    ))))
    marginalized_region = float(np.exp(np.mean(np.log(
        empirical_timing["forming_region_usd"].to_numpy(dtype=float)
    ))))
    assert np.isclose(
        marginalized_bottom,
        result.summary["dynamic_settled_bottom_estimate_usd"],
    )
    assert result.summary["anchor_marginalization_variants"] == 3
    assert pd.Timestamp(result.summary["current_cycle_observation_window_end"]) == pd.Timestamp("2027-02-22")
    assert "promoted automatically" in result.summary["rolling_cycle_status"]
    assert np.isclose(marginalized_region, result.summary["forming_bottom_region_usd"])
    assert np.isclose(result.summary["forming_bottom_region_usd"], 233650.05575375573)
    assert np.isclose(result.summary["dynamic_settled_bottom_estimate_usd"], 224829.51421970947)
    assert np.isclose(result.summary["dynamic_fair_value_usd"], 248219.60196226233)
    assert np.isclose(result.summary["next_bottom_core_usd"], 495801.47379315965)
    assert np.isclose(
        empirical_timing["forming_evidence_weight"].mean(),
        result.summary["forming_evidence_weight"],
    )
    assert result.summary["forming_bottom_region_anchor_low_usd"] <= result.summary[
        "forming_bottom_region_usd"
    ] <= result.summary["forming_bottom_region_anchor_high_usd"]
    assert result.summary["anchor_timing_empirical_bottom_low_usd"] <= result.summary[
        "anchor_timing_empirical_bottom_high_usd"
    ]
    assert "not yet validated" in result.summary["bottom_projection_horizon_status"]
    assert result.bottom_sensitivity["available"].any()
    assert result.bottom_sensitivity.loc[
        result.bottom_sensitivity["available"], "mature_cycle_next_bottom_usd"
    ].notna().all()
    assert result.summary["bottom_sensitivity_dynamic_low_usd"] <= result.summary["bottom_sensitivity_dynamic_high_usd"]
    print("Bottom-anchored price-model checks passed.")


if __name__ == "__main__":
    main()
