from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
page = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")

assert 'st.title("₿ Bitcoin Fair Value")' in page
assert "Research Lab — calibration and evidence" in page
assert "The current cycle is still settling" in page
assert "Bottom-definition sensitivity" in page
assert "Historical fake-today test" in page
assert "Empirical settling-speed calibration" in page
assert "Conservative calibrated curve" in page
assert "Dependence on individual completed cycles" in page
assert "Current bottom-anchor timing" in page
assert "Original rough anchor" in page
assert "Learned timing center" in page
assert "Marginalized forming region" in page
assert "equal-weight geometric mean" in page
assert 'column.endswith("_date")' in page
assert 'column.endswith("_anchor")' in page
assert "public_model_region_usd" in page
assert "exact_anchor_region_usd" in page
assert "Lifecycle boundary" in page
assert "Automatic promotion" in page
assert "Ten-year projection readiness" in page
assert "Recursive bottom" in page
assert "Leave-one-cycle-out range" in page
assert "not a 95% confidence interval" in page
assert "Mature-cycle estimate" in page
assert "Observed mature bottom growth" in page
assert "price_model = reload(price_model)" in page
assert "required_summary_keys" in page
assert "The page and model engine are from different project versions" in page
guard_position = page.index("The page and model engine are from different project versions")
ratio_position = page.index('ratio = float(summary["price_to_dynamic_fair_value"])')
assert guard_position < ratio_position
assert "price_model.SUMMARY_SCHEMA" not in page
assert "Your scenario — never used as model evidence" in page
assert "Copy/paste research diagnostics" in page
assert "bitcoin-dynamic-settling-copy-block-v7" in page
assert "loaded_engine_source" in page
assert "forming_bottom_prior_forecasts" in page
assert "candidate_forecasts" not in page
assert "Broader model disagreement" not in page
assert "future peak compression has not " in page
assert "been modeled" in page
assert "mature_cycle_forecast" in page
assert "settling_calibration_detail" in page
assert "settling_leave_one_out" in page
assert "settling_cycle_dependence" in page
assert "anchor_timing_sensitivity" in page
assert "bottom_walk_forward" in page
assert "dynamic_settling_summary" in page
assert "bottom_sensitivity" in page
assert "bitcoin_dynamic_settling_diagnostics.txt" in page
assert "st.navigation" not in page
assert "BTC Financial Independence" not in page
print("Single-page UI contract checks passed.")
