from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
page = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")

assert 'st.title("₿ Bitcoin Fair Value")' in page
assert "Research Lab — calibration and evidence" in page
assert "The current cycle is still settling" in page
assert "Bottom-definition sensitivity" in page
assert "Historical fake-today test" in page
assert "Your scenario — never used as model evidence" in page
assert "Copy/paste research diagnostics" in page
assert "bitcoin-dynamic-settling-copy-block-v2" in page
assert "loaded_engine_source" in page
assert "candidate_forecasts" in page
assert "bottom_walk_forward" in page
assert "dynamic_settling_summary" in page
assert "bottom_sensitivity" in page
assert "bitcoin_dynamic_settling_diagnostics.txt" in page
assert "st.navigation" not in page
assert "BTC Financial Independence" not in page
print("Single-page UI contract checks passed.")
