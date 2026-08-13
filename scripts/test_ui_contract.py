from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
page = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")

assert 'st.title("₿ Bitcoin Fair Value")' in page
assert "Research Lab — evidence, competing models, and scenarios" in page
assert "Your scenario — never used as model evidence" in page
assert "Copy/paste research diagnostics" in page
assert "bitcoin-bottom-fair-value-copy-block-v1" in page
assert "loaded_engine_source" in page
assert "candidate_forecasts" in page
assert "walk_forward" in page
assert "bitcoin_bottom_fair_value_diagnostics.txt" in page
assert "st.navigation" not in page
assert "BTC Financial Independence" not in page
print("Single-page UI contract checks passed.")
