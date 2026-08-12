from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
router = (ROOT / "streamlit_app.py").read_text()

expected_sources = [
    "app_pages/1_Data_Management.py",
    "app_pages/2_Price_Model.py",
    "app_pages/5_Price_Model_v2.py",
    "app_pages/3_Calibrated_Price_Model.py",
    "app_pages/4_BTC_Financial_Independence.py",
]
for source in expected_sources:
    assert source in router, f"Missing navigation source: {source}"

assert '"pages/' not in router and "'pages/" not in router, (
    "The explicit router must not source scripts from Streamlit's reserved pages/ directory."
)

routes = re.findall(r'url_path="([^"]+)"', router)
assert len(routes) == 5, routes
assert len(routes) == len(set(routes)), routes

for source in expected_sources:
    assert (ROOT / source).exists(), source

print("Navigation layout uses only app_pages/ with five unique explicit routes.")
