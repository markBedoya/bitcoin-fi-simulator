## v3.20.3 — Restore explicit Streamlit navigation

- Fixed the v3.20.2 packaging regression that overwrote the explicit `st.navigation(...)` router with an older single-page `streamlit_app.py`.
- Moved Price Model v2 out of Streamlit's reserved top-level `pages/` directory and into `app_pages/5_Price_Model_v2.py`.
- Added **Price Model v2** to the explicit left navigation with stable route `price-model-v2`.
- Preserved the existing Data Management, Price Model, Calibrated Price Model, and BTC Financial Independence routes.
- Added a migration note requiring deletion of the accidentally created `pages/2a_Price_Model_v2.py` file from repositories that deployed v3.20.2.
- Updated the navigation regression test to require five unique explicit routes and prohibit `pages/` route sources.
