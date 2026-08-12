## v3.20.2 — Expanded Price Model v2 fit set

- Expanded **Price Model v2** to include the earlier complete cycle: 2011 trough → 2014 peak → 2015 trough.
- Added all contiguous complete-cycle combinations now available on the page:
  - Cycle 0 only
  - Cycle 1 only
  - Cycle 2 only
  - Cycles 0–1 combined
  - Cycles 1–2 combined
  - Cycles 0–2 combined
- Added the additional live-data exploratory fits requested by the user:
  - 2011 trough → latest available Bitcoin price data
  - 2015 trough → latest available Bitcoin price data
  - 2018 trough → latest available Bitcoin price data
- Replaced the earlier 2013 peak anchor on this page with the cycle-derived 2014 peak anchor used in the broader project.
- Updated the page copy, sidebar, and fit summary table to distinguish complete-cycle fits from live-data fits.
- Expanded the Price Model v2 regression script to validate all 9 exploratory fits.

## v3.20.1 — Price Model v2 cycle-by-cycle power-law explorer

- Added a new **Price Model v2** Streamlit page for visually comparing cycle-specific power-law centerlines.
- Overlaid the established historical cycle anchor points directly on the chart.
- Added power-law fits for:
  - Cycle 1: 2015 trough → 2017 peak → 2018 trough
  - Cycle 2: 2018 trough → 2021 peak → 2022 trough
  - Combined Cycles 1–2 fit
- Displayed each fitted centerline across the full Bitcoin history while emphasizing the actual fitted window.
- Added fit diagnostics table including exponent, log-RMSE, log-R², and fitted start/end centerline values.
- Updated the landing page instructions to mention the new Price Model v2 page.

# Changelog

## 3.0.0-cloud

- Added Streamlit Community Cloud entrypoint.
- Removed shared active-model JSON persistence.
- Isolated Price Model configuration by visitor session.
- Added cloud-safe temporary Coin Metrics cache.
- Added Streamlit configuration and Codespaces support.
- Added deployment, contribution, and disclaimer documentation.
