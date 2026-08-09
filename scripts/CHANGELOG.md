# Changelog

## 3.13.0 — Walk-Forward Calibrated Price Model

- Froze the existing Price Model v3.12 mathematics as the parent/base model.
- Added a separate Calibrated Price Model page.
- Added Jan 14, 2015 as a hard walk-forward calibration data floor.
- Added standardized 4Y + 8Y fake-today tests every six months from Jan 2023 onward.
- Added 12-month held-out scoring with no future price data available to each fake model.
- Added robust learned structural-growth factor G and cycle-amplitude factor K.
- Added leave-one-fake-date-out validation, stability scoring, and calibration fingerprints.
- Added a calibrated future centerline, peaks, troughs, and daily Bitcoin price path.
- Preserved the live-conditioned 2025–26 bear path before calibration takes over.
- BTC Financial Independence now defaults to the validated calibrated model when available and allows the frozen v3.12 model for comparison.

## 3.0.0-cloud

- Added Streamlit Community Cloud entrypoint.
- Removed shared active-model JSON persistence.
- Isolated Price Model configuration by visitor session.
- Added cloud-safe temporary Coin Metrics cache.
- Added Streamlit configuration and Codespaces support.
- Added deployment, contribution, and disclaimer documentation.
