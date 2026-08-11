## v3.18.3 — plot the cycle-derived 2011 trough anchor

- Promotes the cycle-derived 2011-02-14 point from a shortcut/catalog boundary to a true historical trough/intersection anchor.
- Uses the actual imported Coin Metrics price on the derived date (nearest positive observation within 3 days).
- The Price Model chart now shows the 2011 trough with the same historical-anchor marker whenever the selected training range includes it.
- The historical fitted path is constrained to intersect the observed 2011 anchor price exactly.
- Mature phase-shape and amplitude-learning windows remain unchanged, so the early anchor does not alter mature-cycle template learning.

## v3.18.2 — observed price at cycle-derived start

- Keeps the early cycle-start date derived from the fixed 1428-day schedule (2011-02-14).
- Replaces the temporary theoretical $0 display with the actual imported Coin Metrics price on that derived date (nearest observation within 3 days).
- The pre-2015 peak remains cycle-date-derived with its price sourced from Coin Metrics.

## v3.18.1 — cycle-derived early anchors

- Replaces the data-discovered early peak and dataset-start shortcut with two dates derived by running the same fixed 1428-day cycle clock backward from the current schedule.
- Adds a cycle-derived pre-2015 peak date of 2014-01-13; its price is still the observed Coin Metrics price on that derived date.
- Adds a cycle-start $0 reference date of 2011-02-14. The $0 value is explicitly theoretical and is never inserted into log-price fitting; selecting it as a training start fits the real Bitcoin observations from that date forward.
- Keeps the existing observed 2015/2017/2018/2021/2022/2025 historical anchors unchanged.
- Updates anchor shortcut diagnostics and regression tests to verify both early dates are consequences of the fixed 1064-day bull / 364-day bear schedule rather than data-picked dates.

## v3.18.0 — Price Model anchor exploration

- Pauses calibrated-model development and resumes targeted Price Model research changes.
- Extends the historical fitted-path anchor sequence by one data-derived peak before the Jan-14-2015 trough.
- Derives that early peak from the active Coin Metrics dataset as the highest observed pre-2015-trough price; no early peak price is hard-coded.
- Adds the first usable Coin Metrics observation as a user-facing dataset-start anchor shortcut.
- Adds Start, End, and Apply Range anchor controls so either training boundary can snap to any anchor without dragging the date slider.
- Adds a Historical Anchor Catalog with exact active-dataset dates/prices and source descriptions.
- Keeps mature phase-shape, mature amplitude, future 1428-day timing, centerline, and symmetric future-envelope logic unchanged.
- Adds regression coverage for the data-derived dataset-start and pre-2015 peak anchors.

## v3.17.2 — reserved `pages/` routing migration

- Moves all Streamlit page scripts from the reserved `pages/` directory to `app_pages/`.
- Keeps `st.navigation` as the single navigation authority with explicit stable URL paths.
- Prevents Streamlit's legacy `_mpa_v1` router from running before `streamlit_app.py` on fresh Cloud processes.
- Adds a static navigation-layout regression test and a one-time deletion migration note.
- Does not change calibration mathematics or the frozen v3.12 Price Model.

## v3.17.0 — independent cycle-regime calibration

- Keeps the frozen Price Model v3.12 byte-for-byte unchanged.
- Makes realized trough→peak→trough Bitcoin regimes the primary independent K-learning unit instead of pooled turning-point amplitude residuals.
- Fits K against direct peak-price, trough-price, and realized peak→trough drawdown error so the optimizer can no longer win by collapsing K toward zero.
- Adds continuous small-sample shrinkage of the best cycle-regime K toward neutral K=1.0 based on the effective number of independent regimes.
- Uses a one-standard-error rule for structural G trust so a small structural improvement no longer automatically receives 100% correction weight.
- Allows the 2022 trough parent to earn its own partial out-of-sample evidence as soon as the frozen model's 1000-observation training minimum is reached; the completed 2025 peak can contribute before the future trough is confirmed.
- Fits K maturity trends only from complete independent cycle regimes; partial current-cycle evidence helps the level estimate but cannot manufacture trend confidence.
- Learns an uncertainty-shrunk mature bear-drawdown relationship from complete cycles and converts it into a data-derived future K floor in addition to the pure no-inversion geometry floor.
- Validates future geometry against both peak>trough and the evidence-backed minimum bear decline before FI can consume the calibrated path.
- Adds diagnostics for complete/partial cycle regimes, unshrunk vs shrunk K, K sample confidence, expected/required bear drawdown, geometric K, drawdown K, and effective K.

## v3.16.0 — cycle-disciplined dynamic calibration

- Keeps the frozen Price Model v3.12 byte-for-byte unchanged.
- Replaces calendar-year K extrapolation with independent Bitcoin cycle-index maturity learning.
- Collapses repeated forecasts of the same realized turning point before fitting the K trend, preventing pseudo-replication from overstating confidence.
- Learns an out-of-sample blend between constant K and trend K instead of choosing 100% one or the other.
- Learns an out-of-sample structural trust weight so weak G evidence only partially adjusts the parent-ensemble centerline.
- Extends cycle-envelope evidence to as much as 96 months when history permits and reports direct next-cycle / second-cycle validation.
- Gives new trough-aligned parents a small maturity-matched historical prior when they do not yet own enough OOS evidence; their own evidence replaces the prior over time.
- Adds a mathematical peak→trough geometry floor for K derived from calibrated-centerline growth and raw cycle amplitudes; no discretionary K/price floor is used.
- FI now requires both calibration PASS and valid forward peak/trough geometry before using the calibrated path by default.
- Adds forward diagnostics for unconstrained K, minimum geometric K, effective K, and whether each turning point was geometry-constrained.

## v3.15.0 — dynamic cycle-parent ensemble + maturity trend

- Keeps the frozen Price Model v3.12 byte-for-byte unchanged.
- Separates the model-derived current bear trough price from the frozen schedule's fixed Oct-5-2026 trough date; the calibrated page now states this explicitly.
- Replaces selected-start dependence with a production cycle-aligned parent ensemble. Current observed trough parents are 2015, 2018, and 2022.
- Scores each parent using genuine out-of-sample structural/envelope evidence; newer parents start cautiously and can gain weight as more outcomes mature.
- Adds automatic future parent discovery after enough actual data exists around later cycle windows.
- Validates structural G independently; rejected structural calibration falls back to effective G = 1.0 instead of worsening the centerline.
- Tests constant cycle-amplitude K against a time-trending K under held-out envelope forecasts and uses the trend only when it wins out of sample.
- Projects an accepted K maturity trend forward with confidence shrinkage, allowing each future peak/trough to use a different learned amplitude factor.
- Makes the calibrated production path independent of the Price Model page's selected training start; that selection remains available only as a frozen-model research comparison.
- FI consumes the validated dynamic calibrated ensemble by default when calibration status is PASS.

## v3.14.1 — calibration schema hotfix

- Force-reloads a stale in-memory calibration engine after Streamlit hot deploys.
- Rejects saved calibration results that do not contain the v2 cycle-aware summary schema.
- Replaces UI KeyError crashes with an actionable rerun message.
- Adds a regression test covering v1-to-v2 calibration schema compatibility.

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

## v3.14.0 — Cycle-Aware Multi-Horizon Calibration

- Keeps the frozen Price Model v3.12 engine byte-for-byte unchanged.
- Replaces the first 12-month pooled calibration objective with a two-part walk-forward objective:
  - structural growth factor G from 12/24/36/48-month low-frequency realized growth;
  - cycle amplitude factor K from realized historical peak/trough envelope magnitudes.
- Uses a hard Jan 14, 2015 calibration floor.
- Starts true 4Y fake-today tests in Jan 2019 and true 8Y tests in Jan 2023.
- Uses 180-day trailing log-price medians for structural outcomes and 31-day turning-point medians for envelope outcomes.
- Reports structural and envelope held-out errors separately as well as a combined cycle-aware OOS score.
- Adds PASS / MODEST / UNSTABLE / INSUFFICIENT_EVIDENCE / NO_IMPROVEMENT calibration states; FI only auto-selects a PASS calibration.
- Adds detailed calibration-evidence tables for inspecting implied G/K observations.
- Uses explicit Streamlit navigation so the sidebar order is Data Management → Price Model → Calibrated Price Model → BTC Financial Independence.
