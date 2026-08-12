# Price Model V2 update

Replace the matching repository files with the files in this package.

The ZIP preserves the repository paths:

- `app_pages/5_Price_Model_v2.py`
- `src/price_model_v2.py`
- `scripts/test_price_model_v2_cycle_fits.py`

This update:

- reduces Price Model V2 to the useful chart and diagnostics;
- adds a cycle-balanced backbone fit where each completed cycle has weight `1.0`;
- gives the open 2022-to-live cycle weight equal to its estimated completion percentage;
- treats October 2025 as the confirmed peak and the latest price as a forming trough;
- compares every peak, trough, and live observation against the same progress-weighted backbone;
- measures peak compression separately from the mature-cycle trough floor;
- shows how close the forming bottom is to the median 2015/2018/2022 trough multiple;
- keeps the forming trough partial until the expected cycle closes;
- preserves the lowest observed post-peak price as the forming trough even if price rebounds;
- measures the backbone's downward recalibration separately from peak compression;
- adds a compact maturity-transition summary instead of prematurely fitting a four-point forward curve;
- compares a one-cycle regime hold with bounded convergence toward the centerline;
- rejects an unbounded amplitude trend when it predicts a future peak below the centerline;
- adds a compact tab-separated copy/paste results block;
- converts only structurally valid envelope candidates into a next-cycle dollar range;
- learns next-peak timing from the median completed trough-to-peak offset;
- labels the geometric midpoint as planning math rather than a third fitted model;

Delete `app_pages/2a_Price_Model_v2.py` from the GitHub repository if it still exists. The active page is `app_pages/5_Price_Model_v2.py`.
# Forward structure diagnostics

The compact export now includes `FORWARD STRUCTURE CANDIDATES`. It compares a
one-cycle maturity-regime hold, bounded convergence of peak excess toward the
centerline, and an intentionally unbounded log-amplitude trend. The last row is
a rejection test: it is marked structurally invalid whenever it predicts a
cycle peak below the 1.0x centerline.

The current cycle remains partial. These rows test the shape of the future
envelope; they do not yet select a production projection or publish dollar
price forecasts.

# Next-cycle price envelope

The compact export now also includes `NEXT-CYCLE PRICE ENVELOPE`. It evaluates
the bounded-convergence and regime-hold multipliers against the weighted
backbone on the historically timed next peak date. The rejected unbounded model
is intentionally excluded from dollar projections.
