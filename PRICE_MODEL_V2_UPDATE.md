# Price Model V2 update

Replace the matching repository files with the files in this package.

This update:

- reduces Price Model V2 to the useful chart and diagnostics;
- adds a cycle-balanced backbone fit where each completed cycle has weight `1.0`;
- gives the open 2022-to-live cycle weight equal to its estimated completion percentage;
- treats October 2025 as the confirmed peak and the latest price as a forming trough;
- compares every peak, trough, and live observation against the same progress-weighted backbone;
- measures peak compression separately from the mature-cycle trough floor;
- shows how close the forming bottom is to the median 2015/2018/2022 trough multiple;
- keeps the forming trough partial until the expected cycle closes;
- adds a compact tab-separated copy/paste results block;
- fixes the Price Model V2 navigation route test.

Delete `app_pages/2a_Price_Model_v2.py` from the GitHub repository if it still exists. The active page is `app_pages/5_Price_Model_v2.py`.
