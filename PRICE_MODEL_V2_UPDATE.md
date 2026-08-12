# Price Model V2 update

Replace the matching repository files with the files in this package.

This update:

- reduces Price Model V2 to the useful chart and diagnostics;
- adds a cycle-balanced backbone fit where each completed cycle has weight `1.0`;
- gives the open 2022-to-live cycle weight equal to its estimated completion percentage;
- compares every peak and trough against one shared completed-history centerline;
- adds a compact tab-separated copy/paste results block;
- fixes the Price Model V2 navigation route test.

Delete `app_pages/2a_Price_Model_v2.py` from the GitHub repository if it still exists. The active page is `app_pages/5_Price_Model_v2.py`.
