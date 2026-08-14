# Bitcoin Fair Value

Current model: `bottom-anchored-dynamic-settling-v0.8.0`

A single-page Streamlit research application for estimating Bitcoin fair value from bear-market bottom regions that settle gradually as new evidence arrives.

The public page shows:

- current Bitcoin price;
- a dynamic bottom-derived fair-value estimate;
- observed bottom and peak regions;
- a long-term chart with observed history and a mature-cycle decay projection;
- a Research Lab with empirical bottom-anchor timing, settling-speed calibration, leave-one-cycle-out dependence, bottom-definition sensitivity, fair-value calibration, and walk-forward validation;
- a copy-ready JSON diagnostic block for fast model review.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Project structure

```text
streamlit_app.py          Single public page and Research Lab
src/data_pipeline.py      Coin Metrics data loading and cache
src/price_model.py        Dynamic bottom, fair-value, sensitivity, and validation engine
scripts/                  Focused offline tests
docs/MODEL_METHOD.md      Method, assumptions, and research boundaries
```

The project intentionally contains no FI simulator, multipage router, legacy V2 model, separate calibration page, embedded virtual environment, IDE state, cache, or Git database.

## Research status

The model is explicitly `RESEARCH_ONLY`. Bitcoin has only a few independent completed cycles. The mature-cycle range measures sensitivity to transparent bottom definitions, the leave-one-cycle-out range measures dependence on any single historical cycle, and the anchor-timing range measures dependence on the estimated 2026 turning date. The public forming endpoint is marginalized across the empirical early, central, and late anchor models so one hard window boundary does not control it. None of these ranges is a probability interval or guaranteed floor. The forming bottom remains provisional, and user-entered scenarios never become model evidence.

The public projection ends one bottom beyond the active rolling target. A fixed ten-year bottom line still requires validated recursive projection, while a ten-year fair-value line additionally requires a validated model of future fair-value-multiple or peak compression.

## Rolling lifecycle

New data changes the active region, evidence, bottom, fair-value, and sensitivity calculations automatically after the app receives it. The runtime price cache expires after 24 hours, and the sidebar button remains available for an immediate refresh.

The lifecycle engine has three explicit states: `pre_window`, `forming`, and `settled`. Once a complete bottom window closes, the region is promoted deterministically into the completed catalog; later prices cannot enter that closed window. Its empirical region date becomes the timing origin for a newly generated bottom target. The same mechanism creates and settles the next eligible peak region. Because the process is recalculated entirely from price history on every run, it needs no hidden database or manual cycle switch.

Automatic rollover does not make the forecasts certain. Timing is learned from very few cycles, and the public graph still shows only one bottom beyond the active target rather than an unvalidated fixed ten-year recursion.

## Data

Daily `PriceUSD` observations come from the Coin Metrics Community API and are cached in `/tmp` at runtime. No market dataset is committed to the project.

## Tests

```bash
python scripts/test_price_model.py
python scripts/test_lifecycle.py
python scripts/test_ui_contract.py
python scripts/test_project_manifest.py
```

## License

MIT. See `LICENSE`.
