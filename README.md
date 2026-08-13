# Bitcoin Fair Value

A single-page Streamlit research application for estimating Bitcoin fair value from bear-market bottom regions that settle gradually as new evidence arrives.

The public page shows:

- current Bitcoin price;
- a dynamic bottom-derived fair-value estimate;
- observed bottom and peak regions;
- a long-term chart with observed history and clearly marked research projections;
- a Research Lab with bottom-definition sensitivity, fair-value calibration, dynamic-settling backtests, and competing internal bottom models;
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

The model is explicitly `RESEARCH_ONLY`. Bitcoin has only a few independent completed cycles. Candidate ranges are structural comparisons, not probability intervals or guaranteed floors. The forming bottom remains provisional, and user-entered scenarios never become model evidence.

## Data

Daily `PriceUSD` observations come from the Coin Metrics Community API and are cached in `/tmp` at runtime. No market dataset is committed to the project.

## Tests

```bash
python scripts/test_price_model.py
python scripts/test_ui_contract.py
python scripts/test_project_manifest.py
```

## License

MIT. See `LICENSE`.
