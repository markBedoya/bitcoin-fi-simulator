# Bitcoin FI Simulator

Bitcoin FI Simulator is an experimental Streamlit application for exploring:

- Bitcoin structural-centerline and cycle-adjusted price projections
- Earliest financial-independence age
- Target FI contribution requirements
- Coast-FI plans
- Mixed, 100% Bitcoin, and 100% traditional-investment comparisons

## Important disclaimer

This project is an educational simulation. It does not provide investment,
tax, legal, or financial advice. Bitcoin projections are hypothetical model
outputs and are not guarantees.

## Run locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

Use:

- Repository: `markBedoya/bitcoin-fi-simulator`
- Branch: `main`
- Entrypoint: `streamlit_app.py`
- Requested app URL: `bitcoin-fi-simulator`

## Development workflow

1. Create a branch.
2. Make and test one focused change.
3. Commit the change.
4. Push the branch.
5. Merge to `main` after review.
6. Streamlit Community Cloud redeploys from GitHub automatically.

See `docs/DEPLOYMENT.md` and `CONTRIBUTING.md`.
