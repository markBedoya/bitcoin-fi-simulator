# Contributing

## Branch naming

- `feature/<short-name>`
- `fix/<short-name>`
- `docs/<short-name>`

## Commit examples

- `feat(fi): add saved scenarios`
- `fix(price-model): preserve training cutoff`
- `docs: clarify Streamlit deployment`

## Before committing

```powershell
python -m compileall streamlit_app.py pages src
streamlit run streamlit_app.py
```

Confirm that:

- Data Management loads Coin Metrics data.
- Price Model changes update the active model fingerprint.
- BTC Financial Independence clears stale results after model changes.
- Visitors' settings remain session-specific.
