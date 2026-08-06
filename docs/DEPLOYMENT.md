# Deployment

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Connect your GitHub account.
4. Create an app from an existing repository.
5. Select:
   - Repository: `markBedoya/bitcoin-fi-simulator`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: `bitcoin-fi-simulator`
6. Choose Python 3.12 in Advanced settings.
7. Deploy.

## Updating the public app

Push an approved commit to `main`. Streamlit Community Cloud reads GitHub as
the source of truth and redeploys automatically.

## Cloud behavior

- Coin Metrics data is cached in temporary cloud storage.
- Model settings are stored in each visitor's Streamlit session.
- No visitor can overwrite another visitor's active Price Model.
- The application currently has no user accounts or persistent saved scenarios.
