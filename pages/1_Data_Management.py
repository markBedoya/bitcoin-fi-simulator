import pandas as pd
import streamlit as st
from src.data_pipeline import load_coinmetrics

st.title("Data Management System")
st.caption("Refresh and validate the Coin Metrics Community Bitcoin price cache.")

refresh = st.button("Refresh Coin Metrics data", type="primary")
try:
    df, meta = load_coinmetrics(refresh=refresh)
except Exception as exc:
    st.error(f"Coin Metrics data could not be loaded: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Source", meta["source"])
c2.metric("Rows", f'{meta["rows"]:,}')
c3.metric("First date", meta["first_date"])
c4.metric("Latest date", meta["latest_date"])

missing = pd.date_range(df["date"].min(), df["date"].max(), freq="D").difference(pd.DatetimeIndex(df["date"]))
st.subheader("Validation")
st.json({
    "missing_dates": int(len(missing)),
    "duplicate_dates": int(df["date"].duplicated().sum()),
    "invalid_prices": int((df["price_usd"] <= 0).sum()),
})
st.dataframe(df.tail(250), use_container_width=True, hide_index=True)
