import streamlit as st

st.set_page_config(
    page_title="Bitcoin FI Simulator",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def home_page():
    st.title("₿ Bitcoin FI Simulator")
    st.caption(
        "Explore Bitcoin price-model scenarios, walk-forward calibrated forecasts, "
        "financial-independence timelines, target contributions, and coast-FI plans."
    )
    st.warning(
        "Experimental educational simulation only. This application does not provide "
        "investment, tax, legal, or financial advice. Bitcoin projections are "
        "hypothetical model outputs and are not guarantees."
    )
    st.markdown(
        """
        ### Start here

        1. Open **Data Management** to review or refresh the Bitcoin dataset.
        2. Open **Price Model** to select the frozen v3.12 training period and projection horizon.
        3. Open **Calibrated Price Model** to run the 4Y + 8Y cycle-aware walk-forward calibration.
        4. Open **BTC Financial Independence** to model your FI plan using the validated calibrated path when available.

        Each visitor has an independent session. Your settings do not change the
        defaults or calculations shown to other visitors.
        """
    )


# Explicit navigation keeps the product workflow in the intended order and
# prevents filename prefixes (for example "2a_") from leaking into page labels.
pg = st.navigation([
    st.Page(home_page, title="Bitcoin FI Simulator", default=True),
    st.Page("pages/1_Data_Management.py", title="Data Management"),
    st.Page("pages/2_Price_Model.py", title="Price Model"),
    st.Page("pages/2a_Calibrated_Price_Model.py", title="Calibrated Price Model"),
    st.Page("pages/3_BTC_Financial_Independence.py", title="BTC Financial Independence"),
])
pg.run()
