import streamlit as st

st.set_page_config(
    page_title="Bitcoin FI Simulator",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("₿ Bitcoin FI Simulator")
st.caption(
    "Explore Bitcoin price-model scenarios, financial-independence timelines, "
    "target contributions, and coast-FI plans."
)

st.warning(
    "Experimental educational simulation only. This application does not provide "
    "investment, tax, legal, or financial advice. Bitcoin projections are "
    "hypothetical model outputs and are not guarantees."
)

st.markdown(
    '''
    ### Start here

    1. Open **Data Management System** to review or refresh the Bitcoin dataset.
    2. Open **Price Model** to select the training period and projection horizon.
    3. Open **Price Model v2** for the cycle-by-cycle power-law comparison view.
    4. Open **BTC Financial Independence** to model your FI plan.

    Each visitor has an independent session. Your settings do not change the
    defaults or calculations shown to other visitors.
    '''
)
