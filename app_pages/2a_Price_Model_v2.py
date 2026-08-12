import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from src.data_pipeline import load_coinmetrics
from src.price_model_v2 import (
    COMPLETE_CYCLES,
    LIVE_STARTS,
    get_cycle_anchor_df,
    fit_cycle_combo_centerlines,
)

st.title("Price Model v2 — Cycle-by-Cycle Power Law View")
st.caption(
    "This page is an exploratory comparison view. It overlays Bitcoin price history, the "
    "historical cycle anchor points we have established, and multiple power-law centerlines "
    "fit to individual complete cycles, contiguous combinations of complete cycles, and "
    "selected trough-to-live windows. The goal is to visually compare how each cycle-specific "
    "centerline behaves before making the next modeling decision."
)

st.info(
    "This page now includes the earlier 2011 → 2014 → 2015 cycle, the 2015 → 2017 → 2018 cycle, "
    "the 2018 → 2021 → 2022 cycle, every contiguous combination of those complete cycles, plus "
    "the additional 2011→live, 2015→live, and 2018→live exploratory fits you requested."
)

try:
    prices, meta = load_coinmetrics(refresh=False)
except Exception as exc:
    st.error(f"Coin Metrics data is unavailable: {exc}")
    st.stop()

anchor_df = get_cycle_anchor_df(prices)
fits_df, curves_df = fit_cycle_combo_centerlines(prices)

palette = [
    "#F48FB1", "#81C784", "#FFB74D", "#64B5F6", "#BA68C8",
    "#4DD0E1", "#AED581", "#FF8A65", "#90CAF9", "#CE93D8",
]
fit_ids = fits_df["fit_id"].tolist()
fit_label_lookup = dict(zip(fits_df["fit_id"], fits_df["label"]))
color_lookup = {fit_id: palette[i % len(palette)] for i, fit_id in enumerate(fit_ids)}

complete_default = fits_df.loc[fits_df["fit_group"] == "Complete cycle fits", "fit_id"].tolist()

with st.sidebar:
    st.header("View settings")
    log_scale = st.toggle("Logarithmic price scale", value=True)
    show_extrapolated = st.toggle(
        "Show each centerline across the full chart",
        value=True,
        help=(
            "When enabled, each fitted centerline is shown across the full visible history. "
            "The portion inside the actual fitted date window is emphasized with a thicker solid line."
        ),
    )
    selected_fits = st.multiselect(
        "Centerlines to show",
        options=fit_ids,
        default=complete_default,
        format_func=lambda x: fit_label_lookup.get(x, x),
    )
    if not selected_fits:
        st.warning("Select at least one fitted centerline to display.")

    st.markdown("### Complete cycles used")
    for cycle in COMPLETE_CYCLES:
        st.caption(
            f"**{cycle['name']}**  \n"
            f"Start: {cycle['start'].date()}  \n"
            f"Peak: {cycle['peak'].date()}  \n"
            f"End: {cycle['end'].date()}"
        )

    st.markdown("### Live-window fits used")
    latest_date = pd.Timestamp(prices["date"].max()).date()
    for item in LIVE_STARTS:
        st.caption(f"**{item['cycle_span']}**  \nStart: {item['start'].date()}  \nEnd: {latest_date}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("BTC observations", f"{len(prices):,}")
col2.metric("Historical anchors shown", f"{anchor_df['price_usd'].notna().sum():,}")
col3.metric("Total fits", f"{len(fits_df):,}")
col4.metric("Latest BTC date", f"{pd.Timestamp(prices['date'].max()).date()}")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=prices["date"],
        y=prices["price_usd"],
        mode="lines",
        name="Actual Bitcoin price",
        line=dict(color="#7EC8FF", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>Actual Bitcoin price</extra>",
    )
)

peak_anchors = anchor_df[(anchor_df["type"] == "peak") & anchor_df["price_usd"].notna()].copy()
trough_anchors = anchor_df[(anchor_df["type"] == "trough") & anchor_df["price_usd"].notna()].copy()

if not peak_anchors.empty:
    fig.add_trace(
        go.Scatter(
            x=peak_anchors["date"],
            y=peak_anchors["price_usd"],
            mode="markers+text",
            name="Peak anchors",
            marker=dict(symbol="diamond-open", size=12, color="#FFD54F", line=dict(width=2, color="#FFD54F")),
            text=peak_anchors["label"],
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>%{text}</extra>",
        )
    )

if not trough_anchors.empty:
    fig.add_trace(
        go.Scatter(
            x=trough_anchors["date"],
            y=trough_anchors["price_usd"],
            mode="markers+text",
            name="Trough anchors",
            marker=dict(symbol="circle-open", size=12, color="#FFD54F", line=dict(width=2, color="#FFD54F")),
            text=trough_anchors["label"],
            textposition="bottom center",
            textfont=dict(size=10),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>%{text}</extra>",
        )
    )

for fit_id in selected_fits:
    curve = curves_df[curves_df["fit_id"] == fit_id].copy()
    if curve.empty:
        continue
    color = color_lookup[fit_id]
    label = fit_label_lookup[fit_id]

    if show_extrapolated:
        fig.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["centerline_price_usd"],
                mode="lines",
                name=f"{label} (full)",
                showlegend=False,
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.40,
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>Extrapolated centerline</extra>",
            )
        )

    window_curve = curve[curve["inside_fit_window"]].copy()
    fig.add_trace(
        go.Scatter(
            x=window_curve["date"],
            y=window_curve["centerline_price_usd"],
            mode="lines",
            name=label,
            line=dict(color=color, width=3),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>" + label + "</extra>",
        )
    )

for cycle in COMPLETE_CYCLES:
    fig.add_vrect(
        x0=cycle["start"],
        x1=cycle["end"],
        line_width=0,
        fillcolor="#FFFFFF",
        opacity=0.03,
        layer="below",
        annotation_text=cycle["name"],
        annotation_position="top left",
        annotation_font_size=10,
    )

fig.update_layout(
    title="Bitcoin price history with cycle-anchor power-law centerlines",
    template="plotly_dark",
    height=820,
    legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
    margin=dict(l=30, r=30, t=70, b=30),
    xaxis_title="Date",
    yaxis_title="Bitcoin price (USD)",
)
fig.update_yaxes(type="log" if log_scale else "linear")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Fit summary")
st.caption(
    "Each row is a separate power-law centerline fit. The exponent is the slope in log(price) vs log(days since Bitcoin genesis). "
    "The start/end centerline values are the fitted fair-value line at the window boundaries."
)

display = fits_df.copy()
display["fit_start"] = pd.to_datetime(display["fit_start"]).dt.date
display["fit_end"] = pd.to_datetime(display["fit_end"]).dt.date
st.dataframe(
    display[
        [
            "fit_group",
            "label",
            "fit_start",
            "fit_end",
            "days_used",
            "slope",
            "rmse_log",
            "r2_log",
            "centerline_start_usd",
            "centerline_end_usd",
        ]
    ].style.format(
        {
            "slope": "{:.4f}",
            "rmse_log": "{:.4f}",
            "r2_log": "{:.4f}",
            "centerline_start_usd": "${:,.0f}",
            "centerline_end_usd": "${:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Anchor points used on this page")
st.dataframe(
    anchor_df.assign(date=anchor_df["date"].dt.date)
    [["date", "type", "label", "price_usd"]]
    .style.format({"price_usd": "${:,.2f}"}),
    use_container_width=True,
    hide_index=True,
)

st.markdown("### What to look for")
st.markdown(
    "- Compare the **individual complete-cycle fits** against the **combined complete-cycle fits**.  \n"
    "- Compare the earlier complete-cycle structure against the **2011→live**, **2015→live**, and **2018→live** fits.  \n"
    "- Look at where each line places Bitcoin's fair-value centerline **before**, **during**, and **after** its fitted window.  \n"
    "- Use this page as a visual lab before deciding how to update the main Price Model or the calibration logic."
)
