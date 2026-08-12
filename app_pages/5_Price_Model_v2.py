import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from src.data_pipeline import load_coinmetrics
from src.price_model_v2 import (
    COMPLETE_CYCLES,
    get_cycle_anchor_df,
    fit_cycle_combo_centerlines,
    fit_progress_weighted_backbone,
    build_model_diagnostics,
)

st.title("Price Model v2 — Cycle-by-Cycle Power Law View")
st.caption("Find the stable long-term backbone, then measure how Bitcoin's cycle highs and lows compress around it.")

try:
    prices, meta = load_coinmetrics(refresh=False)
except Exception as exc:
    st.error(f"Coin Metrics data is unavailable: {exc}")
    st.stop()

anchor_df = get_cycle_anchor_df(prices)
fits_df, curves_df = fit_cycle_combo_centerlines(prices)
weighted_fit, weighted_curve = fit_progress_weighted_backbone(prices)
expanding_df, deviations_df = build_model_diagnostics(prices, fits_df)

palette = [
    "#F48FB1", "#81C784", "#FFB74D", "#64B5F6", "#BA68C8",
    "#4DD0E1", "#AED581", "#FF8A65", "#90CAF9", "#CE93D8",
]
fit_ids = fits_df["fit_id"].tolist()
fit_label_lookup = dict(zip(fits_df["fit_id"], fits_df["label"]))
color_lookup = {fit_id: palette[i % len(palette)] for i, fit_id in enumerate(fit_ids)}

complete_default = ["cycles_0_2", "live_2011"]

with st.sidebar:
    st.header("View settings")
    log_scale = st.toggle("Logarithmic price scale", value=True)
    show_extrapolated = True
    selected_fits = st.multiselect(
        "Centerlines to show",
        options=fit_ids,
        default=complete_default,
        format_func=lambda x: fit_label_lookup.get(x, x),
    )
    if not selected_fits:
        st.warning("Select at least one fitted centerline to display.")

    st.caption("The default view keeps only the longest completed-history fit and the comparable 2011→live fit.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Open-cycle progress", f"{weighted_fit['progress']:.1%}")
col2.metric("Estimated cycle end", f"{weighted_fit['estimated_cycle_end'].date()}")
col3.metric("Weighted backbone slope", f"{weighted_fit['slope']:.3f}")
col4.metric("Live ÷ centerline", f"{weighted_fit['live_actual_to_centerline']:.2f}×")

st.info(
    f"The 2022→live cycle is {weighted_fit['progress']:.1%} complete using the median length of the three completed cycles "
    f"({weighted_fit['expected_cycle_days']:,} days). Its observations therefore receive {weighted_fit['evidence_weight']:.1%} "
    "of a completed cycle's weight. It influences the backbone almost fully, but is not labeled complete."
)

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

fig.add_trace(
    go.Scatter(
        x=weighted_curve["date"],
        y=weighted_curve["centerline_price_usd"],
        mode="lines",
        name="Progress-weighted backbone",
        line=dict(color="#FFFFFF", width=3, dash="dash"),
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

st.subheader("Expanding-history convergence")
st.caption("Same 2011 start, with progressively more cycle history. This is the clean test of whether the backbone stabilizes.")
expanding_display = expanding_df.copy()
expanding_display["fit_start"] = pd.to_datetime(expanding_display["fit_start"]).dt.date
expanding_display["fit_end"] = pd.to_datetime(expanding_display["fit_end"]).dt.date
st.dataframe(
    expanding_display.style.format({"slope": "{:.4f}", "rmse_log": "{:.4f}", "r2_log": "{:.4f}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Cycle deviation from one shared centerline")
st.caption("Every peak and trough uses the same completed-history backbone. Falling peak multiples and changing trough multiples reveal maturity without moving the ruler.")
deviation_display = deviations_df.copy()
deviation_display["date"] = pd.to_datetime(deviation_display["date"]).dt.date
st.dataframe(deviation_display.style.format({"price_usd": "${:,.0f}", "shared_centerline_usd": "${:,.0f}", "actual_to_centerline": "{:.2f}×"}), use_container_width=True, hide_index=True)

st.subheader("Copy/paste results for analysis")
st.caption(
    "Copy everything in the block below and paste it back into ChatGPT. It contains only the "
    "cycle-progress, stable-backbone, and shared-deviation diagnostics needed for the next decision."
)

copy_text = (
    "PRICE MODEL V2 — COMPACT DIAGNOSTICS\n"
    + "CURRENT CYCLE EVIDENCE\n"
    + pd.DataFrame([weighted_fit]).drop(columns=["intercept"]).to_csv(index=False, sep="\t")
    + "\nEXPANDING-HISTORY BACKBONE\n"
    + expanding_display.to_csv(index=False, sep="\t")
    + "\nSHARED-CENTERLINE CYCLE DEVIATIONS\n"
    + deviation_display.to_csv(index=False, sep="\t")
)
st.code(copy_text, language="text")
