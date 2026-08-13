from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import load_coinmetrics
import src.price_model as price_model


st.set_page_config(
    page_title="Bitcoin Fair Value",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def money(value: float) -> str:
    return f"${value:,.0f}"


def copy_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): copy_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [copy_safe(item) for item in value]
    return value


def compact_records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    available = [column for column in columns if column in frame.columns]
    view = frame[available].copy()
    for column in view.columns:
        if "date" in column:
            view[column] = pd.to_datetime(view[column], errors="coerce").dt.date.astype(str)
    return copy_safe(view.to_dict(orient="records"))


with st.sidebar:
    st.header("Model controls")
    refresh = st.button("Refresh Bitcoin data", use_container_width=True)
    log_scale = st.toggle("Logarithmic chart", value=True)
    show_published = st.toggle("Show published formula benchmark", value=True)
    show_all_price = st.toggle("Show all-price backbone", value=False)
    st.divider()
    st.caption(
        "Research model. Turning dates identify approximate regime regions; "
        "the exact daily high or low may occur before or after them."
    )

try:
    prices, meta = load_coinmetrics(refresh=refresh)
    model = price_model.fit_bottom_anchored_model(prices)
except Exception as exc:
    st.error(f"The Bitcoin model could not be loaded: {exc}")
    st.stop()

summary = model.summary
latest_date = pd.Timestamp(summary["latest_date"])

st.title("₿ Bitcoin Fair Value")
st.caption(
    "A transparent bottom-anchored research model. Bitcoin's relatively stable bottom regions "
    "form the structural foundation; cycle-neutral fair value and maturing peaks are measured above it."
)
st.warning(
    "Experimental educational research only—not investment advice or a guaranteed price floor. "
    "The current model has very few independent Bitcoin cycles and remains research-only."
)

ratio = float(summary["price_to_experimental_fair_value"])
if ratio < 0.75:
    valuation_label = "Well below experimental fair value"
elif ratio < 0.95:
    valuation_label = "Below experimental fair value"
elif ratio <= 1.05:
    valuation_label = "Near experimental fair value"
else:
    valuation_label = "Above experimental fair value"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Bitcoin price", money(summary["latest_price_usd"]))
m2.metric("Experimental fair value", money(summary["experimental_fair_value_usd"]))
m3.metric("Price / fair value", f"{ratio:.3f}×")
m4.metric("Observed bottom region", money(summary["current_bottom_region_usd"]))
m5.metric("Valuation", valuation_label)

st.info(
    f"The current estimate is derived from an observed bottom foundation near "
    f"**{money(summary['current_bottom_foundation_usd'])}** and a cycle-neutral multiplier of "
    f"**{summary['cycle_neutral_fair_multiple']:.3f}×**. The published `5.82 × 0.71` benchmark is "
    f"**{money(summary['published_fair_value_benchmark_usd'])}**. Their proximity is encouraging, "
    "but it is not independent proof that either estimate is correct."
)

curve = model.curve
historical_curve = curve[curve["row_type"] == "historical"]
future_curve = curve[curve["row_type"] == "projected"]

figure = go.Figure()
future_band = future_curve[future_curve["date"] >= latest_date]
if not future_band.empty:
    figure.add_trace(go.Scatter(
        x=future_band["date"],
        y=future_band["experimental_fair_value_high_usd"],
        mode="lines",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
        name="Candidate high",
    ))
    figure.add_trace(go.Scatter(
        x=future_band["date"],
        y=future_band["experimental_fair_value_low_usd"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(247,147,26,0.12)",
        hoverinfo="skip",
        name="Candidate-model range",
    ))

figure.add_trace(go.Scatter(
    x=prices["date"],
    y=prices["price_usd"],
    mode="lines",
    name="Bitcoin price",
    line={"color": "#7EC8FF", "width": 2},
))
figure.add_trace(go.Scatter(
    x=historical_curve["date"],
    y=historical_curve["bottom_foundation_usd"],
    mode="lines",
    name="Bottom foundation (observed)",
    line={"color": "#72D572", "width": 3},
))
figure.add_trace(go.Scatter(
    x=future_curve["date"],
    y=future_curve["bottom_foundation_usd"],
    mode="lines",
    name="Bottom foundation (candidate ensemble)",
    line={"color": "#72D572", "width": 2, "dash": "dash"},
))
figure.add_trace(go.Scatter(
    x=historical_curve["date"],
    y=historical_curve["experimental_fair_value_usd"],
    mode="lines",
    name="Cycle-neutral fair value (descriptive)",
    line={"color": "#F7931A", "width": 3},
))
figure.add_trace(go.Scatter(
    x=future_curve["date"],
    y=future_curve["experimental_fair_value_usd"],
    mode="lines",
    name="Cycle-neutral fair value (candidate)",
    line={"color": "#F7931A", "width": 2, "dash": "dash"},
))
figure.add_trace(go.Scatter(
    x=model.bottom_regions["region_date"],
    y=model.bottom_regions["region_price_usd"],
    mode="markers",
    name="Bottom regions",
    marker={"size": 11, "color": "#72D572", "symbol": "circle-open", "line": {"width": 2}},
))
figure.add_trace(go.Scatter(
    x=model.peak_regions["region_date"],
    y=model.peak_regions["region_price_usd"],
    mode="markers",
    name="Peak regions",
    marker={"size": 10, "color": "#FFD166", "symbol": "diamond-open", "line": {"width": 2}},
))
if show_published:
    figure.add_trace(go.Scatter(
        x=curve["date"],
        y=curve["published_fair_value_usd"],
        mode="lines",
        name="Published 5.82 × 0.71 fair benchmark",
        line={"color": "#B39DDB", "width": 1.5, "dash": "dot"},
    ))
    figure.add_trace(go.Scatter(
        x=curve["date"],
        y=curve["published_bottom_usd"],
        mode="lines",
        name="Published 5.82 × 0.42 bottom benchmark",
        line={"color": "#90A4AE", "width": 1.25, "dash": "dot"},
    ))
if show_all_price:
    figure.add_trace(go.Scatter(
        x=model.all_price_curve["date"],
        y=model.all_price_curve["all_price_backbone_usd"],
        mode="lines",
        name="Cycle-balanced all-price backbone",
        line={"color": "#EF9A9A", "width": 1.5, "dash": "dashdot"},
    ))

figure.add_vline(
    x=str(latest_date.date()),
    line_dash="dot",
    line_color="#9E9E9E",
    annotation_text="Latest actual data",
)
figure.update_layout(
    title="Bitcoin price, bottom foundation, and experimental fair value",
    xaxis_title="Date",
    yaxis_title="Bitcoin price (USD)",
    yaxis_type="log" if log_scale else "linear",
    hovermode="x unified",
    height=690,
    legend={"orientation": "h", "y": -0.16},
    margin={"b": 120},
)
st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})

st.caption(
    "Solid green/orange lines describe observed history. Dashed lines depend on competing future bottom models. "
    "The shaded area is the full candidate-model range—not a statistical confidence interval."
)

with st.expander("How the model works", expanded=False):
    st.markdown(
        """
1. **Bottom regions:** each approximate bear-to-bull turning region is represented by the median of its seven lowest daily closes inside a 241-day window. This avoids letting one wick define the model.
2. **Bottom foundation:** observed bottom regions are joined in log-price space. They are the most stable structural evidence in the model.
3. **Cycle-neutral fair value:** each observed peak is measured above the bottom foundation. Fair value is the log midpoint between the bottom foundation and that cycle's peak multiple.
4. **Unknown future:** four transparent bottom models forecast the next region. Their influence is based on walk-forward errors against later bottom regions.
5. **No hidden assumption:** the `$100K–$120K` user scenario is displayed only in the Research Lab and never becomes training evidence.
        """
    )
    st.markdown(
        r"The current fair-value calculation is: "
        r"$F_t = B_t \times \sqrt{P_{peak}/B_{peak}}$, where $B$ is the bottom foundation."
    )

with st.expander("Research Lab — evidence, competing models, and scenarios", expanded=True):
    st.subheader("Observed bottom regions")
    bottom_view = model.bottom_regions[[
        "cycle", "label", "anchor_date", "region_date", "region_price_usd",
        "extreme_date", "extreme_price_usd", "bottom_to_bottom_multiple",
        "bottom_to_bottom_cagr", "status",
    ]].copy()
    st.dataframe(
        bottom_view.style.format({
            "region_price_usd": "${:,.2f}",
            "extreme_price_usd": "${:,.2f}",
            "bottom_to_bottom_multiple": "{:.2f}×",
            "bottom_to_bottom_cagr": "{:.1%}",
        }, na_rep="—"),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Anchor dates locate broad regime turns. Region dates come from the actual low-close cluster, "
        "so the exact market low is not forced onto the anchor date."
    )

    st.subheader("Peak compression above the bottom foundation")
    peak_view = model.peak_regions[[
        "cycle", "label", "region_date", "region_price_usd", "bottom_foundation_usd",
        "peak_to_bottom_foundation", "cycle_neutral_fair_multiple", "status",
    ]].copy()
    st.dataframe(
        peak_view.style.format({
            "region_price_usd": "${:,.2f}",
            "bottom_foundation_usd": "${:,.2f}",
            "peak_to_bottom_foundation": "{:.3f}×",
            "cycle_neutral_fair_multiple": "{:.3f}×",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader(f"Competing estimates for the {pd.Timestamp(summary['next_bottom_anchor']).date()} bottom region")
    forecast_view = model.candidate_forecasts[[
        "model", "predicted_bottom_usd", "ensemble_weight",
        "validation_holdouts", "approx_typical_pct_error",
    ]].copy()
    st.dataframe(
        forecast_view.style.format({
            "predicted_bottom_usd": "${:,.0f}",
            "ensemble_weight": "{:.1%}",
            "approx_typical_pct_error": "{:.1%}",
        }),
        hide_index=True,
        use_container_width=True,
    )
    f1, f2, f3 = st.columns(3)
    f1.metric("Validation-weighted estimate", money(summary["next_bottom_ensemble_usd"]))
    f2.metric("Lowest candidate", money(summary["next_bottom_candidate_low_usd"]))
    f3.metric("Highest candidate", money(summary["next_bottom_candidate_high_usd"]))
    st.warning(
        "The candidate range is wide because the historical evidence does not yet identify one reliable decay law. "
        "A lower mature-cycle outcome remains plausible, but it has not historically outperformed the fixed-formula models."
    )

    st.subheader("Your scenario — never used as model evidence")
    s1, s2, s3, s4 = st.columns(4)
    scenario_bottom_low = s1.number_input("2030 bottom low", 1_000.0, 10_000_000.0, 100_000.0, 5_000.0)
    scenario_bottom_high = s2.number_input("2030 bottom high", 1_000.0, 10_000_000.0, 120_000.0, 5_000.0)
    scenario_peak_low = s3.number_input("2029 peak low", 1_000.0, 10_000_000.0, 150_000.0, 5_000.0)
    scenario_peak_high = s4.number_input("2029 peak high", 1_000.0, 10_000_000.0, 180_000.0, 5_000.0)

    scenario_valid = scenario_bottom_low <= scenario_bottom_high < scenario_peak_low <= scenario_peak_high
    if scenario_valid:
        shallow_drawdown = 1.0 - scenario_bottom_high / scenario_peak_low
        deep_drawdown = 1.0 - scenario_bottom_low / scenario_peak_high
        st.info(
            f"This scenario implies a post-peak decline of roughly **{shallow_drawdown:.1%}–{deep_drawdown:.1%}**. "
            "It is economically coherent, but remains a user scenario until historical validation supports its decay rate."
        )
    else:
        shallow_drawdown = float("nan")
        deep_drawdown = float("nan")
        st.error("Scenario ordering must be: bottom low ≤ bottom high < peak low ≤ peak high.")

    forecast_chart = go.Figure(go.Bar(
        x=model.candidate_forecasts["model"],
        y=model.candidate_forecasts["predicted_bottom_usd"],
        marker_color="#F7931A",
        text=model.candidate_forecasts["predicted_bottom_usd"].map(lambda value: f"${value:,.0f}"),
        textposition="outside",
    ))
    if scenario_bottom_low <= scenario_bottom_high:
        forecast_chart.add_hrect(
            y0=scenario_bottom_low,
            y1=scenario_bottom_high,
            fillcolor="#72D572",
            opacity=0.16,
            line_width=0,
            annotation_text="User scenario only",
        )
    forecast_chart.update_layout(
        title="Next-bottom candidates vs your scenario",
        xaxis_title="Candidate model",
        yaxis_title="Projected bottom-region price (USD)",
        height=480,
    )
    st.plotly_chart(forecast_chart, use_container_width=True, config={"displaylogo": False})

    st.subheader("Walk-forward validation")
    st.caption(
        "For each row, the target bottom was hidden and the model used only earlier bottom regions. "
        "The forming 2026 region receives half the validation weight of a completed region."
    )
    walk_view = model.walk_forward[[
        "model", "target_date", "target_status", "training_bottoms",
        "predicted_price_usd", "actual_region_price_usd", "prediction_ratio",
        "absolute_log_error",
    ]].copy()
    st.dataframe(
        walk_view.style.format({
            "predicted_price_usd": "${:,.0f}",
            "actual_region_price_usd": "${:,.0f}",
            "prediction_ratio": "{:.3f}×",
            "absolute_log_error": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Structural comparisons")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bottom-derived fair value", money(summary["experimental_fair_value_usd"]))
    c2.metric("Published 5.82 × 0.71", money(summary["published_fair_value_benchmark_usd"]))
    c3.metric("All-price backbone", money(summary["all_price_backbone_usd"]))
    c4.metric("All-price exponent", f"{summary['all_price_backbone_exponent']:.3f}")
    st.caption(
        "The all-price backbone is retained only as a research comparison. It is not labeled fair value, "
        "because early bull-market peaks can pull it above the cycle-neutral price level."
    )

st.divider()
st.subheader("Copy/paste research diagnostics")
st.caption(
    "Use the copy button in the upper-right corner of this block and paste everything into ChatGPT. "
    "It contains the exact data, model version, evidence, candidate forecasts, validation, and your scenario."
)

diagnostics = {
    "diagnostics_schema": "bitcoin-bottom-fair-value-copy-block-v1",
    "deployment": {
        "model_version": price_model.MODEL_VERSION,
        "loaded_engine_source": getattr(price_model, "__file__", "UNKNOWN"),
    },
    "data": meta,
    "summary": copy_safe(summary),
    "bottom_regions": compact_records(model.bottom_regions, [
        "cycle", "label", "anchor_date", "region_date", "region_price_usd",
        "cluster_low_usd", "cluster_high_usd", "extreme_date", "extreme_price_usd",
        "bottom_to_bottom_multiple", "bottom_to_bottom_cagr", "status",
    ]),
    "peak_regions": compact_records(model.peak_regions, [
        "cycle", "label", "region_date", "region_price_usd", "bottom_foundation_usd",
        "peak_to_bottom_foundation", "cycle_neutral_fair_multiple", "status",
    ]),
    "candidate_forecasts": compact_records(model.candidate_forecasts, [
        "model_id", "model", "next_bottom_anchor", "predicted_bottom_usd",
        "ensemble_weight", "validation_holdouts", "approx_typical_pct_error",
    ]),
    "validation_summary": compact_records(model.validation_summary, [
        "model_id", "model", "holdouts", "effective_holdouts",
        "mean_absolute_log_error", "rms_log_error", "approx_typical_pct_error",
        "ensemble_weight",
    ]),
    "walk_forward": compact_records(model.walk_forward, [
        "model_id", "target_date", "target_status", "training_bottoms",
        "predicted_price_usd", "actual_region_price_usd", "prediction_ratio",
        "absolute_log_error", "evidence_weight",
    ]),
    "user_scenario_not_training_evidence": {
        "next_peak_anchor": summary["next_peak_anchor"],
        "peak_low_usd": scenario_peak_low,
        "peak_high_usd": scenario_peak_high,
        "next_bottom_anchor": summary["next_bottom_anchor"],
        "bottom_low_usd": scenario_bottom_low,
        "bottom_high_usd": scenario_bottom_high,
        "implied_drawdown_low": shallow_drawdown,
        "implied_drawdown_high": deep_drawdown,
        "geometry_valid": scenario_valid,
    },
}
diagnostics_text = json.dumps(copy_safe(diagnostics), indent=2, sort_keys=True, allow_nan=False)
st.code(diagnostics_text, language="json", wrap_lines=False)
st.download_button(
    "Download compact diagnostics TXT",
    data=diagnostics_text.encode("utf-8"),
    file_name="bitcoin_bottom_fair_value_diagnostics.txt",
    mime="text/plain",
)

st.caption(
    f"Data through {latest_date.date()} · Model {price_model.MODEL_VERSION} · "
    "Coin Metrics PriceUSD · MIT licensed"
)
