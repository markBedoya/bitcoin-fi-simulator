from __future__ import annotations

import json
from importlib import reload

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import load_coinmetrics
import src.price_model as price_model


# Streamlit reruns the page in a long-lived process. Reloading prevents a newly
# deployed page from retaining an older price_model module in memory.
price_model = reload(price_model)
EXPECTED_SUMMARY_SCHEMA = "bitcoin-dynamic-settling-summary-v4"


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
        if "date" in column or "anchor" in column:
            view[column] = pd.to_datetime(view[column], errors="coerce").dt.date.astype(str)
    return copy_safe(view.to_dict(orient="records"))


with st.sidebar:
    st.header("Model controls")
    refresh = st.button("Refresh Bitcoin data", use_container_width=True)
    log_scale = st.toggle("Logarithmic main chart", value=True)
    show_all_price = st.toggle("Show all-price research curve", value=False)
    st.divider()
    st.caption(
        "Research model. Turning dates locate broad market regions; exact daily highs and lows can occur "
        "before or after them."
    )

try:
    prices, meta = load_coinmetrics(refresh=refresh)
    model = price_model.fit_bottom_anchored_model(prices)
except Exception as exc:
    st.error(f"The Bitcoin model could not be loaded: {exc}")
    st.stop()

summary = model.summary
required_summary_keys = {
    "summary_schema",
    "latest_date",
    "latest_price_usd",
    "price_to_dynamic_fair_value",
    "dynamic_fair_value_usd",
    "dynamic_settled_bottom_estimate_usd",
    "linear_window_progress",
    "forming_evidence_weight",
    "next_bottom_core_usd",
    "next_bottom_core_low_usd",
    "next_bottom_core_high_usd",
}
missing_summary_keys = sorted(required_summary_keys.difference(summary))
missing_result_sections = sorted(
    section for section in (
        "bottom_sensitivity",
        "mature_cycle_forecast",
        "settling_calibration",
        "settling_calibration_detail",
        "fair_value_methods",
        "dynamic_settling",
        "dynamic_settling_summary",
    )
    if not hasattr(model, section)
)
if (
    missing_summary_keys
    or missing_result_sections
    or summary.get("summary_schema") != EXPECTED_SUMMARY_SCHEMA
):
    loaded_version = getattr(price_model, "MODEL_VERSION", "unknown")
    st.error(
        "The page and model engine are from different project versions. Update both "
        "`streamlit_app.py` and `src/price_model.py` from the same ZIP, then reboot the Streamlit app."
    )
    st.code(
        json.dumps({
            "loaded_model_version": loaded_version,
            "expected_summary_schema": EXPECTED_SUMMARY_SCHEMA,
            "loaded_summary_schema": summary.get("summary_schema"),
            "missing_summary_keys": missing_summary_keys,
            "missing_result_sections": missing_result_sections,
        }, indent=2),
        language="json",
    )
    st.stop()
latest_date = pd.Timestamp(summary["latest_date"])

st.title("₿ Bitcoin Fair Value")
st.caption(
    "A bottom-anchored dynamic-settling model. Stable bear-market regions form the foundation, "
    "and fair value updates as an unfinished cycle supplies new evidence."
)
st.warning(
    "Experimental educational research only—not investment advice or a guaranteed price floor. "
    "Bitcoin provides only a few independent completed cycles."
)

ratio = float(summary["price_to_dynamic_fair_value"])
if ratio < 0.75:
    valuation_label = "Well below fair value"
elif ratio < 0.95:
    valuation_label = "Below fair value"
elif ratio <= 1.05:
    valuation_label = "Near fair value"
else:
    valuation_label = "Above fair value"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Bitcoin price", money(summary["latest_price_usd"]))
m2.metric("Dynamic fair value", money(summary["dynamic_fair_value_usd"]))
m3.metric("Price / fair value", f"{ratio:.3f}×")
m4.metric("Settling bottom estimate", money(summary["dynamic_settled_bottom_estimate_usd"]))
m5.metric("Valuation", valuation_label)

st.info(
    f"The forming bottom region is **{money(summary['forming_bottom_region_usd'])}**. "
    f"The window is **{summary['linear_window_progress']:.1%}** complete, while the calibrated historical "
    f"evidence weight is **{summary['forming_evidence_weight']:.1%}**. The observed region is blended "
    f"with the pre-observation estimate of **{money(summary['pre_observation_bottom_forecast_usd'])}**. "
    f"That produces a settling bottom estimate of **{money(summary['dynamic_settled_bottom_estimate_usd'])}** "
    f"and fair value of **{money(summary['dynamic_fair_value_usd'])}**."
)

curve = model.curve
historical_curve = curve[curve["row_type"] == "historical"]
future_curve = curve[curve["row_type"] == "projected"]

figure = go.Figure()
figure.add_trace(go.Scatter(
    x=prices["date"], y=prices["price_usd"], mode="lines", name="Bitcoin price",
    line={"color": "#7EC8FF", "width": 2},
))
figure.add_trace(go.Scatter(
    x=historical_curve["date"], y=historical_curve["bottom_foundation_usd"], mode="lines",
    name="Bottom foundation", line={"color": "#72D572", "width": 3},
))
figure.add_trace(go.Scatter(
    x=future_curve["date"], y=future_curve["bottom_foundation_usd"], mode="lines",
    name="Bottom foundation (research projection)",
    line={"color": "#72D572", "width": 2, "dash": "dash"},
))
figure.add_trace(go.Scatter(
    x=historical_curve["date"], y=historical_curve["dynamic_fair_value_usd"], mode="lines",
    name="Dynamic fair value", line={"color": "#F7931A", "width": 3},
))
figure.add_trace(go.Scatter(
    x=model.bottom_regions["region_date"], y=model.bottom_regions["region_price_usd"],
    mode="markers", name="Observed bottom regions",
    marker={"size": 11, "color": "#72D572", "symbol": "circle-open", "line": {"width": 2}},
))
figure.add_trace(go.Scatter(
    x=model.peak_regions["region_date"], y=model.peak_regions["region_price_usd"],
    mode="markers", name="Observed peak regions",
    marker={"size": 10, "color": "#FFD166", "symbol": "diamond-open", "line": {"width": 2}},
))
if show_all_price:
    figure.add_trace(go.Scatter(
        x=model.all_price_curve["date"], y=model.all_price_curve["all_price_backbone_usd"],
        mode="lines", name="Cycle-balanced all-price curve",
        line={"color": "#EF9A9A", "width": 1.5, "dash": "dashdot"},
    ))
figure.add_vline(
    x=str(latest_date.date()), line_dash="dot", line_color="#9E9E9E",
    annotation_text="Latest data",
)
figure.update_layout(
    title="Bitcoin price, dynamic bottom foundation, and fair value",
    xaxis_title="Date", yaxis_title="Bitcoin price (USD)",
    yaxis_type="log" if log_scale else "linear", hovermode="x unified",
    height=690, legend={"orientation": "h", "y": -0.16}, margin={"b": 120},
)
st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
p1, p2, p3 = st.columns(3)
p1.metric("2030 mature-cycle bottom", money(summary["next_bottom_core_usd"]))
p2.metric(
    "Definition range",
    f"{money(summary['next_bottom_core_low_usd'])}–{money(summary['next_bottom_core_high_usd'])}",
)
p3.metric("Projected bottom growth", f"{summary['next_bottom_core_multiple']:.3f}×")
st.caption(
    "The dashed green line extends the mature-cycle bottom estimate. Fair value stops at today because "
    "future peak compression has not been modeled. The definition range is not a confidence interval."
)

with st.expander("How the model works", expanded=False):
    st.markdown(
        """
1. **Bottom regions:** each turning area is represented by a cluster of low daily closes, so one wick does not define the model.
2. **Dynamic settling:** an internal forecast is blended with the observed forming region using an empirical settling-speed curve learned from completed bottoms.
3. **Bottom foundation:** settled regions are joined in log-price space, creating the structural curve beneath market cycles.
4. **Fair value:** four cycle-neutral definitions are tested on earlier completed cycles and combined using walk-forward validation weights.
5. **Mature-cycle projection:** decaying bottom growth across the 2015→2018, 2018→2022, and forming 2022→2026 transitions produces the core next-bottom estimate.
6. **Pre-observation prior:** several internal rules create a forecast before a forming bottom is observed. They help the 2026 estimate settle but are not presented as 2030 paths.
7. **Scenario separation:** user-entered future prices remain scenarios and never become training evidence.
        """
    )

with st.expander("Research Lab — calibration and evidence", expanded=True):
    st.subheader("The current cycle is still settling")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pre-observation estimate", money(summary["pre_observation_bottom_forecast_usd"]))
    c2.metric("Forming observed region", money(summary["forming_bottom_region_usd"]))
    c3.metric("Linear window progress", f"{summary['linear_window_progress']:.1%}")
    c4.metric("Empirical evidence", f"{summary['forming_evidence_weight']:.1%}")
    c5.metric("Dynamic settled estimate", money(summary["dynamic_settled_bottom_estimate_usd"]))
    st.caption(
        "The estimate is allowed to move while the current turning window is incomplete. It is not treated "
        "as a finished $60K bottom merely because that is the lowest region observed so far. Evidence weight "
        "describes blending influence, not the probability that the market bottom has occurred."
    )

    st.subheader("Empirical settling-speed calibration")
    calibration_chart = go.Figure()
    calibration_chart.add_trace(go.Scatter(
        x=model.settling_calibration["window_fraction"],
        y=model.settling_calibration["linear_time_weight"],
        mode="lines", name="Old linear schedule",
        line={"color": "#9E9E9E", "dash": "dot"},
    ))
    calibration_chart.add_trace(go.Scatter(
        x=model.settling_calibration["window_fraction"],
        y=model.settling_calibration["raw_empirical_weight"],
        mode="lines+markers", name="Raw completed-cycle curve",
        line={"color": "#72D572", "dash": "dash"},
    ))
    calibration_chart.add_trace(go.Scatter(
        x=model.settling_calibration["window_fraction"],
        y=model.settling_calibration["empirical_evidence_weight"],
        mode="lines+markers", name="Conservative calibrated curve",
        line={"color": "#B39DDB", "width": 3},
    ))
    calibration_chart.add_trace(go.Scatter(
        x=[summary["linear_window_progress"]],
        y=[summary["forming_evidence_weight"]],
        mode="markers+text", name="Today",
        text=[f"Today: {summary['forming_evidence_weight']:.1%}"],
        textposition="top center", marker={"size": 13, "color": "#F7931A"},
    ))
    calibration_chart.update_layout(
        title="How quickly historical bottom regions settled",
        xaxis_title="Fraction of turning window elapsed",
        yaxis_title="Evidence weight", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05], height=450, hovermode="x unified",
    )
    st.plotly_chart(calibration_chart, use_container_width=True, config={"displaylogo": False})
    st.caption(
        f"The raw curve uses {summary['settling_calibration_cycles']} completed bottom regions. "
        "Because the sample is small, it is conservatively shrunk toward the old linear schedule using "
        "two linear-prior cycle equivalents."
    )

    completed_paths = model.dynamic_settling[model.dynamic_settling["target_status"] == "completed"].copy()
    settling_chart = go.Figure()
    for cycle, group in completed_paths.groupby("target_cycle"):
        label = pd.Timestamp(group["target_anchor"].iloc[0]).year
        settling_chart.add_trace(go.Scatter(
            x=group["reveal_date"], y=group["dynamic_reference_fair_value_usd"],
            mode="lines+markers", name=f"{label} estimate as evidence arrived",
        ))
        settling_chart.add_trace(go.Scatter(
            x=group["reveal_date"],
            y=np.repeat(group["settled_reference_fair_value_usd"].iloc[-1], len(group)),
            mode="lines", name=f"{label} final settled reference", line={"dash": "dot"},
        ))
    settling_chart.update_layout(
        title="Historical fake-today test: fixed-date fair value settling with later bottom evidence",
        xaxis_title="Date evidence was revealed", yaxis_title="Fair value at the fixed reference date (USD)",
        height=470, hovermode="x unified",
    )
    st.plotly_chart(settling_chart, use_container_width=True, config={"displaylogo": False})

    settling_summary = model.dynamic_settling_summary.copy()
    st.dataframe(
        settling_summary.style.format({
            "first_dynamic_bottom_usd": "${:,.0f}", "latest_dynamic_bottom_usd": "${:,.0f}",
            "settled_bottom_usd": "${:,.0f}", "first_reference_fair_value_usd": "${:,.0f}",
            "latest_reference_fair_value_usd": "${:,.0f}",
            "settled_reference_fair_value_usd": "${:,.0f}", "first_bottom_error_pct": "{:.1%}",
            "latest_bottom_error_pct": "{:.1%}", "first_fair_value_error_pct": "{:.1%}",
            "latest_fair_value_error_pct": "{:.1%}",
        }, na_rep="—"),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "For completed cycles, the final column is calculated only after the full bottom window is known. "
        "The forming cycle has no final answer yet."
    )

    st.subheader("Bottom-definition sensitivity")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tested definitions", str(summary["bottom_sensitivity_variants"]))
    s2.metric(
        "Settling bottom range",
        f"{money(summary['bottom_sensitivity_dynamic_low_usd'])}–{money(summary['bottom_sensitivity_dynamic_high_usd'])}",
    )
    s3.metric(
        "Fair-value range",
        f"{money(summary['bottom_sensitivity_fair_low_usd'])}–{money(summary['bottom_sensitivity_fair_high_usd'])}",
    )
    s4.metric(
        "Fair-method range",
        f"{money(summary['dynamic_fair_value_low_usd'])}–{money(summary['dynamic_fair_value_high_usd'])}",
    )
    sensitivity_view = model.bottom_sensitivity[model.bottom_sensitivity["available"]].copy()
    sensitivity_chart = go.Figure()
    for statistic, group in sensitivity_view.groupby("statistic"):
        sensitivity_chart.add_trace(go.Scatter(
            x=group["dynamic_current_bottom_usd"], y=group["fair_value_if_multiplier_held_usd"],
            mode="markers", name=statistic.replace("_", " ").title(),
            text=(
                "window ±" + group["half_window_days"].astype(str)
                + "d · cluster " + group["cluster_days"].astype(str)
            ),
            hovertemplate="%{text}<br>bottom $%{x:,.0f}<br>fair $%{y:,.0f}<extra></extra>",
        ))
    sensitivity_chart.update_layout(
        title="How region width, cluster size, and statistic change today's estimate",
        xaxis_title="Dynamic settled bottom estimate (USD)",
        yaxis_title="Fair value with the current multiplier (USD)", height=430,
    )
    st.plotly_chart(sensitivity_chart, use_container_width=True, config={"displaylogo": False})

    st.subheader("Observed bottom regions")
    bottom_view = model.bottom_regions[[
        "cycle", "label", "anchor_date", "region_date", "region_price_usd",
        "extreme_date", "extreme_price_usd", "bottom_to_bottom_multiple",
        "bottom_to_bottom_cagr", "status",
    ]].copy()
    st.dataframe(
        bottom_view.style.format({
            "region_price_usd": "${:,.2f}", "extreme_price_usd": "${:,.2f}",
            "bottom_to_bottom_multiple": "{:.2f}×", "bottom_to_bottom_cagr": "{:.1%}",
        }, na_rep="—"),
        hide_index=True, use_container_width=True,
    )

    st.subheader("Fair-value definitions and validation weights")
    fair_view = model.fair_value_methods[[
        "method", "fair_multiple", "ensemble_weight", "completed_holdouts", "mean_combined_score",
    ]].copy()
    st.dataframe(
        fair_view.style.format({
            "fair_multiple": "{:.3f}×", "ensemble_weight": "{:.1%}",
            "mean_combined_score": "{:.3f}",
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Weights use completed-cycle walk-forward behavior. The incomplete current cycle is measured but "
        "does not decide the calibration weights."
    )

    st.subheader(f"Mature-cycle estimate for the {pd.Timestamp(summary['next_bottom_anchor']).date()} bottom region")
    core_view = model.mature_cycle_forecast[[
        "model", "mature_transitions", "observed_growth_path", "next_growth_multiple", "predicted_bottom_usd",
        "definition_range_low_usd", "definition_range_high_usd", "validation_status",
    ]].copy()
    st.dataframe(
        core_view.style.format({
            "next_growth_multiple": "{:.3f}×", "predicted_bottom_usd": "${:,.0f}",
            "definition_range_low_usd": "${:,.0f}", "definition_range_high_usd": "${:,.0f}",
        }),
        hide_index=True, use_container_width=True,
    )
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Core estimate", money(summary["next_bottom_core_usd"]))
    f2.metric(
        "Central definition range",
        f"{money(summary['next_bottom_core_low_usd'])}–{money(summary['next_bottom_core_high_usd'])}",
    )
    f3.metric(
        "Definition stress range",
        f"{money(summary['next_bottom_core_stress_low_usd'])}–{money(summary['next_bottom_core_stress_high_usd'])}",
    )
    f4.metric("Next growth multiple", f"{summary['next_bottom_core_multiple']:.3f}×")
    st.caption(
        f"Observed mature bottom growth: **{summary['mature_observed_growth_path']}**; "
        f"projected next growth: **{summary['next_bottom_core_multiple']:.3f}×**. "
        "The central range is the 10th–90th percentile across available bottom-window, cluster-size, "
        "and region-statistic definitions. It is a definition-sensitivity range, not a probability interval."
    )

    st.subheader("Your scenario — never used as model evidence")
    u1, u2, u3, u4 = st.columns(4)
    scenario_bottom_low = u1.number_input("2030 bottom low", 1_000.0, 10_000_000.0, 100_000.0, 5_000.0)
    scenario_bottom_high = u2.number_input("2030 bottom high", 1_000.0, 10_000_000.0, 120_000.0, 5_000.0)
    scenario_peak_low = u3.number_input("2029 peak low", 1_000.0, 10_000_000.0, 150_000.0, 5_000.0)
    scenario_peak_high = u4.number_input("2029 peak high", 1_000.0, 10_000_000.0, 180_000.0, 5_000.0)
    scenario_valid = scenario_bottom_low <= scenario_bottom_high < scenario_peak_low <= scenario_peak_high
    if scenario_valid:
        shallow_drawdown = 1.0 - scenario_bottom_high / scenario_peak_low
        deep_drawdown = 1.0 - scenario_bottom_low / scenario_peak_high
        st.info(
            f"This scenario implies a post-peak decline of roughly **{shallow_drawdown:.1%}–{deep_drawdown:.1%}**. "
            "It remains a scenario until later evidence can test it."
        )
    else:
        shallow_drawdown = float("nan")
        deep_drawdown = float("nan")
        st.error("Scenario ordering must be: bottom low ≤ bottom high < peak low ≤ peak high.")

    chart_labels = ["Mature-cycle model"]
    chart_values = [summary["next_bottom_core_usd"]]
    chart_high_errors = [summary["next_bottom_core_high_usd"] - summary["next_bottom_core_usd"]]
    chart_low_errors = [summary["next_bottom_core_usd"] - summary["next_bottom_core_low_usd"]]
    chart_colors = ["#B39DDB"]
    if scenario_valid:
        scenario_midpoint = (scenario_bottom_low + scenario_bottom_high) / 2.0
        chart_labels.append("Your scenario")
        chart_values.append(scenario_midpoint)
        chart_high_errors.append(scenario_bottom_high - scenario_midpoint)
        chart_low_errors.append(scenario_midpoint - scenario_bottom_low)
        chart_colors.append("#72D572")
    forecast_chart = go.Figure(go.Scatter(
        x=chart_labels,
        y=chart_values,
        mode="markers+text",
        text=[money(value) for value in chart_values],
        textposition="top center",
        marker={"size": 14, "color": chart_colors},
        error_y={
            "type": "data",
            "symmetric": False,
            "array": chart_high_errors,
            "arrayminus": chart_low_errors,
            "thickness": 2,
            "width": 12,
        },
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    forecast_chart.update_layout(
        title="2030 bottom: mature-cycle definition range vs your scenario",
        xaxis_title="", yaxis_title="Projected bottom-region price (USD)", height=430,
    )
    st.plotly_chart(forecast_chart, use_container_width=True, config={"displaylogo": False})

    st.subheader("Bottom-model walk-forward validation")
    st.caption(
        "Each target bottom is hidden in turn; a model can use only earlier regions. The forming cycle "
        "receives partial evidence weight."
    )
    walk_view = model.walk_forward[[
        "model", "target_date", "target_status", "training_bottoms", "predicted_price_usd",
        "actual_region_price_usd", "prediction_ratio", "absolute_log_error",
    ]].copy()
    st.dataframe(
        walk_view.style.format({
            "predicted_price_usd": "${:,.0f}", "actual_region_price_usd": "${:,.0f}",
            "prediction_ratio": "{:.3f}×", "absolute_log_error": "{:.3f}",
        }),
        hide_index=True, use_container_width=True,
    )

    if show_all_price:
        st.subheader("All-price research comparison")
        a1, a2, a3 = st.columns(3)
        a1.metric("Dynamic fair value", money(summary["dynamic_fair_value_usd"]))
        a2.metric("All-price curve", money(summary["all_price_backbone_usd"]))
        a3.metric("All-price exponent", f"{summary['all_price_backbone_exponent']:.3f}")
        st.caption(
            "The all-price curve is an internal diagnostic, not fair value. Early high prices can pull it "
            "above the cycle-neutral estimate."
        )

st.divider()
st.subheader("Copy/paste research diagnostics")
st.caption(
    "Use the copy button in the upper-right corner and paste the whole block into ChatGPT. It contains "
    "the model version, current estimate, calibration evidence, sensitivity tests, and your scenario."
)

diagnostics = {
    "diagnostics_schema": "bitcoin-dynamic-settling-copy-block-v4",
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
        "cycle", "label", "anchor_date", "region_date", "region_price_usd", "status",
    ]),
    "fair_value_methods": compact_records(model.fair_value_methods, [
        "method_id", "method", "fair_multiple", "ensemble_weight", "completed_holdouts",
        "mean_combined_score",
    ]),
    "fair_value_validation": compact_records(model.fair_value_validation, [
        "method_id", "method", "target_cycle", "target_status", "predicted_multiple",
        "realized_multiple", "median_absolute_log_error", "neutrality_error", "combined_score",
    ]),
    "settling_calibration": compact_records(model.settling_calibration, [
        "window_fraction", "completed_cycles", "median_settling_progress",
        "mean_settling_progress", "within_10_percent_share", "within_20_percent_share",
        "median_partial_log_error", "linear_time_weight", "raw_empirical_weight",
        "calibration_reliability", "empirical_evidence_weight",
    ]),
    "settling_calibration_detail": compact_records(model.settling_calibration_detail, [
        "target_cycle", "target_anchor", "window_fraction", "reveal_date",
        "partial_bottom_usd", "final_bottom_usd", "absolute_log_error",
        "initial_absolute_log_error", "settling_progress", "within_10_percent",
        "within_20_percent",
    ]),
    "mature_cycle_forecast": compact_records(model.mature_cycle_forecast, [
        "model_id", "model", "target_date", "mature_start_cycle", "mature_transitions",
        "includes_forming_current_bottom", "observed_growth_path", "next_growth_multiple", "predicted_bottom_usd",
        "definition_range_low_usd", "definition_range_high_usd", "definition_stress_low_usd",
        "definition_stress_high_usd", "definition_variants", "range_definition", "validation_status",
    ]),
    "forming_bottom_prior_forecasts": compact_records(model.forming_prior_forecasts, [
        "model_id", "model", "target_date", "predicted_bottom_usd", "ensemble_weight",
        "holdouts", "approx_typical_pct_error", "forecast_role",
    ]),
    "bottom_validation_summary": compact_records(model.validation_summary, [
        "model_id", "model", "holdouts", "effective_holdouts", "mean_absolute_log_error",
        "rms_log_error", "approx_typical_pct_error", "ensemble_weight",
    ]),
    "bottom_walk_forward": compact_records(model.walk_forward, [
        "model_id", "target_date", "target_status", "training_bottoms", "predicted_price_usd",
        "actual_region_price_usd", "prediction_ratio", "absolute_log_error", "evidence_weight",
    ]),
    "dynamic_settling_summary": compact_records(model.dynamic_settling_summary, [
        "target_cycle", "target_status", "target_anchor", "reference_date",
        "first_linear_window_progress", "first_empirical_evidence_weight",
        "latest_linear_window_progress", "latest_empirical_evidence_weight",
        "first_dynamic_bottom_usd", "latest_dynamic_bottom_usd", "settled_bottom_usd",
        "first_reference_fair_value_usd", "latest_reference_fair_value_usd",
        "settled_reference_fair_value_usd", "first_bottom_error_pct", "latest_bottom_error_pct",
        "first_fair_value_error_pct", "latest_fair_value_error_pct",
    ]),
    "bottom_sensitivity": compact_records(model.bottom_sensitivity, [
        "half_window_days", "cluster_days", "statistic", "available", "forming_region_usd",
        "linear_window_progress", "forming_evidence_weight", "dynamic_current_bottom_usd", "current_foundation_usd",
        "fair_value_if_multiplier_held_usd", "mature_cycle_next_multiple",
        "mature_cycle_next_bottom_usd",
    ]),
    "user_scenario_not_training_evidence": {
        "next_peak_anchor": summary["next_peak_anchor"], "peak_low_usd": scenario_peak_low,
        "peak_high_usd": scenario_peak_high, "next_bottom_anchor": summary["next_bottom_anchor"],
        "bottom_low_usd": scenario_bottom_low, "bottom_high_usd": scenario_bottom_high,
        "implied_drawdown_low": shallow_drawdown, "implied_drawdown_high": deep_drawdown,
        "geometry_valid": scenario_valid,
    },
}
diagnostics_text = json.dumps(copy_safe(diagnostics), indent=2, sort_keys=True, allow_nan=False)
st.code(diagnostics_text, language="json", wrap_lines=False)
st.download_button(
    "Download compact diagnostics TXT", data=diagnostics_text.encode("utf-8"),
    file_name="bitcoin_dynamic_settling_diagnostics.txt", mime="text/plain",
)

st.caption(
    f"Data through {latest_date.date()} · Model {price_model.MODEL_VERSION} · "
    "Coin Metrics PriceUSD · MIT licensed"
)
