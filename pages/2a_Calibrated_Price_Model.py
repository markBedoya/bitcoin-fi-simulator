import importlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.active_model_config import build_model_fingerprint
from src.data_pipeline import load_coinmetrics
from src.price_model import NEXT_TROUGH, fit_price_model
from src.theme import REFERENCE_LINE_COLOR
import src.walk_forward_calibration as _wfc

_EXPECTED_CALIBRATION_VERSION = "walk-forward-calibration-v4.0.0-cycle-disciplined-learning"
if getattr(_wfc, "CALIBRATION_VERSION", None) != _EXPECTED_CALIBRATION_VERSION:
    importlib.invalidate_caches()
    _wfc = importlib.reload(_wfc)

CALIBRATION_FLOOR = _wfc.CALIBRATION_FLOOR
CALIBRATION_VERSION = _wfc.CALIBRATION_VERSION
REQUIRED_SUMMARY_KEYS = _wfc.REQUIRED_SUMMARY_KEYS
build_calibrated_price_model = _wfc.build_calibrated_price_model
build_calibrated_projection_fingerprint = _wfc.build_calibrated_projection_fingerprint
calibration_is_current = _wfc.calibration_is_current
run_walk_forward_calibration = _wfc.run_walk_forward_calibration

st.title("Calibrated Price Model v4 — Cycle-Disciplined Learning")
st.caption(
    "Price Model v3.12 remains frozen. This production forecast learns from a growing cycle-aligned parent ensemble, "
    "separates structural and cycle-envelope learning, measures maturity by Bitcoin cycle rather than calendar year, "
    "and enforces mathematically valid future peak/trough geometry."
)

try:
    prices, meta = load_coinmetrics(refresh=False)
except Exception as exc:
    st.error(f"Coin Metrics data is unavailable: {exc}")
    st.stop()

max_date = pd.Timestamp(prices["date"].max())
active = st.session_state.get("active_price_model_config")
if active:
    selected_training_start = pd.Timestamp(active["training_start"])
    selected_training_end = pd.Timestamp(active["training_end"])
    projection_years = int(active["projection_years"])
else:
    selected_training_start = pd.Timestamp("2018-12-15")
    selected_training_end = max_date
    projection_years = 10

try:
    selected_base_model = fit_price_model(
        prices=prices,
        training_start=selected_training_start,
        training_end=selected_training_end,
        projection_years=projection_years,
    )
except Exception as exc:
    st.error(f"Frozen Price Model v3.12 could not be fitted: {exc}")
    st.stop()

selected_base_fingerprint = build_model_fingerprint(
    training_start=selected_training_start,
    training_end=selected_training_end,
    projection_years=projection_years,
    latest_data_date=prices["date"].max(),
    model_daily=selected_base_model.daily,
)

st.info(
    f"Selected frozen Price Model comparison: **{selected_training_start.date()} → {selected_training_end.date()}**, "
    f"**{projection_years} years**, fingerprint **{selected_base_fingerprint}**.  \n"
    "The **production calibrated projection is independent of this selected start date**. It automatically builds and "
    "scores every eligible cycle-aligned trough parent."
)

st.subheader("How v4 learns")
st.markdown(
    """
- **Growing parent ensemble:** 2015, 2018 and 2022 are current trough-aligned parents; future confirmed troughs can join automatically. A young parent can receive a small maturity-matched historical prior before it owns enough out-of-sample evidence.
- **Structural G has a learned trust weight:** the engine learns both a raw G and how much of G actually improved held-out structural forecasts. Weak evidence produces only a small centerline adjustment.
- **K maturity is cycle-indexed:** repeated forecasts of the same realized peak/trough are collapsed before fitting the maturity trend. Trend confidence therefore comes from independent realized Bitcoin cycles, not from counting the same market event many times.
- **Constant K and trend K are blended out of sample:** the engine learns how much to trust the maturity trend rather than choosing 100% constant or 100% trend.
- **Next-cycle / second-cycle evidence:** cycle-envelope tests can look as far as 96 months when history permits, so the system can directly evaluate the next one or two projected cycles.
- **Geometry is enforced mathematically:** if a very small K would make a projected trough exceed the preceding peak, the minimum K is derived from centerline growth and the raw peak/trough amplitudes. No target price or discretionary K floor is used.
    """
)

st.warning(
    f"Timing note: the frozen model's exact next trough date **{pd.Timestamp(NEXT_TROUGH).date()}** still comes from its fixed "
    "1428-day cycle schedule. The trough **price** is model-derived. This calibration version learns future price levels, "
    "centerline behavior and maturity of cycle amplitude; it does not yet learn future turning-point dates."
)

existing = st.session_state.get("walk_forward_calibration_result")
if existing is not None and not calibration_is_current(existing, prices):
    st.session_state.pop("walk_forward_calibration_result", None)
    st.session_state.pop("active_calibrated_price_model_config", None)
    st.session_state.pop("active_calibrated_price_model_result", None)
    existing = None
    st.warning("Bitcoin data or the calibration engine changed. The previous calibration was cleared and must be rerun.")

run = st.button("Run / Update Dynamic Walk-Forward Calibration", type="primary", use_container_width=True)
if run:
    progress = st.progress(0, text="Preparing rolling and cycle-parent walk-forward tests...")

    def update_progress(done: int, total: int, label: str):
        progress.progress(int(done / max(total, 1) * 100), text=f"Test {done} of {total}: {label}")

    try:
        calibration = run_walk_forward_calibration(prices, progress_callback=update_progress)
        st.session_state["walk_forward_calibration_result"] = calibration
        progress.empty()
        st.rerun()
    except Exception as exc:
        progress.empty()
        st.error(f"Dynamic walk-forward calibration failed: {exc}")
        st.stop()

calibration = st.session_state.get("walk_forward_calibration_result")
if calibration is None:
    st.warning(
        "No current calibrated model exists in this session. Run the dynamic calibration above. "
        "FI will continue using the frozen Price Model until a validated calibrated result exists."
    )
    st.stop()

summary = calibration.summary
missing = sorted(REQUIRED_SUMMARY_KEYS.difference(summary.keys()))
if missing:
    st.session_state.pop("walk_forward_calibration_result", None)
    st.session_state.pop("active_calibrated_price_model_config", None)
    st.session_state.pop("active_calibrated_price_model_result", None)
    st.warning("The saved calibration uses an older schema and was cleared. Rerun the dynamic calibration.")
    st.caption("Missing fields: " + ", ".join(missing))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Learned structural G", f"{summary['growth_factor']:.3f}×")
c2.metric("Effective structural G", f"{summary['effective_growth_factor']:.3f}×")
c3.metric("G trust weight", f"{summary['structural_blend_weight']:.0%}")
c4.metric("Calibration status", summary["status"])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Current cycle K", f"{summary['amplitude_factor']:.3f}×")
k2.metric("Constant K", f"{summary['amplitude_constant_factor']:.3f}×")
k3.metric("Trend trust weight", f"{summary['amplitude_trend_blend_weight']:.0%}")
k4.metric("Envelope mode", summary["amplitude_mode"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Structural status", summary["growth_status"])
m2.metric("Envelope status", summary["envelope_status"])
m3.metric("Independent cycle points", f"{summary.get('independent_cycle_points', 0)}")
m4.metric("Cycle parents", f"{summary['cycle_parent_count']}")

if summary["amplitude_mode"] == "BLENDED_TREND":
    st.success(
        f"Maturity trend: **{summary['amplitude_trend_direction']}**. After statistical shrinkage, the trend changes K by "
        f"about **{summary['amplitude_trend_change_per_cycle']:+.1%} per Bitcoin cycle**. The held-out tests trust "
        f"**{summary['amplitude_trend_blend_weight']:.0%}** of that trend and **{1-summary['amplitude_trend_blend_weight']:.0%}** "
        f"of the constant-K estimate. Independent-cycle trend confidence is **{summary['amplitude_trend_confidence']:.1%}** "
        f"with R² **{summary['amplitude_trend_r2']:.2f}**."
    )
elif summary["amplitude_mode"] == "CONSTANT":
    st.info("Held-out cycle evidence currently prefers a near-constant K. The maturity trend remains visible and will be retested as new cycles mature.")
else:
    st.warning("Envelope calibration did not improve held-out cycle forecasts, so FI will not trust this calibration automatically.")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Raw structural OOS", f"{summary['raw_structural_cv_error']:.1%}")
s2.metric("Effective structural OOS", f"{summary['calibrated_structural_cv_error']:.1%}")
s3.metric("Raw envelope OOS", f"{summary['raw_envelope_cv_error']:.1%}")
s4.metric("Calibrated envelope OOS", f"{summary['calibrated_envelope_cv_error']:.1%}")

e1, e2, e3 = st.columns(3)
e1.metric("Raw combined OOS", f"{summary['raw_cv_error']:.1%}")
e2.metric("Calibrated combined OOS", f"{summary['calibrated_cv_error']:.1%}")
e3.metric("Combined improvement", f"{summary['cv_improvement']:+.1%}")

if summary["status"] == "PASS":
    st.success("Calibration validation: PASS — held-out cycle-envelope learning materially improves the frozen model. Geometry will be checked again on the actual forward path before FI uses it.")
elif summary["status"] == "MODEST":
    st.warning("Calibration validation: MODEST — improvement exists but is not yet strong enough to become the FI default.")
else:
    st.warning(f"Calibration validation: {summary['status']} — FI will remain on the frozen model for now.")

st.subheader("Cycle-aligned parent ensemble")
parents = pd.DataFrame(summary["cycle_parents"])
if not parents.empty:
    parents["start_date"] = pd.to_datetime(parents["start_date"]).dt.date
    parent_cols = [
        "start_date", "parent_age_years", "tests", "structural_points", "envelope_points",
        "raw_structural_error", "raw_envelope_error", "raw_total_error",
        "evidence_confidence", "weight_source", "weight",
    ]
    parent_cols = [c for c in parent_cols if c in parents.columns]
    st.dataframe(
        parents[parent_cols].style.format({
            "parent_age_years": "{:.1f}", "raw_structural_error": "{:.1%}",
            "raw_envelope_error": "{:.1%}", "raw_total_error": "{:.1%}",
            "evidence_confidence": "{:.1%}", "weight": "{:.1%}",
        }), hide_index=True, use_container_width=True,
    )
    st.caption(
        "Parents with their own matured OOS forecasts are weighted from that evidence. A newer parent with insufficient outcomes can receive a small "
        "maturity-matched prior based on how older parents performed at the same model age; its own evidence replaces that prior as time passes."
    )

st.subheader("Direct next-cycle / second-cycle validation")
cycle_validation = pd.DataFrame(summary.get("direct_cycle_validation", []))
if cycle_validation.empty:
    st.caption("Not enough independent held-out cycle outcomes are available yet for this table.")
else:
    cycle_validation["Forecast target"] = cycle_validation["cycle_horizon"].map(lambda x: "Next cycle" if x == 1 else f"Cycle +{x}")
    st.dataframe(
        cycle_validation[["Forecast target", "observations", "raw_error", "calibrated_error", "improvement"]].style.format({
            "raw_error": "{:.1%}", "calibrated_error": "{:.1%}", "improvement": "{:+.1%}",
        }), hide_index=True, use_container_width=True,
    )
    st.caption("Envelope evidence can extend to 96 months when history permits, so older fake-today tests can evaluate a second future cycle rather than only the following 12 months.")

st.subheader("4Y vs 8Y rolling evidence")
rows = []
for years in (4, 8):
    item = summary[f"lookback_{years}y"]
    rows.append({
        "Lookback": f"{years} years", "G": item["growth_factor"], "K": item["amplitude_factor"],
        "Raw structural": item["raw_structural_cv_error"], "Calibrated structural": item["calibrated_structural_cv_error"],
        "Raw envelope": item["raw_envelope_cv_error"], "Calibrated envelope": item["calibrated_envelope_cv_error"],
        "Tests": item["tests"], "Envelope outcomes": item["envelope_points"],
    })
st.dataframe(
    pd.DataFrame(rows).style.format({
        "G": "{:.3f}×", "K": "{:.3f}×", "Raw structural": "{:.1%}", "Calibrated structural": "{:.1%}",
        "Raw envelope": "{:.1%}", "Calibrated envelope": "{:.1%}",
    }), hide_index=True, use_container_width=True,
)

st.subheader("Cycle-amplitude maturity evidence")
trend_obs = calibration.observations.copy()
if not trend_obs.empty and "metric_type" in trend_obs.columns:
    trend_obs = trend_obs[trend_obs["metric_type"] == "amplitude_cycle_trend"].copy()
else:
    trend_obs = pd.DataFrame()
if trend_obs.empty:
    st.caption("Not enough independent realized cycle evidence exists yet to display a maturity trend.")
else:
    trend_chart = go.Figure()
    trend_chart.add_trace(go.Scatter(x=trend_obs["cycle_index"], y=trend_obs["factor"], mode="markers+lines", name="Realized cycle K"))
    if "constant_factor" in trend_obs.columns:
        trend_chart.add_trace(go.Scatter(x=trend_obs["cycle_index"], y=trend_obs["constant_factor"], mode="lines", name="Constant K", line={"dash": "dot"}))
    if "trend_factor" in trend_obs.columns:
        trend_chart.add_trace(go.Scatter(x=trend_obs["cycle_index"], y=trend_obs["trend_factor"], mode="lines", name="Shrunk cycle trend", line={"dash": "dash"}))
    if "predicted_factor" in trend_obs.columns:
        trend_chart.add_trace(go.Scatter(x=trend_obs["cycle_index"], y=trend_obs["predicted_factor"], mode="lines", name="OOS-weighted K", line={"width": 3}))
    trend_chart.update_layout(xaxis_title="Bitcoin cycle index", yaxis_title="Cycle amplitude factor K", hovermode="x unified")
    st.plotly_chart(trend_chart, use_container_width=True)

with st.expander("Historical rolling fake-today tests"):
    tests = calibration.tests.copy()
    for col in ("fake_today", "training_start"):
        if col in tests.columns:
            tests[col] = pd.to_datetime(tests[col]).dt.date
    cols = [
        "fake_today", "lookback_years", "training_start", "max_horizon_months",
        "structural_points", "envelope_points", "snapshot_growth_factor", "snapshot_amplitude_factor",
        "cv_growth_factor", "cv_structural_blend_weight", "cv_effective_growth_factor",
        "raw_structural_error", "calibrated_structural_error",
    ]
    cols = [c for c in cols if c in tests.columns]
    st.dataframe(tests[cols], hide_index=True, use_container_width=True)

try:
    calibrated = build_calibrated_price_model(selected_base_model, calibration, prices=prices)
except Exception as exc:
    st.error(f"Calibrated parent ensemble could not be built: {exc}")
    st.stop()

geometry_valid = bool(calibrated.diagnostics.get("geometry_valid", False))
effective_fingerprint = build_calibrated_projection_fingerprint(selected_base_fingerprint, calibrated)
st.session_state["active_calibrated_price_model_result"] = calibrated
st.session_state["active_calibrated_price_model_config"] = {
    "calibration_fingerprint": calibration.fingerprint,
    "effective_fingerprint": effective_fingerprint,
    "effective_growth_factor": float(summary["effective_growth_factor"]),
    "structural_blend_weight": float(summary["structural_blend_weight"]),
    "current_amplitude_factor": float(summary["amplitude_factor"]),
    "amplitude_trend_blend_weight": float(summary["amplitude_trend_blend_weight"]),
    "amplitude_mode": summary["amplitude_mode"],
    "status": summary["status"],
    "geometry_valid": geometry_valid,
    "projection_years": int(projection_years),
    "latest_data_date": prices["date"].max().date().isoformat(),
    "cycle_parents": summary["cycle_parents"],
}

st.subheader("Forward geometry validation")
if geometry_valid:
    st.success("PASS — every displayed calibrated peak remains above the following calibrated trough.")
else:
    st.error("FAIL — forward cycle geometry is invalid. FI will not use this calibrated path.")
geometry_table = calibrated.diagnostics.get("geometry_constraint_table")
if geometry_table is not None and not geometry_table.empty:
    gt = geometry_table.copy()
    st.dataframe(
        gt.style.format({
            "peak_raw_amplitude": "{:.3f}", "trough_raw_amplitude": "{:.3f}",
            "calibrated_centerline_growth_log": "{:.3f}", "minimum_geometric_K": "{:.3f}×",
        }), hide_index=True, use_container_width=True,
    )
    st.caption("The geometric minimum is derived from each peak→trough pair. It is not a manually chosen K floor or target Bitcoin price.")

st.subheader("Calibrated Bitcoin future projection")
st.caption(
    f"Effective fingerprint **{effective_fingerprint}**. The calibrated forecast is based on the cycle-parent ensemble, not the selected Price Model start. "
    f"Live-cycle pricing is preserved through **{calibrated.diagnostics['calibration_start_date']}**; complete future cycles then use the learned centerline, "
    "cycle-index maturity model and any required mathematical geometry constraint."
)

show_selected_raw = st.toggle("Show selected frozen Price Model comparison", value=False)
log_scale = st.toggle("Logarithmic price scale", value=True)
chart = go.Figure()
chart.add_trace(go.Scatter(x=prices["date"], y=prices["price_usd"], mode="lines", name="Actual Bitcoin price"))
proj = calibrated.daily[calibrated.daily["row_type"] == "projected"].copy()
chart.add_trace(go.Scatter(x=proj["date"], y=proj["raw_centerline_usd"], mode="lines", name="Parent-ensemble centerline", line={"dash": "dot", "width": 1}))
chart.add_trace(go.Scatter(x=proj["date"], y=proj["calibrated_centerline_usd"], mode="lines", name="Calibrated centerline", line={"dash": "dash"}))
chart.add_trace(go.Scatter(x=proj["date"], y=proj["calibrated_price_usd"], mode="lines", name="Calibrated future price", line={"width": 3}))
if show_selected_raw:
    selected_proj = selected_base_model.daily[selected_base_model.daily["row_type"] == "projected"]
    chart.add_trace(go.Scatter(x=selected_proj["date"], y=selected_proj["structural_centerline_usd"], mode="lines", name="Selected v3.12 centerline", line={"dash": "dot", "width": 1}))
    chart.add_trace(go.Scatter(x=selected_proj["date"], y=selected_proj["fitted_or_projected_price_usd"], mode="lines", name="Selected v3.12 future price", line={"dash": "dot", "width": 1}))
turns = calibrated.turning_points
if not turns.empty:
    chart.add_trace(go.Scatter(
        x=turns["date"], y=turns["calibrated_price_usd"], mode="markers", name="Calibrated turning points",
        marker={"size": 10, "color": REFERENCE_LINE_COLOR, "symbol": "circle-open", "line": {"width": 2, "color": REFERENCE_LINE_COLOR}},
    ))
chart.add_vline(
    x=pd.Timestamp(calibrated.diagnostics["calibration_start_date"]).timestamp() * 1000,
    line_dash="dash", annotation_text="Calibration takes over", annotation_position="top", line_color=REFERENCE_LINE_COLOR,
)
chart.update_layout(
    xaxis_title="Date", yaxis_title="Bitcoin price (USD)", yaxis_type="log" if log_scale else "linear",
    hovermode="x unified", legend={"orientation": "h", "y": -0.18},
)
st.plotly_chart(chart, use_container_width=True)

if not turns.empty:
    st.subheader("Calibrated future turning points")
    turn_cols = [
        "date", "type", "cycle", "raw_price_usd", "calibrated_price_usd",
        "raw_centerline_usd", "calibrated_centerline_usd",
        "raw_price_over_centerline", "calibrated_price_over_centerline",
        "unconstrained_amplitude_factor_K", "minimum_geometric_K", "amplitude_factor_K", "geometry_constrained",
    ]
    turn_cols = [c for c in turn_cols if c in turns.columns]
    st.dataframe(
        turns[turn_cols].style.format({
            "raw_price_usd": "${:,.0f}", "calibrated_price_usd": "${:,.0f}",
            "raw_centerline_usd": "${:,.0f}", "calibrated_centerline_usd": "${:,.0f}",
            "raw_price_over_centerline": "{:.3f}×", "calibrated_price_over_centerline": "{:.3f}×",
            "unconstrained_amplitude_factor_K": "{:.3f}×", "minimum_geometric_K": "{:.3f}×", "amplitude_factor_K": "{:.3f}×",
        }), hide_index=True, use_container_width=True,
    )

st.subheader("Model chain")
st.code(
    "Frozen Price Model v3.12  →  Growing trough-aligned parent ensemble  →  "
    "OOS structural trust + cycle-index envelope learning  →  Geometry-valid Calibrated Price Model  →  FI"
)
st.caption(
    f"Calibration engine: {CALIBRATION_VERSION}. New Bitcoin prices can mature existing 12–96 month tests, add new fake-today tests, "
    "change parent reliability, alter structural/trend trust weights, and eventually add newly confirmed cycle parents."
)
