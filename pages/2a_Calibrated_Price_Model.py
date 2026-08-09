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

_EXPECTED_CALIBRATION_VERSION = "walk-forward-calibration-v3.0.0-dynamic-cycle-ensemble"
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

st.title("Calibrated Price Model v3 — Dynamic Cycle Ensemble")
st.caption(
    "Price Model v3.12 stays frozen. The calibrated model is now a separate production forecast: "
    "it learns from a growing set of cycle-aligned parent models, validates structural growth and "
    "cycle-envelope learning separately, and can project a maturity trend in future cycle amplitude."
)

try:
    prices, meta = load_coinmetrics(refresh=False)
except Exception as exc:
    st.error(f"Coin Metrics data is unavailable: {exc}")
    st.stop()

min_date = pd.Timestamp(prices["date"].min())
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

# The selected Price Model remains available as a research comparison only.
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
    "The **production calibrated projection no longer depends on this selected start date**. "
    "It builds its own cycle-aligned parent ensemble through the latest Bitcoin price."
)

st.subheader("How the dynamic calibration learns")
st.markdown(
    """
- **Rolling walk-forward evidence:** exact 4Y tests begin in Jan-2019 and exact 8Y tests begin in Jan-2023. Each fake-today model sees only data available then and can accumulate 12/24/36/48-month realized outcomes as time passes.
- **Cycle-aligned parent ensemble:** observed major troughs become independent parent models. Today that includes the 2015, 2018, and 2022 trough regimes. Future confirmed troughs can join automatically as enough actual data becomes available.
- **Structural G is optional:** it is applied only if it improves held-out structural forecasts. If it makes them worse, effective G is exactly 1.0.
- **Cycle-envelope K can trend:** the engine tests whether the historical K values are better explained by a maturity trend or by one constant factor. It uses whichever wins genuine held-out envelope forecasts.
- **Parent weights are learned:** each cycle-aligned parent earns influence from its out-of-sample accuracy and amount of matured evidence. New parents begin cautiously and can gain weight over time.
    """
)

st.warning(
    f"Timing note: the frozen model's exact next trough date **{pd.Timestamp(NEXT_TROUGH).date()}** still comes from its fixed "
    "1428-day cycle schedule. The trough **price** is model-derived from the live bear fit; this calibration version learns "
    "future price levels/amplitudes, not future turning-point dates."
)

existing = st.session_state.get("walk_forward_calibration_result")
if existing is not None and not calibration_is_current(existing, prices):
    st.session_state.pop("walk_forward_calibration_result", None)
    st.session_state.pop("active_calibrated_price_model_config", None)
    st.session_state.pop("active_calibrated_price_model_result", None)
    existing = None
    st.warning("Bitcoin data or the calibration engine changed. The previous calibration was cleared and must be rerun.")

run = st.button(
    "Run / Update Dynamic Walk-Forward Calibration",
    type="primary",
    use_container_width=True,
)
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
c3.metric("Current cycle K", f"{summary['amplitude_factor']:.3f}×")
c4.metric("Calibration status", summary["status"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Structural status", summary["growth_status"])
m2.metric("Envelope status", summary["envelope_status"])
m3.metric("Envelope mode", summary["amplitude_mode"])
m4.metric("Cycle parents", f"{summary['cycle_parent_count']}")

if summary["amplitude_mode"] == "TREND":
    annual = summary["amplitude_trend_annual_change"]
    st.success(
        f"Maturity trend selected: **{summary['amplitude_trend_direction']}**. "
        f"The conservatively shrunk K trend changes about **{annual:+.1%} per year** in multiplicative amplitude, "
        f"with confidence **{summary['amplitude_trend_confidence']:.1%}** and R² **{summary['amplitude_trend_r2']:.2f}**."
    )
else:
    st.info(
        "A constant K beat the trend model in held-out envelope forecasts, so the engine is not forcing a maturity trend. "
        "It will retest this automatically when new Bitcoin evidence is available."
    )

s1, s2, s3, s4 = st.columns(4)
s1.metric("Raw structural OOS", f"{summary['raw_structural_cv_error']:.1%}")
s2.metric("Effective structural OOS", f"{summary['calibrated_structural_cv_error']:.1%}")
s3.metric("Raw envelope OOS", f"{summary['raw_envelope_cv_error']:.1%}")
s4.metric("Calibrated envelope OOS", f"{summary['calibrated_envelope_cv_error']:.1%}")

e1, e2, e3 = st.columns(3)
e1.metric("Raw combined OOS", f"{summary['raw_cv_error']:.1%}")
e2.metric("Calibrated combined OOS", f"{summary['calibrated_cv_error']:.1%}")
e3.metric("Combined improvement", f"{summary['cv_improvement']:+.1%}")

if summary["growth_status"] == "REJECTED":
    st.warning(
        "Structural G was rejected because it did not improve held-out structural forecasts. "
        "The calibrated centerline therefore uses the cycle-parent ensemble with **effective G = 1.000×** instead of forcing a harmful structural correction."
    )

if summary["status"] == "PASS":
    st.success(
        "Calibration validation: PASS — cycle-envelope learning materially improves held-out forecasts. "
        "The calibrated path is eligible to be the default FI Bitcoin projection."
    )
elif summary["status"] == "MODEST":
    st.warning("Calibration validation: MODEST — improvement exists but is not yet strong enough to become the FI default.")
else:
    st.warning(f"Calibration validation: {summary['status']} — FI will remain on the frozen model for now.")

st.subheader("Cycle-aligned parent ensemble")
parents = pd.DataFrame(summary["cycle_parents"])
if not parents.empty:
    parents["start_date"] = pd.to_datetime(parents["start_date"]).dt.date
    st.dataframe(
        parents[[
            "start_date", "tests", "structural_points", "envelope_points",
            "raw_structural_error", "raw_envelope_error", "raw_total_error",
            "evidence_confidence", "weight",
        ]].style.format({
            "raw_structural_error": "{:.1%}", "raw_envelope_error": "{:.1%}",
            "raw_total_error": "{:.1%}", "evidence_confidence": "{:.1%}", "weight": "{:.1%}",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "The ensemble grows over time. A newer cycle parent starts with limited influence because it has little matured "
        "out-of-sample evidence; its weight can rise automatically as additional 12–48 month outcomes become known."
    )

st.subheader("4Y vs 8Y rolling evidence")
rows = []
for years in (4, 8):
    item = summary[f"lookback_{years}y"]
    rows.append({
        "Lookback": f"{years} years",
        "G": item["growth_factor"], "K": item["amplitude_factor"],
        "Raw structural": item["raw_structural_cv_error"], "Calibrated structural": item["calibrated_structural_cv_error"],
        "Raw envelope": item["raw_envelope_cv_error"], "Calibrated envelope": item["calibrated_envelope_cv_error"],
        "Tests": item["tests"], "Envelope outcomes": item["envelope_points"],
    })
st.dataframe(
    pd.DataFrame(rows).style.format({
        "G": "{:.3f}×", "K": "{:.3f}×",
        "Raw structural": "{:.1%}", "Calibrated structural": "{:.1%}",
        "Raw envelope": "{:.1%}", "Calibrated envelope": "{:.1%}",
    }),
    hide_index=True, use_container_width=True,
)

st.subheader("Amplitude maturity evidence")
trend_obs = calibration.observations.copy()
if not trend_obs.empty and "metric_type" in trend_obs.columns:
    trend_obs = trend_obs[trend_obs["metric_type"] == "amplitude_trend"].copy()
else:
    trend_obs = pd.DataFrame()
if trend_obs.empty:
    st.caption("Not enough realized envelope evidence exists yet to display a K trend.")
else:
    trend_chart = go.Figure()
    trend_chart.add_trace(go.Scatter(
        x=trend_obs["date"], y=trend_obs["factor"], mode="markers+lines", name="Observed walk-forward K",
    ))
    if "predicted_factor" in trend_obs.columns:
        trend_chart.add_trace(go.Scatter(
            x=trend_obs["date"], y=trend_obs["predicted_factor"], mode="lines", name="Shrunk maturity trend", line={"dash": "dash"},
        ))
    trend_chart.update_layout(xaxis_title="Fake today", yaxis_title="Cycle amplitude factor K", hovermode="x unified")
    st.plotly_chart(trend_chart, use_container_width=True)

with st.expander("Historical rolling fake-today tests"):
    tests = calibration.tests.copy()
    for col in ("fake_today", "training_start"):
        if col in tests.columns:
            tests[col] = pd.to_datetime(tests[col]).dt.date
    cols = [
        "fake_today", "lookback_years", "training_start", "max_horizon_months",
        "structural_points", "envelope_points", "snapshot_growth_factor", "snapshot_amplitude_factor",
        "cv_growth_factor", "cv_amplitude_factor", "raw_structural_error", "calibrated_structural_error",
        "raw_envelope_error", "calibrated_envelope_error",
    ]
    cols = [c for c in cols if c in tests.columns]
    st.dataframe(tests[cols], hide_index=True, use_container_width=True)

# Build the production calibrated model from the dynamic parent ensemble.  The
# selected frozen model supplies only the requested projection horizon/comparison.
try:
    calibrated = build_calibrated_price_model(selected_base_model, calibration, prices=prices)
except Exception as exc:
    st.error(f"Calibrated parent ensemble could not be built: {exc}")
    st.stop()

effective_fingerprint = build_calibrated_projection_fingerprint(selected_base_fingerprint, calibrated)
st.session_state["active_calibrated_price_model_result"] = calibrated
st.session_state["active_calibrated_price_model_config"] = {
    "calibration_fingerprint": calibration.fingerprint,
    "effective_fingerprint": effective_fingerprint,
    "effective_growth_factor": float(summary["effective_growth_factor"]),
    "current_amplitude_factor": float(summary["amplitude_factor"]),
    "amplitude_mode": summary["amplitude_mode"],
    "status": summary["status"],
    "projection_years": int(projection_years),
    "latest_data_date": prices["date"].max().date().isoformat(),
    "cycle_parents": summary["cycle_parents"],
}

st.subheader("Calibrated Bitcoin future projection")
st.caption(
    f"Effective fingerprint **{effective_fingerprint}**. The calibrated forecast is based on the cycle-parent ensemble, "
    f"not the selected Price Model start. Live-cycle pricing is preserved through **{calibrated.diagnostics['calibration_start_date']}**; "
    "complete future cycles then use the learned centerline ensemble and the validated K mode."
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
    st.dataframe(
        turns[[
            "date", "type", "raw_price_usd", "calibrated_price_usd",
            "raw_centerline_usd", "calibrated_centerline_usd",
            "raw_price_over_centerline", "calibrated_price_over_centerline", "amplitude_factor_K",
        ]].style.format({
            "raw_price_usd": "${:,.0f}", "calibrated_price_usd": "${:,.0f}",
            "raw_centerline_usd": "${:,.0f}", "calibrated_centerline_usd": "${:,.0f}",
            "raw_price_over_centerline": "{:.3f}×", "calibrated_price_over_centerline": "{:.3f}×",
            "amplitude_factor_K": "{:.3f}×",
        }),
        hide_index=True, use_container_width=True,
    )

st.subheader("Model chain")
st.code(
    "Frozen Price Model v3.12  →  Growing cycle-aligned parent ensemble  →  "
    "Walk-forward structural/envelope learning  →  Calibrated Price Model  →  FI"
)
st.caption(
    f"Calibration engine: {CALIBRATION_VERSION}. New actual Bitcoin data can mature existing tests, add new fake-today tests, "
    "change parent reliability weights, strengthen/weaken the K trend, and eventually add newly confirmed cycle parents."
)
