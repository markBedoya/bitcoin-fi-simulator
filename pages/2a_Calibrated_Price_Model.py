import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.active_model_config import build_model_fingerprint
from src.data_pipeline import load_coinmetrics
from src.price_model import fit_price_model
from src.theme import REFERENCE_LINE_COLOR
import importlib
import src.walk_forward_calibration as _wfc

_EXPECTED_CALIBRATION_VERSION = "walk-forward-calibration-v2.0.1-cycle-aware"
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

st.title("Calibrated Price Model v2 — Multi-Horizon + Cycle-Aware")
st.caption(
    "The frozen Price Model v3.12 remains the parent model and is not modified. "
    "This calibration layer uses multi-horizon structural backtests to learn G and "
    "realized historical turning-point envelope errors to learn K. It then creates its "
    "own future centerline, peaks, troughs, and daily Bitcoin path for FI planning."
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
    training_start = pd.Timestamp(active["training_start"])
    training_end = pd.Timestamp(active["training_end"])
    projection_years = int(active["projection_years"])
else:
    training_start = min_date
    training_end = max_date
    projection_years = 10

try:
    base_model = fit_price_model(
        prices=prices,
        training_start=training_start,
        training_end=training_end,
        projection_years=projection_years,
    )
except Exception as exc:
    st.error(f"Frozen Price Model v3.12 could not be fitted: {exc}")
    st.stop()

base_fingerprint = build_model_fingerprint(
    training_start=training_start,
    training_end=training_end,
    projection_years=projection_years,
    latest_data_date=prices["date"].max(),
    model_daily=base_model.daily,
)

st.info(
    f"Parent Price Model: **{training_start.date()} → {training_end.date()}**, "
    f"**{projection_years} years**, fingerprint **{base_fingerprint}**.  \n"
    f"Walk-forward calibration floor: **{CALIBRATION_FLOOR.date()}**."
)

st.subheader("Walk-forward calibration")
st.markdown(
    """
The calibration runs fake-**today** experiments every six months while enforcing a hard
**Jan 14, 2015** training-data floor. A 4-year test can therefore begin in **Jan 2019**;
an 8-year test begins in **Jan 2023**. Each fake model sees only information available at
that fake date.

The two corrections are learned differently on purpose:

- **G — structural growth factor:** learned from realized low-frequency Bitcoin growth at
  **12, 24, 36, and 48 months** versus the frozen model's centerline growth. Realized
  structural level uses a trailing 180-day median of log price rather than a single day.
- **K — cycle amplitude factor:** learned from how large the frozen model said future
  historical peaks/troughs would be versus what actually materialized. Realized turning
  points use a 31-day median around the anchor so one wick cannot dominate the answer.

This makes K specifically answer the question we care about: **how much of the frozen
model's projected cycle envelope historically materialized?** Longer-horizon evidence
receives more weight, and held-out fake dates are still scored using factors learned only
from the other dates.
    """
)

existing = st.session_state.get("walk_forward_calibration_result")
if existing is not None and not calibration_is_current(existing, prices):
    st.session_state.pop("walk_forward_calibration_result", None)
    st.session_state.pop("active_calibrated_price_model_config", None)
    existing = None
    st.warning("Bitcoin data or the frozen model engine changed. The previous calibration was cleared and must be rerun.")

run = st.button(
    "Run / Update 4Y + 8Y Walk-Forward Calibration",
    type="primary",
    use_container_width=True,
)
if run:
    progress = st.progress(0, text="Preparing walk-forward tests...")

    def update_progress(done: int, total: int, label: str):
        progress.progress(
            int(done / max(total, 1) * 100),
            text=f"Walk-forward test {done} of {total}: {label}",
        )

    try:
        calibration = run_walk_forward_calibration(prices, progress_callback=update_progress)
        st.session_state["walk_forward_calibration_result"] = calibration
        progress.empty()
        st.rerun()
    except Exception as exc:
        progress.empty()
        st.error(f"Walk-forward calibration failed: {exc}")
        st.stop()

calibration = st.session_state.get("walk_forward_calibration_result")
if calibration is None:
    st.warning(
        "No calibrated model exists in this session yet. Run the walk-forward calibration above. "
        "BTC Financial Independence will continue using the frozen Price Model until a valid calibration is available."
    )
    st.stop()

summary = calibration.summary
missing_summary_keys = sorted(REQUIRED_SUMMARY_KEYS.difference(summary.keys()))
if missing_summary_keys:
    st.session_state.pop("walk_forward_calibration_result", None)
    st.session_state.pop("active_calibrated_price_model_config", None)
    st.warning(
        "The saved calibration result came from an older calibration schema and was cleared. "
        "Run the 4Y + 8Y walk-forward calibration again to create the cycle-aware v2 result."
    )
    st.caption("Missing fields: " + ", ".join(missing_summary_keys))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Structural growth factor G", f"{summary['growth_factor']:.3f}×")
c2.metric("Cycle amplitude factor K", f"{summary['amplitude_factor']:.3f}×")
c3.metric("Calibration stability", summary["stability"])
c4.metric("Calibration status", summary["status"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Raw cycle-aware OOS error", f"{summary['raw_cv_error']:.1%}")
m2.metric("Calibrated OOS error", f"{summary['calibrated_cv_error']:.1%}")
m3.metric("OOS improvement", f"{summary['cv_improvement']:+.1%}")
m4.metric("Walk-forward tests", f"{summary['total_tests']}")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Raw structural error", f"{summary['raw_structural_cv_error']:.1%}")
s2.metric("Calibrated structural error", f"{summary['calibrated_structural_cv_error']:.1%}")
s3.metric("Raw envelope error", f"{summary['raw_envelope_cv_error']:.1%}")
s4.metric("Calibrated envelope error", f"{summary['calibrated_envelope_cv_error']:.1%}")

status = summary["status"]
if status == "PASS":
    st.success(
        "Calibration validation: PASS — multi-horizon structural growth and historical cycle-envelope "
        "calibration reduce held-out error by at least 10% with acceptable stability. This path is "
        "eligible to become the default FI Bitcoin projection."
    )
elif status == "MODEST":
    st.warning(
        "Calibration validation: MODEST — held-out error improved, but by less than 10%. The calibrated "
        "path is shown for research, while FI stays on frozen v3.12 until the evidence becomes stronger."
    )
elif status == "UNSTABLE":
    st.warning(
        "Calibration validation: UNSTABLE — historical G/K estimates vary too much for FI to trust automatically."
    )
elif status == "INSUFFICIENT_EVIDENCE":
    st.warning(
        "Calibration validation: INSUFFICIENT EVIDENCE — more completed multi-year/turning-point outcomes are needed."
    )
else:
    st.warning(
        "Calibration validation: NO IMPROVEMENT — the cycle-aware held-out tests did not beat frozen v3.12, "
        "so FI remains on the frozen model."
    )

st.caption(
    f"Evidence: {summary['total_structural_points']} structural horizon observations and "
    f"{summary['total_envelope_points']} realized peak/trough envelope observations. "
    f"Maximum evaluation horizon: {summary['max_evaluation_months']} months."
)

l4 = summary["lookback_4y"]
l8 = summary["lookback_8y"]
st.subheader("4Y vs 8Y learned corrections")
lookback_table = pd.DataFrame([
    {
        "Lookback": "4 years",
        "G": l4["growth_factor"],
        "K": l4["amplitude_factor"],
        "Raw total OOS": l4["raw_cv_error"],
        "Calibrated total OOS": l4["calibrated_cv_error"],
        "Raw structural": l4["raw_structural_cv_error"],
        "Calibrated structural": l4["calibrated_structural_cv_error"],
        "Raw envelope": l4["raw_envelope_cv_error"],
        "Calibrated envelope": l4["calibrated_envelope_cv_error"],
        "Tests": l4["tests"],
        "Envelope outcomes": l4["envelope_points"],
        "Final weight": summary["lookback_weights"].get("4", np.nan),
    },
    {
        "Lookback": "8 years",
        "G": l8["growth_factor"],
        "K": l8["amplitude_factor"],
        "Raw total OOS": l8["raw_cv_error"],
        "Calibrated total OOS": l8["calibrated_cv_error"],
        "Raw structural": l8["raw_structural_cv_error"],
        "Calibrated structural": l8["calibrated_structural_cv_error"],
        "Raw envelope": l8["raw_envelope_cv_error"],
        "Calibrated envelope": l8["calibrated_envelope_cv_error"],
        "Tests": l8["tests"],
        "Envelope outcomes": l8["envelope_points"],
        "Final weight": summary["lookback_weights"].get("8", np.nan),
    },
])
st.dataframe(
    lookback_table.style.format({
        "G": "{:.3f}×", "K": "{:.3f}×",
        "Raw total OOS": "{:.1%}", "Calibrated total OOS": "{:.1%}",
        "Raw structural": "{:.1%}", "Calibrated structural": "{:.1%}",
        "Raw envelope": "{:.1%}", "Calibrated envelope": "{:.1%}",
        "Final weight": "{:.1%}",
    }),
    hide_index=True, use_container_width=True,
)

st.subheader("Historical fake-today tests")
tests = calibration.tests.copy()
tests["fake_today"] = pd.to_datetime(tests["fake_today"]).dt.date
tests["training_start"] = pd.to_datetime(tests["training_start"]).dt.date
show_cols = [
    "fake_today", "lookback_years", "training_start", "max_horizon_months",
    "structural_points", "envelope_points", "snapshot_growth_factor",
    "snapshot_amplitude_factor", "raw_error", "calibrated_error",
    "raw_structural_error", "calibrated_structural_error",
    "raw_envelope_error", "calibrated_envelope_error",
    "cv_growth_factor", "cv_amplitude_factor",
]
show_cols = [c for c in show_cols if c in tests.columns]
st.dataframe(
    tests[show_cols].style.format({
        "snapshot_growth_factor": "{:.3f}×", "snapshot_amplitude_factor": "{:.3f}×",
        "cv_growth_factor": "{:.3f}×", "cv_amplitude_factor": "{:.3f}×",
        "raw_error": "{:.1%}", "calibrated_error": "{:.1%}",
        "raw_structural_error": "{:.1%}", "calibrated_structural_error": "{:.1%}",
        "raw_envelope_error": "{:.1%}", "calibrated_envelope_error": "{:.1%}",
    }),
    hide_index=True, use_container_width=True,
)
st.caption(
    "4Y snapshots begin in Jan-2019; 8Y snapshots begin in Jan-2023. A snapshot can contribute "
    "12/24/36/48-month structural evidence as it matures, plus any realized historical cycle turning "
    "points inside its forward window."
)

with st.expander("Inspect calibration evidence"):
    obs = calibration.observations.copy()
    if obs.empty:
        st.caption("No detailed evidence rows are available.")
    else:
        structural_evidence = obs[obs["metric_type"] == "structural"].copy()
        envelope_evidence = obs[obs["metric_type"] == "envelope"].copy()
        t1, t2 = st.tabs(["Structural horizons", "Cycle envelope outcomes"])
        with t1:
            if structural_evidence.empty:
                st.caption("No structural evidence rows.")
            else:
                cols = ["fake_today", "lookback_years", "horizon_months",
                        "raw_structural_log_growth", "actual_structural_log_growth",
                        "implied_growth_factor", "evidence_weight"]
                st.dataframe(structural_evidence[cols], hide_index=True, use_container_width=True)
        with t2:
            if envelope_evidence.empty:
                st.caption("No realized turning-point outcomes are available yet.")
            else:
                cols = ["fake_today", "lookback_years", "anchor_date", "anchor_type",
                        "months_forward", "raw_projected_anchor_price_usd",
                        "actual_anchor_price_usd", "raw_amplitude",
                        "actual_amplitude_using_snapshot_G", "implied_amplitude_factor",
                        "evidence_weight"]
                st.dataframe(
                    envelope_evidence[cols].style.format({
                        "raw_projected_anchor_price_usd": "${:,.0f}",
                        "actual_anchor_price_usd": "${:,.0f}",
                        "implied_amplitude_factor": "{:.3f}×",
                    }), hide_index=True, use_container_width=True,
                )

calibrated = build_calibrated_price_model(base_model, calibration)
effective_fingerprint = build_calibrated_projection_fingerprint(base_fingerprint, calibrated)
st.session_state["active_calibrated_price_model_config"] = {
    "base_fingerprint": base_fingerprint,
    "calibration_fingerprint": calibration.fingerprint,
    "effective_fingerprint": effective_fingerprint,
    "growth_factor": float(summary["growth_factor"]),
    "amplitude_factor": float(summary["amplitude_factor"]),
    "status": summary["status"],
    "training_start": training_start.date().isoformat(),
    "training_end": training_end.date().isoformat(),
    "projection_years": int(projection_years),
    "latest_data_date": prices["date"].max().date().isoformat(),
}

st.subheader("Calibrated Bitcoin future projection")
st.caption(
    f"Effective calibrated fingerprint: **{effective_fingerprint}**. Calibration begins after "
    f"**{calibrated.diagnostics['calibration_start_date']}**. The currently conditioned 2025–26 bear path "
    "is preserved when it is still in progress; complete future cycles then use the new calibrated centerline and amplitude."
)

show_raw = st.toggle("Show frozen v3.12 comparison", value=False)
log_scale = st.toggle("Logarithmic price scale", value=True)

chart = go.Figure()
chart.add_trace(go.Scatter(
    x=prices["date"], y=prices["price_usd"], mode="lines", name="Actual Bitcoin price",
))
proj = calibrated.daily[calibrated.daily["row_type"] == "projected"]
chart.add_trace(go.Scatter(
    x=proj["date"],
    y=proj["calibrated_centerline_usd"],
    mode="lines",
    name="Calibrated centerline",
    line={"dash": "dash"},
))
chart.add_trace(go.Scatter(
    x=proj["date"],
    y=proj["calibrated_price_usd"],
    mode="lines",
    name="Calibrated future price",
    line={"width": 3},
))
if show_raw:
    chart.add_trace(go.Scatter(
        x=proj["date"], y=proj["raw_centerline_usd"], mode="lines",
        name="Frozen v3.12 centerline", line={"dash": "dot", "width": 1},
    ))
    chart.add_trace(go.Scatter(
        x=proj["date"], y=proj["raw_price_usd"], mode="lines",
        name="Frozen v3.12 future price", line={"dash": "dot", "width": 1},
    ))

turns = calibrated.turning_points
if not turns.empty:
    chart.add_trace(go.Scatter(
        x=turns["date"], y=turns["calibrated_price_usd"], mode="markers",
        name="Calibrated turning points",
        marker={
            "size": 10,
            "color": REFERENCE_LINE_COLOR,
            "symbol": "circle-open",
            "line": {"width": 2, "color": REFERENCE_LINE_COLOR},
        },
        hovertemplate="%{x|%Y-%m-%d}<br>Calibrated: $%{y:,.0f}<extra></extra>",
    ))

chart.add_vline(
    x=pd.Timestamp(calibrated.diagnostics["calibration_start_date"]).timestamp() * 1000,
    line_dash="dash",
    annotation_text="Calibration takes over",
    annotation_position="top",
    line_color=REFERENCE_LINE_COLOR,
)
chart.update_layout(
    xaxis_title="Date",
    yaxis_title="Bitcoin price (USD)",
    yaxis_type="log" if log_scale else "linear",
    hovermode="x unified",
    legend={"orientation": "h", "y": -0.18},
)
st.plotly_chart(chart, use_container_width=True)

if not turns.empty:
    st.subheader("Calibrated future turning points")
    st.dataframe(
        turns[[
            "date", "type", "raw_price_usd", "calibrated_price_usd",
            "raw_centerline_usd", "calibrated_centerline_usd",
            "raw_price_over_centerline", "calibrated_price_over_centerline",
        ]].style.format({
            "raw_price_usd": "${:,.0f}",
            "calibrated_price_usd": "${:,.0f}",
            "raw_centerline_usd": "${:,.0f}",
            "calibrated_centerline_usd": "${:,.0f}",
            "raw_price_over_centerline": "{:.3f}×",
            "calibrated_price_over_centerline": "{:.3f}×",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("Model chain")
st.code(
    "Frozen Price Model v3.12  →  Walk-Forward Calibration v2  →  "
    "Calibrated Price Model  →  BTC Financial Independence"
)
st.caption(
    f"Calibration engine: {CALIBRATION_VERSION}. The FI page will use this calibrated model by default "
    "when its status is PASS and its data/model fingerprints are current."
)
