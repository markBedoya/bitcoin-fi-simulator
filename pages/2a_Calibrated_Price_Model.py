import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.active_model_config import build_model_fingerprint
from src.data_pipeline import load_coinmetrics
from src.price_model import fit_price_model
from src.theme import REFERENCE_LINE_COLOR
from src.walk_forward_calibration import (
    CALIBRATION_FLOOR,
    CALIBRATION_VERSION,
    build_calibrated_price_model,
    build_calibrated_projection_fingerprint,
    calibration_is_current,
    run_walk_forward_calibration,
)

st.title("Calibrated Price Model v1 — Walk-Forward Learned")
st.caption(
    "The frozen Price Model v3.12 remains the parent model. This page never changes its mathematics. "
    "Instead, standardized 4-year and 8-year historical walk-forward tests learn two corrections: "
    "G for future structural-centerline growth and K for future cycle amplitude. The calibrated model "
    "then produces its own future centerline, peaks, troughs, and daily Bitcoin path for FI planning."
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
The calibration runs standardized fake-**today** experiments every six months beginning in January 2023.
At every fake date it fits the frozen model twice — once with exactly 4 years of trailing history and once
with exactly 8 years — and then scores the next 12 months against Bitcoin prices that the fake model was
not allowed to see. No calibration training data is allowed before **Jan 14, 2015**.

The backtest learns only:

- **G — structural growth factor:** changes how quickly the *new calibrated centerline* grows in the future.
- **K — cycle amplitude factor:** changes the size of highs/lows around that calibrated centerline.

Both are regularized toward **1.000×**, so weak historical evidence cannot radically rewrite the frozen model.
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
c1, c2, c3, c4 = st.columns(4)
c1.metric("Structural growth factor G", f"{summary['growth_factor']:.3f}×")
c2.metric("Cycle amplitude factor K", f"{summary['amplitude_factor']:.3f}×")
c3.metric("Calibration stability", summary["stability"])
c4.metric("Calibration status", summary["status"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Raw OOS error", f"{summary['raw_cv_error']:.1%}")
m2.metric("Calibrated OOS error", f"{summary['calibrated_cv_error']:.1%}")
m3.metric("OOS improvement", f"{summary['cv_improvement']:+.1%}")
m4.metric("Walk-forward tests", f"{summary['total_tests']}")

if summary["status"] == "PASS":
    st.success(
        "Calibration validation: PASS — the learned factors are stable enough and reduce genuine held-out "
        "walk-forward error. This calibrated path is eligible to become the default FI Bitcoin projection."
    )
elif summary["status"] == "UNSTABLE":
    st.warning(
        "Calibration validation: UNSTABLE — the page will still show the learned projection for research, "
        "but FI will fall back to the frozen Price Model."
    )
else:
    st.warning(
        "Calibration validation: NO IMPROVEMENT — the learned projection is shown for research, but the "
        "held-out tests did not beat the frozen model, so FI will fall back to the frozen Price Model."
    )

l4 = summary["lookback_4y"]
l8 = summary["lookback_8y"]
st.subheader("4Y vs 8Y learned corrections")
lookback_table = pd.DataFrame([
    {
        "Lookback": "4 years",
        "G": l4["growth_factor"],
        "K": l4["amplitude_factor"],
        "Raw OOS error": l4["raw_cv_error"],
        "Calibrated OOS error": l4["calibrated_cv_error"],
        "Stability": l4["stability"],
        "Tests": l4["tests"],
        "Final weight": summary["lookback_weights"].get("4", np.nan),
    },
    {
        "Lookback": "8 years",
        "G": l8["growth_factor"],
        "K": l8["amplitude_factor"],
        "Raw OOS error": l8["raw_cv_error"],
        "Calibrated OOS error": l8["calibrated_cv_error"],
        "Stability": l8["stability"],
        "Tests": l8["tests"],
        "Final weight": summary["lookback_weights"].get("8", np.nan),
    },
])
st.dataframe(
    lookback_table.style.format({
        "G": "{:.3f}×",
        "K": "{:.3f}×",
        "Raw OOS error": "{:.1%}",
        "Calibrated OOS error": "{:.1%}",
        "Final weight": "{:.1%}",
    }),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Historical fake-today tests")
tests = calibration.tests.copy()
tests["fake_today"] = pd.to_datetime(tests["fake_today"]).dt.date
tests["training_start"] = pd.to_datetime(tests["training_start"]).dt.date
st.dataframe(
    tests[[
        "fake_today",
        "lookback_years",
        "training_start",
        "snapshot_growth_factor",
        "snapshot_amplitude_factor",
        "raw_error",
        "calibrated_error",
        "cv_growth_factor",
        "cv_amplitude_factor",
        "raw_12m_price_usd",
        "actual_12m_price_usd",
        "raw_12m_error_pct",
    ]].style.format({
        "snapshot_growth_factor": "{:.3f}×",
        "snapshot_amplitude_factor": "{:.3f}×",
        "raw_error": "{:.1%}",
        "calibrated_error": "{:.1%}",
        "cv_growth_factor": "{:.3f}×",
        "cv_amplitude_factor": "{:.3f}×",
        "raw_12m_price_usd": "${:,.0f}",
        "actual_12m_price_usd": "${:,.0f}",
        "raw_12m_error_pct": "{:+.1%}",
    }),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Snapshot G/K values are the regularized correction learned by each individual fake-today experiment. "
    "The final 4Y and 8Y corrections use their medians, and each held-out cutoff is scored using only the other cutoffs."
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
    "Frozen Price Model v3.12  →  Walk-Forward Calibration v1  →  "
    "Calibrated Price Model  →  BTC Financial Independence"
)
st.caption(
    f"Calibration engine: {CALIBRATION_VERSION}. The FI page will use this calibrated model by default "
    "when its status is PASS and its data/model fingerprints are current."
)
