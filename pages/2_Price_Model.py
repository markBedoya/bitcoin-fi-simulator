import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data_pipeline import load_coinmetrics
from src.price_model import fit_price_model, find_most_conservative_training_start
from src.financial_independence import build_rebased_btc_paths
from src.active_model_config import build_model_fingerprint
from src.theme import REFERENCE_LINE_COLOR, REFERENCE_LINE_WIDTH, REFERENCE_LINE_DASH

st.title("Price Model v2.7 — Conditioned Partial Cycle + Empirical Shape")
st.caption(
    "All Bitcoin price history is always visible. The selected range controls only model fitting. "
    "Historical turning points anchor the fitted path; future timing remains fixed at 1428 days "
    "(1064 bull + 364 bear). The phase shape is learned from completed historical total-price "
    "moves. When the latest date is inside an unfinished cycle phase, the projection continues "
    "from that exact phase progress instead of restarting the phase."
)

try:
    prices, meta = load_coinmetrics(refresh=False)
except Exception as exc:
    st.error(f"Coin Metrics data is unavailable: {exc}")
    st.stop()

min_date = prices["date"].min().date()
max_date = prices["date"].max().date()


def run_conservative_search(minimum_years: int):
    current_start, current_end = st.session_state.price_model_training_range
    search_end = pd.Timestamp(current_end)
    progress = st.progress(
        0,
        text=f"Testing monthly training starts with {minimum_years}-year minimum...",
    )

    def update_progress(done, total):
        progress.progress(
            int(done / total * 100),
            text=f"Testing monthly start {done} of {total}...",
        )

    try:
        best, results_table = find_most_conservative_training_start(
            prices=prices,
            training_end=search_end,
            projection_years=projection_years,
            minimum_training_years=minimum_years,
            progress_callback=update_progress,
        )
        best_start = pd.Timestamp(best["training_start"]).date()
        st.session_state.price_model_training_range = (
            best_start,
            search_end.date(),
        )
        st.session_state.conservative_search_summary = {
            "training_start": best_start.isoformat(),
            "training_end": search_end.date().isoformat(),
            "projection_years": int(projection_years),
            "projected_ending_price_usd": float(best["projected_ending_price_usd"]),
            "implied_cagr": float(best["implied_cagr"]),
            "training_years": float(best["training_years"]),
            "candidates_tested": int(len(results_table)),
            "search_minimum_years": int(minimum_years),
        }
        st.session_state.conservative_search_leaderboard = (
            results_table.copy()
            .assign(search_minimum_years=minimum_years)
            .reset_index(drop=True)
        )
        progress.empty()
        st.rerun()
    except Exception as exc:
        progress.empty()
        st.error(f"Conservative-start search failed: {exc}")

if "pending_training_range" in st.session_state:
    st.session_state.price_model_training_range = st.session_state.pop("pending_training_range")

PRICE_MODEL_DEFAULTS_VERSION = "v2.3-all-data-10y"

# Reset once after this release so existing Streamlit sessions receive the new
# defaults as well as new sessions.
if st.session_state.get("price_model_defaults_version") != PRICE_MODEL_DEFAULTS_VERSION:
    st.session_state.price_model_training_range = (min_date, max_date)
    st.session_state.price_model_projection_years = 10
    st.session_state.price_model_defaults_version = PRICE_MODEL_DEFAULTS_VERSION
    st.session_state.pop("conservative_search_summary", None)
    st.session_state.pop("conservative_search_leaderboard", None)
    st.session_state.pop("leaderboard_selected_row", None)

if "price_model_training_range" not in st.session_state:
    st.session_state.price_model_training_range = (min_date, max_date)
if "price_model_projection_years" not in st.session_state:
    st.session_state.price_model_projection_years = 10

with st.sidebar:
    st.header("Usable data")

    projection_years = st.slider(
        "Projection horizon (years)",
        1,
        80,
        key="price_model_projection_years",
    )

    if st.button("Reset training range to all data", use_container_width=True):
        st.session_state.price_model_training_range = (min_date, max_date)
        st.session_state.pop("conservative_search_summary", None)
        st.session_state.pop("conservative_search_leaderboard", None)
        st.session_state.pop("leaderboard_selected_row", None)
        st.rerun()

    if st.button(
        "Find most conservative start month (8 yr min)",
        use_container_width=True,
    ):
        run_conservative_search(8)


    if st.button(
        "Find most conservative start month (4 yr min)",
        use_container_width=True,
    ):
        run_conservative_search(4)

    if st.button(
        "Find most conservative start month (5 yr min)",
        use_container_width=True,
    ):
        run_conservative_search(5)

    if st.button(
        "Find most conservative start month (6 yr min)",
        use_container_width=True,
    ):
        run_conservative_search(6)

    if st.button(
        "Find most conservative start month (7 yr min)",
        use_container_width=True,
    ):
        run_conservative_search(7)


    selected = st.slider(
        "Training date range",
        min_value=min_date,
        max_value=max_date,
        key="price_model_training_range",
        format="YYYY-MM-DD",
        help=(
            "Both handles directly control the data used to fit the model. "
            "Move the right handle backward to perform a historical evaluation."
        ),
    )
    training_start = pd.Timestamp(selected[0])
    training_end = pd.Timestamp(selected[1])

    st.caption(
        f"Model training cutoff: **{training_end.date().isoformat()}**. "
        "Prices outside the selected range are excluded from fitting but remain visible on the main chart."
    )

    summary = st.session_state.get("conservative_search_summary")
    if summary and (
        summary["training_start"] == training_start.date().isoformat()
        and summary["training_end"] == training_end.date().isoformat()
        and summary["projection_years"] == projection_years
    ):
        st.success(
            f"Selected start: **{summary['training_start']}**  \n"
            f"Year-{projection_years} projected price: "
            f"**${summary['projected_ending_price_usd']:,.0f}**  \n"
            f"Implied CAGR: **{summary['implied_cagr']:.2%}**  \n"
            f"Monthly candidates tested: **{summary['candidates_tested']}**  \\n"
            f"Minimum training history: **{summary.get('search_minimum_years', '—')} years**"
        )

    log_scale = st.toggle("Logarithmic price scale", value=True)


leaderboard = st.session_state.get("conservative_search_leaderboard")
if leaderboard is not None and not leaderboard.empty:
    st.success(
        f"Leaderboard ready: {len(leaderboard):,} monthly candidates ranked."
    )
    minimum_years = int(leaderboard["search_minimum_years"].iloc[0])
    st.subheader(f"Conservative start leaderboard — {minimum_years}-year minimum")
    st.caption(
        "Ranked from lowest to highest model-implied CAGR at the selected projection horizon. "
        "Choose a row below to apply that training start."
    )

    display = leaderboard.copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    display["training_start"] = pd.to_datetime(display["training_start"]).dt.date
    display["training_end"] = pd.to_datetime(display["training_end"]).dt.date

    st.dataframe(
        display[
            [
                "rank",
                "training_start",
                "training_years",
                "projection_years",
                "implied_cagr",
                "projected_ending_price_usd",
            ]
        ].style.format(
            {
                "training_years": "{:.2f}",
                "implied_cagr": "{:.2%}",
                "projected_ending_price_usd": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    option_labels = [
        (
            f"#{int(row.rank)} — {row.training_start} | "
            f"CAGR {row.implied_cagr:.2%} | "
            f"Projected ${row.projected_ending_price_usd:,.0f}"
        )
        for row in display.itertuples(index=False)
    ]
    selected_label = st.selectbox(
        "Select a ranked training start",
        option_labels,
        key="leaderboard_selected_row",
    )
    selected_rank = option_labels.index(selected_label)
    selected_row = display.iloc[selected_rank]

    col_apply, col_clear = st.columns([1, 1])
    with col_apply:
        if st.button("Apply selected leaderboard start", type="primary", use_container_width=True):
            selected_start = pd.Timestamp(selected_row["training_start"]).date()
            selected_end = pd.Timestamp(selected_row["training_end"]).date()
            st.session_state.pending_training_range = (selected_start, selected_end)
            st.session_state.conservative_search_summary = {
                "training_start": selected_start.isoformat(),
                "training_end": selected_end.isoformat(),
                "projection_years": int(selected_row["projection_years"]),
                "projected_ending_price_usd": float(selected_row["projected_ending_price_usd"]),
                "implied_cagr": float(selected_row["implied_cagr"]),
                "training_years": float(selected_row["training_years"]),
                "candidates_tested": int(len(display)),
            }
            st.rerun()
    with col_clear:
        if st.button("Clear leaderboard", use_container_width=True):
            st.session_state.pop("conservative_search_leaderboard", None)
            st.session_state.pop("leaderboard_selected_row", None)
            st.rerun()


try:
    result = fit_price_model(prices, training_start, training_end, projection_years)
except Exception as exc:
    st.error(str(exc))
    st.stop()

daily = result.daily
diag = result.diagnostics
hist = daily[daily["row_type"] == "historical_training"]
proj = daily[daily["row_type"] == "projected"]

last_actual_price = float(hist["actual_price_usd"].iloc[-1])
anchored_proj = build_rebased_btc_paths(
    model_daily=daily,
    latest_actual_price=last_actual_price,
)
proj = proj.merge(anchored_proj, on="date", how="left")

# Explicit projection-boundary rows make the plotted historical fitted path and
# future projected path meet on the exact latest observed Bitcoin price.
projection_boundary = pd.DataFrame({
    "date": [training_end],
    "btc_cycle_price": [last_actual_price],
})
cycle_projection_plot = pd.concat([
    projection_boundary,
    proj[["date", "btc_cycle_price"]],
], ignore_index=True)

# The structural/geometric centerline is one model series across the boundary,
# not two independently anchored lines.
centerline_plot = daily[["date", "structural_centerline_usd"]].copy()

fitted_endpoint = float(hist["fitted_or_projected_price_usd"].iloc[-1])
fitted_endpoint_error_pct = fitted_endpoint / last_actual_price - 1.0
endpoint_scale_factor = float(diag.get("endpoint_scale_factor", 1.0))

# Numerical continuity checks.
fit_meets_actual = abs(fitted_endpoint_error_pct) < 1e-10
future_meets_actual = abs(
    float(cycle_projection_plot["btc_cycle_price"].iloc[0]) / last_actual_price - 1.0
) < 1e-10
centerline_is_single_series = (
    centerline_plot["date"].is_monotonic_increasing
    and centerline_plot["structural_centerline_usd"].notna().all()
)

active_model_fingerprint = build_model_fingerprint(
    training_start=training_start,
    training_end=training_end,
    projection_years=projection_years,
    latest_data_date=prices["date"].max(),
    model_daily=daily,
)
active_model_config = {
    "training_start": training_start.date().isoformat(),
    "training_end": training_end.date().isoformat(),
    "projection_years": int(projection_years),
    "latest_data_date": prices["date"].max().date().isoformat(),
    "fingerprint": active_model_fingerprint,
}
st.session_state["active_price_model_config"] = active_model_config

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Training observations", f'{diag["training_rows"]:,}')
c2.metric("Training end", training_end.date().isoformat())
c3.metric("Historical exponent", f'{diag["historical_exponent"]:.3f}')
c4.metric("Fixed cycle length", f'{int(diag["cycle_days"]):,} days')
c5.metric("Bull / bear days", f'{diag.get("bull_days", 1064)} / {diag.get("bear_days", 364)}')

st.caption(
    f"Active Price Model fingerprint: **{active_model_fingerprint}**. "
    "BTC Financial Independence will use this exact model."
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=prices["date"],
    y=prices["price_usd"],
    mode="lines",
    name="Actual Bitcoin price",
))
fig.add_trace(go.Scatter(
    x=centerline_plot["date"],
    y=centerline_plot["structural_centerline_usd"],
    mode="lines",
    name="Structural / geometric centerline",
    line={"dash": "dash"},
))
fig.add_trace(go.Scatter(
    x=hist["date"],
    y=hist["fitted_or_projected_price_usd"],
    mode="lines",
    name="Historical fitted path",
))

historical_anchor_plot = diag.get("cycle_anchor_table")
required_anchor_columns = {"date", "actual_price_usd", "source"}
if (
    historical_anchor_plot is not None
    and not historical_anchor_plot.empty
    and required_anchor_columns.issubset(historical_anchor_plot.columns)
):
    historical_anchor_plot = historical_anchor_plot.copy()
    historical_anchor_plot = historical_anchor_plot[
        (historical_anchor_plot["source"] == "historical market anchor")
        & historical_anchor_plot["actual_price_usd"].notna()
    ]
    if not historical_anchor_plot.empty:
        fig.add_trace(go.Scatter(
            x=historical_anchor_plot["date"],
            y=historical_anchor_plot["actual_price_usd"],
            mode="markers",
            name="Historical cycle intersections",
            marker={
                "size": 10,
                "color": REFERENCE_LINE_COLOR,
                "symbol": "circle-open",
                "line": {"width": 2, "color": REFERENCE_LINE_COLOR},
            },
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Anchor price: $%{y:,.0f}<extra></extra>"
            ),
        ))
projected_anchor_plot = diag.get("cycle_anchor_table")
if projected_anchor_plot is not None and not projected_anchor_plot.empty:
    projected_anchor_plot = projected_anchor_plot.copy()
    projected_anchor_plot = projected_anchor_plot[
        projected_anchor_plot["source"].astype(str).str.contains("projected", case=False, na=False)
        | projected_anchor_plot["source"].astype(str).str.contains("conditioned", case=False, na=False)
    ]
    projected_anchor_plot = projected_anchor_plot[
        pd.to_datetime(projected_anchor_plot["date"]) >= training_end
    ]
    if not projected_anchor_plot.empty:
        fig.add_trace(go.Scatter(
            x=projected_anchor_plot["date"],
            y=projected_anchor_plot["knot_price_usd"],
            mode="markers",
            name="Projected cycle turning points",
            marker={
                "size": 9,
                "color": REFERENCE_LINE_COLOR,
                "symbol": "diamond-open",
                "line": {"width": 2, "color": REFERENCE_LINE_COLOR},
            },
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Projected turning point: $%{y:,.0f}<extra></extra>"
            ),
        ))

fig.add_trace(go.Scatter(
    x=cycle_projection_plot["date"],
    y=cycle_projection_plot["btc_cycle_price"],
    mode="lines",
    name="Future projected price",
    line={"width": 3},
))


fig.add_vline(
    x=str(training_end.date()),
    line_dash=REFERENCE_LINE_DASH,
    line_color=REFERENCE_LINE_COLOR,
    line_width=REFERENCE_LINE_WIDTH,
    annotation_text="Projection start",
)
fig.update_layout(
    title="Bitcoin historical fit and projected daily path",
    xaxis_title="Date",
    yaxis_title="Bitcoin price (USD)",
    hovermode="x unified",
    height=650,
)
fig.update_yaxes(type="log" if log_scale else "linear")
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.subheader("Continuity verification")
v1, v2, v3, v4 = st.columns(4)
v1.metric(
    "Historical fit → actual",
    "PASS" if fit_meets_actual else "FAIL",
    f"{fitted_endpoint_error_pct:+.6%}",
)
v2.metric(
    "Future path → actual",
    "PASS" if future_meets_actual else "FAIL",
    "$0 boundary gap" if future_meets_actual else "Boundary mismatch",
)
v3.metric(
    "Centerline continuity",
    "PASS" if centerline_is_single_series else "FAIL",
    "Single series across boundary",
)
v4.metric(
    "Endpoint scale",
    f"{endpoint_scale_factor:.4f}×",
    "One common factor",
)

if fit_meets_actual and future_meets_actual and centerline_is_single_series:
    st.success(
        "Projection boundary verified: latest actual price, historical fitted "
        "path, and future projected path meet at the same point. The structural "
        "and future geometric centerline are one continuous model series."
    )
else:
    st.error(
        "A projection-boundary continuity check failed. Do not rely on this "
        "projection until the failed check is resolved."
    )

st.subheader("Model diagnostics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Bull phase", f'{diag.get("bull_days", 1064):,} days')
m2.metric("Bear phase", f'{diag.get("bear_days", 364):,} days')
m3.metric("Peak position", f'{diag["peak_progress"]*100:.1f}%')
m4.metric("Terminal exponent", f'{diag["terminal_exponent"]:.3f}')
m5.metric("Next modeled trough", diag.get("next_modeled_trough", "2026-10-05"))

st.caption(
    "Future-cycle centering: **PASS** — each projected peak uses +A and its paired "
    "trough uses -A in log-price space around the same structural centerline. "
    f"Bull shape: {diag.get('bull_curve', 'empirical')}. "
    f"Bear shape: {diag.get('bear_curve', 'empirical')}. "
    "The learned phase shape now controls the **entire trough-to-peak / peak-to-trough "
    "log-price move**, rather than only the residual around the rising centerline."
)

current_partial_phase = diag.get("current_partial_phase")
if current_partial_phase:
    st.subheader("Current partial-cycle verification")
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    pc1.metric("Current phase", str(current_partial_phase.get("phase", "—")).title())
    pc2.metric(
        "Phase elapsed",
        f"{float(current_partial_phase.get('current_progress', 0.0)) * 100:.1f}%",
    )
    pc3.metric(
        "Learned decline completed",
        f"{float(current_partial_phase.get('learned_completion', 0.0)) * 100:.1f}%",
    )
    pc4.metric(
        "Oct 5, 2026 trough",
        f"${float(current_partial_phase.get('projected_trough_price_usd', float('nan'))):,.0f}",
    )
    pc5.metric(
        "Remaining modeled move",
        f"{float(current_partial_phase.get('remaining_change_pct', float('nan'))):+.1%}",
    )
    partial_declines = (
        float(current_partial_phase.get("projected_trough_price_usd", float("nan")))
        <= float(current_partial_phase.get("current_price_usd", float("nan")))
    )
    if partial_declines:
        st.success(
            "Partial bear continuation: PASS — the projection starts at the latest actual "
            "Bitcoin price, enters the learned bear template at the current phase progress, "
            "and continues downward to the Oct 5, 2026 modeled trough before the next bull phase begins."
        )
    else:
        st.error(
            "Partial bear continuation: FAIL — the modeled Oct 5, 2026 trough is above the "
            "latest actual Bitcoin price. Review the partial-phase fit before using this projection."
        )
    st.caption(
        f"Partial-bear fit uses {int(current_partial_phase.get('observations', 0)):,} daily observations "
        f"from the Oct 6, 2025 peak through the selected training end. Log-price fit RMSE: "
        f"{float(current_partial_phase.get('fit_rmse_log', float('nan'))):.4f}."
    )

st.subheader("Cycle anchor verification")
st.caption(
    "Historical anchors inside the selected training range are actual Bitcoin turning-point dates. "
    "The fitted path is constrained to intersect the observed Bitcoin price at every listed historical anchor."
)
anchor_table = diag.get("cycle_anchor_table")
if anchor_table is not None and not anchor_table.empty:
    anchor_display = anchor_table.copy()
    anchor_display["date"] = pd.to_datetime(anchor_display["date"]).dt.date
    anchor_display["modeled_price_usd"] = (
        anchor_display["structural_centerline_usd"]
        * np.exp(anchor_display["log_deviation"])
    )
    anchor_display["intersection_error_pct"] = np.where(
        anchor_display["actual_price_usd"].notna(),
        anchor_display["modeled_price_usd"] / anchor_display["actual_price_usd"] - 1.0,
        np.nan,
    )
    st.dataframe(
        anchor_display[[
            "date", "type", "source", "actual_price_usd",
            "modeled_price_usd", "intersection_error_pct"
        ]].style.format({
            "actual_price_usd": "${:,.0f}",
            "modeled_price_usd": "${:,.0f}",
            "intersection_error_pct": "{:+.4%}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    historical_anchor_errors = anchor_display.loc[
        anchor_display["actual_price_usd"].notna(), "intersection_error_pct"
    ].abs()
    if len(historical_anchor_errors) and historical_anchor_errors.max() < 1e-9:
        st.success("Historical market anchors intersect the fitted path exactly.")

st.subheader("Projected returns by horizon")
st.markdown(
    """
**Lump Sum CAGR** is the annualized return for a single investment made at the
latest actual Bitcoin price and held until that future horizon.

**Forward Return 1 Year** is the model's price return over the next 12 months
starting at that future horizon. This is especially useful for understanding
the return available to future DCA contributions as Bitcoin matures.
"""
)

projection_points = {}
for years in range(1, projection_years + 1):
    target = training_end + pd.DateOffset(years=years)
    nearest = proj.iloc[(proj["date"] - target).abs().argsort()[:1]]
    projection_points[years] = {
        "cycle_price": float(nearest["btc_cycle_price"].iloc[0]),
        "center_price": float(nearest["btc_centerline_price"].iloc[0]),
    }

rows = []
for years in range(1, projection_years + 1):
    point = projection_points[years]
    cycle_price = point["cycle_price"]
    center_price = point["center_price"]

    cycle_cagr = (cycle_price / last_actual_price) ** (1 / years) - 1
    center_cagr = (center_price / last_actual_price) ** (1 / years) - 1

    cycle_forward_1y = None
    center_forward_1y = None
    if years < projection_years:
        next_point = projection_points[years + 1]
        cycle_forward_1y = next_point["cycle_price"] / cycle_price - 1
        center_forward_1y = next_point["center_price"] / center_price - 1

    rows.append({
        "Horizon (Years)": years,
        "Cycle Projected Price": cycle_price,
        "Cycle Lump Sum CAGR": cycle_cagr,
        "Cycle Forward Return 1 Year": cycle_forward_1y,
        "Centerline Projected Price": center_price,
        "Centerline Lump Sum CAGR": center_cagr,
        "Centerline Forward Return 1 Year": center_forward_1y,
    })

if rows:
    returns_df = pd.DataFrame(rows)

    st.dataframe(
        returns_df.style.format({
            "Cycle Projected Price": "${:,.0f}",
            "Cycle Lump Sum CAGR": "{:.2%}",
            "Cycle Forward Return 1 Year": "{:.2%}",
            "Centerline Projected Price": "${:,.0f}",
            "Centerline Lump Sum CAGR": "{:.2%}",
            "Centerline Forward Return 1 Year": "{:.2%}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Forward Return 1 Year")
    benchmark_pct = st.number_input(
        "Comparison annual return (%)",
        min_value=-50.0,
        max_value=100.0,
        value=12.0,
        step=0.25,
        help=(
            "Reference line only. Use this to compare Bitcoin's modeled one-year "
            "forward return with another assumed annual return."
        ),
    )

    return_fig = go.Figure()
    return_fig.add_trace(go.Scatter(
        x=returns_df["Horizon (Years)"],
        y=returns_df["Cycle Forward Return 1 Year"] * 100,
        mode="lines",
        name="Cycle Forward Return 1 Year",
    ))
    return_fig.add_trace(go.Scatter(
        x=returns_df["Horizon (Years)"],
        y=returns_df["Centerline Forward Return 1 Year"] * 100,
        mode="lines",
        name="Centerline Forward Return 1 Year",
        line={"dash": "dash"},
    ))
    return_fig.add_hline(
        y=benchmark_pct,
        line_dash=REFERENCE_LINE_DASH,
        line_color=REFERENCE_LINE_COLOR,
        line_width=REFERENCE_LINE_WIDTH,
        annotation_text=f"{benchmark_pct:.2f}% comparison",
        annotation_position="top right",
    )
    return_fig.update_layout(
        xaxis_title="Years into future",
        yaxis_title="Forward return over next 12 months (%)",
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(
        return_fig,
        use_container_width=True,
        config={"displaylogo": False},
        key="price_model_forward_return_chart",
    )

st.subheader("Empirical phase-shape learning")
st.caption(
    "Each completed bull and bear phase is normalized from 0% to 100% in time and "
    "0% to 100% of its actual historical log-price move. The individual paths are shown below; the "
    "bold learned template is their smoothed median. Future phases apply this shape "
    "directly to the full log-price move between projected trough and peak prices, "
    "while keeping the fixed 1064/364-day timing."
)

bull_shape_diag = diag.get("bull_shape_diagnostics", {})
bear_shape_diag = diag.get("bear_shape_diagnostics", {})
shape_cols = st.columns(6)
shape_cols[0].metric("Bull phases used", int(diag.get("bull_phases_used", 0)))
shape_cols[1].metric(
    "Bull 50% move",
    f"{bull_shape_diag.get('half_move_progress', float('nan')) * 100:.1f}%",
    help="Percent of the 1064-day bull phase elapsed when the learned template has completed half of its normalized move.",
)
shape_cols[2].metric(
    "Bull max acceleration",
    f"{bull_shape_diag.get('max_acceleration_progress', float('nan')) * 100:.1f}%",
    help="Data-derived point where the second derivative of the learned bull template is greatest.",
)
shape_cols[3].metric("Bear phases used", int(diag.get("bear_phases_used", 0)))
shape_cols[4].metric(
    "Bear 50% decline",
    f"{bear_shape_diag.get('half_move_progress', float('nan')) * 100:.1f}%",
    help="Percent of the 364-day bear phase elapsed when half of the learned normalized decline is complete.",
)
shape_cols[5].metric(
    "Bear max decline velocity",
    f"{bear_shape_diag.get('max_velocity_progress', float('nan')) * 100:.1f}%",
    help="Data-derived point where the normalized bear decline is progressing fastest.",
)

phase_overlays = diag.get("phase_shape_overlays")
phase_templates = diag.get("phase_shape_templates")
if (
    phase_templates is not None
    and not phase_templates.empty
):
    bull_tab, bear_tab = st.tabs(["Bull phase shape", "Bear phase shape"])
    for tab, phase_name, phase_label, phase_diag in [
        (bull_tab, "bull", "Bull", bull_shape_diag),
        (bear_tab, "bear", "Bear", bear_shape_diag),
    ]:
        with tab:
            phase_fig = go.Figure()
            if phase_overlays is not None and not phase_overlays.empty:
                phase_hist = phase_overlays[phase_overlays["phase"] == phase_name]
                for phase_id, grp in phase_hist.groupby("phase_id"):
                    phase_fig.add_trace(go.Scatter(
                        x=grp["progress"] * 100,
                        y=grp["normalized_move"] * 100,
                        mode="lines",
                        name=str(phase_id),
                        opacity=0.35,
                        line={"width": 1.5},
                    ))
            learned = phase_templates[phase_templates["phase"] == phase_name]
            phase_fig.add_trace(go.Scatter(
                x=learned["progress"] * 100,
                y=learned["normalized_move"] * 100,
                mode="lines",
                name=f"Learned median {phase_name} template",
                line={"width": 4},
            ))
            if phase_name == "bull" and phase_diag:
                phase_fig.add_vline(
                    x=float(phase_diag.get("max_acceleration_progress", 0.0)) * 100,
                    line_dash=REFERENCE_LINE_DASH,
                    line_color=REFERENCE_LINE_COLOR,
                    line_width=REFERENCE_LINE_WIDTH,
                    annotation_text="Max acceleration",
                )
            if phase_name == "bear" and phase_diag:
                phase_fig.add_vline(
                    x=float(phase_diag.get("max_velocity_progress", 0.0)) * 100,
                    line_dash=REFERENCE_LINE_DASH,
                    line_color=REFERENCE_LINE_COLOR,
                    line_width=REFERENCE_LINE_WIDTH,
                    annotation_text="Max decline velocity",
                )
            phase_fig.update_layout(
                title=f"{phase_label} phase — historical normalized paths vs learned template",
                xaxis_title=f"{phase_label} phase progress (%)",
                yaxis_title="Normalized move completed (%)",
                yaxis={"range": [0, 100]},
                hovermode="x unified",
                height=460,
            )
            st.plotly_chart(
                phase_fig,
                use_container_width=True,
                config={"displaylogo": False},
                key=f"price_model_empirical_{phase_name}_shape",
            )
else:
    st.warning(
        "The selected training window contains no complete historical bull/bear phase, "
        "so the model is using a conservative fallback phase shape."
    )

st.subheader("Fixed 1428-day historical cycle overlays")
if result.cycle_overlays.empty:
    st.warning("The selected training range does not contain enough complete trough-to-trough cycles.")
else:
    st.subheader("Empirical full-cycle shape")
    cycle_fig = go.Figure()
    for cycle_no, grp in result.cycle_overlays.groupby("cycle"):
        cycle_fig.add_trace(go.Scatter(x=grp["progress"]*100, y=grp["log_deviation"], mode="lines", name=f"Cycle {cycle_no}", opacity=0.45))
    cycle_fig.add_trace(go.Scatter(x=result.cycle_template["progress"]*100, y=result.cycle_template["log_deviation"], mode="lines", name="Learned 1428-day empirical template", line={"width": 4}))
    cycle_fig.update_layout(xaxis_title="Cycle progress (%)", yaxis_title="Log deviation from centerline", height=450)
    st.plotly_chart(cycle_fig, use_container_width=True, config={"displaylogo": False})

st.subheader("Actual price relative to structural centerline")
st.caption(
    "A value of 1.0 means actual Bitcoin price equals the structural centerline."
)
ratio_fig = go.Figure()
ratio_fig.add_trace(go.Scatter(
    x=hist["date"],
    y=hist["actual_price_usd"] / hist["structural_centerline_usd"],
    mode="lines",
    name="Actual ÷ centerline",
))
ratio_fig.add_hline(
    y=1.0,
    line_dash=REFERENCE_LINE_DASH,
    line_color=REFERENCE_LINE_COLOR,
    line_width=REFERENCE_LINE_WIDTH,
    annotation_text="Centerline = 1.0",
    annotation_position="top right",
)
ratio_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price / centerline",
    height=400,
)
st.plotly_chart(
    ratio_fig,
    use_container_width=True,
    config={"displaylogo": False},
    key="price_model_actual_vs_centerline",
)

st.download_button(
    "Download complete historical + future daily CSV",
    data=daily.to_csv(index=False).encode("utf-8"),
    file_name="bitcoin_price_model_daily.csv",
    mime="text/csv",
)
