import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data_pipeline import load_coinmetrics
from src.price_model import fit_price_model, find_most_conservative_training_start
from src.financial_independence import build_rebased_btc_paths
from src.active_model_config import build_model_fingerprint

st.title("Price Model v2.0.1")
st.caption("Use all available data by default, or change the usable training period and refit the daily future path.")

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

default_training_start = max(min_date, pd.Timestamp("2018-12-31").date())

if "price_model_training_range" not in st.session_state:
    st.session_state.price_model_training_range = (default_training_start, max_date)
if "price_model_projection_years" not in st.session_state:
    st.session_state.price_model_projection_years = 80

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
        "Prices after this date are excluded from fitting."
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

    overlay_excluded = st.toggle("Overlay excluded actual prices", value=True)
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

raw_first_cycle = float(proj["fitted_or_projected_price_usd"].iloc[0])
raw_first_center = float(proj["structural_centerline_usd"].iloc[0])
cycle_anchor_ratio = raw_first_cycle / last_actual_price
center_anchor_ratio = raw_first_center / last_actual_price

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
c4.metric("Estimated cycle length", f'{diag["cycle_days"]/365.25:.2f} years')
c5.metric("Complete cycles found", diag["complete_cycles"])

st.caption(
    f"Active Price Model fingerprint: **{active_model_fingerprint}**. "
    "BTC Financial Independence will use this exact model."
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual_price_usd"], mode="lines", name="Actual price used"))
fig.add_trace(go.Scatter(x=hist["date"], y=hist["structural_centerline_usd"], mode="lines", name="Structural centerline"))
fig.add_trace(go.Scatter(x=hist["date"], y=hist["fitted_or_projected_price_usd"], mode="lines", name="Historical fitted path"))
fig.add_trace(go.Scatter(x=proj["date"], y=proj["btc_cycle_price"], mode="lines", name="Future projected price — anchored to latest actual", line={"width": 3}))
fig.add_trace(go.Scatter(x=proj["date"], y=proj["btc_centerline_price"], mode="lines", name="Future centerline — anchored to latest actual", line={"dash": "dash"}))

if overlay_excluded and training_end < prices["date"].max():
    excluded = prices[prices["date"] > training_end]
    fig.add_trace(go.Scatter(x=excluded["date"], y=excluded["price_usd"], mode="lines", name="Excluded actual prices", line={"dash": "dot"}, opacity=0.45))

fig.add_vline(x=str(training_end.date()), line_dash="dash", annotation_text="Projection start")
fig.update_layout(title="Bitcoin historical fit and projected daily path", xaxis_title="Date", yaxis_title="Bitcoin price (USD)", hovermode="x unified", height=650)
fig.update_yaxes(type="log" if log_scale else "linear")
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.caption(
    "Future paths are anchored to the latest actual Bitcoin price before "
    "calculating returns. This is the same path used by BTC Financial Independence."
)
if abs(cycle_anchor_ratio - 1.0) > 0.05 or abs(center_anchor_ratio - 1.0) > 0.05:
    st.warning(
        "The raw fitted model does not begin at the latest actual Bitcoin price. "
        f"Raw cycle path starts at {cycle_anchor_ratio:.2f}× actual and raw "
        f"centerline starts at {center_anchor_ratio:.2f}× actual. Those artificial "
        "starting gaps are excluded from the CAGR table and FI calculations."
    )

st.subheader("Model diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cycle peak position", f'{diag["peak_progress"]*100:.1f}%')
m2.metric("Amplitude scale", f'{diag["amplitude_scale"]:.2f}×')
m3.metric("Amplitude retained per cycle", f'{diag["amplitude_retained_per_cycle"]*100:.1f}%')
m4.metric("Terminal exponent", f'{diag["terminal_exponent"]:.3f}')

rows = []
for years in range(1, projection_years + 1):
    target = training_end + pd.DateOffset(years=years)
    nearest = proj.iloc[(proj["date"] - target).abs().argsort()[:1]]
    cycle_price = float(nearest["btc_cycle_price"].iloc[0])
    center_price = float(nearest["btc_centerline_price"].iloc[0])
    cycle_cagr = (cycle_price / last_actual_price) ** (1 / years) - 1
    center_cagr = (center_price / last_actual_price) ** (1 / years) - 1
    rows.append({
        "horizon_years": years,
        "cycle_projected_price_usd": cycle_price,
        "cycle_implied_cagr": cycle_cagr,
        "centerline_projected_price_usd": center_price,
        "centerline_implied_cagr": center_cagr,
    })
if rows:
    st.dataframe(
        pd.DataFrame(rows).style.format({
            "cycle_projected_price_usd": "${:,.0f}",
            "cycle_implied_cagr": "{:.2%}",
            "centerline_projected_price_usd": "${:,.0f}",
            "centerline_implied_cagr": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Normalized historical cycle overlays")
if result.cycle_overlays.empty:
    st.warning("The selected training range does not contain enough complete trough-to-trough cycles.")
else:
    cycle_fig = go.Figure()
    for cycle_no, grp in result.cycle_overlays.groupby("cycle"):
        cycle_fig.add_trace(go.Scatter(x=grp["progress"]*100, y=grp["log_deviation"], mode="lines", name=f"Cycle {cycle_no}", opacity=0.45))
    cycle_fig.add_trace(go.Scatter(x=result.cycle_template["progress"]*100, y=result.cycle_template["log_deviation"], mode="lines", name="Learned empirical template", line={"width": 4}))
    cycle_fig.update_layout(xaxis_title="Cycle progress (%)", yaxis_title="Log deviation from centerline", height=450)
    st.plotly_chart(cycle_fig, use_container_width=True, config={"displaylogo": False})

st.subheader("Actual price relative to structural centerline")
ratio_fig = go.Figure()
ratio_fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual_price_usd"]/hist["structural_centerline_usd"], mode="lines", name="Actual ÷ centerline"))
ratio_fig.add_hline(y=1.0, line_dash="dash")
ratio_fig.update_layout(xaxis_title="Date", yaxis_title="Price / centerline", height=400)
st.plotly_chart(ratio_fig, use_container_width=True, config={"displaylogo": False})

st.download_button(
    "Download complete historical + future daily CSV",
    data=daily.to_csv(index=False).encode("utf-8"),
    file_name="bitcoin_price_model_daily.csv",
    mime="text/csv",
)
