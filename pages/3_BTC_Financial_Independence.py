import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import load_coinmetrics
from src.price_model import fit_price_model
from src.active_model_config import build_model_fingerprint
from src.financial_independence import (
    build_rebased_btc_paths,
    solve_earliest_financial_independence,
    solve_required_monthly_contribution,
)

st.title("BTC Financial Independence")
st.caption(
    "Compare Bitcoin and traditional-investment paths, solve an earliest FI age, "
    "or calculate the monthly contribution needed for a target or coast plan."
)

try:
    prices, meta = load_coinmetrics(refresh=False)
except Exception as exc:
    st.error(f"Coin Metrics data is unavailable: {exc}")
    st.stop()

min_date = prices["date"].min().date()
max_date = prices["date"].max().date()
default_start = max(min_date, pd.Timestamp("2018-12-31").date())

active_model = st.session_state.get("active_price_model_config")
if active_model:
    training_start = pd.Timestamp(active_model["training_start"])
    training_end = pd.Timestamp(active_model["training_end"])
    projection_years = int(active_model["projection_years"])
else:
    training_start = pd.Timestamp(default_start)
    training_end = pd.Timestamp(max_date)
    projection_years = 80

try:
    model = fit_price_model(
        prices=prices,
        training_start=training_start,
        training_end=training_end,
        projection_years=projection_years,
    )
except Exception as exc:
    st.error(f"Price model could not be fitted: {exc}")
    st.stop()

model_fingerprint = build_model_fingerprint(
    training_start,
    training_end,
    projection_years,
    prices["date"].max(),
    model.daily,
)
latest_actual = float(
    prices.loc[prices["date"] <= training_end, "price_usd"].iloc[-1]
)
btc_paths = build_rebased_btc_paths(model.daily, latest_actual)

st.info(
    f"Active Price Model: **{training_start.date()} → {training_end.date()}**, "
    f"**{projection_years} years**, fingerprint **{model_fingerprint}**."
)

saved = st.session_state.get("btc_fi_last_results")
if saved and saved.get("model_fingerprint") != model_fingerprint:
    st.session_state.pop("btc_fi_last_results", None)
    st.warning(
        "The active Price Model changed. Previous FI results were cleared. "
        "Run the calculation again."
    )

with st.form("btc_fi_form"):
    mode = st.radio(
        "Planning mode",
        [
            "Find earliest FI age",
            "Target FI age",
            "Coast to FI age",
        ],
        horizontal=True,
    )

    st.subheader("Personal and FI goal")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        current_age = st.number_input(
            "Current age", 18.0, 89.0, 35.0, 1.0
        )
    with p2:
        end_age = st.number_input(
            "Plan through age", 19.0, 120.0, 100.0, 1.0
        )
    with p3:
        target_fi_age = st.number_input(
            "Desired FI age",
            min_value=current_age,
            max_value=max(end_age - 1, current_age),
            value=min(55.0, max(end_age - 1, current_age)),
            step=1.0,
            disabled=mode == "Find earliest FI age",
        )
    with p4:
        coast_age = st.number_input(
            "Coast age",
            min_value=current_age,
            max_value=max(target_fi_age, current_age),
            value=min(45.0, max(target_fi_age, current_age)),
            step=1.0,
            disabled=mode != "Coast to FI age",
            help=(
                "Contributions stop at this age. The portfolio then grows without "
                "new contributions until the desired FI age."
            ),
        )

    g1, g2, g3 = st.columns(3)
    with g1:
        monthly_spending = st.number_input(
            "Monthly income needed in today's dollars",
            min_value=0.0,
            value=4000.0,
            step=250.0,
        )
    with g2:
        inflation_rate = st.number_input(
            "Annual inflation rate (%)",
            -5.0,
            20.0,
            3.0,
            0.25,
        ) / 100
    with g3:
        conviction = st.slider(
            "Bitcoin conviction for new monthly contributions (%)",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            disabled=mode == "Find earliest FI age",
            help=(
                "The solver allocates this share of required monthly contributions "
                "to Bitcoin and the remainder to all other investments."
            ),
        ) / 100

    withdrawal_source = st.selectbox(
        "Monthly income withdrawal source",
        ["Proportional", "BTC first", "Other investments first"],
        index=0,
    )

    st.subheader("Current balances and current strategy")
    b1, b2, o1, o2 = st.columns(4)
    with b1:
        btc_principal = st.number_input(
            "Current BTC investment ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )
    with b2:
        btc_monthly = st.number_input(
            "Current monthly BTC contribution ($)",
            min_value=0.0,
            value=0.0,
            step=25.0,
            disabled=mode != "Find earliest FI age",
        )
    with o1:
        other_principal = st.number_input(
            "Current other investments ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )
    with o2:
        other_monthly = st.number_input(
            "Current monthly other contribution ($)",
            min_value=0.0,
            value=0.0,
            step=25.0,
            disabled=mode != "Find earliest FI age",
        )

    st.subheader("Other-investment assumptions")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        other_return_contributing = st.number_input(
            "Annual return (%) while contributing",
            -20.0, 40.0, 12.0, 0.25
        ) / 100
    with a2:
        other_return_drawing = st.number_input(
            "Annual return (%) after FI",
            -20.0, 40.0, 7.0, 0.25
        ) / 100
    with a3:
        compounds = st.selectbox(
            "Compound interest",
            [1, 2, 4, 12, 52, 365],
            index=3,
            format_func=lambda x: f"{x} time(s) annually",
        )
    with a4:
        additions_at_start = (
            st.radio(
                "Make additions at",
                ["Start", "End"],
                index=1,
                horizontal=True,
            )
            == "Start"
        )

    run = st.form_submit_button(
        "Run FI Calculation",
        type="primary",
        use_container_width=True,
    )

if run:
    if end_age <= current_age:
        st.error("Plan-through age must exceed current age.")
        st.stop()
    if projection_years < end_age - current_age:
        st.error(
            "Increase the Price Model projection horizon so it reaches the "
            "selected ending age."
        )
        st.stop()

    common = dict(
        current_age=current_age,
        end_age=end_age,
        other_principal=other_principal,
        other_annual_return_while_contributing=other_return_contributing,
        other_annual_return_while_drawing=other_return_drawing,
        compounds_per_year=compounds,
        additions_at_start=additions_at_start,
        btc_principal_usd=btc_principal,
        monthly_spending_today=monthly_spending,
        inflation_rate=inflation_rate,
        withdrawal_source=withdrawal_source,
    )

    with st.spinner("Running FI calculations..."):
        if mode == "Find earliest FI age":
            center = solve_earliest_financial_independence(
                "BTC structural centerline",
                btc_paths["date"],
                btc_paths["btc_centerline_price"],
                btc_monthly_contribution=btc_monthly,
                other_monthly_contribution=other_monthly,
                **common,
            )
            cycle = solve_earliest_financial_independence(
                "BTC cycle-adjusted path",
                btc_paths["date"],
                btc_paths["btc_cycle_price"],
                btc_monthly_contribution=btc_monthly,
                other_monthly_contribution=other_monthly,
                **common,
            )

            all_other_common = dict(common)
            all_other_common["other_principal"] = other_principal + btc_principal
            all_other_common["btc_principal_usd"] = 0.0
            all_other = solve_earliest_financial_independence(
                "100% other investments",
                btc_paths["date"],
                pd.Series(1.0, index=btc_paths.index),
                btc_monthly_contribution=0.0,
                other_monthly_contribution=other_monthly + btc_monthly,
                withdrawal_source="Other investments first",
                **{
                    k: v for k, v in all_other_common.items()
                    if k != "withdrawal_source"
                },
            )

            all_btc_common = dict(common)
            all_btc_common["other_principal"] = 0.0
            all_btc_common["btc_principal_usd"] = btc_principal + other_principal
            all_btc = solve_earliest_financial_independence(
                "100% Bitcoin",
                btc_paths["date"],
                btc_paths["btc_cycle_price"],
                btc_monthly_contribution=btc_monthly + other_monthly,
                other_monthly_contribution=0.0,
                withdrawal_source="BTC first",
                **{
                    k: v for k, v in all_btc_common.items()
                    if k != "withdrawal_source"
                },
            )

            payload = {
                "mode": mode,
                "center": center,
                "cycle": cycle,
                "all_other": all_other,
                "all_btc": all_btc,
            }
        else:
            coast_value = coast_age if mode == "Coast to FI age" else None
            center_target = solve_required_monthly_contribution(
                "BTC structural centerline",
                btc_paths["date"],
                btc_paths["btc_centerline_price"],
                target_financial_independence_age=target_fi_age,
                coast_age=coast_value,
                bitcoin_conviction=conviction,
                **common,
            )
            cycle_target = solve_required_monthly_contribution(
                "BTC cycle-adjusted path",
                btc_paths["date"],
                btc_paths["btc_cycle_price"],
                target_financial_independence_age=target_fi_age,
                coast_age=coast_value,
                bitcoin_conviction=conviction,
                **common,
            )
            payload = {
                "mode": mode,
                "center_target": center_target,
                "cycle_target": cycle_target,
            }

    payload["model_fingerprint"] = model_fingerprint
    payload["inputs"] = {
        "current_age": current_age,
        "end_age": end_age,
        "monthly_spending": monthly_spending,
        "inflation_rate": inflation_rate,
        "target_fi_age": target_fi_age,
        "coast_age": coast_age,
        "conviction": conviction,
    }
    st.session_state["btc_fi_last_results"] = payload

results = st.session_state.get("btc_fi_last_results")
if results:
    st.subheader("FI Results")

    if results["mode"] == "Find earliest FI age":
        scenarios = [
            results["center"],
            results["cycle"],
            results["all_other"],
            results["all_btc"],
        ]
        rows = []
        cols = st.columns(4)
        for col, result in zip(cols, scenarios):
            with col:
                st.markdown(f"#### {result.scenario}")
                if result and result.sustainable:
                    st.metric(
                        "FI age",
                        f"{result.financial_independence_age:.2f}",
                    )
                    st.metric(
                        "Value at FI",
                        f"${result.financial_independence_balance:,.0f}",
                    )
                    st.metric(
                        "Ending value",
                        f"${result.ending_balance:,.0f}",
                    )
                else:
                    st.error("Not sustainable")

            if result:
                rows.append({
                    "scenario": result.scenario,
                    "fi_age": (
                        result.financial_independence_age
                        if result.sustainable else None
                    ),
                    "value_at_fi": result.financial_independence_balance,
                    "ending_value": result.ending_balance,
                    "total_contributions": result.total_contributions,
                    "total_withdrawals": result.total_withdrawals,
                    "sustainable": result.sustainable,
                })

        st.dataframe(
            pd.DataFrame(rows).style.format({
                "fi_age": "{:.2f}",
                "value_at_fi": "${:,.0f}",
                "ending_value": "${:,.0f}",
                "total_contributions": "${:,.0f}",
                "total_withdrawals": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Portfolio paths")
        display_mode = st.radio(
            "Display",
            ["Total Portfolio", "Portfolio Components", "Portfolio Comparison"],
            horizontal=True,
            key="fi_portfolio_display_mode",
        )
        tabs = st.tabs(["Structural centerline", "Cycle-adjusted path"])
        for tab, result in zip(tabs, scenarios[:2]):
            with tab:
                fig = go.Figure()
                if display_mode == "Total Portfolio":
                    fig.add_trace(go.Scatter(x=result.path["age"], y=result.path["total_portfolio"], mode="lines", name="Combined portfolio", line={"width": 3}))
                elif display_mode == "Portfolio Components":
                    fig.add_trace(go.Scatter(x=result.path["age"], y=result.path["btc_value"], mode="lines", name="Bitcoin value", stackgroup="components"))
                    fig.add_trace(go.Scatter(x=result.path["age"], y=result.path["other_investments"], mode="lines", name="Other investments", stackgroup="components"))
                    fig.add_trace(go.Scatter(x=result.path["age"], y=result.path["total_portfolio"], mode="lines", name="Combined portfolio", line={"width": 3}))
                else:
                    fig.add_trace(go.Scatter(x=result.path["age"], y=result.path["total_portfolio"], mode="lines", name="Mixed BTC + other portfolio", line={"width": 3}))
                    fig.add_trace(go.Scatter(x=results["all_other"].path["age"], y=results["all_other"].path["total_portfolio"], mode="lines", name="100% other investments", line={"dash": "dash", "width": 3}))
                    fig.add_trace(go.Scatter(x=results["all_btc"].path["age"], y=results["all_btc"].path["total_portfolio"], mode="lines", name="100% Bitcoin", line={"dash": "dot", "width": 3}))
                fig.add_vline(x=result.financial_independence_age, line_dash="dash", annotation_text="FI")
                fig.update_layout(title=f"{result.scenario} — {display_mode}", xaxis_title="Age", yaxis_title="Portfolio value (USD)", hovermode="x unified", height=600)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key=f"fi_paths_{result.scenario}_{display_mode}")

    else:
        targets = [
            results["center_target"],
            results["cycle_target"],
        ]
        cols = st.columns(2)
        rows = []
        for col, target in zip(cols, targets):
            with col:
                st.markdown(f"### {target.scenario}")
                if target.sustainable:
                    st.metric(
                        "Required monthly total",
                        f"${target.required_total_monthly:,.0f}",
                    )
                    st.metric(
                        "Required monthly BTC",
                        f"${target.required_btc_monthly:,.0f}",
                    )
                    st.metric(
                        "Required monthly other investments",
                        f"${target.required_other_monthly:,.0f}",
                    )
                    if target.coast_age is not None:
                        st.caption(
                            f"Contribute through age **{target.coast_age:.0f}**, "
                            f"then coast until FI at age "
                            f"**{target.target_financial_independence_age:.0f}**."
                        )
                else:
                    st.error(
                        "The target was not achievable within the solver limit."
                    )

            rows.append({
                "scenario": target.scenario,
                "desired_fi_age": target.target_financial_independence_age,
                "coast_age": target.coast_age,
                "bitcoin_conviction": target.bitcoin_conviction,
                "monthly_total": target.required_total_monthly,
                "monthly_btc": target.required_btc_monthly,
                "monthly_other": target.required_other_monthly,
                "sustainable": target.sustainable,
            })

        st.dataframe(
            pd.DataFrame(rows).style.format({
                "bitcoin_conviction": "{:.0%}",
                "monthly_total": "${:,.0f}",
                "monthly_btc": "${:,.0f}",
                "monthly_other": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Bitcoin conviction controls the split of new monthly contributions. "
            "Current BTC and other-investment balances remain as entered."
        )
