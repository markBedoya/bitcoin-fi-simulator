"""Fast regression checks for FI accounting and scenario ordering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.financial_independence import (
    _simulate_prepared,
    prepare_monthly_path,
    solve_earliest_financial_independence,
    validate_result_accounting,
)


def assert_solver_matches_exhaustive_scan(
    dates: pd.Series,
    prices: pd.Series,
    common: dict,
) -> None:
    prepared = prepare_monthly_path(
        dates, prices, common["current_age"], common["end_age"]
    )
    kwargs = {
        key: value
        for key, value in common.items()
        if key not in {"current_age", "end_age"}
    }
    expected_month = None
    max_month = int((common["end_age"] - common["current_age"]) * 12) - 1
    for month in range(max_month + 1):
        age = common["current_age"] + month / 12
        result = _simulate_prepared(
            "exhaustive",
            prepared,
            common["current_age"],
            age,
            common["end_age"],
            contribution_end_age=age,
            build_path=False,
            **kwargs,
        )
        if result.sustainable:
            expected_month = month
            break

    solved = solve_earliest_financial_independence(
        "solver",
        dates,
        prices,
        **common,
    )
    solved_month = round(
        (solved.financial_independence_age - common["current_age"]) * 12
    )
    assert solved_month == expected_month


def run() -> None:
    months = 80 * 12 + 1
    dates = pd.Series(pd.date_range("2026-08-01", periods=months, freq="MS"))

    btc_prices = pd.Series(65_000 * (1.15 ** (np.arange(months) / 12)))
    flat_prices = pd.Series(np.ones(months))

    common = dict(
        current_age=35,
        end_age=100,
        other_principal=0.0,
        other_annual_return_while_contributing=0.12,
        other_annual_return_while_drawing=0.07,
        compounds_per_year=12,
        additions_at_start=False,
        btc_principal_usd=0.0,
        monthly_spending_today=4_000.0,
        inflation_rate=0.03,
    )

    btc = solve_earliest_financial_independence(
        "BTC regression",
        dates,
        btc_prices,
        btc_monthly_contribution=2_000.0,
        other_monthly_contribution=0.0,
        withdrawal_source="BTC first",
        **common,
    )
    other = solve_earliest_financial_independence(
        "Other regression",
        dates,
        flat_prices,
        btc_monthly_contribution=0.0,
        other_monthly_contribution=2_000.0,
        withdrawal_source="Other investments first",
        **common,
    )

    assert validate_result_accounting(btc)["passed"]
    assert validate_result_accounting(other)["passed"]
    # A smooth 15% BTC path should not lose to 12% nominal APR accumulation.
    assert btc.financial_independence_age <= other.financial_independence_age

    # The public solver must return the same earliest month as a direct scan.
    assert_solver_matches_exhaustive_scan(
        dates,
        btc_prices,
        {
            **common,
            "btc_monthly_contribution": 2_000.0,
            "other_monthly_contribution": 0.0,
            "withdrawal_source": "BTC first",
        },
    )

    print("FI accounting and earliest-age regression checks passed.")


if __name__ == "__main__":
    run()
