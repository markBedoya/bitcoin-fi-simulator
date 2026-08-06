from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass
class FinancialIndependenceResult:
    scenario: str
    sustainable: bool
    financial_independence_age: float
    financial_independence_date: pd.Timestamp
    ending_balance: float
    financial_independence_balance: float
    total_contributions: float
    total_withdrawals: float
    path: pd.DataFrame


@dataclass
class ContributionTargetResult:
    scenario: str
    sustainable: bool
    target_financial_independence_age: float
    coast_age: float | None
    bitcoin_conviction: float
    required_total_monthly: float | None
    required_btc_monthly: float | None
    required_other_monthly: float | None
    simulation: FinancialIndependenceResult | None


@dataclass(frozen=True)
class PreparedMonthlyPath:
    dates: pd.DatetimeIndex
    btc_prices: np.ndarray
    ages: np.ndarray
    years_from_start: np.ndarray
    start_date: pd.Timestamp


def build_rebased_btc_paths(model_daily: pd.DataFrame, latest_actual_price: float) -> pd.DataFrame:
    projected = model_daily[model_daily["row_type"] == "projected"].copy()
    if projected.empty:
        raise ValueError("The price model contains no future projection rows.")
    first_center = float(projected["structural_centerline_usd"].iloc[0])
    first_cycle = float(projected["fitted_or_projected_price_usd"].iloc[0])
    if min(first_center, first_cycle, latest_actual_price) <= 0:
        raise ValueError("Bitcoin prices must be positive.")
    projected["btc_centerline_price"] = latest_actual_price * projected["structural_centerline_usd"] / first_center
    projected["btc_cycle_price"] = latest_actual_price * projected["fitted_or_projected_price_usd"] / first_cycle
    return projected[["date", "btc_centerline_price", "btc_cycle_price"]].reset_index(drop=True)


def prepare_monthly_path(
    dates: pd.Series,
    btc_prices: pd.Series,
    current_age: float,
    end_age: float,
) -> PreparedMonthlyPath:
    """Convert a daily projected price path to one observation per month once."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "btc_price": pd.to_numeric(btc_prices, errors="coerce"),
    }).dropna()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError("The Bitcoin projection path is empty.")

    start_date = pd.Timestamp(frame["date"].iloc[0]).normalize()
    end_date = start_date + pd.Timedelta(days=round((end_age - current_age) * 365.25))
    frame = frame[frame["date"] <= end_date].copy()
    if frame.empty or frame["date"].max() < end_date - pd.Timedelta(days=35):
        raise ValueError("Projection horizon is too short for the selected ending age.")

    # First available observation in each month matches contribution/withdrawal timing.
    monthly = (
        frame.assign(month=frame["date"].dt.to_period("M"))
        .groupby("month", as_index=False)
        .first()
    )
    dates_index = pd.DatetimeIndex(monthly["date"])
    elapsed_days = (dates_index - start_date).days.to_numpy(dtype=float)
    years = elapsed_days / 365.25
    ages = current_age + years
    return PreparedMonthlyPath(
        dates=dates_index,
        btc_prices=monthly["btc_price"].to_numpy(dtype=float),
        ages=ages,
        years_from_start=years,
        start_date=start_date,
    )


def _monthly_growth_factor(annual_rate: float, compounds_per_year: int) -> float:
    if annual_rate <= -1:
        raise ValueError("Annual return must be greater than -100%.")
    m = max(int(compounds_per_year), 1)
    annual_factor = (1 + annual_rate / m) ** m
    return annual_factor ** (1 / 12)


def _simulate_prepared(
    scenario_name: str,
    prepared: PreparedMonthlyPath,
    current_age: float,
    financial_independence_age: float,
    end_age: float,
    other_principal: float,
    other_monthly_contribution: float,
    other_annual_return_while_contributing: float,
    other_annual_return_while_drawing: float,
    compounds_per_year: int,
    additions_at_start: bool,
    btc_principal_usd: float,
    btc_monthly_contribution: float,
    monthly_spending_today: float,
    inflation_rate: float,
    withdrawal_source: str,
    contribution_end_age: float | None = None,
    build_path: bool = True,
) -> FinancialIndependenceResult:
    if financial_independence_age < current_age:
        raise ValueError("Financial independence age cannot be below current age.")
    if end_age <= financial_independence_age:
        raise ValueError("Ending age must be above financial independence age.")

    contribution_end_age = financial_independence_age if contribution_end_age is None else contribution_end_age
    if contribution_end_age < current_age:
        raise ValueError("Contribution end age cannot be below current age.")
    if contribution_end_age > financial_independence_age:
        raise ValueError("Contribution end age cannot be above financial independence age.")

    fi_index = int(np.searchsorted(prepared.ages, financial_independence_age, side="left"))
    contribution_end_index = int(np.searchsorted(prepared.ages, contribution_end_age, side="left"))
    end_index = int(np.searchsorted(prepared.ages, end_age, side="right"))
    end_index = min(end_index, len(prepared.dates))
    if fi_index >= end_index:
        raise ValueError("Financial independence age is outside the prepared horizon.")

    fi_date = prepared.dates[fi_index]
    first_btc_price = float(prepared.btc_prices[0])
    btc_units = max(float(btc_principal_usd), 0.0) / first_btc_price
    other_balance = max(float(other_principal), 0.0)
    accumulation_factor = _monthly_growth_factor(other_annual_return_while_contributing, compounds_per_year)
    drawing_factor = _monthly_growth_factor(other_annual_return_while_drawing, compounds_per_year)

    total_contributions = max(float(other_principal), 0.0) + max(float(btc_principal_usd), 0.0)
    total_withdrawals = 0.0
    fi_balance = 0.0
    insolvent = False
    rows: list[dict] | None = [] if build_path else None

    for index in range(end_index):
        btc_price = float(prepared.btc_prices[index])
        financially_independent = index >= fi_index
        contributing = index < contribution_end_index

        if contributing and additions_at_start:
            other_balance += other_monthly_contribution
            btc_units += btc_monthly_contribution / btc_price
            total_contributions += other_monthly_contribution + btc_monthly_contribution

        other_balance *= drawing_factor if financially_independent else accumulation_factor

        if contributing and not additions_at_start:
            other_balance += other_monthly_contribution
            btc_units += btc_monthly_contribution / btc_price
            total_contributions += other_monthly_contribution + btc_monthly_contribution

        total_value = other_balance + btc_units * btc_price
        if index == fi_index:
            fi_balance = total_value

        if financially_independent:
            withdrawal = monthly_spending_today * ((1 + inflation_rate) ** prepared.years_from_start[index])
            if total_value + 1e-9 < withdrawal:
                insolvent = True
                total_withdrawals += max(total_value, 0.0)
                other_balance = 0.0
                btc_units = 0.0
            else:
                if withdrawal_source == "Proportional":
                    other_share = other_balance / total_value if total_value > 0 else 0.0
                    from_other = withdrawal * other_share
                    from_btc = withdrawal - from_other
                elif withdrawal_source == "BTC first":
                    from_btc = min(withdrawal, btc_units * btc_price)
                    from_other = withdrawal - from_btc
                elif withdrawal_source == "Other investments first":
                    from_other = min(withdrawal, other_balance)
                    from_btc = withdrawal - from_other
                else:
                    raise ValueError(f"Unknown withdrawal source: {withdrawal_source}")
                other_balance -= from_other
                btc_units -= from_btc / btc_price
                total_withdrawals += withdrawal

        total_value = max(other_balance + btc_units * btc_price, 0.0)
        if rows is not None:
            rows.append({
                "date": prepared.dates[index],
                "age": float(prepared.ages[index]),
                "other_investments": max(other_balance, 0.0),
                "btc_units": max(btc_units, 0.0),
                "btc_price": btc_price,
                "btc_value": max(btc_units * btc_price, 0.0),
                "total_portfolio": total_value,
                "financially_independent": financially_independent,
                "contributing": contributing,
            })
        if insolvent:
            break

    path = pd.DataFrame(rows) if rows is not None else pd.DataFrame()
    ending_balance = max(other_balance + btc_units * float(prepared.btc_prices[min(index, end_index - 1)]), 0.0)
    return FinancialIndependenceResult(
        scenario=scenario_name,
        sustainable=not insolvent,
        financial_independence_age=financial_independence_age,
        financial_independence_date=fi_date,
        ending_balance=float(ending_balance),
        financial_independence_balance=float(fi_balance),
        total_contributions=float(total_contributions),
        total_withdrawals=float(total_withdrawals),
        path=path,
    )


def simulate_financial_independence(
    scenario_name: str,
    dates: pd.Series,
    btc_prices: pd.Series,
    current_age: float,
    financial_independence_age: float,
    end_age: float,
    **kwargs,
) -> FinancialIndependenceResult:
    prepared = prepare_monthly_path(dates, btc_prices, current_age, end_age)
    return _simulate_prepared(
        scenario_name,
        prepared,
        current_age,
        financial_independence_age,
        end_age,
        **kwargs,
    )


def solve_earliest_financial_independence(
    scenario_name: str,
    dates: pd.Series,
    btc_prices: pd.Series,
    current_age: float,
    end_age: float,
    **kwargs,
) -> FinancialIndependenceResult:
    """Find the monthly FI boundary without brute-forcing every daily path."""
    prepared = prepare_monthly_path(dates, btc_prices, current_age, end_age)
    max_month = max(int(math.floor((end_age - current_age) * 12)) - 1, 0)

    def run(month: int, build_path: bool = False) -> FinancialIndependenceResult:
        age = current_age + month / 12
        return _simulate_prepared(
            scenario_name,
            prepared,
            current_age,
            age,
            end_age,
            contribution_end_age=age,
            build_path=build_path,
            **kwargs,
        )

    # If even the latest possible FI date fails, return that result immediately.
    latest = run(max_month, build_path=False)
    if not latest.sustainable:
        return run(max_month, build_path=True)

    # Sustainability is expected to be monotonic as FI is delayed: more months
    # of contributions and fewer months of withdrawals. Find the first success.
    low, high = 0, max_month
    while low < high:
        middle = (low + high) // 2
        if run(middle, build_path=False).sustainable:
            high = middle
        else:
            low = middle + 1

    # Verify the boundary locally in case rounding places it one month earlier.
    first_success = low
    for month in range(max(0, low - 2), low + 1):
        if run(month, build_path=False).sustainable:
            first_success = month
            break
    return run(first_success, build_path=True)


def solve_required_monthly_contribution(
    scenario_name: str,
    dates: pd.Series,
    btc_prices: pd.Series,
    current_age: float,
    target_financial_independence_age: float,
    end_age: float,
    bitcoin_conviction: float,
    coast_age: float | None,
    max_total_monthly: float = 400000.0,
    tolerance: float = 1.0,
    **kwargs,
) -> ContributionTargetResult:
    if not 0 <= bitcoin_conviction <= 1:
        raise ValueError("Bitcoin conviction must be between 0% and 100%.")
    contribution_end_age = target_financial_independence_age if coast_age is None else coast_age
    if contribution_end_age > target_financial_independence_age:
        raise ValueError("Coast age must be at or below the desired FI age.")

    prepared = prepare_monthly_path(dates, btc_prices, current_age, end_age)

    def run(total_monthly: float, build_path: bool = False) -> FinancialIndependenceResult:
        btc_monthly = total_monthly * bitcoin_conviction
        other_monthly = total_monthly - btc_monthly
        return _simulate_prepared(
            scenario_name,
            prepared,
            current_age,
            target_financial_independence_age,
            end_age,
            contribution_end_age=contribution_end_age,
            btc_monthly_contribution=btc_monthly,
            other_monthly_contribution=other_monthly,
            build_path=build_path,
            **kwargs,
        )

    zero = run(0.0, build_path=False)
    if zero.sustainable:
        final = run(0.0, build_path=True)
        return ContributionTargetResult(scenario_name, True, target_financial_independence_age, coast_age, bitcoin_conviction, 0.0, 0.0, 0.0, final)

    upper = 400.0
    upper_result = None
    while upper <= max_total_monthly:
        upper_result = run(upper, build_path=False)
        if upper_result.sustainable:
            break
        upper *= 2
    if upper_result is None or not upper_result.sustainable:
        return ContributionTargetResult(scenario_name, False, target_financial_independence_age, coast_age, bitcoin_conviction, None, None, None, upper_result)

    lower = 0.0
    while upper - lower > tolerance:
        middle = (lower + upper) / 2
        if run(middle, build_path=False).sustainable:
            upper = middle
        else:
            lower = middle

    final = run(upper, build_path=True)
    return ContributionTargetResult(
        scenario_name,
        True,
        target_financial_independence_age,
        coast_age,
        bitcoin_conviction,
        upper,
        upper * bitcoin_conviction,
        upper * (1 - bitcoin_conviction),
        final,
    )
