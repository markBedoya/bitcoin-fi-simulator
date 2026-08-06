from __future__ import annotations

from dataclasses import dataclass
import math
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


def _daily_rate(annual_rate: float, compounds_per_year: int) -> float:
    if annual_rate <= -1:
        raise ValueError("Annual return must be greater than -100%.")
    m = max(int(compounds_per_year), 1)
    annual_factor = (1 + annual_rate / m) ** m
    return annual_factor ** (1 / 365.25) - 1


def simulate_financial_independence(
    scenario_name: str,
    dates: pd.Series,
    btc_prices: pd.Series,
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

    start_date = pd.Timestamp(dates.iloc[0]).normalize()
    fi_date = start_date + pd.Timedelta(days=round((financial_independence_age-current_age)*365.25))
    contribution_end_date = start_date + pd.Timedelta(days=round((contribution_end_age-current_age)*365.25))
    end_date = start_date + pd.Timedelta(days=round((end_age-current_age)*365.25))

    frame = pd.DataFrame({"date": pd.to_datetime(dates), "btc_price": pd.to_numeric(btc_prices)})
    frame = frame[frame["date"] <= end_date].copy()
    if frame.empty or frame["date"].max() < end_date - pd.Timedelta(days=3):
        raise ValueError("Projection horizon is too short for the selected ending age.")

    btc_units = max(float(btc_principal_usd),0.0)/float(frame["btc_price"].iloc[0])
    other_balance = max(float(other_principal),0.0)
    accumulation_daily_rate = _daily_rate(other_annual_return_while_contributing, compounds_per_year)
    drawing_daily_rate = _daily_rate(other_annual_return_while_drawing, compounds_per_year)
    total_contributions = max(float(other_principal),0.0)+max(float(btc_principal_usd),0.0)
    total_withdrawals=0.0
    fi_balance=None
    last_contribution_month=None
    last_withdrawal_month=None
    insolvent=False
    rows=[]

    for row in frame.itertuples(index=False):
        date=pd.Timestamp(row.date); btc_price=float(row.btc_price)
        financially_independent=date>=fi_date
        contributing=date<contribution_end_date
        month_key=(date.year,date.month)
        should_add=contributing and month_key!=last_contribution_month

        if should_add and additions_at_start:
            other_balance += other_monthly_contribution
            btc_units += btc_monthly_contribution/btc_price
            total_contributions += other_monthly_contribution+btc_monthly_contribution
            last_contribution_month=month_key

        other_balance *= 1 + (drawing_daily_rate if financially_independent else accumulation_daily_rate)

        if should_add and not additions_at_start:
            other_balance += other_monthly_contribution
            btc_units += btc_monthly_contribution/btc_price
            total_contributions += other_monthly_contribution+btc_monthly_contribution
            last_contribution_month=month_key

        total_value=other_balance+btc_units*btc_price
        if financially_independent and fi_balance is None:
            fi_balance=total_value

        if financially_independent and month_key!=last_withdrawal_month:
            years_from_start=(date-start_date).days/365.25
            withdrawal=monthly_spending_today*((1+inflation_rate)**years_from_start)
            if total_value < withdrawal:
                insolvent=True; total_withdrawals += max(total_value,0.0); other_balance=0.0; btc_units=0.0
            else:
                if withdrawal_source=="Proportional":
                    other_share=other_balance/total_value if total_value>0 else 0.0
                    from_other=withdrawal*other_share; from_btc=withdrawal-from_other
                elif withdrawal_source=="BTC first":
                    from_btc=min(withdrawal,btc_units*btc_price); from_other=withdrawal-from_btc
                elif withdrawal_source=="Other investments first":
                    from_other=min(withdrawal,other_balance); from_btc=withdrawal-from_other
                else:
                    raise ValueError(f"Unknown withdrawal source: {withdrawal_source}")
                other_balance -= from_other; btc_units -= from_btc/btc_price; total_withdrawals += withdrawal
            last_withdrawal_month=month_key

        total_value=max(other_balance+btc_units*btc_price,0.0)
        rows.append({"date":date,"age":current_age+(date-start_date).days/365.25,"other_investments":max(other_balance,0.0),"btc_units":max(btc_units,0.0),"btc_price":btc_price,"btc_value":max(btc_units*btc_price,0.0),"total_portfolio":total_value,"financially_independent":financially_independent,"contributing":contributing})
        if insolvent: break

    path=pd.DataFrame(rows)
    ending_balance=float(path["total_portfolio"].iloc[-1]) if not path.empty else 0.0
    return FinancialIndependenceResult(scenario_name,not insolvent,financial_independence_age,fi_date,ending_balance,float(fi_balance or 0.0),float(total_contributions),float(total_withdrawals),path)


def solve_earliest_financial_independence(scenario_name: str, dates: pd.Series, btc_prices: pd.Series, current_age: float, end_age: float, **kwargs) -> FinancialIndependenceResult:
    months=int(math.floor((end_age-current_age)*12)); last=None
    for month in range(months+1):
        age=current_age+month/12
        if age>=end_age: break
        last=simulate_financial_independence(scenario_name,dates,btc_prices,current_age,age,end_age,contribution_end_age=age,**kwargs)
        if last.sustainable: return last
    return last


def solve_required_monthly_contribution(scenario_name: str, dates: pd.Series, btc_prices: pd.Series, current_age: float, target_financial_independence_age: float, end_age: float, bitcoin_conviction: float, coast_age: float | None, max_total_monthly: float=400000.0, tolerance: float=1.0, **kwargs) -> ContributionTargetResult:
    if not 0<=bitcoin_conviction<=1: raise ValueError("Bitcoin conviction must be between 0% and 100%.")
    contribution_end_age=target_financial_independence_age if coast_age is None else coast_age
    if contribution_end_age>target_financial_independence_age: raise ValueError("Coast age must be at or below the desired FI age.")
    def run(total_monthly: float):
        btc_monthly=total_monthly*bitcoin_conviction; other_monthly=total_monthly-btc_monthly
        return simulate_financial_independence(scenario_name,dates,btc_prices,current_age,target_financial_independence_age,end_age,contribution_end_age=contribution_end_age,btc_monthly_contribution=btc_monthly,other_monthly_contribution=other_monthly,**kwargs)
    zero=run(0.0)
    if zero.sustainable:
        return ContributionTargetResult(scenario_name,True,target_financial_independence_age,coast_age,bitcoin_conviction,0.0,0.0,0.0,zero)
    upper=400.0; upper_result=None
    while upper<=max_total_monthly:
        upper_result=run(upper)
        if upper_result.sustainable: break
        upper*=2
    if upper_result is None or not upper_result.sustainable:
        return ContributionTargetResult(scenario_name,False,target_financial_independence_age,coast_age,bitcoin_conviction,None,None,None,upper_result)
    lower=0.0; best=upper_result
    for _ in range(70):
        if upper-lower<=tolerance: break
        middle=(lower+upper)/2; result=run(middle)
        if result.sustainable: upper=middle; best=result
        else: lower=middle
    return ContributionTargetResult(scenario_name,True,target_financial_independence_age,coast_age,bitcoin_conviction,upper,upper*bitcoin_conviction,upper*(1-bitcoin_conviction),best)
