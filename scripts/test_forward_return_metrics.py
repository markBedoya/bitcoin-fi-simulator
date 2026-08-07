"""Regression checks for Price Model return metric definitions."""

def lump_sum_cagr(start_price, end_price, years):
    return (end_price / start_price) ** (1 / years) - 1

def forward_return_1_year(price_at_horizon, price_one_year_later):
    return price_one_year_later / price_at_horizon - 1

today = 100.0
year_10 = 750.0
year_11 = 837.0

cagr_10 = lump_sum_cagr(today, year_10, 10)
forward_10 = forward_return_1_year(year_10, year_11)

assert cagr_10 > forward_10
assert abs(forward_10 - 0.116) < 0.002
assert forward_return_1_year(500.0, 500.0) == 0.0

print("Forward Return 1 Year metric checks passed.")
