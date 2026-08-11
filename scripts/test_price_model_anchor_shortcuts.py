import numpy as np
import pandas as pd

from src.price_model import (
    FIXED_BEAR_DAYS,
    FIXED_BULL_DAYS,
    FIXED_CYCLE_DAYS,
    NEXT_TROUGH,
    SCHEDULED_PRE_2015_PEAK,
    SCHEDULED_PRE_2015_TROUGH,
    SCHEDULED_CYCLE_START,
    SCHEDULED_ZERO_START,
    _fixed_cycle_anchors,
    historical_cycle_anchors,
    fit_price_model,
    get_price_model_anchor_catalog,
)

# The early dates must be consequences of the SAME fixed cycle clock, not typed
# historical dates or a search for the largest early Bitcoin price.
assert SCHEDULED_PRE_2015_TROUGH == NEXT_TROUGH - pd.Timedelta(days=3 * FIXED_CYCLE_DAYS)
assert SCHEDULED_PRE_2015_PEAK == SCHEDULED_PRE_2015_TROUGH - pd.Timedelta(days=FIXED_BEAR_DAYS)
assert SCHEDULED_CYCLE_START == SCHEDULED_PRE_2015_PEAK - pd.Timedelta(days=FIXED_BULL_DAYS)
assert SCHEDULED_ZERO_START == SCHEDULED_CYCLE_START
assert SCHEDULED_PRE_2015_TROUGH == pd.Timestamp("2015-01-12")
assert SCHEDULED_PRE_2015_PEAK == pd.Timestamp("2014-01-13")
assert SCHEDULED_CYCLE_START == pd.Timestamp("2011-02-14")

# Synthetic daily data deliberately puts a MUCH larger price on 2013-11-30.
# The shortcut must ignore that maximum and still use the cycle-derived 2014-01-13 date.
dates = pd.date_range("2010-07-18", "2026-08-07", freq="D")
prices = np.exp(np.linspace(np.log(0.08), np.log(70000.0), len(dates)))
data = pd.DataFrame({"date": dates, "price_usd": prices})
data.loc[data["date"] == pd.Timestamp("2013-11-30"), "price_usd"] = 99999.0
data.loc[data["date"] == SCHEDULED_PRE_2015_PEAK, "price_usd"] = 876.54
data.loc[data["date"] == SCHEDULED_CYCLE_START, "price_usd"] = 0.71

for d, value in {
    "2015-01-14": 180.0,
    "2017-12-17": 19000.0,
    "2018-12-15": 3200.0,
    "2021-11-08": 67500.0,
    "2022-11-07": 20500.0,
    "2025-10-06": 125000.0,
}.items():
    data.loc[data["date"] == pd.Timestamp(d), "price_usd"] = value

catalog = get_price_model_anchor_catalog(data)
start = catalog[catalog["label"] == "Cycle-derived 2011 trough"]
assert len(start) == 1
assert pd.Timestamp(start.iloc[0]["date"]) == SCHEDULED_CYCLE_START
assert abs(float(start.iloc[0]["price_usd"]) - 0.71) < 1e-12
assert start.iloc[0]["type"] == "trough"

pre = catalog[catalog["label"] == "Cycle-derived pre-2015 peak"]
assert len(pre) == 1
assert pd.Timestamp(pre.iloc[0]["date"]) == SCHEDULED_PRE_2015_PEAK
assert abs(float(pre.iloc[0]["price_usd"]) - 876.54) < 1e-9

for label in [
    "2015 trough",
    "2017 peak",
    "2018 trough",
    "2021 peak",
    "2022 trough",
    "2025 peak",
]:
    assert label in set(catalog["label"])

schedule = _fixed_cycle_anchors(
    pd.Timestamp("2010-07-18"),
    pd.Timestamp("2030-01-01"),
    data=data,
)
scheduled_start = schedule[(schedule["date"] == SCHEDULED_CYCLE_START) & (schedule["type"] == "trough")]
assert len(scheduled_start) == 1
assert int(scheduled_start.iloc[0]["cycle"]) == -3
early = schedule[(schedule["date"] == SCHEDULED_PRE_2015_PEAK) & (schedule["type"] == "peak")]
assert len(early) == 1
assert int(early.iloc[0]["cycle"]) == -3

# If training spans the derived peak date, the fitted historical path intersects
# the actual Bitcoin price observed there. The cycle-start shortcut likewise uses
# the imported positive Bitcoin price at its cycle-derived date.
result = fit_price_model(
    prices=data,
    training_start=SCHEDULED_CYCLE_START,
    training_end=dates.max(),
    projection_years=3,
)
anchor_table = result.diagnostics["cycle_anchor_table"]
model_start = anchor_table[
    (pd.to_datetime(anchor_table["requested_anchor_date"]) == SCHEDULED_CYCLE_START)
    & (anchor_table["type"] == "trough")
    & (anchor_table["source"] == "historical market anchor")
]
assert len(model_start) == 1
assert abs(float(model_start.iloc[0]["actual_price_usd"]) - 0.71) < 1e-12

model_early = anchor_table[
    (pd.to_datetime(anchor_table["requested_anchor_date"]) == SCHEDULED_PRE_2015_PEAK)
    & (anchor_table["type"] == "peak")
    & (anchor_table["source"] == "historical market anchor")
]
assert len(model_early) == 1
assert abs(float(model_early.iloc[0]["actual_price_usd"]) - 876.54) < 1e-9

print("Cycle-derived 2011 trough and pre-2015 peak are plotted/intersection anchors.")
