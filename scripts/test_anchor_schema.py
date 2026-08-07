import pandas as pd

from src.price_model import HISTORICAL_CYCLE_ANCHORS, _fixed_cycle_anchors

schedule = _fixed_cycle_anchors(
    pd.Timestamp("2014-01-01"),
    pd.Timestamp("2030-01-01"),
)

assert {"date", "type", "cycle"}.issubset(schedule.columns)

required_dates = {
    pd.Timestamp("2015-01-14"),
    pd.Timestamp("2017-12-17"),
    pd.Timestamp("2018-12-15"),
    pd.Timestamp("2021-11-08"),
    pd.Timestamp("2022-11-07"),
    pd.Timestamp("2025-10-06"),
}
assert required_dates.issubset(set(schedule["date"]))

print("Anchor schema and requested historical dates verified.")
