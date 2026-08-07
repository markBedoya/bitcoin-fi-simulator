import pandas as pd

from src.price_model import (
    FIXED_BEAR_DAYS,
    FIXED_BULL_DAYS,
    FIXED_CYCLE_DAYS,
    HISTORICAL_CYCLE_ANCHORS,
    NEXT_TROUGH,
    REFERENCE_PEAK,
    REFERENCE_TROUGH,
    _fixed_cycle_anchors,
)

assert FIXED_BULL_DAYS == 1064
assert FIXED_BEAR_DAYS == 364
assert FIXED_CYCLE_DAYS == 1428
assert FIXED_BULL_DAYS + FIXED_BEAR_DAYS == FIXED_CYCLE_DAYS

assert REFERENCE_TROUGH == pd.Timestamp("2022-11-07")
assert REFERENCE_PEAK == pd.Timestamp("2025-10-06")
assert NEXT_TROUGH == pd.Timestamp("2026-10-05")
assert (REFERENCE_PEAK - REFERENCE_TROUGH).days == 1064
assert (NEXT_TROUGH - REFERENCE_PEAK).days == 364

historical = {(d, t) for d, t, _ in HISTORICAL_CYCLE_ANCHORS}
required = {
    (pd.Timestamp("2015-01-14"), "trough"),
    (pd.Timestamp("2017-12-17"), "peak"),
    (pd.Timestamp("2018-12-15"), "trough"),
    (pd.Timestamp("2021-11-08"), "peak"),
    (pd.Timestamp("2022-11-07"), "trough"),
    (pd.Timestamp("2025-10-06"), "peak"),
}
assert required.issubset(historical)

schedule = _fixed_cycle_anchors(
    pd.Timestamp("2014-01-01"),
    pd.Timestamp("2035-01-01"),
)

for date, anchor_type in required:
    match = schedule[
        (schedule["date"] == date)
        & (schedule["type"] == anchor_type)
    ]
    assert len(match) == 1

future_troughs = schedule[
    (schedule["type"] == "trough")
    & (schedule["date"] >= NEXT_TROUGH)
]["date"].sort_values().tolist()
assert len(future_troughs) >= 2
assert (future_troughs[1] - future_troughs[0]).days == 1428

first_future_peak = schedule[
    (schedule["type"] == "peak")
    & (schedule["date"] > NEXT_TROUGH)
]["date"].min()
assert (first_future_peak - NEXT_TROUGH).days == 1064

print("Actual historical anchors and fixed 1428-day future cycle checks passed.")
