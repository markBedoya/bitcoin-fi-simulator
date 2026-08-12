from __future__ import annotations

import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")

# Historical anchor points for the exploratory comparison page.
# The earlier peak follows the cycle-derived date the user chose.
CYCLE_ANCHORS = [
    {"date": pd.Timestamp("2011-02-14"), "type": "trough", "label": "2011 trough"},
    {"date": pd.Timestamp("2014-01-13"), "type": "peak", "label": "2014 peak"},
    {"date": pd.Timestamp("2015-01-14"), "type": "trough", "label": "2015 trough"},
    {"date": pd.Timestamp("2017-12-17"), "type": "peak", "label": "2017 peak"},
    {"date": pd.Timestamp("2018-12-15"), "type": "trough", "label": "2018 trough"},
    {"date": pd.Timestamp("2021-11-08"), "type": "peak", "label": "2021 peak"},
    {"date": pd.Timestamp("2022-11-07"), "type": "trough", "label": "2022 trough"},
    {"date": pd.Timestamp("2025-10-06"), "type": "peak", "label": "2025 peak"},
]

# Complete historical cycles currently available for this comparison page.
COMPLETE_CYCLES = [
    {
        "cycle_id": 0,
        "name": "Cycle 0",
        "display_name": "Cycle 0 — 2011 trough → 2014 peak → 2015 trough",
        "start": pd.Timestamp("2011-02-14"),
        "peak": pd.Timestamp("2014-01-13"),
        "end": pd.Timestamp("2015-01-14"),
    },
    {
        "cycle_id": 1,
        "name": "Cycle 1",
        "display_name": "Cycle 1 — 2015 trough → 2017 peak → 2018 trough",
        "start": pd.Timestamp("2015-01-14"),
        "peak": pd.Timestamp("2017-12-17"),
        "end": pd.Timestamp("2018-12-15"),
    },
    {
        "cycle_id": 2,
        "name": "Cycle 2",
        "display_name": "Cycle 2 — 2018 trough → 2021 peak → 2022 trough",
        "start": pd.Timestamp("2018-12-15"),
        "peak": pd.Timestamp("2021-11-08"),
        "end": pd.Timestamp("2022-11-07"),
    },
]

LIVE_STARTS = [
    {"fit_id": "live_2011", "label": "2011 trough → live data", "cycle_span": "2011→live", "start": pd.Timestamp("2011-02-14")},
    {"fit_id": "live_2015", "label": "2015 trough → live data", "cycle_span": "2015→live", "start": pd.Timestamp("2015-01-14")},
    {"fit_id": "live_2018", "label": "2018 trough → live data", "cycle_span": "2018→live", "start": pd.Timestamp("2018-12-15")},
]


def get_cycle_anchor_df(prices: pd.DataFrame) -> pd.DataFrame:
    price_lookup = prices.set_index("date")["price_usd"]
    rows = []
    for anchor in CYCLE_ANCHORS:
        price = price_lookup.get(anchor["date"], np.nan)
        rows.append(
            {
                "date": anchor["date"],
                "type": anchor["type"],
                "label": anchor["label"],
                "price_usd": float(price) if pd.notna(price) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_fit_windows(latest_date: pd.Timestamp) -> list[dict]:
    windows: list[dict] = []

    # Individual complete-cycle fits plus every contiguous complete-cycle combination.
    n = len(COMPLETE_CYCLES)
    for i in range(n):
        for j in range(i, n):
            selected = COMPLETE_CYCLES[i : j + 1]
            start = selected[0]["start"]
            end = selected[-1]["end"]
            cycle_ids = [str(c["cycle_id"]) for c in selected]
            if len(selected) == 1:
                label = selected[0]["display_name"]
                span = selected[0]["name"]
            else:
                label = (
                    f"Cycles {selected[0]['cycle_id']}–{selected[-1]['cycle_id']} combined"
                    f" — {start.date()} → {end.date()}"
                )
                span = f"Cycles {'-'.join(cycle_ids)}"
            windows.append(
                {
                    "fit_id": f"cycles_{selected[0]['cycle_id']}_{selected[-1]['cycle_id']}",
                    "label": label,
                    "cycle_span": span,
                    "start": start,
                    "end": end,
                    "group": "Complete cycle fits",
                }
            )

    # Additional live-data exploratory fits the user requested.
    for live_cfg in LIVE_STARTS:
        windows.append(
            {
                "fit_id": live_cfg["fit_id"],
                "label": f"{live_cfg['label']} — {live_cfg['start'].date()} → {latest_date.date()}",
                "cycle_span": live_cfg["cycle_span"],
                "start": live_cfg["start"],
                "end": latest_date,
                "group": "Live-data fits",
            }
        )
    return windows


def _fit_power_law(
    prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[float, float, pd.DataFrame, float, float]:
    segment = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
    if len(segment) < 3:
        raise ValueError(f"Not enough observations between {start.date()} and {end.date()}.")

    x_days = (segment["date"] - GENESIS).dt.days.to_numpy(dtype=float)
    y_price = segment["price_usd"].to_numpy(dtype=float)

    mask = (x_days > 0) & (y_price > 0)
    x_days = x_days[mask]
    y_price = y_price[mask]
    segment = segment.loc[mask].reset_index(drop=True)

    log_x = np.log(x_days)
    log_y = np.log(y_price)

    slope, intercept = np.polyfit(log_x, log_y, 1)
    log_y_hat = intercept + slope * log_x
    y_hat = np.exp(log_y_hat)

    rmse_log = float(np.sqrt(np.mean((log_y - log_y_hat) ** 2)))
    ss_res = float(np.sum((log_y - log_y_hat) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    fitted = segment[["date"]].copy()
    fitted["actual_price_usd"] = segment["price_usd"].to_numpy(dtype=float)
    fitted["centerline_price_usd"] = y_hat
    fitted["fit_start"] = start
    fitted["fit_end"] = end

    return float(slope), float(intercept), fitted, rmse_log, r2_log


def fit_cycle_combo_centerlines(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_dates = prices[["date"]].copy()
    x_days_all = (plot_dates["date"] - GENESIS).dt.days.to_numpy(dtype=float)
    latest_date = pd.Timestamp(prices["date"].max())

    fit_rows = []
    curve_rows = []

    for window in build_fit_windows(latest_date):
        slope, intercept, fitted, rmse_log, r2_log = _fit_power_law(
            prices,
            window["start"],
            window["end"],
        )
        full_curve = np.exp(intercept + slope * np.log(x_days_all))
        curve = plot_dates.copy()
        curve["fit_id"] = window["fit_id"]
        curve["label"] = window["label"]
        curve["cycle_span"] = window["cycle_span"]
        curve["fit_group"] = window["group"]
        curve["fit_start"] = window["start"]
        curve["fit_end"] = window["end"]
        curve["centerline_price_usd"] = full_curve
        curve["inside_fit_window"] = (curve["date"] >= window["start"]) & (curve["date"] <= window["end"])
        curve_rows.append(curve)

        start_idx = curve.index[curve["date"] == window["start"]]
        end_idx = curve.index[curve["date"] == window["end"]]
        centerline_start = float(curve.loc[start_idx[0], "centerline_price_usd"]) if len(start_idx) else float("nan")
        centerline_end = float(curve.loc[end_idx[0], "centerline_price_usd"]) if len(end_idx) else float("nan")

        fit_rows.append(
            {
                "fit_id": window["fit_id"],
                "label": window["label"],
                "cycle_span": window["cycle_span"],
                "fit_group": window["group"],
                "fit_start": window["start"],
                "fit_end": window["end"],
                "days_used": int(len(fitted)),
                "slope": slope,
                "intercept": intercept,
                "rmse_log": rmse_log,
                "r2_log": r2_log,
                "centerline_start_usd": centerline_start,
                "centerline_end_usd": centerline_end,
            }
        )

    fits_df = pd.DataFrame(fit_rows)
    curves_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    return fits_df, curves_df
