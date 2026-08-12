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


def get_current_cycle_progress(prices: pd.DataFrame) -> dict:
    """Estimate how much of the open cycle has elapsed from completed-cycle lengths."""
    latest_date = pd.Timestamp(prices["date"].max())
    current_start = COMPLETE_CYCLES[-1]["end"]
    completed_days = np.array(
        [(cycle["end"] - cycle["start"]).days for cycle in COMPLETE_CYCLES],
        dtype=float,
    )
    expected_days = int(round(float(np.median(completed_days))))
    elapsed_days = max(0, int((latest_date - current_start).days))
    progress = float(np.clip(elapsed_days / expected_days, 0.0, 1.0))
    return {
        "cycle_start": current_start,
        "latest_date": latest_date,
        "expected_cycle_days": expected_days,
        "estimated_cycle_end": current_start + pd.Timedelta(days=expected_days),
        "elapsed_days": elapsed_days,
        "progress": progress,
        "evidence_weight": progress,
    }


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


def fit_progress_weighted_backbone(prices: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit 2011→live while discounting the open cycle by its completion fraction.

    Each completed cycle receives total weight 1 regardless of small differences
    in duration. The open cycle receives total weight equal to its completion
    fraction, so a 96%-complete cycle supplies 0.96 cycle-equivalents of evidence.
    """
    progress = get_current_cycle_progress(prices)
    start = COMPLETE_CYCLES[0]["start"]
    latest = progress["latest_date"]
    segment = prices[(prices["date"] >= start) & (prices["date"] <= latest)].copy()
    x_days = (segment["date"] - GENESIS).dt.days.to_numpy(dtype=float)
    y_price = segment["price_usd"].to_numpy(dtype=float)
    valid = (x_days > 0) & (y_price > 0)
    segment = segment.loc[valid].reset_index(drop=True)
    log_x = np.log(x_days[valid])
    log_y = np.log(y_price[valid])
    weights = np.zeros(len(segment), dtype=float)
    for cycle in COMPLETE_CYCLES:
        in_cycle = (segment["date"] >= cycle["start"]) & (segment["date"] < cycle["end"])
        count = int(in_cycle.sum())
        if count:
            weights[in_cycle.to_numpy()] = 1.0 / count
    in_open_cycle = segment["date"] >= progress["cycle_start"]
    open_count = int(in_open_cycle.sum())
    if open_count:
        weights[in_open_cycle.to_numpy()] = progress["evidence_weight"] / open_count
    slope, intercept = np.polyfit(log_x, log_y, 1, w=np.sqrt(weights))
    fitted_log = intercept + slope * log_x
    rmse_log = float(np.sqrt(np.average((log_y - fitted_log) ** 2, weights=weights)))

    curve = prices[["date"]].copy()
    curve_days = (curve["date"] - GENESIS).dt.days.to_numpy(dtype=float)
    curve["centerline_price_usd"] = np.exp(intercept + slope * np.log(curve_days))
    row = {
        **progress,
        "fit_id": "progress_weighted_backbone",
        "label": "Progress-weighted 2011→live backbone",
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_log": rmse_log,
        "effective_current_cycle_weight": float(progress["evidence_weight"]),
        "live_centerline_usd": float(curve.iloc[-1]["centerline_price_usd"]),
        "live_actual_usd": float(prices.iloc[-1]["price_usd"]),
    }
    row["live_actual_to_centerline"] = row["live_actual_usd"] / row["live_centerline_usd"]
    return row, curve


def build_model_diagnostics(
    prices: pd.DataFrame,
    fits_df: pd.DataFrame,
    weighted_fit: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the compact diagnostics used to learn the mature-cycle geometry.

    October 2025 is treated as the confirmed current-cycle peak.  The latest
    observation is treated as a forming trough: it contributes in proportion
    to elapsed cycle progress but is never silently promoted to a completed
    historical trough.
    """
    expanding_ids = ["cycles_0_0", "cycles_0_1", "cycles_0_2", "live_2011"]
    expanding = fits_df.set_index("fit_id").loc[expanding_ids].reset_index()
    latest_date = pd.Timestamp(prices["date"].max())
    latest_price = float(prices.loc[prices["date"].idxmax(), "price_usd"])
    latest_days = float((latest_date - GENESIS).days)
    expanding["centerline_at_live_usd"] = np.exp(
        expanding["intercept"] + expanding["slope"] * np.log(latest_days)
    )
    expanding["live_actual_to_centerline"] = latest_price / expanding["centerline_at_live_usd"]
    expanding = expanding[[
        "fit_id", "label", "fit_start", "fit_end", "slope", "rmse_log", "r2_log",
        "centerline_at_live_usd", "live_actual_to_centerline",
    ]]

    anchors = get_cycle_anchor_df(prices)
    confirmed_peak_date = pd.Timestamp("2025-10-06")
    post_peak = prices[prices["date"] >= confirmed_peak_date].sort_values("date")
    trough_index = post_peak["price_usd"].idxmin()
    forming_trough_date = pd.Timestamp(prices.loc[trough_index, "date"])
    forming_trough_price = float(prices.loc[trough_index, "price_usd"])
    latest = pd.DataFrame([{
        "date": forming_trough_date,
        "type": "forming_trough",
        "label": "Current forming trough",
        "price_usd": forming_trough_price,
    }])
    deviations = pd.concat([anchors, latest], ignore_index=True)
    days = (deviations["date"] - GENESIS).dt.days.to_numpy(dtype=float)
    deviations["weighted_centerline_usd"] = np.exp(
        float(weighted_fit["intercept"]) + float(weighted_fit["slope"]) * np.log(days)
    )
    deviations["actual_to_centerline"] = deviations["price_usd"] / deviations["weighted_centerline_usd"]
    deviations["anchor_status"] = "completed"
    deviations.loc[deviations["type"] == "forming_trough", "anchor_status"] = "partial"
    deviations.loc[deviations["label"] == "2025 peak", "anchor_status"] = "confirmed"

    peak_rows = deviations[deviations["type"] == "peak"].reset_index(drop=True)
    trough_rows = deviations[deviations["type"].isin(["trough", "forming_trough"])].reset_index(drop=True)
    cycle_rows = []
    for i in range(min(len(peak_rows), len(trough_rows) - 1)):
        peak = peak_rows.iloc[i]
        following_trough = trough_rows.iloc[i + 1]
        peak_multiple = float(peak["actual_to_centerline"])
        trough_multiple = float(following_trough["actual_to_centerline"])
        cycle_rows.append({
            "cycle": i,
            "peak_label": peak["label"],
            "peak_multiple": peak_multiple,
            "trough_label": following_trough["label"],
            "trough_multiple": trough_multiple,
            "trough_status": following_trough["anchor_status"],
            "peak_to_trough_multiple_ratio": peak_multiple / trough_multiple,
            "log_peak_to_trough_amplitude": float(np.log(peak_multiple / trough_multiple)),
        })
    cycle_geometry = pd.DataFrame(cycle_rows)
    cycle_geometry["peak_compression_vs_prior"] = cycle_geometry["peak_multiple"].pct_change()
    cycle_geometry["amplitude_change_vs_prior"] = cycle_geometry[
        "log_peak_to_trough_amplitude"
    ].diff()

    mature_troughs = deviations[
        (deviations["type"] == "trough")
        & (deviations["date"] >= pd.Timestamp("2015-01-14"))
    ]["actual_to_centerline"]
    mature_floor = float(mature_troughs.median())
    current = deviations.iloc[-1]
    latest_centerline = float(
        np.exp(float(weighted_fit["intercept"]) + float(weighted_fit["slope"]) * np.log(latest_days))
    )
    floor_assessment = pd.DataFrame([{
        "forming_trough_date": forming_trough_date,
        "current_price_usd": float(current["price_usd"]),
        "current_centerline_usd": float(current["weighted_centerline_usd"]),
        "current_multiple": float(current["actual_to_centerline"]),
        "latest_date": latest_date,
        "latest_price_usd": latest_price,
        "latest_actual_to_centerline": latest_price / latest_centerline,
        "mature_completed_trough_median": mature_floor,
        "current_vs_mature_floor_pct": float(current["actual_to_centerline"] / mature_floor - 1.0),
        "cycle_progress": float(weighted_fit["progress"]),
        "remaining_expected_days": max(0, int(weighted_fit["expected_cycle_days"] - weighted_fit["elapsed_days"])),
        "model_interpretation": "forming bottom near mature-cycle floor",
        "market_context_assumption": "October 2025 confirmed peak; bear market; negative news no longer making new lows",
    }])

    completed_geometry = cycle_geometry[cycle_geometry["trough_status"] == "completed"]
    early_peak_median = float(completed_geometry.iloc[:2]["peak_multiple"].median())
    prior_peak = float(completed_geometry.iloc[-1]["peak_multiple"])
    current_peak = float(cycle_geometry.iloc[-1]["peak_multiple"])
    completed_backbone_live = float(
        expanding.loc[expanding["fit_id"] == "cycles_0_2", "centerline_at_live_usd"].iloc[0]
    )
    weighted_backbone_live = float(weighted_fit["live_centerline_usd"])
    maturity_transition = pd.DataFrame([{
        "early_peak_median_cycles_0_1": early_peak_median,
        "cycle_2_peak_multiple": prior_peak,
        "cycle_3_peak_multiple": current_peak,
        "cycle_2_vs_early_peak_pct": prior_peak / early_peak_median - 1.0,
        "cycle_3_vs_early_peak_pct": current_peak / early_peak_median - 1.0,
        "cycle_3_vs_cycle_2_peak_pct": current_peak / prior_peak - 1.0,
        "completed_trough_min": float(mature_troughs.min()),
        "completed_trough_max": float(mature_troughs.max()),
        "forming_trough_multiple": float(current["actual_to_centerline"]),
        "completed_cycle_backbone_live_usd": completed_backbone_live,
        "weighted_backbone_live_usd": weighted_backbone_live,
        "backbone_recalibration_pct": weighted_backbone_live / completed_backbone_live - 1.0,
        "pattern_read": "stable mature trough band; two-stage upside compression",
    }])

    # Compare forward structures without prematurely promoting one to the price model.
    # A valid mature-cycle upper boundary must remain above both the trough boundary
    # and the 1.0x centerline.  This guard exposes when an unconstrained amplitude
    # trend has crossed into structurally invalid territory.
    completed_floor_min = float(mature_troughs.min())
    completed_floor_max = float(mature_troughs.max())
    forming_floor = float(current["actual_to_centerline"])
    robust_floor = float(np.median([mature_floor, forming_floor]))
    current_amplitude = float(cycle_geometry.iloc[-1]["log_peak_to_trough_amplitude"])

    recent = cycle_geometry.iloc[1:].copy()
    x = recent["cycle"].to_numpy(dtype=float)
    y = recent["log_peak_to_trough_amplitude"].to_numpy(dtype=float)
    amplitude_slope, amplitude_intercept = np.polyfit(x, y, 1)
    next_cycle_id = int(cycle_geometry["cycle"].max()) + 1
    naive_next_amplitude = float(amplitude_intercept + amplitude_slope * next_cycle_id)
    naive_next_peak = float(robust_floor * np.exp(naive_next_amplitude))

    current_peak_excess = max(0.0, current_peak - 1.0)
    prior_peak_excess = max(0.0, prior_peak - 1.0)
    recent_excess_retention = (
        current_peak_excess / prior_peak_excess if prior_peak_excess > 0 else 0.0
    )
    bounded_next_peak = 1.0 + current_peak_excess * recent_excess_retention

    candidate_rows = [
        {
            "candidate": "regime_hold",
            "next_peak_multiple": current_peak,
            "next_trough_multiple": robust_floor,
            "next_log_amplitude": float(np.log(current_peak / robust_floor)),
            "structurally_valid": bool(current_peak >= 1.0 and current_peak > robust_floor),
            "role": "upper scenario",
            "interpretation": "2025 maturity arena persists for one more cycle",
        },
        {
            "candidate": "bounded_centerline_convergence",
            "next_peak_multiple": bounded_next_peak,
            "next_trough_multiple": robust_floor,
            "next_log_amplitude": float(np.log(bounded_next_peak / robust_floor)),
            "structurally_valid": bool(bounded_next_peak >= 1.0 and bounded_next_peak > robust_floor),
            "role": "lower scenario",
            "interpretation": "peak excess decays toward, but never through, the centerline",
        },
        {
            "candidate": "naive_log_amplitude_trend",
            "next_peak_multiple": naive_next_peak,
            "next_trough_multiple": robust_floor,
            "next_log_amplitude": naive_next_amplitude,
            "structurally_valid": bool(naive_next_peak >= 1.0 and naive_next_peak > robust_floor),
            "role": "rejection test",
            "interpretation": "unbounded recent amplitude trend; reject if peak falls below centerline",
        },
    ]
    forward_candidates = pd.DataFrame(candidate_rows)
    forward_candidates["forming_floor_vs_completed_min_pct"] = (
        forming_floor / completed_floor_min - 1.0
    )
    forward_candidates["forming_floor_vs_completed_max_pct"] = (
        forming_floor / completed_floor_max - 1.0
    )
    forward_candidates["recent_peak_excess_retention"] = recent_excess_retention
    forward_candidates["current_cycle_is_partial"] = True

    # Translate only the structurally valid envelope candidates into a single
    # forward cycle. Timing is learned separately from completed-cycle geometry:
    # the next peak occurs at the historical median trough-to-peak offset, not at
    # the estimated cycle boundary. The midpoint is descriptive scenario math,
    # not a third fitted law.
    peak_offsets = np.array(
        [(cycle["peak"] - cycle["start"]).days for cycle in COMPLETE_CYCLES],
        dtype=float,
    )
    decline_offsets = np.array(
        [(cycle["end"] - cycle["peak"]).days for cycle in COMPLETE_CYCLES],
        dtype=float,
    )
    cycle_lengths = np.array(
        [(cycle["end"] - cycle["start"]).days for cycle in COMPLETE_CYCLES],
        dtype=float,
    )
    timing_stability = pd.DataFrame([{
        "completed_cycles": len(COMPLETE_CYCLES),
        "trough_to_peak_min_days": int(peak_offsets.min()),
        "trough_to_peak_median_days": int(round(float(np.median(peak_offsets)))),
        "trough_to_peak_max_days": int(peak_offsets.max()),
        "peak_timing_range_days": int(peak_offsets.max() - peak_offsets.min()),
        "peak_to_trough_min_days": int(decline_offsets.min()),
        "peak_to_trough_median_days": int(round(float(np.median(decline_offsets)))),
        "peak_to_trough_max_days": int(decline_offsets.max()),
        "decline_timing_range_days": int(decline_offsets.max() - decline_offsets.min()),
        "cycle_length_min_days": int(cycle_lengths.min()),
        "cycle_length_median_days": int(round(float(np.median(cycle_lengths)))),
        "cycle_length_max_days": int(cycle_lengths.max()),
        "timing_read": "historical cycle timing is unusually stable",
    }])
    next_cycle_start = pd.Timestamp(weighted_fit["estimated_cycle_end"])
    median_peak_offset_days = int(round(float(np.median(peak_offsets))))
    next_peak_date = next_cycle_start + pd.Timedelta(days=median_peak_offset_days)
    next_trough_date = next_cycle_start + pd.Timedelta(days=int(weighted_fit["expected_cycle_days"]))

    def backbone_at(date: pd.Timestamp) -> float:
        model_days = float((date - GENESIS).days)
        return float(np.exp(float(weighted_fit["intercept"]) + float(weighted_fit["slope"]) * np.log(model_days)))

    peak_backbone = backbone_at(next_peak_date)
    trough_backbone = backbone_at(next_trough_date)
    valid = forward_candidates[forward_candidates["structurally_valid"]].copy()
    lower_peak_multiple = float(valid["next_peak_multiple"].min())
    upper_peak_multiple = float(valid["next_peak_multiple"].max())
    midpoint_peak_multiple = float(np.sqrt(lower_peak_multiple * upper_peak_multiple))
    projected_floor_multiple = robust_floor
    forward_price_scenarios = pd.DataFrame([
        {
            "scenario": "bounded_convergence_lower",
            "next_cycle_start": next_cycle_start,
            "projected_peak_date": next_peak_date,
            "peak_timing_basis": "median completed trough-to-peak offset",
            "peak_offset_days": median_peak_offset_days,
            "peak_backbone_usd": peak_backbone,
            "peak_multiple": lower_peak_multiple,
            "projected_peak_usd": peak_backbone * lower_peak_multiple,
            "projected_trough_date": next_trough_date,
            "trough_backbone_usd": trough_backbone,
            "trough_multiple": projected_floor_multiple,
            "projected_trough_usd": trough_backbone * projected_floor_multiple,
            "scenario_role": "lower boundary",
        },
        {
            "scenario": "geometric_midpoint",
            "next_cycle_start": next_cycle_start,
            "projected_peak_date": next_peak_date,
            "peak_timing_basis": "median completed trough-to-peak offset",
            "peak_offset_days": median_peak_offset_days,
            "peak_backbone_usd": peak_backbone,
            "peak_multiple": midpoint_peak_multiple,
            "projected_peak_usd": peak_backbone * midpoint_peak_multiple,
            "projected_trough_date": next_trough_date,
            "trough_backbone_usd": trough_backbone,
            "trough_multiple": projected_floor_multiple,
            "projected_trough_usd": trough_backbone * projected_floor_multiple,
            "scenario_role": "planning midpoint; not independently fitted",
        },
        {
            "scenario": "regime_hold_upper",
            "next_cycle_start": next_cycle_start,
            "projected_peak_date": next_peak_date,
            "peak_timing_basis": "median completed trough-to-peak offset",
            "peak_offset_days": median_peak_offset_days,
            "peak_backbone_usd": peak_backbone,
            "peak_multiple": upper_peak_multiple,
            "projected_peak_usd": peak_backbone * upper_peak_multiple,
            "projected_trough_date": next_trough_date,
            "trough_backbone_usd": trough_backbone,
            "trough_multiple": projected_floor_multiple,
            "projected_trough_usd": trough_backbone * projected_floor_multiple,
            "scenario_role": "upper boundary",
        },
    ])
    forward_price_scenarios["peak_to_trough_drawdown_pct"] = (
        forward_price_scenarios["projected_trough_usd"]
        / forward_price_scenarios["projected_peak_usd"]
        - 1.0
    )
    forward_price_scenarios["peak_to_trough_days"] = (
        forward_price_scenarios["projected_trough_date"]
        - forward_price_scenarios["projected_peak_date"]
    ).dt.days
    forward_price_scenarios["trough_above_current_peak_pct"] = (
        forward_price_scenarios["projected_trough_usd"]
        / float(deviations.loc[deviations["label"] == "2025 peak", "price_usd"].iloc[0])
        - 1.0
    )

    # Keep floor uncertainty separate from peak uncertainty. These three levels
    # are all observed-data summaries, not newly fitted decay laws: the partial
    # forming low, the robust partial/completed blend, and the completed mature
    # trough median. This makes the sensitivity explicit before extending the
    # envelope across many future cycles.
    current_peak_price = float(
        deviations.loc[deviations["label"] == "2025 peak", "price_usd"].iloc[0]
    )
    floor_sensitivity = pd.DataFrame([
        {
            "floor_path": "conservative_forming_floor_hold",
            "trough_multiple": forming_floor,
            "evidence_basis": "partial June 2026 forming trough",
            "projected_trough_date": next_trough_date,
            "projected_trough_usd": trough_backbone * forming_floor,
            "assumption_role": "lower floor boundary",
        },
        {
            "floor_path": "robust_partial_completed_blend",
            "trough_multiple": robust_floor,
            "evidence_basis": "median of partial floor and completed mature median",
            "projected_trough_date": next_trough_date,
            "projected_trough_usd": trough_backbone * robust_floor,
            "assumption_role": "baseline floor",
        },
        {
            "floor_path": "completed_mature_median",
            "trough_multiple": mature_floor,
            "evidence_basis": "median completed 2015, 2018, and 2022 troughs",
            "projected_trough_date": next_trough_date,
            "projected_trough_usd": trough_backbone * mature_floor,
            "assumption_role": "gently rising floor boundary",
        },
    ])
    lower_peak_usd = peak_backbone * lower_peak_multiple
    upper_peak_usd = peak_backbone * upper_peak_multiple
    floor_sensitivity["drawdown_from_lower_peak_pct"] = (
        floor_sensitivity["projected_trough_usd"] / lower_peak_usd - 1.0
    )
    floor_sensitivity["drawdown_from_upper_peak_pct"] = (
        floor_sensitivity["projected_trough_usd"] / upper_peak_usd - 1.0
    )
    floor_sensitivity["trough_above_current_peak_pct"] = (
        floor_sensitivity["projected_trough_usd"] / current_peak_price - 1.0
    )
    floor_sensitivity["trough_range_span_pct"] = (
        floor_sensitivity["projected_trough_usd"].max()
        / floor_sensitivity["projected_trough_usd"].min()
        - 1.0
    )

    # Extend the bounded structures across five cycles as a structural stress
    # test. The lower peak path compounds the observed retention of excess above
    # 1.0x. The upper path allows the 2025 regime to persist for one more cycle,
    # then converges by the same bounded rule. Floor paths are deliberately kept
    # independent: fixed partial, fixed robust, or a labeled half-gap recovery
    # toward the completed mature median.
    multi_cycle_rows = []
    lower_excess = current_peak_excess
    upper_excess = current_peak_excess
    recovery_floor = robust_floor
    future_cycle_count = 5
    for horizon_cycle in range(1, future_cycle_count + 1):
        cycle_start = next_cycle_start + pd.Timedelta(
            days=(horizon_cycle - 1) * int(weighted_fit["expected_cycle_days"])
        )
        peak_date = cycle_start + pd.Timedelta(days=median_peak_offset_days)
        trough_date = cycle_start + pd.Timedelta(days=int(weighted_fit["expected_cycle_days"]))
        lower_excess *= recent_excess_retention
        if horizon_cycle == 1:
            upper_excess = current_peak_excess
        else:
            upper_excess *= recent_excess_retention
        lower_multiple = 1.0 + lower_excess
        upper_multiple = 1.0 + upper_excess
        midpoint_multiple = float(np.sqrt(lower_multiple * upper_multiple))
        if horizon_cycle > 1:
            recovery_floor += 0.5 * (mature_floor - recovery_floor)
        peak_backbone_h = backbone_at(peak_date)
        trough_backbone_h = backbone_at(trough_date)
        for peak_path, peak_multiple_h in [
            ("bounded_convergence_lower", lower_multiple),
            ("geometric_planning_midpoint", midpoint_multiple),
            ("one_cycle_regime_hold_upper", upper_multiple),
        ]:
            for floor_path, floor_multiple_h in [
                ("conservative_forming_floor_hold", forming_floor),
                ("robust_floor_hold", robust_floor),
                ("half_gap_recovery_to_completed_median", recovery_floor),
            ]:
                peak_usd_h = peak_backbone_h * peak_multiple_h
                trough_usd_h = trough_backbone_h * floor_multiple_h
                multi_cycle_rows.append({
                    "horizon_cycle": horizon_cycle,
                    "cycle_start": cycle_start,
                    "projected_peak_date": peak_date,
                    "projected_trough_date": trough_date,
                    "peak_path": peak_path,
                    "floor_path": floor_path,
                    "peak_multiple": peak_multiple_h,
                    "floor_multiple": floor_multiple_h,
                    "peak_backbone_usd": peak_backbone_h,
                    "trough_backbone_usd": trough_backbone_h,
                    "projected_peak_usd": peak_usd_h,
                    "projected_trough_usd": trough_usd_h,
                    "peak_to_trough_drawdown_pct": trough_usd_h / peak_usd_h - 1.0,
                    "structural_status": "bounded diagnostic; not production projection",
                })
    multi_cycle_stress_test = pd.DataFrame(multi_cycle_rows)

    # Long-horizon dollar outcomes are increasingly dominated by the backbone,
    # not the cycle envelope. Anchor every sensitivity path to the same live
    # centerline, then retain 100%, 90%, or 80% of the learned log-time exponent.
    # The reduced-exponent paths are transparent stress tests, not fitted laws.
    live_model_days = float((latest_date - GENESIS).days)
    live_centerline = float(weighted_fit["live_centerline_usd"])
    backbone_sensitivity_rows = []
    backbone_paths = [
        ("learned_exponent_hold", 1.00, "baseline structural extrapolation"),
        ("moderate_exponent_compression", 0.90, "sensitivity: retain 90% of learned exponent"),
        ("conservative_exponent_compression", 0.80, "sensitivity: retain 80% of learned exponent"),
    ]
    anchor_dates = sorted(
        set(multi_cycle_stress_test["projected_peak_date"].tolist())
        | set(multi_cycle_stress_test["projected_trough_date"].tolist())
    )
    peak_dates_set = set(multi_cycle_stress_test["projected_peak_date"].tolist())
    for path_name, exponent_retention, path_role in backbone_paths:
        effective_exponent = float(weighted_fit["slope"]) * exponent_retention
        for anchor_date in anchor_dates:
            anchor_date = pd.Timestamp(anchor_date)
            anchor_days = float((anchor_date - GENESIS).days)
            years_from_live = (anchor_date - latest_date).days / 365.2425
            projected_backbone = live_centerline * (anchor_days / live_model_days) ** effective_exponent
            backbone_sensitivity_rows.append({
                "backbone_path": path_name,
                "anchor_type": "peak" if anchor_date in peak_dates_set else "trough",
                "anchor_date": anchor_date,
                "exponent_retention": exponent_retention,
                "effective_exponent": effective_exponent,
                "projected_backbone_usd": projected_backbone,
                "cagr_from_live_actual_pct": (
                    (projected_backbone / latest_price) ** (1.0 / years_from_live) - 1.0
                ),
                "cagr_from_live_centerline_pct": (
                    (projected_backbone / live_centerline) ** (1.0 / years_from_live) - 1.0
                ),
                "path_role": path_role,
            })
    backbone_sensitivity = pd.DataFrame(backbone_sensitivity_rows)

    # Test a decay mechanism tied to observed model recalibration rather than an
    # arbitrary terminal retention. The completed-history exponent is compared
    # with the progress-weighted exponent. Paths are integrated piecewise in
    # log-time so exponent changes never create price discontinuities.
    completed_exponent = float(
        expanding.loc[expanding["fit_id"] == "cycles_0_2", "slope"].iloc[0]
    )
    raw_exponent_retention = float(weighted_fit["slope"]) / completed_exponent
    recent_exponent_retention = float(np.clip(raw_exponent_retention, 0.0, 1.0))
    recursive_rows = []
    recursive_paths = [
        ("learned_exponent_hold", "hold current learned exponent"),
        ("one_time_recalibration_then_hold", "apply observed retention once, then hold"),
        ("repeat_observed_recalibration", "repeat observed retention once per future cycle"),
    ]

    def recursive_backbone_at(target_date: pd.Timestamp, path_name: str) -> tuple[float, float, int]:
        target_date = pd.Timestamp(target_date)
        price_level = live_centerline
        segment_start = latest_date
        cycle_number = 0
        active_exponent = float(weighted_fit["slope"])
        boundaries = [next_cycle_start]
        for boundary_number in range(1, future_cycle_count + 1):
            boundaries.append(
                next_cycle_start
                + pd.Timedelta(days=boundary_number * int(weighted_fit["expected_cycle_days"]))
            )
        for boundary in boundaries:
            segment_end = min(target_date, boundary)
            if segment_end > segment_start:
                start_days = float((segment_start - GENESIS).days)
                end_days = float((segment_end - GENESIS).days)
                price_level *= (end_days / start_days) ** active_exponent
                segment_start = segment_end
            if target_date <= boundary:
                break
            cycle_number += 1
            if path_name == "one_time_recalibration_then_hold":
                active_exponent = float(weighted_fit["slope"]) * recent_exponent_retention
            elif path_name == "repeat_observed_recalibration":
                active_exponent = float(weighted_fit["slope"]) * recent_exponent_retention ** cycle_number
        return price_level, active_exponent, cycle_number

    for path_name, path_role in recursive_paths:
        for anchor_date in anchor_dates:
            anchor_date = pd.Timestamp(anchor_date)
            projected_backbone, active_exponent, cycle_number = recursive_backbone_at(
                anchor_date, path_name
            )
            years_from_live = (anchor_date - latest_date).days / 365.2425
            recursive_rows.append({
                "recursive_path": path_name,
                "anchor_type": "peak" if anchor_date in peak_dates_set else "trough",
                "anchor_date": anchor_date,
                "future_cycle": cycle_number,
                "completed_history_exponent": completed_exponent,
                "current_weighted_exponent": float(weighted_fit["slope"]),
                "raw_exponent_retention": raw_exponent_retention,
                "observed_exponent_retention": recent_exponent_retention,
                "active_segment_exponent": active_exponent,
                "projected_backbone_usd": projected_backbone,
                "cagr_from_live_actual_pct": (
                    (projected_backbone / latest_price) ** (1.0 / years_from_live) - 1.0
                ),
                "cagr_from_live_centerline_pct": (
                    (projected_backbone / live_centerline) ** (1.0 / years_from_live) - 1.0
                ),
                "continuity_rule": "piecewise log-time integration; no boundary reset",
                "path_role": path_role,
            })
    recursive_exponent_candidates = pd.DataFrame(recursive_rows)

    return (
        expanding,
        deviations,
        cycle_geometry,
        floor_assessment,
        maturity_transition,
        forward_candidates,
        forward_price_scenarios,
        timing_stability,
        floor_sensitivity,
        multi_cycle_stress_test,
        backbone_sensitivity,
        recursive_exponent_candidates,
    )


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


def build_common_date_comparison(
    fits_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every fitted power law at the same structural checkpoint dates.

    Returns:
        comparison: one row per fit plus an Actual BTC row.
        spread_summary: cross-fit min/median/max/spread at each checkpoint.
    """
    latest_date = pd.Timestamp(prices["date"].max())
    checkpoint_specs = [
        ("2011 trough", pd.Timestamp("2011-02-14")),
        ("2014 peak", pd.Timestamp("2014-01-13")),
        ("2015 trough", pd.Timestamp("2015-01-14")),
        ("2018 trough", pd.Timestamp("2018-12-15")),
        ("2022 trough", pd.Timestamp("2022-11-07")),
        ("2025 peak", pd.Timestamp("2025-10-06")),
        ("Live", latest_date),
    ]

    price_lookup = prices.set_index("date")["price_usd"]
    actual_row = {
        "fit_group": "Actual BTC",
        "label": "Actual Bitcoin price",
        "fit_start": pd.NaT,
        "fit_end": pd.NaT,
        "slope": np.nan,
    }
    for label, date in checkpoint_specs:
        actual = price_lookup.get(date, np.nan)
        actual_row[f"{label} | {date.date()}"] = float(actual) if pd.notna(actual) else np.nan

    rows = [actual_row]
    for _, fit in fits_df.iterrows():
        row = {
            "fit_group": fit["fit_group"],
            "label": fit["label"],
            "fit_start": pd.Timestamp(fit["fit_start"]),
            "fit_end": pd.Timestamp(fit["fit_end"]),
            "slope": float(fit["slope"]),
        }
        for checkpoint_label, date in checkpoint_specs:
            days = float((date - GENESIS).days)
            value = np.exp(float(fit["intercept"]) + float(fit["slope"]) * np.log(days))
            row[f"{checkpoint_label} | {date.date()}"] = float(value)
        rows.append(row)

    comparison = pd.DataFrame(rows)

    fit_only = comparison[comparison["fit_group"] != "Actual BTC"].copy()
    spread_rows = []
    for checkpoint_label, date in checkpoint_specs:
        col = f"{checkpoint_label} | {date.date()}"
        vals = pd.to_numeric(fit_only[col], errors="coerce").dropna()
        actual = price_lookup.get(date, np.nan)
        if vals.empty:
            continue
        minimum = float(vals.min())
        median = float(vals.median())
        maximum = float(vals.max())
        spread_rows.append(
            {
                "checkpoint": checkpoint_label,
                "date": date,
                "actual_btc_usd": float(actual) if pd.notna(actual) else np.nan,
                "centerline_min_usd": minimum,
                "centerline_median_usd": median,
                "centerline_max_usd": maximum,
                "max_min_ratio": maximum / minimum if minimum > 0 else np.nan,
                "range_pct_of_median": (maximum - minimum) / median if median > 0 else np.nan,
                "median_vs_actual_ratio": median / float(actual) if pd.notna(actual) and float(actual) > 0 else np.nan,
            }
        )

    spread_summary = pd.DataFrame(spread_rows)
    return comparison, spread_summary
