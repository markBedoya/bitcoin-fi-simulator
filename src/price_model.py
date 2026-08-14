from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MODEL_VERSION = "bottom-anchored-dynamic-settling-v0.5.0"
SUMMARY_SCHEMA = "bitcoin-dynamic-settling-summary-v5"
GENESIS = pd.Timestamp("2009-01-03")
BOTTOM_WINDOW_DAYS = 120
PEAK_WINDOW_DAYS = 90
REGION_CLUSTER_DAYS = 7
VALIDATION_TEMPERATURE = 0.35
REFERENCE_DAYS_BEFORE_TURN = 56
SETTLING_GRID = tuple(np.round(np.arange(0.05, 1.001, 0.05), 2))
SETTLING_LINEAR_PRIOR_CYCLES = 2.0

BOTTOM_TURNING_REGIONS = [
    {"anchor_date": pd.Timestamp("2011-11-18"), "label": "2011 bottom region"},
    {"anchor_date": pd.Timestamp("2015-01-14"), "label": "2015 bottom region"},
    {"anchor_date": pd.Timestamp("2018-12-15"), "label": "2018 bottom region"},
    {"anchor_date": pd.Timestamp("2022-11-21"), "label": "2022 bottom region"},
    {"anchor_date": pd.Timestamp("2026-10-07"), "label": "2026 forming bottom region"},
]

PEAK_TURNING_REGIONS = [
    {"anchor_date": pd.Timestamp("2013-12-04"), "label": "2013 peak region"},
    {"anchor_date": pd.Timestamp("2017-12-17"), "label": "2017 peak region"},
    {"anchor_date": pd.Timestamp("2021-11-08"), "label": "2021 peak region"},
    {"anchor_date": pd.Timestamp("2025-10-06"), "label": "2025 peak region"},
]

CANDIDATE_LABELS = {
    "expanding_power_law": "Expanding-history bottom power law",
    "all_cycle_excess_decay": "All-cycle excess-growth decay",
    "recent_excess_decay": "Recent excess-growth decay",
    "local_exponent": "Recency-weighted local exponent",
}

FAIR_METHOD_LABELS = {
    "peak_bottom_midpoint": "Peak/bottom log midpoint",
    "cycle_log_median": "Time-weighted cycle median",
    "cycle_log_mean": "Time-weighted geometric mean",
    "central_50_midpoint": "Central-50% log midpoint",
}


@dataclass
class BottomAnchoredModelResult:
    summary: dict
    bottom_regions: pd.DataFrame
    peak_regions: pd.DataFrame
    curve: pd.DataFrame
    mature_cycle_forecast: pd.DataFrame
    forming_prior_forecasts: pd.DataFrame
    validation_summary: pd.DataFrame
    walk_forward: pd.DataFrame
    bottom_sensitivity: pd.DataFrame
    fair_value_cycles: pd.DataFrame
    fair_value_validation: pd.DataFrame
    fair_value_methods: pd.DataFrame
    settling_calibration: pd.DataFrame
    settling_calibration_detail: pd.DataFrame
    settling_leave_one_out: pd.DataFrame
    settling_cycle_dependence: pd.DataFrame
    dynamic_settling: pd.DataFrame
    dynamic_settling_summary: pd.DataFrame
    all_price_curve: pd.DataFrame


def _normalise(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices[["date", "price_usd"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    data["price_usd"] = pd.to_numeric(data["price_usd"], errors="coerce")
    data = data.dropna(subset=["date", "price_usd"])
    data = data[np.isfinite(data["price_usd"]) & (data["price_usd"] > 0)]
    return data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _median_timestamp(values: pd.Series) -> pd.Timestamp:
    ordered = pd.to_datetime(values).sort_values().reset_index(drop=True)
    return pd.Timestamp(ordered.iloc[len(ordered) // 2])


def _region_price(values: pd.Series, statistic: str) -> float:
    array = values.to_numpy(dtype=float)
    if statistic == "median":
        return float(np.median(array))
    if statistic == "geometric_mean":
        return float(np.exp(np.mean(np.log(array))))
    raise ValueError(f"Unknown region statistic: {statistic}")


def _extract_regions(
    data: pd.DataFrame,
    specifications: list[dict],
    half_window_days: int,
    direction: str,
    cluster_days: int = REGION_CLUSTER_DAYS,
    statistic: str = "median",
) -> pd.DataFrame:
    latest = pd.Timestamp(data["date"].max())
    rows: list[dict] = []
    for cycle, specification in enumerate(specifications):
        anchor = pd.Timestamp(specification["anchor_date"])
        start = anchor - pd.Timedelta(days=half_window_days)
        end = min(anchor + pd.Timedelta(days=half_window_days), latest)
        window = data[(data["date"] >= start) & (data["date"] <= end)].copy()
        if len(window) < cluster_days:
            continue
        if direction == "bottom":
            cluster = window.nsmallest(cluster_days, "price_usd").copy()
            extreme_row = window.loc[window["price_usd"].idxmin()]
        else:
            cluster = window.nlargest(cluster_days, "price_usd").copy()
            extreme_row = window.loc[window["price_usd"].idxmax()]
        status = "completed" if latest >= anchor + pd.Timedelta(days=half_window_days) else "forming"
        rows.append({
            "cycle": cycle,
            "label": specification["label"],
            "anchor_date": anchor,
            "region_date": _median_timestamp(cluster["date"]),
            "region_price_usd": _region_price(cluster["price_usd"], statistic),
            "cluster_low_usd": float(cluster["price_usd"].min()),
            "cluster_high_usd": float(cluster["price_usd"].max()),
            "extreme_date": pd.Timestamp(extreme_row["date"]),
            "extreme_price_usd": float(extreme_row["price_usd"]),
            "window_start": start,
            "window_end": end,
            "observations": int(len(window)),
            "cluster_days": int(cluster_days),
            "statistic": statistic,
            "status": status,
        })
    return pd.DataFrame(rows)


def _log_interpolate(target_dates, anchor_dates, anchor_values) -> np.ndarray:
    target = pd.DatetimeIndex(target_dates).asi8.astype(float)
    dates = pd.DatetimeIndex(anchor_dates).asi8.astype(float)
    values = np.asarray(anchor_values, dtype=float)
    return np.exp(np.interp(target, dates, np.log(values)))


def _candidate_prediction(
    model_id: str,
    region_dates: list[pd.Timestamp],
    region_values: np.ndarray,
    target_date: pd.Timestamp,
) -> float:
    values = np.asarray(region_values, dtype=float)
    dates = pd.DatetimeIndex(region_dates)
    if len(values) < 2:
        return float("nan")
    model_days = (dates - GENESIS).days.to_numpy(dtype=float)
    target_days = float((target_date - GENESIS).days)

    if model_id == "expanding_power_law":
        slope, intercept = np.polyfit(np.log(model_days), np.log(values), 1)
        return float(np.exp(intercept + slope * np.log(target_days)))

    if model_id == "local_exponent":
        local_exponents = np.log(values[1:] / values[:-1]) / np.log(model_days[1:] / model_days[:-1])
        recency_weights = np.arange(1, len(local_exponents) + 1, dtype=float)
        effective_exponent = float(np.average(local_exponents, weights=recency_weights))
        return float(values[-1] * (target_days / model_days[-1]) ** effective_exponent)

    if len(values) < 3:
        return float("nan")
    growth = values[1:] / values[:-1]
    if np.any(growth <= 1.0):
        return float("nan")
    if model_id == "recent_excess_decay":
        growth = growth[-2:]
    elif model_id != "all_cycle_excess_decay":
        raise KeyError(model_id)
    x = np.arange(len(growth), dtype=float)
    slope, intercept = np.polyfit(x, np.log(growth - 1.0), 1)
    next_growth = 1.0 + float(np.exp(intercept + slope * len(growth)))
    return float(values[-1] * next_growth)


def _walk_forward(bottoms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for target_index in range(2, len(bottoms)):
        training = bottoms.iloc[:target_index]
        target = bottoms.iloc[target_index]
        for model_id, label in CANDIDATE_LABELS.items():
            predicted = _candidate_prediction(
                model_id,
                training["region_date"].tolist(),
                training["region_price_usd"].to_numpy(dtype=float),
                pd.Timestamp(target["anchor_date"]),
            )
            if not np.isfinite(predicted) or predicted <= 0:
                continue
            actual = float(target["region_price_usd"])
            evidence_weight = 0.5 if target["status"] == "forming" else 1.0
            rows.append({
                "model_id": model_id,
                "model": label,
                "target_cycle": int(target["cycle"]),
                "target_date": pd.Timestamp(target["anchor_date"]),
                "target_status": target["status"],
                "training_bottoms": int(len(training)),
                "predicted_price_usd": predicted,
                "actual_region_price_usd": actual,
                "prediction_ratio": predicted / actual,
                "absolute_log_error": float(abs(np.log(predicted / actual))),
                "evidence_weight": evidence_weight,
            })
    walk = pd.DataFrame(rows)
    if walk.empty:
        return walk, pd.DataFrame()
    summaries: list[dict] = []
    for model_id, label in CANDIDATE_LABELS.items():
        subset = walk[walk["model_id"] == model_id]
        if subset.empty:
            continue
        weights = subset["evidence_weight"].to_numpy(dtype=float)
        errors = subset["absolute_log_error"].to_numpy(dtype=float)
        mean_error = float(np.average(errors, weights=weights))
        evidence = float(weights.sum())
        coverage = min(1.0, evidence / 2.0)
        raw_weight = float(coverage * np.exp(-mean_error / VALIDATION_TEMPERATURE))
        summaries.append({
            "model_id": model_id,
            "model": label,
            "holdouts": int(len(subset)),
            "effective_holdouts": evidence,
            "mean_absolute_log_error": mean_error,
            "rms_log_error": float(np.sqrt(np.average(np.square(errors), weights=weights))),
            "approx_typical_pct_error": float(np.exp(mean_error) - 1.0),
            "raw_validation_weight": raw_weight,
        })
    validation = pd.DataFrame(summaries)
    if not validation.empty:
        total = float(validation["raw_validation_weight"].sum())
        validation["ensemble_weight"] = (
            validation["raw_validation_weight"] / total if total > 0 else 1.0 / len(validation)
        )
    return walk, validation


def _forecast_candidates(
    bottoms: pd.DataFrame,
    target_date: pd.Timestamp,
    validation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for model_id, label in CANDIDATE_LABELS.items():
        prediction = _candidate_prediction(
            model_id,
            bottoms["region_date"].tolist(),
            bottoms["region_price_usd"].to_numpy(dtype=float),
            target_date,
        )
        if np.isfinite(prediction) and prediction > 0:
            rows.append({
                "model_id": model_id,
                "model": label,
                "target_date": target_date,
                "predicted_bottom_usd": prediction,
            })
    forecasts = pd.DataFrame(rows)
    if forecasts.empty:
        return forecasts
    if validation is None or validation.empty:
        forecasts["ensemble_weight"] = 1.0 / len(forecasts)
    else:
        forecasts = forecasts.merge(
            validation[["model_id", "ensemble_weight", "holdouts", "approx_typical_pct_error"]],
            on="model_id",
            how="left",
        )
        forecasts["ensemble_weight"] = forecasts["ensemble_weight"].fillna(0.0)
        if forecasts["ensemble_weight"].sum() <= 0:
            forecasts["ensemble_weight"] = 1.0 / len(forecasts)
        else:
            forecasts["ensemble_weight"] /= forecasts["ensemble_weight"].sum()
    return forecasts


def _ensemble_value(forecasts: pd.DataFrame) -> float:
    valid = forecasts[
        np.isfinite(forecasts["predicted_bottom_usd"])
        & (forecasts["predicted_bottom_usd"] > 0)
        & (forecasts["ensemble_weight"] > 0)
    ]
    if valid.empty:
        return float("nan")
    weights = valid["ensemble_weight"].to_numpy(dtype=float)
    weights /= weights.sum()
    return float(np.exp(np.sum(weights * np.log(valid["predicted_bottom_usd"].to_numpy(dtype=float)))))


def _mature_cycle_decay(bottoms: pd.DataFrame, mature_start_cycle: int = 1) -> dict:
    """Project the next bottom from decaying mature-cycle growth above 1x.

    Cycle zero is Bitcoin's early micro-cap regime. Starting with the 2015
    bottom leaves the three later transitions that describe the maturing
    market structure: 2015→2018, 2018→2022, and 2022→forming 2026.
    """
    mature = bottoms[bottoms["cycle"] >= mature_start_cycle].sort_values("cycle")
    values = mature["region_price_usd"].to_numpy(dtype=float)
    if len(values) < 4:
        return {
            "available": False,
            "mature_start_cycle": mature_start_cycle,
            "mature_transitions": max(0, len(values) - 1),
        }
    growth = values[1:] / values[:-1]
    if np.any(~np.isfinite(growth)) or np.any(growth <= 1.0):
        return {
            "available": False,
            "mature_start_cycle": mature_start_cycle,
            "mature_transitions": int(len(growth)),
        }
    x = np.arange(len(growth), dtype=float)
    slope, intercept = np.polyfit(x, np.log(growth - 1.0), 1)
    next_growth = 1.0 + float(np.exp(intercept + slope * len(growth)))
    return {
        "available": True,
        "mature_start_cycle": mature_start_cycle,
        "mature_transitions": int(len(growth)),
        "growth_multiples": growth.tolist(),
        "log_excess_decay_slope": float(slope),
        "next_growth_multiple": next_growth,
        "predicted_bottom_usd": float(values[-1] * next_growth),
    }


def _window_progress(anchor_date: pd.Timestamp, reveal_date: pd.Timestamp, half_window: int) -> float:
    start = anchor_date - pd.Timedelta(days=half_window)
    return float(np.clip((reveal_date - start).days / (2.0 * half_window), 0.0, 1.0))


def _settling_calibration_from_detail(detail: pd.DataFrame) -> pd.DataFrame:
    """Build a regularized monotonic settling curve from historical replay rows."""
    if detail.empty:
        return pd.DataFrame({
            "window_fraction": [0.0, 1.0],
            "completed_cycles": [0, 0],
            "median_settling_progress": [0.0, 1.0],
            "mean_settling_progress": [0.0, 1.0],
            "within_10_percent_share": [np.nan, np.nan],
            "within_20_percent_share": [np.nan, np.nan],
            "median_partial_log_error": [np.nan, 0.0],
            "linear_time_weight": [0.0, 1.0],
            "raw_empirical_weight": [0.0, 1.0],
            "calibration_reliability": [0.0, 0.0],
            "empirical_evidence_weight": [0.0, 1.0],
        })

    calibration = detail.groupby("window_fraction", as_index=False).agg(
        completed_cycles=("target_cycle", "nunique"),
        median_settling_progress=("settling_progress", "median"),
        mean_settling_progress=("settling_progress", "mean"),
        within_10_percent_share=("within_10_percent", "mean"),
        within_20_percent_share=("within_20_percent", "mean"),
        median_partial_log_error=("absolute_log_error", "median"),
    )
    calibration = pd.concat([
        pd.DataFrame({
            "window_fraction": [0.0],
            "completed_cycles": [int(detail["target_cycle"].nunique())],
            "median_settling_progress": [0.0],
            "mean_settling_progress": [0.0],
            "within_10_percent_share": [0.0],
            "within_20_percent_share": [0.0],
            "median_partial_log_error": [np.nan],
        }),
        calibration,
    ], ignore_index=True).sort_values("window_fraction").reset_index(drop=True)
    calibration["linear_time_weight"] = calibration["window_fraction"]
    calibration["raw_empirical_weight"] = np.maximum.accumulate(
        calibration["median_settling_progress"].to_numpy(dtype=float)
    )
    calibration["calibration_reliability"] = (
        calibration["completed_cycles"]
        / (calibration["completed_cycles"] + SETTLING_LINEAR_PRIOR_CYCLES)
    )
    calibration["empirical_evidence_weight"] = (
        calibration["calibration_reliability"] * calibration["raw_empirical_weight"]
        + (1.0 - calibration["calibration_reliability"]) * calibration["linear_time_weight"]
    )
    calibration["empirical_evidence_weight"] = np.maximum.accumulate(
        calibration["empirical_evidence_weight"].to_numpy(dtype=float)
    )
    calibration.loc[calibration.index[0], "empirical_evidence_weight"] = 0.0
    calibration.loc[calibration.index[-1], "empirical_evidence_weight"] = 1.0
    return calibration


def _empirical_settling_calibration(
    data: pd.DataFrame,
    completed_bottoms: pd.DataFrame,
    half_window: int = BOTTOM_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure how quickly partial historical bottom regions approached their final value."""
    detail_rows: list[dict] = []
    for _, target in completed_bottoms.sort_values("cycle").iterrows():
        cycle = int(target["cycle"])
        anchor = pd.Timestamp(target["anchor_date"])
        start = anchor - pd.Timedelta(days=half_window)
        final_bottom = float(target["region_price_usd"])
        partial_rows: list[dict] = []
        for fraction in SETTLING_GRID:
            reveal_date = start + pd.Timedelta(days=round(fraction * 2 * half_window))
            partial = _extract_regions(
                data[data["date"] <= reveal_date],
                [BOTTOM_TURNING_REGIONS[cycle]],
                half_window,
                "bottom",
            )
            if partial.empty:
                continue
            partial_bottom = float(partial["region_price_usd"].iloc[0])
            partial_rows.append({
                "target_cycle": cycle,
                "target_anchor": anchor,
                "window_fraction": float(fraction),
                "reveal_date": reveal_date,
                "partial_bottom_usd": partial_bottom,
                "final_bottom_usd": final_bottom,
                "absolute_log_error": float(abs(np.log(partial_bottom / final_bottom))),
            })
        if not partial_rows:
            continue
        initial_error = float(partial_rows[0]["absolute_log_error"])
        for row in partial_rows:
            error = float(row["absolute_log_error"])
            if initial_error <= 1e-12:
                progress = 1.0
            else:
                progress = float(np.clip(1.0 - error / initial_error, 0.0, 1.0))
            row["initial_absolute_log_error"] = initial_error
            row["settling_progress"] = progress
            row["within_10_percent"] = bool(error <= np.log(1.10))
            row["within_20_percent"] = bool(error <= np.log(1.20))
            detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    return detail, _settling_calibration_from_detail(detail)


def _settling_leave_one_out(detail: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the settling curve after omitting each completed cycle in turn."""
    if detail.empty or detail["target_cycle"].nunique() < 2:
        return pd.DataFrame()
    variants: list[pd.DataFrame] = []
    anchors = detail.groupby("target_cycle")["target_anchor"].first()
    for omitted_cycle in sorted(detail["target_cycle"].unique()):
        subset = detail[detail["target_cycle"] != omitted_cycle].copy()
        calibration = _settling_calibration_from_detail(subset)
        calibration.insert(0, "omitted_cycle", int(omitted_cycle))
        calibration.insert(1, "omitted_anchor", pd.Timestamp(anchors.loc[omitted_cycle]))
        variants.append(calibration)
    return pd.concat(variants, ignore_index=True)


def _forming_evidence_weight(
    anchor_date: pd.Timestamp,
    reveal_date: pd.Timestamp,
    half_window: int,
    calibration: pd.DataFrame | None = None,
) -> float:
    progress = _window_progress(anchor_date, reveal_date, half_window)
    if calibration is None or calibration.empty:
        return progress
    return float(np.interp(
        progress,
        calibration["window_fraction"].to_numpy(dtype=float),
        calibration["empirical_evidence_weight"].to_numpy(dtype=float),
    ))


def _dynamic_bottom_estimate(
    prior_bottoms: pd.DataFrame,
    target_row: pd.Series,
    reveal_date: pd.Timestamp,
    validation: pd.DataFrame | None = None,
    half_window: int = BOTTOM_WINDOW_DAYS,
    settling_calibration: pd.DataFrame | None = None,
) -> dict:
    anchor = pd.Timestamp(target_row["anchor_date"])
    forecasts = _forecast_candidates(prior_bottoms, anchor, validation)
    forecast = _ensemble_value(forecasts)
    observed = float(target_row["region_price_usd"])
    linear_progress = _window_progress(anchor, reveal_date, half_window)
    evidence = _forming_evidence_weight(anchor, reveal_date, half_window, settling_calibration)
    dynamic = float(np.exp((1.0 - evidence) * np.log(forecast) + evidence * np.log(observed)))
    return {
        "forecast_bottom_usd": forecast,
        "observed_forming_bottom_usd": observed,
        "forming_evidence_weight": evidence,
        "linear_window_progress": linear_progress,
        "dynamic_bottom_usd": dynamic,
        "forecast_low_usd": float(forecasts["predicted_bottom_usd"].min()),
        "forecast_high_usd": float(forecasts["predicted_bottom_usd"].max()),
    }


def _settling_cycle_dependence(
    leave_one_out: pd.DataFrame,
    observed_bottoms: pd.DataFrame,
    target_row: pd.Series,
    current_dynamic: dict,
    latest: pd.Timestamp,
    current_anchor: pd.Timestamp,
    fair_multiplier: float,
) -> pd.DataFrame:
    """Propagate leave-one-cycle-out settling curves into current and next-bottom estimates."""
    if leave_one_out.empty:
        return pd.DataFrame()
    previous = observed_bottoms.iloc[-2]
    forecast = float(current_dynamic["forecast_bottom_usd"])
    observed = float(target_row["region_price_usd"])
    progress = float(current_dynamic["linear_window_progress"])
    rows: list[dict] = []
    for omitted_cycle, calibration in leave_one_out.groupby("omitted_cycle"):
        calibration = calibration.sort_values("window_fraction")
        evidence = float(np.interp(
            progress,
            calibration["window_fraction"].to_numpy(dtype=float),
            calibration["empirical_evidence_weight"].to_numpy(dtype=float),
        ))
        dynamic_bottom = float(np.exp(
            (1.0 - evidence) * np.log(forecast) + evidence * np.log(observed)
        ))
        current_foundation = _foundation_at(
            latest, previous["region_date"], previous["region_price_usd"],
            current_anchor, dynamic_bottom,
        )
        variant_bottoms = observed_bottoms.copy()
        variant_bottoms.loc[variant_bottoms.index[-1], "region_price_usd"] = dynamic_bottom
        variant_bottoms.loc[variant_bottoms.index[-1], "region_date"] = current_anchor
        variant_bottoms = _add_bottom_growth(variant_bottoms)
        mature_variant = _mature_cycle_decay(variant_bottoms)
        rows.append({
            "omitted_cycle": int(omitted_cycle),
            "omitted_anchor": pd.Timestamp(calibration["omitted_anchor"].iloc[0]),
            "remaining_cycles": int(calibration["completed_cycles"].max()),
            "linear_window_progress": progress,
            "forming_evidence_weight": evidence,
            "full_calibration_evidence_weight": float(current_dynamic["forming_evidence_weight"]),
            "evidence_weight_change": evidence - float(current_dynamic["forming_evidence_weight"]),
            "dynamic_current_bottom_usd": dynamic_bottom,
            "current_foundation_usd": current_foundation,
            "dynamic_fair_value_usd": current_foundation * fair_multiplier,
            "mature_cycle_next_multiple": (
                float(mature_variant["next_growth_multiple"]) if mature_variant.get("available") else np.nan
            ),
            "mature_cycle_next_bottom_usd": (
                float(mature_variant["predicted_bottom_usd"]) if mature_variant.get("available") else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values("omitted_cycle").reset_index(drop=True)


def _add_bottom_growth(bottoms: pd.DataFrame) -> pd.DataFrame:
    result = bottoms.copy().sort_values("region_date").reset_index(drop=True)
    result["bottom_to_bottom_multiple"] = result["region_price_usd"].pct_change() + 1.0
    result["bottom_to_bottom_cagr"] = np.nan
    for index in range(1, len(result)):
        years = (result.loc[index, "region_date"] - result.loc[index - 1, "region_date"]).days / 365.2425
        result.loc[index, "bottom_to_bottom_cagr"] = (
            (result.loc[index, "region_price_usd"] / result.loc[index - 1, "region_price_usd"]) ** (1.0 / years) - 1.0
        )
    return result


def _foundation_at(date: pd.Timestamp, left_date, left_value, right_date, right_value) -> float:
    span = max((pd.Timestamp(right_date) - pd.Timestamp(left_date)).total_seconds(), 1.0)
    progress = float(np.clip((pd.Timestamp(date) - pd.Timestamp(left_date)).total_seconds() / span, 0.0, 1.0))
    return float(np.exp(np.log(left_value) + progress * (np.log(right_value) - np.log(left_value))))


def _build_fair_value_cycles(
    data: pd.DataFrame,
    bottoms: pd.DataFrame,
    peaks: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    cycle_count = min(len(peaks), len(bottoms) - 1)
    for cycle in range(cycle_count):
        left = bottoms.iloc[cycle]
        right = bottoms.iloc[cycle + 1]
        peak = peaks.iloc[cycle]
        cycle_end = min(pd.Timestamp(right["anchor_date"]), pd.Timestamp(data["date"].max()))
        segment = data[(data["date"] >= pd.Timestamp(left["region_date"])) & (data["date"] <= cycle_end)].copy()
        if segment.empty:
            continue
        right_date = pd.Timestamp(right["anchor_date"] if right["status"] == "forming" else right["region_date"])
        foundation = _log_interpolate(
            segment["date"],
            [pd.Timestamp(left["region_date"]), right_date],
            [float(left["region_price_usd"]), float(right["region_price_usd"])],
        )
        ratio = segment["price_usd"].to_numpy(dtype=float) / foundation
        peak_foundation = _foundation_at(
            pd.Timestamp(peak["region_date"]),
            pd.Timestamp(left["region_date"]),
            float(left["region_price_usd"]),
            right_date,
            float(right["region_price_usd"]),
        )
        peak_multiple = float(peak["region_price_usd"] / peak_foundation)
        method_values = {
            "peak_bottom_midpoint": float(np.sqrt(peak_multiple)),
            "cycle_log_median": float(np.exp(np.median(np.log(ratio)))),
            "cycle_log_mean": float(np.exp(np.mean(np.log(ratio)))),
            "central_50_midpoint": float(np.sqrt(np.quantile(ratio, 0.25) * np.quantile(ratio, 0.75))),
        }
        for method_id, multiple in method_values.items():
            rows.append({
                "cycle": cycle,
                "method_id": method_id,
                "method": FAIR_METHOD_LABELS[method_id],
                "start_date": pd.Timestamp(left["region_date"]),
                "end_date": cycle_end,
                "target_status": right["status"],
                "observations": int(len(segment)),
                "fair_multiple": multiple,
                "peak_foundation_multiple": peak_multiple,
            })
    return pd.DataFrame(rows)


def _fair_value_walk_forward(
    data: pd.DataFrame,
    bottoms: pd.DataFrame,
    fair_cycles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    cycles = sorted(fair_cycles["cycle"].unique())
    for target_cycle in cycles[1:]:
        target_status = str(fair_cycles.loc[fair_cycles["cycle"] == target_cycle, "target_status"].iloc[0])
        evidence_weight = 1.0 if target_status == "completed" else 0.0
        left = bottoms.iloc[target_cycle]
        right = bottoms.iloc[target_cycle + 1]
        right_date = pd.Timestamp(right["anchor_date"] if right["status"] == "forming" else right["region_date"])
        cycle_end = min(right_date, pd.Timestamp(data["date"].max()))
        segment = data[(data["date"] >= pd.Timestamp(left["region_date"])) & (data["date"] <= cycle_end)].copy()
        foundation = _log_interpolate(
            segment["date"],
            [pd.Timestamp(left["region_date"]), right_date],
            [float(left["region_price_usd"]), float(right["region_price_usd"])],
        )
        for method_id, label in FAIR_METHOD_LABELS.items():
            history = fair_cycles[
                (fair_cycles["cycle"] < target_cycle)
                & (fair_cycles["method_id"] == method_id)
            ].sort_values("cycle")
            if history.empty:
                continue
            predicted_multiple = float(history["fair_multiple"].iloc[-1])
            residual = np.log(segment["price_usd"].to_numpy(dtype=float) / (foundation * predicted_multiple))
            median_abs = float(np.median(np.abs(residual)))
            neutrality = float(abs(np.mean(residual >= 0.0) - 0.5))
            rows.append({
                "method_id": method_id,
                "method": label,
                "target_cycle": int(target_cycle),
                "target_status": target_status,
                "predicted_multiple": predicted_multiple,
                "realized_multiple": float(fair_cycles.loc[
                    (fair_cycles["cycle"] == target_cycle)
                    & (fair_cycles["method_id"] == method_id),
                    "fair_multiple",
                ].iloc[0]),
                "median_absolute_log_error": median_abs,
                "absolute_log_bias": float(abs(np.median(residual))),
                "time_above_fair_value": float(np.mean(residual >= 0.0)),
                "neutrality_error": neutrality,
                "combined_score": median_abs + neutrality,
                "evidence_weight": evidence_weight,
            })
    walk = pd.DataFrame(rows)
    summaries: list[dict] = []
    for method_id, label in FAIR_METHOD_LABELS.items():
        subset = walk[(walk["method_id"] == method_id) & (walk["evidence_weight"] > 0)]
        if subset.empty:
            continue
        weights = subset["evidence_weight"].to_numpy(dtype=float)
        score = float(np.average(subset["combined_score"], weights=weights))
        summaries.append({
            "method_id": method_id,
            "method": label,
            "completed_holdouts": int(len(subset)),
            "mean_combined_score": score,
            "mean_median_absolute_log_error": float(np.average(subset["median_absolute_log_error"], weights=weights)),
            "mean_neutrality_error": float(np.average(subset["neutrality_error"], weights=weights)),
            "raw_validation_weight": float(np.exp(-score / 0.6)),
        })
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["ensemble_weight"] = summary["raw_validation_weight"] / summary["raw_validation_weight"].sum()
    return walk, summary


def _dynamic_settling_backtest(
    data: pd.DataFrame,
    observed_bottoms: pd.DataFrame,
    peaks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths: list[dict] = []
    summaries: list[dict] = []
    latest = pd.Timestamp(data["date"].max())
    for target_index in range(2, len(observed_bottoms)):
        target = observed_bottoms.iloc[target_index]
        prior = observed_bottoms.iloc[:target_index].copy()
        anchor = pd.Timestamp(target["anchor_date"])
        reference_date = anchor - pd.Timedelta(days=REFERENCE_DAYS_BEFORE_TURN)
        reveal_end = min(anchor + pd.Timedelta(days=BOTTOM_WINDOW_DAYS), latest)
        if reveal_end < reference_date:
            continue
        _, prior_validation = _walk_forward(prior)
        _, prior_settling_calibration = _empirical_settling_calibration(data, prior)
        reveal_dates = list(pd.date_range(reference_date, reveal_end, freq="30D"))
        if reveal_dates[-1] != reveal_end:
            reveal_dates.append(reveal_end)
        left = prior.iloc[-1]
        peak = peaks.iloc[target_index - 1]
        settled_bottom = float(target["region_price_usd"])
        settled_reference_foundation = _foundation_at(
            reference_date, left["region_date"], left["region_price_usd"], anchor, settled_bottom
        )
        settled_peak_foundation = _foundation_at(
            peak["region_date"], left["region_date"], left["region_price_usd"], anchor, settled_bottom
        )
        settled_multiplier = float(np.sqrt(float(peak["region_price_usd"]) / settled_peak_foundation))
        settled_reference_fair = settled_reference_foundation * settled_multiplier

        cycle_rows: list[dict] = []
        for reveal_date in reveal_dates:
            partial = data[data["date"] <= reveal_date]
            partial_target = _extract_regions(
                partial, [BOTTOM_TURNING_REGIONS[target_index]], BOTTOM_WINDOW_DAYS, "bottom"
            )
            forecasts = _forecast_candidates(prior, anchor, prior_validation)
            forecast = _ensemble_value(forecasts)
            if partial_target.empty:
                observed = np.nan
                evidence = 0.0
                linear_progress = _window_progress(anchor, reveal_date, BOTTOM_WINDOW_DAYS)
                dynamic_bottom = forecast
            else:
                observed = float(partial_target["region_price_usd"].iloc[0])
                linear_progress = _window_progress(anchor, reveal_date, BOTTOM_WINDOW_DAYS)
                evidence = _forming_evidence_weight(
                    anchor, reveal_date, BOTTOM_WINDOW_DAYS, prior_settling_calibration
                )
                dynamic_bottom = float(np.exp((1.0 - evidence) * np.log(forecast) + evidence * np.log(observed)))
            reference_foundation = _foundation_at(
                reference_date, left["region_date"], left["region_price_usd"], anchor, dynamic_bottom
            )
            peak_foundation = _foundation_at(
                peak["region_date"], left["region_date"], left["region_price_usd"], anchor, dynamic_bottom
            )
            fair_multiplier = float(np.sqrt(float(peak["region_price_usd"]) / peak_foundation))
            reference_fair = reference_foundation * fair_multiplier
            row = {
                "target_cycle": int(target["cycle"]),
                "target_status": target["status"],
                "target_anchor": anchor,
                "reference_date": reference_date,
                "reveal_date": reveal_date,
                "forecast_bottom_usd": forecast,
                "forming_bottom_usd": observed,
                "linear_window_progress": linear_progress,
                "forming_evidence_weight": evidence,
                "dynamic_bottom_usd": dynamic_bottom,
                "settled_bottom_usd": settled_bottom if target["status"] == "completed" else np.nan,
                "dynamic_reference_fair_value_usd": reference_fair,
                "settled_reference_fair_value_usd": settled_reference_fair if target["status"] == "completed" else np.nan,
                "bottom_error_vs_settled_log": float(abs(np.log(dynamic_bottom / settled_bottom))) if target["status"] == "completed" else np.nan,
                "fair_value_error_vs_settled_log": float(abs(np.log(reference_fair / settled_reference_fair))) if target["status"] == "completed" else np.nan,
            }
            cycle_rows.append(row)
            paths.append(row)
        cycle_path = pd.DataFrame(cycle_rows)
        first = cycle_path.iloc[0]
        last = cycle_path.iloc[-1]
        summaries.append({
            "target_cycle": int(target["cycle"]),
            "target_status": target["status"],
            "target_anchor": anchor,
            "reference_date": reference_date,
            "first_linear_window_progress": float(first["linear_window_progress"]),
            "first_empirical_evidence_weight": float(first["forming_evidence_weight"]),
            "latest_linear_window_progress": float(last["linear_window_progress"]),
            "latest_empirical_evidence_weight": float(last["forming_evidence_weight"]),
            "first_dynamic_bottom_usd": float(first["dynamic_bottom_usd"]),
            "latest_dynamic_bottom_usd": float(last["dynamic_bottom_usd"]),
            "settled_bottom_usd": settled_bottom if target["status"] == "completed" else np.nan,
            "first_reference_fair_value_usd": float(first["dynamic_reference_fair_value_usd"]),
            "latest_reference_fair_value_usd": float(last["dynamic_reference_fair_value_usd"]),
            "settled_reference_fair_value_usd": settled_reference_fair if target["status"] == "completed" else np.nan,
            "first_bottom_error_pct": float(abs(first["dynamic_bottom_usd"] / settled_bottom - 1.0)) if target["status"] == "completed" else np.nan,
            "latest_bottom_error_pct": float(abs(last["dynamic_bottom_usd"] / settled_bottom - 1.0)) if target["status"] == "completed" else np.nan,
            "first_fair_value_error_pct": float(abs(first["dynamic_reference_fair_value_usd"] / settled_reference_fair - 1.0)) if target["status"] == "completed" else np.nan,
            "latest_fair_value_error_pct": float(abs(last["dynamic_reference_fair_value_usd"] / settled_reference_fair - 1.0)) if target["status"] == "completed" else np.nan,
        })
    return pd.DataFrame(paths), pd.DataFrame(summaries)


def _bottom_sensitivity(
    data: pd.DataFrame,
    fair_multiplier: float,
    settling_calibration: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    latest = pd.Timestamp(data["date"].max())
    for half_window in (60, 90, 120, 180):
        for cluster_days in (3, 7, 14, 30):
            for statistic in ("median", "geometric_mean"):
                regions = _extract_regions(
                    data, BOTTOM_TURNING_REGIONS, half_window, "bottom",
                    cluster_days=cluster_days, statistic=statistic,
                )
                current = regions[regions["cycle"] == len(BOTTOM_TURNING_REGIONS) - 1]
                completed = regions[regions["cycle"] < len(BOTTOM_TURNING_REGIONS) - 1]
                if current.empty or len(completed) < 3:
                    rows.append({
                        "half_window_days": half_window,
                        "cluster_days": cluster_days,
                        "statistic": statistic,
                        "available": False,
                    })
                    continue
                _, prior_validation = _walk_forward(completed)
                target = current.iloc[0]
                dynamic = _dynamic_bottom_estimate(
                    completed, target, latest, prior_validation, half_window=half_window,
                    settling_calibration=settling_calibration,
                )
                adjusted = pd.concat([completed, current], ignore_index=True)
                adjusted.loc[adjusted.index[-1], "region_price_usd"] = dynamic["dynamic_bottom_usd"]
                adjusted.loc[adjusted.index[-1], "region_date"] = pd.Timestamp(target["anchor_date"])
                mature_core = _mature_cycle_decay(adjusted)
                previous = completed.iloc[-1]
                current_foundation = _foundation_at(
                    latest, previous["region_date"], previous["region_price_usd"],
                    target["anchor_date"], dynamic["dynamic_bottom_usd"],
                )
                rows.append({
                    "half_window_days": half_window,
                    "cluster_days": cluster_days,
                    "statistic": statistic,
                    "available": True,
                    "forming_region_usd": float(target["region_price_usd"]),
                    "forming_evidence_weight": dynamic["forming_evidence_weight"],
                    "linear_window_progress": dynamic["linear_window_progress"],
                    "dynamic_current_bottom_usd": dynamic["dynamic_bottom_usd"],
                    "current_foundation_usd": current_foundation,
                    "fair_value_if_multiplier_held_usd": current_foundation * fair_multiplier,
                    "mature_cycle_next_multiple": mature_core.get("next_growth_multiple", np.nan),
                    "mature_cycle_next_bottom_usd": mature_core.get("predicted_bottom_usd", np.nan),
                })
    return pd.DataFrame(rows)


def _cycle_balanced_all_price_backbone(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    complete_cycles = [
        (pd.Timestamp("2011-02-14"), pd.Timestamp("2015-01-14")),
        (pd.Timestamp("2015-01-14"), pd.Timestamp("2018-12-15")),
        (pd.Timestamp("2018-12-15"), pd.Timestamp("2022-11-07")),
    ]
    latest = pd.Timestamp(data["date"].max())
    segment = data[data["date"] >= complete_cycles[0][0]].copy().reset_index(drop=True)
    x = np.log((segment["date"] - GENESIS).dt.days.to_numpy(dtype=float))
    y = np.log(segment["price_usd"].to_numpy(dtype=float))
    weights = np.zeros(len(segment), dtype=float)
    for start, end in complete_cycles:
        mask = (segment["date"] >= start) & (segment["date"] < end)
        if mask.any():
            weights[mask.to_numpy()] = 1.0 / int(mask.sum())
    expected_days = int(round(float(np.median([(end - start).days for start, end in complete_cycles]))))
    open_start = complete_cycles[-1][1]
    open_mask = segment["date"] >= open_start
    progress = float(np.clip((latest - open_start).days / expected_days, 0.0, 1.0))
    if open_mask.any():
        weights[open_mask.to_numpy()] = progress / int(open_mask.sum())
    valid = weights > 0
    slope, intercept = np.polyfit(x[valid], y[valid], 1, w=np.sqrt(weights[valid]))
    curve = data[["date"]].copy()
    curve_days = (curve["date"] - GENESIS).dt.days.to_numpy(dtype=float)
    curve["all_price_backbone_usd"] = np.exp(intercept + slope * np.log(curve_days))
    return curve, {
        "exponent": float(slope),
        "latest_backbone_usd": float(curve["all_price_backbone_usd"].iloc[-1]),
        "open_cycle_evidence_weight": progress,
    }


def fit_bottom_anchored_model(prices: pd.DataFrame) -> BottomAnchoredModelResult:
    data = _normalise(prices)
    if data.empty:
        raise ValueError("No valid Bitcoin prices are available.")
    latest = pd.Timestamp(data["date"].max())

    observed_bottoms = _add_bottom_growth(_extract_regions(
        data, BOTTOM_TURNING_REGIONS, BOTTOM_WINDOW_DAYS, "bottom"
    ))
    peaks = _extract_regions(data, PEAK_TURNING_REGIONS, PEAK_WINDOW_DAYS, "peak")
    if len(observed_bottoms) < 5 or len(peaks) < 4:
        raise ValueError("The complete bottom/peak research catalog is not observable yet.")

    mature_anchor_dates = [pd.Timestamp(item["anchor_date"]) for item in BOTTOM_TURNING_REGIONS[1:]]
    mature_cycle_days = np.diff(pd.DatetimeIndex(mature_anchor_dates).asi8) / (24 * 60 * 60 * 1e9)
    expected_cycle_days = int(round(float(np.median(mature_cycle_days))))
    current_anchor = pd.Timestamp(BOTTOM_TURNING_REGIONS[-1]["anchor_date"])
    next_bottom_anchor = current_anchor + pd.Timedelta(days=expected_cycle_days)
    mature_bull_days = [
        (pd.Timestamp(PEAK_TURNING_REGIONS[index]["anchor_date"]) - pd.Timestamp(BOTTOM_TURNING_REGIONS[index]["anchor_date"])).days
        for index in range(1, len(PEAK_TURNING_REGIONS))
    ]
    expected_bull_days = int(round(float(np.median(mature_bull_days))))
    next_peak_anchor = current_anchor + pd.Timedelta(days=expected_bull_days)

    completed_bottoms = observed_bottoms[observed_bottoms["status"] == "completed"].copy()
    settling_calibration_detail, settling_calibration = _empirical_settling_calibration(
        data, completed_bottoms
    )
    settling_leave_one_out = _settling_leave_one_out(settling_calibration_detail)
    _, prior_validation = _walk_forward(completed_bottoms)
    current_observed = observed_bottoms.iloc[-1]
    forming_prior_forecasts = _forecast_candidates(completed_bottoms, current_anchor, prior_validation)
    forming_prior_forecasts["forecast_role"] = "pre-observation prior for the forming 2026 bottom"
    current_dynamic = _dynamic_bottom_estimate(
        completed_bottoms, current_observed, latest, prior_validation,
        settling_calibration=settling_calibration,
    )

    settled_bottoms = observed_bottoms.copy()
    settled_bottoms.loc[settled_bottoms.index[-1], "region_price_usd"] = current_dynamic["dynamic_bottom_usd"]
    settled_bottoms.loc[settled_bottoms.index[-1], "region_date"] = current_anchor
    settled_bottoms = _add_bottom_growth(settled_bottoms)

    fair_cycles = _build_fair_value_cycles(data, settled_bottoms, peaks)
    fair_walk, fair_methods = _fair_value_walk_forward(data, settled_bottoms, fair_cycles)
    current_cycle_id = int(fair_cycles["cycle"].max())
    current_methods = fair_cycles[fair_cycles["cycle"] == current_cycle_id].merge(
        fair_methods[["method_id", "ensemble_weight", "completed_holdouts", "mean_combined_score"]],
        on="method_id", how="left",
    )
    current_methods["ensemble_weight"] = current_methods["ensemble_weight"].fillna(1.0 / len(current_methods))
    current_methods["ensemble_weight"] /= current_methods["ensemble_weight"].sum()
    fair_multiplier = float(np.exp(np.sum(
        current_methods["ensemble_weight"].to_numpy(dtype=float)
        * np.log(current_methods["fair_multiple"].to_numpy(dtype=float))
    )))
    fair_multiplier_low = float(current_methods["fair_multiple"].min())
    fair_multiplier_high = float(current_methods["fair_multiple"].max())

    bottom_walk, bottom_validation = _walk_forward(settled_bottoms)
    mature_core = _mature_cycle_decay(settled_bottoms)
    if not mature_core["available"]:
        raise ValueError("The mature-cycle bottom projection is not observable yet.")

    previous = settled_bottoms.iloc[-2]
    current_foundation = _foundation_at(
        latest, previous["region_date"], previous["region_price_usd"],
        current_anchor, current_dynamic["dynamic_bottom_usd"],
    )
    current_fair = current_foundation * fair_multiplier
    current_fair_low = current_foundation * fair_multiplier_low
    current_fair_high = current_foundation * fair_multiplier_high

    settling_cycle_dependence = _settling_cycle_dependence(
        settling_leave_one_out, observed_bottoms, current_observed, current_dynamic,
        latest, current_anchor, fair_multiplier,
    )
    if settling_cycle_dependence.empty:
        raise ValueError("Leave-one-cycle-out settling calibration is not observable yet.")
    settling_next_bottoms = settling_cycle_dependence["mature_cycle_next_bottom_usd"].dropna()
    if settling_next_bottoms.empty:
        raise ValueError("Leave-one-cycle-out next-bottom propagation is not observable yet.")

    sensitivity = _bottom_sensitivity(data, fair_multiplier, settling_calibration)
    available_sensitivity = sensitivity[sensitivity["available"]].copy()
    mature_sensitivity = available_sensitivity["mature_cycle_next_bottom_usd"].dropna()
    if mature_sensitivity.empty:
        raise ValueError("No mature-cycle bottom sensitivity variants are available.")
    next_bottom_core = float(mature_core["predicted_bottom_usd"])
    next_bottom_core_low = min(float(mature_sensitivity.quantile(0.10)), next_bottom_core)
    next_bottom_core_high = max(float(mature_sensitivity.quantile(0.90)), next_bottom_core)
    observed_growth_path = " → ".join(f"{multiple:.3f}×" for multiple in mature_core["growth_multiples"])
    mature_forecast = pd.DataFrame([{
        "model_id": "mature_cycle_excess_decay",
        "model": "Mature-cycle excess-growth decay",
        "target_date": next_bottom_anchor,
        "mature_start_cycle": mature_core["mature_start_cycle"],
        "mature_transitions": mature_core["mature_transitions"],
        "includes_forming_current_bottom": True,
        "observed_growth_path": observed_growth_path,
        "next_growth_multiple": mature_core["next_growth_multiple"],
        "predicted_bottom_usd": next_bottom_core,
        "definition_range_low_usd": next_bottom_core_low,
        "definition_range_high_usd": next_bottom_core_high,
        "definition_stress_low_usd": float(mature_sensitivity.min()),
        "definition_stress_high_usd": float(mature_sensitivity.max()),
        "definition_variants": int(len(mature_sensitivity)),
        "range_definition": "10th–90th percentile across available bottom-region definitions",
        "validation_status": "LOW_EVIDENCE — three mature transitions; current endpoint is forming",
    }])
    dynamic_paths, dynamic_summary = _dynamic_settling_backtest(data, observed_bottoms, peaks)

    curve_dates = pd.date_range(pd.Timestamp(settled_bottoms["region_date"].min()), next_bottom_anchor, freq="D")
    bottom_anchor_dates = settled_bottoms["region_date"].tolist() + [next_bottom_anchor]
    bottom_anchor_values = settled_bottoms["region_price_usd"].tolist() + [next_bottom_core]
    bottom_foundation = _log_interpolate(curve_dates, bottom_anchor_dates, bottom_anchor_values)

    cycle_multiplier_rows = fair_cycles.merge(
        fair_methods[["method_id", "ensemble_weight"]], on="method_id", how="left"
    )
    cycle_multipliers = []
    for cycle, group in cycle_multiplier_rows.groupby("cycle"):
        weights = group["ensemble_weight"].fillna(1.0 / len(group)).to_numpy(dtype=float)
        weights /= weights.sum()
        cycle_multipliers.append({
            "date": pd.Timestamp(peaks.iloc[int(cycle)]["region_date"]),
            "multiple": float(np.exp(np.sum(weights * np.log(group["fair_multiple"].to_numpy(dtype=float))))),
        })
    multiplier_dates = [row["date"] for row in cycle_multipliers] + [next_peak_anchor]
    multiplier_values = [row["multiple"] for row in cycle_multipliers] + [fair_multiplier]
    fair_multiplier_curve = _log_interpolate(curve_dates, multiplier_dates, multiplier_values)
    dynamic_fair_value = bottom_foundation * fair_multiplier_curve
    dynamic_fair_value[curve_dates > latest] = np.nan
    curve = pd.DataFrame({
        "date": curve_dates,
        "row_type": np.where(curve_dates <= latest, "historical", "projected"),
        "bottom_foundation_usd": bottom_foundation,
        "fair_value_multiple": fair_multiplier_curve,
        "dynamic_fair_value_usd": dynamic_fair_value,
    })

    all_price_curve, all_price_summary = _cycle_balanced_all_price_backbone(data)
    latest_price = float(data["price_usd"].iloc[-1])
    selected_method = str(fair_methods.sort_values("ensemble_weight", ascending=False)["method"].iloc[0])
    summary = {
        "summary_schema": SUMMARY_SCHEMA,
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY",
        "confidence": "LOW — only a few independent Bitcoin cycles exist",
        "latest_date": latest,
        "latest_price_usd": latest_price,
        "forming_bottom_region_usd": float(current_observed["region_price_usd"]),
        "forming_bottom_extreme_usd": float(current_observed["extreme_price_usd"]),
        "linear_window_progress": current_dynamic["linear_window_progress"],
        "forming_evidence_weight": current_dynamic["forming_evidence_weight"],
        "settling_calibration_cycles": (
            int(settling_calibration_detail["target_cycle"].nunique())
            if not settling_calibration_detail.empty else 0
        ),
        "settling_calibration_method": "median normalized log-error convergence across completed bottom regions",
        "settling_cycle_dependence_method": "full range after omitting each completed bottom cycle in turn",
        "settling_cycle_dependence_variants": int(len(settling_cycle_dependence)),
        "forming_evidence_weight_cycle_low": float(settling_cycle_dependence["forming_evidence_weight"].min()),
        "forming_evidence_weight_cycle_high": float(settling_cycle_dependence["forming_evidence_weight"].max()),
        "pre_observation_bottom_forecast_usd": current_dynamic["forecast_bottom_usd"],
        "dynamic_settled_bottom_estimate_usd": current_dynamic["dynamic_bottom_usd"],
        "dynamic_settled_bottom_cycle_low_usd": float(settling_cycle_dependence["dynamic_current_bottom_usd"].min()),
        "dynamic_settled_bottom_cycle_high_usd": float(settling_cycle_dependence["dynamic_current_bottom_usd"].max()),
        "dynamic_bottom_forecast_low_usd": current_dynamic["forecast_low_usd"],
        "dynamic_bottom_forecast_high_usd": current_dynamic["forecast_high_usd"],
        "current_bottom_foundation_usd": current_foundation,
        "dynamic_fair_value_usd": current_fair,
        "dynamic_fair_value_cycle_low_usd": float(settling_cycle_dependence["dynamic_fair_value_usd"].min()),
        "dynamic_fair_value_cycle_high_usd": float(settling_cycle_dependence["dynamic_fair_value_usd"].max()),
        "dynamic_fair_value_low_usd": current_fair_low,
        "dynamic_fair_value_high_usd": current_fair_high,
        "price_to_dynamic_fair_value": latest_price / current_fair,
        "fair_value_multiple": fair_multiplier,
        "fair_value_method_leader": selected_method,
        "fair_value_completed_holdouts": int(fair_methods["completed_holdouts"].max()),
        "all_price_backbone_usd": all_price_summary["latest_backbone_usd"],
        "all_price_backbone_exponent": all_price_summary["exponent"],
        "all_price_open_cycle_weight": all_price_summary["open_cycle_evidence_weight"],
        "expected_cycle_days": expected_cycle_days,
        "expected_bull_days": expected_bull_days,
        "next_peak_anchor": next_peak_anchor,
        "next_bottom_anchor": next_bottom_anchor,
        "next_bottom_core_usd": next_bottom_core,
        "next_bottom_cycle_low_usd": float(settling_next_bottoms.min()),
        "next_bottom_cycle_high_usd": float(settling_next_bottoms.max()),
        "next_bottom_core_multiple": mature_core["next_growth_multiple"],
        "mature_observed_growth_path": observed_growth_path,
        "next_bottom_core_low_usd": next_bottom_core_low,
        "next_bottom_core_high_usd": next_bottom_core_high,
        "next_bottom_core_stress_low_usd": float(mature_sensitivity.min()),
        "next_bottom_core_stress_high_usd": float(mature_sensitivity.max()),
        "bottom_sensitivity_variants": int(len(available_sensitivity)),
        "bottom_sensitivity_dynamic_low_usd": float(available_sensitivity["dynamic_current_bottom_usd"].min()),
        "bottom_sensitivity_dynamic_high_usd": float(available_sensitivity["dynamic_current_bottom_usd"].max()),
        "bottom_sensitivity_fair_low_usd": float(available_sensitivity["fair_value_if_multiplier_held_usd"].min()),
        "bottom_sensitivity_fair_high_usd": float(available_sensitivity["fair_value_if_multiplier_held_usd"].max()),
        "candidate_weight_method": "internal walk-forward error weighting",
        "core_forecast_method": "mature-cycle excess-growth decay beginning with the 2015 bottom; 10th–90th percentile definition range",
        "dynamic_settling_method": "pre-observation internal forecast blended with forming bottom evidence using an empirical historical settling curve",
        "fair_value_method": "validation-weighted ensemble of four internally derived cycle-neutral definitions",
        "future_fair_value_status": "deferred until peak-compression behavior is modeled",
    }
    return BottomAnchoredModelResult(
        summary=summary,
        bottom_regions=observed_bottoms,
        peak_regions=peaks,
        curve=curve,
        mature_cycle_forecast=mature_forecast,
        forming_prior_forecasts=forming_prior_forecasts.sort_values("predicted_bottom_usd").reset_index(drop=True),
        validation_summary=bottom_validation.sort_values("ensemble_weight", ascending=False).reset_index(drop=True),
        walk_forward=bottom_walk.sort_values(["target_date", "model"]).reset_index(drop=True),
        bottom_sensitivity=sensitivity,
        fair_value_cycles=fair_cycles,
        fair_value_validation=fair_walk,
        fair_value_methods=current_methods.sort_values("ensemble_weight", ascending=False).reset_index(drop=True),
        settling_calibration=settling_calibration,
        settling_calibration_detail=settling_calibration_detail,
        settling_leave_one_out=settling_leave_one_out,
        settling_cycle_dependence=settling_cycle_dependence,
        dynamic_settling=dynamic_paths,
        dynamic_settling_summary=dynamic_summary,
        all_price_curve=all_price_curve,
    )
