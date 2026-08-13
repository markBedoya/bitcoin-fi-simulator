from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MODEL_VERSION = "bottom-anchored-fair-value-research-v0.1.0"
GENESIS = pd.Timestamp("2009-01-03")
BOTTOM_WINDOW_DAYS = 120
PEAK_WINDOW_DAYS = 90
REGION_CLUSTER_DAYS = 7
VALIDATION_TEMPERATURE = 0.35

# Dates locate broad market-regime turning regions. The observed lowest/highest
# daily close may occur before or after the date. The 2026 date remains forming.
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
    "learned_power_law": "Learned fixed bottom power law",
    "published_bottom_formula": "Published 5.82 × 0.42 bottom formula",
    "all_cycle_excess_decay": "All-cycle excess-growth decay",
    "recent_excess_decay": "Recent excess-growth decay",
}


@dataclass
class BottomAnchoredModelResult:
    summary: dict
    bottom_regions: pd.DataFrame
    peak_regions: pd.DataFrame
    curve: pd.DataFrame
    candidate_forecasts: pd.DataFrame
    validation_summary: pd.DataFrame
    walk_forward: pd.DataFrame
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


def _extract_regions(
    data: pd.DataFrame,
    specifications: list[dict],
    half_window_days: int,
    direction: str,
) -> pd.DataFrame:
    latest = pd.Timestamp(data["date"].max())
    rows: list[dict] = []
    for cycle, specification in enumerate(specifications):
        anchor = pd.Timestamp(specification["anchor_date"])
        start = anchor - pd.Timedelta(days=half_window_days)
        end = min(anchor + pd.Timedelta(days=half_window_days), latest)
        window = data[(data["date"] >= start) & (data["date"] <= end)].copy()
        if len(window) < REGION_CLUSTER_DAYS:
            continue
        if direction == "bottom":
            cluster = window.nsmallest(REGION_CLUSTER_DAYS, "price_usd").copy()
            extreme_row = window.loc[window["price_usd"].idxmin()]
        else:
            cluster = window.nlargest(REGION_CLUSTER_DAYS, "price_usd").copy()
            extreme_row = window.loc[window["price_usd"].idxmax()]

        status = "completed" if latest >= anchor + pd.Timedelta(days=half_window_days) else "forming"
        rows.append({
            "cycle": cycle,
            "label": specification["label"],
            "anchor_date": anchor,
            "region_date": _median_timestamp(cluster["date"]),
            "region_price_usd": float(cluster["price_usd"].median()),
            "cluster_low_usd": float(cluster["price_usd"].min()),
            "cluster_high_usd": float(cluster["price_usd"].max()),
            "extreme_date": pd.Timestamp(extreme_row["date"]),
            "extreme_price_usd": float(extreme_row["price_usd"]),
            "window_start": start,
            "window_end": end,
            "observations": int(len(window)),
            "status": status,
        })
    return pd.DataFrame(rows)


def _log_interpolate(
    target_dates: pd.DatetimeIndex | pd.Series,
    anchor_dates: pd.Series | list,
    anchor_values: pd.Series | list,
) -> np.ndarray:
    target = pd.DatetimeIndex(target_dates).asi8.astype(float)
    dates = pd.DatetimeIndex(anchor_dates).asi8.astype(float)
    values = np.asarray(anchor_values, dtype=float)
    return np.exp(np.interp(target, dates, np.log(values)))


def _published_lines(dates: pd.DatetimeIndex | pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_days = (pd.DatetimeIndex(dates) - GENESIS).days.to_numpy(dtype=float)
    raw = 1.0117e-17 * np.power(model_days, 5.82)
    return raw, 0.71 * raw, 0.42 * raw


def _candidate_prediction(
    model_id: str,
    region_dates: list[pd.Timestamp],
    region_values: np.ndarray,
    target_date: pd.Timestamp,
) -> float:
    values = np.asarray(region_values, dtype=float)
    dates = pd.DatetimeIndex(region_dates)
    if model_id == "published_bottom_formula":
        return float(_published_lines(pd.DatetimeIndex([target_date]))[2][0])
    if model_id == "learned_power_law":
        if len(values) < 2:
            return float("nan")
        x = np.log((dates - GENESIS).days.to_numpy(dtype=float))
        slope, intercept = np.polyfit(x, np.log(values), 1)
        target_days = float((target_date - GENESIS).days)
        return float(np.exp(intercept + slope * np.log(target_days)))

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
    model_ids = list(CANDIDATE_LABELS)
    for target_index in range(2, len(bottoms)):
        training = bottoms.iloc[:target_index]
        target = bottoms.iloc[target_index]
        for model_id in model_ids:
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
                "model": CANDIDATE_LABELS[model_id],
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
    summaries: list[dict] = []
    for model_id, label in CANDIDATE_LABELS.items():
        subset = walk[walk["model_id"] == model_id]
        if subset.empty:
            continue
        weights = subset["evidence_weight"].to_numpy(dtype=float)
        errors = subset["absolute_log_error"].to_numpy(dtype=float)
        mean_error = float(np.average(errors, weights=weights))
        rms_error = float(np.sqrt(np.average(np.square(errors), weights=weights)))
        evidence = float(weights.sum())
        coverage = min(1.0, evidence / 2.0)
        raw_weight = float(coverage * np.exp(-mean_error / VALIDATION_TEMPERATURE))
        summaries.append({
            "model_id": model_id,
            "model": label,
            "holdouts": int(len(subset)),
            "effective_holdouts": evidence,
            "mean_absolute_log_error": mean_error,
            "rms_log_error": rms_error,
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
        "intercept": float(intercept),
        "latest_backbone_usd": float(curve["all_price_backbone_usd"].iloc[-1]),
        "open_cycle_evidence_weight": progress,
    }


def fit_bottom_anchored_model(prices: pd.DataFrame) -> BottomAnchoredModelResult:
    data = _normalise(prices)
    if data.empty:
        raise ValueError("No valid Bitcoin prices are available.")

    bottoms = _extract_regions(
        data, BOTTOM_TURNING_REGIONS, BOTTOM_WINDOW_DAYS, "bottom"
    )
    peaks = _extract_regions(
        data, PEAK_TURNING_REGIONS, PEAK_WINDOW_DAYS, "peak"
    )
    if len(bottoms) < 4:
        raise ValueError("At least four observable Bitcoin bottom regions are required.")

    bottoms = bottoms.sort_values("region_date").reset_index(drop=True)
    bottoms["bottom_to_bottom_multiple"] = bottoms["region_price_usd"].pct_change() + 1.0
    bottoms["bottom_to_bottom_cagr"] = np.nan
    for index in range(1, len(bottoms)):
        years = (bottoms.loc[index, "region_date"] - bottoms.loc[index - 1, "region_date"]).days / 365.2425
        bottoms.loc[index, "bottom_to_bottom_cagr"] = (
            (bottoms.loc[index, "region_price_usd"] / bottoms.loc[index - 1, "region_price_usd"]) ** (1.0 / years) - 1.0
        )

    mature_anchor_dates = [pd.Timestamp(item["anchor_date"]) for item in BOTTOM_TURNING_REGIONS[1:]]
    mature_cycle_days = np.diff(pd.DatetimeIndex(mature_anchor_dates).asi8) / (24 * 60 * 60 * 1e9)
    expected_cycle_days = int(round(float(np.median(mature_cycle_days))))
    current_turning_anchor = pd.Timestamp(BOTTOM_TURNING_REGIONS[-1]["anchor_date"])
    next_bottom_anchor = current_turning_anchor + pd.Timedelta(days=expected_cycle_days)
    mature_bull_days = [
        (pd.Timestamp(PEAK_TURNING_REGIONS[index]["anchor_date"]) - pd.Timestamp(BOTTOM_TURNING_REGIONS[index]["anchor_date"])).days
        for index in range(1, len(PEAK_TURNING_REGIONS))
    ]
    expected_bull_days = int(round(float(np.median(mature_bull_days))))
    next_peak_anchor = current_turning_anchor + pd.Timedelta(days=expected_bull_days)

    walk, validation = _walk_forward(bottoms)
    forecasts: list[dict] = []
    for model_id, label in CANDIDATE_LABELS.items():
        prediction = _candidate_prediction(
            model_id,
            bottoms["region_date"].tolist(),
            bottoms["region_price_usd"].to_numpy(dtype=float),
            next_bottom_anchor,
        )
        validation_row = validation[validation["model_id"] == model_id]
        forecasts.append({
            "model_id": model_id,
            "model": label,
            "next_bottom_anchor": next_bottom_anchor,
            "predicted_bottom_usd": prediction,
            "ensemble_weight": (
                float(validation_row["ensemble_weight"].iloc[0])
                if not validation_row.empty else 0.0
            ),
            "validation_holdouts": (
                int(validation_row["holdouts"].iloc[0])
                if not validation_row.empty else 0
            ),
            "approx_typical_pct_error": (
                float(validation_row["approx_typical_pct_error"].iloc[0])
                if not validation_row.empty else np.nan
            ),
        })
    forecast_df = pd.DataFrame(forecasts)
    valid_forecasts = forecast_df[
        np.isfinite(forecast_df["predicted_bottom_usd"])
        & (forecast_df["predicted_bottom_usd"] > 0)
        & (forecast_df["ensemble_weight"] > 0)
    ].copy()
    valid_forecasts["ensemble_weight"] /= valid_forecasts["ensemble_weight"].sum()
    forecast_df = forecast_df.drop(columns="ensemble_weight").merge(
        valid_forecasts[["model_id", "ensemble_weight"]], on="model_id", how="left"
    )
    forecast_df["ensemble_weight"] = forecast_df["ensemble_weight"].fillna(0.0)
    central_next_bottom = float(np.exp(np.sum(
        valid_forecasts["ensemble_weight"].to_numpy(dtype=float)
        * np.log(valid_forecasts["predicted_bottom_usd"].to_numpy(dtype=float))
    )))
    low_next_bottom = float(valid_forecasts["predicted_bottom_usd"].min())
    high_next_bottom = float(valid_forecasts["predicted_bottom_usd"].max())

    observed_foundation_at_peaks = _log_interpolate(
        peaks["region_date"], bottoms["region_date"], bottoms["region_price_usd"]
    )
    peaks = peaks.copy()
    peaks["bottom_foundation_usd"] = observed_foundation_at_peaks
    peaks["peak_to_bottom_foundation"] = peaks["region_price_usd"] / peaks["bottom_foundation_usd"]
    peaks["cycle_neutral_fair_multiple"] = np.sqrt(peaks["peak_to_bottom_foundation"])

    latest_date = pd.Timestamp(data["date"].max())
    first_curve_date = pd.Timestamp(bottoms["region_date"].min())
    curve_dates = pd.date_range(first_curve_date, next_bottom_anchor, freq="D")
    last_bottom_date = pd.Timestamp(bottoms["region_date"].iloc[-1])
    last_bottom_price = float(bottoms["region_price_usd"].iloc[-1])

    central_anchor_dates = bottoms["region_date"].tolist() + [next_bottom_anchor]
    central_anchor_values = bottoms["region_price_usd"].tolist() + [central_next_bottom]
    low_anchor_values = bottoms["region_price_usd"].tolist() + [low_next_bottom]
    high_anchor_values = bottoms["region_price_usd"].tolist() + [high_next_bottom]
    foundation_central = _log_interpolate(curve_dates, central_anchor_dates, central_anchor_values)
    foundation_low = _log_interpolate(curve_dates, central_anchor_dates, low_anchor_values)
    foundation_high = _log_interpolate(curve_dates, central_anchor_dates, high_anchor_values)

    neutral_multiplier = _log_interpolate(
        curve_dates,
        peaks["region_date"],
        peaks["cycle_neutral_fair_multiple"],
    )
    published_raw_curve, published_fair_curve, published_bottom_curve = _published_lines(curve_dates)
    curve = pd.DataFrame({
        "date": curve_dates,
        "row_type": np.where(curve_dates <= latest_date, "historical", "projected"),
        "bottom_foundation_usd": foundation_central,
        "bottom_foundation_low_usd": foundation_low,
        "bottom_foundation_high_usd": foundation_high,
        "cycle_neutral_multiple": neutral_multiplier,
        "experimental_fair_value_usd": foundation_central * neutral_multiplier,
        "experimental_fair_value_low_usd": foundation_low * neutral_multiplier,
        "experimental_fair_value_high_usd": foundation_high * neutral_multiplier,
        "published_raw_power_law_usd": published_raw_curve,
        "published_fair_value_usd": published_fair_curve,
        "published_bottom_usd": published_bottom_curve,
    })

    current_curve_row = curve.iloc[(curve["date"] - latest_date).abs().argsort()[:1]].iloc[0]
    raw_published, fair_published, bottom_published = _published_lines(pd.DatetimeIndex([latest_date]))
    all_price_curve, all_price_summary = _cycle_balanced_all_price_backbone(data)
    current_fair = float(current_curve_row["experimental_fair_value_usd"])
    latest_price = float(data["price_usd"].iloc[-1])

    summary = {
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY",
        "confidence": "LOW — only a few independent Bitcoin cycles exist",
        "latest_date": latest_date,
        "latest_price_usd": latest_price,
        "current_bottom_region_date": pd.Timestamp(bottoms["region_date"].iloc[-1]),
        "current_bottom_region_usd": last_bottom_price,
        "current_bottom_extreme_usd": float(bottoms["extreme_price_usd"].iloc[-1]),
        "current_bottom_status": bottoms["status"].iloc[-1],
        "current_bottom_foundation_usd": float(current_curve_row["bottom_foundation_usd"]),
        "latest_peak_region_usd": float(peaks["region_price_usd"].iloc[-1]),
        "latest_peak_foundation_multiple": float(peaks["peak_to_bottom_foundation"].iloc[-1]),
        "cycle_neutral_fair_multiple": float(peaks["cycle_neutral_fair_multiple"].iloc[-1]),
        "experimental_fair_value_usd": current_fair,
        "price_to_experimental_fair_value": latest_price / current_fair,
        "published_raw_power_law_usd": float(raw_published[0]),
        "published_fair_value_benchmark_usd": float(fair_published[0]),
        "published_bottom_benchmark_usd": float(bottom_published[0]),
        "all_price_backbone_usd": all_price_summary["latest_backbone_usd"],
        "all_price_backbone_exponent": all_price_summary["exponent"],
        "all_price_open_cycle_weight": all_price_summary["open_cycle_evidence_weight"],
        "expected_cycle_days": expected_cycle_days,
        "expected_bull_days": expected_bull_days,
        "next_peak_anchor": next_peak_anchor,
        "next_bottom_anchor": next_bottom_anchor,
        "next_bottom_ensemble_usd": central_next_bottom,
        "next_bottom_candidate_low_usd": low_next_bottom,
        "next_bottom_candidate_high_usd": high_next_bottom,
        "next_bottom_range_type": "full range of transparent candidate models; not a probability interval",
        "fair_value_method": "log midpoint between the bottom foundation and the latest observed peak multiple",
        "candidate_weight_method": "walk-forward error weighting with forming-cycle evidence at half weight",
    }
    return BottomAnchoredModelResult(
        summary=summary,
        bottom_regions=bottoms,
        peak_regions=peaks,
        curve=curve,
        candidate_forecasts=forecast_df.sort_values("predicted_bottom_usd").reset_index(drop=True),
        validation_summary=validation.sort_values("ensemble_weight", ascending=False).reset_index(drop=True),
        walk_forward=walk.sort_values(["target_date", "model"]).reset_index(drop=True),
        all_price_curve=all_price_curve,
    )
