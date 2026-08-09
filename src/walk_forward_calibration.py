from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd

from src.price_model import (
    HISTORICAL_CYCLE_ANCHORS,
    NEXT_TROUGH,
    PRICE_MODEL_ENGINE_VERSION,
    PriceModelResult,
    fit_price_model,
)

# v2 deliberately leaves the frozen Price Model untouched.  It learns two
# post-model corrections from historical walk-forward outcomes:
#   G = structural-centerline growth correction
#   K = cycle-envelope amplitude correction
CALIBRATION_VERSION = "walk-forward-calibration-v2.0.1-cycle-aware"
CALIBRATION_FLOOR = pd.Timestamp("2015-01-14")
LOOKBACK_YEARS = (4, 8)
FAKE_TODAY_STEP_MONTHS = 6
EVALUATION_HORIZONS_MONTHS = (12, 24, 36, 48)
MAX_EVALUATION_MONTHS = max(EVALUATION_HORIZONS_MONTHS)
STRUCTURAL_SMOOTHING_DAYS = 180
ANCHOR_SMOOTHING_DAYS = 31

# The calibration is allowed to move materially when the out-of-sample data
# supports it, but remains bounded against pathological small samples.
GROWTH_FACTOR_MIN = 0.45
GROWTH_FACTOR_MAX = 1.30
AMPLITUDE_FACTOR_MIN = 0.25
AMPLITUDE_FACTOR_MAX = 1.30

# One pseudo-observation at the frozen-model value is enough to stabilize weak
# samples without forcing the answer back toward 1.0 as strongly as v1 did.
GROWTH_PRIOR_WEIGHT = 1.0
AMPLITUDE_PRIOR_WEIGHT = 0.75

REQUIRED_SUMMARY_KEYS = frozenset({
    "growth_factor",
    "amplitude_factor",
    "raw_cv_error",
    "calibrated_cv_error",
    "raw_structural_cv_error",
    "calibrated_structural_cv_error",
    "raw_envelope_cv_error",
    "calibrated_envelope_cv_error",
    "cv_improvement",
    "stability",
    "status",
    "total_tests",
    "total_structural_points",
    "total_envelope_points",
    "lookback_4y",
    "lookback_8y",
})


@dataclass
class WalkForwardCalibrationResult:
    summary: dict
    tests: pd.DataFrame
    observations: pd.DataFrame
    fingerprint: str


@dataclass
class CalibratedPriceModelResult:
    daily: pd.DataFrame
    turning_points: pd.DataFrame
    diagnostics: dict


def _weighted_median(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(weights.sum())
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _regularized_factor(
    values,
    weights,
    *,
    lower: float,
    upper: float,
    prior_weight: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = np.clip(values[mask], lower, upper)
    weights = weights[mask]
    if len(values) == 0:
        return 1.0
    values = np.concatenate([values, np.array([1.0])])
    weights = np.concatenate([weights, np.array([max(float(prior_weight), 1e-9)])])
    return float(np.clip(_weighted_median(values, weights), lower, upper))


def _weighted_equivalent_pct_error(log_errors, weights=None) -> float:
    values = np.asarray(log_errors, dtype=float)
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    median_abs = _weighted_median(np.abs(values[mask]), weights[mask])
    return float(np.expm1(max(median_abs, 0.0)))


def _nearest_price_on_or_before(data: pd.DataFrame, target: pd.Timestamp) -> pd.Series:
    eligible = data[data["date"] <= pd.Timestamp(target)]
    if eligible.empty:
        raise ValueError(f"No Bitcoin price exists on or before {pd.Timestamp(target).date()}.")
    return eligible.iloc[-1]


def _nearest_price_on_or_after(data: pd.DataFrame, target: pd.Timestamp) -> pd.Series:
    eligible = data[data["date"] >= pd.Timestamp(target)]
    if eligible.empty:
        raise ValueError(f"No Bitcoin price exists on or after {pd.Timestamp(target).date()}.")
    return eligible.iloc[0]


def _smoothed_price_level(
    data: pd.DataFrame,
    end_date: pd.Timestamp,
    window_days: int = STRUCTURAL_SMOOTHING_DAYS,
) -> float:
    """Robust low-frequency realized price level ending at end_date.

    A trailing median of log prices is intentionally used instead of one daily
    close.  The calibration is trying to measure structural growth, not whether
    a fake-today snapshot happened to land on a local wick.
    """
    end_date = pd.Timestamp(end_date)
    start_date = end_date - pd.Timedelta(days=max(int(window_days) - 1, 1))
    window = data[(data["date"] >= start_date) & (data["date"] <= end_date)]
    if window.empty:
        row = _nearest_price_on_or_before(data, end_date)
        return float(row["price_usd"])
    return float(np.exp(np.median(np.log(window["price_usd"].to_numpy(dtype=float)))))


def _smoothed_anchor_price(
    data: pd.DataFrame,
    anchor_date: pd.Timestamp,
    window_days: int = ANCHOR_SMOOTHING_DAYS,
) -> float:
    """Sustained price around a realized turning point rather than one wick."""
    anchor_date = pd.Timestamp(anchor_date)
    half = max(int(window_days) // 2, 1)
    window = data[
        (data["date"] >= anchor_date - pd.Timedelta(days=half))
        & (data["date"] <= anchor_date + pd.Timedelta(days=half))
    ]
    if window.empty:
        row = _nearest_price_on_or_before(data, anchor_date)
        return float(row["price_usd"])
    return float(np.exp(np.median(np.log(window["price_usd"].to_numpy(dtype=float)))))


def first_fake_today_for_lookback(lookback_years: int) -> pd.Timestamp:
    return CALIBRATION_FLOOR + pd.DateOffset(years=int(lookback_years))


def generate_fake_today_dates(
    prices: pd.DataFrame,
    lookback_years: int | None = None,
) -> list[pd.Timestamp]:
    """Generate standardized six-month fake-today dates.

    Four-year tests begin in Jan-2019 and eight-year tests in Jan-2023 so no
    training window ever crosses the Jan-14-2015 calibration floor.  A fake
    date is eligible only after at least 12 months of actual future data exist.
    """
    latest = pd.Timestamp(prices["date"].max()).normalize()
    latest_eligible = latest - pd.DateOffset(months=min(EVALUATION_HORIZONS_MONTHS))

    lookbacks = LOOKBACK_YEARS if lookback_years is None else (int(lookback_years),)
    dates: list[pd.Timestamp] = []
    for lb in lookbacks:
        cursor = first_fake_today_for_lookback(lb)
        while cursor <= latest_eligible:
            actual = pd.Timestamp(_nearest_price_on_or_before(prices, cursor)["date"]).normalize()
            if actual - pd.DateOffset(years=lb) >= CALIBRATION_FLOOR:
                dates.append(actual)
            cursor = cursor + pd.DateOffset(months=FAKE_TODAY_STEP_MONTHS)
    return sorted(set(dates))


def _available_horizons(training_end: pd.Timestamp, latest: pd.Timestamp) -> list[int]:
    return [
        h for h in EVALUATION_HORIZONS_MONTHS
        if training_end + pd.DateOffset(months=h) <= latest
    ]


def _snapshot_growth_factor(structural_rows: pd.DataFrame) -> float:
    if structural_rows.empty:
        return 1.0
    return _regularized_factor(
        structural_rows["implied_growth_factor"],
        structural_rows["evidence_weight"],
        lower=GROWTH_FACTOR_MIN,
        upper=GROWTH_FACTOR_MAX,
        prior_weight=GROWTH_PRIOR_WEIGHT,
    )


def _actual_anchor_amplitude(row: pd.Series, growth_factor: float) -> float:
    c0 = float(row["start_centerline_usd"])
    raw_center = float(row["raw_centerline_usd"])
    cal_center = c0 * (raw_center / c0) ** float(growth_factor)
    actual_price = float(row["actual_anchor_price_usd"])
    signed = float(row["expected_sign"]) * np.log(max(actual_price, 1e-12) / max(cal_center, 1e-12))
    return float(max(signed, 0.0))


def _snapshot_amplitude_factor(envelope_rows: pd.DataFrame, growth_factor: float) -> float:
    if envelope_rows.empty:
        return 1.0
    candidates = []
    weights = []
    for _, row in envelope_rows.iterrows():
        raw_amp = float(row["raw_amplitude"])
        if not np.isfinite(raw_amp) or raw_amp <= 0.02:
            continue
        actual_amp = _actual_anchor_amplitude(row, growth_factor)
        candidates.append(actual_amp / raw_amp)
        weights.append(float(row["evidence_weight"]))
    return _regularized_factor(
        candidates,
        weights,
        lower=AMPLITUDE_FACTOR_MIN,
        upper=AMPLITUDE_FACTOR_MAX,
        prior_weight=AMPLITUDE_PRIOR_WEIGHT,
    )


def _build_backtest_snapshot(
    prices: pd.DataFrame,
    fake_today: pd.Timestamp,
    lookback_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fake_today = pd.Timestamp(fake_today).normalize()
    requested_start = fake_today - pd.DateOffset(years=int(lookback_years))
    if requested_start < CALIBRATION_FLOOR:
        raise ValueError("Walk-forward training may not use data before the Jan 14, 2015 calibration floor.")

    train_start_row = _nearest_price_on_or_after(prices, requested_start)
    train_end_row = _nearest_price_on_or_before(prices, fake_today)
    training_start = pd.Timestamp(train_start_row["date"]).normalize()
    training_end = pd.Timestamp(train_end_row["date"]).normalize()
    latest = pd.Timestamp(prices["date"].max()).normalize()
    horizons = _available_horizons(training_end, latest)
    if not horizons:
        raise ValueError("At least 12 months of unseen future data are required for a walk-forward snapshot.")

    max_horizon = max(horizons)
    projection_years = max(2, int(np.ceil(max_horizon / 12.0)) + 1)
    asof_prices = prices[prices["date"] <= training_end].copy()
    model = fit_price_model(
        prices=asof_prices,
        training_start=training_start,
        training_end=training_end,
        projection_years=projection_years,
    )
    daily = model.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.set_index("date")

    c0 = float(daily.loc[training_end, "structural_centerline_usd"])
    actual_start_level = _smoothed_price_level(prices, training_end)

    structural_rows = []
    for horizon in horizons:
        target = training_end + pd.DateOffset(months=horizon)
        actual_row = _nearest_price_on_or_before(prices, target)
        eval_date = pd.Timestamp(actual_row["date"]).normalize()
        if eval_date not in daily.index:
            continue
        raw_center = float(daily.loc[eval_date, "structural_centerline_usd"])
        raw_growth = float(np.log(raw_center / c0))
        if abs(raw_growth) <= 0.01:
            continue
        actual_level = _smoothed_price_level(prices, eval_date)
        actual_growth = float(np.log(actual_level / actual_start_level))
        implied_g = float(actual_growth / raw_growth)
        # Longer horizons and larger structural moves carry more information
        # about a multi-year forward centerline than a 12-month snapshot.
        horizon_weight = float(horizon / 12.0)
        signal_weight = float(np.clip(abs(raw_growth) / 0.20, 0.50, 2.00))
        evidence_weight = horizon_weight * signal_weight
        structural_rows.append({
            "metric_type": "structural",
            "fake_today": training_end,
            "lookback_years": int(lookback_years),
            "training_start": training_start,
            "eval_date": eval_date,
            "horizon_months": int(horizon),
            "start_centerline_usd": c0,
            "raw_centerline_usd": raw_center,
            "actual_start_level_usd": actual_start_level,
            "actual_level_usd": actual_level,
            "raw_structural_log_growth": raw_growth,
            "actual_structural_log_growth": actual_growth,
            "implied_growth_factor": implied_g,
            "evidence_weight": evidence_weight,
        })
    structural = pd.DataFrame(structural_rows)
    G_snapshot = _snapshot_growth_factor(structural)

    evaluation_end = min(
        training_end + pd.DateOffset(months=MAX_EVALUATION_MONTHS),
        latest,
    )
    envelope_rows = []
    for anchor_date, anchor_type, cycle in HISTORICAL_CYCLE_ANCHORS:
        anchor_date = pd.Timestamp(anchor_date).normalize()
        if not (training_end < anchor_date <= evaluation_end):
            continue
        if anchor_date not in daily.index:
            continue
        raw_center = float(daily.loc[anchor_date, "structural_centerline_usd"])
        raw_price = float(daily.loc[anchor_date, "fitted_or_projected_price_usd"])
        expected_sign = 1.0 if anchor_type == "peak" else -1.0
        raw_signed = expected_sign * np.log(max(raw_price, 1e-12) / max(raw_center, 1e-12))
        raw_amp = float(max(raw_signed, 0.0))
        if raw_amp <= 0.02:
            continue
        actual_anchor = _smoothed_anchor_price(prices, anchor_date)
        months_forward = max((anchor_date - training_end).days / 30.4375, 0.0)
        evidence_weight = float(np.clip(months_forward / 12.0, 0.75, 4.0) * np.clip(raw_amp / 0.25, 0.5, 2.0))
        envelope_rows.append({
            "metric_type": "envelope",
            "fake_today": training_end,
            "lookback_years": int(lookback_years),
            "training_start": training_start,
            "anchor_date": anchor_date,
            "anchor_type": anchor_type,
            "cycle": int(cycle),
            "months_forward": float(months_forward),
            "start_centerline_usd": c0,
            "raw_centerline_usd": raw_center,
            "raw_projected_anchor_price_usd": raw_price,
            "actual_anchor_price_usd": actual_anchor,
            "raw_amplitude": raw_amp,
            "expected_sign": expected_sign,
            "evidence_weight": evidence_weight,
        })
    envelope = pd.DataFrame(envelope_rows)
    K_snapshot = _snapshot_amplitude_factor(envelope, G_snapshot)

    # Store the snapshot's own realized amplitude evidence for transparency.
    if not envelope.empty:
        envelope = envelope.copy()
        envelope["actual_amplitude_using_snapshot_G"] = envelope.apply(
            lambda r: _actual_anchor_amplitude(r, G_snapshot), axis=1
        )
        envelope["implied_amplitude_factor"] = np.where(
            envelope["raw_amplitude"] > 0,
            envelope["actual_amplitude_using_snapshot_G"] / envelope["raw_amplitude"],
            np.nan,
        )

    meta = {
        "fake_today": training_end,
        "lookback_years": int(lookback_years),
        "training_start": training_start,
        "training_end": training_end,
        "max_horizon_months": int(max_horizon),
        "structural_points": int(len(structural)),
        "envelope_points": int(len(envelope)),
        "snapshot_growth_factor": float(G_snapshot),
        "snapshot_amplitude_factor": float(K_snapshot),
        "structural_evidence_weight": float(structural["evidence_weight"].sum()) if not structural.empty else 0.0,
        "envelope_evidence_weight": float(envelope["evidence_weight"].sum()) if not envelope.empty else 0.0,
    }
    return structural, envelope, meta


def _fit_snapshot_factor_medians(tests: pd.DataFrame) -> tuple[float, float]:
    if tests.empty:
        return 1.0, 1.0
    G = _regularized_factor(
        tests["snapshot_growth_factor"],
        np.maximum(tests["structural_evidence_weight"].to_numpy(dtype=float), 0.25),
        lower=GROWTH_FACTOR_MIN,
        upper=GROWTH_FACTOR_MAX,
        prior_weight=GROWTH_PRIOR_WEIGHT,
    )
    k_mask = tests["envelope_points"].to_numpy(dtype=float) > 0
    K = _regularized_factor(
        tests.loc[k_mask, "snapshot_amplitude_factor"],
        np.maximum(tests.loc[k_mask, "envelope_evidence_weight"].to_numpy(dtype=float), 0.25),
        lower=AMPLITUDE_FACTOR_MIN,
        upper=AMPLITUDE_FACTOR_MAX,
        prior_weight=AMPLITUDE_PRIOR_WEIGHT,
    ) if np.any(k_mask) else 1.0
    return float(G), float(K)


def _score_holdout(
    structural: pd.DataFrame,
    envelope: pd.DataFrame,
    growth_factor: float,
    amplitude_factor: float,
) -> dict:
    if structural.empty:
        raw_struct = cal_struct = float("nan")
    else:
        y = structural["actual_structural_log_growth"].to_numpy(dtype=float)
        x = structural["raw_structural_log_growth"].to_numpy(dtype=float)
        w = structural["evidence_weight"].to_numpy(dtype=float)
        raw_struct = _weighted_equivalent_pct_error(y - x, w)
        cal_struct = _weighted_equivalent_pct_error(y - growth_factor * x, w)

    if envelope.empty:
        raw_env = cal_env = float("nan")
    else:
        weights = envelope["evidence_weight"].to_numpy(dtype=float)
        raw_pred = envelope["raw_amplitude"].to_numpy(dtype=float)
        raw_actual = np.array([_actual_anchor_amplitude(r, 1.0) for _, r in envelope.iterrows()])
        cal_actual = np.array([_actual_anchor_amplitude(r, growth_factor) for _, r in envelope.iterrows()])
        raw_env = _weighted_equivalent_pct_error(raw_actual - raw_pred, weights)
        cal_env = _weighted_equivalent_pct_error(cal_actual - amplitude_factor * raw_pred, weights)

    raw_parts = [x for x in (raw_struct, raw_env) if np.isfinite(x)]
    cal_parts = [x for x in (cal_struct, cal_env) if np.isfinite(x)]
    raw_total = float(np.mean(raw_parts)) if raw_parts else float("nan")
    cal_total = float(np.mean(cal_parts)) if cal_parts else float("nan")
    return {
        "raw_structural_error": raw_struct,
        "calibrated_structural_error": cal_struct,
        "raw_envelope_error": raw_env,
        "calibrated_envelope_error": cal_env,
        "raw_error": raw_total,
        "calibrated_error": cal_total,
    }


def _cross_validate_one_lookback(
    structural: pd.DataFrame,
    envelope: pd.DataFrame,
    snapshot_tests: pd.DataFrame,
) -> dict:
    groups = sorted(pd.to_datetime(snapshot_tests["fake_today"].unique()))
    rows = []
    for holdout in groups:
        train = snapshot_tests[pd.to_datetime(snapshot_tests["fake_today"]) != holdout]
        if train.empty:
            continue
        G, K = _fit_snapshot_factor_medians(train)
        s = structural[pd.to_datetime(structural["fake_today"]) == holdout] if not structural.empty else structural
        e = envelope[pd.to_datetime(envelope["fake_today"]) == holdout] if not envelope.empty else envelope
        score = _score_holdout(s, e, G, K)
        rows.append({
            "fake_today": pd.Timestamp(holdout),
            "cv_growth_factor": G,
            "cv_amplitude_factor": K,
            **score,
        })
    holdouts = pd.DataFrame(rows)
    if holdouts.empty:
        return {"holdouts": holdouts}

    def med(col):
        values = holdouts[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else float("nan")

    return {
        "holdouts": holdouts,
        "raw_cv_error": med("raw_error"),
        "calibrated_cv_error": med("calibrated_error"),
        "raw_structural_cv_error": med("raw_structural_error"),
        "calibrated_structural_cv_error": med("calibrated_structural_error"),
        "raw_envelope_cv_error": med("raw_envelope_error"),
        "calibrated_envelope_cv_error": med("calibrated_envelope_error"),
    }


def _stability_label(snapshot_factors: pd.DataFrame) -> tuple[str, float]:
    if snapshot_factors.empty or len(snapshot_factors) < 3:
        return "LOW", float("nan")
    g = snapshot_factors["snapshot_growth_factor"].dropna()
    k = snapshot_factors.loc[snapshot_factors["envelope_points"] > 0, "snapshot_amplitude_factor"].dropna()
    g_iqr = float(g.quantile(0.75) - g.quantile(0.25)) if len(g) >= 2 else 0.0
    k_iqr = float(k.quantile(0.75) - k.quantile(0.25)) if len(k) >= 2 else 0.0
    spread = max(g_iqr, k_iqr)
    if spread <= 0.18:
        return "HIGH", spread
    if spread <= 0.35:
        return "MEDIUM", spread
    return "LOW", spread


def _fingerprint(summary: dict, tests: pd.DataFrame, latest_data_date: pd.Timestamp) -> str:
    payload = {
        "version": CALIBRATION_VERSION,
        "price_model_engine": PRICE_MODEL_ENGINE_VERSION,
        "latest_data_date": pd.Timestamp(latest_data_date).date().isoformat(),
        "floor": CALIBRATION_FLOOR.date().isoformat(),
        "G": round(float(summary.get("growth_factor", 1.0)), 10),
        "K": round(float(summary.get("amplitude_factor", 1.0)), 10),
        "status": str(summary.get("status", "UNKNOWN")),
        "tests": [
            {
                "fake_today": pd.Timestamp(r.fake_today).date().isoformat(),
                "lookback": int(r.lookback_years),
                "max_horizon": int(r.max_horizon_months),
            }
            for r in tests[["fake_today", "lookback_years", "max_horizon_months"]].itertuples(index=False)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def run_walk_forward_calibration(
    prices: pd.DataFrame,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> WalkForwardCalibrationResult:
    data = prices[["date", "price_usd"]].copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None).dt.normalize()
    latest = pd.Timestamp(data["date"].max()).normalize()

    schedule = []
    for lookback in LOOKBACK_YEARS:
        for fake_today in generate_fake_today_dates(data, lookback):
            schedule.append((lookback, fake_today))
    if not schedule:
        raise ValueError("Not enough post-Jan-2015 history exists for walk-forward calibration.")

    total = len(schedule)
    structural_frames = []
    envelope_frames = []
    test_meta = []
    for done, (lookback, fake_today) in enumerate(schedule, start=1):
        s, e, meta = _build_backtest_snapshot(data, fake_today, lookback)
        if not s.empty:
            structural_frames.append(s)
        if not e.empty:
            envelope_frames.append(e)
        test_meta.append(meta)
        if progress_callback is not None:
            progress_callback(done, total, f"{lookback}Y as-of {pd.Timestamp(fake_today).date()}")

    structural = pd.concat(structural_frames, ignore_index=True) if structural_frames else pd.DataFrame()
    envelope = pd.concat(envelope_frames, ignore_index=True) if envelope_frames else pd.DataFrame()
    tests = pd.DataFrame(test_meta).sort_values(["lookback_years", "fake_today"]).reset_index(drop=True)

    lookback_summaries = {}
    cv_frames = []
    blend_rows = []
    for lookback in LOOKBACK_YEARS:
        test_lb = tests[tests["lookback_years"] == lookback].copy()
        struct_lb = structural[structural["lookback_years"] == lookback].copy() if not structural.empty else structural
        env_lb = envelope[envelope["lookback_years"] == lookback].copy() if not envelope.empty else envelope
        G, K = _fit_snapshot_factor_medians(test_lb)
        cv = _cross_validate_one_lookback(struct_lb, env_lb, test_lb)
        hold = cv.get("holdouts", pd.DataFrame())
        if not hold.empty:
            hold = hold.copy()
            hold["lookback_years"] = int(lookback)
            cv_frames.append(hold)
        stability, spread = _stability_label(test_lb)
        cal_err = float(cv.get("calibrated_cv_error", np.nan))
        evidence = max(float(len(test_lb)), 1.0)
        accuracy_weight = np.sqrt(evidence) / max(cal_err if np.isfinite(cal_err) else 1.0, 0.03)
        blend_rows.append((lookback, G, K, accuracy_weight))
        lookback_summaries[lookback] = {
            "growth_factor": G,
            "amplitude_factor": K,
            "raw_cv_error": float(cv.get("raw_cv_error", np.nan)),
            "calibrated_cv_error": cal_err,
            "raw_structural_cv_error": float(cv.get("raw_structural_cv_error", np.nan)),
            "calibrated_structural_cv_error": float(cv.get("calibrated_structural_cv_error", np.nan)),
            "raw_envelope_cv_error": float(cv.get("raw_envelope_cv_error", np.nan)),
            "calibrated_envelope_cv_error": float(cv.get("calibrated_envelope_cv_error", np.nan)),
            "stability": stability,
            "stability_spread": spread,
            "tests": int(len(test_lb)),
            "structural_points": int(len(struct_lb)),
            "envelope_points": int(len(env_lb)),
            "first_fake_today": pd.Timestamp(test_lb["fake_today"].min()).date().isoformat() if not test_lb.empty else None,
            "last_fake_today": pd.Timestamp(test_lb["fake_today"].max()).date().isoformat() if not test_lb.empty else None,
            "accuracy_weight": float(accuracy_weight),
        }

    weights = np.asarray([x[3] for x in blend_rows], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(blend_rows), dtype=float)
    weights /= weights.sum()
    combined_G = float(np.clip(np.sum(weights * np.asarray([x[1] for x in blend_rows])), GROWTH_FACTOR_MIN, GROWTH_FACTOR_MAX))
    combined_K = float(np.clip(np.sum(weights * np.asarray([x[2] for x in blend_rows])), AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX))

    def blended(metric: str) -> float:
        vals = np.asarray([lookback_summaries[y][metric] for y in LOOKBACK_YEARS], dtype=float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            return float("nan")
        w = weights[mask]
        w = w / w.sum()
        return float(np.sum(w * vals[mask]))

    raw_cv_error = blended("raw_cv_error")
    calibrated_cv_error = blended("calibrated_cv_error")
    raw_struct = blended("raw_structural_cv_error")
    cal_struct = blended("calibrated_structural_cv_error")
    raw_env = blended("raw_envelope_cv_error")
    cal_env = blended("calibrated_envelope_cv_error")
    improvement = (
        float(1.0 - calibrated_cv_error / raw_cv_error)
        if np.isfinite(raw_cv_error) and raw_cv_error > 0 and np.isfinite(calibrated_cv_error)
        else float("nan")
    )

    overall_stability = "HIGH"
    if any(lookback_summaries[y]["stability"] == "LOW" for y in LOOKBACK_YEARS):
        overall_stability = "LOW"
    elif any(lookback_summaries[y]["stability"] == "MEDIUM" for y in LOOKBACK_YEARS):
        overall_stability = "MEDIUM"

    enough_tests = lookback_summaries[4]["tests"] >= 5 and lookback_summaries[8]["tests"] >= 3
    enough_envelope = sum(lookback_summaries[y]["envelope_points"] for y in LOOKBACK_YEARS) >= 4
    stable_enough = overall_stability != "LOW"
    if not stable_enough:
        status = "UNSTABLE"
    elif not enough_tests or not enough_envelope:
        status = "INSUFFICIENT_EVIDENCE"
    elif np.isfinite(improvement) and improvement >= 0.10:
        status = "PASS"
    elif np.isfinite(improvement) and improvement > 0.0:
        status = "MODEST"
    else:
        status = "NO_IMPROVEMENT"

    summary = {
        "version": CALIBRATION_VERSION,
        "price_model_engine_version": PRICE_MODEL_ENGINE_VERSION,
        "calibration_floor": CALIBRATION_FLOOR.date().isoformat(),
        "fake_today_step_months": FAKE_TODAY_STEP_MONTHS,
        "evaluation_horizons_months": list(EVALUATION_HORIZONS_MONTHS),
        "max_evaluation_months": MAX_EVALUATION_MONTHS,
        "structural_smoothing_days": STRUCTURAL_SMOOTHING_DAYS,
        "anchor_smoothing_days": ANCHOR_SMOOTHING_DAYS,
        "growth_factor": combined_G,
        "amplitude_factor": combined_K,
        "raw_cv_error": raw_cv_error,
        "calibrated_cv_error": calibrated_cv_error,
        "raw_structural_cv_error": raw_struct,
        "calibrated_structural_cv_error": cal_struct,
        "raw_envelope_cv_error": raw_env,
        "calibrated_envelope_cv_error": cal_env,
        "cv_improvement": improvement,
        "stability": overall_stability,
        "status": status,
        "total_tests": int(len(tests)),
        "total_structural_points": int(len(structural)),
        "total_envelope_points": int(len(envelope)),
        "lookback_4y": lookback_summaries[4],
        "lookback_8y": lookback_summaries[8],
        "lookback_weights": {str(item[0]): float(w) for item, w in zip(blend_rows, weights)},
        "latest_data_date": latest.date().isoformat(),
    }

    if cv_frames:
        holdouts = pd.concat(cv_frames, ignore_index=True)
        tests = tests.merge(
            holdouts,
            on=["fake_today", "lookback_years"],
            how="left",
            suffixes=("", "_cv"),
        )

    observations = pd.concat(
        [x for x in [structural, envelope] if x is not None and not x.empty],
        ignore_index=True,
        sort=False,
    ) if (not structural.empty or not envelope.empty) else pd.DataFrame()

    fingerprint = _fingerprint(summary, tests, latest)
    summary["fingerprint"] = fingerprint
    tests = tests.copy()
    tests["final_growth_factor"] = combined_G
    tests["final_amplitude_factor"] = combined_K
    return WalkForwardCalibrationResult(summary, tests, observations, fingerprint)


def calibration_is_current(
    calibration: WalkForwardCalibrationResult | None,
    prices: pd.DataFrame,
) -> bool:
    if calibration is None:
        return False
    summary = getattr(calibration, "summary", {})
    return (
        summary.get("version") == CALIBRATION_VERSION
        and summary.get("price_model_engine_version") == PRICE_MODEL_ENGINE_VERSION
        and summary.get("latest_data_date") == pd.Timestamp(prices["date"].max()).date().isoformat()
        and REQUIRED_SUMMARY_KEYS.issubset(summary.keys())
    )


def _calibrated_centerline_from_raw(
    raw_center: np.ndarray,
    dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    growth_factor: float,
) -> np.ndarray:
    raw_center = np.asarray(raw_center, dtype=float)
    out = raw_center.copy()
    start_date = pd.Timestamp(start_date)
    idx = int(np.searchsorted(dates.values, np.datetime64(start_date), side="left"))
    idx = min(max(idx, 0), len(dates) - 1)
    c0 = float(raw_center[idx])
    mask = np.arange(len(dates)) > idx
    ratio = np.maximum(raw_center[mask] / c0, 1e-12)
    out[mask] = c0 * np.power(ratio, float(growth_factor))
    return out


def build_calibrated_price_model(
    base_model: PriceModelResult,
    calibration: WalkForwardCalibrationResult,
) -> CalibratedPriceModelResult:
    daily = base_model.daily.copy().sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(daily["date"]))
    raw_center = daily["structural_centerline_usd"].to_numpy(dtype=float)
    raw_price = daily["fitted_or_projected_price_usd"].to_numpy(dtype=float)
    training_end = pd.Timestamp(base_model.diagnostics["training_end"])
    projection_end = pd.Timestamp(base_model.diagnostics["projection_end_date"])

    G = float(calibration.summary.get("growth_factor", 1.0))
    K = float(calibration.summary.get("amplitude_factor", 1.0))

    # Preserve the currently observed/conditioned 2025-26 bear path. Once that
    # live-cycle trough is reached, calibration takes over the structural growth
    # and all complete future cycle amplitudes.
    if training_end < NEXT_TROUGH <= projection_end:
        calibration_start = NEXT_TROUGH
        live_cycle_preserved = True
    else:
        calibration_start = training_end
        live_cycle_preserved = False

    calibrated_center = _calibrated_centerline_from_raw(raw_center, dates, calibration_start, G)
    calibrated_price = raw_price.copy()

    projected_mask = daily["row_type"].eq("projected").to_numpy()
    after_start = dates > calibration_start
    direct_mask = projected_mask & after_start
    raw_deviation = np.log(np.maximum(raw_price, 1e-12) / np.maximum(raw_center, 1e-12))
    calibrated_price[direct_mask] = calibrated_center[direct_mask] * np.exp(K * raw_deviation[direct_mask])

    # Connect the preserved live-cycle boundary continuously to the first fully
    # calibrated turning point while preserving the frozen model's phase shape.
    anchor_frames = []
    for key in ("cycle_anchor_table", "cycle_anchor_lookahead_table"):
        frame = base_model.diagnostics.get(key)
        if frame is not None and not frame.empty:
            anchor_frames.append(frame.copy())
    anchors = (
        pd.concat(anchor_frames, ignore_index=True, sort=False)
        .drop_duplicates(subset=["date", "type"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
        if anchor_frames else pd.DataFrame()
    )
    future_turns = anchors[
        (pd.to_datetime(anchors.get("date")) > calibration_start)
        & anchors.get("type", pd.Series(dtype=str)).isin(["peak", "trough"])
    ].copy() if not anchors.empty else pd.DataFrame()

    first_turn_date = None
    if not future_turns.empty:
        first_turn = future_turns.iloc[0]
        first_turn_date = pd.Timestamp(first_turn["date"])
        if first_turn_date in dates and calibration_start in dates:
            turn_idx = int(np.where(dates == first_turn_date)[0][0])
            start_idx = int(np.where(dates == calibration_start)[0][0])
            if turn_idx > start_idx:
                raw_start = float(raw_price[start_idx])
                raw_end = float(raw_price[turn_idx])
                cal_start = raw_start
                cal_end = float(calibrated_center[turn_idx] * np.exp(K * raw_deviation[turn_idx]))
                denom = np.log(raw_end) - np.log(raw_start)
                if abs(denom) > 1e-12:
                    segment = np.arange(start_idx, turn_idx + 1)
                    progress = (
                        np.log(np.maximum(raw_price[segment], 1e-12)) - np.log(raw_start)
                    ) / denom
                    progress = np.clip(progress, 0.0, 1.0)
                    calibrated_price[segment] = np.exp(
                        np.log(cal_start) + progress * (np.log(cal_end) - np.log(cal_start))
                    )

    preserve_mask = (~projected_mask) | (dates <= calibration_start)
    calibrated_price[preserve_mask] = raw_price[preserve_mask]
    calibrated_center[dates <= calibration_start] = raw_center[dates <= calibration_start]

    out = daily[["date", "row_type", "actual_price_usd"]].copy()
    out["raw_centerline_usd"] = raw_center
    out["raw_price_usd"] = raw_price
    out["calibrated_centerline_usd"] = calibrated_center
    out["calibrated_price_usd"] = calibrated_price
    out["calibration_active"] = dates > calibration_start

    turning_rows = []
    if not anchors.empty:
        visible = anchors[
            (pd.to_datetime(anchors["date"]) >= training_end)
            & (pd.to_datetime(anchors["date"]) <= projection_end)
            & anchors["type"].isin(["peak", "trough"])
        ].copy()
        for row in visible.itertuples(index=False):
            d = pd.Timestamp(row.date)
            if d not in dates:
                continue
            i = int(np.where(dates == d)[0][0])
            turning_rows.append({
                "date": d,
                "type": row.type,
                "cycle": getattr(row, "cycle", np.nan),
                "raw_price_usd": float(raw_price[i]),
                "raw_centerline_usd": float(raw_center[i]),
                "calibrated_price_usd": float(calibrated_price[i]),
                "calibrated_centerline_usd": float(calibrated_center[i]),
                "raw_price_over_centerline": float(raw_price[i] / raw_center[i]),
                "calibrated_price_over_centerline": float(calibrated_price[i] / calibrated_center[i]),
                "source": getattr(row, "source", ""),
            })
    turning_points = pd.DataFrame(turning_rows)

    diagnostics = {
        "version": "calibrated-price-model-v2.0-cycle-aware",
        "base_model_version": base_model.diagnostics.get("model_version", PRICE_MODEL_ENGINE_VERSION),
        "calibration_version": calibration.summary.get("version", CALIBRATION_VERSION),
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_status": calibration.summary.get("status", "UNKNOWN"),
        "growth_factor": G,
        "amplitude_factor": K,
        "calibration_start_date": calibration_start.date().isoformat(),
        "live_cycle_preserved_through_calibration_start": live_cycle_preserved,
        "first_calibrated_turning_point": first_turn_date.date().isoformat() if first_turn_date is not None else None,
        "structural_centerline_rebased": True,
        "cycle_amplitude_rebased": True,
    }
    return CalibratedPriceModelResult(out, turning_points, diagnostics)


def build_calibrated_projection_fingerprint(
    base_fingerprint: str,
    calibrated: CalibratedPriceModelResult,
) -> str:
    d = calibrated.diagnostics
    payload = {
        "base": str(base_fingerprint),
        "calibration": str(d.get("calibration_fingerprint")),
        "G": round(float(d.get("growth_factor", 1.0)), 10),
        "K": round(float(d.get("amplitude_factor", 1.0)), 10),
        "start": str(d.get("calibration_start_date")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
