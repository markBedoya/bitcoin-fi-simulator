from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd

from src.price_model import (
    FIXED_BULL_DAYS,
    FIXED_CYCLE_DAYS,
    HISTORICAL_CYCLE_ANCHORS,
    NEXT_TROUGH,
    PRICE_MODEL_ENGINE_VERSION,
    PriceModelResult,
    fit_price_model,
)

# The frozen Price Model remains untouched.  This layer learns from genuine
# walk-forward outcomes and maintains a growing ensemble of cycle-aligned
# parent models.  New confirmed troughs can join the parent set over time.
CALIBRATION_VERSION = "walk-forward-calibration-v4.0.0-cycle-disciplined-learning"
CALIBRATION_FLOOR = pd.Timestamp("2015-01-14")
LOOKBACK_YEARS = (4, 8)
FAKE_TODAY_STEP_MONTHS = 6
EVALUATION_HORIZONS_MONTHS = (12, 24, 36, 48)
MAX_EVALUATION_MONTHS = max(EVALUATION_HORIZONS_MONTHS)
ENVELOPE_MAX_EVALUATION_MONTHS = 96
STRUCTURAL_SMOOTHING_DAYS = 180
ANCHOR_SMOOTHING_DAYS = 31

# Dynamic cycle-parent discovery.  The historical anchors are already observed
# facts in the frozen model.  After those, an expected cycle window becomes an
# observed parent only once enough actual data exists on both sides to identify
# the realized local trough/peak.  This does not rewrite the frozen model.
ANCHOR_DISCOVERY_HALF_WINDOW_DAYS = 180
ANCHOR_CONFIRMATION_DAYS = 180
PARENT_MIN_TRAIN_YEARS = 2
PARENT_TEST_STEP_MONTHS = 12

GROWTH_FACTOR_MIN = 0.45
GROWTH_FACTOR_MAX = 1.30
AMPLITUDE_FACTOR_MIN = 0.01
AMPLITUDE_FACTOR_MAX = 1.30
GROWTH_PRIOR_WEIGHT = 1.0
AMPLITUDE_PRIOR_WEIGHT = 0.75
GEOMETRY_NUMERICAL_EPS_LOG = 1e-10

REQUIRED_SUMMARY_KEYS = frozenset({
    "growth_factor",
    "effective_growth_factor",
    "structural_blend_weight",
    "growth_status",
    "amplitude_factor",
    "amplitude_constant_factor",
    "amplitude_trend_blend_weight",
    "amplitude_mode",
    "amplitude_trend_direction",
    "amplitude_trend_confidence",
    "amplitude_trend_change_per_cycle",
    "geometry_guard_enabled",
    "raw_cv_error",
    "calibrated_cv_error",
    "raw_structural_cv_error",
    "calibrated_structural_cv_error",
    "raw_envelope_cv_error",
    "calibrated_envelope_cv_error",
    "cv_improvement",
    "envelope_status",
    "status",
    "total_tests",
    "total_structural_points",
    "total_envelope_points",
    "cycle_parents",
    "cycle_parent_count",
    "latest_data_date",
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


def _normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices[["date", "price_usd"]].copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None).dt.normalize()
    data = data.drop_duplicates(subset=["date"], keep="last")
    data = data[np.isfinite(data["price_usd"]) & (data["price_usd"] > 0)].reset_index(drop=True)
    return data


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


def _regularized_factor(values, weights, *, lower: float, upper: float, prior_weight: float) -> float:
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


def _smoothed_price_level(data: pd.DataFrame, end_date: pd.Timestamp, window_days: int = STRUCTURAL_SMOOTHING_DAYS) -> float:
    end_date = pd.Timestamp(end_date)
    start_date = end_date - pd.Timedelta(days=max(int(window_days) - 1, 1))
    window = data[(data["date"] >= start_date) & (data["date"] <= end_date)]
    if window.empty:
        return float(_nearest_price_on_or_before(data, end_date)["price_usd"])
    return float(np.exp(np.median(np.log(window["price_usd"].to_numpy(dtype=float)))))


def _smoothed_anchor_price(data: pd.DataFrame, anchor_date: pd.Timestamp, window_days: int = ANCHOR_SMOOTHING_DAYS) -> float:
    anchor_date = pd.Timestamp(anchor_date)
    half = max(int(window_days) // 2, 1)
    window = data[(data["date"] >= anchor_date - pd.Timedelta(days=half)) & (data["date"] <= anchor_date + pd.Timedelta(days=half))]
    if window.empty:
        return float(_nearest_price_on_or_before(data, anchor_date)["price_usd"])
    return float(np.exp(np.median(np.log(window["price_usd"].to_numpy(dtype=float)))))


def discover_observed_cycle_anchors(prices: pd.DataFrame) -> pd.DataFrame:
    """Return observed cycle anchors available as of the latest actual price.

    The known 2015/2018/2022 troughs and intervening peaks come from the frozen
    model's observed-anchor table.  For later cycles, the calibration layer can
    grow automatically: once 180 days of confirming data exist after a modeled
    cycle window, it records the realized local min/max inside +/-180 days.
    """
    data = _normalise_prices(prices)
    latest = pd.Timestamp(data["date"].max())
    rows = []
    for d, typ, cycle in HISTORICAL_CYCLE_ANCHORS:
        d = pd.Timestamp(d).normalize()
        if CALIBRATION_FLOOR <= d <= latest:
            rows.append({"date": d, "type": typ, "cycle": int(cycle), "source": "frozen observed historical anchor"})

    # Grow beyond the currently hard-coded historical anchor list when enough
    # actual data is available to identify what really happened near the next
    # modeled cycle windows.
    trough = pd.Timestamp(NEXT_TROUGH).normalize()
    cycle = 1
    while trough + pd.Timedelta(days=ANCHOR_CONFIRMATION_DAYS) <= latest:
        for typ, expected in (("trough", trough), ("peak", trough + pd.Timedelta(days=FIXED_BULL_DAYS))):
            if expected + pd.Timedelta(days=ANCHOR_CONFIRMATION_DAYS) > latest:
                continue
            if any(abs((pd.Timestamp(r["date"]) - expected).days) <= ANCHOR_DISCOVERY_HALF_WINDOW_DAYS and r["type"] == typ for r in rows):
                continue
            window = data[
                (data["date"] >= expected - pd.Timedelta(days=ANCHOR_DISCOVERY_HALF_WINDOW_DAYS))
                & (data["date"] <= expected + pd.Timedelta(days=ANCHOR_DISCOVERY_HALF_WINDOW_DAYS))
            ]
            if window.empty:
                continue
            idx = window["price_usd"].idxmin() if typ == "trough" else window["price_usd"].idxmax()
            realized = pd.Timestamp(data.loc[idx, "date"]).normalize()
            rows.append({
                "date": realized,
                "type": typ,
                "cycle": int(cycle),
                "source": "calibration-discovered realized local turning point",
                "expected_window_center": expected,
            })
        trough = trough + pd.Timedelta(days=FIXED_CYCLE_DAYS)
        cycle += 1

    anchors = pd.DataFrame(rows).sort_values("date").drop_duplicates(subset=["date", "type"], keep="first").reset_index(drop=True)
    return anchors


def discover_cycle_aligned_parent_starts(prices: pd.DataFrame) -> list[pd.Timestamp]:
    anchors = discover_observed_cycle_anchors(prices)
    starts = [pd.Timestamp(d).normalize() for d in anchors.loc[anchors["type"] == "trough", "date"]]
    return sorted(set(d for d in starts if d >= CALIBRATION_FLOOR))


def first_fake_today_for_lookback(lookback_years: int) -> pd.Timestamp:
    return CALIBRATION_FLOOR + pd.DateOffset(years=int(lookback_years))


def generate_fake_today_dates(prices: pd.DataFrame, lookback_years: int | None = None) -> list[pd.Timestamp]:
    data = _normalise_prices(prices)
    latest = pd.Timestamp(data["date"].max()).normalize()
    latest_eligible = latest - pd.DateOffset(months=min(EVALUATION_HORIZONS_MONTHS))
    lookbacks = LOOKBACK_YEARS if lookback_years is None else (int(lookback_years),)
    dates: list[pd.Timestamp] = []
    for lb in lookbacks:
        cursor = first_fake_today_for_lookback(lb)
        while cursor <= latest_eligible:
            actual = pd.Timestamp(_nearest_price_on_or_before(data, cursor)["date"]).normalize()
            if actual - pd.DateOffset(years=lb) >= CALIBRATION_FLOOR:
                dates.append(actual)
            cursor = cursor + pd.DateOffset(months=FAKE_TODAY_STEP_MONTHS)
    return sorted(set(dates))


def _available_horizons(training_end: pd.Timestamp, latest: pd.Timestamp) -> list[int]:
    return [h for h in EVALUATION_HORIZONS_MONTHS if training_end + pd.DateOffset(months=h) <= latest]


def _snapshot_growth_factor(structural_rows: pd.DataFrame) -> float:
    if structural_rows.empty:
        return 1.0
    return _regularized_factor(
        structural_rows["implied_growth_factor"], structural_rows["evidence_weight"],
        lower=GROWTH_FACTOR_MIN, upper=GROWTH_FACTOR_MAX, prior_weight=GROWTH_PRIOR_WEIGHT,
    )


def _actual_anchor_amplitude(row: pd.Series, growth_factor: float = 1.0) -> float:
    c0 = float(row["start_centerline_usd"])
    raw_center = float(row["raw_centerline_usd"])
    cal_center = c0 * (raw_center / c0) ** float(growth_factor)
    actual_price = float(row["actual_anchor_price_usd"])
    signed = float(row["expected_sign"]) * np.log(max(actual_price, 1e-12) / max(cal_center, 1e-12))
    return float(max(signed, 0.0))


def _snapshot_amplitude_factor(envelope_rows: pd.DataFrame) -> float:
    """Learn envelope compression independently of structural G."""
    if envelope_rows.empty:
        return 1.0
    candidates, weights = [], []
    for _, row in envelope_rows.iterrows():
        raw_amp = float(row["raw_amplitude"])
        if not np.isfinite(raw_amp) or raw_amp <= 0.02:
            continue
        actual_amp = _actual_anchor_amplitude(row, 1.0)
        candidates.append(actual_amp / raw_amp)
        weights.append(float(row["evidence_weight"]))
    return _regularized_factor(
        candidates, weights, lower=AMPLITUDE_FACTOR_MIN, upper=AMPLITUDE_FACTOR_MAX,
        prior_weight=AMPLITUDE_PRIOR_WEIGHT,
    )


def _build_backtest_snapshot_for_start(
    prices: pd.DataFrame,
    fake_today: pd.Timestamp,
    training_start_requested: pd.Timestamp,
    *,
    lookback_years: int | None = None,
    parent_start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = _normalise_prices(prices)
    fake_today = pd.Timestamp(fake_today).normalize()
    requested_start = pd.Timestamp(training_start_requested).normalize()
    if requested_start < CALIBRATION_FLOOR:
        raise ValueError("Walk-forward training may not use data before the Jan 14, 2015 calibration floor.")

    train_start_row = _nearest_price_on_or_after(data, requested_start)
    train_end_row = _nearest_price_on_or_before(data, fake_today)
    training_start = pd.Timestamp(train_start_row["date"]).normalize()
    training_end = pd.Timestamp(train_end_row["date"]).normalize()
    if training_start >= training_end:
        raise ValueError("Walk-forward training start must precede fake today.")
    latest = pd.Timestamp(data["date"].max()).normalize()
    horizons = _available_horizons(training_end, latest)
    if not horizons:
        raise ValueError("At least 12 months of unseen future data are required for a walk-forward snapshot.")

    max_horizon = max(horizons)
    projection_years = max(2, int(np.ceil(max(max_horizon, ENVELOPE_MAX_EVALUATION_MONTHS) / 12.0)) + 1)
    asof_prices = data[data["date"] <= training_end].copy()
    model = fit_price_model(
        prices=asof_prices,
        training_start=training_start,
        training_end=training_end,
        projection_years=projection_years,
    )
    daily = model.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.set_index("date")
    if training_end not in daily.index:
        raise ValueError("Frozen model did not contain the fake-today boundary.")

    c0 = float(daily.loc[training_end, "structural_centerline_usd"])
    actual_start_level = _smoothed_price_level(data, training_end)
    structural_rows = []
    for horizon in horizons:
        target = training_end + pd.DateOffset(months=horizon)
        actual_row = _nearest_price_on_or_before(data, target)
        eval_date = pd.Timestamp(actual_row["date"]).normalize()
        if eval_date not in daily.index:
            continue
        raw_center = float(daily.loc[eval_date, "structural_centerline_usd"])
        raw_growth = float(np.log(raw_center / c0))
        if abs(raw_growth) <= 0.01:
            continue
        actual_level = _smoothed_price_level(data, eval_date)
        actual_growth = float(np.log(actual_level / actual_start_level))
        implied_g = float(actual_growth / raw_growth)
        horizon_weight = float(horizon / 12.0)
        signal_weight = float(np.clip(abs(raw_growth) / 0.20, 0.50, 2.00))
        structural_rows.append({
            "metric_type": "structural",
            "fake_today": training_end,
            "lookback_years": lookback_years,
            "parent_start": parent_start,
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
            "evidence_weight": horizon_weight * signal_weight,
        })
    structural = pd.DataFrame(structural_rows)
    G_snapshot = _snapshot_growth_factor(structural)

    evaluation_end = min(training_end + pd.DateOffset(months=ENVELOPE_MAX_EVALUATION_MONTHS), latest)
    envelope_rows = []
    anchors = discover_observed_cycle_anchors(data)
    eligible_anchors = anchors[(pd.to_datetime(anchors["date"]) > training_end) & (pd.to_datetime(anchors["date"]) <= evaluation_end)].copy()
    eligible_anchors = eligible_anchors.sort_values("date").reset_index(drop=True)
    for ahead_idx, anchor in enumerate(eligible_anchors.itertuples(index=False), start=1):
        anchor_date = pd.Timestamp(anchor.date).normalize()
        anchor_type = str(anchor.type)
        if anchor_date not in daily.index:
            continue
        raw_center = float(daily.loc[anchor_date, "structural_centerline_usd"])
        raw_price = float(daily.loc[anchor_date, "fitted_or_projected_price_usd"])
        expected_sign = 1.0 if anchor_type == "peak" else -1.0
        raw_signed = expected_sign * np.log(max(raw_price, 1e-12) / max(raw_center, 1e-12))
        raw_amp = float(max(raw_signed, 0.0))
        if raw_amp <= 0.02:
            continue
        actual_anchor = _smoothed_anchor_price(data, anchor_date)
        months_forward = max((anchor_date - training_end).days / 30.4375, 0.0)
        evidence_weight = float(np.clip(months_forward / 12.0, 0.75, 4.0) * np.clip(raw_amp / 0.25, 0.5, 2.0))
        envelope_rows.append({
            "metric_type": "envelope",
            "fake_today": training_end,
            "lookback_years": lookback_years,
            "parent_start": parent_start,
            "training_start": training_start,
            "anchor_date": anchor_date,
            "anchor_type": anchor_type,
            "cycle": int(getattr(anchor, "cycle", 0)),
            "turning_point_ahead": int(ahead_idx),
            "cycle_horizon": int((ahead_idx + 1) // 2),
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
    K_snapshot = _snapshot_amplitude_factor(envelope)
    if not envelope.empty:
        envelope = envelope.copy()
        envelope["actual_amplitude_raw_centerline"] = envelope.apply(lambda r: _actual_anchor_amplitude(r, 1.0), axis=1)
        envelope["implied_amplitude_factor"] = np.where(
            envelope["raw_amplitude"] > 0,
            envelope["actual_amplitude_raw_centerline"] / envelope["raw_amplitude"],
            np.nan,
        )

    meta = {
        "fake_today": training_end,
        "lookback_years": lookback_years,
        "parent_start": parent_start,
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


def _build_backtest_snapshot(prices: pd.DataFrame, fake_today: pd.Timestamp, lookback_years: int):
    fake_today = pd.Timestamp(fake_today).normalize()
    return _build_backtest_snapshot_for_start(
        prices, fake_today, fake_today - pd.DateOffset(years=int(lookback_years)),
        lookback_years=int(lookback_years),
    )


def _fit_constant_growth(structural: pd.DataFrame) -> float:
    if structural.empty:
        return 1.0
    return _regularized_factor(
        structural["implied_growth_factor"], structural["evidence_weight"],
        lower=GROWTH_FACTOR_MIN, upper=GROWTH_FACTOR_MAX, prior_weight=GROWTH_PRIOR_WEIGHT,
    )


def _fit_constant_amplitude_from_tests(tests: pd.DataFrame) -> float:
    if tests.empty:
        return 1.0
    mask = tests["envelope_points"].to_numpy(dtype=float) > 0
    if not np.any(mask):
        return 1.0
    return _regularized_factor(
        tests.loc[mask, "snapshot_amplitude_factor"],
        np.maximum(tests.loc[mask, "envelope_evidence_weight"].to_numpy(dtype=float), 0.25),
        lower=AMPLITUDE_FACTOR_MIN, upper=AMPLITUDE_FACTOR_MAX, prior_weight=AMPLITUDE_PRIOR_WEIGHT,
    )


def _score_structural(structural: pd.DataFrame, growth_factor: float) -> tuple[float, float]:
    if structural.empty:
        return float("nan"), float("nan")
    y = structural["actual_structural_log_growth"].to_numpy(dtype=float)
    x = structural["raw_structural_log_growth"].to_numpy(dtype=float)
    w = structural["evidence_weight"].to_numpy(dtype=float)
    return _weighted_equivalent_pct_error(y - x, w), _weighted_equivalent_pct_error(y - growth_factor * x, w)


def _score_envelope(envelope: pd.DataFrame, amplitude_factor: float) -> tuple[float, float]:
    if envelope.empty:
        return float("nan"), float("nan")
    w = envelope["evidence_weight"].to_numpy(dtype=float)
    raw_pred = envelope["raw_amplitude"].to_numpy(dtype=float)
    actual = np.array([_actual_anchor_amplitude(r, 1.0) for _, r in envelope.iterrows()])
    return _weighted_equivalent_pct_error(actual - raw_pred, w), _weighted_equivalent_pct_error(actual - amplitude_factor * raw_pred, w)


def _cross_validate_structural(structural: pd.DataFrame) -> dict:
    if structural.empty:
        return {"raw": float("nan"), "calibrated": float("nan"), "holdouts": pd.DataFrame()}
    groups = sorted(pd.to_datetime(structural["fake_today"].dropna().unique()))
    rows = []
    for holdout in groups:
        train = structural[pd.to_datetime(structural["fake_today"]) != holdout]
        test = structural[pd.to_datetime(structural["fake_today"]) == holdout]
        if train.empty or test.empty:
            continue
        G = _fit_constant_growth(train)
        raw, cal = _score_structural(test, G)
        rows.append({"fake_today": pd.Timestamp(holdout), "cv_growth_factor": G, "raw_structural_error": raw, "calibrated_structural_error": cal})
    holdouts = pd.DataFrame(rows)
    if holdouts.empty:
        return {"raw": float("nan"), "calibrated": float("nan"), "holdouts": holdouts}
    return {
        "raw": float(np.nanmedian(holdouts["raw_structural_error"])),
        "calibrated": float(np.nanmedian(holdouts["calibrated_structural_error"])),
        "holdouts": holdouts,
    }


def _amplitude_cycle_points(envelope: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated forecasts into independent realized cycle-regime evidence.

    Multiple fake-today forecasts can target the same realized turning point.  They
    are useful forecast-origin observations, but they are not independent market
    cycles.  Trend fitting therefore first aggregates by realized anchor and then
    by cycle index so one 2025 peak cannot masquerade as many separate maturity
    observations.
    """
    if envelope is None or envelope.empty or "cycle" not in envelope.columns:
        return pd.DataFrame()
    work = envelope.copy()
    if "implied_amplitude_factor" not in work.columns:
        work["actual_amplitude_raw_centerline"] = work.apply(lambda r: _actual_anchor_amplitude(r, 1.0), axis=1)
        work["implied_amplitude_factor"] = np.where(
            work["raw_amplitude"].to_numpy(dtype=float) > 0,
            work["actual_amplitude_raw_centerline"].to_numpy(dtype=float) / work["raw_amplitude"].to_numpy(dtype=float),
            np.nan,
        )
    work = work[np.isfinite(work["implied_amplitude_factor"]) & (work["raw_amplitude"] > 0.02)].copy()
    if work.empty:
        return pd.DataFrame()

    anchor_rows = []
    for (cycle, anchor_date, anchor_type), grp in work.groupby(["cycle", "anchor_date", "anchor_type"], dropna=False):
        vals = grp["implied_amplitude_factor"].to_numpy(dtype=float)
        w = np.maximum(grp["evidence_weight"].to_numpy(dtype=float), 1e-9)
        anchor_rows.append({
            "cycle_index": int(cycle),
            "anchor_date": pd.Timestamp(anchor_date),
            "anchor_type": str(anchor_type),
            "factor": float(np.clip(_weighted_median(vals, w), AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX)),
            # Repeated fake-today predictions do not multiply independent-cycle
            # confidence.  The anchor gets one unit, mildly increased when both
            # long and short forecast origins agree.
            "weight": float(np.clip(np.sqrt(len(grp)), 1.0, 2.0)),
            "forecast_origins": int(len(grp)),
        })
    anchors = pd.DataFrame(anchor_rows)
    rows = []
    for cycle, grp in anchors.groupby("cycle_index"):
        vals = grp["factor"].to_numpy(dtype=float)
        w = grp["weight"].to_numpy(dtype=float)
        rows.append({
            "cycle_index": int(cycle),
            "date": pd.Timestamp(grp["anchor_date"].max()),
            "factor": float(_weighted_median(vals, w)),
            "weight": float(len(grp)),  # at most one independent vote per realized turning point
            "realized_turning_points": int(len(grp)),
            "forecast_origins": int(grp["forecast_origins"].sum()),
        })
    return pd.DataFrame(rows).sort_values("cycle_index").reset_index(drop=True)


def _amplitude_trend_points(tests: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible helper used by older tests/tools.

    v4 production trend fitting uses `_amplitude_cycle_points` instead.  When a
    caller passes date/factor observations directly this function still produces
    the previous fake-today aggregation shape.
    """
    if tests is None or tests.empty:
        return pd.DataFrame()
    if {"date", "factor", "weight"}.issubset(tests.columns):
        return tests[["date", "factor", "weight"]].copy().sort_values("date").reset_index(drop=True)
    rows = []
    valid = tests[tests.get("envelope_points", 0) > 0].copy() if "envelope_points" in tests.columns else pd.DataFrame()
    if valid.empty:
        return pd.DataFrame()
    for d, grp in valid.groupby(pd.to_datetime(valid["fake_today"])):
        vals = grp["snapshot_amplitude_factor"].to_numpy(dtype=float)
        weights = np.maximum(grp["envelope_evidence_weight"].to_numpy(dtype=float), 0.25)
        rows.append({"date": pd.Timestamp(d), "factor": _weighted_median(vals, weights), "weight": float(np.sum(weights))})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _fit_amplitude_trend(points: pd.DataFrame) -> dict:
    """Fit a conservatively shrunk maturity trend.

    Production points use Bitcoin cycle index rather than calendar time.  This
    prevents a -15%/year rule from compounding four times inside one market cycle
    and, more importantly, bases confidence on independent realized cycles.
    """
    if points is None or points.empty:
        return {
            "mode": "CONSTANT", "direction": "NO_CLEAR_TREND", "confidence": 0.0,
            "r2": float("nan"), "raw_log_slope_per_cycle": 0.0, "effective_log_slope_per_cycle": 0.0,
            "center_cycle": 0.0, "center_log_factor": 0.0, "constant_factor": 1.0, "n_eff": 0.0,
            "change_per_cycle": 0.0,
        }
    p = points.copy()
    factor = np.clip(p["factor"].to_numpy(dtype=float), AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX)
    w = np.maximum(p["weight"].to_numpy(dtype=float), 1e-9)
    if "cycle_index" in p.columns:
        x = p["cycle_index"].to_numpy(dtype=float)
    else:
        dates = pd.to_datetime(p["date"])
        origin = pd.Timestamp(dates.min())
        # Compatibility conversion: one x-unit is one fixed Bitcoin cycle.
        x = np.array([(pd.Timestamp(d) - origin).days / FIXED_CYCLE_DAYS for d in dates], dtype=float)
    y = np.log(factor)
    xbar = float(np.average(x, weights=w))
    ybar = float(np.average(y, weights=w))
    denom = float(np.sum(w * (x - xbar) ** 2))
    slope = float(np.sum(w * (x - xbar) * (y - ybar)) / denom) if denom > 1e-12 else 0.0
    pred = ybar + slope * (x - xbar)
    sse = float(np.sum(w * (y - pred) ** 2))
    sst = float(np.sum(w * (y - ybar) ** 2))
    r2 = float(np.clip(1.0 - sse / sst, 0.0, 1.0)) if sst > 1e-12 else 0.0
    n_eff = float((w.sum() ** 2) / max(np.sum(w ** 2), 1e-12))
    # With only 2-3 independent mature cycles, trend confidence must remain
    # modest even when R² happens to look excellent.
    confidence = float(np.clip((n_eff / (n_eff + 4.0)) * r2, 0.0, 1.0))
    effective_slope = slope * confidence
    constant_factor = _regularized_factor(
        factor, w, lower=AMPLITUDE_FACTOR_MIN, upper=AMPLITUDE_FACTOR_MAX, prior_weight=AMPLITUDE_PRIOR_WEIGHT
    )
    if confidence < 0.10 or abs(effective_slope) < 1e-4:
        direction = "NO_CLEAR_TREND"
    else:
        direction = "DECLINING" if effective_slope < 0 else "INCREASING"
    return {
        "mode": "TREND",
        "direction": direction,
        "confidence": confidence,
        "r2": r2,
        "raw_log_slope_per_cycle": slope,
        "effective_log_slope_per_cycle": effective_slope,
        "change_per_cycle": float(np.expm1(effective_slope)),
        "center_cycle": float(xbar),
        "center_log_factor": float(ybar),
        "constant_factor": float(constant_factor),
        "n_eff": n_eff,
    }


def _trend_factor_at_cycle(trend: dict, cycle_index: float) -> float:
    log_factor = float(trend.get("center_log_factor", np.log(max(trend.get("constant_factor", 1.0), 1e-12))))
    log_factor += float(trend.get("effective_log_slope_per_cycle", 0.0)) * (float(cycle_index) - float(trend.get("center_cycle", 0.0)))
    return float(np.clip(np.exp(log_factor), AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX))


def _trend_factor_at_date(trend: dict, date: pd.Timestamp) -> float:
    """Compatibility wrapper for older tests.

    If a date-based center is supplied, convert elapsed time to fixed-cycle units;
    v4 production forecasts call `_trend_factor_at_cycle` directly.
    """
    if trend.get("center_cycle") is not None:
        if trend.get("center_date") is not None:
            cycle = float(trend.get("center_cycle", 0.0)) + (pd.Timestamp(date) - pd.Timestamp(trend["center_date"])).days / FIXED_CYCLE_DAYS
        else:
            cycle = float(trend.get("center_cycle", 0.0))
        return _trend_factor_at_cycle(trend, cycle)
    if trend.get("center_date") is None:
        return float(trend.get("constant_factor", 1.0))
    cycles = (pd.Timestamp(date) - pd.Timestamp(trend["center_date"])).days / FIXED_CYCLE_DAYS
    slope_cycle = float(trend.get("effective_log_slope_per_cycle", 0.0))
    return float(np.clip(np.exp(float(trend.get("center_log_factor", 0.0)) + slope_cycle * cycles), AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX))


def _fit_constant_amplitude_from_cycle_points(points: pd.DataFrame) -> float:
    if points is None or points.empty:
        return 1.0
    return _regularized_factor(
        points["factor"].to_numpy(dtype=float), points["weight"].to_numpy(dtype=float),
        lower=AMPLITUDE_FACTOR_MIN, upper=AMPLITUDE_FACTOR_MAX, prior_weight=AMPLITUDE_PRIOR_WEIGHT,
    )


def _cross_validate_structural_blend(structural: pd.DataFrame) -> dict:
    """Learn how much of G to trust using held-out fake-today forecasts."""
    if structural is None or structural.empty:
        return {"raw": float("nan"), "calibrated": float("nan"), "blend_weight": 0.0, "holdouts": pd.DataFrame()}
    groups = sorted(pd.to_datetime(structural["fake_today"].dropna().unique()))
    folds = []
    for holdout in groups:
        train = structural[pd.to_datetime(structural["fake_today"]) != holdout]
        test = structural[pd.to_datetime(structural["fake_today"]) == holdout]
        if train.empty or test.empty:
            continue
        folds.append((pd.Timestamp(holdout), _fit_constant_growth(train), test))
    if not folds:
        return {"raw": float("nan"), "calibrated": float("nan"), "blend_weight": 0.0, "holdouts": pd.DataFrame()}

    alphas = np.linspace(0.0, 1.0, 101)
    scores = []
    for alpha in alphas:
        fold_errors = []
        for _, g_train, test in folds:
            g_eff = 1.0 + float(alpha) * (float(g_train) - 1.0)
            _, cal = _score_structural(test, g_eff)
            if np.isfinite(cal):
                fold_errors.append(cal)
        scores.append(float(np.median(fold_errors)) if fold_errors else float("inf"))
    best_i = int(np.argmin(scores))
    alpha = float(alphas[best_i])
    rows, raw_errors, cal_errors = [], [], []
    for holdout, g_train, test in folds:
        g_eff = 1.0 + alpha * (float(g_train) - 1.0)
        raw, cal = _score_structural(test, g_eff)
        raw_errors.append(raw); cal_errors.append(cal)
        rows.append({
            "fake_today": holdout, "cv_growth_factor": float(g_train), "cv_structural_blend_weight": alpha,
            "cv_effective_growth_factor": g_eff, "raw_structural_error": raw, "calibrated_structural_error": cal,
        })
    return {
        "raw": float(np.nanmedian(raw_errors)), "calibrated": float(np.nanmedian(cal_errors)),
        "blend_weight": alpha, "holdouts": pd.DataFrame(rows),
    }


def _cross_validate_amplitude_blend(cycle_points: pd.DataFrame, envelope: pd.DataFrame) -> dict:
    """Choose constant-vs-trend blend by leaving entire realized cycles out.

    This is deliberately stricter than fake-today holdouts: all forecasts that
    target the same realized market cycle are withheld together.
    """
    if cycle_points is None or cycle_points.empty or envelope is None or envelope.empty:
        return {"raw": float("nan"), "calibrated": float("nan"), "blend_weight": 0.0, "holdouts": pd.DataFrame(), "observation_scores": pd.DataFrame()}
    cycles = sorted(int(c) for c in cycle_points["cycle_index"].unique())
    if len(cycles) < 2:
        return {"raw": float("nan"), "calibrated": float("nan"), "blend_weight": 0.0, "holdouts": pd.DataFrame(), "observation_scores": pd.DataFrame()}

    fold_specs = []
    for hold in cycles:
        train_points = cycle_points[cycle_points["cycle_index"] != hold]
        test_env = envelope[envelope["cycle"].astype(int) == hold]
        if train_points.empty or test_env.empty:
            continue
        const = _fit_constant_amplitude_from_cycle_points(train_points)
        trend = _fit_amplitude_trend(train_points)
        trend_k = _trend_factor_at_cycle(trend, hold)
        fold_specs.append((hold, const, trend_k, test_env))
    if not fold_specs:
        return {"raw": float("nan"), "calibrated": float("nan"), "blend_weight": 0.0, "holdouts": pd.DataFrame(), "observation_scores": pd.DataFrame()}

    alphas = np.linspace(0.0, 1.0, 101)
    alpha_scores = []
    for alpha in alphas:
        errs = []
        for _, const, trend_k, test_env in fold_specs:
            k = (1.0 - alpha) * const + alpha * trend_k
            _, cal = _score_envelope(test_env, k)
            if np.isfinite(cal): errs.append(cal)
        alpha_scores.append(float(np.median(errs)) if errs else float("inf"))
    best_i = int(np.argmin(alpha_scores))
    alpha = float(alphas[best_i])

    hold_rows, obs_rows, raw_errors, cal_errors = [], [], [], []
    for hold, const, trend_k, test_env in fold_specs:
        k = float(np.clip((1.0 - alpha) * const + alpha * trend_k, AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX))
        raw, cal = _score_envelope(test_env, k)
        raw_errors.append(raw); cal_errors.append(cal)
        hold_rows.append({
            "cycle": hold, "cv_constant_K": const, "cv_trend_K": trend_k,
            "cv_amplitude_blend_weight": alpha, "cv_amplitude_factor": k,
            "raw_envelope_error": raw, "calibrated_envelope_error": cal,
        })
        for _, r in test_env.iterrows():
            actual = _actual_anchor_amplitude(r, 1.0)
            raw_amp = float(r["raw_amplitude"])
            obs_rows.append({
                "cycle": hold,
                "fake_today": pd.Timestamp(r["fake_today"]),
                "anchor_date": pd.Timestamp(r["anchor_date"]),
                "anchor_type": str(r["anchor_type"]),
                "turning_point_ahead": int(r.get("turning_point_ahead", 0) or 0),
                "cycle_horizon": int(r.get("cycle_horizon", 0) or 0),
                "evidence_weight": float(r["evidence_weight"]),
                "raw_abs_log_error": abs(actual - raw_amp),
                "calibrated_abs_log_error": abs(actual - k * raw_amp),
                "cv_K": k,
            })
    return {
        "raw": float(np.nanmedian(raw_errors)), "calibrated": float(np.nanmedian(cal_errors)),
        "blend_weight": alpha, "holdouts": pd.DataFrame(hold_rows), "observation_scores": pd.DataFrame(obs_rows),
    }


def _direct_cycle_validation(observation_scores: pd.DataFrame) -> list[dict]:
    if observation_scores is None or observation_scores.empty:
        return []
    rows = []
    for horizon, grp in observation_scores.groupby("cycle_horizon"):
        if int(horizon) <= 0:
            continue
        w = grp["evidence_weight"].to_numpy(dtype=float)
        raw = _weighted_equivalent_pct_error(grp["raw_abs_log_error"].to_numpy(dtype=float), w)
        cal = _weighted_equivalent_pct_error(grp["calibrated_abs_log_error"].to_numpy(dtype=float), w)
        rows.append({
            "cycle_horizon": int(horizon), "observations": int(len(grp)),
            "raw_error": raw, "calibrated_error": cal,
            "improvement": float(1.0 - cal / raw) if np.isfinite(raw) and raw > 0 and np.isfinite(cal) else float("nan"),
        })
    return sorted(rows, key=lambda r: r["cycle_horizon"])


def _cross_validate_amplitude_mode(tests: pd.DataFrame, envelope: pd.DataFrame, mode: str) -> dict:
    """Compatibility shim retained for older tests.

    v4 production uses cycle-level blended CV.  Constant mode still maps to the
    older fake-today implementation when explicitly requested.
    """
    if mode == "TREND":
        points = _amplitude_cycle_points(envelope)
        return _cross_validate_amplitude_blend(points, envelope)
    if tests is None or tests.empty or envelope is None or envelope.empty:
        return {"raw": float("nan"), "calibrated": float("nan"), "holdouts": pd.DataFrame()}
    rows = []
    for holdout in sorted(pd.to_datetime(tests["fake_today"].unique())):
        train_tests = tests[pd.to_datetime(tests["fake_today"]) != holdout]
        test_env = envelope[pd.to_datetime(envelope["fake_today"]) == holdout]
        if train_tests.empty or test_env.empty: continue
        k = _fit_constant_amplitude_from_tests(train_tests)
        raw, cal = _score_envelope(test_env, k)
        rows.append({"fake_today": pd.Timestamp(holdout), "cv_amplitude_factor": k, "raw_envelope_error": raw, "calibrated_envelope_error": cal})
    holdouts = pd.DataFrame(rows)
    return {
        "raw": float(np.nanmedian(holdouts["raw_envelope_error"])) if not holdouts.empty else float("nan"),
        "calibrated": float(np.nanmedian(holdouts["calibrated_envelope_error"])) if not holdouts.empty else float("nan"),
        "holdouts": holdouts,
    }

def _component_status(raw_error: float, calibrated_error: float) -> tuple[str, float]:
    if not np.isfinite(raw_error) or not np.isfinite(calibrated_error) or raw_error <= 0:
        return "INSUFFICIENT_EVIDENCE", float("nan")
    improvement = float(1.0 - calibrated_error / raw_error)
    if improvement >= 0.10:
        return "PASS", improvement
    if improvement > 0:
        return "MODEST", improvement
    return "REJECTED", improvement


def _score_cycle_aligned_parents(prices: pd.DataFrame, progress_callback=None, progress_offset=0, progress_total=None):
    data = _normalise_prices(prices)
    latest = pd.Timestamp(data["date"].max()).normalize()
    latest_eligible = latest - pd.DateOffset(months=12)
    starts = discover_cycle_aligned_parent_starts(data)
    rows = []
    all_parent_obs = []
    done = progress_offset

    for start in starts:
        first = pd.Timestamp(start) + pd.DateOffset(years=PARENT_MIN_TRAIN_YEARS)
        cursor = first
        structural_frames, envelope_frames = [], []
        tests = 0
        while cursor <= latest_eligible:
            fake = pd.Timestamp(_nearest_price_on_or_before(data, cursor)["date"]).normalize()
            try:
                st, env, _ = _build_backtest_snapshot_for_start(data, fake, start, parent_start=start)
                if not st.empty: structural_frames.append(st)
                if not env.empty: envelope_frames.append(env)
                tests += 1
            except Exception:
                pass
            done += 1
            if progress_callback is not None and progress_total:
                progress_callback(done, progress_total, f"Parent {start.date()} as-of {fake.date()}")
            cursor = cursor + pd.DateOffset(months=PARENT_TEST_STEP_MONTHS)

        structural = pd.concat(structural_frames, ignore_index=True) if structural_frames else pd.DataFrame()
        envelope = pd.concat(envelope_frames, ignore_index=True) if envelope_frames else pd.DataFrame()
        raw_s, _ = _score_structural(structural, 1.0)
        raw_e, _ = _score_envelope(envelope, 1.0)
        parts = [v for v in (raw_s, raw_e) if np.isfinite(v)]
        raw_error = float(np.mean(parts)) if parts else float("nan")
        confidence = float(tests / (tests + 2.0)) if tests > 0 else 0.0
        rows.append({
            "start_date": pd.Timestamp(start),
            "parent_age_years": float((latest - pd.Timestamp(start)).days / 365.25),
            "tests": int(tests),
            "structural_points": int(len(structural)),
            "envelope_points": int(len(envelope)),
            "raw_structural_error": raw_s,
            "raw_envelope_error": raw_e,
            "raw_total_error": raw_error,
            "evidence_confidence": confidence,
            "weight_source": "own OOS evidence" if tests > 0 and np.isfinite(raw_error) else "pending maturity-matched prior",
        })
        if not structural.empty:
            tmp = structural.copy(); tmp["parent_candidate"] = pd.Timestamp(start); all_parent_obs.append(tmp)
        if not envelope.empty:
            tmp = envelope.copy(); tmp["parent_candidate"] = pd.Timestamp(start); all_parent_obs.append(tmp)

    table = pd.DataFrame(rows)
    if table.empty:
        return table, pd.DataFrame(), done
    obs = pd.concat(all_parent_obs, ignore_index=True, sort=False) if all_parent_obs else pd.DataFrame()

    # New cycle parents often have not yet accumulated a full 12-month scored
    # forecast.  Instead of assigning them zero influence, borrow a small prior
    # from how older parents performed at the same model age.  As soon as the
    # new parent owns OOS evidence, that evidence replaces the prior naturally.
    prior_errors = []
    prior_confidences = []
    for idx, row in table.iterrows():
        own_error = float(row["raw_total_error"]) if np.isfinite(row["raw_total_error"]) else float("nan")
        own_conf = float(row["evidence_confidence"])
        if np.isfinite(own_error) and own_conf > 0:
            prior_errors.append(own_error)
            prior_confidences.append(own_conf)
            continue

        start = pd.Timestamp(row["start_date"])
        age_years = float(row["parent_age_years"])
        analog_errors = []
        older_starts = [pd.Timestamp(x) for x in table.loc[table["start_date"] < start, "start_date"]]
        for older in older_starts:
            cutoff = older + pd.Timedelta(days=age_years * 365.25)
            sgrp = obs[(obs.get("parent_candidate") == older) & (obs.get("metric_type") == "structural")].copy() if not obs.empty else pd.DataFrame()
            egrp = obs[(obs.get("parent_candidate") == older) & (obs.get("metric_type") == "envelope")].copy() if not obs.empty else pd.DataFrame()
            if not sgrp.empty and "fake_today" in sgrp.columns:
                sgrp = sgrp[pd.to_datetime(sgrp["fake_today"]) <= cutoff]
            if not egrp.empty and "fake_today" in egrp.columns:
                egrp = egrp[pd.to_datetime(egrp["fake_today"]) <= cutoff]
            rs, _ = _score_structural(sgrp, 1.0)
            re, _ = _score_envelope(egrp, 1.0)
            vals = [v for v in (rs, re) if np.isfinite(v)]
            if vals: analog_errors.append(float(np.mean(vals)))
        if analog_errors:
            prior_error = float(np.median(analog_errors))
            analog_count = len(analog_errors)
            maturity_fraction = float(np.clip(age_years / (ENVELOPE_MAX_EVALUATION_MONTHS / 12.0), 0.0, 1.0))
            prior_conf = float((analog_count / (analog_count + 2.0)) * maturity_fraction)
            table.at[idx, "raw_total_error"] = prior_error
            table.at[idx, "evidence_confidence"] = prior_conf
            table.at[idx, "weight_source"] = "maturity-matched historical prior"
            prior_errors.append(prior_error)
            prior_confidences.append(prior_conf)
        else:
            table.at[idx, "evidence_confidence"] = 0.0
            table.at[idx, "weight_source"] = "insufficient parent evidence"

    raw_weights = []
    for row in table.itertuples(index=False):
        err = float(row.raw_total_error) if np.isfinite(row.raw_total_error) else float("nan")
        conf = float(row.evidence_confidence)
        raw_weights.append(conf / max(err, 0.05) if np.isfinite(err) and conf > 0 else 0.0)
    raw_weights = np.asarray(raw_weights, dtype=float)
    if raw_weights.sum() <= 0:
        raw_weights = np.ones(len(table), dtype=float)
    table["raw_weight"] = raw_weights
    table["weight"] = raw_weights / raw_weights.sum()
    return table, obs, done

def _lookback_summary(tests: pd.DataFrame, structural: pd.DataFrame, envelope: pd.DataFrame, lookback: int) -> dict:
    t = tests[tests["lookback_years"] == lookback].copy()
    s = structural[structural["lookback_years"] == lookback].copy() if not structural.empty else structural
    e = envelope[envelope["lookback_years"] == lookback].copy() if not envelope.empty else envelope
    G = _fit_constant_growth(s)
    K = _fit_constant_amplitude_from_tests(t)
    raw_s, cal_s = _score_structural(s, G)
    raw_e, cal_e = _score_envelope(e, K)
    return {
        "growth_factor": G, "amplitude_factor": K,
        "raw_structural_cv_error": raw_s, "calibrated_structural_cv_error": cal_s,
        "raw_envelope_cv_error": raw_e, "calibrated_envelope_cv_error": cal_e,
        "tests": int(len(t)), "structural_points": int(len(s)), "envelope_points": int(len(e)),
        "first_fake_today": pd.Timestamp(t["fake_today"].min()).date().isoformat() if not t.empty else None,
        "last_fake_today": pd.Timestamp(t["fake_today"].max()).date().isoformat() if not t.empty else None,
    }


def _fingerprint(summary: dict, tests: pd.DataFrame, latest_data_date: pd.Timestamp) -> str:
    payload = {
        "version": CALIBRATION_VERSION,
        "price_model_engine": PRICE_MODEL_ENGINE_VERSION,
        "latest_data_date": pd.Timestamp(latest_data_date).date().isoformat(),
        "floor": CALIBRATION_FLOOR.date().isoformat(),
        "effective_G": round(float(summary.get("effective_growth_factor", 1.0)), 10),
        "K": round(float(summary.get("amplitude_factor", 1.0)), 10),
        "K_mode": summary.get("amplitude_mode"),
        "K_slope_cycle": round(float(summary.get("amplitude_trend_effective_log_slope_per_cycle", 0.0)), 12),
        "K_blend": round(float(summary.get("amplitude_trend_blend_weight", 0.0)), 8),
        "G_blend": round(float(summary.get("structural_blend_weight", 0.0)), 8),
        "parents": [(p.get("start_date"), round(float(p.get("weight", 0.0)), 8)) for p in summary.get("cycle_parents", [])],
        "status": str(summary.get("status", "UNKNOWN")),
        "tests": [
            {"fake_today": pd.Timestamp(r.fake_today).date().isoformat(), "lookback": int(r.lookback_years), "max_horizon": int(r.max_horizon_months)}
            for r in tests[["fake_today", "lookback_years", "max_horizon_months"]].itertuples(index=False)
        ] if not tests.empty else [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def run_walk_forward_calibration(prices: pd.DataFrame, progress_callback: Callable[[int, int, str], None] | None = None) -> WalkForwardCalibrationResult:
    data = _normalise_prices(prices)
    latest = pd.Timestamp(data["date"].max()).normalize()

    schedule = [(lb, d) for lb in LOOKBACK_YEARS for d in generate_fake_today_dates(data, lb)]
    if not schedule:
        raise ValueError("Not enough post-Jan-2015 history exists for walk-forward calibration.")

    parent_starts = discover_cycle_aligned_parent_starts(data)
    latest_eligible = latest - pd.DateOffset(months=12)
    parent_test_count = 0
    for start_date in parent_starts:
        cursor = pd.Timestamp(start_date) + pd.DateOffset(years=PARENT_MIN_TRAIN_YEARS)
        while cursor <= latest_eligible:
            parent_test_count += 1
            cursor = cursor + pd.DateOffset(months=PARENT_TEST_STEP_MONTHS)
    progress_total = len(schedule) + parent_test_count

    structural_frames, envelope_frames, test_meta = [], [], []
    done = 0
    for lookback, fake_today in schedule:
        structural_part, envelope_part, meta = _build_backtest_snapshot(data, fake_today, lookback)
        if not structural_part.empty: structural_frames.append(structural_part)
        if not envelope_part.empty: envelope_frames.append(envelope_part)
        test_meta.append(meta)
        done += 1
        if progress_callback is not None:
            progress_callback(done, progress_total, f"{lookback}Y as-of {pd.Timestamp(fake_today).date()}")

    structural = pd.concat(structural_frames, ignore_index=True) if structural_frames else pd.DataFrame()
    envelope = pd.concat(envelope_frames, ignore_index=True) if envelope_frames else pd.DataFrame()
    tests = pd.DataFrame(test_meta).sort_values(["lookback_years", "fake_today"]).reset_index(drop=True)

    # G is no longer all-or-nothing.  Held-out fake-today forecasts learn how
    # much of the structural correction to trust, so weak evidence only nudges
    # the ensemble centerline rather than applying the entire fitted G.
    learned_G = _fit_constant_growth(structural)
    struct_cv = _cross_validate_structural_blend(structural)
    structural_blend = float(struct_cv.get("blend_weight", 0.0))
    effective_G = float(1.0 + structural_blend * (learned_G - 1.0))
    growth_status, growth_improvement = _component_status(struct_cv["raw"], struct_cv["calibrated"])
    if growth_status == "REJECTED":
        structural_blend = 0.0
        effective_G = 1.0
        struct_cv = dict(struct_cv)
        struct_cv["calibrated"] = struct_cv["raw"]

    # Parent models are scored before the maturity trend so their older
    # cycle-aligned fake-today forecasts can contribute independent cycle
    # outcomes (including evidence earlier than the rolling 4Y schedule).
    parent_table, parent_obs, done = _score_cycle_aligned_parents(
        data, progress_callback=progress_callback, progress_offset=done, progress_total=progress_total
    )
    if parent_table.empty:
        raise ValueError("No cycle-aligned parent models could be scored.")

    parent_envelope = pd.DataFrame()
    if parent_obs is not None and not parent_obs.empty and "metric_type" in parent_obs.columns:
        parent_envelope = parent_obs[parent_obs["metric_type"] == "envelope"].copy()
    amplitude_evidence = pd.concat(
        [x for x in (envelope, parent_envelope) if x is not None and not x.empty],
        ignore_index=True, sort=False,
    ) if (not envelope.empty or not parent_envelope.empty) else pd.DataFrame()

    # Trend fitting now operates on independent Bitcoin cycle indices, not
    # calendar years.  Repeated forecasts of the same realized anchor are first
    # collapsed, preventing pseudo-replication from inflating trend confidence.
    cycle_points = _amplitude_cycle_points(amplitude_evidence)
    trend_fit = _fit_amplitude_trend(cycle_points)
    constant_K = _fit_constant_amplitude_from_cycle_points(cycle_points)
    amp_cv = _cross_validate_amplitude_blend(cycle_points, amplitude_evidence)
    amplitude_blend = float(amp_cv.get("blend_weight", 0.0))
    envelope_status, envelope_improvement = _component_status(amp_cv["raw"], amp_cv["calibrated"])
    if envelope_status == "REJECTED":
        amplitude_blend = 0.0
        constant_K = 1.0

    observed_anchors = discover_observed_cycle_anchors(data)
    current_cycle_index = int(observed_anchors["cycle"].max()) if not observed_anchors.empty else 0
    trend_K_current = _trend_factor_at_cycle(trend_fit, current_cycle_index)
    current_K = float(np.clip(
        (1.0 - amplitude_blend) * constant_K + amplitude_blend * trend_K_current,
        AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX,
    ))
    if envelope_status == "REJECTED":
        current_K = 1.0

    if amplitude_blend > 0.05 and trend_fit.get("direction") != "NO_CLEAR_TREND" and envelope_status in ("PASS", "MODEST"):
        amplitude_mode = "BLENDED_TREND"
    elif envelope_status in ("PASS", "MODEST"):
        amplitude_mode = "CONSTANT"
    else:
        amplitude_mode = "REJECTED"

    direct_cycle_validation = _direct_cycle_validation(amp_cv.get("observation_scores", pd.DataFrame()))

    enough_parents = len(parent_table) >= 2
    if envelope_status == "PASS" and enough_parents:
        status = "PASS"
    elif envelope_status == "MODEST" and enough_parents:
        status = "MODEST"
    elif envelope_status == "INSUFFICIENT_EVIDENCE":
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "NO_IMPROVEMENT"

    raw_struct = float(struct_cv["raw"])
    cal_struct = float(struct_cv["calibrated"])
    raw_env = float(amp_cv["raw"])
    cal_env = float(amp_cv["calibrated"])
    raw_parts = [x for x in (raw_struct, raw_env) if np.isfinite(x)]
    cal_parts = [x for x in (cal_struct, cal_env) if np.isfinite(x)]
    raw_total = float(np.mean(raw_parts)) if raw_parts else float("nan")
    cal_total = float(np.mean(cal_parts)) if cal_parts else float("nan")
    overall_improvement = float(1.0 - cal_total / raw_total) if np.isfinite(raw_total) and raw_total > 0 and np.isfinite(cal_total) else float("nan")

    tests_out = tests.copy()
    if not struct_cv.get("holdouts", pd.DataFrame()).empty:
        tests_out = tests_out.merge(struct_cv["holdouts"], on="fake_today", how="left")

    parent_records = []
    for row in parent_table.itertuples(index=False):
        parent_records.append({
            "start_date": pd.Timestamp(row.start_date).date().isoformat(),
            "parent_age_years": float(row.parent_age_years),
            "tests": int(row.tests),
            "structural_points": int(row.structural_points),
            "envelope_points": int(row.envelope_points),
            "raw_structural_error": float(row.raw_structural_error) if np.isfinite(row.raw_structural_error) else None,
            "raw_envelope_error": float(row.raw_envelope_error) if np.isfinite(row.raw_envelope_error) else None,
            "raw_total_error": float(row.raw_total_error) if np.isfinite(row.raw_total_error) else None,
            "evidence_confidence": float(row.evidence_confidence),
            "weight_source": str(row.weight_source),
            "weight": float(row.weight),
        })

    summary = {
        "version": CALIBRATION_VERSION,
        "price_model_engine_version": PRICE_MODEL_ENGINE_VERSION,
        "calibration_floor": CALIBRATION_FLOOR.date().isoformat(),
        "fake_today_step_months": FAKE_TODAY_STEP_MONTHS,
        "evaluation_horizons_months": list(EVALUATION_HORIZONS_MONTHS),
        "max_evaluation_months": MAX_EVALUATION_MONTHS,
        "envelope_max_evaluation_months": ENVELOPE_MAX_EVALUATION_MONTHS,
        "growth_factor": float(learned_G),
        "effective_growth_factor": float(effective_G),
        "structural_blend_weight": float(structural_blend),
        "growth_status": growth_status,
        "growth_cv_improvement": growth_improvement,
        "amplitude_factor": float(current_K),
        "amplitude_constant_factor": float(constant_K),
        "amplitude_trend_blend_weight": float(amplitude_blend),
        "amplitude_mode": amplitude_mode,
        "amplitude_trend_direction": trend_fit.get("direction", "NO_CLEAR_TREND"),
        "amplitude_trend_confidence": float(trend_fit.get("confidence", 0.0)),
        "amplitude_trend_r2": float(trend_fit.get("r2", np.nan)),
        "amplitude_trend_raw_log_slope_per_cycle": float(trend_fit.get("raw_log_slope_per_cycle", 0.0)),
        "amplitude_trend_effective_log_slope_per_cycle": float(trend_fit.get("effective_log_slope_per_cycle", 0.0)),
        "amplitude_trend_change_per_cycle": float(trend_fit.get("change_per_cycle", 0.0)),
        "amplitude_trend_center_cycle": float(trend_fit.get("center_cycle", 0.0)),
        "amplitude_trend_center_log_factor": float(trend_fit.get("center_log_factor", 0.0)),
        "current_cycle_index": int(current_cycle_index),
        "envelope_status": envelope_status,
        "envelope_cv_improvement": envelope_improvement,
        "geometry_guard_enabled": True,
        "direct_cycle_validation": direct_cycle_validation,
        "raw_cv_error": raw_total,
        "calibrated_cv_error": cal_total,
        "raw_structural_cv_error": raw_struct,
        "calibrated_structural_cv_error": cal_struct,
        "raw_envelope_cv_error": raw_env,
        "calibrated_envelope_cv_error": cal_env,
        "cv_improvement": overall_improvement,
        "status": status,
        "total_tests": int(len(tests)),
        "total_structural_points": int(len(structural)),
        "total_envelope_points": int(len(amplitude_evidence)),
        "independent_cycle_points": int(len(cycle_points)),
        "lookback_4y": _lookback_summary(tests, structural, envelope, 4),
        "lookback_8y": _lookback_summary(tests, structural, envelope, 8),
        "cycle_parents": parent_records,
        "cycle_parent_count": int(len(parent_records)),
        "latest_data_date": latest.date().isoformat(),
        "timing_source": "frozen v3.12 fixed 1428-day cycle schedule; calibration learns price levels and cycle-regime maturation, not future turning-point dates",
    }

    trend_obs = cycle_points.copy()
    if not trend_obs.empty:
        trend_obs["metric_type"] = "amplitude_cycle_trend"
        trend_obs["constant_factor"] = float(constant_K)
        trend_obs["trend_factor"] = trend_obs["cycle_index"].apply(lambda c: _trend_factor_at_cycle(trend_fit, c))
        trend_obs["predicted_factor"] = (
            (1.0 - amplitude_blend) * trend_obs["constant_factor"] + amplitude_blend * trend_obs["trend_factor"]
        )

    cycle_cv_obs = amp_cv.get("observation_scores", pd.DataFrame()).copy()
    if not cycle_cv_obs.empty:
        cycle_cv_obs["metric_type"] = "cycle_relative_validation"

    observations = pd.concat(
        [x for x in (structural, envelope, trend_obs, parent_obs, cycle_cv_obs) if x is not None and not x.empty],
        ignore_index=True, sort=False,
    ) if any(x is not None and not x.empty for x in (structural, envelope, trend_obs, parent_obs, cycle_cv_obs)) else pd.DataFrame()

    fingerprint = _fingerprint(summary, tests_out, latest)
    summary["fingerprint"] = fingerprint
    tests_out["final_effective_growth_factor"] = effective_G
    tests_out["final_structural_blend_weight"] = structural_blend
    tests_out["final_current_amplitude_factor"] = current_K
    tests_out["final_amplitude_trend_blend_weight"] = amplitude_blend
    return WalkForwardCalibrationResult(summary, tests_out, observations, fingerprint)

def calibration_is_current(calibration: WalkForwardCalibrationResult | None, prices: pd.DataFrame) -> bool:
    if calibration is None:
        return False
    summary = getattr(calibration, "summary", {})
    return (
        summary.get("version") == CALIBRATION_VERSION
        and summary.get("price_model_engine_version") == PRICE_MODEL_ENGINE_VERSION
        and summary.get("latest_data_date") == pd.Timestamp(prices["date"].max()).date().isoformat()
        and REQUIRED_SUMMARY_KEYS.issubset(summary.keys())
    )


def _calibrated_centerline_from_raw(raw_center: np.ndarray, dates: pd.DatetimeIndex, start_date: pd.Timestamp, growth_factor: float) -> np.ndarray:
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


def _summary_trend_dict(summary: dict) -> dict:
    return {
        "center_cycle": float(summary.get("amplitude_trend_center_cycle", 0.0)),
        "center_log_factor": float(summary.get("amplitude_trend_center_log_factor", np.log(max(summary.get("amplitude_constant_factor", 1.0), 1e-12)))),
        "effective_log_slope_per_cycle": float(summary.get("amplitude_trend_effective_log_slope_per_cycle", 0.0)),
        "constant_factor": float(summary.get("amplitude_constant_factor", summary.get("amplitude_factor", 1.0))),
        "blend_weight": float(summary.get("amplitude_trend_blend_weight", 0.0)),
    }


def _blended_amplitude_factor_for_cycle(trend: dict, cycle_index: float) -> float:
    const = float(trend.get("constant_factor", 1.0))
    alpha = float(np.clip(trend.get("blend_weight", 0.0), 0.0, 1.0))
    trended = _trend_factor_at_cycle(trend, cycle_index)
    return float(np.clip((1.0 - alpha) * const + alpha * trended, AMPLITUDE_FACTOR_MIN, AMPLITUDE_FACTOR_MAX))

def _fit_current_parent_models(prices: pd.DataFrame, calibration: WalkForwardCalibrationResult, projection_years: int):
    data = _normalise_prices(prices)
    latest = pd.Timestamp(data["date"].max()).normalize()
    parent_rows = calibration.summary.get("cycle_parents", [])
    models = []
    for parent in parent_rows:
        start = pd.Timestamp(parent["start_date"]).normalize()
        weight = float(parent.get("weight", 0.0))
        if weight <= 0 or start >= latest:
            continue
        try:
            model = fit_price_model(data, start, latest, int(projection_years))
            models.append((start, weight, model))
        except Exception:
            continue
    if not models:
        raise ValueError("No cycle-aligned parent model could be fitted to current Bitcoin data.")
    weights = np.asarray([m[1] for m in models], dtype=float)
    weights = weights / weights.sum()
    return [(m[0], float(w), m[2]) for m, w in zip(models, weights)]


def _collect_model_anchors(model: PriceModelResult) -> pd.DataFrame:
    frames = []
    for key in ("cycle_anchor_table", "cycle_anchor_lookahead_table"):
        frame = model.diagnostics.get(key)
        if frame is not None and not frame.empty:
            frames.append(frame.copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.drop_duplicates(subset=["date", "type"], keep="first").sort_values("date").reset_index(drop=True)


def _cycle_coordinate_for_dates(dates: pd.DatetimeIndex, anchors: pd.DataFrame, default_cycle: float) -> np.ndarray:
    if anchors is None or anchors.empty or "cycle" not in anchors.columns:
        return np.full(len(dates), float(default_cycle), dtype=float)
    a = anchors.dropna(subset=["cycle"]).copy().sort_values("date")
    if a.empty:
        return np.full(len(dates), float(default_cycle), dtype=float)
    x = pd.DatetimeIndex(pd.to_datetime(a["date"])).asi8.astype(float)
    y = a["cycle"].to_numpy(dtype=float)
    q = dates.asi8.astype(float)
    coords = np.interp(q, x, y, left=y[0], right=y[-1])
    # After the final known anchor, let the cycle coordinate continue slowly at
    # the frozen cycle cadence rather than staying permanently fixed.
    right = q > x[-1]
    if np.any(right):
        days = (q[right] - x[-1]) / 1e9 / 86400.0
        coords[right] = y[-1] + days / FIXED_CYCLE_DAYS
    return coords


def _geometry_k_floor_schedule(
    dates: pd.DatetimeIndex,
    anchors: pd.DataFrame,
    calibration_start: pd.Timestamp,
    calibrated_center: np.ndarray,
    raw_dev: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return the mathematical K floor required to preserve peak>next trough.

    For a peak p and following trough t:
      Cp*exp(K*Ap) > Ct*exp(-K*At)
    therefore K > log(Ct/Cp)/(Ap+At).  No target-price floor is chosen by hand;
    the required floor is entirely implied by centerline growth and raw cycle
    geometry.
    """
    floor = np.zeros(len(dates), dtype=float)
    rows = []
    if anchors is None or anchors.empty:
        return floor, pd.DataFrame(rows)
    turns = anchors[(pd.to_datetime(anchors["date"]) > pd.Timestamp(calibration_start)) & anchors["type"].isin(["peak", "trough"])].copy()
    turns = turns.sort_values("date").reset_index(drop=True)
    for i, row in turns.iterrows():
        if str(row["type"]) != "peak":
            continue
        after = turns.iloc[i + 1:]
        after = after[after["type"] == "trough"]
        if after.empty:
            continue
        trough_row = after.iloc[0]
        pdte = pd.Timestamp(row["date"]); tdte = pd.Timestamp(trough_row["date"])
        if pdte not in dates or tdte not in dates:
            continue
        pi = int(np.where(dates == pdte)[0][0]); ti = int(np.where(dates == tdte)[0][0])
        ap = float(max(raw_dev[pi], 0.0)); at = float(max(-raw_dev[ti], 0.0))
        denom = ap + at
        center_growth = float(np.log(max(calibrated_center[ti], 1e-12) / max(calibrated_center[pi], 1e-12)))
        kmin = float(max((center_growth + GEOMETRY_NUMERICAL_EPS_LOG) / denom, 0.0)) if denom > 1e-12 else 0.0
        if kmin > 0:
            kmin = float(np.nextafter(kmin, np.inf))
            floor[pi:ti + 1] = np.maximum(floor[pi:ti + 1], kmin)
        rows.append({
            "peak_date": pdte, "trough_date": tdte,
            "peak_cycle": int(row.get("cycle", 0)) if pd.notna(row.get("cycle", np.nan)) else None,
            "trough_cycle": int(trough_row.get("cycle", 0)) if pd.notna(trough_row.get("cycle", np.nan)) else None,
            "peak_raw_amplitude": ap, "trough_raw_amplitude": at,
            "calibrated_centerline_growth_log": center_growth,
            "minimum_geometric_K": kmin,
        })
    return floor, pd.DataFrame(rows)

def build_calibrated_price_model(
    base_model: PriceModelResult,
    calibration: WalkForwardCalibrationResult,
    prices: pd.DataFrame | None = None,
) -> CalibratedPriceModelResult:
    """Build the dynamic calibrated production projection without modifying v3.12."""
    if prices is None:
        daily = base_model.daily.copy().sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(daily["date"]).dt.normalize())
        raw_center = daily["structural_centerline_usd"].to_numpy(dtype=float)
        raw_price = daily["fitted_or_projected_price_usd"].to_numpy(dtype=float)
        latest = pd.Timestamp(base_model.diagnostics["training_end"]).normalize()
        projection_end = pd.Timestamp(base_model.diagnostics["projection_end_date"]).normalize()
        ref_model = base_model
        parent_models = [(pd.Timestamp(base_model.diagnostics.get("training_start", latest)), 1.0, base_model)]
    else:
        data = _normalise_prices(prices)
        latest = pd.Timestamp(data["date"].max()).normalize()
        projection_years = int(base_model.diagnostics.get("projection_years", 10))
        parent_models = _fit_current_parent_models(data, calibration, projection_years)
        ref_model = parent_models[0][2]
        ref = ref_model.daily.copy().sort_values("date").reset_index(drop=True)
        ref["date"] = pd.to_datetime(ref["date"]).dt.normalize()
        dates = pd.DatetimeIndex(ref["date"])
        daily = ref.copy()

        center_stack, dev_stack, used_weights = [], [], []
        for start_date, weight, model in parent_models:
            frame = model.daily.copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame.set_index("date").reindex(dates)
            center = frame["structural_centerline_usd"].to_numpy(dtype=float)
            price = frame["fitted_or_projected_price_usd"].to_numpy(dtype=float)
            valid = np.isfinite(center) & np.isfinite(price) & (center > 0) & (price > 0)
            if not np.all(valid[dates >= latest]):
                continue
            center_stack.append(np.log(np.maximum(center, 1e-12)))
            dev_stack.append(np.log(np.maximum(price, 1e-12) / np.maximum(center, 1e-12)))
            used_weights.append(weight)
        if not center_stack:
            raise ValueError("Cycle-aligned parent projections could not be aligned on a common date grid.")
        weights = np.asarray(used_weights, dtype=float); weights = weights / weights.sum()
        wc = weights[:, None]
        center_arr = np.vstack(center_stack); dev_arr = np.vstack(dev_stack)
        center_valid = np.isfinite(center_arr); dev_valid = np.isfinite(dev_arr)
        center_denom = np.sum(wc * center_valid, axis=0); dev_denom = np.sum(wc * dev_valid, axis=0)
        log_center = np.divide(np.nansum(wc * center_arr, axis=0), center_denom, out=np.full(center_denom.shape, np.nan), where=center_denom > 0)
        raw_deviation = np.divide(np.nansum(wc * dev_arr, axis=0), dev_denom, out=np.full(dev_denom.shape, np.nan), where=dev_denom > 0)
        raw_center = np.exp(log_center); raw_price = raw_center * np.exp(raw_deviation)
        projection_end = pd.Timestamp(dates.max()).normalize()

    dates = pd.DatetimeIndex(pd.to_datetime(daily["date"]).normalize()) if isinstance(daily["date"], pd.DatetimeIndex) else pd.DatetimeIndex(pd.to_datetime(daily["date"]).dt.normalize())
    projected_mask = dates > latest
    G = float(calibration.summary.get("effective_growth_factor", 1.0))
    trend = _summary_trend_dict(calibration.summary)

    if latest < NEXT_TROUGH <= projection_end:
        calibration_start = pd.Timestamp(NEXT_TROUGH).normalize()
        live_cycle_preserved = True
    else:
        calibration_start = latest
        live_cycle_preserved = False

    calibrated_center = _calibrated_centerline_from_raw(raw_center, dates, calibration_start, G)
    raw_dev = np.log(np.maximum(raw_price, 1e-12) / np.maximum(raw_center, 1e-12))
    anchors = _collect_model_anchors(ref_model)
    current_cycle = int(calibration.summary.get("current_cycle_index", 0))
    cycle_coords = _cycle_coordinate_for_dates(dates, anchors, current_cycle)
    unconstrained_K = np.array([_blended_amplitude_factor_for_cycle(trend, c) for c in cycle_coords], dtype=float)

    geometry_floor, geometry_table = _geometry_k_floor_schedule(
        dates, anchors, calibration_start, calibrated_center, raw_dev
    )
    effective_K = np.maximum(unconstrained_K, geometry_floor)
    calibrated_price = raw_price.copy()
    direct_mask = projected_mask & (dates > calibration_start)
    calibrated_price[direct_mask] = calibrated_center[direct_mask] * np.exp(effective_K[direct_mask] * raw_dev[direct_mask])

    # Preserve the live current bear through the model-derived Oct-2026 trough.
    preserve = dates <= calibration_start
    calibrated_price[preserve] = raw_price[preserve]
    calibrated_center[preserve] = raw_center[preserve]

    # Join the live-cycle trough continuously to the first fully calibrated turn.
    first_turn_date = None
    if not anchors.empty:
        future_turns = anchors[(pd.to_datetime(anchors["date"]) > calibration_start) & anchors["type"].isin(["peak", "trough"])].copy()
        if not future_turns.empty:
            first_turn_date = pd.Timestamp(future_turns.iloc[0]["date"]).normalize()
            if first_turn_date in dates and calibration_start in dates:
                start_idx = int(np.where(dates == calibration_start)[0][0]); turn_idx = int(np.where(dates == first_turn_date)[0][0])
                if turn_idx > start_idx:
                    raw_start = float(raw_price[start_idx]); raw_end = float(raw_price[turn_idx])
                    cal_end = float(calibrated_center[turn_idx] * np.exp(effective_K[turn_idx] * raw_dev[turn_idx]))
                    denom = np.log(max(raw_end, 1e-12)) - np.log(max(raw_start, 1e-12))
                    if abs(denom) > 1e-12:
                        seg = np.arange(start_idx, turn_idx + 1)
                        progress = (np.log(np.maximum(raw_price[seg], 1e-12)) - np.log(max(raw_start, 1e-12))) / denom
                        progress = np.clip(progress, 0.0, 1.0)
                        calibrated_price[seg] = np.exp(np.log(max(raw_start, 1e-12)) + progress * (np.log(max(cal_end, 1e-12)) - np.log(max(raw_start, 1e-12))))

    out = pd.DataFrame({
        "date": dates,
        "row_type": np.where(dates <= latest, "historical_training", "projected"),
        "actual_price_usd": np.nan,
        "raw_centerline_usd": raw_center,
        "raw_price_usd": raw_price,
        "calibrated_centerline_usd": calibrated_center,
        "calibrated_price_usd": calibrated_price,
        "cycle_coordinate": cycle_coords,
        "unconstrained_amplitude_factor_K": unconstrained_K,
        "minimum_geometric_K": geometry_floor,
        "amplitude_factor_K": effective_K,
        "geometry_constrained": effective_K > unconstrained_K + 1e-12,
        "calibration_active": dates > calibration_start,
    })

    turning_rows = []
    visible = pd.DataFrame()
    if not anchors.empty:
        visible = anchors[(pd.to_datetime(anchors["date"]) >= latest) & (pd.to_datetime(anchors["date"]) <= projection_end) & anchors["type"].isin(["peak", "trough"])].copy()
        for row in visible.itertuples(index=False):
            d = pd.Timestamp(row.date).normalize()
            if d not in dates: continue
            i = int(np.where(dates == d)[0][0])
            turning_rows.append({
                "date": d, "type": row.type, "cycle": getattr(row, "cycle", np.nan),
                "raw_price_usd": float(raw_price[i]), "calibrated_price_usd": float(calibrated_price[i]),
                "raw_centerline_usd": float(raw_center[i]), "calibrated_centerline_usd": float(calibrated_center[i]),
                "raw_price_over_centerline": float(raw_price[i] / raw_center[i]),
                "calibrated_price_over_centerline": float(calibrated_price[i] / calibrated_center[i]),
                "unconstrained_amplitude_factor_K": float(unconstrained_K[i]),
                "minimum_geometric_K": float(geometry_floor[i]),
                "amplitude_factor_K": float(effective_K[i]),
                "geometry_constrained": bool(effective_K[i] > unconstrained_K[i] + 1e-12),
                "source": getattr(row, "source", ""),
            })
    turning_points = pd.DataFrame(turning_rows)

    geometry_valid = True
    if not turning_points.empty:
        ordered = turning_points.sort_values("date").reset_index(drop=True)
        for i, row in ordered.iterrows():
            if row["type"] != "peak": continue
            after = ordered.iloc[i + 1:]
            after = after[after["type"] == "trough"]
            if after.empty: continue
            if not (float(row["calibrated_price_usd"]) > float(after.iloc[0]["calibrated_price_usd"])):
                geometry_valid = False; break

    if not geometry_table.empty:
        geometry_table = geometry_table.copy()
        geometry_table["was_binding"] = geometry_table["minimum_geometric_K"] > 0

    parent_diag = [{"start_date": start.date().isoformat(), "weight": weight} for start, weight, _ in parent_models]
    parent_payload = json.dumps(parent_diag, sort_keys=True)
    parent_fingerprint = hashlib.sha256(parent_payload.encode("utf-8")).hexdigest()[:16]
    diagnostics = {
        "version": "calibrated-price-model-v4.0-cycle-disciplined-learning",
        "base_model_version": PRICE_MODEL_ENGINE_VERSION,
        "calibration_version": calibration.summary.get("version", CALIBRATION_VERSION),
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_status": calibration.summary.get("status", "UNKNOWN"),
        "growth_factor": G,
        "structural_blend_weight": float(calibration.summary.get("structural_blend_weight", 0.0)),
        "amplitude_factor_at_latest_data": float(calibration.summary.get("amplitude_factor", 1.0)),
        "amplitude_mode": calibration.summary.get("amplitude_mode", "CONSTANT"),
        "amplitude_trend_direction": calibration.summary.get("amplitude_trend_direction", "NO_CLEAR_TREND"),
        "amplitude_trend_blend_weight": float(calibration.summary.get("amplitude_trend_blend_weight", 0.0)),
        "calibration_start_date": calibration_start.date().isoformat(),
        "live_cycle_preserved_through_calibration_start": live_cycle_preserved,
        "first_calibrated_turning_point": first_turn_date.date().isoformat() if first_turn_date is not None else None,
        "cycle_parent_models": parent_diag,
        "cycle_parent_ensemble_fingerprint": parent_fingerprint,
        "projection_years": int(base_model.diagnostics.get("projection_years", 10)),
        "latest_data_date": latest.date().isoformat(),
        "timing_source": calibration.summary.get("timing_source"),
        "selected_price_model_start_independent": prices is not None,
        "geometry_guard_enabled": True,
        "geometry_valid": bool(geometry_valid),
        "geometry_constraint_table": geometry_table,
    }
    return CalibratedPriceModelResult(out, turning_points, diagnostics)

def build_calibrated_projection_fingerprint(base_fingerprint: str | None, calibrated: CalibratedPriceModelResult) -> str:
    """Fingerprint the calibrated production path independently of selected start."""
    d = calibrated.diagnostics
    payload = {
        "engine": PRICE_MODEL_ENGINE_VERSION,
        "calibration": str(d.get("calibration_fingerprint")),
        "parents": str(d.get("cycle_parent_ensemble_fingerprint")),
        "G": round(float(d.get("growth_factor", 1.0)), 10),
        "K": round(float(d.get("amplitude_factor_at_latest_data", 1.0)), 10),
        "mode": str(d.get("amplitude_mode")),
        "start": str(d.get("calibration_start_date")),
        "years": int(d.get("projection_years", 10)),
        "latest": str(d.get("latest_data_date")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
