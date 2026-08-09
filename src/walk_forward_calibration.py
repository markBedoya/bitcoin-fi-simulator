from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd

from src.price_model import (
    NEXT_TROUGH,
    PRICE_MODEL_ENGINE_VERSION,
    PriceModelResult,
    fit_price_model,
)

CALIBRATION_VERSION = "walk-forward-calibration-v1.0"
CALIBRATION_FLOOR = pd.Timestamp("2015-01-14")
FIRST_FAKE_TODAY = pd.Timestamp("2023-01-14")
LOOKBACK_YEARS = (4, 8)
FAKE_TODAY_STEP_MONTHS = 6
EVALUATION_MONTHS = 12
RIDGE_STRENGTH = 4.0
FACTOR_MIN = 0.35
FACTOR_MAX = 1.35


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


def _equivalent_pct_error(log_errors: np.ndarray | pd.Series) -> float:
    values = np.asarray(log_errors, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.expm1(np.median(np.abs(values))))


def _fit_growth_amplitude_factors(
    observations: pd.DataFrame,
    ridge_strength: float = RIDGE_STRENGTH,
) -> tuple[float, float]:
    """Fit structural-growth (G) and cycle-amplitude (K) factors.

    The response is log price movement from each historical fake-today date.
    The two features decompose the frozen model's same movement into structural
    centerline growth and change in cycle deviation. Ridge regularization pulls
    both factors toward 1.0 when the historical sample is weak.
    """
    if observations.empty:
        return 1.0, 1.0

    X = observations[["structural_log_growth", "cycle_log_change"]].to_numpy(dtype=float)
    y = observations["actual_log_return"].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if len(y) < 4:
        return 1.0, 1.0

    prior = np.array([1.0, 1.0], dtype=float)
    weights = np.ones(len(y), dtype=float)
    beta = prior.copy()

    # Robust ridge via a few Huber IRLS passes. This keeps one unusual month
    # from dictating the calibration while preserving the no-intercept boundary
    # condition: at fake-today, modeled log return is exactly zero.
    for _ in range(8):
        WX = X * np.sqrt(weights)[:, None]
        Wy = y * np.sqrt(weights)
        lhs = WX.T @ WX + ridge_strength * np.eye(2)
        rhs = WX.T @ Wy + ridge_strength * prior
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

        residual = y - X @ beta
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(1.4826 * mad, 1e-4)
        cutoff = 1.5 * scale
        abs_resid = np.abs(residual)
        weights = np.where(abs_resid <= cutoff, 1.0, cutoff / np.maximum(abs_resid, 1e-12))

    beta = np.clip(beta, FACTOR_MIN, FACTOR_MAX)
    return float(beta[0]), float(beta[1])


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


def generate_fake_today_dates(prices: pd.DataFrame) -> list[pd.Timestamp]:
    latest = pd.Timestamp(prices["date"].max()).normalize()
    latest_eligible = latest - pd.DateOffset(months=EVALUATION_MONTHS)
    dates: list[pd.Timestamp] = []
    cursor = FIRST_FAKE_TODAY
    while cursor <= latest_eligible:
        # Use the actual daily observation on or before the standardized cutoff.
        actual = pd.Timestamp(_nearest_price_on_or_before(prices, cursor)["date"]).normalize()
        if not dates or actual > dates[-1]:
            dates.append(actual)
        cursor = cursor + pd.DateOffset(months=FAKE_TODAY_STEP_MONTHS)
    return dates


def _build_backtest_observations(
    prices: pd.DataFrame,
    fake_today: pd.Timestamp,
    lookback_years: int,
) -> tuple[pd.DataFrame, dict]:
    fake_today = pd.Timestamp(fake_today).normalize()
    requested_start = fake_today - pd.DateOffset(years=int(lookback_years))
    if requested_start < CALIBRATION_FLOOR:
        raise ValueError("Walk-forward training may not use data before the Jan 14, 2015 calibration floor.")

    train_start_row = _nearest_price_on_or_after(prices, requested_start)
    train_end_row = _nearest_price_on_or_before(prices, fake_today)
    training_start = pd.Timestamp(train_start_row["date"]).normalize()
    training_end = pd.Timestamp(train_end_row["date"]).normalize()

    # Two projection years keep the 12-month scoring horizon safely inside the
    # model even when DateOffset/calendar endpoints differ by a day.
    asof_prices = prices[prices["date"] <= training_end].copy()
    model = fit_price_model(
        prices=asof_prices,
        training_start=training_start,
        training_end=training_end,
        projection_years=2,
    )
    daily = model.daily.set_index("date")

    p0 = float(train_end_row["price_usd"])
    c0 = float(daily.loc[training_end, "structural_centerline_usd"])
    d0 = float(np.log(p0 / c0))

    rows = []
    endpoint_raw_price = np.nan
    endpoint_actual_price = np.nan
    for month in range(1, EVALUATION_MONTHS + 1):
        target = training_end + pd.DateOffset(months=month)
        actual_row = _nearest_price_on_or_before(prices, target)
        eval_date = pd.Timestamp(actual_row["date"]).normalize()
        if eval_date <= training_end or eval_date not in daily.index:
            continue

        actual_price = float(actual_row["price_usd"])
        raw_center = float(daily.loc[eval_date, "structural_centerline_usd"])
        raw_price = float(daily.loc[eval_date, "fitted_or_projected_price_usd"])
        structural_growth = float(np.log(raw_center / c0))
        raw_deviation = float(np.log(raw_price / raw_center))
        cycle_change = raw_deviation - d0
        actual_return = float(np.log(actual_price / p0))
        raw_log_return = structural_growth + cycle_change
        rows.append({
            "fake_today": training_end,
            "lookback_years": int(lookback_years),
            "training_start": training_start,
            "eval_date": eval_date,
            "months_forward": month,
            "start_price_usd": p0,
            "actual_price_usd": actual_price,
            "raw_price_usd": raw_price,
            "raw_centerline_usd": raw_center,
            "structural_log_growth": structural_growth,
            "cycle_log_change": cycle_change,
            "actual_log_return": actual_return,
            "raw_log_return": raw_log_return,
        })
        if month == EVALUATION_MONTHS:
            endpoint_raw_price = raw_price
            endpoint_actual_price = actual_price

    obs = pd.DataFrame(rows)
    if obs.empty:
        raise ValueError("No out-of-sample observations were available for this walk-forward test.")

    raw_error = _equivalent_pct_error(obs["actual_log_return"] - obs["raw_log_return"])
    endpoint_error = (
        float(endpoint_raw_price / endpoint_actual_price - 1.0)
        if np.isfinite(endpoint_raw_price) and np.isfinite(endpoint_actual_price)
        else float("nan")
    )
    meta = {
        "fake_today": training_end,
        "lookback_years": int(lookback_years),
        "training_start": training_start,
        "training_end": training_end,
        "raw_error": raw_error,
        "raw_12m_price_usd": float(endpoint_raw_price),
        "actual_12m_price_usd": float(endpoint_actual_price),
        "raw_12m_error_pct": endpoint_error,
    }
    return obs, meta


def _cross_validate_one_lookback(
    observations: pd.DataFrame,
    snapshot_tests: pd.DataFrame,
) -> dict:
    """Leave one fake-today date out and aggregate calibration robustly.

    Each fake-today test first learns its own regularized G/K pair.  The
    walk-forward calibration then uses the median of the *other* historical
    snapshots to calibrate the held-out period.  This is more stable than a
    single pooled regression because structural growth and cycle deviation are
    strongly correlated over short 12-month windows.
    """
    groups = sorted(pd.to_datetime(observations["fake_today"].unique()))
    raw_residuals = []
    calibrated_residuals = []
    holdout_rows = []

    for holdout in groups:
        train_factors = snapshot_tests[pd.to_datetime(snapshot_tests["fake_today"]) != holdout]
        test = observations[pd.to_datetime(observations["fake_today"]) == holdout]
        if train_factors.empty or test.empty:
            continue
        G = float(train_factors["snapshot_growth_factor"].median())
        K = float(train_factors["snapshot_amplitude_factor"].median())
        G = float(np.clip(G, FACTOR_MIN, FACTOR_MAX))
        K = float(np.clip(K, FACTOR_MIN, FACTOR_MAX))
        pred = (
            G * test["structural_log_growth"].to_numpy(dtype=float)
            + K * test["cycle_log_change"].to_numpy(dtype=float)
        )
        y = test["actual_log_return"].to_numpy(dtype=float)
        raw_pred = test["raw_log_return"].to_numpy(dtype=float)
        raw_residuals.extend((y - raw_pred).tolist())
        calibrated_residuals.extend((y - pred).tolist())
        holdout_rows.append({
            "fake_today": pd.Timestamp(holdout),
            "cv_growth_factor": G,
            "cv_amplitude_factor": K,
            "raw_error": _equivalent_pct_error(y - raw_pred),
            "calibrated_error": _equivalent_pct_error(y - pred),
        })

    return {
        "raw_cv_error": _equivalent_pct_error(raw_residuals),
        "calibrated_cv_error": _equivalent_pct_error(calibrated_residuals),
        "holdouts": pd.DataFrame(holdout_rows),
    }

def _stability_label(snapshot_factors: pd.DataFrame) -> tuple[str, float]:
    if snapshot_factors.empty or len(snapshot_factors) < 3:
        return "LOW", float("nan")
    g_iqr = float(snapshot_factors["snapshot_growth_factor"].quantile(0.75) - snapshot_factors["snapshot_growth_factor"].quantile(0.25))
    k_iqr = float(snapshot_factors["snapshot_amplitude_factor"].quantile(0.75) - snapshot_factors["snapshot_amplitude_factor"].quantile(0.25))
    spread = max(g_iqr, k_iqr)
    if spread <= 0.15:
        return "HIGH", spread
    if spread <= 0.30:
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
            }
            for r in tests[["fake_today", "lookback_years"]].itertuples(index=False)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def run_walk_forward_calibration(
    prices: pd.DataFrame,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> WalkForwardCalibrationResult:
    data = prices[["date", "price_usd"]].copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None).dt.normalize()
    fake_dates = generate_fake_today_dates(data)
    if len(fake_dates) < 3:
        raise ValueError(
            "Not enough post-Jan-2015 history exists for 4Y + 8Y walk-forward calibration. "
            "At least three eligible fake-today dates are required."
        )

    total = len(fake_dates) * len(LOOKBACK_YEARS)
    done = 0
    all_obs = []
    test_meta = []

    for fake_today in fake_dates:
        for lookback in LOOKBACK_YEARS:
            obs, meta = _build_backtest_observations(data, fake_today, lookback)
            G_snap, K_snap = _fit_growth_amplitude_factors(obs, ridge_strength=8.0)
            meta["snapshot_growth_factor"] = G_snap
            meta["snapshot_amplitude_factor"] = K_snap
            all_obs.append(obs)
            test_meta.append(meta)
            done += 1
            if progress_callback is not None:
                progress_callback(done, total, f"{lookback}Y as-of {pd.Timestamp(fake_today).date()}")

    observations = pd.concat(all_obs, ignore_index=True)
    tests = pd.DataFrame(test_meta).sort_values(["fake_today", "lookback_years"]).reset_index(drop=True)

    lookback_summaries = {}
    factors = []
    cv_holdout_frames = []
    for lookback in LOOKBACK_YEARS:
        obs_lb = observations[observations["lookback_years"] == lookback].copy()
        test_lb = tests[tests["lookback_years"] == lookback].copy()
        # The final lookback calibration is the median historical fake-today
        # correction. Each snapshot factor is already ridge-shrunk toward 1.0,
        # so the median is a robust dynamic-learning estimate rather than an
        # overfit pooled regression.
        G = float(np.clip(test_lb["snapshot_growth_factor"].median(), FACTOR_MIN, FACTOR_MAX))
        K = float(np.clip(test_lb["snapshot_amplitude_factor"].median(), FACTOR_MIN, FACTOR_MAX))
        cv = _cross_validate_one_lookback(obs_lb, test_lb)
        if not cv["holdouts"].empty:
            hold = cv["holdouts"].copy()
            hold["lookback_years"] = int(lookback)
            cv_holdout_frames.append(hold)
        stability, spread = _stability_label(test_lb)
        raw_err = float(cv["raw_cv_error"])
        cal_err = float(cv["calibrated_cv_error"])
        if not np.isfinite(cal_err) or cal_err <= 0:
            accuracy_weight = 1.0
        else:
            accuracy_weight = 1.0 / max(cal_err, 0.01) ** 2
        factors.append((lookback, G, K, accuracy_weight))
        lookback_summaries[lookback] = {
            "growth_factor": G,
            "amplitude_factor": K,
            "raw_cv_error": raw_err,
            "calibrated_cv_error": cal_err,
            "stability": stability,
            "stability_spread": spread,
            "tests": int(len(test_lb)),
            "accuracy_weight": accuracy_weight,
        }

    weights = np.array([item[3] for item in factors], dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(len(factors), dtype=float)
    weights = weights / weights.sum()
    combined_G = float(np.sum(weights * np.array([item[1] for item in factors], dtype=float)))
    combined_K = float(np.sum(weights * np.array([item[2] for item in factors], dtype=float)))
    combined_G = float(np.clip(combined_G, FACTOR_MIN, FACTOR_MAX))
    combined_K = float(np.clip(combined_K, FACTOR_MIN, FACTOR_MAX))

    # Out-of-sample score uses each lookback's genuine leave-one-cutoff-out predictions.
    raw_errors = [lookback_summaries[y]["raw_cv_error"] for y in LOOKBACK_YEARS]
    cal_errors = [lookback_summaries[y]["calibrated_cv_error"] for y in LOOKBACK_YEARS]
    raw_cv_error = float(np.sum(weights * np.array(raw_errors, dtype=float)))
    calibrated_cv_error = float(np.sum(weights * np.array(cal_errors, dtype=float)))
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

    enough_tests = all(lookback_summaries[y]["tests"] >= 3 for y in LOOKBACK_YEARS)
    improves = bool(np.isfinite(improvement) and improvement > 0.0)
    stable_enough = overall_stability != "LOW"
    status = "PASS" if enough_tests and improves and stable_enough else (
        "UNSTABLE" if not stable_enough else "NO_IMPROVEMENT"
    )

    summary = {
        "version": CALIBRATION_VERSION,
        "price_model_engine_version": PRICE_MODEL_ENGINE_VERSION,
        "calibration_floor": CALIBRATION_FLOOR.date().isoformat(),
        "first_fake_today": pd.Timestamp(fake_dates[0]).date().isoformat(),
        "last_fake_today": pd.Timestamp(fake_dates[-1]).date().isoformat(),
        "fake_today_step_months": FAKE_TODAY_STEP_MONTHS,
        "evaluation_months": EVALUATION_MONTHS,
        "growth_factor": combined_G,
        "amplitude_factor": combined_K,
        "raw_cv_error": raw_cv_error,
        "calibrated_cv_error": calibrated_cv_error,
        "cv_improvement": improvement,
        "stability": overall_stability,
        "status": status,
        "eligible_fake_today_dates": int(len(fake_dates)),
        "total_tests": int(len(tests)),
        "lookback_4y": lookback_summaries[4],
        "lookback_8y": lookback_summaries[8],
        "lookback_weights": {str(item[0]): float(w) for item, w in zip(factors, weights)},
        "latest_data_date": pd.Timestamp(data["date"].max()).date().isoformat(),
    }
    if cv_holdout_frames:
        holdouts = pd.concat(cv_holdout_frames, ignore_index=True)
        tests = tests.merge(
            holdouts, on=["fake_today", "lookback_years"], how="left",
            suffixes=("", "_cv"),
        )

    fingerprint = _fingerprint(summary, tests, data["date"].max())
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

    calibrated_center = _calibrated_centerline_from_raw(
        raw_center, dates, calibration_start, G
    )
    calibrated_price = raw_price.copy()

    projected_mask = daily["row_type"].eq("projected").to_numpy()
    after_start = dates > calibration_start
    direct_mask = projected_mask & after_start

    raw_deviation = np.log(np.maximum(raw_price, 1e-12) / np.maximum(raw_center, 1e-12))
    calibrated_price[direct_mask] = (
        calibrated_center[direct_mask] * np.exp(K * raw_deviation[direct_mask])
    )

    # The first segment after the calibration boundary must connect continuously
    # from the exact preserved boundary price to the first calibrated turning
    # point. Map the frozen model's own normalized log-price progress onto those
    # two calibrated endpoint prices; this preserves its empirical phase shape.
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
        if first_turn_date in dates:
            turn_idx = int(np.where(dates == first_turn_date)[0][0])
            start_idx = int(np.where(dates == calibration_start)[0][0]) if calibration_start in dates else None
            if start_idx is not None and turn_idx > start_idx:
                raw_start = float(raw_price[start_idx])
                raw_end = float(raw_price[turn_idx])
                cal_start = float(raw_start)  # preserved boundary price
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

    # Historical rows and any preserved current-cycle rows remain exactly the
    # frozen model, so calibration cannot rewrite observed history.
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
        visible_anchors = anchors[
            (pd.to_datetime(anchors["date"]) >= training_end)
            & (pd.to_datetime(anchors["date"]) <= projection_end)
            & anchors["type"].isin(["peak", "trough"])
        ].copy()
        for row in visible_anchors.itertuples(index=False):
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
        "version": "calibrated-price-model-v1.0",
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
