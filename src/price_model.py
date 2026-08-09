from dataclasses import dataclass
import numpy as np
import pandas as pd

PRICE_MODEL_ENGINE_VERSION = "price-model-v3.2.0-symmetric-cycle-envelope"

GENESIS = pd.Timestamp("2009-01-03")
FIXED_CYCLE_DAYS = 1428
FIXED_BULL_DAYS = 1064
FIXED_BEAR_DAYS = 364
REFERENCE_TROUGH = pd.Timestamp("2022-11-07")
REFERENCE_PEAK = pd.Timestamp("2025-10-06")
NEXT_TROUGH = pd.Timestamp("2026-10-05")

# Historical turning points are actual market dates used as price-intersection
# anchors when they fall inside the selected training range. Older cycles were
# close to, but not exactly, 1428 days. The future schedule is fixed at
# 1064 bull days + 364 bear days.
HISTORICAL_CYCLE_ANCHORS = [
    (pd.Timestamp("2015-01-14"), "trough", -2),
    (pd.Timestamp("2017-12-17"), "peak", -2),
    (pd.Timestamp("2018-12-15"), "trough", -1),
    (pd.Timestamp("2021-11-08"), "peak", -1),
    (pd.Timestamp("2022-11-07"), "trough", 0),
    (pd.Timestamp("2025-10-06"), "peak", 0),
]

# Mature amplitude anchors provide candidate turning points, but amplitude
# decay is evaluated only where the selected structural centerline is actually
# trained. This keeps peak/trough deviations mathematically coherent with the
# user's chosen structural fit instead of extrapolating that fit backward.
MATURE_AMPLITUDE_START = pd.Timestamp("2017-12-17")
# Phase-shape learning is also independent of the user-selected structural
# start. This keeps both mature bear phases (2017-18 and 2021-22) and the two
# mature bull phases (2018-21 and 2022-25) available for the empirical shape.
MATURE_PHASE_START = pd.Timestamp("2017-12-17")
MATURE_BULL_GAIN_START = pd.Timestamp("2018-12-15")
MATURE_AMPLITUDE_ANCHORS = [
    anchor for anchor in HISTORICAL_CYCLE_ANCHORS
    if anchor[0] >= MATURE_AMPLITUDE_START
]


@dataclass
class PriceModelResult:
    daily: pd.DataFrame
    diagnostics: dict
    cycle_overlays: pd.DataFrame
    cycle_template: pd.DataFrame


def _fit_centerline(train: pd.DataFrame, future_dates: pd.DatetimeIndex):
    days = (train["date"] - GENESIS).dt.days.to_numpy(dtype=float)
    log_days = np.log(days)
    log_price = np.log(train["price_usd"].to_numpy(dtype=float))

    weights = np.linspace(0.5, 1.0, len(train))
    X = np.column_stack([np.ones(len(log_days)), log_days])
    W = np.sqrt(weights)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, log_price * W[:, 0], rcond=None)
    intercept, exponent = map(float, beta)
    hist = np.exp(intercept + exponent * log_days)

    checkpoints = np.linspace(max(700, len(train) // 4), len(train), 10, dtype=int)
    exp_history = []
    for n in checkpoints:
        x = log_days[:n]
        y = log_price[:n]
        w = np.linspace(0.5, 1.0, n)
        XX = np.column_stack([np.ones(n), x])
        WW = np.sqrt(w)[:, None]
        b, *_ = np.linalg.lstsq(XX * WW, y * WW[:, 0], rcond=None)
        exp_history.append(float(b[1]))

    drift = np.polyfit(np.arange(len(exp_history)), exp_history, 1)[0] if len(exp_history) >= 3 else 0.0
    terminal_exponent = max(exponent + min(drift, 0.0) * 5.0, 0.0)
    half_life = 12.0

    years_future = np.maximum((future_dates - train["date"].max()).days / 365.25, 0)
    effective_exponent = terminal_exponent + (exponent - terminal_exponent) * np.exp(
        -np.log(2) * years_future / half_life
    )

    last_day = days[-1]
    last_trend = hist[-1]
    future_day = (future_dates - GENESIS).days.to_numpy(dtype=float)
    ratio = np.maximum(future_day / last_day, 1.0)
    future = last_trend * np.exp(effective_exponent * np.log(ratio))

    return np.concatenate([hist, future]), {
        "historical_exponent": exponent,
        "terminal_exponent": float(terminal_exponent),
        "exponent_half_life_years": half_life,
        "expanding_exponents": exp_history,
    }


def _fixed_cycle_anchors(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Return historical actual turning-point dates plus the deterministic future
    1428-day schedule.

    Historical anchors intentionally use observed market dates. Future cycles
    begin with the 2026-10-05 trough and use exactly 1064 bull days followed by
    364 bear days.
    """
    rows = [
        {"date": date, "type": anchor_type, "cycle": cycle}
        for date, anchor_type, cycle in HISTORICAL_CYCLE_ANCHORS
    ]

    cycle = 1
    trough = NEXT_TROUGH
    while trough <= end + pd.Timedelta(days=FIXED_CYCLE_DAYS):
        peak = trough + pd.Timedelta(days=FIXED_BULL_DAYS)
        rows.append({"date": trough, "type": "trough", "cycle": cycle})
        rows.append({"date": peak, "type": "peak", "cycle": cycle})
        trough = trough + pd.Timedelta(days=FIXED_CYCLE_DAYS)
        cycle += 1

    schedule = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["date", "type"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return schedule[
        (schedule["date"] >= start - pd.Timedelta(days=FIXED_CYCLE_DAYS))
        & (schedule["date"] <= end + pd.Timedelta(days=FIXED_CYCLE_DAYS))
    ].reset_index(drop=True)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Least-squares nondecreasing fit using the pool-adjacent-violators algorithm."""
    y = np.asarray(values, dtype=float)
    if len(y) <= 1:
        return y.copy()

    block_values = []
    block_weights = []
    block_counts = []
    for value in y:
        block_values.append(float(value))
        block_weights.append(1.0)
        block_counts.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            w1, w2 = block_weights[-2], block_weights[-1]
            merged_weight = w1 + w2
            merged_value = (
                block_values[-2] * w1 + block_values[-1] * w2
            ) / merged_weight
            merged_count = block_counts[-2] + block_counts[-1]
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_counts[-2:] = [merged_count]

    out = np.empty(len(y), dtype=float)
    pos = 0
    for value, count in zip(block_values, block_counts):
        out[pos:pos + count] = value
        pos += count
    return out


def _smooth_monotone_curve(values: np.ndarray, window: int = 11) -> np.ndarray:
    """Smooth a normalized phase curve and enforce 0 -> 1 monotonicity."""
    y = np.asarray(values, dtype=float)
    if len(y) == 0:
        return y.copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    smooth = (
        pd.Series(y)
        .rolling(window, center=True, min_periods=1)
        .median()
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    smooth = np.clip(smooth, 0.0, 1.0)
    smooth[0] = 0.0
    smooth[-1] = 1.0
    smooth = _isotonic_increasing(smooth)
    lo, hi = float(smooth[0]), float(smooth[-1])
    if hi - lo > 1e-12:
        smooth = (smooth - lo) / (hi - lo)
    smooth[0] = 0.0
    smooth[-1] = 1.0
    return np.clip(smooth, 0.0, 1.0)


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson/PCHIP-style slopes for monotone shape-preserving cubic interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.zeros_like(y)
    h = np.diff(x)
    delta = np.diff(y) / h
    if n == 2:
        return np.array([delta[0], delta[0]], dtype=float)

    m = np.zeros(n, dtype=float)
    for k in range(1, n - 1):
        d0, d1 = delta[k - 1], delta[k]
        if d0 == 0.0 or d1 == 0.0 or np.sign(d0) != np.sign(d1):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d0 + w2 / d1)

    # Endpoint slopes from the standard PCHIP one-sided estimate, then limit
    # them to preserve monotonicity.
    m[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(m[0]) != np.sign(delta[0]):
        m[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(m[0]) > abs(3.0 * delta[0]):
        m[0] = 3.0 * delta[0]

    m[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(m[-1]) != np.sign(delta[-1]):
        m[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(m[-1]) > abs(3.0 * delta[-1]):
        m[-1] = 3.0 * delta[-1]
    return m


def _monotone_cubic_eval(
    x: np.ndarray,
    y: np.ndarray,
    value: float,
    slopes: np.ndarray | None = None,
) -> float:
    """Evaluate a monotone cubic Hermite interpolation at one normalized point."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    value = float(np.clip(value, x[0], x[-1]))
    if value <= x[0]:
        return float(y[0])
    if value >= x[-1]:
        return float(y[-1])
    j = int(np.searchsorted(x, value) - 1)
    h = x[j + 1] - x[j]
    t = (value - x[j]) / h
    m = _pchip_slopes(x, y) if slopes is None else np.asarray(slopes, dtype=float)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return float(
        h00 * y[j]
        + h10 * h * m[j]
        + h01 * y[j + 1]
        + h11 * h * m[j + 1]
    )


def _combine_phase_curves(curves: list[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Median-combine historical normalized phase curves into one empirical template."""
    if not curves:
        return np.asarray(grid, dtype=float)
    matrix = np.vstack(curves)
    median_curve = np.nanmedian(matrix, axis=0)
    return _smooth_monotone_curve(median_curve, window=max(9, len(grid) // 35))


def _phase_curve_diagnostics(grid: np.ndarray, template: np.ndarray) -> dict:
    grid = np.asarray(grid, dtype=float)
    template = np.asarray(template, dtype=float)
    velocity = np.gradient(template, grid)
    acceleration = np.gradient(velocity, grid)
    interior = (grid >= 0.05) & (grid <= 0.95)
    interior_idx = np.where(interior)[0]
    if len(interior_idx):
        max_velocity_idx = interior_idx[int(np.argmax(velocity[interior]))]
        max_accel_idx = interior_idx[int(np.argmax(acceleration[interior]))]
    else:
        max_velocity_idx = int(np.argmax(velocity))
        max_accel_idx = int(np.argmax(acceleration))
    half_progress = float(np.interp(0.5, template, grid))
    return {
        "half_move_progress": half_progress,
        "max_velocity_progress": float(grid[max_velocity_idx]),
        "max_acceleration_progress": float(grid[max_accel_idx]),
        "max_velocity": float(velocity[max_velocity_idx]),
        "max_acceleration": float(acceleration[max_accel_idx]),
    }


def _lookup_price_near(data: pd.DataFrame, date: pd.Timestamp, tolerance_days: int = 3):
    nearest = data.iloc[(data["date"] - date).abs().argsort()[:1]].iloc[0]
    actual_date = pd.Timestamp(nearest["date"])
    if abs((actual_date - date).days) > tolerance_days:
        return None
    return actual_date, float(nearest["price_usd"])


def _learn_empirical_phase_templates(
    data: pd.DataFrame,
    center_series: pd.Series,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
):
    """Learn bull and bear timing shapes from completed historical phases.

    Each completed historical phase is normalized in time (0..1) and by its
    *total log-price move* from the actual starting turning point to the actual
    ending turning point. This makes the learned template describe the visible
    Bitcoin price trajectory itself rather than a residual around the structural
    centerline. Noise/corrections are handled with an isotonic fit, then the
    completed phases are median-combined and shape-preserving cubic interpolation
    is used when projecting future paths.
    """
    grid = np.linspace(0.0, 1.0, 401)
    schedule = sorted(HISTORICAL_CYCLE_ANCHORS, key=lambda item: item[0])
    bull_curves = []
    bear_curves = []
    overlay_rows = []

    for (start_date, start_type, _), (end_date, end_type, _) in zip(schedule[:-1], schedule[1:]):
        if (start_type, end_type) not in [("trough", "peak"), ("peak", "trough")]:
            continue
        if start_date < MATURE_PHASE_START or end_date > training_end:
            continue
        start_lookup = _lookup_price_near(data, start_date)
        end_lookup = _lookup_price_near(data, end_date)
        if start_lookup is None or end_lookup is None:
            continue
        actual_start_date, _ = start_lookup
        actual_end_date, _ = end_lookup
        seg = data[
            (data["date"] >= actual_start_date)
            & (data["date"] <= actual_end_date)
        ].copy()
        if len(seg) < 180:
            continue

        log_price = np.log(seg["price_usd"].to_numpy(dtype=float))
        start_log_price = float(log_price[0])
        end_log_price = float(log_price[-1])
        phase = "bull" if start_type == "trough" else "bear"

        if phase == "bull":
            denom = end_log_price - start_log_price
            normalized = (
                (log_price - start_log_price) / denom
                if abs(denom) > 1e-9 else None
            )
        else:
            denom = start_log_price - end_log_price
            normalized = (
                (start_log_price - log_price) / denom
                if abs(denom) > 1e-9 else None
            )
        if normalized is None:
            continue

        # Smooth daily noise before learning the underlying monotone move.
        day_window = max(9, int(round(len(seg) * 0.025)))
        if day_window % 2 == 0:
            day_window += 1
        normalized = (
            pd.Series(normalized)
            .rolling(day_window, center=True, min_periods=1)
            .median()
            .rolling(day_window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized[0] = 0.0
        normalized[-1] = 1.0
        normalized = _isotonic_increasing(normalized)
        elapsed = (seg["date"] - actual_start_date).dt.days.to_numpy(dtype=float)
        duration = max((actual_end_date - actual_start_date).days, 1)
        phase_progress = np.clip(elapsed / duration, 0.0, 1.0)
        curve = np.interp(grid, phase_progress, normalized)
        curve = _smooth_monotone_curve(curve, window=11)

        if phase == "bull":
            bull_curves.append(curve)
        else:
            bear_curves.append(curve)

        phase_id = f"{actual_start_date.date()} → {actual_end_date.date()}"
        overlay_rows.extend({
            "phase": phase,
            "phase_id": phase_id,
            "start_date": actual_start_date,
            "end_date": actual_end_date,
            "progress": float(g),
            "normalized_move": float(v),
        } for g, v in zip(grid, curve))

    # Use historical data whenever available. The fallbacks only prevent model
    # failure when a user selects a training window containing no complete phase.
    if bull_curves:
        bull_template = _combine_phase_curves(bull_curves, grid)
    else:
        bull_template = _smooth_monotone_curve(grid ** 2.7, window=9)
    if bear_curves:
        bear_template = _combine_phase_curves(bear_curves, grid)
    else:
        bear_template = _smooth_monotone_curve(1.0 - (1.0 - grid) ** 2.2, window=9)

    bull_diag = _phase_curve_diagnostics(grid, bull_template)
    bear_diag = _phase_curve_diagnostics(grid, bear_template)
    overlay_df = pd.DataFrame(overlay_rows)
    template_df = pd.concat([
        pd.DataFrame({
            "phase": "bull",
            "progress": grid,
            "normalized_move": bull_template,
        }),
        pd.DataFrame({
            "phase": "bear",
            "progress": grid,
            "normalized_move": bear_template,
        }),
    ], ignore_index=True)
    return grid, bull_template, bear_template, overlay_df, template_df, bull_diag, bear_diag


def _symmetric_cycle_amplitude_decay(history: pd.DataFrame) -> dict:
    """Estimate one decaying log-amplitude for complete future cycles.

    Each completed historical cycle contributes one amplitude equal to the mean
    of its peak and trough absolute log deviations from the structural centerline.
    Peak and trough are therefore treated as one cycle envelope rather than two
    independent forecasts.  The empirical retention rate is shrunk toward 1.0
    when only a few completed cycle-to-cycle transitions exist.
    """
    if history.empty:
        return {
            "observations": 0, "transitions": 0, "sample_confidence": 0.0,
            "latest_cycle": 0.0, "latest_amplitude": 0.35,
            "raw_retention_per_cycle": 0.90, "retention_per_cycle": 0.90,
            "method": "fallback", "cycle_amplitudes": pd.DataFrame(),
        }

    rows = []
    for cycle, grp in history.dropna(subset=["cycle"]).groupby("cycle"):
        peaks = grp.loc[grp["type"] == "peak", "log_deviation"].dropna()
        troughs = grp.loc[grp["type"] == "trough", "log_deviation"].dropna()
        if peaks.empty or troughs.empty:
            continue
        peak_amp = float(np.median(np.abs(peaks.to_numpy(dtype=float))))
        trough_amp = float(np.median(np.abs(troughs.to_numpy(dtype=float))))
        rows.append({
            "cycle": float(cycle),
            "peak_amplitude": peak_amp,
            "trough_amplitude": trough_amp,
            "cycle_amplitude": 0.5 * (peak_amp + trough_amp),
        })

    cycle_df = pd.DataFrame(rows).sort_values("cycle") if rows else pd.DataFrame()
    if cycle_df.empty:
        amp = float(np.median(np.abs(history["log_deviation"].dropna().to_numpy(dtype=float))))
        return {
            "observations": 0, "transitions": 0, "sample_confidence": 0.0,
            "latest_cycle": 0.0, "latest_amplitude": max(amp, 0.05),
            "raw_retention_per_cycle": 0.90, "retention_per_cycle": 0.90,
            "method": "anchor-median fallback", "cycle_amplitudes": cycle_df,
        }

    cycles = cycle_df["cycle"].to_numpy(dtype=float)
    amps = np.maximum(cycle_df["cycle_amplitude"].to_numpy(dtype=float), 1e-6)
    raw_retention = 1.0
    if len(amps) >= 2:
        log_amp = np.log(amps)
        slopes = []
        for i in range(len(amps) - 1):
            for j in range(i + 1, len(amps)):
                dc = cycles[j] - cycles[i]
                if dc > 1e-12:
                    slopes.append((log_amp[j] - log_amp[i]) / dc)
        robust_slope = min(float(np.median(slopes)) if slopes else 0.0, 0.0)
        robust_retention = float(np.exp(robust_slope))
        dc = cycles[-1] - cycles[-2]
        recent_retention = float((amps[-1] / amps[-2]) ** (1.0 / dc)) if dc > 1e-12 else 1.0
        raw_retention = float(np.clip(min(robust_retention, recent_retention, 1.0), 0.25, 1.0))

    transitions = max(len(amps) - 1, 0)
    if transitions:
        confidence = float(transitions / (transitions + 1.0))
        retention = float(np.exp(confidence * np.log(raw_retention)))
    else:
        confidence = 0.0
        retention = 0.90
    retention = float(np.clip(retention, raw_retention, 1.0))

    return {
        "observations": int(len(amps)),
        "transitions": int(transitions),
        "sample_confidence": confidence,
        "latest_cycle": float(cycles[-1]),
        "latest_amplitude": float(amps[-1]),
        "raw_retention_per_cycle": raw_retention,
        "retention_per_cycle": retention,
        "method": "symmetric completed-cycle envelope with small-sample shrinkage",
        "cycle_amplitudes": cycle_df,
    }


def _project_symmetric_cycle_amplitude(decay: dict, future_cycle: int) -> float:
    steps = max(float(future_cycle) - float(decay["latest_cycle"]), 0.0)
    amp = float(decay["latest_amplitude"]) * float(decay["retention_per_cycle"]) ** steps
    return float(max(amp, 0.05))


def _estimate_future_cycle_amplitude(history: pd.DataFrame, future_cycle: int) -> float:
    """Compatibility wrapper for the symmetric complete-cycle envelope."""
    return _project_symmetric_cycle_amplitude(_symmetric_cycle_amplitude_decay(history), future_cycle)



def _historical_centerline_power_law_params(
    train: pd.DataFrame,
    center_series: pd.Series,
) -> tuple[float, float]:
    """Recover the fitted historical power-law line for arbitrary past dates."""
    dates = pd.DatetimeIndex(train["date"])
    centers = center_series.reindex(dates).to_numpy(dtype=float)
    days = (dates - GENESIS).days.to_numpy(dtype=float)
    valid = np.isfinite(centers) & (centers > 0) & (days > 0)
    if valid.sum() < 2:
        raise ValueError("Not enough structural-centerline points for amplitude diagnostics.")
    x = np.log(days[valid])
    y = np.log(centers[valid])
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _build_amplitude_anchor_history(
    data: pd.DataFrame,
    train: pd.DataFrame,
    center_series: pd.Series,
    training_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build amplitude anchors that are coherent with the selected centerline.

    Peak/trough amplitude is defined as log(actual / structural centerline).
    Because the structural centerline itself is fitted only on the selected
    training range, only mature anchors inside that range are used to estimate
    future amplitude decay. This avoids letting a backward extrapolation of the
    centerline distort the 2017/2021/2025 amplitude trend when a later training
    start is selected.
    """
    columns = [
        "date", "requested_anchor_date", "type", "cycle",
        "actual_price_usd", "structural_centerline_usd", "log_deviation",
        "knot_price_usd", "source", "inside_structural_training",
        "used_for_amplitude_decay",
    ]
    train_start = pd.Timestamp(train["date"].min())
    train_end = pd.Timestamp(train["date"].max())
    rows = []
    for requested_date, anchor_type, cycle in MATURE_AMPLITUDE_ANCHORS:
        if requested_date > training_end:
            continue
        lookup = _lookup_price_near(data, requested_date)
        if lookup is None:
            continue
        actual_date, actual_price = lookup
        inside = bool(train_start <= actual_date <= train_end)
        if not inside or actual_date not in center_series.index:
            continue
        center = float(center_series.loc[actual_date])
        rows.append({
            "date": actual_date,
            "requested_anchor_date": requested_date,
            "type": anchor_type,
            "cycle": int(cycle),
            "actual_price_usd": float(actual_price),
            "structural_centerline_usd": center,
            "log_deviation": float(np.log(actual_price / center)),
            "knot_price_usd": float(actual_price),
            "source": "mature amplitude anchor inside structural training",
            "inside_structural_training": True,
            "used_for_amplitude_decay": True,
        })
    return pd.DataFrame(rows, columns=columns)



def _build_mature_bull_gain_history(data: pd.DataFrame, training_end: pd.Timestamp) -> pd.DataFrame:
    """Build completed mature trough->peak bull-run gains from actual prices.

    This dataset is independent of the selected structural-training start. It
    measures the total log-price gain of each completed mature bull phase, so a
    future peak cannot expand simply because the structural centerline moved.
    """
    cols = [
        "cycle", "start_date", "peak_date", "start_price_usd", "peak_price_usd",
        "bull_multiple", "bull_log_gain", "source",
    ]
    schedule = sorted(HISTORICAL_CYCLE_ANCHORS, key=lambda item: item[0])
    rows = []
    for (start_date, start_type, _), (peak_date, peak_type, peak_cycle) in zip(schedule[:-1], schedule[1:]):
        if start_type != "trough" or peak_type != "peak":
            continue
        if start_date < MATURE_BULL_GAIN_START or peak_date > training_end:
            continue
        start_lookup = _lookup_price_near(data, start_date)
        peak_lookup = _lookup_price_near(data, peak_date)
        if start_lookup is None or peak_lookup is None:
            continue
        actual_start, start_price = start_lookup
        actual_peak, peak_price = peak_lookup
        if start_price <= 0 or peak_price <= start_price:
            continue
        multiple = float(peak_price / start_price)
        rows.append({
            "cycle": int(peak_cycle),
            "start_date": actual_start,
            "peak_date": actual_peak,
            "start_price_usd": float(start_price),
            "peak_price_usd": float(peak_price),
            "bull_multiple": multiple,
            "bull_log_gain": float(np.log(multiple)),
            "source": "observed mature bull",
        })
    return pd.DataFrame(rows, columns=cols)


def _bull_gain_decay(history: pd.DataFrame) -> dict:
    """Fit monotone decay in completed bull-run *log gains*.

    With two mature completed bulls, the observed ratio of their log gains is
    the exact recent retention estimate. More history can contribute a robust
    median pairwise slope. Retention is capped at 100%, so the model can never
    project an expanding bull log gain after a shrinking sequence.
    """
    if history is None or history.empty:
        return {
            "observations": 0,
            "latest_cycle": 0.0,
            "latest_log_gain": float(np.log(4.0)),
            "latest_multiple": 4.0,
            "retention_per_cycle": 0.90,
            "robust_retention_per_cycle": 0.90,
            "recent_retention_per_cycle": np.nan,
            "method": "fallback",
        }
    h = history.sort_values(["cycle", "peak_date"]).drop_duplicates("cycle", keep="last")
    cycles = h["cycle"].to_numpy(dtype=float)
    gains = np.maximum(h["bull_log_gain"].to_numpy(dtype=float), 1e-9)
    robust = 0.90
    recent = np.nan
    if len(gains) >= 2:
        slopes = []
        log_g = np.log(gains)
        for i in range(len(gains)-1):
            for j in range(i+1, len(gains)):
                dc = cycles[j]-cycles[i]
                if dc > 0:
                    slopes.append((log_g[j]-log_g[i])/dc)
        slope = min(float(np.median(slopes)) if slopes else 0.0, 0.0)
        robust = float(np.clip(np.exp(slope), 0.25, 1.0))
        dc = cycles[-1]-cycles[-2]
        if dc > 0:
            recent = float(np.clip((gains[-1]/gains[-2])**(1.0/dc), 0.25, 1.0))
    retention = min(robust, float(recent) if np.isfinite(recent) else 1.0, 1.0)
    retention = float(np.clip(retention, 0.25, 1.0))
    return {
        "observations": int(len(gains)),
        "latest_cycle": float(cycles[-1]),
        "latest_log_gain": float(gains[-1]),
        "latest_multiple": float(np.exp(gains[-1])),
        "retention_per_cycle": retention,
        "robust_retention_per_cycle": robust,
        "recent_retention_per_cycle": float(recent) if np.isfinite(recent) else np.nan,
        "method": "min(robust mature bull-log-gain trend, most recent bull-log-gain retention)",
    }


def _project_bull_log_gain(decay: dict, future_cycle: int) -> float:
    steps = max(float(future_cycle) - float(decay["latest_cycle"]), 0.0)
    return float(max(
        float(decay["latest_log_gain"]) * (float(decay["retention_per_cycle"]) ** steps),
        0.0,
    ))

def _anchor_amplitude_decay(history: pd.DataFrame, anchor_type: str) -> dict:
    """Estimate conservative monotone peak/trough amplitude decay.

    A robust median pairwise slope is estimated across all mature anchors. The
    most recent observed cycle-to-cycle amplitude retention is then used as an
    additional upper bound. This gives the recent 2021->2025 compression direct
    influence instead of allowing older cycles to dilute it. Future retention is
    never above 100%, so projected cycle amplitudes cannot expand.
    """
    subset = history[
        (history["type"] == anchor_type)
        & history["cycle"].notna()
        & history["log_deviation"].notna()
    ].copy()
    if subset.empty:
        return {
            "anchor_type": anchor_type,
            "observations": 0,
            "latest_cycle": 0.0,
            "latest_amplitude": 0.35,
            "log_slope_per_cycle": float(np.log(0.90)),
            "retention_per_cycle": 0.90,
            "raw_retention_per_cycle": 0.90,
            "robust_retention_per_cycle": 0.90,
            "recent_retention_per_cycle": np.nan,
            "transitions": 0,
            "sample_confidence": 0.0,
            "retention_method": "fallback",
        }

    subset["amplitude"] = np.maximum(
        np.abs(subset["log_deviation"].to_numpy(dtype=float)), 1e-6
    )
    subset = subset.sort_values(["cycle", "date"]).drop_duplicates(
        subset=["cycle"], keep="last"
    )
    cycles = subset["cycle"].to_numpy(dtype=float)
    amps = subset["amplitude"].to_numpy(dtype=float)

    robust_retention = 0.90
    if len(amps) >= 2:
        slopes = []
        log_amp = np.log(amps)
        for i in range(len(amps) - 1):
            for j in range(i + 1, len(amps)):
                dc = cycles[j] - cycles[i]
                if abs(dc) > 1e-12:
                    slopes.append((log_amp[j] - log_amp[i]) / dc)
        slope = float(np.median(slopes)) if slopes else 0.0
        slope = min(slope, 0.0)
        robust_retention = float(np.clip(np.exp(slope), 0.25, 1.0))

        dc_recent = cycles[-1] - cycles[-2]
        if dc_recent > 1e-12:
            recent_retention = float(
                (amps[-1] / amps[-2]) ** (1.0 / dc_recent)
            )
            recent_retention = float(np.clip(recent_retention, 0.25, 1.0))
        else:
            recent_retention = np.nan
    else:
        recent_retention = np.nan

    # First form the raw empirical retention from the mature-history trend and
    # the latest observed same-type change.  With only two observed peaks (or
    # troughs), however, that raw number is based on a *single* cycle-to-cycle
    # transition and can wildly overfit one unusually large compression move.
    #
    # Apply sample-size shrinkage in log-retention space toward the neutral
    # hypothesis of 1.0 (no further compression).  The data receive weight
    # transitions / (transitions + 1), so one observed transition gets 50%
    # weight, two transitions 67%, three 75%, etc.  As more cycles arrive the
    # shrinkage naturally disappears.  This is deliberately not a hard-coded
    # amplitude floor: it only controls how much confidence we place in a tiny
    # sample when extrapolating the *rate* of decay.
    finite_recent = np.isfinite(recent_retention)
    raw_retention = min(
        robust_retention,
        float(recent_retention) if finite_recent else 1.0,
        1.0,
    )
    raw_retention = float(np.clip(raw_retention, 0.25, 1.0))
    transition_count = max(int(len(amps)) - 1, 0)
    if transition_count > 0:
        sample_confidence = float(transition_count / (transition_count + 1.0))
        retention = float(np.exp(sample_confidence * np.log(raw_retention)))
    else:
        sample_confidence = 0.0
        retention = 0.90
    retention = float(np.clip(retention, raw_retention, 1.0))
    slope = float(np.log(retention))

    return {
        "anchor_type": anchor_type,
        "observations": int(len(amps)),
        "transitions": int(transition_count),
        "sample_confidence": float(sample_confidence),
        "latest_cycle": float(cycles[-1]),
        "latest_amplitude": float(amps[-1]),
        "log_slope_per_cycle": slope,
        "retention_per_cycle": retention,
        "raw_retention_per_cycle": raw_retention,
        "robust_retention_per_cycle": float(robust_retention),
        "recent_retention_per_cycle": float(recent_retention) if finite_recent else np.nan,
        "retention_method": "small-sample shrinkage of empirical retention toward 100%",
    }


def _project_anchor_amplitude(decay: dict, future_cycle: int) -> float:
    steps = max(float(future_cycle) - float(decay["latest_cycle"]), 0.0)
    amp = float(decay["latest_amplitude"]) * (
        float(decay["retention_per_cycle"]) ** steps
    )
    return float(max(amp, 0.03))


def _fit_historical_segment_path(train: pd.DataFrame, knots: pd.DataFrame) -> np.ndarray:
    """Fit each historical anchor-to-anchor segment from its own observed shape.

    This keeps the historical fitted path descriptive rather than forcing one
    global bull template onto every era. Each segment is smoothed in log price,
    normalized between its exact boundary anchors, and made monotone with an
    isotonic fit. The future projection still uses the aggregate empirical
    bull/bear templates learned from completed cycles.
    """
    dates = pd.DatetimeIndex(train["date"])
    actual = train["price_usd"].to_numpy(dtype=float)
    out = actual.copy()
    hist_knots = knots[knots["date"] <= dates.max()].copy()
    hist_knots = (
        hist_knots.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if len(hist_knots) < 2:
        return out

    for left, right in zip(hist_knots.iloc[:-1].itertuples(index=False),
                           hist_knots.iloc[1:].itertuples(index=False)):
        mask = (dates >= pd.Timestamp(left.date)) & (dates <= pd.Timestamp(right.date))
        idxs = np.where(mask)[0]
        if len(idxs) < 3:
            continue
        seg_prices = actual[idxs]
        logp = np.log(seg_prices)
        start_log = float(np.log(left.knot_price_usd))
        end_log = float(np.log(right.knot_price_usd))
        delta = end_log - start_log
        if abs(delta) < 1e-9:
            out[idxs] = np.exp(np.linspace(start_log, end_log, len(idxs)))
            continue

        if delta > 0:
            normalized = (logp - start_log) / delta
        else:
            normalized = (start_log - logp) / (-delta)

        window = max(15, int(round(len(idxs) * 0.06)))
        if window % 2 == 0:
            window += 1
        normalized = (
            pd.Series(normalized)
            .rolling(window, center=True, min_periods=1)
            .median()
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized[0] = 0.0
        normalized[-1] = 1.0
        normalized = _isotonic_increasing(normalized)
        normalized = _smooth_monotone_curve(normalized, window=max(9, window // 3))

        if delta > 0:
            fitted_log = start_log + delta * normalized
        else:
            fitted_log = start_log - (-delta) * normalized
        out[idxs] = np.exp(fitted_log)

    # Exact endpoint/turning-point intersections are invariants.
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    for row in hist_knots.itertuples(index=False):
        d = pd.Timestamp(row.date)
        if d in date_to_idx:
            out[date_to_idx[d]] = float(row.knot_price_usd)
    return out


def _template_value(
    grid: np.ndarray,
    template: np.ndarray,
    t: float,
    slopes: np.ndarray | None = None,
) -> float:
    return _monotone_cubic_eval(
        grid, template, float(np.clip(t, 0.0, 1.0)), slopes=slopes
    )


def _fit_current_partial_bear(
    data: pd.DataFrame,
    training_end: pd.Timestamp,
    phase_grid: np.ndarray,
    bear_template: np.ndarray,
):
    """Estimate the Oct-2026 trough from the current partial bear market.

    The current fixed bear phase runs from the observed 2025-10-06 peak to the
    modeled 2026-10-05 trough. The learned historical bear template determines
    how much of a full bear decline is normally complete at each phase-progress
    point. We fit the full log decline to all observed daily prices in the
    current bear phase, then constrain it so the projected trough cannot sit
    above a decline already observed. The future path still begins at the exact
    latest actual price, preserving projection-boundary continuity.
    """
    training_end = pd.Timestamp(training_end)
    if not (REFERENCE_PEAK < training_end < NEXT_TROUGH):
        return None

    peak_lookup = _lookup_price_near(data, REFERENCE_PEAK)
    current_lookup = _lookup_price_near(data, training_end, tolerance_days=3)
    if peak_lookup is None or current_lookup is None:
        return None

    peak_date, peak_price = peak_lookup
    current_date, current_price = current_lookup
    seg = data[(data["date"] >= peak_date) & (data["date"] <= current_date)].copy()
    if len(seg) < 30:
        return None

    phase_progress = np.clip(
        (seg["date"] - REFERENCE_PEAK).dt.days.to_numpy(dtype=float)
        / FIXED_BEAR_DAYS,
        0.0,
        1.0,
    )
    bear_slopes = _pchip_slopes(phase_grid, bear_template)
    shape = np.array([
        _template_value(phase_grid, bear_template, value, slopes=bear_slopes)
        for value in phase_progress
    ], dtype=float)

    # Smooth only for estimating the eventual trough; the displayed projection
    # itself always starts from the exact latest actual price.
    log_prices = np.log(seg["price_usd"].to_numpy(dtype=float))
    smooth_log_prices = (
        pd.Series(log_prices)
        .rolling(14, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    observed_decline = np.log(peak_price) - smooth_log_prices
    weights = np.linspace(0.5, 1.0, len(seg))
    valid = shape > 1e-4
    denom = float(np.sum(weights[valid] * shape[valid] ** 2))
    if denom <= 1e-12:
        return None
    fitted_total_decline = float(
        np.sum(weights[valid] * shape[valid] * observed_decline[valid]) / denom
    )

    current_progress = float(np.clip(
        (current_date - REFERENCE_PEAK).days / FIXED_BEAR_DAYS, 0.0, 1.0
    ))
    learned_completion = _template_value(
        phase_grid, bear_template, current_progress, slopes=bear_slopes
    )

    # Endpoint-conditioned decline keeps the current observation aligned with
    # the learned phase progress without using a hand-selected remaining drop.
    recent = seg.tail(min(14, len(seg)))
    recent_geometric_price = float(
        np.exp(np.mean(np.log(recent["price_usd"].to_numpy(dtype=float))))
    )
    recent_decline = max(float(np.log(peak_price / recent_geometric_price)), 0.0)
    endpoint_implied_decline = (
        recent_decline / learned_completion
        if learned_completion > 1e-4 else recent_decline
    )

    # A future trough cannot be above the lowest price already observed during
    # this bear phase. This is a logical trough constraint, not a tuned percent.
    minimum_observed_price = float(seg["price_usd"].min())
    minimum_total_decline = max(float(np.log(peak_price / minimum_observed_price)), 0.0)
    total_decline = max(
        fitted_total_decline, endpoint_implied_decline, minimum_total_decline
    )
    projected_trough_price = float(peak_price * np.exp(-total_decline))

    fitted_decline_path = total_decline * shape
    rmse_log = float(np.sqrt(np.average(
        (observed_decline - fitted_decline_path) ** 2, weights=weights
    )))

    return {
        "phase": "bear",
        "phase_start": REFERENCE_PEAK,
        "phase_end": NEXT_TROUGH,
        "current_date": current_date,
        "current_progress": current_progress,
        "learned_completion": float(learned_completion),
        "peak_price_usd": float(peak_price),
        "current_price_usd": float(current_price),
        "projected_trough_price_usd": projected_trough_price,
        "remaining_change_pct": projected_trough_price / current_price - 1.0,
        "fitted_total_log_decline": float(total_decline),
        "fit_rmse_log": rmse_log,
        "observations": int(len(seg)),
    }


def _phase_interpolate(
    start_type: str,
    end_type: str,
    t: float,
    bull_grid: np.ndarray,
    bull_template: np.ndarray,
    bear_template: np.ndarray,
    bull_slopes: np.ndarray | None = None,
    bear_slopes: np.ndarray | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if start_type == "trough" and end_type == "peak":
        return _template_value(bull_grid, bull_template, t, slopes=bull_slopes)
    if start_type == "peak" and end_type == "trough":
        return _template_value(bull_grid, bear_template, t, slopes=bear_slopes)

    # A latest-actual boundary may occur partway through the current fixed phase.
    # Continue from that phase's learned progress rather than restarting the full
    # bull/bear template at the projection boundary.
    if end_type == "trough" and start_date is not None and end_date is not None:
        phase_start = end_date - pd.Timedelta(days=FIXED_BEAR_DAYS)
        t0 = float(np.clip((start_date - phase_start).days / FIXED_BEAR_DAYS, 0.0, 1.0))
        absolute_t = t0 + (1.0 - t0) * t
        y0 = _template_value(bull_grid, bear_template, t0, slopes=bear_slopes)
        y1 = _template_value(bull_grid, bear_template, absolute_t, slopes=bear_slopes)
        return 0.0 if 1.0 - y0 < 1e-12 else float((y1 - y0) / (1.0 - y0))
    if end_type == "peak" and start_date is not None and end_date is not None:
        phase_start = end_date - pd.Timedelta(days=FIXED_BULL_DAYS)
        t0 = float(np.clip((start_date - phase_start).days / FIXED_BULL_DAYS, 0.0, 1.0))
        absolute_t = t0 + (1.0 - t0) * t
        y0 = _template_value(bull_grid, bull_template, t0, slopes=bull_slopes)
        y1 = _template_value(bull_grid, bull_template, absolute_t, slopes=bull_slopes)
        return 0.0 if 1.0 - y0 < 1e-12 else float((y1 - y0) / (1.0 - y0))
    return float(_smoothstep(np.array([t]))[0])


def _interp_cycle_knots(
    dates: pd.DatetimeIndex,
    knots: pd.DataFrame,
    phase_grid: np.ndarray,
    bull_template: np.ndarray,
    bear_template: np.ndarray,
) -> np.ndarray:
    ords = np.array([pd.Timestamp(d).toordinal() for d in dates], dtype=float)
    knot_dates = [pd.Timestamp(d) for d in knots["date"]]
    kd = np.array([d.toordinal() for d in knot_dates], dtype=float)
    kv = knots["log_deviation"].to_numpy(dtype=float)
    kt = knots["type"].astype(str).tolist()
    out = np.empty(len(ords), dtype=float)
    bull_slopes = _pchip_slopes(phase_grid, bull_template)
    bear_slopes = _pchip_slopes(phase_grid, bear_template)

    for i, x in enumerate(ords):
        if x <= kd[0]:
            out[i] = kv[0]
            continue
        if x >= kd[-1]:
            out[i] = kv[-1]
            continue
        j = np.searchsorted(kd, x) - 1
        span = kd[j + 1] - kd[j]
        t = 0.0 if span <= 0 else (x - kd[j]) / span
        s = _phase_interpolate(
            kt[j], kt[j + 1], t,
            phase_grid, bull_template, bear_template,
            bull_slopes=bull_slopes, bear_slopes=bear_slopes,
            start_date=knot_dates[j], end_date=knot_dates[j + 1],
        )
        out[i] = kv[j] + (kv[j + 1] - kv[j]) * s
    return out


def _interp_cycle_price_knots(
    dates: pd.DatetimeIndex,
    knots: pd.DataFrame,
    phase_grid: np.ndarray,
    bull_template: np.ndarray,
    bear_template: np.ndarray,
) -> np.ndarray:
    """Interpolate the *total log-price move* between cycle turning points.

    This is intentionally different from interpolating only the residual around
    the structural centerline. The empirical bull/bear template therefore
    controls the visible trough-to-peak and peak-to-trough price trajectory.
    """
    ords = np.array([pd.Timestamp(d).toordinal() for d in dates], dtype=float)
    knot_dates = [pd.Timestamp(d) for d in knots["date"]]
    kd = np.array([d.toordinal() for d in knot_dates], dtype=float)
    kp = knots["knot_price_usd"].to_numpy(dtype=float)
    if np.any(~np.isfinite(kp)) or np.any(kp <= 0):
        raise ValueError("Cycle knot prices must be finite and positive.")
    log_kp = np.log(kp)
    kt = knots["type"].astype(str).tolist()
    out_log = np.empty(len(ords), dtype=float)
    bull_slopes = _pchip_slopes(phase_grid, bull_template)
    bear_slopes = _pchip_slopes(phase_grid, bear_template)

    for i, x in enumerate(ords):
        if x <= kd[0]:
            out_log[i] = log_kp[0]
            continue
        if x >= kd[-1]:
            out_log[i] = log_kp[-1]
            continue
        j = np.searchsorted(kd, x) - 1
        span = kd[j + 1] - kd[j]
        t = 0.0 if span <= 0 else (x - kd[j]) / span
        s = _phase_interpolate(
            kt[j], kt[j + 1], t,
            phase_grid, bull_template, bear_template,
            bull_slopes=bull_slopes, bear_slopes=bear_slopes,
            start_date=knot_dates[j], end_date=knot_dates[j + 1],
        )
        out_log[i] = log_kp[j] + (log_kp[j + 1] - log_kp[j]) * s
    return np.exp(out_log)


def _build_cycle_fit(
    data: pd.DataFrame,
    train: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    structural_centerline: np.ndarray,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    lightweight: bool = False,
):
    raw_structural_centerline = np.asarray(structural_centerline, dtype=float).copy()
    center_series = pd.Series(raw_structural_centerline, index=all_dates)
    schedule = _fixed_cycle_anchors(training_start, all_dates.max())
    (
        phase_grid,
        bull_template,
        bear_template,
        phase_overlays,
        phase_templates,
        bull_shape_diag,
        bear_shape_diag,
    ) = _learn_empirical_phase_templates(
        data=data,
        center_series=center_series,
        training_start=training_start,
        training_end=training_end,
    )

    current_partial_phase = _fit_current_partial_bear(
        data=data,
        training_end=training_end,
        phase_grid=phase_grid,
        bear_template=bear_template,
    )

    # Historical anchor intersections are only learned from prices inside the selected training range.
    hist_schedule = schedule[
        (schedule["date"] >= training_start) & (schedule["date"] <= training_end)
    ].copy()

    anchor_columns = [
        "date",
        "type",
        "cycle",
        "actual_price_usd",
        "structural_centerline_usd",
        "log_deviation",
        "knot_price_usd",
        "source",
    ]
    anchor_rows = []
    for row in hist_schedule.itertuples(index=False):
        # Coin Metrics is daily, but use the nearest available observation so a
        # missing calendar date can never silently remove a requested anchor.
        nearest = data.iloc[(data["date"] - row.date).abs().argsort()[:1]].iloc[0]
        actual_date = pd.Timestamp(nearest["date"])
        if abs((actual_date - row.date).days) > 3:
            continue
        if actual_date not in center_series.index:
            continue
        actual = float(nearest["price_usd"])
        center = float(center_series.loc[actual_date])
        anchor_rows.append({
            "date": actual_date,
            "requested_anchor_date": row.date,
            "type": row.type,
            "cycle": int(row.cycle),
            "actual_price_usd": actual,
            "structural_centerline_usd": center,
            "log_deviation": float(np.log(actual / center)),
            "knot_price_usd": actual,
            "source": "historical market anchor",
        })

    anchor_columns_with_requested = [
        "date",
        "requested_anchor_date",
        "type",
        "cycle",
        "actual_price_usd",
        "structural_centerline_usd",
        "log_deviation",
        "knot_price_usd",
        "source",
    ]
    anchors = pd.DataFrame(anchor_rows, columns=anchor_columns_with_requested)

    # Always force the fitted path to intersect the actual first and last training observations.
    endpoint_rows = []
    for d, label in [(training_start, "training_start"), (training_end, "latest_actual")]:
        nearest = train.iloc[(train["date"] - d).abs().argsort()[:1]].iloc[0]
        actual_d = pd.Timestamp(nearest["date"])
        actual = float(nearest["price_usd"])
        center = float(center_series.loc[actual_d])
        endpoint_rows.append({
            "date": actual_d,
            "requested_anchor_date": actual_d,
            "type": label,
            "cycle": np.nan,
            "actual_price_usd": actual,
            "structural_centerline_usd": center,
            "log_deviation": float(np.log(actual / center)),
            "knot_price_usd": actual,
            "source": "boundary anchor",
        })

    amplitude_history = _build_amplitude_anchor_history(
        data=data,
        train=train,
        center_series=center_series,
        training_end=training_end,
    )
    history_for_forecast = amplitude_history.copy()

    # Complete future cycles use ONE symmetric log-amplitude around the locked
    # structural centerline. This restores the geometric meaning of "centerline":
    # for a complete projected cycle, peak and trough are equal log distances
    # above and below it. The live Oct-2026 trough is the only exception because
    # it is conditioned from the already-partially-observed current bear market.
    future_schedule = schedule[schedule["date"] > training_end].copy()
    future_rows = []
    amplitude_by_cycle = {}
    symmetric_decay = _symmetric_cycle_amplitude_decay(history_for_forecast)

    conditioned_cycle1_trough_amp = None
    if current_partial_phase is not None and NEXT_TROUGH in center_series.index:
        trough_center = float(center_series.loc[NEXT_TROUGH])
        trough_price = float(current_partial_phase["projected_trough_price_usd"])
        conditioned_cycle1_trough_amp = max(
            abs(float(np.log(trough_price / trough_center))), 0.03
        )

    for row in future_schedule.itertuples(index=False):
        if row.date > all_dates.max():
            continue
        cycle_id = int(row.cycle)
        center = float(center_series.loc[row.date]) if row.date in center_series.index else np.nan
        raw_center = center

        if (
            current_partial_phase is not None
            and row.date == NEXT_TROUGH
            and row.type == "trough"
        ):
            knot_price = float(current_partial_phase["projected_trough_price_usd"])
            dev = float(np.log(knot_price / center))
            amp = abs(dev)
            source = "current bear-conditioned projected trough (live-cycle exception)"
        else:
            amp = _project_symmetric_cycle_amplitude(symmetric_decay, cycle_id)
            dev = amp if row.type == "peak" else -amp
            knot_price = float(center * np.exp(dev))
            source = "projected symmetric cycle envelope around locked structural centerline"

        amplitude_by_cycle.setdefault(cycle_id, {})[row.type] = float(abs(dev))
        future_rows.append({
            "date": row.date,
            "requested_anchor_date": row.date,
            "type": row.type,
            "cycle": cycle_id,
            "actual_price_usd": np.nan,
            "raw_structural_centerline_usd": raw_center,
            "structural_centerline_usd": center,
            "centerline_scale": 1.0,
            "log_deviation": dev,
            "knot_price_usd": knot_price,
            "source": source,
        })

    # Legacy diagnostics retained for UI/backward compatibility, but they are
    # no longer used to forecast future peak/trough geometry.
    peak_decay = _anchor_amplitude_decay(history_for_forecast, "peak")
    trough_decay = _anchor_amplitude_decay(history_for_forecast, "trough")
    bull_gain_history = _build_mature_bull_gain_history(data, training_end)
    bull_gain_decay = _bull_gain_decay(bull_gain_history)
    projected_bull_gain_rows = []
    centerline_conflicts = []
    # Locked structural backbone: the displayed centerline is exactly the raw
    # fitted/extrapolated centerline for every date. No maturity rule may alter it.
    display_centerline = raw_structural_centerline.copy()
    scale_df = pd.DataFrame([{
        "date": pd.Timestamp(training_end),
        "scale": 1.0,
        "reason": "structural centerline locked",
    }])

    future_amplitude_rows = pd.DataFrame(future_rows)
    if not future_amplitude_rows.empty:
        future_amplitude_rows = future_amplitude_rows.copy()
        future_amplitude_rows["inside_structural_training"] = False
        future_amplitude_rows["used_for_amplitude_decay"] = False
    amplitude_anchor_table = pd.concat(
        [amplitude_history, future_amplitude_rows],
        ignore_index=True,
        sort=False,
    )
    if not amplitude_anchor_table.empty:
        amplitude_anchor_table = amplitude_anchor_table.sort_values("date").reset_index(drop=True)

    projected_bull_gain_table = pd.DataFrame(projected_bull_gain_rows)
    bull_gain_table = pd.concat(
        [bull_gain_history, projected_bull_gain_table], ignore_index=True, sort=False
    )
    if not bull_gain_table.empty:
        bull_gain_table = bull_gain_table.sort_values(["peak_date", "cycle"]).reset_index(drop=True)

    knots = pd.concat(
        [pd.DataFrame(endpoint_rows), anchors, pd.DataFrame(future_rows)],
        ignore_index=True,
    )
    knots = (
        knots.sort_values(["date", "source"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(knots) < 2:
        raise ValueError("Not enough cycle anchors exist for the selected training range.")

    if lightweight:
        # Search ranking never consumes historical fitted values. Evaluate only
        # the sparse future dates (turning points + final horizon) so the exact
        # endpoint can be scored without thousands of unnecessary interpolations.
        cycle_price = np.full(len(all_dates), np.nan, dtype=float)
        future_slice = slice(len(train), len(all_dates))
        cycle_price[future_slice] = _interp_cycle_price_knots(
            all_dates[future_slice], knots, phase_grid, bull_template, bear_template
        )
        deviation_path = np.full(len(all_dates), np.nan, dtype=float)
        deviation_path[future_slice] = np.log(
            cycle_price[future_slice] / display_centerline[future_slice]
        )
    else:
        cycle_price = _interp_cycle_price_knots(
            all_dates, knots, phase_grid, bull_template, bear_template
        )
        # Historical fitted values use each observed segment's own smoothed timing
        # shape. Future values continue to use the aggregate empirical templates.
        historical_fit = _fit_historical_segment_path(train, knots)
        cycle_price[:len(train)] = historical_fit
        deviation_path = np.log(cycle_price / display_centerline)

    # Fixed phase: trough=0, peak=1064/1428, next trough=1.
    elapsed = np.array([(d - REFERENCE_TROUGH).days for d in all_dates], dtype=float)
    cycle_progress = np.mod(elapsed, FIXED_CYCLE_DAYS) / FIXED_CYCLE_DAYS
    peak_progress = FIXED_BULL_DAYS / FIXED_CYCLE_DAYS

    if not lightweight:
        # Build normalized cycle overlays from actual residuals for visualization only.
        overlays = []
        grid = np.linspace(0, 1, 301)
        complete_cycles = 0
        historical_troughs = (
            schedule[
                (schedule["type"] == "trough")
                & (schedule["date"] <= training_end)
            ]["date"]
            .sort_values()
            .tolist()
        )
        for trough_date, next_trough in zip(historical_troughs[:-1], historical_troughs[1:]):
            if trough_date < training_start or next_trough > training_end:
                continue
            seg = train[
                (train["date"] >= trough_date)
                & (train["date"] <= next_trough)
            ].copy()
            if len(seg) < 900:
                continue
            centers = center_series.reindex(
                pd.DatetimeIndex(seg["date"])
            ).to_numpy(dtype=float)
            residual = np.log(
                seg["price_usd"].to_numpy(dtype=float) / centers
            )
            cycle_days_actual = max((next_trough - trough_date).days, 1)
            progress = (
                (seg["date"] - trough_date).dt.days / cycle_days_actual
            ).to_numpy(dtype=float)
            interp = np.interp(grid, progress, residual)
            complete_cycles += 1
            overlays.append(pd.DataFrame({
                "cycle": complete_cycles,
                "progress": grid,
                "log_deviation": interp,
            }))

        overlay_df = pd.concat(overlays, ignore_index=True) if overlays else pd.DataFrame()

        # Future whole-cycle template uses the empirically learned phase shapes.
        template_amp = _estimate_future_cycle_amplitude(anchors, 1) if not anchors.empty else 0.6
        template_vals = np.empty_like(grid)
        bull_slopes = _pchip_slopes(phase_grid, bull_template)
        bear_slopes = _pchip_slopes(phase_grid, bear_template)
        for i, p in enumerate(grid):
            if p <= peak_progress:
                t = p / peak_progress
                s = _template_value(phase_grid, bull_template, t, slopes=bull_slopes)
                template_vals[i] = -template_amp + (2.0 * template_amp) * s
            else:
                t = (p - peak_progress) / (1.0 - peak_progress)
                s = _template_value(phase_grid, bear_template, t, slopes=bear_slopes)
                template_vals[i] = template_amp - (2.0 * template_amp) * s
        template_df = pd.DataFrame({"progress": grid, "log_deviation": template_vals})

    else:
        overlay_df = pd.DataFrame()
        template_df = pd.DataFrame()
        complete_cycles = 0

    return cycle_price, deviation_path, cycle_progress, knots, overlay_df, template_df, display_centerline, {
        "cycle_days": float(FIXED_CYCLE_DAYS),
        "bull_days": int(FIXED_BULL_DAYS),
        "bear_days": int(FIXED_BEAR_DAYS),
        "peak_progress": float(peak_progress),
        "complete_cycles": int(complete_cycles),
        "turning_points": anchors.copy(),
        "next_modeled_trough": NEXT_TROUGH.date().isoformat(),
        "future_cycle_amplitudes": amplitude_by_cycle,
        "symmetric_cycle_amplitude_decay": symmetric_decay,
        "peak_amplitude_decay": peak_decay,
        "trough_amplitude_decay": trough_decay,
        "amplitude_anchor_table": amplitude_anchor_table,
        "amplitude_training_start": (
            pd.Timestamp(amplitude_history["date"].min()).date().isoformat()
            if not amplitude_history.empty else None
        ),
        "amplitude_training_end": training_end.date().isoformat(),
        "amplitude_training_independent_of_structural_start": False,
        "bull_gain_decay": bull_gain_decay,
        "bull_gain_table": bull_gain_table,
        "bull_gain_monotone_guardrail": False,
        "bull_gain_guardrail_mode": "diagnostic only; symmetric cycle envelope is the forecast",
        "bull_gain_ceiling_basis_multiple": float(bull_gain_decay.get("latest_multiple", np.nan)),
        "bull_gain_centerline_conflicts": pd.DataFrame(centerline_conflicts),
        "phase_shape_training_start": MATURE_PHASE_START.date().isoformat(),
        "phase_shape_training_independent_of_structural_start": True,
        "future_peak_amplitude_monotone": True,
        "future_trough_amplitude_monotone": True,
        "future_cycle_centered": True,
        "future_centerline_reconciled": False,
        "future_centerline_reconciliation_applied": False,
        "future_centerline_min_scale": 1.0,
        "future_centerline_scale_table": scale_df.copy(),
        "structural_centerline_locked": True,
        "raw_structural_centerline_preserved_for_diagnostics": True,
        "phase_shape_applied_to": "total log-price path",
        "bull_curve": f"empirical median from {int(phase_overlays.loc[phase_overlays['phase'] == 'bull', 'phase_id'].nunique()) if not phase_overlays.empty else 0} completed bull phases",
        "bear_curve": f"empirical median from {int(phase_overlays.loc[phase_overlays['phase'] == 'bear', 'phase_id'].nunique()) if not phase_overlays.empty else 0} completed bear phases",
        "phase_shape_overlays": phase_overlays,
        "phase_shape_templates": phase_templates,
        "bull_shape_diagnostics": bull_shape_diag,
        "bear_shape_diagnostics": bear_shape_diag,
        "bull_phases_used": int(phase_overlays.loc[phase_overlays["phase"] == "bull", "phase_id"].nunique()) if not phase_overlays.empty else 0,
        "bear_phases_used": int(phase_overlays.loc[phase_overlays["phase"] == "bear", "phase_id"].nunique()) if not phase_overlays.empty else 0,
        "phase_shape_basis": "total log-price move",
        "historical_fit_method": "phase-specific smoothed monotone segment fit",
        "current_partial_phase": current_partial_phase,
    }


def fit_price_model(prices, training_start, training_end, projection_years):
    data = prices.sort_values("date").copy()
    training_start = pd.Timestamp(training_start)
    training_end = pd.Timestamp(training_end)
    train = data[(data["date"] >= training_start) & (data["date"] <= training_end)].copy()
    if len(train) < 1000:
        raise ValueError("Select at least 1,000 daily training observations.")

    projection_end = training_end + pd.DateOffset(years=projection_years)
    future_dates = pd.date_range(
        training_end + pd.Timedelta(days=1),
        projection_end,
        freq="D",
    )

    # The visible horizon can end halfway through a bull or bear phase.  The
    # phase interpolator needs the *next* turning point to know where that
    # unfinished segment is heading.  Model through one look-ahead anchor, then
    # trim the public daily output back to the requested horizon.  Without this,
    # the projection became flat after the last in-horizon turning point.
    lookahead_schedule = _fixed_cycle_anchors(training_start, projection_end)
    later_anchors = lookahead_schedule.loc[
        lookahead_schedule["date"] > projection_end, "date"
    ]
    model_end = (
        pd.Timestamp(later_anchors.min())
        if not later_anchors.empty
        else pd.Timestamp(projection_end)
    )
    model_future_dates = pd.date_range(
        training_end + pd.Timedelta(days=1),
        model_end,
        freq="D",
    )

    structural_centerline_model, trend_diag = _fit_centerline(train, model_future_dates)
    all_model_dates = pd.DatetimeIndex(train["date"].tolist() + model_future_dates.tolist())

    fitted_model, deviation_model, progress_model, knots, overlays, template, display_centerline_model, cycle_diag = _build_cycle_fit(
        data=data,
        train=train,
        all_dates=all_model_dates,
        structural_centerline=structural_centerline_model,
        training_start=training_start,
        training_end=training_end,
    )

    visible_len = len(train) + len(future_dates)
    all_dates = all_model_dates[:visible_len]
    fitted_or_projected = fitted_model[:visible_len]
    deviation_path = deviation_model[:visible_len]
    cycle_progress = progress_model[:visible_len]
    display_centerline = display_centerline_model[:visible_len]
    structural_centerline = structural_centerline_model[:visible_len]

    latest_actual_price = float(train["price_usd"].iloc[-1])
    fitted_endpoint = float(fitted_or_projected[len(train) - 1])
    endpoint_error = fitted_endpoint / latest_actual_price - 1.0

    daily = pd.DataFrame({
        "date": all_dates,
        "row_type": ["historical_training"] * len(train) + ["projected"] * len(future_dates),
        "actual_price_usd": list(train["price_usd"]) + [np.nan] * len(future_dates),
        "included_in_training": [True] * len(train) + [False] * len(future_dates),
        "structural_centerline_usd": display_centerline,
        "raw_structural_centerline_usd": structural_centerline,
        "cycle_progress": cycle_progress,
        "cycle_shape_value": deviation_path,
        "cycle_amplitude": np.abs(deviation_path),
        "fitted_or_projected_price_usd": fitted_or_projected,
        "model_version": PRICE_MODEL_ENGINE_VERSION,
        "training_start_date": training_start.date().isoformat(),
        "training_end_date": training_end.date().isoformat(),
    })
    daily["cycle_phase"] = np.where(
        daily["cycle_progress"] <= cycle_diag["peak_progress"], "bull", "bear"
    )

    diagnostics = {
        **trend_diag,
        "training_rows": len(train),
        "training_start": training_start.date().isoformat(),
        "training_end": training_end.date().isoformat(),
        "projection_years": projection_years,
        "model_version": PRICE_MODEL_ENGINE_VERSION,
        "amplitude_scale": 1.0,
        "amplitude_retained_per_cycle": np.nan,
        "template_log_mean": 0.0,
        "endpoint_scale_factor": 1.0,
        "latest_actual_price": latest_actual_price,
        "fitted_endpoint_price": fitted_endpoint,
        "fitted_endpoint_error_pct": float(endpoint_error),
        "cycle_anchor_table": knots[knots["date"] <= projection_end].copy(),
        "cycle_anchor_lookahead_table": knots[knots["date"] > projection_end].copy(),
        "projection_end_date": pd.Timestamp(projection_end),
        "projection_lookahead_anchor_date": pd.Timestamp(model_end),
        "projection_tail_uses_lookahead_anchor": bool(model_end > projection_end),
        **cycle_diag,
    }
    return PriceModelResult(daily, diagnostics, overlays, template)



def _score_projection_endpoint_exact(
    prices: pd.DataFrame,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    projection_years: int,
) -> float:
    """Return the exact model endpoint without constructing the full future daily path.

    Conservative-start ranking only needs the price at the selected horizon.
    We therefore evaluate the structural model on historical training dates plus
    the handful of future turning-point dates and the final horizon date. The
    endpoint is mathematically identical to a full daily projection because the
    interpolation and structural formulas are date-local between the same knots.
    """
    data = prices.sort_values("date").copy()
    training_start = pd.Timestamp(training_start)
    training_end = pd.Timestamp(training_end)
    train = data[(data["date"] >= training_start) & (data["date"] <= training_end)].copy()
    if len(train) < 1000:
        raise ValueError("Select at least 1,000 daily training observations.")

    target_date = training_end + pd.DateOffset(years=projection_years)
    schedule = _fixed_cycle_anchors(training_start, target_date)
    future_anchor_dates = schedule.loc[
        (schedule["date"] > training_end) & (schedule["date"] <= target_date),
        "date",
    ].tolist()
    after_target = schedule.loc[schedule["date"] > target_date, "date"]
    if not after_target.empty:
        # Include exactly one anchor beyond the target so a horizon ending
        # mid-phase is scored on the same interpolated segment as the full
        # daily model instead of being flat-filled from the prior turning point.
        future_anchor_dates.append(pd.Timestamp(after_target.min()))
    sparse_future = pd.DatetimeIndex(sorted(set(future_anchor_dates + [target_date])))

    structural_centerline, _ = _fit_centerline(train, sparse_future)
    all_dates = pd.DatetimeIndex(train["date"].tolist() + sparse_future.tolist())
    fitted, *_ = _build_cycle_fit(
        data=data,
        train=train,
        all_dates=all_dates,
        structural_centerline=structural_centerline,
        training_start=training_start,
        training_end=training_end,
        lightweight=True,
    )
    target_idx = all_dates.get_loc(target_date)
    if isinstance(target_idx, slice):
        target_idx = target_idx.stop - 1
    return float(fitted[int(target_idx)])


def find_most_conservative_training_start(
    prices: pd.DataFrame,
    training_end: pd.Timestamp,
    projection_years: int,
    minimum_training_years: int = 8,
    progress_callback=None,
    score_cache: dict | None = None,
):
    data = prices.sort_values("date").copy()
    data = data[data["date"] <= training_end].copy()
    if data.empty:
        raise ValueError("No data exists on or before the selected training end date.")

    latest_allowed_start = training_end - pd.DateOffset(years=minimum_training_years)
    eligible = data[data["date"] <= latest_allowed_start].copy()
    if eligible.empty:
        raise ValueError(
            f"The selected training end date does not allow {minimum_training_years} years of history."
        )

    candidate_dates = (
        eligible.assign(month=eligible["date"].dt.to_period("M"))
        .groupby("month", as_index=False)["date"]
        .max()["date"]
        .tolist()
    )

    end_price = float(data["price_usd"].iloc[-1])
    rows = []
    total = len(candidate_dates)

    for idx, candidate_start in enumerate(candidate_dates, start=1):
        try:
            cache_key = (
                "price-model-v3.1.0",
                pd.Timestamp(candidate_start).date().isoformat(),
                training_end.date().isoformat(),
                int(projection_years),
                int(len(data)),
                round(end_price, 8),
            )
            if score_cache is not None and cache_key in score_cache:
                ending_price = float(score_cache[cache_key])
            else:
                ending_price = _score_projection_endpoint_exact(
                    prices=data,
                    training_start=pd.Timestamp(candidate_start),
                    training_end=training_end,
                    projection_years=projection_years,
                )
                if score_cache is not None:
                    score_cache[cache_key] = float(ending_price)
            implied_cagr = (ending_price / end_price) ** (1.0 / projection_years) - 1.0
            rows.append({
                "training_start": pd.Timestamp(candidate_start),
                "training_end": training_end,
                "projection_years": int(projection_years),
                "projected_ending_price_usd": ending_price,
                "implied_cagr": implied_cagr,
                "training_years": (training_end - pd.Timestamp(candidate_start)).days / 365.25,
            })
        except Exception:
            pass

        if progress_callback:
            progress_callback(idx, total)

    if not rows:
        raise ValueError("No eligible monthly candidate produced a valid model.")

    results = pd.DataFrame(rows).sort_values(
        ["implied_cagr", "projected_ending_price_usd", "training_start"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return results.iloc[0].to_dict(), results
