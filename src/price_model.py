from dataclasses import dataclass
import numpy as np
import pandas as pd

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


def _interp_knots(dates: pd.DatetimeIndex, knot_dates, knot_values):
    ords = np.array([pd.Timestamp(d).toordinal() for d in dates], dtype=float)
    kd = np.array([pd.Timestamp(d).toordinal() for d in knot_dates], dtype=float)
    kv = np.asarray(knot_values, dtype=float)
    out = np.empty(len(ords), dtype=float)

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
        s = float(_smoothstep(np.array([t]))[0])
        out[i] = kv[j] + (kv[j + 1] - kv[j]) * s
    return out


def _estimate_future_cycle_amplitude(history: pd.DataFrame, future_cycle: int) -> float:
    """Estimate one symmetric log-amplitude for a future cycle.

    Using one amplitude for both the peak (+A) and trough (-A) keeps the
    structural centerline as the geometric midpoint of every projected cycle.
    Historical cycles may be asymmetric because they are forced through actual
    market turning points; only the future projection is symmetrically centered.
    """
    if history.empty:
        return 0.6

    cycle_amplitudes = []
    for cycle, grp in history.dropna(subset=["cycle"]).groupby("cycle"):
        peaks = grp.loc[grp["type"] == "peak", "log_deviation"]
        troughs = grp.loc[grp["type"] == "trough", "log_deviation"]
        if peaks.empty or troughs.empty:
            continue
        peak_amp = float(np.median(np.abs(peaks.to_numpy(dtype=float))))
        trough_amp = float(np.median(np.abs(troughs.to_numpy(dtype=float))))
        cycle_amplitudes.append((float(cycle), 0.5 * (peak_amp + trough_amp)))

    if not cycle_amplitudes:
        amp = float(np.median(np.abs(history["log_deviation"].to_numpy(dtype=float))))
        return max(amp, 0.05)

    cycle_amplitudes.sort(key=lambda item: item[0])
    cycles = np.array([item[0] for item in cycle_amplitudes], dtype=float)
    amps = np.array([item[1] for item in cycle_amplitudes], dtype=float)
    latest_cycle = float(cycles[-1])
    latest_amp = float(amps[-1])

    # Learn only amplitude decay; never let projected volatility expand.
    if len(amps) >= 2:
        use = min(3, len(amps))
        slope, _ = np.polyfit(cycles[-use:], np.log(np.maximum(amps[-use:], 1e-6)), 1)
        retention = float(np.clip(np.exp(min(float(slope), 0.0)), 0.78, 1.0))
    else:
        retention = 0.92

    steps = max(float(future_cycle) - latest_cycle, 0.0)
    amp = latest_amp * (retention ** steps)
    return float(max(amp, 0.05))


def _bull_ease(t: float) -> float:
    """Curved 1064-day bull path: gradual early rise, faster middle, soft peak.

    The symmetric power-logistic form has an average of 0.5 over the phase,
    helping the projected cycle remain centered on the structural line.
    """
    t = float(np.clip(t, 0.0, 1.0))
    power = 2.35
    a = t ** power
    b = (1.0 - t) ** power
    return 0.0 if a + b == 0 else a / (a + b)


def _bear_ease(t: float) -> float:
    """Shorter bear path with a faster initial decline and slower bottoming.

    This form is also symmetric around 0.5 in aggregate, so it does not create
    a persistent upward/downward bias relative to the structural centerline.
    """
    t = float(np.clip(t, 0.0, 1.0))
    power = 0.72
    a = t ** power
    b = (1.0 - t) ** power
    return 0.0 if a + b == 0 else a / (a + b)


def _phase_interpolate(start_type: str, end_type: str, t: float) -> float:
    if start_type == "trough" and end_type == "peak":
        return _bull_ease(t)
    if start_type == "peak" and end_type == "trough":
        return _bear_ease(t)
    # A boundary anchor can sit partway through a bull/bear leg. Use the
    # destination turning point to preserve the expected phase character.
    if end_type == "peak":
        return _bull_ease(t)
    if end_type == "trough":
        return _bear_ease(t)
    return float(_smoothstep(np.array([t]))[0])


def _interp_cycle_knots(dates: pd.DatetimeIndex, knots: pd.DataFrame) -> np.ndarray:
    ords = np.array([pd.Timestamp(d).toordinal() for d in dates], dtype=float)
    knot_dates = [pd.Timestamp(d) for d in knots["date"]]
    kd = np.array([d.toordinal() for d in knot_dates], dtype=float)
    kv = knots["log_deviation"].to_numpy(dtype=float)
    kt = knots["type"].astype(str).tolist()
    out = np.empty(len(ords), dtype=float)

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
        s = _phase_interpolate(kt[j], kt[j + 1], t)
        out[i] = kv[j] + (kv[j + 1] - kv[j]) * s
    return out


def _build_cycle_fit(
    data: pd.DataFrame,
    train: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    structural_centerline: np.ndarray,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
):
    center_series = pd.Series(structural_centerline, index=all_dates)
    schedule = _fixed_cycle_anchors(training_start, all_dates.max())

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
            "source": "boundary anchor",
        })

    history_for_forecast = anchors.copy()

    # Future peak/trough anchors use one symmetric log-amplitude per cycle.
    # This guarantees peak = centerline * exp(+A) and trough = centerline * exp(-A),
    # so the centerline cannot drift above or below the projected cycle geometry.
    future_schedule = schedule[schedule["date"] > training_end].copy()
    future_rows = []
    amplitude_by_cycle = {}
    for row in future_schedule.itertuples(index=False):
        if row.date > all_dates.max():
            continue
        cycle_id = int(row.cycle)
        if cycle_id not in amplitude_by_cycle:
            amplitude_by_cycle[cycle_id] = _estimate_future_cycle_amplitude(
                history_for_forecast, cycle_id
            )
        amp = amplitude_by_cycle[cycle_id]
        dev = amp if row.type == "peak" else -amp
        center = float(center_series.loc[row.date]) if row.date in center_series.index else np.nan
        future_rows.append({
            "date": row.date,
            "requested_anchor_date": row.date,
            "type": row.type,
            "cycle": cycle_id,
            "actual_price_usd": np.nan,
            "structural_centerline_usd": center,
            "log_deviation": dev,
            "source": "projected centered cycle anchor",
        })

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

    deviation_path = _interp_cycle_knots(all_dates, knots)
    cycle_price = structural_centerline * np.exp(deviation_path)

    # Fixed phase: trough=0, peak=1064/1428, next trough=1.
    elapsed = np.array([(d - REFERENCE_TROUGH).days for d in all_dates], dtype=float)
    cycle_progress = np.mod(elapsed, FIXED_CYCLE_DAYS) / FIXED_CYCLE_DAYS
    peak_progress = FIXED_BULL_DAYS / FIXED_CYCLE_DAYS

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

    # Deterministic future template: one symmetric amplitude around the structural
    # centerline, with a curved bull phase and faster bear phase.
    template_amp = _estimate_future_cycle_amplitude(anchors, 1) if not anchors.empty else 0.6
    template_vals = np.empty_like(grid)
    for i, p in enumerate(grid):
        if p <= peak_progress:
            t = p / peak_progress
            s = _bull_ease(t)
            template_vals[i] = -template_amp + (2.0 * template_amp) * s
        else:
            t = (p - peak_progress) / (1.0 - peak_progress)
            s = _bear_ease(t)
            template_vals[i] = template_amp - (2.0 * template_amp) * s
    template_df = pd.DataFrame({"progress": grid, "log_deviation": template_vals})

    return cycle_price, deviation_path, cycle_progress, knots, overlay_df, template_df, {
        "cycle_days": float(FIXED_CYCLE_DAYS),
        "bull_days": int(FIXED_BULL_DAYS),
        "bear_days": int(FIXED_BEAR_DAYS),
        "peak_progress": float(peak_progress),
        "complete_cycles": int(complete_cycles),
        "turning_points": anchors.copy(),
        "next_modeled_trough": NEXT_TROUGH.date().isoformat(),
        "future_cycle_amplitudes": amplitude_by_cycle,
        "future_cycle_centered": True,
        "bull_curve": "power-logistic 2.35 (gradual / accelerating)",
        "bear_curve": "power-logistic 0.72 (faster initial decline)",
    }


def fit_price_model(prices, training_start, training_end, projection_years):
    data = prices.sort_values("date").copy()
    training_start = pd.Timestamp(training_start)
    training_end = pd.Timestamp(training_end)
    train = data[(data["date"] >= training_start) & (data["date"] <= training_end)].copy()
    if len(train) < 1000:
        raise ValueError("Select at least 1,000 daily training observations.")

    future_dates = pd.date_range(
        training_end + pd.Timedelta(days=1),
        training_end + pd.DateOffset(years=projection_years),
        freq="D",
    )

    structural_centerline, trend_diag = _fit_centerline(train, future_dates)
    all_dates = pd.DatetimeIndex(train["date"].tolist() + future_dates.tolist())

    fitted_or_projected, deviation_path, cycle_progress, knots, overlays, template, cycle_diag = _build_cycle_fit(
        data=data,
        train=train,
        all_dates=all_dates,
        structural_centerline=structural_centerline,
        training_start=training_start,
        training_end=training_end,
    )

    latest_actual_price = float(train["price_usd"].iloc[-1])
    fitted_endpoint = float(fitted_or_projected[len(train) - 1])
    endpoint_error = fitted_endpoint / latest_actual_price - 1.0

    daily = pd.DataFrame({
        "date": all_dates,
        "row_type": ["historical_training"] * len(train) + ["projected"] * len(future_dates),
        "actual_price_usd": list(train["price_usd"]) + [np.nan] * len(future_dates),
        "included_in_training": [True] * len(train) + [False] * len(future_dates),
        "structural_centerline_usd": structural_centerline,
        "cycle_progress": cycle_progress,
        "cycle_shape_value": deviation_path,
        "cycle_amplitude": np.abs(deviation_path),
        "fitted_or_projected_price_usd": fitted_or_projected,
        "model_version": "price-model-v2.4-centered-curved-cycle",
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
        "amplitude_scale": 1.0,
        "amplitude_retained_per_cycle": np.nan,
        "template_log_mean": 0.0,
        "endpoint_scale_factor": 1.0,
        "latest_actual_price": latest_actual_price,
        "fitted_endpoint_price": fitted_endpoint,
        "fitted_endpoint_error_pct": float(endpoint_error),
        "cycle_anchor_table": knots,
        **cycle_diag,
    }
    return PriceModelResult(daily, diagnostics, overlays, template)


def find_most_conservative_training_start(
    prices: pd.DataFrame,
    training_end: pd.Timestamp,
    projection_years: int,
    minimum_training_years: int = 8,
    progress_callback=None,
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
            result = fit_price_model(
                prices=data,
                training_start=pd.Timestamp(candidate_start),
                training_end=training_end,
                projection_years=projection_years,
            )
            projected = result.daily[result.daily["row_type"] == "projected"]
            if projected.empty:
                continue
            ending_price = float(projected["fitted_or_projected_price_usd"].iloc[-1])
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
