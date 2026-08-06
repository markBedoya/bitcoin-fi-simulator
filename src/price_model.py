from dataclasses import dataclass
import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")

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

    # Estimate exponent maturity from expanding historical fits.
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

def _turning_points(residual: np.ndarray, dates: pd.Series):
    smooth = pd.Series(residual).rolling(121, center=True, min_periods=1).median().to_numpy()
    raw = []
    for i in range(2, len(smooth) - 2):
        if smooth[i] >= smooth[i-1] and smooth[i] >= smooth[i+1]:
            raw.append((i, "peak", float(smooth[i])))
        elif smooth[i] <= smooth[i-1] and smooth[i] <= smooth[i+1]:
            raw.append((i, "trough", float(smooth[i])))

    selected = []
    min_separation = 500
    for idx, typ, value in raw:
        if not selected:
            selected.append([idx, typ, value])
            continue
        if idx - selected[-1][0] < min_separation:
            if typ == selected[-1][1]:
                better = value > selected[-1][2] if typ == "peak" else value < selected[-1][2]
                if better:
                    selected[-1] = [idx, typ, value]
            continue
        if typ == selected[-1][1]:
            better = value > selected[-1][2] if typ == "peak" else value < selected[-1][2]
            if better:
                selected[-1] = [idx, typ, value]
        else:
            selected.append([idx, typ, value])

    return pd.DataFrame(
        [{"idx": i, "date": dates.iloc[i], "type": t, "value": v} for i, t, v in selected]
    )

def _cycle_template(train: pd.DataFrame, residual: np.ndarray):
    tp = _turning_points(residual, train["date"])
    trough_idx = tp.loc[tp["type"] == "trough", "idx"].astype(int).tolist() if not tp.empty else []
    grid = np.linspace(0, 1, 301)
    cycles, frames = [], []

    for num, (a, b) in enumerate(zip(trough_idx[:-1], trough_idx[1:]), start=1):
        if b - a < 700:
            continue
        segment = residual[a:b+1]
        interp = np.interp(grid, np.linspace(0, 1, len(segment)), segment)
        cycles.append(interp)
        frames.append(pd.DataFrame({"cycle": num, "progress": grid, "log_deviation": interp}))

    if cycles:
        matrix = np.vstack(cycles)
        weights = np.arange(1, len(cycles) + 1, dtype=float)
        weights /= weights.sum()
        template = np.average(matrix, axis=0, weights=weights)
    else:
        template = np.zeros_like(grid)

    template = pd.Series(template).rolling(11, center=True, min_periods=1).mean().to_numpy(copy=True)

    # The learned cycle template can have a non-zero mean in log space.
    # A positive mean makes the structural line sit below the visual middle
    # of the projected oscillations. Keep the mean for an exact algebraic
    # reparameterization later, then expose a zero-mean template.
    template_log_mean = float(np.trapz(template, grid) / (grid[-1] - grid[0]))
    centered_template = template - template_log_mean

    if len(trough_idx) >= 2:
        ordinals = [train["date"].iloc[i].toordinal() for i in trough_idx]
        cycle_days = float(np.median(np.diff(ordinals)))
    else:
        cycle_days = 1461.0

    overlays = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not overlays.empty:
        overlays["log_deviation"] = overlays["log_deviation"] - template_log_mean
    template_df = pd.DataFrame({"progress": grid, "log_deviation": centered_template})
    diagnostics = {
        "turning_points": tp,
        "complete_cycles": len(cycles),
        "cycle_days": cycle_days,
        "peak_progress": float(grid[np.argmax(centered_template)]) if len(centered_template) else 0.5,
        "template_log_mean": template_log_mean,
    }
    return overlays, template_df, diagnostics

def fit_price_model(prices, training_start, training_end, projection_years):
    data = prices.sort_values("date").copy()
    train = data[(data["date"] >= training_start) & (data["date"] <= training_end)].copy()
    if len(train) < 1000:
        raise ValueError("Select at least 1,000 daily training observations.")

    future_dates = pd.date_range(
        training_end + pd.Timedelta(days=1),
        training_end + pd.DateOffset(years=projection_years),
        freq="D",
    )

    centerline, trend_diag = _fit_centerline(train, future_dates)
    hist_center = centerline[:len(train)]
    residual = np.log(train["price_usd"].to_numpy() / hist_center)

    overlays, template, cycle_diag = _cycle_template(train, residual)
    grid = template["progress"].to_numpy()
    centered_shape = template["log_deviation"].to_numpy()
    template_log_mean = float(cycle_diag.get("template_log_mean", 0.0))
    raw_shape = centered_shape + template_log_mean
    tp = cycle_diag["turning_points"]

    troughs = tp[tp["type"] == "trough"] if not tp.empty else pd.DataFrame()
    if not troughs.empty:
        last_trough_idx = int(troughs.iloc[-1]["idx"])
        days_since_trough = (training_end - train["date"].iloc[last_trough_idx]).days
    else:
        days_since_trough = 0

    cycle_days = max(cycle_diag["cycle_days"], 365.0)
    hist_progress = ((np.arange(len(train)) - (len(train)-1-days_since_trough)) / cycle_days) % 1.0
    future_steps = np.arange(1, len(future_dates) + 1)
    future_progress = ((days_since_trough + future_steps) / cycle_days) % 1.0

    raw_hist_shape = np.interp(hist_progress, grid, raw_shape)
    raw_future_shape = np.interp(future_progress, grid, raw_shape)
    hist_shape = np.interp(hist_progress, grid, centered_shape)
    future_shape = np.interp(future_progress, grid, centered_shape)

    valid = np.abs(raw_hist_shape) > 0.05
    amplitude_scale = float(np.median(np.abs(residual[valid] / raw_hist_shape[valid]))) if valid.any() else 1.0
    amplitude_scale = float(np.clip(amplitude_scale, 0.4, 2.5))

    amplitudes = np.abs(tp["value"].to_numpy()) if not tp.empty else np.array([1.0])
    if len(amplitudes) >= 4:
        slope = np.polyfit(np.arange(len(amplitudes)), np.log(np.maximum(amplitudes, 1e-6)), 1)[0]
        retain_per_cycle = float(np.exp(min(slope, 0.0)))
    else:
        retain_per_cycle = 0.85

    future_amp = amplitude_scale * np.power(retain_per_cycle, future_steps / cycle_days)

    # Reparameterize without changing fitted/projected prices:
    # raw_center * exp(raw_shape * amplitude)
    # == centered_center * exp(centered_shape * amplitude).
    # This makes structural_centerline_usd the true geometric center of the
    # oscillation rather than the underlying regression baseline.
    centered_hist_center = hist_center * np.exp(template_log_mean * amplitude_scale)
    raw_future_center = centerline[len(train):]
    centered_future_center = raw_future_center * np.exp(template_log_mean * future_amp)
    centered_centerline = np.concatenate([centered_hist_center, centered_future_center])

    fitted_hist = centered_hist_center * np.exp(hist_shape * amplitude_scale)
    projected = centered_future_center * np.exp(future_shape * future_amp)

    all_dates = pd.DatetimeIndex(train["date"].tolist() + future_dates.tolist())
    daily = pd.DataFrame({
        "date": all_dates,
        "row_type": ["historical_training"] * len(train) + ["projected"] * len(future_dates),
        "actual_price_usd": list(train["price_usd"]) + [np.nan] * len(future_dates),
        "included_in_training": [True] * len(train) + [False] * len(future_dates),
        "structural_centerline_usd": centered_centerline,
        "cycle_progress": np.concatenate([hist_progress, future_progress]),
        "cycle_shape_value": np.concatenate([hist_shape, future_shape]),
        "cycle_amplitude": np.concatenate([np.full(len(train), amplitude_scale), future_amp]),
        "fitted_or_projected_price_usd": np.concatenate([fitted_hist, projected]),
        "model_version": "price-model-v2.0.1",
        "training_start_date": training_start.date().isoformat(),
        "training_end_date": training_end.date().isoformat(),
    })
    daily["cycle_phase"] = np.where(
        daily["cycle_progress"] <= cycle_diag["peak_progress"], "rising", "falling"
    )

    diagnostics = {
        **trend_diag,
        "training_rows": len(train),
        "training_start": training_start.date().isoformat(),
        "training_end": training_end.date().isoformat(),
        "projection_years": projection_years,
        "amplitude_scale": amplitude_scale,
        "amplitude_retained_per_cycle": retain_per_cycle,
        "template_log_mean": template_log_mean,
        **{k: v for k, v in cycle_diag.items() if k != "turning_points"},
        "turning_points": tp,
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
