from __future__ import annotations

import hashlib
import json
import pandas as pd


def build_model_fingerprint(
    training_start,
    training_end,
    projection_years: int,
    latest_data_date,
    model_daily: pd.DataFrame,
) -> str:
    projected = model_daily[model_daily["row_type"] == "projected"]
    historical = model_daily[model_daily["row_type"] == "historical_training"]

    samples = []
    for frame, column in [
        (historical, "structural_centerline_usd"),
        (projected, "structural_centerline_usd"),
        (projected, "fitted_or_projected_price_usd"),
    ]:
        if not frame.empty:
            indices = sorted(
                set([0, len(frame) // 4, len(frame) // 2, 3 * len(frame) // 4, len(frame) - 1])
            )
            samples.extend(round(float(frame.iloc[i][column]), 8) for i in indices)

    payload = {
        "training_start": pd.Timestamp(training_start).date().isoformat(),
        "training_end": pd.Timestamp(training_end).date().isoformat(),
        "projection_years": int(projection_years),
        "latest_data_date": pd.Timestamp(latest_data_date).date().isoformat(),
        "samples": samples,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
