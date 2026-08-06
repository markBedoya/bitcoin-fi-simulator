from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.price_model import fit_price_model


def main() -> None:
    dates = pd.date_range("2015-01-01", "2026-08-05", freq="D")
    t = np.arange(len(dates), dtype=float)
    # Positive synthetic path with structural growth plus an oscillation.
    price = 250.0 * np.exp(0.0012 * t) * np.exp(0.35 * np.sin(2 * np.pi * t / 1461.0))
    prices = pd.DataFrame({"date": dates, "price_usd": price})

    result = fit_price_model(
        prices,
        pd.Timestamp("2018-12-31"),
        pd.Timestamp("2026-08-05"),
        10,
    )
    daily = result.daily
    hist = daily[daily["row_type"] == "historical_training"]
    proj = daily[daily["row_type"] == "projected"]

    actual = float(hist["actual_price_usd"].iloc[-1])
    fitted = float(hist["fitted_or_projected_price_usd"].iloc[-1])
    assert abs(fitted / actual - 1.0) < 1e-12

    assert hist["structural_centerline_usd"].notna().all()
    assert proj["structural_centerline_usd"].notna().all()
    assert float(hist["structural_centerline_usd"].iloc[-1]) > 0
    assert float(proj["structural_centerline_usd"].iloc[0]) > 0

    print("Price Model continuity test passed.")


if __name__ == "__main__":
    main()
