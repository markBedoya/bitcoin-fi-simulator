from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.price_model import fit_bottom_anchored_model  # noqa: E402


def synthetic_prices() -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", "2026-08-12", freq="D")
    genesis = pd.Timestamp("2009-01-03")
    days = (dates - genesis).days.to_numpy(dtype=float)
    floor = 0.42 * 1.0117e-17 * np.power(days, 5.82)
    progress = np.mod(np.arange(len(dates)), 1431) / 1431.0
    cycle_premium = np.exp(1.15 * np.square(np.sin(np.pi * progress)))
    price = np.maximum(floor * cycle_premium, 0.01)
    return pd.DataFrame({"date": dates, "price_usd": price})


def main() -> None:
    result = fit_bottom_anchored_model(synthetic_prices())
    assert len(result.bottom_regions) == 5
    assert len(result.peak_regions) == 4
    assert len(result.candidate_forecasts) == 4
    assert np.isclose(result.candidate_forecasts["ensemble_weight"].sum(), 1.0)
    assert result.summary["experimental_fair_value_usd"] > 0
    assert result.summary["next_bottom_candidate_low_usd"] > 0
    assert result.summary["next_bottom_candidate_high_usd"] >= result.summary["next_bottom_candidate_low_usd"]
    assert result.summary["status"] == "RESEARCH_ONLY"
    assert result.curve["date"].max() == result.summary["next_bottom_anchor"]
    assert result.curve[[
        "bottom_foundation_usd",
        "experimental_fair_value_usd",
        "published_fair_value_usd",
    ]].gt(0).all().all()
    assert not result.walk_forward.empty
    print("Bottom-anchored price-model checks passed.")


if __name__ == "__main__":
    main()
