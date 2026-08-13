from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


CACHE = Path("/tmp/bitcoin-bottom-fair-value/coinmetrics_btc_daily.csv")
URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = (
        pd.to_datetime(data["date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    data["price_usd"] = pd.to_numeric(data["price_usd"], errors="coerce")
    data = data.dropna(subset=["date", "price_usd"])
    data = data[data["price_usd"] > 0]
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    return data[["date", "price_usd"]].reset_index(drop=True)


def fetch_coinmetrics() -> pd.DataFrame:
    params = {
        "assets": "btc",
        "metrics": "PriceUSD",
        "frequency": "1d",
        "page_size": 10000,
        "paging_from": "start",
    }
    rows: list[dict] = []
    next_url: str | None = URL
    while next_url:
        response = requests.get(
            next_url,
            params=params if next_url == URL else None,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("data", []))
        next_url = payload.get("next_page_url")
    if not rows:
        raise RuntimeError("Coin Metrics returned no Bitcoin PriceUSD observations.")

    data = normalize_prices(
        pd.DataFrame(rows).rename(columns={"time": "date", "PriceUSD": "price_usd"})
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(CACHE, index=False)
    return data


def load_coinmetrics(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    if refresh or not CACHE.exists():
        data = fetch_coinmetrics()
        source = "Coin Metrics Community (fresh)"
    else:
        data = normalize_prices(pd.read_csv(CACHE))
        source = "Coin Metrics Community (cached)"
    return data, {
        "source": source,
        "rows": len(data),
        "first_date": data["date"].min().date().isoformat(),
        "latest_date": data["date"].max().date().isoformat(),
        "cache_path": str(CACHE),
    }
