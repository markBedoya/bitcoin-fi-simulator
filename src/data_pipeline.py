from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CACHE = Path("/tmp/bitcoin-bottom-fair-value/coinmetrics_btc_daily.csv")
CACHE_MAX_AGE = pd.Timedelta(hours=24)
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
    import requests

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


def cache_is_stale(cache_path: Path = CACHE, now: datetime | None = None) -> bool:
    if not cache_path.exists():
        return True
    checked_at = now or datetime.now(timezone.utc)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    modified_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    return checked_at - modified_at >= CACHE_MAX_AGE.to_pytimedelta()


def load_coinmetrics(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    stale = cache_is_stale(CACHE)
    if refresh or stale:
        data = fetch_coinmetrics()
        source = "Coin Metrics Community (fresh — manual or daily refresh)"
    else:
        data = normalize_prices(pd.read_csv(CACHE))
        source = "Coin Metrics Community (cached)"
    return data, {
        "source": source,
        "rows": len(data),
        "first_date": data["date"].min().date().isoformat(),
        "latest_date": data["date"].max().date().isoformat(),
        "cache_path": str(CACHE),
        "cache_max_age_hours": int(CACHE_MAX_AGE / pd.Timedelta(hours=1)),
        "automatic_refresh_enabled": True,
    }
