from pathlib import Path
import pandas as pd
import requests

APP_DIR = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/bitcoin-fi-simulator/coinmetrics_btc_daily.csv")
URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    out["price_usd"] = pd.to_numeric(out["price_usd"], errors="coerce")
    out = out.dropna(subset=["date", "price_usd"])
    out = out[out["price_usd"] > 0]
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out[["date", "price_usd"]].reset_index(drop=True)

def fetch_coinmetrics() -> pd.DataFrame:
    params = {
        "assets": "btc",
        "metrics": "PriceUSD",
        "frequency": "1d",
        "page_size": 10000,
        "paging_from": "start",
    }
    rows = []
    next_url = URL
    while next_url:
        response = requests.get(next_url, params=params if next_url == URL else None, timeout=60)
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("data", []))
        next_url = payload.get("next_page_url")
    if not rows:
        raise RuntimeError("Coin Metrics returned no BTC PriceUSD data.")
    df = pd.DataFrame(rows).rename(columns={"time": "date", "PriceUSD": "price_usd"})
    df = normalize(df)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df

def load_coinmetrics(refresh: bool = False):
    if refresh or not CACHE.exists():
        df = fetch_coinmetrics()
        source = "Coin Metrics Community (fresh)"
    else:
        df = normalize(pd.read_csv(CACHE))
        source = "Coin Metrics Community (cached)"
    return df, {
        "source": source,
        "rows": len(df),
        "first_date": df["date"].min().date().isoformat(),
        "latest_date": df["date"].max().date().isoformat(),
        "cache_path": str(CACHE),
    }
