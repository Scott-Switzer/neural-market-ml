"""Data loading: free daily SPY bars with a local parquet cache.

Primary source is Yahoo Finance (free, no API key). We first try the
``yfinance`` package; if that fails (some networks filter its user agent),
we fall back to the public Yahoo chart API directly with a browser
user-agent — same data, same terms. Downloaded bars are cached to
``data/spy_daily.parquet`` so repeat runs work offline.

If the network is down and no cache exists, this raises a clear
RuntimeError instead of silently producing garbage.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
from pathlib import Path

import pandas as pd

TICKER = "SPY"
START = "2015-01-01"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _yahoo_chart_api(ticker: str, start: str) -> pd.DataFrame:
    """Fetch daily OHLCV + adjclose from Yahoo's public chart endpoint."""
    period1 = int(dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
    period2 = int(dt.datetime.now(dt.timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={period1}&period2={period2}&events=div%2Csplit"
    )
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode())
    result = payload["chart"]["result"]
    if not result:
        raise RuntimeError(f"Yahoo chart API returned no result for {ticker}")
    r = result[0]
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    q = r["indicators"]["quote"][0]
    adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame(
        {
            "Open": q["open"],
            "High": q["high"],
            "Low": q["low"],
            "Close": q["close"],
            "Volume": q["volume"],
        },
        index=idx,
    )
    if adj:
        df["AdjClose"] = adj
    else:
        df["AdjClose"] = df["Close"]
    df.index.name = "Date"
    return df.dropna()


def _via_yfinance(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, timeout=30)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    return df[["Open", "High", "Low", "Close", "AdjClose", "Volume"]].dropna()


def load_spy_daily(
    ticker: str = TICKER,
    start: str = START,
    cache_dir: Path | str = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load daily bars, from cache when available, else download.

    Raises RuntimeError with a clear message if download fails and no
    cache exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker.lower()}_daily.parquet"

    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    errors: list[str] = []
    for name, fn in (("yfinance", _via_yfinance), ("yahoo-chart-api", _yahoo_chart_api)):
        try:
            df = fn(ticker, start)
            if df is not None and not df.empty:
                df.to_parquet(cache_path)
                return df
            errors.append(f"{name}: empty result")
        except Exception as e:  # noqa: BLE001 - we want the fallback chain
            errors.append(f"{name}: {e}")

    if cache_path.exists():
        # Stale cache is better than nothing; warn loudly.
        print(f"WARNING: download failed ({'; '.join(errors)}); using cached bars.")
        return pd.read_parquet(cache_path)
    raise RuntimeError(
        f"Could not download {ticker} bars and no cache exists at {cache_path}. "
        f"Errors: {'; '.join(errors)}"
    )
