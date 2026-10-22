"""Point-in-time (PIT) data loading from Scott's own data stack.

The Yahoo path in :mod:`neural_market_ml.data` returns whatever history
Yahoo serves *today*, including every later correction. This module loads
daily bars from the point-in-time stores built by Project-Portfolio-Engine
(PPE), where every observation carries an ``available_at`` timestamp: the
moment that observation became knowable. A read with cutoff ``as_of``
returns, for each session date, the newest revision whose
``available_at <= as_of``. A correction published after ``as_of`` can never
leak into a frame read at ``as_of``. That property is proven by the
fixture tests in ``tests/test_pit_data.py``.

Three input flavors are supported:

1. **Canonical lakehouse parquet** (preferred). Parquet in PPE's
   ``daily-price-observations-v1`` schema (see PPE
   ``lakehouse/daily_prices.py``): ``instrument_id``, ``session_date``,
   ``available_at``, ``ingested_at``, decimal ``open`` / ``high`` /
   ``low`` / ``close`` / ``adjusted_close``, ``volume``, plus lineage
   columns. Accepts a single file (e.g. the serving artifact produced by
   PPE ``scripts/build_remote_daily_prices_serving.py``) or a directory
   of such files. Full revision-level PIT semantics.

2. **Market-terminal release file**. The raw per-security
   ``price_history.parquet`` files published in the
   ``financial-system-datasets`` R2 bucket under
   ``gold/market-terminal/releases/<release>/securities/<TICKER>/``.
   Columns are ``date`` plus OHLCV (OHLC stored as decimal strings).
   These files carry no availability timestamp, so availability is
   synthesized with PPE's own canonicalization rule
   (``lakehouse/daily_prices.py::normalize_price_file``): a session's bar
   becomes available at the 16:00 America/New_York close of its session
   date. One revision per session exists in this flavor. A release root
   directory is also accepted; the loader resolves
   ``securities/<TICKER>/price_history.parquet`` inside it.

3. **PPE legacy DuckDB export**. An ``engine.duckdb`` file with PPE's
   ``prices`` table (``ticker, date, open, high, low, close, adj_close,
   volume``; see PPE ``engine/store.py``). WARNING: this schema stores
   one row per (ticker, date) with NO availability timestamp, so
   revision-level PIT cannot be proven from it. ``as_of`` is applied at
   date granularity only, and the frame reflects whatever values were in
   the store when it was exported. Prefer flavor 1 when it exists.

Output contract is identical to :func:`neural_market_ml.data.load_spy_daily`:
a DataFrame indexed by tz-naive ``Date`` (sorted ascending) with float
columns ``Open, High, Low, Close, AdjClose, Volume`` and no NaNs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
CANONICAL_REQUIRED = {"instrument_id", "session_date", "available_at", "close"}
RELEASE_REQUIRED = {"date", "close"}
_CLOSE_LOCAL = dt.time(16, 0)  # NYSE close, America/New_York


def _parse_as_of(as_of) -> pd.Timestamp | None:
    """Normalize an as_of cutoff to a tz-aware UTC Timestamp.

    Date-only values mean "end of that calendar day, UTC". Naive
    datetimes are interpreted as UTC. ``None`` means no cutoff (latest
    known state, matching the Yahoo path's behavior).
    """
    if as_of is None:
        return None
    if isinstance(as_of, pd.Timestamp):
        ts = as_of
    elif isinstance(as_of, dt.datetime):
        ts = pd.Timestamp(as_of)
    elif isinstance(as_of, dt.date):
        ts = pd.Timestamp(as_of) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    elif isinstance(as_of, str):
        s = as_of.strip()
        if len(s) == 10:  # YYYY-MM-DD
            return _parse_as_of(dt.date.fromisoformat(s))
        ts = pd.Timestamp(s)
    else:
        raise TypeError(f"Unsupported as_of type: {type(as_of)!r}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_float(series: pd.Series) -> pd.Series:
    """Convert Decimal / string / numeric series to float, NaN on junk."""
    return pd.to_numeric(series, errors="coerce").astype(float)


def _finalize(df: pd.DataFrame, ticker: str, start: str, source_desc: str) -> pd.DataFrame:
    """Apply the shared output contract (index, columns, order, NaNs)."""
    if df.empty:
        raise RuntimeError(
            f"No bars for {ticker} at or after {start} in {source_desc}. "
            "Check the ticker spelling and the store's coverage."
        )
    df = df[df.index >= pd.Timestamp(start)]
    df = df[OUTPUT_COLUMNS].dropna().sort_index()
    if df.empty:
        raise RuntimeError(
            f"Bars for {ticker} exist in {source_desc} but none survive "
            f"the start={start} filter with complete OHLCV."
        )
    df.index.name = "Date"
    return df


def _select_pit_revisions(df: pd.DataFrame, cutoff: pd.Timestamp | None) -> pd.DataFrame:
    """Per session_date, keep the newest revision available at cutoff.

    Rows are observations with parsed ``_session`` (Timestamp),
    ``_available`` and ``_ingested`` (tz-aware UTC) columns. Sorting is
    fully deterministic: availability, then ingestion, then id.
    """
    if cutoff is not None:
        df = df[df["_available"] <= cutoff]
    if df.empty:
        return df
    sort_cols = ["_session", "_available", "_ingested"]
    if "_obs_id" in df.columns:
        sort_cols.append("_obs_id")
    df = df.sort_values(sort_cols, kind="mergesort")
    return df.drop_duplicates(subset=["_session"], keep="last")


def _bars_from_observations(obs: pd.DataFrame) -> pd.DataFrame:
    """Map selected observations to the output contract's columns."""
    # NOTE: build from raw arrays, never Series. A Series would be
    # label-aligned against the new DatetimeIndex and silently become NaN.
    close = _to_float(obs["close"]).to_numpy()
    if "adjusted_close" in obs:
        adj = _to_float(obs["adjusted_close"]).to_numpy()
        adj = np.where(np.isnan(adj), close, adj)  # Yahoo path's fallback
    else:
        adj = close
    out = pd.DataFrame(
        {
            "Open": _to_float(obs["open"]).to_numpy() if "open" in obs else close,
            "High": _to_float(obs["high"]).to_numpy() if "high" in obs else close,
            "Low": _to_float(obs["low"]).to_numpy() if "low" in obs else close,
            "Close": close,
            "AdjClose": adj,
            "Volume": (
                _to_float(obs["volume"]).to_numpy()
                if "volume" in obs
                else np.full(len(obs), np.nan)
            ),
        },
        index=pd.DatetimeIndex(obs["_session"].to_numpy()),
    )
    return out


def _load_canonical(files: list[Path], ticker: str, start: str, cutoff) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    for f in files:
        table = pq.read_table(f)
        missing = CANONICAL_REQUIRED - set(table.column_names)
        if missing:
            raise RuntimeError(f"{f} is missing canonical columns: {sorted(missing)}")
        frames.append(table.to_pandas())
    obs = pd.concat(frames, ignore_index=True)
    obs = obs[obs["instrument_id"].astype(str).str.upper() == ticker.upper()]
    if obs.empty:
        raise RuntimeError(
            f"No canonical observations for {ticker} in {[str(f) for f in files]}. "
            "The bounded lakehouse pilot only covers AAPL/MSFT/NVDA."
        )
    obs = obs.copy()
    obs["_session"] = pd.to_datetime(obs["session_date"]).dt.normalize()
    obs["_available"] = pd.to_datetime(obs["available_at"], utc=True)
    obs["_ingested"] = (
        pd.to_datetime(obs["ingested_at"], utc=True)
        if "ingested_at" in obs
        else obs["_available"]
    )
    obs["_obs_id"] = obs["observation_id"].astype(str) if "observation_id" in obs else ""
    selected = _select_pit_revisions(obs, cutoff)
    return _finalize(_bars_from_observations(selected), ticker, start, str(files[0]))


def _load_release_file(file: Path, ticker: str, start: str, cutoff) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(file)
    missing = RELEASE_REQUIRED - set(table.column_names)
    if missing:
        raise RuntimeError(f"{file} is missing release columns: {sorted(missing)}")
    raw = table.to_pandas()
    sessions = pd.to_datetime(raw["date"].astype(str).str.slice(0, 10)).dt.normalize()
    # PPE's canonicalization rule: available at the 16:00 ET session close.
    local_close = (sessions + pd.Timedelta(hours=_CLOSE_LOCAL.hour)).dt.tz_localize(
        "America/New_York"
    )
    obs = pd.DataFrame(
        {
            "_session": sessions,
            "_available": local_close.dt.tz_convert("UTC"),
            "_ingested": local_close.dt.tz_convert("UTC"),
            "open": raw.get("open"),
            "high": raw.get("high"),
            "low": raw.get("low"),
            "close": raw.get("close"),
            "adjusted_close": raw.get("adjusted_close"),
            "volume": raw.get("volume"),
        }
    )
    selected = _select_pit_revisions(obs, cutoff)
    return _finalize(_bars_from_observations(selected), ticker, start, str(file))


def _load_duckdb(path: Path, ticker: str, start: str, cutoff) -> pd.DataFrame:
    try:
        import duckdb
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "Reading a PPE DuckDB export requires the 'duckdb' package "
            "(pip install duckdb, or use the canonical parquet flavor)."
        ) from e
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "prices" not in tables:
            raise RuntimeError(f"{path} has no 'prices' table (found: {sorted(tables)})")
        rows = con.execute(
            "SELECT date, open, high, low, close, adj_close, volume "
            "FROM prices WHERE upper(ticker) = upper(?) ORDER BY date",
            [ticker],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise RuntimeError(f"No rows for {ticker} in the prices table of {path}.")
    raw = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
    sessions = pd.to_datetime(raw["date"]).dt.normalize()
    if cutoff is not None:
        # Date granularity only: the legacy schema has no availability
        # timestamps, so this is a session-date cutoff, not a true PIT read.
        raw = raw[sessions <= cutoff.tz_localize(None).normalize()]
        sessions = pd.to_datetime(raw["date"]).dt.normalize()
    out = pd.DataFrame(
        {
            "Open": _to_float(raw["open"]).to_numpy(),
            "High": _to_float(raw["high"]).to_numpy(),
            "Low": _to_float(raw["low"]).to_numpy(),
            "Close": _to_float(raw["close"]).to_numpy(),
            "AdjClose": _to_float(raw["adj_close"]).to_numpy(),
            "Volume": _to_float(raw["volume"]).to_numpy(),
        },
        index=pd.DatetimeIndex(sessions.to_numpy()),
    )
    # One row per (ticker, date) by schema; keep last on any duplicates.
    out = out[~out.index.duplicated(keep="last")]
    return _finalize(out, ticker, start, str(path))


def load_bars_pit(
    path: Path | str,
    ticker: str = "SPY",
    start: str = "2015-01-01",
    as_of=None,
) -> pd.DataFrame:
    """Load daily bars for ``ticker`` from a point-in-time store.

    Parameters
    ----------
    path:
        Canonical parquet file or directory, market-terminal release
        file or release root directory, or a PPE ``engine.duckdb`` file.
    ticker:
        Symbol to load (case-insensitive). Note the current lakehouse
        pilot and market-terminal release do NOT contain SPY; see
        DATA_ACCESS.md.
    start:
        First session date to include (``YYYY-MM-DD``).
    as_of:
        Knowledge cutoff. Only observations with ``available_at <=
        as_of`` are visible, and per session the newest such revision
        wins. ``None`` reads the latest known state. Date-only values
        mean end of that day, UTC.

    Returns the same contract as ``load_spy_daily``: tz-naive ``Date``
    index, float ``Open, High, Low, Close, AdjClose, Volume``, no NaNs.
    Raises RuntimeError with a precise message on any coverage or
    schema problem, never silently returning partial garbage.
    """
    path = Path(path)
    cutoff = _parse_as_of(as_of)
    if not path.exists():
        raise RuntimeError(f"PIT store path does not exist: {path}")

    if path.is_file() and path.suffix == ".duckdb":
        return _load_duckdb(path, ticker, start, cutoff)

    if path.is_dir():
        release_file = path / "securities" / ticker.upper() / "price_history.parquet"
        if release_file.exists():
            return _load_release_file(release_file, ticker, start, cutoff)
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise RuntimeError(f"No parquet files under PIT store directory: {path}")
        return _load_canonical(files, ticker, start, cutoff)

    if path.is_file() and path.suffix == ".parquet":
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(path).schema_arrow.names)
        if CANONICAL_REQUIRED <= names:
            return _load_canonical([path], ticker, start, cutoff)
        if RELEASE_REQUIRED <= names:
            return _load_release_file(path, ticker, start, cutoff)
        raise RuntimeError(
            f"{path} matches neither the canonical lakehouse schema "
            f"(needs {sorted(CANONICAL_REQUIRED)}) nor the release schema "
            f"(needs {sorted(RELEASE_REQUIRED)}). Columns found: {sorted(names)}"
        )

    raise RuntimeError(f"Unsupported PIT store path: {path}")


__all__ = ["load_bars_pit", "OUTPUT_COLUMNS"]
