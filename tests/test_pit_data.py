"""PIT loader tests: the property that matters is no future knowledge.

Fixtures are built in PPE's real schemas:
- canonical ``daily-price-observations-v1`` parquet (decimal OHLC,
  ``available_at`` / ``ingested_at`` ISO strings), as produced by PPE
  ``lakehouse/daily_prices.py``;
- the raw market-terminal release ``price_history.parquet`` shape
  (OHLC as decimal strings), as published in R2;
- a DuckDB file with PPE ``engine/store.py``'s ``prices`` table.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neural_market_ml.features import build_features
from neural_market_ml.pit_data import load_bars_pit

CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.string(), nullable=False),
        pa.field("world_id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("session_date", pa.date32(), nullable=False),
        pa.field("available_at", pa.string(), nullable=False),
        pa.field("ingested_at", pa.string(), nullable=False),
        pa.field("open", pa.decimal128(38, 24)),
        pa.field("high", pa.decimal128(38, 24)),
        pa.field("low", pa.decimal128(38, 24)),
        pa.field("close", pa.decimal128(38, 24), nullable=False),
        pa.field("adjusted_close", pa.decimal128(38, 24)),
        pa.field("volume", pa.int64()),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("evidence_id", pa.string(), nullable=False),
        pa.field("quality_status", pa.string(), nullable=False),
        pa.field("producer_release", pa.string(), nullable=False),
    ]
)


def _canonical_row(symbol, session, close, available_at, *, adjusted=None, obs_suffix="a"):
    """One canonical observation; OHLC derived from close for brevity."""
    session = dt.date.fromisoformat(session)
    c = Decimal(str(close))
    return {
        "observation_id": f"obs-{symbol}-{session}-{obs_suffix}",
        "world_id": "real",
        "entity_id": f"real:equity:{symbol}",
        "instrument_id": symbol,
        "session_date": session,
        "available_at": available_at,
        "ingested_at": available_at,
        "open": c,
        "high": c,
        "low": c,
        "close": c,
        "adjusted_close": None if adjusted is None else Decimal(str(adjusted)),
        "volume": 1_000_000,
        "unit": "USD/share",
        "currency": "USD",
        "source_id": "fixture",
        "source_record_id": f"{symbol}:{session}:fixture",
        "evidence_id": f"ev-{symbol}-{session}-{obs_suffix}",
        "quality_status": "REPORTED",
        "producer_release": "fixture-release",
    }


def _write_canonical(tmp_path, rows, name="daily_prices.parquet"):
    table = pa.Table.from_pylist(rows, schema=CANONICAL_SCHEMA)
    path = tmp_path / name
    pq.write_table(table, path)
    return path


# A session (2021-01-04) whose close is corrected two days later, plus a
# normal session (2021-01-05). Original close 100, correction 101.
REVISION_ROWS = [
    _canonical_row("SPY", "2021-01-04", 100.0, "2021-01-04T21:00:00Z"),
    _canonical_row("SPY", "2021-01-04", 101.0, "2021-01-06T14:00:00Z", obs_suffix="b"),
    _canonical_row("SPY", "2021-01-05", 102.0, "2021-01-05T21:00:00Z"),
]


def test_canonical_loads_with_loader_contract(tmp_path):
    path = _write_canonical(
        tmp_path,
        [
            _canonical_row("SPY", "2021-01-04", 100.0, "2021-01-04T21:00:00Z", adjusted=99.5),
            _canonical_row("SPY", "2021-01-05", 102.0, "2021-01-05T21:00:00Z", adjusted=101.5),
            _canonical_row("AAPL", "2021-01-04", 130.0, "2021-01-04T21:00:00Z"),
        ],
    )
    df = load_bars_pit(path, ticker="SPY", start="2021-01-01")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    assert df.index.name == "Date"
    assert df.index.is_monotonic_increasing
    assert list(df.index) == [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")]
    assert df.loc["2021-01-04", "Close"] == pytest.approx(100.0)
    assert df.loc["2021-01-04", "AdjClose"] == pytest.approx(99.5)
    assert df["Close"].dtype == float


def test_canonical_adjclose_falls_back_to_close(tmp_path):
    path = _write_canonical(
        tmp_path, [_canonical_row("SPY", "2021-01-04", 100.0, "2021-01-04T21:00:00Z")]
    )
    df = load_bars_pit(path, ticker="SPY", start="2021-01-01")
    assert df.loc["2021-01-04", "AdjClose"] == pytest.approx(100.0)


def test_pit_revision_hidden_until_available(tmp_path):
    """The corrected close (101) must not appear in a read taken before
    the correction was published (2021-01-06T14:00Z)."""
    path = _write_canonical(tmp_path, REVISION_ROWS)
    df = load_bars_pit(path, ticker="SPY", start="2021-01-01", as_of="2021-01-05")
    assert df.loc["2021-01-04", "Close"] == pytest.approx(100.0)


def test_pit_revision_visible_after_available(tmp_path):
    path = _write_canonical(tmp_path, REVISION_ROWS)
    df = load_bars_pit(path, ticker="SPY", start="2021-01-01", as_of="2021-01-07")
    assert df.loc["2021-01-04", "Close"] == pytest.approx(101.0)
    # Latest-known-state read (no cutoff) also sees the correction.
    df_now = load_bars_pit(path, ticker="SPY", start="2021-01-01")
    assert df_now.loc["2021-01-04", "Close"] == pytest.approx(101.0)


def test_pit_truncation_equivalence(tmp_path):
    """load(as_of=T) == load of a store physically truncated to rows
    available by T. This is the loader-level no-lookahead proof."""
    full = _write_canonical(tmp_path, REVISION_ROWS, name="full.parquet")
    truncated_rows = [r for r in REVISION_ROWS if r["available_at"] <= "2021-01-05T23:59:59Z"]
    truncated = _write_canonical(tmp_path, truncated_rows, name="truncated.parquet")
    a = load_bars_pit(full, ticker="SPY", start="2021-01-01", as_of="2021-01-05")
    b = load_bars_pit(truncated, ticker="SPY", start="2021-01-01")
    pd.testing.assert_frame_equal(a, b)


def test_pit_backfilled_session_absent_from_earlier_read(tmp_path):
    """A bar for an old session that only enters the store later (a
    backfill) must not appear in reads taken before it arrived."""
    rows = [
        _canonical_row("SPY", "2021-01-05", 102.0, "2021-01-05T21:00:00Z"),
        # Session 2021-01-04 backfilled on 2021-01-20:
        _canonical_row("SPY", "2021-01-04", 100.0, "2021-01-20T15:00:00Z"),
    ]
    path = _write_canonical(tmp_path, rows)
    early = load_bars_pit(path, ticker="SPY", start="2021-01-01", as_of="2021-01-10")
    assert list(early.index) == [pd.Timestamp("2021-01-05")]
    late = load_bars_pit(path, ticker="SPY", start="2021-01-01", as_of="2021-01-21")
    assert list(late.index) == [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")]


def test_release_file_shape_and_close_time_availability(tmp_path):
    """Raw release parquet (OHLC as strings). Availability is the
    16:00 ET session close per PPE's canonicalization rule: on a winter
    (EST) day that is 21:00Z, so a 17:00Z read must NOT see that day's
    bar, and a 22:00Z read must."""
    raw = pd.DataFrame(
        {
            "date": ["2021-01-14", "2021-01-15"],
            "open": ["100.0", "101.0"],
            "high": ["101.0", "102.0"],
            "low": ["99.0", "100.0"],
            "close": [100.5, 101.5],
            "volume": [1_000_000.0, 1_100_000.0],
            "adjusted_close": ["100.5", "101.5"],
        }
    )
    path = tmp_path / "price_history.parquet"
    raw.to_parquet(path)

    before_close = load_bars_pit(
        path, ticker="SPY", start="2021-01-01", as_of="2021-01-15T17:00:00Z"
    )
    assert list(before_close.index) == [pd.Timestamp("2021-01-14")]
    after_close = load_bars_pit(
        path, ticker="SPY", start="2021-01-01", as_of="2021-01-15T22:00:00Z"
    )
    assert list(after_close.index) == [pd.Timestamp("2021-01-14"), pd.Timestamp("2021-01-15")]
    assert after_close.loc["2021-01-15", "Close"] == pytest.approx(101.5)


def test_duckdb_export_loads_and_maps(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "engine.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE prices(ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE, "
        "low DOUBLE, close DOUBLE, adj_close DOUBLE, volume DOUBLE)"
    )
    con.execute(
        "INSERT INTO prices VALUES "
        "('SPY', DATE '2021-01-04', 100, 101, 99, 100.5, 99.9, 1000000), "
        "('SPY', DATE '2021-01-05', 101, 102, 100, 101.5, 100.9, 1100000), "
        "('AAPL', DATE '2021-01-04', 130, 131, 129, 130.5, 130.1, 900000)"
    )
    con.close()
    df = load_bars_pit(path, ticker="SPY", start="2021-01-01")
    assert list(df.index) == [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")]
    assert df.loc["2021-01-04", "AdjClose"] == pytest.approx(99.9)  # adj_close -> AdjClose


def test_missing_ticker_raises_clear_error(tmp_path):
    path = _write_canonical(
        tmp_path, [_canonical_row("AAPL", "2021-01-04", 130.0, "2021-01-04T21:00:00Z")]
    )
    with pytest.raises(RuntimeError, match="AAPL/MSFT/NVDA|No canonical observations"):
        load_bars_pit(path, ticker="SPY", start="2021-01-01")


def test_pit_frame_features_stay_causal(tmp_path):
    """End to end: features built on a PIT frame obey the same
    truncation property as features on Yahoo bars."""
    rng = np.random.default_rng(11)
    n = 160
    sessions = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    rows = []
    for s, c in zip(sessions, close):
        avail = (s + pd.Timedelta(hours=21)).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = _canonical_row("SPY", s.strftime("%Y-%m-%d"), float(c), avail)
        row["volume"] = int(rng.integers(1_000_000, 5_000_000))
        rows.append(row)
    path = _write_canonical(tmp_path, rows)
    bars = load_bars_pit(path, ticker="SPY", start="2021-01-01")
    full = build_features(bars)
    t = bars.index[120]
    truncated = build_features(bars.loc[:t])
    pd.testing.assert_frame_equal(full.loc[truncated.index], truncated, check_dtype=False)
