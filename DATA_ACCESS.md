# Data access: point-in-time path status

Status as of 2026-10-21, verified from the assistant's VM. This file
records exactly what the PIT loader (`neural_market_ml/pit_data.py`) can
and cannot read today, and what is needed for a like-for-like SPY rerun
on point-in-time data.

## What works

- **Bucket `financial-system-datasets` (Cloudflare R2): listing and
  object GET both work** with the current Cloudflare token. This is the
  bucket that holds the market-terminal releases and the R2 Data
  Catalog (Iceberg) tables.
- The only market-terminal prices release is
  `gold/market-terminal/releases/19c7c32e43fb83f3e4e4efeb1e9b5c3eec484b10581d6e0b3b3e9d48189275ee`,
  with one `securities/<TICKER>/price_history.parquet` per symbol.
  The AAPL file was downloaded and run end to end through the PIT
  loader and the walk-forward backtest (see README, "Point-in-time
  run"). Verification cross-checks, all passing:
  - 968 AAPL bars from 2015-01-02, exactly the count PPE publishes for
    AAPL 2015-2018 in `docs/PRICES_DAILY_LAKEHOUSE_V1.md`.
  - First bar (2015-01-02) close 109.33000183105469 and volume
    53,204,600, exactly PPE's published canary values.
  - Loader closes are bit-identical to the raw release file.

## The blocker for a SPY rerun

**No SPY daily bars exist in any in-stack source reachable from this
VM, and every reachable price series ends 2018-11-02.**

- The market-terminal release has no `securities/SPY/` entry (listing
  checked 2026-10-21; SPYD and SPYX exist, SPY does not). QQQ is also
  absent (QQQC and QQQX exist). The release data ends 2018-11-02.
- The canonical lakehouse pilot (`daily_prices_v1`, R2 Data Catalog)
  covers only AAPL, MSFT, and NVDA, and also ends 2018-11-02.
- PPE's legacy DuckDB store (`engine/engine.duckdb`) might contain
  more, but it is a local-only artifact that is deliberately not in
  Git (see PPE `engine/local_store.py`), and no copy exists on this
  VM.
- The `financial-data-wrds-lakehouse` bucket contains only small R2
  Data Catalog pilot tables (data files of about 1.3 KB); no usable
  daily price history was found there. The
  `financial-data-wrds-private` bucket still returns 404 on object GET
  with the current token (listing works), the blocker recorded in
  earlier WRDS work.

## What Scott or Codex can provide (any one unblocks the SPY rerun)

1. **A market-terminal release or lakehouse export that includes
   SPY**, published to `financial-system-datasets`. No token change is
   needed; GET already works there. Extending coverage past 2018 would
   also fix the short-window problem. The loader reads it with no code
   change (flavor 1 or 2 in `pit_data.py`).
2. **A copy of PPE's `engine.duckdb`**, or a parquet export of its
   `prices` table, at any path this repo can read. Handled by loader
   flavor 3. Caveat: that schema has no `available_at`, so the read is
   date-granular, not revision-level PIT.
3. **A serving artifact with SPY**, produced by extending PPE
   `scripts/build_remote_daily_prices_serving.py` beyond the
   AAPL/MSFT/NVDA pilot. Handled by loader flavor 1.

If future data lands only in `financial-data-wrds-private`, that
bucket's object GET (404 today) must be fixed first, either token
scope or bucket policy.
