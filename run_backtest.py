"""Walk-forward backtest. Prints honest metrics, saves results JSON.

Usage:
  python run_backtest.py [--refresh] [--threshold 0.55] [--cost-bps 5]
  python run_backtest.py --data-source pit --pit-path <store> \
      [--ticker AAPL] [--as-of 2018-11-02] [--out results/backtest_pit.json]

The default source is Yahoo Finance via ``neural_market_ml.data``
(unchanged behavior). The ``pit`` source loads bars from Scott's own
point-in-time stack via ``neural_market_ml.pit_data`` (PPE lakehouse
parquet, market-terminal release files, or a PPE DuckDB export).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_market_ml.backtest import run_walk_forward
from neural_market_ml.data import START, load_spy_daily
from neural_market_ml.features import build_features, make_target
from neural_market_ml.models import get_models
from neural_market_ml.pit_data import load_bars_pit

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--cost-bps", type=float, default=5.0, dest="cost_bps")
    ap.add_argument(
        "--data-source",
        choices=("yahoo", "pit"),
        default="yahoo",
        help="yahoo: free Yahoo Finance bars (default). pit: point-in-time store.",
    )
    ap.add_argument("--pit-path", type=str, default=None,
                    help="PIT store path (parquet file/dir, release file/dir, or .duckdb).")
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--start", type=str, default=START)
    ap.add_argument("--as-of", type=str, default=None,
                    help="PIT knowledge cutoff (date or ISO timestamp, UTC).")
    ap.add_argument("--out", type=str, default=None,
                    help="Results JSON path. Defaults: results/backtest.json "
                         "(yahoo) or results/backtest_pit.json (pit).")
    args = ap.parse_args()

    if args.data_source == "pit":
        if not args.pit_path:
            ap.error("--data-source pit requires --pit-path")
        df = load_bars_pit(args.pit_path, ticker=args.ticker, start=args.start,
                           as_of=args.as_of)
        print(f"data: PIT store {args.pit_path} ticker={args.ticker} "
              f"as_of={args.as_of or 'latest'} rows={len(df)} "
              f"range {df.index[0].date()} -> {df.index[-1].date()}")
        default_out = RESULTS_DIR / "backtest_pit.json"
    else:
        df = load_spy_daily(refresh=args.refresh)
        print(f"data: Yahoo Finance ticker=SPY rows={len(df)} "
              f"range {df.index[0].date()} -> {df.index[-1].date()}")
        default_out = RESULTS_DIR / "backtest.json"

    X = build_features(df)
    y, next_ret = make_target(df, X.index)
    X = X.loc[y.index]

    from sklearn.base import clone

    protos = get_models()
    factories = {name: (lambda p=proto: clone(p)) for name, proto in protos.items()}
    results = run_walk_forward(
        X, y, next_ret,
        factories,
        threshold=args.threshold,
        cost_bps=args.cost_bps,
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else default_out
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
