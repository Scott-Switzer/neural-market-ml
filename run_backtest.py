"""Walk-forward backtest. Prints honest metrics, saves results/backtest.json.

Usage:  python run_backtest.py [--refresh] [--threshold 0.55] [--cost-bps 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_market_ml.backtest import run_walk_forward
from neural_market_ml.data import load_spy_daily
from neural_market_ml.features import build_features, make_target
from neural_market_ml.models import get_models

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--cost-bps", type=float, default=5.0, dest="cost_bps")
    args = ap.parse_args()

    df = load_spy_daily(refresh=args.refresh)
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
    out = RESULTS_DIR / "backtest.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
