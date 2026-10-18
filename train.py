"""Train final models on the full dataset and save them to models/.

Usage:  python train.py [--refresh]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from neural_market_ml.data import load_spy_daily
from neural_market_ml.features import build_features, make_target
from neural_market_ml.models import get_models

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download bars")
    args = ap.parse_args()

    df = load_spy_daily(refresh=args.refresh)
    print(f"bars: {len(df)} ({df.index[0].date()} -> {df.index[-1].date()})")
    X = build_features(df)
    y, _ = make_target(df, X.index)
    X, y = X.loc[y.index], y
    print(f"feature rows: {len(X)}, positives: {y.mean():.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in get_models().items():
        model.fit(X, y)
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
