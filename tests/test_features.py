"""Feature tests: no-lookahead is the property that matters."""

import numpy as np
import pandas as pd
import pytest

from neural_market_ml.features import FEATURE_COLUMNS, build_features, make_target


@pytest.fixture()
def bars() -> pd.DataFrame:
    """Synthetic OHLCV with a fixed seed — deterministic across runs."""
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0.0005, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_feature_columns_present(bars):
    feats = build_features(bars)
    assert list(feats.columns) == FEATURE_COLUMNS


def test_no_nan_leakage_first_valid_row(bars):
    """After warmup, the FIRST valid row must be fully finite.

    A NaN in the first kept row would mean an indicator silently reached
    beyond the available history (or a bug in min_periods handling).
    """
    feats = build_features(bars)
    assert len(feats) > 100
    assert feats.iloc[0].notna().all(), feats.iloc[0][feats.iloc[0].isna()]
    assert np.isfinite(feats.values).all()


def test_no_lookahead_truncation(bars):
    """Features at t must not depend on data after t.

    Compute features on the full frame and on a frame truncated at T;
    every row <= T must match exactly.
    """
    full = build_features(bars)
    t = bars.index[200]
    truncated = build_features(bars.loc[:t])
    common = truncated.index
    pd.testing.assert_frame_equal(full.loc[common], truncated, check_dtype=False)


def test_no_lookahead_future_shock(bars):
    """Perturbing future bars must not change past features."""
    base = build_features(bars)
    shocked = bars.copy()
    shocked.loc[shocked.index[250:], "Close"] *= 1.5  # big future move
    shocked_feats = build_features(shocked)
    cutoff = shocked.index[249]
    pd.testing.assert_frame_equal(
        base.loc[:cutoff], shocked_feats.loc[:cutoff], check_dtype=False
    )


def test_target_alignment(bars):
    feats = build_features(bars)
    y, next_ret = make_target(bars, feats.index)
    # y[t] == 1 iff close[t+1] > close[t]
    close = bars["Close"]
    t = y.index[10]
    expected = int(close.loc[close.index[close.index.get_loc(t) + 1]] > close.loc[t])
    assert y.loc[t] == expected
    assert len(y) == len(feats) - 1  # last row has no next day
    assert y.index.equals(next_ret.index)
