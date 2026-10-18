"""Backtest tests: fold hygiene and cost accounting."""

import numpy as np
import pandas as pd
import pytest

from neural_market_ml.backtest import (
    performance_metrics,
    quarter_starts,
    signal_stats,
    strategy_returns,
    walk_forward_splits,
)


@pytest.fixture()
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", "2024-12-31")


def test_walk_forward_monotonic_no_overlap(dates):
    splits = walk_forward_splits(dates, warmup_years=3.0)
    assert len(splits) > 20
    prev_test_end = None
    for train_idx, test_idx in splits:
        # train strictly before test
        assert train_idx.max() < test_idx.min()
        # test folds strictly increasing and non-overlapping
        if prev_test_end is not None:
            assert test_idx.min() > prev_test_end
        prev_test_end = test_idx.max()
        # train is expanding
        assert len(train_idx) > 50


def test_quarter_starts_sorted_unique(dates):
    starts = quarter_starts(dates, dates[0] + pd.DateOffset(years=3))
    assert starts == sorted(set(starts))
    assert all(s >= dates[0] + pd.DateOffset(years=3) for s in starts)


def test_cost_accounting_round_trip():
    """A 0->1->0 round trip must deduct exactly 2 * 5bps."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    signals = pd.Series([0, 0, 1, 1, 0, 0], index=idx)
    daily_ret = pd.Series(np.zeros(6), index=idx)  # zero market return isolates costs
    net = strategy_returns(signals, daily_ret, cost_bps=5.0)
    total_cost = float(-net.sum())
    assert total_cost == pytest.approx(2 * 5.0 / 10_000.0)


def test_cost_accounting_flat_signal_no_cost():
    idx = pd.bdate_range("2024-01-01", periods=10)
    signals = pd.Series(np.ones(10, dtype=int), index=idx)
    daily_ret = pd.Series(np.full(10, 0.001), index=idx)
    net = strategy_returns(signals, daily_ret, cost_bps=5.0)
    # Position decided at close of day 0 applies from day 1; day 0 earns 0.
    assert net.iloc[0] == pytest.approx(0.0)
    assert np.allclose(net.iloc[1:].values, 0.001, rtol=1e-9, atol=1e-12)


def test_position_timing_one_day_lag():
    """Signal at close t earns the return of day t+1, not day t."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    signals = pd.Series([0, 1, 1], index=idx)  # go long at close of day 1
    daily_ret = pd.Series([0.10, 0.20, 0.30], index=idx)
    net = strategy_returns(signals, daily_ret, cost_bps=0.0)
    # Day 0: flat. Day 1: signal from day 0 = 0 -> earns 0. Day 2: earns 0.30.
    assert net.iloc[0] == pytest.approx(0.0)
    assert net.iloc[1] == pytest.approx(0.0)
    assert net.iloc[2] == pytest.approx(0.30)


def test_performance_metrics_known_values():
    idx = pd.bdate_range("2024-01-01", periods=252)
    net = pd.Series(np.full(252, 0.001), index=idx)  # constant +10bps/day
    m = performance_metrics(net)
    assert m["total_return"] == pytest.approx((1.001) ** 252 - 1, rel=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    # constant returns -> zero variance -> sharpe defined as 0 by convention
    assert m["annualized_sharpe"] == 0.0


def test_performance_metrics_drawdown():
    idx = pd.bdate_range("2024-01-01", periods=4)
    net = pd.Series([0.0, 0.10, -0.20, 0.0], index=idx)
    m = performance_metrics(net)
    # peak 1.10, trough 0.88 -> dd = 0.88/1.10 - 1 = -0.20
    assert m["max_drawdown"] == pytest.approx(-0.20, rel=1e-9)


def test_signal_stats():
    idx = pd.bdate_range("2024-01-01", periods=4)
    signals = pd.Series([1, 1, 0, 0], index=idx)
    y = pd.Series([1, 0, 0, 1], index=idx)
    s = signal_stats(signals, y)
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["precision_on_longs"] == pytest.approx(0.5)
    assert s["n_longs"] == 2
