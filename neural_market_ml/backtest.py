"""Walk-forward backtest engine with honest cost accounting.

Protocol
--------
* Expanding window: for each test quarter, train on ALL data before it
  (after an initial warmup), predict the quarter, then expand. Retraining
  is quarterly, not daily, for speed. Test folds are strictly increasing
  and never overlap; the train set never contains the test period.
* Never shuffle-split a time series.
* Signal: long (1) when P(up) > ``threshold`` (default 0.55), else flat (0).
* Position timing: the signal computed at the close of day t applies to
  day t+1's return. In code: ``position(t) = signal(t-1)``.
* Costs: 5 bps per one-way trade, applied on every position change:
  ``cost(t) = |signal(t) - signal(t-1)| * 5bps``. A 0 -> 1 -> 0 round trip
  therefore deducts exactly 2 * 5bps (covered by a unit test).

Metrics: hit rate (accuracy), precision on long signals, total return,
annualized Sharpe, max drawdown — all net of costs. Buy-and-hold over
the same window is reported as the benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def quarter_starts(dates: pd.DatetimeIndex, first_test: pd.Timestamp) -> list[pd.Timestamp]:
    """Quarter starts >= first_test, strictly increasing, within the data."""
    starts = pd.date_range(first_test, dates[-1], freq="QS")
    return [s for s in starts if s <= dates[-1]]


def walk_forward_splits(
    dates: pd.DatetimeIndex, warmup_years: float = 3.0
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Return [(train_idx, test_idx), ...] with expanding train windows."""
    first_test = dates[0] + pd.DateOffset(years=warmup_years)
    splits = []
    for qstart in quarter_starts(dates, first_test):
        qend = qstart + pd.DateOffset(months=3)
        train_idx = dates[dates < qstart]
        test_idx = dates[(dates >= qstart) & (dates < qend)]
        if len(train_idx) > 50 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def strategy_returns(
    signals: pd.Series, daily_ret: pd.Series, cost_bps: float = 5.0
) -> pd.Series:
    """Net daily strategy returns from a 0/1 signal series.

    ``signals[t]`` is the position decided at the close of day ``t`` and
    applies to the return of day ``t+1``. Costs are charged on changes.
    """
    signals = signals.reindex(daily_ret.index).fillna(0).astype(int)
    position = signals.shift(1).fillna(0)  # held during day t, decided at close t-1
    gross = position * daily_ret
    trades = signals.diff().abs().fillna(0)  # |signal(t) - signal(t-1)|
    costs = trades * (cost_bps / 10_000.0)
    return (gross - costs).rename("strategy_net")


def performance_metrics(net_ret: pd.Series) -> dict:
    """Total return, annualized Sharpe, max drawdown from net daily returns."""
    net_ret = net_ret.dropna()
    n = len(net_ret)
    total = float((1.0 + net_ret).prod() - 1.0)
    ann_factor = TRADING_DAYS / n if n else 0.0
    mean, std = float(net_ret.mean()), float(net_ret.std())
    # Near-zero variance (e.g. constant returns up to float noise) -> 0 by convention.
    sharpe = float(mean / std * np.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0
    cum = (1.0 + net_ret).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "total_return": total,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_days": n,
    }


def signal_stats(signals: pd.Series, y: pd.Series) -> dict:
    """Hit rate and precision-on-longs for out-of-sample signals."""
    s, yy = signals.align(y, join="inner")
    s = s.astype(int)
    hit_rate = float((s == yy).mean()) if len(s) else float("nan")
    n_longs = int(s.sum())
    precision_long = float(yy[s == 1].mean()) if n_longs else float("nan")
    return {
        "hit_rate": hit_rate,
        "precision_on_longs": precision_long,
        "n_signals": len(s),
        "n_longs": n_longs,
    }


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    next_ret: pd.Series,
    models: dict,
    threshold: float = 0.55,
    cost_bps: float = 5.0,
    warmup_years: float = 3.0,
    verbose: bool = True,
) -> dict:
    """Walk-forward evaluation. Returns per-model metrics + benchmark."""
    from .models import predict_up_proba

    dates = X.index
    splits = walk_forward_splits(dates, warmup_years)
    if verbose:
        print(f"walk-forward: {len(splits)} quarterly folds, "
              f"test window {splits[0][1][0].date()} -> {splits[-1][1][-1].date()}")

    results: dict = {}
    for name, make_model in models.items():
        oos_signals: list[pd.Series] = []
        for train_idx, test_idx in splits:
            model = make_model()
            model.fit(X.loc[train_idx], y.loc[train_idx])
            p_up = predict_up_proba(model, X.loc[test_idx])
            oos_signals.append((p_up > threshold).astype(int).rename("signal"))
        signals = pd.concat(oos_signals).sort_index()
        net = strategy_returns(signals, next_ret.loc[signals.index], cost_bps)
        metrics = performance_metrics(net)
        metrics.update(signal_stats(signals, y.loc[signals.index]))
        results[name] = metrics
        if verbose:
            print(f"  {name:8s} hit={metrics['hit_rate']:.3f} "
                  f"prec_long={metrics['precision_on_longs']:.3f} "
                  f"sharpe={metrics['annualized_sharpe']:+.2f} "
                  f"ret={metrics['total_return']:+.1%} "
                  f"maxdd={metrics['max_drawdown']:.1%}")

    # Buy-and-hold benchmark over the exact same out-of-sample window.
    bh_idx = splits[0][1][0]
    bh_ret = next_ret.loc[bh_idx:]
    results["buy_and_hold"] = performance_metrics(bh_ret)
    if verbose:
        b = results["buy_and_hold"]
        print(f"  {'buy_hold':8s} "
              f"sharpe={b['annualized_sharpe']:+.2f} "
              f"ret={b['total_return']:+.1%} "
              f"maxdd={b['max_drawdown']:.1%}")
    return results
