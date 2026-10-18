"""Feature engineering with strict no-lookahead.

Every feature at row ``t`` is computed only from bars with dates <= ``t``.
All indicators use causal rolling / ewm windows (never centered), so the
feature vector known at the close of day ``t`` is exactly what a live
strategy would have had. No ``shift(1)`` is needed on top: causality is
built into the computation, and a truncation test in ``tests/``
proves it (features computed on data truncated at T match the full-frame
features at every row <= T).

Target convention (built by :func:`make_target`): ``y[t]`` is the sign of
the next-day return, i.e. ``sign(close[t+1] / close[t] - 1)``, paired with
the feature row at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_21d",
    "rsi_14",
    "macd_line", "macd_signal", "macd_hist",
    "bb_pctb", "bb_bandwidth",
    "atr_pct",
    "vol_21d",
    "volume_zscore",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, causal by construction (ewm over past deltas only)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr / close


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix. Returns rows with any NaN dropped.

    The returned frame's index is a subset of ``df``'s index; row ``t``
    depends only on bars up to and including ``t``.
    """
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # Trailing N-day returns ending at t (causal: pct_change(N) uses close[t-N..t]).
    for lag in (1, 2, 3, 5, 10, 21):
        feats[f"ret_{lag}d"] = close.pct_change(lag)

    feats["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    feats["macd_line"] = ema12 - ema26
    feats["macd_signal"] = feats["macd_line"].ewm(span=9, min_periods=9, adjust=False).mean()
    feats["macd_hist"] = feats["macd_line"] - feats["macd_signal"]

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    feats["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)
    feats["bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)

    feats["atr_pct"] = _atr_pct(high, low, close, 14)

    daily_ret = close.pct_change()
    feats["vol_21d"] = daily_ret.rolling(21, min_periods=21).std()

    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std()
    feats["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)

    dow = pd.Series(df.index.dayofweek, index=df.index)
    for d in range(5):
        feats[f"dow_{d}"] = (dow == d).astype(float)

    feats = feats[FEATURE_COLUMNS]
    return feats.dropna()


def make_target(df: pd.DataFrame, feature_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Target and next-day return aligned to the feature index.

    ``y[t]`` = 1 if close[t+1] > close[t] else 0. ``next_ret[t]`` is the
    simple return close[t+1]/close[t] - 1 earned by holding through day t+1.
    The final feature row has no next day and is dropped.
    """
    close = df["Close"].astype(float)
    next_ret = close.pct_change().shift(-1)
    y = (next_ret > 0).astype(int)
    aligned = feature_index.intersection(next_ret.dropna().index)
    return y.loc[aligned], next_ret.loc[aligned]
