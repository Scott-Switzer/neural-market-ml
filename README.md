# neural-market-ml

ML market-prediction research pipeline: predict next-day SPY direction with
small neural nets and classic ML, evaluated with a strict walk-forward
backtest. **Research and paper-trading only. Not financial advice.**

## What it is

A clean, honest pipeline for the question "can ML predict daily market
direction from price history?":

- **Data**: free daily SPY bars (Yahoo Finance, no API key), 2015 to
  present, cached to `data/spy_daily.parquet`.
- **Features** (`neural_market_ml/features.py`): 20 features — trailing
  returns (1/2/3/5/10/21d), RSI(14), MACD, Bollinger %B and bandwidth,
  ATR(14)/close, 21d rolling volatility, volume z-score, day-of-week
  one-hots. All computed with strict no-lookahead: every feature at day
  `t` uses only bars with dates <= `t` (proven by truncation tests).
- **Models**: logistic regression (baseline), sklearn `MLPClassifier`
  (32,16) neural net, `HistGradientBoostingClassifier`. All CPU-friendly,
  all seeded for reproducibility.
- **Validation**: expanding-window walk-forward backtest, quarterly
  refits, never shuffle-split. 5 bps per one-way trade on every position
  change. Strategy goes long when P(up) > 0.55, flat otherwise.
  Benchmarked against buy-and-hold SPY over the same window.

## Quickstart

```bash
pip install -r requirements.txt
python train.py            # train final models on full data -> models/
python run_backtest.py     # walk-forward backtest -> results/backtest.json
pytest tests/              # 16 tests
```

## Real results (not cherry-picked)

Walk-forward, 35 quarterly folds, test window 2018-04-02 to 2026-10-15
(2,148 trading days), 3-year warmup, 5 bps one-way costs, long/flat only:

| model    | hit rate | precision on longs | Sharpe | total return | max drawdown |
|----------|----------|--------------------|--------|--------------|--------------|
| logreg   | 0.480    | 0.536              | -0.44  | -47.3%       | -56.6%       |
| mlp      | 0.504    | 0.550              | -0.30  | -35.7%       | -38.4%       |
| hgb      | 0.478    | 0.526              | +0.14  | +9.3%        | -28.1%       |
| buy-hold | —        | —                  | +0.77  | +196.4%      | -34.1%       |

**The honest takeaway: none of the models beat buy-and-hold.** Hit rates
hover near a coin flip (the base up-day rate is 0.543), and trading costs
eat what little signal exists. This is the expected outcome, not a bug:
daily direction from public price history is one of the most-mined
datasets on earth. The value here is the pipeline and the evaluation
discipline, not the P&L. These numbers were produced by `run_backtest.py`
on 2026-10-17 and are saved in `results/backtest.json`.

## Limitations

- Public price data is the most-mined dataset on earth. Do not expect
  this to make money, and do not trade real capital on it.
- Features are price/volume only: no fundamentals, no order flow, no
  alternative data.
- Benchmark uses price returns (dividends excluded), so buy-and-hold is
  slightly understated; it still wins by a mile.
- A PyTorch LSTM was planned but the CPU wheel would not download in the
  build environment; the sklearn MLP is the neural-net representative.
  An LSTM can slot into `get_models()` later behind a torch check.
- This is NOT financial advice. Paper trading only.

## Layout

```
neural_market_ml/
  data.py       Yahoo Finance download + parquet cache (graceful failure)
  features.py   no-lookahead feature engineering
  models.py     logreg / MLP / gradient boosting (+ seeds)
  backtest.py   walk-forward engine, cost accounting, metrics
train.py        fit final models -> models/
run_backtest.py walk-forward -> results/backtest.json
tests/          16 pytest tests (no-lookahead, fold hygiene, costs, determinism)
```

## License

MIT.
