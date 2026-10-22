# neural-market-ml

ML market-prediction research pipeline: predict next-day SPY direction with
small neural nets and classic ML, evaluated with a strict walk-forward
backtest. **Research and paper-trading only. Not financial advice.**

## What it is

A clean, honest pipeline for the question "can ML predict daily market
direction from price history?":

- **Data**: free daily SPY bars (Yahoo Finance, no API key), 2015 to
  present, cached to `data/spy_daily.parquet`. A second, point-in-time
  source reads Scott's own data stack (Project-Portfolio-Engine
  lakehouse and market-terminal releases) through
  `neural_market_ml/pit_data.py`; see "Point-in-time run" below and
  `DATA_ACCESS.md` for what that path can reach today.
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
pytest tests/              # 26 tests
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

## Point-in-time run (own stack, AAPL)

The PIT loader (`neural_market_ml/pit_data.py`) reads daily bars from
Scott's own point-in-time stack behind the same contract as the Yahoo
loader: every observation carries an `available_at` timestamp, a read
at cutoff `as_of` sees only revisions published by then, and fixture
tests prove a later correction cannot leak into an earlier read.

One real run has been executed on that path. Ticker is AAPL because
**the stack contains no SPY bars today** (see `DATA_ACCESS.md`; that,
not the loader, is the blocker for a like-for-like SPY rerun).
Provenance: Cloudflare R2 bucket `financial-system-datasets`, object
`gold/market-terminal/releases/19c7c32e43fb83f3e4e4efeb1e9b5c3eec484b10581d6e0b3b3e9d48189275ee/securities/AAPL/price_history.parquet`,
downloaded 2026-10-21. Input: 968 bars, 2015-01-02 to 2018-11-02
(the release ends in 2018). Walk-forward, 3 quarterly folds, test
window 2018-04-02 to 2018-11-01 (151 trading days), 3-year warmup,
5 bps one-way costs, long/flat only:

| model    | hit rate | precision on longs | Sharpe | total return | max drawdown |
|----------|----------|--------------------|--------|--------------|--------------|
| logreg   | 0.437    | 0.583              | +0.95  | +4.2%        | -3.6%        |
| mlp      | 0.490    | 0.593              | +2.05  | +24.5%       | -6.6%        |
| hgb      | 0.437    | 0.519              | +1.72  | +19.2%       | -6.7%        |
| buy-hold | -        | -                  | +1.60  | +24.5%       | -10.6%       |

Cross-checks: the 968-bar count and the first bar (2015-01-02 close
109.33000183105469, volume 53,204,600) match PPE's published values
for this data exactly, and loader closes are bit-identical to the raw
release file. Results saved in `results/backtest_pit_aapl.json`.

**Read this honestly.** 151 days on one strong ticker in one rising
window is a path verification, not evidence of edge. The MLP matching
buy-and-hold here does not contradict the SPY result above; the
samples do not overlap in ticker, period, or size. Hit rates for
logreg and hgb are still below a coin flip. The point of this run is
that the pipeline now executes end to end on verified, point-in-time
data from Scott's own stack.

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
- The point-in-time path cannot yet rerun the headline SPY experiment:
  no SPY bars exist in Scott's reachable stack sources, and the
  reachable price history ends 2018-11-02. `DATA_ACCESS.md` names the
  three concrete ways to unblock it.
- This is NOT financial advice. Paper trading only.

## Layout

```
neural_market_ml/
  data.py       Yahoo Finance download + parquet cache (graceful failure)
  pit_data.py   point-in-time loader for PPE lakehouse / releases / DuckDB
  features.py   no-lookahead feature engineering
  models.py     logreg / MLP / gradient boosting (+ seeds)
  backtest.py   walk-forward engine, cost accounting, metrics
train.py        fit final models -> models/
run_backtest.py walk-forward -> results/backtest.json (or PIT source)
DATA_ACCESS.md  what the PIT path can reach today, and the SPY blocker
tests/          26 pytest tests (no-lookahead, fold hygiene, costs,
                determinism, PIT revision and truncation proofs)
```

## License

MIT.
