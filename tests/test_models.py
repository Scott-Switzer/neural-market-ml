"""Model tests: valid probabilities and determinism under fixed seeds."""

import numpy as np
import pandas as pd

from neural_market_ml.models import MODEL_NAMES, get_models, predict_up_proba


def _toy_data():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(
        rng.normal(size=(200, 8)),
        index=pd.bdate_range("2022-01-01", periods=200),
        columns=[f"f{i}" for i in range(8)],
    )
    y = pd.Series((X["f0"] + rng.normal(0, 0.5, 200) > 0).astype(int), index=X.index)
    return X, y


def test_all_models_produce_valid_probabilities():
    X, y = _toy_data()
    for name, model in get_models(random_state=42).items():
        model.fit(X, y)
        p = predict_up_proba(model, X)
        assert p.index.equals(X.index)
        assert ((p >= 0) & (p <= 1)).all(), name
        assert p.notna().all(), name


def test_determinism_fixed_seed():
    """Same seed twice -> identical P(up) for every model."""
    X, y = _toy_data()
    for name in MODEL_NAMES:
        p1 = predict_up_proba(get_models(random_state=123)[name].fit(X, y), X)
        p2 = predict_up_proba(get_models(random_state=123)[name].fit(X, y), X)
        pd.testing.assert_series_equal(p1, p2)


def test_model_names_stable():
    assert set(get_models().keys()) == set(MODEL_NAMES)
