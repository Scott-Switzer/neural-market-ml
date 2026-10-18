"""Small CPU-friendly classifiers for next-day direction.

Everything here trains in minutes on a CPU. Each model is seeded via
``random_state`` so runs are reproducible (see determinism test).

PyTorch LSTM was intentionally left out: the CPU wheel would not
download in this environment, and the brief says not to burn time on
the install. The sklearn MLPClassifier below is the neural-net
representative; an LSTM module can slot into ``get_models()`` later
behind a ``torch`` availability check.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["logreg", "mlp", "hgb"]


def get_models(random_state: int = 42) -> dict:
    """Return {name: unfitted sklearn estimator}."""
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=500,
                random_state=random_state,
            ),
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
    }


def predict_up_proba(model, X) -> "pd.Series":
    """P(up) for each row of X, returned as a Series aligned to X.index."""
    import pandas as pd

    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="p_up")
