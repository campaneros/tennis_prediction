import numpy as np
import pandas as pd

from scripts.features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    build_dataset,
)
from scripts.model import _default_model


def make_tiny_df():
    """
    Build a tiny synthetic dataset with two matches:
      - match 1: Player 1 wins
      - match 2: Player 2 wins
    This ensures y has both 0 and 1, so XGBoost's base_score is in (0,1).
    """
    rows = []

    # Match 1: P1 wins the match
    for i in range(6):
        rows.append({
            "match_id": 1,
            "SetNo": 1,
            "GameNo": 1,
            "PointNumber": i + 1,
            "PointServer": 1 if i % 2 == 0 else 2,
            "PointWinner": 1 if i < 4 else 2,  # arbitrary
            "GameWinner": 1,                    # P1 wins match
        })

    # Match 2: P2 wins the match
    for i in range(6):
        rows.append({
            "match_id": 2,
            "SetNo": 1,
            "GameNo": 1,
            "PointNumber": i + 1,
            "PointServer": 2 if i % 2 == 0 else 1,
            "PointWinner": 2 if i < 4 else 1,  # arbitrary
            "GameWinner": 2,                    # P2 wins match
        })

    return pd.DataFrame(rows)

def test_feature_pipeline_and_model():
    df = make_tiny_df()
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=20, short_window=5)
    df = add_leverage_and_momentum(df, alpha=0.3)

    X, y, mask = build_dataset(df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 6  # P_srv_win_long, P_srv_lose_long, P_srv_win_short, P_srv_lose_short, PointServer, momentum

    model = _default_model()
    # Just check that fit() runs without crashing on tiny data
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
