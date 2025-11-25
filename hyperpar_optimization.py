#!/usr/bin/env python3
"""
Hyperparameter optimisation for the XGBoost match-winning model.

- Uses the same feature construction as tennis_prediction.py
- Runs RandomizedSearchCV over a small hyperparameter space
- Saves the best model to xgb_match_model_tuned.json
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Reuse the same config
POINTS_CSV = "2021-wimbledon-points.csv"
TUNED_MODEL_PATH = "xgb_match_model_tuned.json"

MATCH_COL = "match_id"
SET_COL   = "SetNo"
GAME_COL  = "GameNo"
POINT_COL = "PointNumber"

SERVER_COL        = "PointServer"
POINT_WINNER_COL  = "PointWinner"
GAME_WINNER_COL   = "GameWinner"

WINDOW = 20


# --- Functions copied from tennis_prediction.py (minimal subset) ---------
def load_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values([MATCH_COL, SET_COL, GAME_COL, POINT_COL]).reset_index(drop=True)
    return df


def add_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    last_rows = df.groupby(MATCH_COL).tail(1)
    match_winner_map = last_rows.set_index(MATCH_COL)[GAME_WINNER_COL]
    df["match_winner"] = df[MATCH_COL].map(match_winner_map)
    df["p1_wins_match"] = (df["match_winner"] == 1).astype(int)
    return df


def add_rolling_serve_return_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    df = df.copy()
    df[SERVER_COL] = df[SERVER_COL].astype(int)
    df[POINT_WINNER_COL] = df[POINT_WINNER_COL].astype(int)

    for side in (1, 2):
        srv_win_col = f"s{side}_srv_win"
        rcv_win_col = f"s{side}_rcv_win"

        df[srv_win_col] = (
            (df[SERVER_COL] == side) &
            (df[POINT_WINNER_COL] == side)
        ).astype(int)

        df[rcv_win_col] = (
            (df[SERVER_COL] != side) &
            (df[POINT_WINNER_COL] == side)
        ).astype(int)

        df[f"{srv_win_col}_roll"] = (
            df.groupby(MATCH_COL)[srv_win_col]
              .apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
              .reset_index(level=0, drop=True)
        )
        df[f"{rcv_win_col}_roll"] = (
            df.groupby(MATCH_COL)[rcv_win_col]
              .apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
              .reset_index(level=0, drop=True)
        )

    server_side = df[SERVER_COL]

    def pick_col(base_name_template):
        return np.where(
            server_side == 1,
            df[base_name_template.format(side=1)],
            df[base_name_template.format(side=2)]
        )

    n_srv_win = pick_col("s{side}_srv_win_roll")
    n_opp_rcv_win = np.where(
        server_side == 1,
        df["s2_rcv_win_roll"],
        df["s1_rcv_win_roll"]
    )

    P1_unnorm = (n_srv_win + 1.0) / float(window)
    P2_unnorm = (n_opp_rcv_win + 1.0) / float(window)
    norm = P1_unnorm + P2_unnorm

    df["P_srv_win"] = P1_unnorm / norm
    df["P_srv_lose"] = P2_unnorm / norm

    return df


def build_dataset(df: pd.DataFrame):
    feature_cols = ["P_srv_win", "P_srv_lose", SERVER_COL]
    X_all = df[feature_cols].values
    y_all = df["p1_wins_match"].values
    mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[mask]
    y = y_all[mask]
    return X, y


# --- Hyperparameter search -----------------------------------------------
def main():
    df = load_points(POINTS_CSV)
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, window=WINDOW)
    X, y = build_dataset(df)

    print("Hyperopt dataset shape:", X.shape, " positives:", y.sum())

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    param_distributions = {
        "n_estimators":    [200, 400, 600, 800],
        "max_depth":       [3, 4, 5, 6],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "min_child_weight":[1, 3, 5, 7],
        "gamma":           [0.0, 0.1, 0.3, 0.5],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,                # number of random configurations
        scoring="roc_auc",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X, y)

    print("Best parameters:")
    print(search.best_params_)
    print(f"Best CV ROC AUC: {search.best_score_:.3f}")

    best_model = search.best_estimator_
    best_model.save_model(TUNED_MODEL_PATH)
    print(f"Tuned model saved to: {TUNED_MODEL_PATH}")


if __name__ == "__main__":
    main()
