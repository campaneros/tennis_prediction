import numpy as np
import pandas as pd

from .data_loader import (
    MATCH_COL,
    SERVER_COL,
    POINT_WINNER_COL,
    GAME_WINNER_COL,
    WINDOW,
)


def add_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary label 'p1_wins_match' to each point:
    1 if Player 1 wins the match, 0 otherwise.
    """
    df = df.copy()

    last_rows = df.groupby(MATCH_COL).tail(1)
    match_winner_map = last_rows.set_index(MATCH_COL)[GAME_WINNER_COL]

    df["match_winner"] = df[MATCH_COL].map(match_winner_map)
    df["p1_wins_match"] = (df["match_winner"] == 1).astype(int)

    return df


def add_rolling_serve_return_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Build Depken-style rolling serve/return features:

    For each side s ∈ {1,2} within each match:
      s_srv_win: points where s served and won
      s_rcv_win: points where s received and won

    Then at each point, from current server perspective,
      P_srv_win  ∝ (n_srv_win + 1)
      P_srv_lose ∝ (n_opp_rcv_win + 1)
    and normalise so P_srv_win + P_srv_lose = 1.
    """
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
            df[base_name_template.format(side=2)],
        )

    n_srv_win = pick_col("s{side}_srv_win_roll")

    n_opp_rcv_win = np.where(
        server_side == 1,
        df["s2_rcv_win_roll"],
        df["s1_rcv_win_roll"],
    )

    P1_unnorm = (n_srv_win + 1.0) / float(window)
    P2_unnorm = (n_opp_rcv_win + 1.0) / float(window)
    norm = P1_unnorm + P2_unnorm

    df["P_srv_win"] = P1_unnorm / norm
    df["P_srv_lose"] = P2_unnorm / norm

    return df


def build_dataset(df: pd.DataFrame):
    """
    Build feature matrix X and target y for training.

    Features:
      - P_srv_win
      - P_srv_lose
      - SERVER_COL (PointServer)

    Target:
      - p1_wins_match

    Returns: X, y, mask
    """
    df = df.copy()
    feature_cols = ["P_srv_win", "P_srv_lose", SERVER_COL]

    X_all = df[feature_cols].values
    y_all = df["p1_wins_match"].values

    mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[mask]
    y = y_all[mask]

    return X, y, mask
