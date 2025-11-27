import numpy as np
import pandas as pd

from .data_loader import (
    MATCH_COL,
    SERVER_COL,
    POINT_WINNER_COL,
    GAME_WINNER_COL,
)

def _compute_momentum_series(leverage: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exact momentum:

      M_t = [L_t + (1-alpha)L_{t-1} + ... + (1-alpha)^{t-1} L_1] /
            [1   + (1-alpha)       + ... + (1-alpha)^{t-1}     ]

    via recurrence:
      N_t = L_t + r N_{t-1}
      D_t = 1   + r D_{t-1}
      M_t = N_t / D_t

    with r = 1 - alpha.
    Valid for 0 < alpha < 2 (so |r| < 1), allowing alpha > 1.
    """
    r = 1.0 - alpha
    n = len(leverage)
    N = np.zeros(n, dtype=float)
    D = np.zeros(n, dtype=float)
    M = np.zeros(n, dtype=float)

    for i, L in enumerate(leverage):
        if i == 0:
            N[i] = L
            D[i] = 1.0
        else:
            N[i] = L + r * N[i - 1]
            D[i] = 1.0 + r * D[i - 1]

        M[i] = N[i] / D[i] if D[i] != 0.0 else 0.0

    return M

def add_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary label 'p1_wins_match' to each point:
    1 if Player 1 wins the match, 0 otherwise.

    We use the winner of the last game of each match as match winner.
    """
    df = df.copy()

    last_rows = df.groupby(MATCH_COL).tail(1)
    match_winner_map = last_rows.set_index(MATCH_COL)[GAME_WINNER_COL]

    df["match_winner"] = df[MATCH_COL].map(match_winner_map)
    df["p1_wins_match"] = (df["match_winner"] == 1).astype(int)

    return df


def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional features:
      - Momentum_Diff: P1Momentum - P2Momentum (if available in data)
      - Score_Diff: P1Score - P2Score (if available)
      - Game_Diff: P1GamesWon - P2GamesWon (if available)
      - SrvScr: cumulative points won when p1 served in current game
      - RcvScr: cumulative points won when p1 received in current game
    """
    df = df.copy()

    # Helper function to convert tennis score strings to numeric values
    def score_to_numeric(score):
        """Convert tennis score (0, 15, 30, 40, AD) to numeric value."""
        if pd.isna(score):
            return 0
        score_str = str(score).strip().upper()
        score_map = {
            '0': 0,
            '15': 1,
            '30': 2,
            '40': 3,
            'AD': 4,
            'A': 4,  # Sometimes advantage is abbreviated as 'A'
        }
        return score_map.get(score_str, 0)

    # Momentum difference (if columns exist)
    if 'P1Momentum' in df.columns and 'P2Momentum' in df.columns:
        df['Momentum_Diff'] = pd.to_numeric(df['P1Momentum'], errors='coerce').fillna(0) - pd.to_numeric(df['P2Momentum'], errors='coerce').fillna(0)
    else:
        df['Momentum_Diff'] = 0.0

    # Score difference (convert tennis scores to numeric first)
    if 'P1Score' in df.columns and 'P2Score' in df.columns:
        p1_score_numeric = df['P1Score'].apply(score_to_numeric)
        p2_score_numeric = df['P2Score'].apply(score_to_numeric)
        df['Score_Diff'] = p1_score_numeric - p2_score_numeric
    else:
        df['Score_Diff'] = 0.0

    # Game won difference
    if 'P1GamesWon' in df.columns and 'P2GamesWon' in df.columns:
        df['Game_Diff'] = pd.to_numeric(df['P1GamesWon'], errors='coerce').fillna(0) - pd.to_numeric(df['P2GamesWon'], errors='coerce').fillna(0)
    else:
        df['Game_Diff'] = 0.0

    # Served Score and Received Score (cumulative within each game)
    df['p1_srv_win_game'] = (
        (df[SERVER_COL] == 1) & (df[POINT_WINNER_COL] == 1)
    ).astype(int)
    df['p1_rcv_win_game'] = (
        (df[SERVER_COL] != 1) & (df[POINT_WINNER_COL] == 1)
    ).astype(int)

    df['SrvScr'] = (
        df.groupby([MATCH_COL, 'GameNo'])['p1_srv_win_game']
          .cumsum()
    )
    df['RcvScr'] = (
        df.groupby([MATCH_COL, 'GameNo'])['p1_rcv_win_game']
          .cumsum()
    )

    return df


def add_rolling_serve_return_features(
    df: pd.DataFrame,
    long_window: int,
    short_window: int,
    ) -> pd.DataFrame:
    """
    Build Depken-style rolling serve/return features at two time scales:

      - long_window  (e.g. 20 points)  → 'long-term features'
      - short_window (e.g. 5 points)   → 'real-time features'

    For each side s ∈ {1,2} within each match:
      s_srv_win: points where s served and won
      s_rcv_win: points where s received and won

    For each window scale we get, at each point, from the current
    server's perspective:
      P_srv_win_<tag>, P_srv_lose_<tag>

    For compatibility, we also expose:
      P_srv_win  = P_srv_win_long
      P_srv_lose = P_srv_lose_long
    """
    df = df.copy()

    df[SERVER_COL] = df[SERVER_COL].astype(int)
    df[POINT_WINNER_COL] = df[POINT_WINNER_COL].astype(int)

    # Base indicator columns: per side, did they serve+win or receive+win?
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

        # Rolling sums for both long and short windows
        for window, tag in ((long_window, "long"), (short_window, "short")):
            df[f"{srv_win_col}_roll_{tag}"] = (
                df.groupby(MATCH_COL)[srv_win_col]
                  .apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
                  .reset_index(level=0, drop=True)
            )
            df[f"{rcv_win_col}_roll_{tag}"] = (
                df.groupby(MATCH_COL)[rcv_win_col]
                  .apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
                  .reset_index(level=0, drop=True)
            )

    server_side = df[SERVER_COL]
    win_map = {"long": long_window, "short": short_window}

    def pick_col(template, tag):
        """
        template: e.g. 's{side}_srv_win_roll_{tag}'
        tag: 'long' or 'short'
        """
        return np.where(
            server_side == 1,
            df[template.format(side=1, tag=tag)],
            df[template.format(side=2, tag=tag)],
        )

    # Build probabilities at both scales
    for tag in ("long", "short"):
        window = win_map[tag]

        n_srv_win = pick_col("s{side}_srv_win_roll_{tag}", tag)

        n_opp_rcv_win = np.where(
            server_side == 1,
            df[f"s2_rcv_win_roll_{tag}"],
            df[f"s1_rcv_win_roll_{tag}"],
        )

        P1_unnorm = (n_srv_win + 1.0) / float(window)
        P2_unnorm = (n_opp_rcv_win + 1.0) / float(window)
        norm = P1_unnorm + P2_unnorm

        df[f"P_srv_win_{tag}"] = P1_unnorm / norm
        df[f"P_srv_lose_{tag}"] = P2_unnorm / norm

    # For backward compatibility and for leverage/momentum we take
    # the long-term probabilities as "current" P_srv_win/P_srv_lose.
    df["P_srv_win"] = df["P_srv_win_long"]
    df["P_srv_lose"] = df["P_srv_lose_long"]

    return df


def add_leverage_and_momentum(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Compute leverage and exact momentum with the given alpha.
    """
    df = df.copy()

    server_wins = (df[POINT_WINNER_COL] == df[SERVER_COL])

    leverage_raw = df["P_srv_win_long"] - df["P_srv_lose_long"]
    leverage_raw = leverage_raw.clip(lower=0.0)

    df["leverage"] = np.where(server_wins, leverage_raw, 0.0)


    df["momentum"] = (
        df.groupby(MATCH_COL)["leverage"]
          .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    )
    #df["momentum"] = (
    #    df.groupby(MATCH_COL)["leverage"]
    #      .transform(lambda x: _compute_momentum_series(x.to_numpy(), alpha))
    #)


    return df


def build_dataset(df: pd.DataFrame):
    """
    Build feature matrix X and target y for training.

    Features:
      - P_srv_win_long
      - P_srv_lose_long
      - P_srv_win_short   (real-time window)
      - P_srv_lose_short  (real-time window)
      - PointServer (Srv)
      - momentum (EWMA of leverage)
      - Momentum_Diff: P1Momentum - P2Momentum
      - Score_Diff: P1Score - P2Score
      - Game_Diff: P1GamesWon - P2GamesWon
      - SrvScr: cumulative points won when p1 served in game
      - RcvScr: cumulative points won when p1 received in game
      - SetNo (St)
      - GameNo (Gm)
      - PointNumber (Pt)

    Target:
      - p1_wins_match
    """
    df = df.copy()
    feature_cols = [
        "P_srv_win_long",
        "P_srv_lose_long",
        "P_srv_win_short",
        "P_srv_lose_short",
        SERVER_COL,
        "momentum",
        "Momentum_Diff",
        "Score_Diff",
        "Game_Diff",
        "SrvScr",
        "RcvScr",
        "SetNo",
        "GameNo",
        "PointNumber",
    ]

    X_all = df[feature_cols].values
    y_all = df["p1_wins_match"].values

    mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[mask]
    y = y_all[mask]

    return X, y, mask
