#!/usr/bin/env python3
"""
XGBoost model for within-game prediction in tennis using Jeff Sackmann's
Grand Slam point-by-point data and a 20-point rolling serve/return window.

- Label: for each point, does the current server win this game?
- Features: Depken-style rolling serve/return stats (window=20) encoded as
  P_srv_win and P_srv_lose, plus server indicator.
- Counterfactual leverage: use a simple Bayesian update of P_srv_win/P_srv_lose
  for hypothetical win/lose of the current point, then evaluate the model
  at the two post-point states.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# -------------------------------------------------------------------------
# 0. CONFIG: adapt these column names to your CSV
# -------------------------------------------------------------------------
POINTS_CSV = "tennis_slam_pointbypoint/2021-wimbledon-points.csv"  # path to your file

MATCH_COL = "match_id"      # OK
SET_COL   = "SetNo"         # <-- FIXED
GAME_COL  = "GameNo"        # <-- FIXED
POINT_COL = "PointNumber"   # <-- FIXED

SERVER_COL        = "PointServer"   # <-- FIXED
POINT_WINNER_COL  = "PointWinner"   # <-- FIXED


WINDOW = 20  # rolling window size


# -------------------------------------------------------------------------
# 1. Load and basic sorting
# -------------------------------------------------------------------------
def load_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Columns in file:", df.columns.tolist())
    # sort in chronological order within match
    df = df.sort_values([MATCH_COL, SET_COL, GAME_COL, POINT_COL]).reset_index(drop=True)
    return df


# -------------------------------------------------------------------------
# 2. Create game-wise label: does the server win this game?
# -------------------------------------------------------------------------
def add_game_labels(df: pd.DataFrame) -> pd.DataFrame:
    # define game id
    df["game_id"] = (
        df[MATCH_COL].astype(str) + "_" +
        df[SET_COL].astype(str)   + "_" +
        df[GAME_COL].astype(str)
    )

    # server is constant within a game; winner is last point's victor
    game_server = df.groupby("game_id")[SERVER_COL].transform("first")
    game_winner = df.groupby("game_id")[POINT_WINNER_COL].transform("last")

    df["server_wins_game"] = (game_server == game_winner).astype(int)
    return df


# -------------------------------------------------------------------------
# 3. Rolling 20-point serve/return features (Depken style)
# -------------------------------------------------------------------------
def add_rolling_serve_return_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    For each side s in {1,2}, precompute rolling counts over the last `window`
    points within a match:
      - s_srv_win: points where s served and won
      - s_rcv_win: points where s received and won
    Then build, at each row, the Depken-style P_srv_win and P_srv_lose
    from the perspective of the current server.
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

        # rolling sums over previous points (exclude current with shift)
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

    # From server perspective: pick the correct rolling counts
    server_side = df[SERVER_COL]
    opp_side = 3 - server_side  # not used directly, but kept for clarity

    # for server's own serve-win rolling count
    def pick_col(base_name_template):
        return np.where(
            server_side == 1,
            df[base_name_template.format(side=1)],
            df[base_name_template.format(side=2)]
        )

    n_srv_win = pick_col("s{side}_srv_win_roll")

    # opponent's receive-win rolling count
    n_opp_rcv_win = np.where(
        server_side == 1,
        df["s2_rcv_win_roll"],
        df["s1_rcv_win_roll"]
    )

    # Depken-style unnormalized probabilities
    P1_unnorm = (n_srv_win + 1.0) / float(window)      # server wins point
    P2_unnorm = (n_opp_rcv_win + 1.0) / float(window)  # server loses point

    norm = P1_unnorm + P2_unnorm
    df["P_srv_win"] = P1_unnorm / norm
    df["P_srv_lose"] = P2_unnorm / norm

    return df


# -------------------------------------------------------------------------
# 4. Build feature matrix X and target y
# -------------------------------------------------------------------------
def build_dataset(df: pd.DataFrame):
    """
    Minimal feature set: just the serve/return window and an indicator
    for which side is serving (1/2). You can add more scoreboard/context
    features here later.
    """
    feature_cols = ["P_srv_win", "P_srv_lose", SERVER_COL]
    X = df[feature_cols].values
    y = df["server_wins_game"].values

    # Optional: remove NaNs from very early points (before window filled)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y


# -------------------------------------------------------------------------
# 5. Train XGBoost model
# -------------------------------------------------------------------------
def train_xgb_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = float("nan")

    print(f"Test accuracy: {acc:.3f}")
    print(f"Test ROC AUC:  {auc:.3f}")

    return model


# -------------------------------------------------------------------------
# 6. Counterfactual leverage: implemented
# -------------------------------------------------------------------------
def advance_game_state_simple(row_features: np.ndarray,
                              server_wins_point: bool,
                              eff_window: float = WINDOW + 2.0) -> np.ndarray:
    """
    Given the feature vector for a point:
      row_features = [P_srv_win, P_srv_lose, server_id]

    we build a *post-point* feature vector for the hypothetical outcome:
    server_wins_point ∈ {True, False}.

    We interpret P_srv_win / P_srv_lose as the mean of a Bernoulli with
    an effective sample size eff_window, and update the Beta(α,β) counts:

      α = P_srv_win  * eff_window
      β = P_srv_lose * eff_window

    If server wins the point: α' = α + 1, β' = β
    If server loses the point: α' = α,     β' = β + 1

    Then:
      P'_srv_win  = α' / (α' + β')
      P'_srv_lose = β' / (α' + β')

    The server id itself is kept fixed here. If you add scoreboard features
    and a real server-rotation model, you should update them here as well.
    """
    x = row_features.copy()
    P_win = float(x[0])
    P_lose = float(x[1])

    # guard against numerical problems
    if P_win < 0.0:
        P_win = 0.0
    if P_lose < 0.0:
        P_lose = 0.0
    S = P_win + P_lose
    if S <= 0.0:
        # degenerate: fall back to symmetric prior
        P_win = 0.5
        P_lose = 0.5
        S = 1.0

    # renormalise in case of small drift
    P_win /= S
    P_lose /= S

    # effective Beta counts
    alpha = P_win * eff_window
    beta = P_lose * eff_window

    if server_wins_point:
        alpha_new = alpha + 1.0
        beta_new = beta
    else:
        alpha_new = alpha
        beta_new = beta + 1.0

    total_new = alpha_new + beta_new
    P_win_new = alpha_new / total_new
    P_lose_new = beta_new / total_new

    x[0] = P_win_new
    x[1] = P_lose_new
    # x[2] (server id) unchanged in this simple version
    return x


def compute_leverage_for_row(model, row_features: np.ndarray):
    """
    row_features: numpy array of shape (n_features,) for one point.
                  Expected ordering: [P_srv_win, P_srv_lose, server]

    Returns:
      P_win, P_lose, leverage_raw

    where:
      P_win  = model's predicted probability that the *current* server
               eventually wins the game if he wins this point.
      P_lose = same, if he loses this point.
      leverage_raw = P_win - P_lose.
    """
    # Hypothetical post-point states
    x_win_post = advance_game_state_simple(row_features, server_wins_point=True)
    x_lose_post = advance_game_state_simple(row_features, server_wins_point=False)

    x_win_post = x_win_post.reshape(1, -1)
    x_lose_post = x_lose_post.reshape(1, -1)

    P_win = float(model.predict_proba(x_win_post)[:, 1])
    P_lose = float(model.predict_proba(x_lose_post)[:, 1])

    leverage_raw = P_win - P_lose
    return P_win, P_lose, leverage_raw


# -------------------------------------------------------------------------
# 7. Main
# -------------------------------------------------------------------------
def main():
    df = load_points(POINTS_CSV)
    df = add_game_labels(df)
    df = add_rolling_serve_return_features(df, window=WINDOW)

    X, y = build_dataset(df)
    print("Dataset shape:", X.shape, " Target positives:", y.sum())

    model = train_xgb_classifier(X, y)

    # Example: compute leverage for the first point in the dataset
    sample_row = X[0]
    P_win, P_lose, lev = compute_leverage_for_row(model, sample_row)
    print("Example point:")
    print("  P_win  (if server wins point):  ", P_win)
    print("  P_lose (if server loses point): ", P_lose)
    print("  leverage(raw):                  ", lev)


if __name__ == "__main__":
    main()
