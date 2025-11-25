#!/usr/bin/env python3
"""
XGBoost model for match-winning probability in tennis
using Jeff Sackmann-style point-by-point data.

- Label: for each point, does Player 1 eventually win the MATCH?
- Features: Depken-style rolling serve/return stats (window=20)
  encoded as P_srv_win and P_srv_lose, plus server indicator.
- Output for each point:
    prob_p1          = P(Player 1 wins match | current state)
    prob_p2          = 1 - prob_p1
    prob_p1_lose_srv = P(Player 1 wins match | server LOSES this point)
    prob_p2_lose_srv = 1 - prob_p1_lose_srv
- Plot: all four curves vs point index for one match.
- The trained model is saved to disk for reuse on other matches.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 0. CONFIG
# -------------------------------------------------------------------------
POINTS_CSV = "../tennis_slam_pointbypoint/2021-wimbledon-points.csv"  # path to your file
MODEL_PATH = "xgb_match_model.json"      # model file to save / load

MATCH_COL = "match_id"
SET_COL   = "SetNo"
GAME_COL  = "GameNo"
POINT_COL = "PointNumber"

SERVER_COL        = "PointServer"   # 1 or 2
POINT_WINNER_COL  = "PointWinner"   # 1 or 2
GAME_WINNER_COL   = "GameWinner"    # 1 or 2

WINDOW = 20  # rolling window size


# -------------------------------------------------------------------------
# 1. Load and basic sorting
# -------------------------------------------------------------------------
def load_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Columns in file:", df.columns.tolist())
    df = df.sort_values([MATCH_COL, SET_COL, GAME_COL, POINT_COL]).reset_index(drop=True)
    return df


# -------------------------------------------------------------------------
# 2. Create match-wise label: does Player 1 win the match?
# -------------------------------------------------------------------------
def add_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    last_rows = df.groupby(MATCH_COL).tail(1)
    match_winner_map = last_rows.set_index(MATCH_COL)[GAME_WINNER_COL]
    df["match_winner"] = df[MATCH_COL].map(match_winner_map)
    df["p1_wins_match"] = (df["match_winner"] == 1).astype(int)
    return df


# -------------------------------------------------------------------------
# 3. Rolling 20-point serve/return features (Depken style)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# 4. Build feature matrix X and target y
# -------------------------------------------------------------------------
def build_dataset(df: pd.DataFrame):
    feature_cols = ["P_srv_win", "P_srv_lose", SERVER_COL]
    X_all = df[feature_cols].values
    y_all = df["p1_wins_match"].values

    mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[mask]
    y = y_all[mask]

    return X, y, mask


# -------------------------------------------------------------------------
# 5. Train XGBoost model + save it
# -------------------------------------------------------------------------
def train_xgb_classifier(X, y, model_path: str):
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

    # Save model to disk
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    return model


def load_trained_model(model_path: str) -> XGBClassifier:
    """
    Load a previously saved XGBoost model.
    Example usage:
        model = load_trained_model("xgb_match_model.json")
    """
    model = XGBClassifier()
    model.load_model(model_path)
    return model


# -------------------------------------------------------------------------
# 6. Counterfactual update: server loses the point
# -------------------------------------------------------------------------
def advance_game_state_simple(row_features: np.ndarray,
                              server_wins_point: bool,
                              eff_window: float = WINDOW + 2.0) -> np.ndarray:
    x = row_features.copy()
    P_win = float(x[0])
    P_lose = float(x[1])

    if P_win < 0.0:
        P_win = 0.0
    if P_lose < 0.0:
        P_lose = 0.0
    S = P_win + P_lose
    if S <= 0.0:
        P_win = 0.5
        P_lose = 0.5
        S = 1.0

    P_win /= S
    P_lose /= S

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
    return x


def compute_current_and_counterfactual_probs(X, model):
    n = X.shape[0]
    prob_p1 = np.zeros(n)
    prob_p2 = np.zeros(n)
    prob_p1_lose_srv = np.zeros(n)
    prob_p2_lose_srv = np.zeros(n)

    for i in range(n):
        x = X[i]

        p1_now = float(model.predict_proba(x.reshape(1, -1))[:, 1])
        prob_p1[i] = p1_now
        prob_p2[i] = 1.0 - p1_now

        x_lose = advance_game_state_simple(x, server_wins_point=False)
        p1_lose = float(model.predict_proba(x_lose.reshape(1, -1))[:, 1])
        prob_p1_lose_srv[i] = p1_lose
        prob_p2_lose_srv[i] = 1.0 - p1_lose

    return prob_p1, prob_p2, prob_p1_lose_srv, prob_p2_lose_srv


# -------------------------------------------------------------------------
# 7. Plot match-wise probabilities with counterfactual (server loses)
# -------------------------------------------------------------------------
def plot_match_probabilities(df_valid: pd.DataFrame, match_id_to_plot):
    dfm = df_valid[df_valid[MATCH_COL] == match_id_to_plot].reset_index(drop=True)

    if dfm.empty:
        print(f"No rows found for match_id = {match_id_to_plot}")
        return

    max_dev_now = np.max(np.abs(dfm["prob_p1"] + dfm["prob_p2"] - 1.0))
    max_dev_lose = np.max(np.abs(dfm["prob_p1_lose_srv"] + dfm["prob_p2_lose_srv"] - 1.0))
    print(f"Max deviation from P1+P2=1 (current): {max_dev_now:.2e}")
    print(f"Max deviation from P1+P2=1 (server loses): {max_dev_lose:.2e}")

    x = np.arange(len(dfm))

    plt.figure(figsize=(11, 6))

    plt.plot(x, dfm["prob_p1"], label="P1 wins match (current)", linewidth=2)
    plt.plot(x, dfm["prob_p2"], label="P2 wins match (current)", linewidth=2)

    plt.plot(
        x, dfm["prob_p1_lose_srv"],
        "--", label="P1 wins match | server loses point", linewidth=1.5
    )
    plt.plot(
        x, dfm["prob_p2_lose_srv"],
        "--", label="P2 wins match | server loses point", linewidth=1.5
    )

    plt.xlabel("Point index in match")
    plt.ylabel("Match win probability")
    plt.ylim(0.0, 1.0)
    plt.title(f"Match win probabilities and counterfactual (server loses point)\nmatch_id={match_id_to_plot}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# 8. Main
# -------------------------------------------------------------------------
def main():
    df = load_points(POINTS_CSV)
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, window=WINDOW)

    X, y, mask = build_dataset(df)
    print("Dataset shape:", X.shape, " Target positives (P1 wins):", y.sum())

    model = train_xgb_classifier(X, y, MODEL_PATH)

    prob_p1, prob_p2, prob_p1_lose_srv, prob_p2_lose_srv = compute_current_and_counterfactual_probs(X, model)

    df_valid = df[mask].copy()
    df_valid["prob_p1"] = prob_p1
    df_valid["prob_p2"] = prob_p2
    df_valid["prob_p1_lose_srv"] = prob_p1_lose_srv
    df_valid["prob_p2_lose_srv"] = prob_p2_lose_srv

    example_match_id = df_valid[MATCH_COL].iloc[0]
    print("Plotting probabilities for match_id =", example_match_id)
    plot_match_probabilities(df_valid, example_match_id)


if __name__ == "__main__":
    main()
