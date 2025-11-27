import os
import numpy as np

from .data_loader import load_points_multiple, MATCH_COL, WINDOW
from .config import load_config
from .features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    add_additional_features,
    build_dataset,
)
from .model import load_model
from .plotting import plot_match_probabilities


def advance_game_state_simple(row_features, server_wins_point: bool, eff_window: float = 20.0):
    """
    Update features to simulate counterfactual scenario where server loses/wins the point.

    Current feature order (14 features):
    [0] P_srv_win_long
    [1] P_srv_lose_long
    [2] P_srv_win_short
    [3] P_srv_lose_short
    [4] PointServer
    [5] momentum
    [6] Momentum_Diff
    [7] Score_Diff
    [8] Game_Diff
    [9] SrvScr
    [10] RcvScr
    [11] SetNo
    [12] GameNo
    [13] PointNumber

    For the counterfactual, we update:
    - P_srv_win/lose (long and short) using Bayesian update
    - SrvScr/RcvScr based on outcome
    - Leave others unchanged as approximation
    """
    x = row_features.copy()
    
    # Extract long window probabilities
    P_win_long = float(x[0])
    P_lose_long = float(x[1])
    P_win_short = float(x[2])
    P_lose_short = float(x[3])
    server = int(x[4])
    
    # Normalize probabilities
    def normalize_probs(p_win, p_lose):
        if p_win < 0.0:
            p_win = 0.0
        if p_lose < 0.0:
            p_lose = 0.0
        S = p_win + p_lose
        if S <= 0.0:
            return 0.5, 0.5
        return p_win / S, p_lose / S
    
    P_win_long, P_lose_long = normalize_probs(P_win_long, P_lose_long)
    P_win_short, P_lose_short = normalize_probs(P_win_short, P_lose_short)
    
    # Bayesian update for long window
    alpha_long = P_win_long * eff_window
    beta_long = P_lose_long * eff_window
    
    if server_wins_point:
        alpha_long_new = alpha_long + 1.0
        beta_long_new = beta_long
    else:
        alpha_long_new = alpha_long
        beta_long_new = beta_long + 1.0
    
    total_long = alpha_long_new + beta_long_new
    x[0] = alpha_long_new / total_long
    x[1] = beta_long_new / total_long
    
    # Bayesian update for short window (more sensitive)
    short_window = 5.0
    alpha_short = P_win_short * short_window
    beta_short = P_lose_short * short_window
    
    if server_wins_point:
        alpha_short_new = alpha_short + 1.0
        beta_short_new = beta_short
    else:
        alpha_short_new = alpha_short
        beta_short_new = beta_short + 1.0
    
    total_short = alpha_short_new + beta_short_new
    x[2] = alpha_short_new / total_short
    x[3] = beta_short_new / total_short
    
    # Update SrvScr/RcvScr based on who served and won
    if server_wins_point:
        # If server=1, increment SrvScr; if server=2, RcvScr changes for P1
        if server == 1:
            x[9] += 1  # SrvScr
        else:
            x[10] += 0  # RcvScr stays same (server 2 won)
    else:
        # Server lost
        if server == 1:
            x[9] += 0  # SrvScr stays same
        else:
            x[10] += 1  # RcvScr (P1 received and won)
    
    return x


def compute_current_and_counterfactual_probs(X, model):
    """
    Given features X for all points and a trained model, compute:

      prob_p1          = P(Player 1 wins match | current state)
      prob_p2          = 1 - prob_p1
      prob_p1_lose_srv = P(Player 1 wins match | server loses this point)
      prob_p2_lose_srv = 1 - prob_p1_lose_srv
    """
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

def run_prediction(file_paths, model_path: str, match_id: str, plot_dir: str, config_path: str | None = None):
    """
    End-to-end prediction + plotting for a given set of files and one match_id.

    - Loads data
    - Rebuilds features (including momentum)
    - Loads trained model
    - Computes current and counterfactual probabilities
    - Produces a probability trajectory plot for the given match_id
    """
    os.makedirs(plot_dir, exist_ok=True)
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))

    df = load_points_multiple(file_paths)
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_leverage_and_momentum(df, alpha=alpha)
    df = add_additional_features(df)


    X, y, mask = build_dataset(df)
    df_valid = df[mask].copy()

    model = load_model(model_path)

    prob_p1, prob_p2, prob_p1_lose, prob_p2_lose = compute_current_and_counterfactual_probs(X, model)

    df_valid["prob_p1"] = prob_p1
    df_valid["prob_p2"] = prob_p2
    df_valid["prob_p1_lose_srv"] = prob_p1_lose
    df_valid["prob_p2_lose_srv"] = prob_p2_lose

    match_id_str = str(match_id)
    df_valid[MATCH_COL] = df_valid[MATCH_COL].astype(str)

    plot_match_probabilities(df_valid, match_id_str, plot_dir)
