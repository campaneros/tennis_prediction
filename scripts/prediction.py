import os
import numpy as np

from .data_loader import load_points_multiple, MATCH_COL, WINDOW
from .features import add_match_labels, add_rolling_serve_return_features, build_dataset
from .model import load_model
from .plotting import plot_match_probabilities


def advance_game_state_simple(row_features, server_wins_point: bool, eff_window: float = WINDOW + 2.0):
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


def run_prediction(file_paths, model_path: str, match_id: str, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)

    df = load_points_multiple(file_paths)
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, window=WINDOW)

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
