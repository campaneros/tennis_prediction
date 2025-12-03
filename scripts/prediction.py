import os
import numpy as np

from .data_loader import load_points_multiple, MATCH_COL, SERVER_COL, WINDOW
from .config import load_config
from .features import (
    MATCH_FEATURE_COLUMNS,
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    add_additional_features,
    build_dataset,
)
from .model import load_model
from .plotting import plot_match_probabilities


def simulate_score_after_point_loss(row, server_wins: bool):
    """
    Simulate what the actual tennis score would be if the server wins/loses the point.
    Returns a modified row with updated scores and games.
    
    This properly implements tennis scoring rules:
    - 0, 15, 30, 40, deuce, advantage, game
    - Games accumulate to win sets
    - Realistic game/set transitions
    """
    new_row = row.copy()
    
    # Get current state
    p1_score = str(row.get('P1Score', '0')).strip()
    p2_score = str(row.get('P2Score', '0')).strip()
    server = int(row.get('PointServer', 1))
    p1_games = int(row.get('P1GamesWon', 0))
    p2_games = int(row.get('P2GamesWon', 0))
    
    # Score progression map
    score_progression = {'0': '15', '15': '30', '30': '40', '40': 'WIN', 'AD': 'WIN'}
    
    # Determine who wins the point
    if server_wins:
        winner = server
    else:
        winner = 2 if server == 1 else 1
    
    # Update scores based on winner
    game_won = False
    
    # Handle deuce/advantage situations
    if p1_score == '40' and p2_score == '40':
        # Deuce
        if winner == 1:
            new_row['P1Score'] = 'AD'
            new_row['P2Score'] = '40'
        else:
            new_row['P1Score'] = '40'
            new_row['P2Score'] = 'AD'
    elif p1_score == 'AD':
        if winner == 1:
            game_won = True
            new_row['P1GamesWon'] = p1_games + 1
        else:
            new_row['P1Score'] = '40'
            new_row['P2Score'] = '40'
    elif p2_score == 'AD':
        if winner == 2:
            game_won = True
            new_row['P2GamesWon'] = p2_games + 1
        else:
            new_row['P1Score'] = '40'
            new_row['P2Score'] = '40'
    else:
        # Normal scoring
        if winner == 1:
            next_score = score_progression.get(p1_score, '0')
            if next_score == 'WIN':
                game_won = True
                new_row['P1GamesWon'] = p1_games + 1
            else:
                new_row['P1Score'] = next_score
        else:
            next_score = score_progression.get(p2_score, '0')
            if next_score == 'WIN':
                game_won = True
                new_row['P2GamesWon'] = p2_games + 1
            else:
                new_row['P2Score'] = next_score
    
    # If game was won, reset point scores
    if game_won:
        new_row['P1Score'] = '0'
        new_row['P2Score'] = '0'
        # Also set GameWinner
        new_row['GameWinner'] = winner
    
    # Set PointWinner
    new_row['PointWinner'] = winner
    
    return new_row


def recalculate_match_state_from_point_winners(df):
    """
    Recalculate scores and games based solely on PointWinner and PointServer.
    
    IMPORTANT: This function does NOT recalculate SetNo or GameNo - those come from the original dataset.
    It only recalculates: P1Score, P2Score, P1GamesWon, P2GamesWon, SetWinner, GameWinner
    
    This allows us to modify PointWinner and see how scores/games would have changed,
    while maintaining the original match structure (sets/games).
    """
    df = df.copy()
    
    # Initialize state tracking
    df['P1Score'] = '0'
    df['P2Score'] = '0'
    df['GameWinner'] = 0
    df['SetWinner'] = 0
    
    current_p1_score = '0'
    current_p2_score = '0'
    current_set = 1
    current_p1_games_in_set = 0
    current_p2_games_in_set = 0
    
    score_progression = {'0': '15', '15': '30', '30': '40', '40': 'WIN', 'AD': 'WIN'}
    
    for idx in df.index:
        server = int(df.at[idx, 'PointServer'])
        winner = int(df.at[idx, 'PointWinner'])
        set_no = int(df.at[idx, 'SetNo'])
        
        # Check if we're in a new set
        if set_no != current_set:
            # Reset games counter for new set
            current_set = set_no
            current_p1_games_in_set = 0
            current_p2_games_in_set = 0
            current_p1_score = '0'
            current_p2_score = '0'
        
        # Update score based on winner
        game_won = False
        game_winner = 0
        
        # Handle deuce/advantage
        if current_p1_score == '40' and current_p2_score == '40':
            if winner == 1:
                current_p1_score = 'AD'
                current_p2_score = '40'
            else:
                current_p1_score = '40'
                current_p2_score = 'AD'
        elif current_p1_score == 'AD':
            if winner == 1:
                game_won = True
                game_winner = 1
                current_p1_games_in_set += 1
            else:
                current_p1_score = '40'
                current_p2_score = '40'
        elif current_p2_score == 'AD':
            if winner == 2:
                game_won = True
                game_winner = 2
                current_p2_games_in_set += 1
            else:
                current_p1_score = '40'
                current_p2_score = '40'
        else:
            # Normal scoring
            if winner == 1:
                next_score = score_progression.get(current_p1_score, '0')
                if next_score == 'WIN':
                    game_won = True
                    game_winner = 1
                    current_p1_games_in_set += 1
                else:
                    current_p1_score = next_score
            else:
                next_score = score_progression.get(current_p2_score, '0')
                if next_score == 'WIN':
                    game_won = True
                    game_winner = 2
                    current_p2_games_in_set += 1
                else:
                    current_p2_score = next_score
        
        # Record state AFTER this point
        df.at[idx, 'P1Score'] = current_p1_score
        df.at[idx, 'P2Score'] = current_p2_score
        df.at[idx, 'P1GamesWon'] = current_p1_games_in_set
        df.at[idx, 'P2GamesWon'] = current_p2_games_in_set
        df.at[idx, 'GameWinner'] = game_winner
        
        # If game won, reset scores
        if game_won:
            current_p1_score = '0'
            current_p2_score = '0'
            
            # Check for set win
            set_won = False
            set_winner = 0
            
            # Standard set: need to win 6 games with 2-game margin
            if current_p1_games_in_set >= 6 and current_p1_games_in_set - current_p2_games_in_set >= 2:
                set_won = True
                set_winner = 1
            elif current_p2_games_in_set >= 6 and current_p2_games_in_set - current_p1_games_in_set >= 2:
                set_won = True
                set_winner = 2
            # Tiebreak at 6-6 (simplified: assume tiebreak winner has 7 games)
            elif current_p1_games_in_set == 7 and current_p2_games_in_set == 6:
                set_won = True
                set_winner = 1
            elif current_p2_games_in_set == 7 and current_p1_games_in_set == 6:
                set_won = True
                set_winner = 2
            
            if set_won:
                # Mark SetWinner for this point (will propagate to all points in set later)
                df.at[idx, 'SetWinner'] = set_winner
    
    return df


_FEATURE_INDEX = {name: idx for idx, name in enumerate(MATCH_FEATURE_COLUMNS)}


def advance_game_state_simple(row_features, server_wins_point: bool, point_importance: float = 1.0, eff_window: float = 20.0):
    """
    Update features to simulate counterfactual scenario where server loses/wins the point.
    
    Takes into account point_importance to amplify changes for critical points like:
    - Break points (importance ~2-5)
    - Set points (importance ~3-5)
    - Tiebreaks (importance ~3-5)

    Current feature order (22 features):
    [0] P_srv_win_long (dampened when tied)
    [1] P_srv_lose_long (dampened when tied)
    [2] P_srv_win_short
    [3] P_srv_lose_short
    [4] PointServer
    [5] momentum (dampened when tied)
    [6] Momentum_Diff
    [7] Score_Diff
    [8] Game_Diff
    [9] CurrentSetGamesDiff
    [10] SrvScr
    [11] RcvScr
    [12] SetNo
    [13] GameNo
    [14] PointNumber
    [15] point_importance
    [16] SetsWonDiff (scaled by match progress)
    [17] SetsWonAdvantage
    [18] SetWinProbPrior
    [19] SetWinProbEdge
    [20] SetWinProbLogit
    [21] is_decider_tied (1.0 in tied final set)

    For the counterfactual, we update:
    - P_srv_win/lose (long and short) using Bayesian update, amplified by importance
    - SrvScr/RcvScr based on outcome
    - Score_Diff changes (tennis scoring)
    - Game_Diff changes for game-deciding points
    - Momentum changes based on leverage and importance
    - point_importance stays the same (inherent to the game situation)
    """
    x = row_features.copy()
    
    # Extract long window probabilities
    idx = _FEATURE_INDEX
    P_win_long = float(x[idx["P_srv_win_long"]])
    P_lose_long = float(x[idx["P_srv_lose_long"]])
    P_win_short = float(x[idx["P_srv_win_short"]])
    P_lose_short = float(x[idx["P_srv_lose_short"]])
    server = int(x[idx[SERVER_COL]])
    current_momentum = float(x[idx["momentum"]])
    score_diff = float(x[idx["Score_Diff"]])
    game_diff = float(x[idx["Game_Diff"]])
    
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
    
    # Scale updates by importance (critical points have larger impact)
    # importance ranges from 1.0 (normal) to 7.0 (critical break/set point)
    # 
    # NEW APPROACH: Suppress normal points instead of amplifying critical ones
    # Use a sigmoid-like function that keeps normal points very low
    # and only starts growing significantly after importance > 2.5
    
    if point_importance <= 2.0:
        # Very conservative for truly normal points
        importance_factor = point_importance * 0.02  # 1.0->0.02, 2.0->0.04
    elif point_importance <= 3.0:
        # Moderate transition zone
        # Linear interpolation from 0.04 at 2.0 to 0.25 at 3.0
        importance_factor = 0.04 + (point_importance - 2.0) * 0.21
    else:
        # Above 3.0, use linear scaling
        # From 0.25 at 3.0 to 1.5 at 7.0
        importance_factor = 0.25 + (point_importance - 3.0) * 0.3125
        importance_factor = min(importance_factor, 1.5)
    
    # Bayesian update for long window
    alpha_long = P_win_long * eff_window
    beta_long = P_lose_long * eff_window
    
    # Update strength proportional to importance_factor
    # For normal points (importance_factor ~ 0.02), we add very few pseudo-observations
    # For critical points (importance_factor ~ 1.5), we add many
    # Use importance_factor^2.0 to further suppress normal point changes in probabilities
    bayesian_factor = np.power(importance_factor, 2.0)
    update_strength = 0.05 + bayesian_factor * 1.5  # Range: ~0.05 (normal) to 3.4 (critical)
    
    if server_wins_point:
        alpha_long_new = alpha_long + update_strength
        beta_long_new = beta_long
    else:
        alpha_long_new = alpha_long
        beta_long_new = beta_long + update_strength
    
    total_long = alpha_long_new + beta_long_new
    x[idx["P_srv_win_long"]] = alpha_long_new / total_long
    x[idx["P_srv_lose_long"]] = beta_long_new / total_long
    
    # Bayesian update for short window (more sensitive) - also amplified
    short_window = 5.0
    alpha_short = P_win_short * short_window
    beta_short = P_lose_short * short_window
    
    if server_wins_point:
        alpha_short_new = alpha_short + update_strength
        beta_short_new = beta_short
    else:
        alpha_short_new = alpha_short
        beta_short_new = beta_short + update_strength
    
    total_short = alpha_short_new + beta_short_new
    x[idx["P_srv_win_short"]] = alpha_short_new / total_short
    x[idx["P_srv_lose_short"]] = beta_short_new / total_short
    
    # Update momentum based on leverage and importance
    # Critical points have much larger momentum impact
    raw_leverage = max(0.0, x[idx["P_srv_win_long"]] - x[idx["P_srv_lose_long"]])
    
    # Cap leverage for normal points to prevent outlier changes
    # For normal points (importance <= 1.5), cap leverage at 0.15
    # For critical points, use full leverage
    if point_importance <= 1.5:
        leverage = min(raw_leverage, 0.15)
    elif point_importance <= 2.5:
        # Smooth transition: linearly interpolate cap from 0.15 to full leverage
        max_lev = 0.15 + (point_importance - 1.5) * (raw_leverage - 0.15)
        leverage = min(raw_leverage, max_lev)
    else:
        leverage = raw_leverage
    
    # Use importance_factor^2.5 to aggressively suppress momentum changes for normal points
    # 0.002^2.5 = 0.000003, 0.01^2.5 = 0.000003, 1.5^2.5 = 2.76
    momentum_factor = np.power(importance_factor, 2.5)
    momentum_change = 0.15 * leverage * momentum_factor  # Further reduced base from 0.2 to 0.15
    
    if server_wins_point:
        # Winning critical point increases momentum significantly
        x[idx["momentum"]] = current_momentum + momentum_change
    else:
        # Losing critical point decreases momentum significantly
        x[idx["momentum"]] = current_momentum - momentum_change
    
    # Clip momentum to reasonable range
    x[idx["momentum"]] = np.clip(x[idx["momentum"]], -3.0, 3.0)
    
    # Update Game_Diff and Score_Diff using smooth scaling
    # Use importance_factor^2.0 for game/score to suppress normal point changes
    # 0.002^2.0 = 0.000004, 1.5^2.0 = 2.25
    game_change = np.power(importance_factor, 2.0) * 1.0  # Reduced from 1.5 and squared
    score_change = np.power(importance_factor, 2.0) * 1.5  # Reduced from 2.5 and squared
    
    # Apply game differential change
    if server == 1:
        if server_wins_point:
            x[idx["Game_Diff"]] = game_diff + game_change
        else:
            x[idx["Game_Diff"]] = game_diff - game_change
    else:
        if server_wins_point:
            x[idx["Game_Diff"]] = game_diff - game_change
        else:
            x[idx["Game_Diff"]] = game_diff + game_change
    x[idx["Game_Diff"]] = np.clip(x[idx["Game_Diff"]], -3.0, 3.0)
    
    # Apply score differential change
    if server == 1:
        if server_wins_point:
            x[idx["Score_Diff"]] = score_diff + score_change
        else:
            x[idx["Score_Diff"]] = score_diff - score_change
    else:
        if server_wins_point:
            x[idx["Score_Diff"]] = score_diff - score_change
        else:
            x[idx["Score_Diff"]] = score_diff + score_change
    x[idx["Score_Diff"]] = np.clip(x[idx["Score_Diff"]], -2.0, 2.0)
    
    # Update SrvScr/RcvScr based on who served and won
    if server_wins_point:
        if server == 1:
            x[idx["SrvScr"]] += 1  # SrvScr
        # If server=2, P1 metrics don't change
    else:
        # Server lost
        if server == 2:
            x[idx["RcvScr"]] += 1  # RcvScr (P1 received and won)
        # If server=1 lost, P1 metrics don't improve
    
    return x


def compute_counterfactual_with_importance(X, model, point_importances, X_prev=None, point_winners=None):
    """
    Fast counterfactual using point_importance to scale feature changes.
    
    For each point i (AFTER it has been played):
    - Shows current state probabilities
    - Shows what probabilities would be if point i had opposite outcome
    
    This answers: "At this moment, what if the point that just happened went the other way?"
    
    Args:
        X: Feature matrix at current state (after each point)
        model: Trained model
        point_importances: Array of importance values for each point
        X_prev: Feature matrix at previous state (before each point). If None, uses X shifted.
        point_winners: Array of point winners (1 or 2). If provided, used to determine outcome.
    """
    n = X.shape[0]
    prob_p1 = np.zeros(n)
    prob_p2 = np.zeros(n)
    prob_p1_counterfactual = np.zeros(n)
    prob_p2_counterfactual = np.zeros(n)

    idx = _FEATURE_INDEX
    
    for i in range(n):
        x_current = X[i]
        importance_current = point_importances[i] if i < len(point_importances) else 1.0
        
        # Current probabilities (after point i was played)
        p1_now = float(model.predict_proba(x_current.reshape(1, -1))[:, 1])
        prob_p1[i] = p1_now
        prob_p2[i] = 1.0 - p1_now

        if i == 0:
            # First point: no previous state to work with
            prob_p1_counterfactual[i] = p1_now
            prob_p2_counterfactual[i] = 1.0 - p1_now
            continue
        
        # Get state BEFORE point i was played
        x_before = X_prev[i-1] if X_prev is not None else X[i-1]
        server_at_i = int(x_before[idx[SERVER_COL]])
        
        # Determine who won point i
        if point_winners is not None and i < len(point_winners):
            # Use explicit point winner information
            point_winner_i = int(point_winners[i])
        else:
            # Infer from feature changes
            # SrvScr tracks P1's serve points won, RcvScr tracks P1's return points won
            srv_scr_now = x_current[idx["SrvScr"]]
            rcv_scr_now = x_current[idx["RcvScr"]]
            srv_scr_before = x_before[idx["SrvScr"]]
            rcv_scr_before = x_before[idx["RcvScr"]]
            
            # If SrvScr or RcvScr increased, P1 won the point
            if (srv_scr_now > srv_scr_before) or (rcv_scr_now > rcv_scr_before):
                point_winner_i = 1
            else:
                point_winner_i = 2
        
        # Counterfactual: INVERT who won the point
        # If P1 won → counterfactual: P2 wins
        # If P2 won → counterfactual: P1 wins
        counterfactual_winner = 2 if point_winner_i == 1 else 1
        
        # Translate to server_wins_point for simulation
        # If counterfactual winner == server → server wins
        # If counterfactual winner != server → server loses
        server_wins_in_cf = (counterfactual_winner == server_at_i)
        
        x_counter = advance_game_state_simple(
            x_before, 
            server_wins_point=server_wins_in_cf, 
            point_importance=importance_current
        )
        
        # Get probabilities in counterfactual scenario
        p1_counter = float(model.predict_proba(x_counter.reshape(1, -1))[:, 1])
        prob_p1_counterfactual[i] = p1_counter
        prob_p2_counterfactual[i] = 1.0 - p1_counter

    # Statistics
    diffs_p1 = np.abs(prob_p1 - prob_p1_counterfactual)
    diffs_p2 = np.abs(prob_p2 - prob_p2_counterfactual)
    print(f"[counterfactual-importance] Mean change: P1={np.mean(diffs_p1):.4f}, P2={np.mean(diffs_p2):.4f}")
    print(f"[counterfactual-importance] Max change: P1={np.max(diffs_p1):.4f}, P2={np.max(diffs_p2):.4f}")
    print(f"[counterfactual-importance] Points with >10% change: P1={np.sum(diffs_p1 > 0.10)}, P2={np.sum(diffs_p2 > 0.10)}")

    return prob_p1, prob_p2, prob_p1_counterfactual, prob_p2_counterfactual


def compute_counterfactual_point_by_point(df_valid, df_raw_with_labels, model, config_path=None, 
                                          match_id=None, importance_threshold=None, mode="realistic"):
    """
    Compute counterfactual by modifying dataset point by point.
    
    For each point i:
    1. Look at the state BEFORE point i was played (i.e., after point i-1)
    2. Determine what actually happened at point i (who won)
    3. Simulate the OPPOSITE outcome
    4. Rebuild features and get model prediction
    
    This shows: "What if point i had gone the opposite way?"
    
    Args:
        df_valid: DataFrame with all features computed
        df_raw_with_labels: Raw DataFrame for rebuilding features
        model: Trained model
        config_path: Config file path
        match_id: If provided, only process points from this match
        importance_threshold: If provided, only process points above this importance
        mode: "realistic" (all points) or "semi-realistic" (critical only)
    
    Returns:
        prob_p1, prob_p2, prob_p1_counterfactual, prob_p2_counterfactual
    """
    from .features import (
        add_rolling_serve_return_features,
        add_leverage_and_momentum,
        add_additional_features,
        build_dataset,
    )
    from .config import load_config
    
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))
    
    n = len(df_valid)
    prob_p1 = np.zeros(n)
    prob_p2 = np.zeros(n)
    prob_p1_counterfactual = np.zeros(n)
    prob_p2_counterfactual = np.zeros(n)
    
    # Get current features
    X_current, _, _, _, _ = build_dataset(df_valid)
    
    # Compute current probabilities
    for i in range(len(X_current)):
        p1_now = float(model.predict_proba(X_current[i].reshape(1, -1))[:, 1])
        prob_p1[i] = p1_now
        prob_p2[i] = 1.0 - p1_now
    
    # Determine which points to simulate
    point_importances = df_valid['point_importance'].values if 'point_importance' in df_valid.columns else np.ones(n)
    
    if importance_threshold is not None:
        # Semi-realistic: only critical points
        simulate_mask = point_importances > importance_threshold
        n_simulate = np.sum(simulate_mask)
        print(f"[{mode}] Simulating {n_simulate}/{n} points with importance > {importance_threshold}")
    else:
        # Realistic: all points
        simulate_mask = np.ones(n, dtype=bool)
        n_simulate = n
        print(f"[{mode}] Simulating all {n_simulate} points")
    
    # Process each point that needs simulation
    simulated_count = 0
    indices = list(df_valid.index)
    
    for i, idx in enumerate(indices):
        if i == 0:
            # First point: no previous state
            prob_p1_counterfactual[i] = prob_p1[i]
            prob_p2_counterfactual[i] = prob_p2[i]
            continue
            
        if not simulate_mask[i]:
            # Non-critical point: no counterfactual simulation
            prob_p1_counterfactual[i] = prob_p1[i]
            prob_p2_counterfactual[i] = prob_p2[i]
            continue
        
        # Create a FRESH copy of the raw dataset
        cf_df = df_raw_with_labels.copy()
        
        # Get current row to determine actual winner
        current_row = df_raw_with_labels.loc[idx]
        point_winner = int(current_row.get('PointWinner', 1))
        
        # Counterfactual: INVERT who won the point
        # If P1 won → counterfactual: P2 wins
        # If P2 won → counterfactual: P1 wins
        counterfactual_winner = 2 if point_winner == 1 else 1
        
        # Modify ONLY the PointWinner for this point in the counterfactual dataset
        cf_df.at[idx, 'PointWinner'] = counterfactual_winner
        
        # Recalculate ALL scores, games, sets from the beginning based on modified PointWinner
        cf_df = recalculate_match_state_from_point_winners(cf_df)
        
        # Check if the counterfactual would have ended the match at THIS point
        cf_row = cf_df.loc[idx]
        
        # Count total sets won by each player up to this point
        # IMPORTANT: Only look at points UP TO idx in the ORIGINAL match sequence
        # Get the SetNo of the current point
        current_set_no = int(cf_row.get('SetNo', 1))
        
        # Count sets won up to the CURRENT set number (not beyond)
        sets_p1 = 0
        sets_p2 = 0
        
        # Only check sets 1 through current_set_no
        for set_num in range(1, current_set_no + 1):
            # Get all points in this set up to current point
            set_points = cf_df[(cf_df['SetNo'] == set_num) & (cf_df.index <= idx)]
            if len(set_points) > 0:
                # Get the last point - if SetWinner is non-zero, the set is complete
                last_point = set_points.iloc[-1]
                set_winner = last_point.get('SetWinner', 0)
                if set_winner == 0:
                    # Set not yet complete - still in progress
                    continue
                set_winner = int(set_winner)
                if set_winner == 1:
                    sets_p1 += 1
                elif set_winner == 2:
                    sets_p2 += 1
        
        # Check if match would be over (best-of-5: need 3 sets, best-of-3: need 2 sets)
        match_ended = False
        match_winner = 0
        
        # Assume best-of-5 (Grand Slam men's)
        if sets_p1 >= 3:
            match_ended = True
            match_winner = 1
        elif sets_p2 >= 3:
            match_ended = True
            match_winner = 2
        
        if match_ended:
            # Match is over! Set probability directly
            if match_winner == 1:
                prob_p1_counterfactual[i] = 1.0
                prob_p2_counterfactual[i] = 0.0
            else:
                prob_p1_counterfactual[i] = 0.0
                prob_p2_counterfactual[i] = 1.0
        else:
            # Match continues: rebuild features and predict
            # Rebuild ALL features from this modified dataset
            # This will recalculate momentum, serve statistics, etc. based on the new scores/games
            cf_df = add_rolling_serve_return_features(cf_df, long_window=long_window, short_window=short_window)
            cf_df = add_additional_features(cf_df)
            cf_df = add_leverage_and_momentum(cf_df, alpha=alpha)
            
            X_cf, _, mask_cf, _, _ = build_dataset(cf_df)
            
            # The modified point is at position i in the rebuilt array
            if i < len(X_cf) and mask_cf[i]:
                p1_cf = float(model.predict_proba(X_cf[i].reshape(1, -1))[:, 1])
                prob_p1_counterfactual[i] = p1_cf
                prob_p2_counterfactual[i] = 1.0 - p1_cf
            else:
                # Fallback
                prob_p1_counterfactual[i] = prob_p1[i]
                prob_p2_counterfactual[i] = prob_p2[i]
        
        simulated_count += 1
        if simulated_count % 50 == 0:
            print(f"[{mode}] Processed {simulated_count}/{n_simulate} points...")
    
    # Statistics
    diffs = np.abs(prob_p1 - prob_p1_counterfactual)
    print(f"[{mode}] Mean change: {np.mean(diffs):.4f}, Max: {np.max(diffs):.4f}")
    print(f"[{mode}] Points with >10% change: {np.sum(diffs > 0.10)}/{n}")
    
    if importance_threshold is not None:
        simulated_diffs = diffs[simulate_mask]
        if len(simulated_diffs) > 0:
            print(f"[{mode}] Simulated points >10% change: {np.sum(simulated_diffs > 0.10)}/{n_simulate}")

    return prob_p1, prob_p2, prob_p1_counterfactual, prob_p2_counterfactual




def run_prediction(file_paths, model_path: str, match_id: str, plot_dir: str, config_path: str | None = None, 
                   counterfactual_mode: str = "importance"):
    """
    End-to-end prediction + plotting for a given set of files and one match_id.

    - Loads data
    - Rebuilds features (including momentum and point importance)
    - Loads trained model
    - Computes current and counterfactual probabilities
    - Produces probability trajectory plots
    
    Args:
        file_paths: Paths to data files
        model_path: Path to trained model
        match_id: Match ID to analyze
        plot_dir: Directory for plots
        config_path: Path to config file
        counterfactual_mode: "importance" (default), "semi-realistic", or "realistic"
            - importance: Fast scaling using point_importance (always generated)
            - semi-realistic: Dataset modification for critical points only (importance > 3.5)
            - realistic: Dataset modification for ALL points (slow!)
    """
    os.makedirs(plot_dir, exist_ok=True)
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))

    df = load_points_multiple(file_paths)
    df = add_match_labels(df)
    
    # Keep a copy of raw data with labels for counterfactual simulation
    df_raw_with_labels = df.copy()
    
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)

    X, y, mask, sample_weights, _ = build_dataset(df)
    df_valid = df[mask].copy()

    model = load_model(model_path)

    # ALWAYS compute importance-based counterfactual (fast)
    print("\n=== COUNTERFACTUAL: Point Importance Scaling ===")
    # Extract point winners from raw data for accurate counterfactual simulation
    point_winners = df_valid['PointWinner'].values if 'PointWinner' in df_valid.columns else None
    prob_p1, prob_p2, prob_p1_lose, prob_p2_lose = compute_counterfactual_with_importance(
        X, model, sample_weights, X_prev=None, point_winners=point_winners
    )

    df_valid["prob_p1"] = prob_p1
    df_valid["prob_p2"] = prob_p2
    df_valid["prob_p1_lose_srv"] = prob_p1_lose  # Actually: prob if P1 loses this point
    df_valid["prob_p2_lose_srv"] = prob_p2_lose  # Actually: prob if P2 loses this point

    # Filter to match for additional simulations
    match_id_str = str(match_id)
    df_valid[MATCH_COL] = df_valid[MATCH_COL].astype(str)
    match_df = df_valid[df_valid[MATCH_COL] == match_id_str]
    
    # Optionally compute semi-realistic or realistic counterfactuals
    if counterfactual_mode in ["semi-realistic", "realistic"] and not match_df.empty:
        print(f"\n=== ADDITIONAL MODE: {counterfactual_mode.upper()} ===")
        
        if counterfactual_mode == "semi-realistic":
            # Only critical points in this match
            threshold = 3.0  # Lower threshold to include more match points
            prob_p1_2, prob_p2_2, prob_p1_lose_2, prob_p2_lose_2 = compute_counterfactual_point_by_point(
                match_df, df_raw_with_labels, model, config_path,
                match_id=match_id_str, importance_threshold=threshold, mode="semi-realistic"
            )
        else:  # realistic
            # ALL points in this match
            prob_p1_2, prob_p2_2, prob_p1_lose_2, prob_p2_lose_2 = compute_counterfactual_point_by_point(
                match_df, df_raw_with_labels, model, config_path,
                match_id=match_id_str, importance_threshold=None, mode="realistic"
            )
        
        # Add to match dataframe
        match_df = match_df.copy()
        match_df["prob_p1_alt"] = prob_p1_2
        match_df["prob_p2_alt"] = prob_p2_2
        match_df["prob_p1_lose_alt"] = prob_p1_lose_2
        match_df["prob_p2_lose_alt"] = prob_p2_lose_2
        
        # Update df_valid with match results
        for col in ["prob_p1_alt", "prob_p2_alt", "prob_p1_lose_alt", "prob_p2_lose_alt"]:
            df_valid.loc[match_df.index, col] = match_df[col]
    
    # Save probabilities to CSV AFTER all computations
    # Add suffix based on counterfactual mode
    mode_suffix = "" if counterfactual_mode == "importance" else f"_{counterfactual_mode}"
    probs_file = os.path.join(plot_dir, f"match_{match_id_str}_probabilities{mode_suffix}.csv")
    if not match_df.empty:
        # Save essential columns for plotting
        cols_to_save = [MATCH_COL, 'prob_p1', 'prob_p2', 'prob_p1_lose_srv', 'prob_p2_lose_srv']
        if 'point_importance' in match_df.columns:
            cols_to_save.append('point_importance')
        # If we have semi-realistic/realistic alternative probabilities, save them too
        for col in ["prob_p1_alt", "prob_p2_alt", "prob_p1_lose_alt", "prob_p2_lose_alt"]:
            if col in match_df.columns:
                cols_to_save.append(col)
        match_df[cols_to_save].to_csv(probs_file, index=False)
        print(f"[prediction] Saved probabilities to: {probs_file}")
    
    # Print statistics
    if not match_df.empty:
        print(f"\n=== MATCH {match_id_str} STATISTICS ===")
        print(f"  Points in match: {len(match_df)}")
        
        if 'point_importance' in match_df.columns:
            match_importances = match_df['point_importance'].values
            print(f"  Point importance: mean={np.mean(match_importances):.2f}, max={np.max(match_importances):.2f}")
            print(f"  Critical points (>3.5): {np.sum(match_importances > 3.5)}")
        
        # Stats for importance method
        match_diffs = np.abs(match_df["prob_p1"] - match_df["prob_p1_lose_srv"])
        print(f"\n  Importance method:")
        print(f"    Mean change: {np.mean(match_diffs):.4f}, Max: {np.max(match_diffs):.4f}")
        print(f"    Points with >10% change: {np.sum(match_diffs > 0.10)}/{len(match_diffs)}")
        
        # Stats for alternative method if computed
        if f"prob_p1_lose_alt" in match_df.columns:
            match_diffs_alt = np.abs(match_df["prob_p1_alt"] - match_df["prob_p1_lose_alt"])
            print(f"\n  {counterfactual_mode.capitalize()} method:")
            print(f"    Mean change: {np.mean(match_diffs_alt):.4f}, Max: {np.max(match_diffs_alt):.4f}")
            print(f"    Points with >10% change: {np.sum(match_diffs_alt > 0.10)}/{len(match_diffs_alt)}")

    # Generate plots
    print(f"\n=== Generating plots (mode={counterfactual_mode}) ===")
    
    if counterfactual_mode in ["semi-realistic", "realistic"] and "prob_p1_alt" in df_valid.columns:
        # Generate comparison plot
        from .plotting import plot_match_probabilities_comparison
        plot_match_probabilities_comparison(df_valid, match_id_str, plot_dir, counterfactual_mode)
    else:
        # Generate standard plot
        plot_match_probabilities(df_valid, match_id_str, plot_dir)
