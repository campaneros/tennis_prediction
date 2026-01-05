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
from .model import load_model, _predict_proba_model
from .plotting import plot_match_probabilities


def _predict_p1_proba(model, x_vec):
    """Return P1 win probability for classifier, regressor, or neural network models."""
    return float(_predict_proba_model(model, x_vec.reshape(1, -1))[0])


def is_critical_point(row, sets_to_win=3):
    """
    Identify truly critical points based on actual game situation, NOT point_importance.
    
    Critical points are:
    1. Break points: P1BreakPoint=1 or P2BreakPoint=1 (from CSV - already correct)
    2. Set points: winning this point wins the set
    3. Match points: winning this point wins the match
    4. Tiebreak critical points: score >= 3-3
    
    Args:
        row: DataFrame row with point data
        sets_to_win: Number of sets needed to win match (2 for best-of-3, 3 for best-of-5)
    
    Returns:
        bool: True if critical point
    """
    # 1. Break points (from CSV data - these are already correct)
    if 'P1BreakPoint' in row and row.get('P1BreakPoint', 0) == 1:
        return True
    if 'P2BreakPoint' in row and row.get('P2BreakPoint', 0) == 1:
        return True
    
    # 2. Tiebreak critical points (score >= 3-3) AND match/set points in tiebreak
    if 'is_tiebreak' in row and row.get('is_tiebreak', 0) == 1:
        p1_score = row.get('P1Score', 0)
        p2_score = row.get('P2Score', 0)
        try:
            p1_score_int = int(p1_score)
            p2_score_int = int(p2_score)
            
            # Tiebreak critical: both at 3 or higher
            if p1_score_int >= 3 and p2_score_int >= 3:
                # Additional check for match/set points in tiebreak
                # In tiebreak: first to 7 (with 2-point lead) wins
                p1_sets = int(row.get('P1SetsWon', 0))
                p2_sets = int(row.get('P2SetsWon', 0))
                
                # P1 is at match/set point in tiebreak
                if p1_score_int >= 6 and p1_score_int >= p2_score_int + 1:
                    # Match point if P1 is one set away from winning
                    if p1_sets >= sets_to_win - 1:
                        return True  # P1 match point in tiebreak
                    else:
                        return True  # P1 set point in tiebreak
                
                # P2 is at match/set point in tiebreak
                if p2_score_int >= 6 and p2_score_int >= p1_score_int + 1:
                    # Match point if P2 is one set away from winning
                    if p2_sets >= sets_to_win - 1:
                        return True  # P2 match point in tiebreak
                    else:
                        return True  # P2 set point in tiebreak
                
                return True  # Other critical tiebreak point (>=3-3)
        except (ValueError, TypeError):
            pass
    
    # 3. Set points and Match points
    p1_games = int(row.get('P1GamesWon', 0))
    p2_games = int(row.get('P2GamesWon', 0))
    p1_score = str(row.get('P1Score', '0')).strip()
    p2_score = str(row.get('P2Score', '0')).strip()
    p1_sets = int(row.get('P1SetsWon', 0))
    p2_sets = int(row.get('P2SetsWon', 0))
    
    # Helper function to check if winning current point wins the game
    def can_win_game_on_this_point(my_score, opp_score):
        """Check if player can win game by winning this point."""
        # At 40, win point = win game (unless opponent is at 40/AD)
        if my_score == '40':
            return opp_score not in ['40', 'AD']
        # At AD (advantage), win point = win game
        if my_score == 'AD':
            return True
        return False
    
    # Helper function to check if winning game wins the set
    def can_win_set_by_winning_game(my_games, opp_games):
        """Check if winning the current game wins the set."""
        # Standard set: need to reach 6 games with 2-game lead
        # OR 7 games (after winning tiebreak at 6-6)
        # In final set without tiebreak: need 2-game lead at any score >= 6
        
        if my_games >= 5:
            # At 5 games: winning makes it 6, need opponent at <=4 to win set
            if my_games == 5 and opp_games <= 4:
                return True
            # At 6+ games: need 2-game lead (works for extended sets like 8-7, 9-7, etc.)
            if my_games >= 6 and my_games >= opp_games + 1:
                return True
        return False
    
    # Check P1 set point
    is_p1_set_point = False
    if can_win_game_on_this_point(p1_score, p2_score):
        if can_win_set_by_winning_game(p1_games, p2_games):
            is_p1_set_point = True
    
    # Check P2 set point
    is_p2_set_point = False
    if can_win_game_on_this_point(p2_score, p1_score):
        if can_win_set_by_winning_game(p2_games, p1_games):
            is_p2_set_point = True
    
    # Match point: set point AND player is one set away from winning match
    is_p1_match_point = is_p1_set_point and (p1_sets >= sets_to_win - 1)
    is_p2_match_point = is_p2_set_point and (p2_sets >= sets_to_win - 1)
    
    # Return True for ANY critical point
    if is_p1_match_point or is_p2_match_point:
        return True  # Match points
    
    if is_p1_set_point or is_p2_set_point:
        return True  # Set points (not match points)
    
    return False


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


def advance_game_state_simple(row_features, point_winner: int, point_importance: float = 1.0, eff_window: float = 20.0):
    """
    Update features to simulate counterfactual scenario based on which player wins the point.
    
    Args:
        row_features: Current feature vector
        point_winner: 1 if P1 wins the point, 2 if P2 wins the point
        point_importance: Importance weight of the point
        eff_window: Effective window for Bayesian updates
    
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
    
    # Determine if server wins based on point_winner
    server_wins = (point_winner == server)
    
    # Bayesian update for long window
    alpha_long = P_win_long * eff_window
    beta_long = P_lose_long * eff_window
    
    # Update strength proportional to importance_factor
    # For normal points (importance_factor ~ 0.02), we add very few pseudo-observations
    # For critical points (importance_factor ~ 1.5), we add many
    # Use importance_factor^2.0 to further suppress normal point changes in probabilities
    bayesian_factor = np.power(importance_factor, 2.0)
    update_strength = 0.05 + bayesian_factor * 1.5  # Range: ~0.05 (normal) to 3.4 (critical)
    
    if server_wins:
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
    
    if server_wins:
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
    
    if server_wins:
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
    base_game_change = np.power(importance_factor, 2.0) * 1.0  # Reduced from 1.5 and squared
    score_change = np.power(importance_factor, 2.0) * 1.5  # Reduced from 2.5 and squared
    
    # Apply game differential change based on which player wins
    # KEY: Break (winning game in return) should count MORE than hold (winning game on serve)
    # BUT: make weights contextual - in final set, everything matters more equally
    # Determine if we're in a critical situation (final set, close sets)
    set_no = int(x[idx["SetNo"]]) if "SetNo" in idx else 1
    p1_sets = int(x[idx["SetsWonAdvantage"]]) if "SetsWonAdvantage" in idx else 0  # approximation
    
    # Context-dependent weights:
    # Early match (sets 1-3): hold=0.6x, break=2.0x (significant difference)
    # Late match (set 4+): hold=0.75x, break=1.6x (moderate difference)
    # Final set (5): hold=0.85x, break=1.3x (both important)
    if set_no >= 5:
        # Final set: both hold and break are critical
        hold_weight = 0.85
        break_weight = 1.3
    elif set_no >= 4:
        # Late match: holds still less important but not negligible
        hold_weight = 0.75
        break_weight = 1.6
    else:
        # Early match: holds expected, breaks significant
        hold_weight = 0.6
        break_weight = 2.0
    
    # If P1 wins, P1 advantage increases; if P2 wins, P1 advantage decreases
    if point_winner == 1:
        # P1 wins the point
        # Check if P1 is serving: if server==1, this is a hold (weight less)
        # If server==2, this is a break (weight more)
        if server == 1:
            # P1 holding serve
            game_change = base_game_change * hold_weight
        else:
            # P1 breaking serve
            game_change = base_game_change * break_weight
        x[idx["Game_Diff"]] = game_diff + game_change
    else:
        # P2 wins the point
        # Check if P2 is serving: if server==2, this is a hold (weight less)
        # If server==1, this is a break (weight more)
        if server == 2:
            # P2 holding serve
            game_change = base_game_change * hold_weight
        else:
            # P2 breaking serve
            game_change = base_game_change * break_weight
        x[idx["Game_Diff"]] = game_diff - game_change
    x[idx["Game_Diff"]] = np.clip(x[idx["Game_Diff"]], -3.0, 3.0)
    
    # Apply score differential change based on which player wins
    # Also differentiate based on serve/return context (same weights as game)
    if point_winner == 1:
        # P1 wins the point
        if server == 1:
            # P1 holding serve
            effective_score_change = score_change * hold_weight
        else:
            # P1 breaking serve
            effective_score_change = score_change * break_weight
        x[idx["Score_Diff"]] = score_diff + effective_score_change
    else:
        # P2 wins the point
        if server == 2:
            # P2 holding serve
            effective_score_change = score_change * hold_weight
        else:
            # P2 breaking serve
            effective_score_change = score_change * break_weight
        x[idx["Score_Diff"]] = score_diff - effective_score_change
    x[idx["Score_Diff"]] = np.clip(x[idx["Score_Diff"]], -2.0, 2.0)
    
    # Update SrvScr/RcvScr based on who served and who won
    if point_winner == 1:
        if server == 1:
            x[idx["SrvScr"]] += 1  # P1 served and won
        else:
            x[idx["RcvScr"]] += 1  # P1 received and won
    # If P2 wins (point_winner == 2), P1 metrics don't improve
    
    return x


def compute_counterfactual_with_importance(X, model, point_importances, X_prev=None, point_winners=None):
    """
    Fast alternative probability: for each point, estimate the current win prob
    and the win prob if the PLAYER WHO ACTUALLY WON THE POINT had lost it.
    Uses only information up to that point (state before the point, plus flipped
    outcome) with a lightweight feature update.
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
        
        # Current probability (state built from all points up to this one)
        p1_now = _predict_p1_proba(model, x_current)
        prob_p1[i] = p1_now
        prob_p2[i] = 1.0 - p1_now

        if point_winners is None or i == 0:
            prob_p1_counterfactual[i] = p1_now
            prob_p2_counterfactual[i] = 1.0 - p1_now
            continue

        # State BEFORE point i (after point i-1)
        x_before = X_prev[i - 1] if X_prev is not None else X[i - 1]

        actual_winner = int(point_winners[i])
        counterfactual_winner = 2 if actual_winner == 1 else 1

        x_counter = advance_game_state_simple(
            x_before,
            point_winner=counterfactual_winner,
            point_importance=importance_current
        )

        p1_cf = _predict_p1_proba(model, x_counter)
        prob_p1_counterfactual[i] = p1_cf
        prob_p2_counterfactual[i] = 1.0 - p1_cf

    # Statistics
    diffs_p1 = np.abs(prob_p1 - prob_p1_counterfactual)
    diffs_p2 = np.abs(prob_p2 - prob_p2_counterfactual)
    print(f"[counterfactual-importance] Mean change: P1={np.mean(diffs_p1):.4f}, P2={np.mean(diffs_p2):.4f}")
    print(f"[counterfactual-importance] Max change: P1={np.max(diffs_p1):.4f}, P2={np.max(diffs_p2):.4f}")
    print(f"[counterfactual-importance] Points with >10% change: P1={np.sum(diffs_p1 > 0.10)}, P2={np.sum(diffs_p2 > 0.10)}")

    return prob_p1, prob_p2, prob_p1_counterfactual, prob_p2_counterfactual


def compute_counterfactual_point_by_point(df_valid, df_raw_with_labels, model, config_path=None, 
                                          match_id=None, mode="realistic"):
    """
    Compute counterfactual by modifying dataset point by point.
    
    For each point i:
    1. Look at the state BEFORE point i was played (i.e., after point i-1)
    2. Determine what actually happened at point i (who won)
    3. Simulate the OPPOSITE outcome
    4. Rebuild features and get model prediction
    
    Modes:
      - realistic: rebuild entire match with flipped point for ALL points
      - semi-realistic: rebuild ONLY for CRITICAL points (break points, set points, match points, tiebreak >=3-3)
                       AND build dataset point-by-point to avoid future information leakage
      - importance: use feature importance scaling (fast approximation)
    
    Args:
        df_valid: DataFrame with all features computed
        df_raw_with_labels: Raw DataFrame for rebuilding features
        model: Trained model
        config_path: Config file path
        match_id: If provided, only process points from this match
        mode: "realistic" (all points), "semi-realistic" (critical only + point-by-point), or "importance" (scaling)
    
    Returns:
        prob_p1, prob_p2, prob_p1_counterfactual, prob_p2_counterfactual, simulate_mask
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
    
    # Build dataset ONCE for the entire match (much more efficient)
    print(f"[{mode}] Building full match dataset once...")
    
    # Precompute per-match max values to preserve match format and normalization
    set_no_full_max = None
    game_no_full_max = None
    point_no_full_max = None
    if 'SetNo' in df_raw_with_labels.columns:
        set_no_full_max = df_raw_with_labels.groupby(MATCH_COL)['SetNo'].transform('max')
    if 'GameNo' in df_raw_with_labels.columns:
        game_no_full_max = df_raw_with_labels.groupby(MATCH_COL)['GameNo'].transform('max')
    if 'PointNumber' in df_raw_with_labels.columns:
        point_no_full_max = df_raw_with_labels.groupby(MATCH_COL)['PointNumber'].transform('max')
    
    # Build full dataset with features
    full_df = df_raw_with_labels.copy()
    if set_no_full_max is not None:
        full_df['SetNo_full_max'] = set_no_full_max.values
    if game_no_full_max is not None:
        full_df['GameNo_full_max'] = game_no_full_max.values
    if point_no_full_max is not None:
        full_df['PointNumber_full_max'] = point_no_full_max.values
    
    # For point-by-point mode, weight serve wins less and return wins more
    # This prevents over-reaction to normal holds during incremental reconstruction
    use_weighted = (mode == "semi-realistic")
    
    full_df = add_rolling_serve_return_features(full_df, long_window=long_window, short_window=short_window, 
                                                 weight_serve_return=use_weighted)
    full_df = add_additional_features(full_df)
    full_df = add_leverage_and_momentum(full_df, alpha=alpha)
    # NON aggiungiamo break features
    
    X_full, _, mask_full, _, _ = build_dataset(full_df)
    print(f"[{mode}] Full dataset built: {len(X_full)} valid points")
    
    # Create mapping from df_valid index to X_full position
    indices = list(df_valid.index)
    index_to_x_pos = {}
    valid_positions = np.flatnonzero(mask_full)
    
    for i, idx in enumerate(indices):
        if idx in full_df.index:
            try:
                df_pos = full_df.index.get_loc(idx)
                if df_pos < len(mask_full) and mask_full[df_pos]:
                    match_pos = np.where(valid_positions == df_pos)[0]
                    if len(match_pos) > 0:
                        index_to_x_pos[i] = int(match_pos[0])
            except KeyError:
                pass
    
    # For realistic mode, use full dataset for current probabilities
    build_point_by_point = (mode == "semi-realistic")  # Build dataset incrementally
    
    if not build_point_by_point:
        for i in range(n):
            if i in index_to_x_pos:
                x_pos = index_to_x_pos[i]
                if x_pos < len(X_full):
                    p1_now = _predict_p1_proba(model, X_full[x_pos])
                    prob_p1[i] = p1_now
                    prob_p2[i] = 1.0 - p1_now
    
    # Determine which points to simulate based on mode
    if build_point_by_point:
        # Semi-realistic: only critical points (break points, set points, match points, tiebreak >=3-3)
        # Determine sets_to_win for each point
        if 'SetNo_full_max' in df_valid.columns:
            max_sets = df_valid['SetNo_full_max'].values
            sets_to_win_array = np.where(max_sets >= 4, 3, 2)  # 3 for bo5, 2 for bo3
        elif 'SetNo' in df_valid.columns:
            max_sets = df_valid.groupby(MATCH_COL)['SetNo'].transform('max').values
            sets_to_win_array = np.where(max_sets >= 4, 3, 2)
        else:
            sets_to_win_array = np.full(n, 3)  # default to bo5
        
        # Check each row for critical points
        simulate_mask = np.zeros(n, dtype=bool)
        for i, (idx, row) in enumerate(df_valid.iterrows()):
            sets_to_win = int(sets_to_win_array[i])
            simulate_mask[i] = is_critical_point(row, sets_to_win=sets_to_win)
        
        n_simulate = np.sum(simulate_mask)
        print(f"[{mode}] Simulating {n_simulate}/{n} CRITICAL points (break/set/match points, tiebreak >=3-3)")
    else:
        # Realistic: all points
        simulate_mask = np.ones(n, dtype=bool)
        n_simulate = n
        print(f"[{mode}] Simulating all {n_simulate} points")
    
    # Process each point
    simulated_count = 0
    computed_count = 0
    
    for i, idx in enumerate(indices):
        # For semi-realistic mode, ALWAYS build dataset point-by-point (no future information)
        # For other modes, use pre-built X_full
        
        if build_point_by_point:
            # Build dataset with only points up to current index (no future information)
            current_df = df_raw_with_labels.loc[:idx].copy()
            if set_no_full_max is not None:
                current_df['SetNo_full_max'] = set_no_full_max.loc[current_df.index].values
            if game_no_full_max is not None:
                current_df['GameNo_full_max'] = game_no_full_max.loc[current_df.index].values
            if point_no_full_max is not None:
                current_df['PointNumber_full_max'] = point_no_full_max.loc[current_df.index].values
            
            # Rebuild features for this prefix only WITH WEIGHTED SERVE/RETURN
            # This prevents over-reaction to holds during point-by-point reconstruction
            current_df = add_rolling_serve_return_features(current_df, long_window=long_window, short_window=short_window,
                                                           weight_serve_return=True)
            current_df = add_additional_features(current_df)
            current_df = add_leverage_and_momentum(current_df, alpha=alpha)
            # NON aggiungiamo break features
            
            X_curr, _, mask_curr, _, _ = build_dataset(current_df)
            
            # Get the last valid point (which is our current point)
            if len(X_curr) > 0:
                p1_now = _predict_p1_proba(model, X_curr[-1])
                prob_p1[i] = p1_now
                prob_p2[i] = 1.0 - p1_now
                computed_count += 1
                
                if computed_count % 50 == 0:
                    print(f"[{mode}] Computing current probabilities: {computed_count}/{n} points...")
        
        if i == 0:
            # First point: no previous state for counterfactual
            prob_p1_counterfactual[i] = prob_p1[i]
            prob_p2_counterfactual[i] = prob_p2[i]
            continue
            
        if not simulate_mask[i]:
            # Non-critical point: no counterfactual simulation
            prob_p1_counterfactual[i] = prob_p1[i]
            prob_p2_counterfactual[i] = prob_p2[i]
            continue
        
        # COUNTERFACTUAL: Build dataset with INVERTED winner for this specific point
        # We reuse current_df if we're in semi-realistic mode (already built above)
        if build_point_by_point:
            # Reuse the current_df built above, just invert the PointWinner
            cf_df = current_df.copy()
        else:
            # Realistic mode: build from scratch
            cf_df = df_raw_with_labels.loc[:idx].copy()
            if set_no_full_max is not None:
                cf_df['SetNo_full_max'] = set_no_full_max.loc[cf_df.index].values
            if game_no_full_max is not None:
                cf_df['GameNo_full_max'] = game_no_full_max.loc[cf_df.index].values
            if point_no_full_max is not None:
                cf_df['PointNumber_full_max'] = point_no_full_max.loc[cf_df.index].values
        
        # Get current row to determine actual winner
        current_row = df_raw_with_labels.loc[idx]
        point_winner = int(current_row.get('PointWinner', 1))
        
        # Counterfactual: INVERT who won the point
        counterfactual_winner = 2 if point_winner == 1 else 1
        cf_df.at[idx, 'PointWinner'] = counterfactual_winner
        
        # Recalculate ALL scores, games, sets from the beginning based on modified PointWinner
        cf_df = recalculate_match_state_from_point_winners(cf_df)

        # Determine if the flipped outcome would end the match; otherwise avoid marking MatchFinished.
        if "SetNo_full_max" in cf_df.columns:
            sets_needed = 3 if float(cf_df["SetNo_full_max"].max()) >= 4 else 2
        else:
            sets_needed = 3
        current_set_no = int(cf_df.at[idx, "SetNo"]) if "SetNo" in cf_df else 1
        sets_p1 = 0
        sets_p2 = 0

        for set_num in range(1, current_set_no + 1):
            set_points = cf_df[(cf_df["SetNo"] == set_num) & (cf_df.index <= idx)]
            if len(set_points) == 0:
                continue
            last_point = set_points.iloc[-1]
            set_winner = int(last_point.get("SetWinner", 0))
            if set_winner == 1:
                sets_p1 += 1
            elif set_winner == 2:
                sets_p2 += 1

        match_ended = (sets_p1 >= sets_needed) or (sets_p2 >= sets_needed)

        # Rebuild ALL features from this modified dataset using the counterfactual outcome.
        # This recalculates momentum, serve statistics, etc. based on the new scores/games.
        # Use weighted serve/return for point-by-point mode to reduce hold over-reaction
        cf_df = add_rolling_serve_return_features(cf_df, long_window=long_window, short_window=short_window,
                                                  weight_serve_return=(mode == "semi-realistic"))
        cf_df = add_additional_features(cf_df)
        cf_df = add_leverage_and_momentum(cf_df, alpha=alpha)
        # NON aggiungiamo break features

        # If match did NOT end after flipping, ensure MatchFinished stays 0 so the model
        # does not treat this point as terminal.
        if not match_ended and 'MatchFinished' in cf_df.columns:
            cf_df['MatchFinished'] = 0.0
        
        X_cf, _, mask_cf, _, _ = build_dataset(cf_df)
        
        # Map the flipped point to the correct position in X_cf (which is mask-compacted)
        target_pos = None
        if idx in cf_df.index:
            try:
                target_pos = cf_df.index.get_loc(idx)
            except KeyError:
                target_pos = None
        if target_pos is None:
            target_pos = len(cf_df) - 1  # fallback to last row

        valid_positions = np.flatnonzero(mask_cf)
        x_pos = None
        if target_pos < len(mask_cf) and mask_cf[target_pos]:
            # Exact mapping exists; locate its position in the compacted array
            match = np.where(valid_positions == target_pos)[0]
            if len(match) > 0:
                x_pos = int(match[0])
        if x_pos is None:
            # Fallback: use the last valid position at or before target_pos, else the last valid overall
            before = valid_positions[valid_positions <= target_pos]
            if len(before) > 0:
                x_pos = int(np.where(valid_positions == before[-1])[0][0])
            elif len(valid_positions) > 0:
                x_pos = int(len(valid_positions) - 1)

        if x_pos is not None and x_pos < len(X_cf):
            p1_cf = _predict_p1_proba(model, X_cf[x_pos])
        else:
            p1_cf = prob_p1[i]

        prob_p1_counterfactual[i] = p1_cf
        prob_p2_counterfactual[i] = 1.0 - p1_cf
        
        simulated_count += 1
        if simulated_count % 50 == 0:
            print(f"[{mode}] Processed {simulated_count}/{n_simulate} points...")
    
    # Statistics
    diffs = np.abs(prob_p1 - prob_p1_counterfactual)
    print(f"[{mode}] Mean change: {np.mean(diffs):.4f}, Max: {np.max(diffs):.4f}")
    print(f"[{mode}] Points with >10% change: {np.sum(diffs > 0.10)}/{n}")
    
    if build_point_by_point:
        simulated_diffs = diffs[simulate_mask]
        if len(simulated_diffs) > 0:
            print(f"[{mode}] Critical points >10% change: {np.sum(simulated_diffs > 0.10)}/{n_simulate}")

    return prob_p1, prob_p2, prob_p1_counterfactual, prob_p2_counterfactual, simulate_mask


def compute_probabilities_causal(df_raw, model, config_path=None, match_id=None, progress_interval=50):
    """
    Compute match win probabilities point by point in a CAUSAL manner.
    
    For each point i:
    1. Use ONLY data from points 0 to i-1 (no future information)
    2. Rebuild features using only past points
    3. Predict probability for point i
    
    This is the true "online" prediction mode where the model doesn't see future points.
    
    Args:
        df_raw: Raw DataFrame with point-by-point data
        model: Trained model
        config_path: Path to config file
        match_id: Match ID (for filtering if needed)
        progress_interval: Print progress every N points
    
    Returns:
        prob_p1: Array of P1 win probabilities (one per point)
        prob_p2: Array of P2 win probabilities (one per point)
    """
    from .features import (
        add_match_labels,
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
    
    print(f"[causal] Computing probabilities point by point (causal mode)")
    print(f"[causal] Total points to process: {len(df_raw)}")
    
    prob_p1 = np.zeros(len(df_raw))
    prob_p2 = np.zeros(len(df_raw))
    
    # Process each point causally
    for i in range(len(df_raw)):
        # Take only points up to current point (inclusive)
        df_history = df_raw.iloc[:i+1].copy()
        
        # Rebuild features using only historical data
        df_history = add_match_labels(df_history)
        df_history = add_rolling_serve_return_features(df_history, long_window=long_window, short_window=short_window)
        df_history = add_additional_features(df_history)
        df_history = add_leverage_and_momentum(df_history, alpha=alpha)
        
        
        # Get prediction for the current (last) point
        if len(X_history) > 0 and mask_history[-1]:
            # Last point is valid
            x_current = X_history[-1:, :]
            p1_prob = _predict_proba_model(model, x_current)[0]
            prob_p1[i] = p1_prob
            prob_p2[i] = 1.0 - p1_prob
        else:
            # Fallback: uniform probability
            prob_p1[i] = 0.5
            prob_p2[i] = 0.5
        
        # Progress logging
        if (i + 1) % progress_interval == 0 or i == len(df_raw) - 1:
            print(f"[causal] Processed {i+1}/{len(df_raw)} points...")
    
    print(f"[causal] Causal computation complete")
    print(f"[causal] P1 probability: mean={prob_p1.mean():.4f}, min={prob_p1.min():.4f}, max={prob_p1.max():.4f}")
    
    return prob_p1, prob_p2


def run_prediction(file_paths, model_path: str, match_id: str, plot_dir: str, config_path: str | None = None, 
                   counterfactual_mode: str = "importance", gender: str = "male", causal: bool = False):
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
        gender: Filter by gender - "male" (match_id<2000), "female" (match_id>=2000), or "both" (all)
        point_by_point: If True, rebuild dataset point by point (slower but more accurate).
                       If False (default), use full match info (faster).
    """
    os.makedirs(plot_dir, exist_ok=True)
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))

    # Load data
    df = load_points_multiple(file_paths)
    
    # FILTER TO SPECIFIC MATCH IMMEDIATELY (before any processing)
    match_id_str = str(match_id)
    df[MATCH_COL] = df[MATCH_COL].astype(str)
    df = df[df[MATCH_COL] == match_id_str].copy()
    
    if df.empty:
        print(f"[predict] ERROR: No data found for match_id '{match_id_str}'")
        return
    
    print(f"[predict] Filtered to match {match_id_str}: {len(df)} points")
    
    # Filter by gender if specified (should not be necessary if match already filtered, but keep for safety)
    if gender != "both":
        extracted = df['match_id'].str.extract(r'-(\d+)$')[0]
        valid_mask = extracted.notna()
        if valid_mask.any():
            df_temp = df[valid_mask].copy()
            df_temp['match_num'] = extracted[valid_mask].astype(int)
            
            if gender == "male":
                df_temp = df_temp[df_temp['match_num'] < 2000].copy()
            else:  # female
                df_temp = df_temp[df_temp['match_num'] >= 2000].copy()
            
            df_temp = df_temp.drop(columns=['match_num'])
            df = df_temp
            print(f"[predict] After gender filter ({gender}): {len(df)} points")
    
    # Add match labels
    df = add_match_labels(df)
    
    # Keep a copy of raw data with labels for counterfactual simulation
    df_raw_with_labels = df.copy()
    
    # CAUSAL MODE: compute probabilities point-by-point without seeing future
    if causal:
        print(f"\n[CAUSAL MODE] Computing probabilities point-by-point without future information...")
        prob_p1, prob_p2 = compute_probabilities_causal(
            df, model_path, config_path, long_window, short_window, alpha
        )
        
        # Build features for plotting and statistics (using full match info for point_importance)
        df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
        df = add_additional_features(df)
        df = add_leverage_and_momentum(df, alpha=alpha)
        X, y, mask, sample_weights, _ = build_dataset(df)
        df_valid = df[mask].copy()
        
        # Assign causal probabilities
        df_valid["prob_p1"] = prob_p1[mask]
        df_valid["prob_p2"] = prob_p2[mask]
        df_valid["counterfactual_computed"] = False  # No counterfactual in causal mode
        
    else:
        # STANDARD MODE: use full match info (faster)
        print(f"[predict] Building features using FULL MATCH INFO for match {match_id_str} (faster)...")
        df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
        df = add_additional_features(df)
        df = add_leverage_and_momentum(df, alpha=alpha)
        
        X, y, mask, sample_weights, _ = build_dataset(df)
        df_valid = df[mask].copy()
    
    print(f"[predict] Dataset built: {len(df_valid)} valid points for match {match_id_str}")

    # Load model (only if not already loaded in causal mode)
    if not causal:
        model = load_model(model_path)

    # COUNTERFACTUAL ANALYSIS (skip in causal mode)
    if not causal:
        # ALWAYS compute importance-based counterfactual (fast)
        print("\n=== COUNTERFACTUAL: Point Importance Scaling ===")
        point_winners = df_valid['PointWinner'].values if 'PointWinner' in df_valid.columns else None
        point_importances = df_valid['point_importance'].values if 'point_importance' in df_valid.columns else np.ones(len(df_valid))
        
        # Use fast importance-based approach for counterfactual
        prob_p1, prob_p2, prob_p1_lose, prob_p2_lose = compute_counterfactual_with_importance(
            X, model, point_importances, X_prev=None, point_winners=point_winners
        )

        df_valid["prob_p1"] = prob_p1
        df_valid["prob_p2"] = prob_p2
        df_valid["counterfactual_computed"] = True
        # Counterfactual: probability if the actual point winner had lost that point
        df_valid["prob_p1_lose_srv"] = prob_p1_lose
        df_valid["prob_p2_lose_srv"] = prob_p2_lose

    # For semi-realistic/realistic modes, we already have only the match data
    match_df = df_valid.copy()
    
    # Optionally compute semi-realistic or realistic counterfactuals
    if counterfactual_mode in ["semi-realistic", "realistic"]:
        print(f"\n=== ADDITIONAL MODE: {counterfactual_mode.upper()} ===")
        
        if counterfactual_mode == "semi-realistic":
            # Only CRITICAL points: break points, set points, match points, tiebreak >=3-3
            prob_p1_2, prob_p2_2, prob_p1_lose_2, prob_p2_lose_2, simulate_mask = compute_counterfactual_point_by_point(
                match_df, df_raw_with_labels, model, config_path,
                match_id=match_id_str, mode="semi-realistic"
            )
        else:  # realistic
            # ALL points in this match
            prob_p1_2, prob_p2_2, prob_p1_lose_2, prob_p2_lose_2, simulate_mask = compute_counterfactual_point_by_point(
                match_df, df_raw_with_labels, model, config_path,
                match_id=match_id_str, mode="realistic"
            )
        
        # Add to match dataframe
        match_df = match_df.copy()
        match_df["prob_p1_alt"] = prob_p1_2
        match_df["prob_p2_alt"] = prob_p2_2
        match_df["prob_p1_lose_alt"] = prob_p1_lose_2
        match_df["prob_p2_lose_alt"] = prob_p2_lose_2
        match_df["counterfactual_computed"] = simulate_mask
        
        # Update df_valid with match results
        for col in ["prob_p1_alt", "prob_p2_alt", "prob_p1_lose_alt", "prob_p2_lose_alt", "counterfactual_computed"]:
            df_valid.loc[match_df.index, col] = match_df[col]
    else:
        # Ensure columns exist for consistent CSV even if no alt computation
        match_df = match_df.copy()
        if "prob_p1_alt" not in match_df.columns:
            match_df["prob_p1_alt"] = match_df["prob_p1"]
        if "prob_p2_alt" not in match_df.columns:
            match_df["prob_p2_alt"] = match_df["prob_p2"]
        if "prob_p1_lose_alt" not in match_df.columns:
            match_df["prob_p1_lose_alt"] = match_df["prob_p1_lose_srv"]
        if "prob_p2_lose_alt" not in match_df.columns:
            match_df["prob_p2_lose_alt"] = match_df["prob_p2_lose_srv"]
        if "counterfactual_computed" not in match_df.columns:
            match_df["counterfactual_computed"] = True
    
    # Save probabilities to CSV AFTER all computations
    # Add suffix based on counterfactual mode
    mode_suffix = "" if counterfactual_mode == "importance" else f"_{counterfactual_mode}"
    probs_file = os.path.join(plot_dir, f"match_{match_id_str}_probabilities{mode_suffix}.csv")
    if not match_df.empty:
        # Save essential columns for plotting
        cols_to_save = [
            MATCH_COL,
            'prob_p1',
            'prob_p2',
            'prob_p1_lose_srv',
            'prob_p2_lose_srv',
            'point_importance' if 'point_importance' in match_df.columns else None,
            'counterfactual_computed' if 'counterfactual_computed' in match_df.columns else None,
            'prob_p1_alt',
            'prob_p2_alt',
            'prob_p1_lose_alt',
            'prob_p2_lose_alt',
        ]
        # Drop Nones and keep unique order
        cols_to_save = [c for c in cols_to_save if c is not None]
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