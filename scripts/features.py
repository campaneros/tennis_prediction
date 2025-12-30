import numpy as np
import pandas as pd

# Try relative import first (when used as module), fall back to absolute
try:
    from .data_loader import (
        MATCH_COL,
        SERVER_COL,
        POINT_WINNER_COL,
        GAME_WINNER_COL,
    )
except ImportError:
    from data_loader import (
        MATCH_COL,
        SERVER_COL,
        POINT_WINNER_COL,
        GAME_WINNER_COL,
    )

# Canonical order for match-level feature matrices. This ensures training and
# inference share the exact same column ordering, which is critical when using
# numpy arrays instead of column-aware structures.
MATCH_FEATURE_COLUMNS = [
    "P_srv_win_long",
    "P_srv_lose_long",
    "P_srv_win_short",
    "P_srv_lose_short",
    SERVER_COL,
    "momentum",
    "Momentum_Diff",
    "Score_Diff",
    "Game_Diff",
    "CurrentSetGamesDiff",
    "SrvScr",
    "RcvScr",
    "SetNo",
    "GameNo",
    "PointNumber",
    "point_importance",
    "P1SetsWon",
    "P2SetsWon",
    "SetsWonDiff",
    "SetsWonAdvantage",
    "SetWinProbPrior",
    "SetWinProbEdge",
    "SetWinProbLogit",
    "is_decider_tied",
    "DistanceToMatchEnd",
    "MatchFinished",
    "is_tiebreak",
    "is_decisive_tiebreak",
    "tiebreak_score_diff",
    "tiebreak_win_proximity",
    "is_tiebreak_late_stage",
    # Break/Hold features - CRITICAL per distinguere importanza dei game
    "P1BreaksWon",
    "P2BreaksWon",
    "BreaksDiff",
    "P1HoldsWon",
    "P2HoldsWon",
    "HoldRate_P1",
    "HoldRate_P2",
    "BreakRate_P1",
    "BreakRate_P2",
    "weighted_game_diff",
    # Early match indicator
    "is_early_match",
    # Set situation indicators - binari e chiari
    "is_p1_ahead_sets",  # P1 ha vinto più set
    "is_p2_ahead_sets",  # P2 ha vinto più set
    "is_sets_tied",      # Set pari (stesso numero)
    "sets_magnitude",    # Magnitudine del vantaggio (0, 1, 2)
]

# Point-level predictions reuse the exact same feature ordering.
POINT_FEATURE_COLUMNS = MATCH_FEATURE_COLUMNS.copy()

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
      - point_importance: weight/importance of each point
      - SetWinProbPrior / SetWinProbEdge / SetWinProbLogit: calibrated prior for P1 match win
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
    
    def calculate_point_importance(row):
        """
        Calculate importance/weight of a point based on game situation.
        
        Considers:
        - Closeness to winning: 30-40 or 40-AD is more critical than 0-40
        - Deuce/advantage: need 2 points in a row to win
        - Break points weighted by how close to winning
        - Game points weighted by how close to winning
        - Set-deciding points have highest weights
        - Tiebreak points
        
        Returns weight in range [1.0, 7.0]
        """
        base_weight = 1.0
        
        # Get scores and detect tiebreak
        # In tiebreak, scores are numeric (1,2,3,...) not tennis scores (15,30,40,AD)
        p1_score_raw = row.get('P1Score', 0)
        p2_score_raw = row.get('P2Score', 0)
        
        # Detect tiebreak by checking if scores are NOT standard tennis scores
        is_tiebreak = False
        try:
            # Check if scores are purely numeric (tiebreak) vs tennis scores
            p1_str = str(p1_score_raw).strip()
            p2_str = str(p2_score_raw).strip()
            
            # Standard tennis scores
            tennis_scores = {'0', '15', '30', '40', 'AD', 'A'}
            
            # If both scores are NOT in tennis_scores set, it's a tiebreak
            if p1_str not in tennis_scores and p2_str not in tennis_scores:
                # Try to parse as integers (tiebreak scores)
                p1_score = int(p1_str)
                p2_score = int(p2_str)
                is_tiebreak = True
            else:
                p1_score = score_to_numeric(p1_score_raw)
                p2_score = score_to_numeric(p2_score_raw)
        except (ValueError, TypeError):
            # Fallback to standard conversion
            p1_score = score_to_numeric(p1_score_raw)
            p2_score = score_to_numeric(p2_score_raw)
        
        server = row.get(SERVER_COL, 1)
        
        # Get games won (for set point detection)
        p1_games = pd.to_numeric(row.get('P1GamesWon', 0), errors='coerce')
        p2_games = pd.to_numeric(row.get('P2GamesWon', 0), errors='coerce')
        if pd.isna(p1_games):
            p1_games = 0
        if pd.isna(p2_games):
            p2_games = 0
        
        weight = base_weight
        
        # Check if this could be a set-deciding point
        # Set point scenarios: 5-4, 5-3, 6-5 (player ahead could win set)
        # Also check for tiebreak situations (6-6, 12-12 in 5th set, etc.)
        is_set_point_situation = False
        set_point_player = None
        
        # Tiebreak detection: games are tied at 6-6 or higher (12-12 in 5th set)
        is_games_tied_for_tiebreak = (p1_games == p2_games) and (p1_games >= 6)
        
        if p1_games >= 5 and p1_games >= p2_games + 1 and not is_games_tied_for_tiebreak:
            # P1 is ahead and could win set (not in tiebreak)
            is_set_point_situation = True
            set_point_player = 1
        elif p2_games >= 5 and p2_games >= p1_games + 1 and not is_games_tied_for_tiebreak:
            # P2 is ahead and could win set (not in tiebreak)
            is_set_point_situation = True
            set_point_player = 2
        
        # Deuce/Advantage situation: need 2 points in a row to win
        # Higher weight for advantage (closer to winning)
        if p1_score >= 3 and p2_score >= 3:
            if p1_score == p2_score:
                # Deuce (40-40): +1.5
                weight += 1.5
            else:
                # Advantage (need 1 more point to win or back to deuce): +2.0
                weight += 2.0
        
        # Calculate score differential to determine how "close" the point is
        # 0-40: differential = 3, less critical
        # 30-40: differential = 1, more critical
        # 40-AD: differential = 1, very critical
        score_diff = abs(p1_score - p2_score)
        
        # Calculate game differential (how close in games)
        game_diff = abs(p1_games - p2_games)
        
        # Get set scores for set differential
        # Use P1SetsWon and P2SetsWon which are now calculated before this function
        p1_sets = pd.to_numeric(row.get('P1SetsWon', 0), errors='coerce')
        p2_sets = pd.to_numeric(row.get('P2SetsWon', 0), errors='coerce')
        if pd.isna(p1_sets):
            p1_sets = 0
        if pd.isna(p2_sets):
            p2_sets = 0
        set_diff = abs(p1_sets - p2_sets)
        
        # MATCH POINT DETECTION: Maximum importance (7.0)
        # A match point occurs when:
        # 1. A player has won enough sets to be 1 set away from match victory
        # 2. That player is 1 point away from winning the current set/game
        
        set_no = pd.to_numeric(row.get('SetNo', 1), errors='coerce')
        if pd.isna(set_no):
            set_no = 1
        
        # Determine match format and sets needed to win
        # Best-of-5: need 3 sets, best-of-3: need 2 sets
        is_best_of_5 = (set_no >= 4) or (p1_sets + p2_sets >= 3)
        sets_to_win = 3 if is_best_of_5 else 2
        
        # Check if either player has sets_to_win - 1 sets (one set away from victory)
        p1_one_set_away = (p1_sets == sets_to_win - 1)
        p2_one_set_away = (p2_sets == sets_to_win - 1)
        
        # CASE 1: Match point in TIEBREAK
        # Use is_tiebreak (detected by numeric scores) OR is_games_tied_for_tiebreak (6-6, 12-12, etc.)
        in_tiebreak = is_tiebreak or is_games_tied_for_tiebreak
        
        if in_tiebreak and (p1_one_set_away or p2_one_set_away):
            # In tiebreak: need score >= 6 (or >= 10 for super tiebreak) with 2-point lead to win
            # For simplicity, check >= 6 with 2-point lead (works for both regular and super tiebreaks)
            if p1_one_set_away and p1_score >= 6 and p1_score >= p2_score + 1:
                return 7.0  # P1 match point in tiebreak
            elif p2_one_set_away and p2_score >= 6 and p2_score >= p1_score + 1:
                return 7.0  # P2 match point in tiebreak
        
        # CASE 2: Match point in NORMAL GAME
        # Player needs: (a) to be one set away AND (b) one point from winning game AND (c) able to win set
        if not in_tiebreak:
            # Check if winning this game would win the set
            p1_can_win_set = (p1_games >= 5 and p1_games >= p2_games + 1)  # 5-3, 5-4, 6-5, etc.
            p2_can_win_set = (p2_games >= 5 and p2_games >= p1_games + 1)
            
            # P1 match point: one set away + can win set + one point from winning game
            if p1_one_set_away and p1_can_win_set:
                # Check score: need >= 40 and ahead (server or receiver)
                if p1_score >= 3 and p1_score > p2_score:
                    return 7.0  # P1 match point (serving or receiving)
            
            # P2 match point: one set away + can win set + one point from winning game  
            if p2_one_set_away and p2_can_win_set:
                # Check score: need >= 40 and ahead (server or receiver)
                if p2_score >= 3 and p2_score > p1_score:
                    return 7.0  # P2 match point (serving or receiving)
        
        # Break point to win the set: receiver could win game AND set
        if is_set_point_situation:
            if server == 1 and set_point_player == 2 and p2_score >= 3 and p2_score > p1_score:
                # Break point for P2 to win set - weight based on closeness
                if p2_score > 3:  # Advantage
                    weight += 5.0
                elif score_diff == 1:  # 30-40
                    weight += 4.5
                elif score_diff == 2:  # 15-40
                    weight += 4.0
                else:  # 0-40
                    weight += 3.5
            elif server == 2 and set_point_player == 1 and p1_score >= 3 and p1_score > p2_score:
                # Break point for P1 to win set - weight based on closeness
                if p1_score > 3:  # Advantage
                    weight += 5.0
                elif score_diff == 1:  # 30-40
                    weight += 4.5
                elif score_diff == 2:  # 15-40
                    weight += 4.0
                else:  # 0-40
                    weight += 3.5
        
        # Regular break point (receiver could win game) - weighted by closeness
        elif server == 1 and p2_score >= 3 and p2_score > p1_score:
            if p2_score > 3:  # Advantage
                weight += 2.5
            elif score_diff == 1:  # 30-40
                weight += 2.2
            elif score_diff == 2:  # 15-40
                weight += 1.8
            else:  # 0-40
                weight += 1.5
        elif server == 2 and p1_score >= 3 and p1_score > p2_score:
            if p1_score > 3:  # Advantage
                weight += 2.5
            elif score_diff == 1:  # 30-40
                weight += 2.2
            elif score_diff == 2:  # 15-40
                weight += 1.8
            else:  # 0-40
                weight += 1.5
        
        # Set point (to win set): server could win game AND set - weighted by closeness
        if is_set_point_situation:
            if server == 1 and set_point_player == 1 and p1_score >= 3 and p1_score > p2_score:
                if p1_score > 3:  # Advantage
                    weight += 4.5
                elif score_diff == 1:  # 40-30
                    weight += 4.0
                elif score_diff == 2:  # 40-15
                    weight += 3.5
                else:  # 40-0
                    weight += 3.0
            elif server == 2 and set_point_player == 2 and p2_score >= 3 and p2_score > p1_score:
                if p2_score > 3:  # Advantage
                    weight += 4.5
                elif score_diff == 1:  # 40-30
                    weight += 4.0
                elif score_diff == 2:  # 40-15
                    weight += 3.5
                else:  # 40-0
                    weight += 3.0
        
        # Regular game point (server could win game) - weighted by closeness
        elif server == 1 and p1_score >= 3 and p1_score > p2_score:
            if p1_score > 3:  # Advantage
                weight += 1.5
            elif score_diff == 1:  # 40-30
                weight += 1.2
            elif score_diff == 2:  # 40-15
                weight += 0.9
            else:  # 40-0
                weight += 0.7
        elif server == 2 and p2_score >= 3 and p2_score > p1_score:
            if p2_score > 3:  # Advantage
                weight += 1.5
            elif score_diff == 1:  # 40-30
                weight += 1.2
            elif score_diff == 2:  # 40-15
                weight += 0.9
            else:  # 40-0
                weight += 0.7
        
        # Tiebreak: enhanced detection and weighting
        if is_tiebreak or is_games_tied_for_tiebreak:
            # Determine if this is a match-deciding tiebreak
            is_match_deciding_tiebreak = (p1_one_set_away or p2_one_set_away)
            
            if is_match_deciding_tiebreak:
                # DECISIVE TIEBREAK: Every point is critical, but reserve 7.0 for actual match points
                # Base weight for being in decisive tiebreak (very high to compensate for data scarcity: 0.87%)
                # Cap will be applied later to keep below 7.0
                weight += 5.5
                
                # Additional weight based on tiebreak score
                if is_tiebreak:
                    # Close tiebreak: both players have chance
                    score_sum = p1_score + p2_score
                    if score_sum >= 12:  # Very late in tiebreak
                        weight += 1.0
                    elif score_sum >= 8:  # Mid-to-late tiebreak
                        weight += 0.5
                    
                    # One player close to winning (but not match point yet)
                    if (p1_score >= 5 or p2_score >= 5) and abs(p1_score - p2_score) <= 2:
                        weight += 0.5
                
                # Cap at 6.5 to reserve 7.0 for match points only
                weight = min(weight, 6.5)
            else:
                # Regular tiebreak (not match-deciding)
                weight += 2.5
                
                # Tiebreak set point (not match point)
                if is_tiebreak:
                    if p1_score >= 6 and p1_score >= p2_score + 1:
                        weight += 3.5  # P1 tiebreak set point
                    elif p2_score >= 6 and p2_score >= p1_score + 1:
                        weight += 3.5  # P2 tiebreak set point
                    
                    # Close tiebreak
                    score_sum = p1_score + p2_score
                    if score_sum >= 12 and abs(p1_score - p2_score) <= 1:
                        weight += 2.0
                    elif score_sum >= 10:
                        weight += 1.0
                
                # Cap at 6.5 for non-match-deciding situations
                weight = min(weight, 6.5)
        
        # Game differential weight: closer games are more critical
        # 5-4 or 4-5 is more critical than 5-0 or 5-1
        # BUT: Skip these additional weights for tiebreaks to avoid exceeding 6.5 cap
        if not (is_tiebreak or is_games_tied_for_tiebreak):
            if p1_games >= 3 or p2_games >= 3:  # Mid to late set
                if game_diff == 0:  # Tied in games (3-3, 4-4, 5-5)
                    weight += 1.0
                elif game_diff == 1:  # One game apart (4-3, 5-4)
                    weight += 0.7
                elif game_diff == 2:  # Two games apart (5-3, 4-2)
                    weight += 0.3
        
        # Set differential weight: closer in sets means more critical match
        # Best of 5: being 2-1 or 1-1 is more critical than 2-0
        # Best of 3: being 1-1 is more critical than 1-0
        # BUT: Skip for tiebreaks to avoid exceeding 6.5 cap
        if not (is_tiebreak or is_games_tied_for_tiebreak):
            if p1_sets >= 1 or p2_sets >= 1:  # Not first set
                if set_diff == 0:  # Tied in sets (1-1, 2-2)
                    weight += 1.2
                elif set_diff == 1 and (p1_sets >= 1 and p2_sets >= 1):  # 2-1 situation
                    weight += 0.8
        
        # Critical games: score is 30-30 or higher (tight game): +0.5
        if p1_score >= 2 and p2_score >= 2 and not (p1_score >= 3 and p2_score >= 3):
            weight += 0.5
        
        # Final cap: ensure tiebreak points don't exceed 6.5 (reserve 7.0 for match points)
        if is_tiebreak or is_games_tied_for_tiebreak:
            return min(weight, 6.5)
        else:
            return min(weight, 7.0)  # Cap at 7.0 for non-tiebreak situations

    # Set progression features derived from per-set winners (0=no winner yet)
    # MUST be calculated BEFORE point_importance so match point detection works
    match_groups = df[MATCH_COL]
    if 'SetWinner' in df.columns:
        set_winners = pd.to_numeric(df['SetWinner'], errors='coerce').fillna(0.0).astype(int)
    else:
        set_winners = pd.Series(0, index=df.index)

    p1_set_wins = ((set_winners == 1).astype(int)
                   .groupby(match_groups)
                   .cumsum()
                   .astype(float))
    p2_set_wins = ((set_winners == 2).astype(int)
                   .groupby(match_groups)
                   .cumsum()
                   .astype(float))

    df['P1SetsWon'] = p1_set_wins
    df['P2SetsWon'] = p2_set_wins
    
    # Calculate point importance for sample weighting
    # NOW we have P1SetsWon and P2SetsWon available for match point detection
    df['point_importance'] = df.apply(calculate_point_importance, axis=1)
    
    # TIEBREAK-SPECIFIC FEATURES for better probability estimation
    # These help the model understand tiebreak dynamics, especially in decisive sets
    
    def extract_tiebreak_features(row):
        """Extract tiebreak-specific features."""
        p1_games = pd.to_numeric(row.get('P1GamesWon', 0), errors='coerce')
        p2_games = pd.to_numeric(row.get('P2GamesWon', 0), errors='coerce')
        if pd.isna(p1_games): p1_games = 0
        if pd.isna(p2_games): p2_games = 0
        
        # Detect tiebreak: 6-6 games OR higher (super tiebreak at 12-12, etc.)
        is_tiebreak = (p1_games == p2_games) and (p1_games >= 6)
        
        # Get scores
        p1_score_raw = row.get('P1Score', 0)
        p2_score_raw = row.get('P2Score', 0)
        
        # Parse tiebreak scores (numeric)
        tiebreak_p1_score = 0
        tiebreak_p2_score = 0
        if is_tiebreak:
            try:
                p1_str = str(p1_score_raw).strip()
                p2_str = str(p2_score_raw).strip()
                if p1_str not in {'0', '15', '30', '40', 'AD', 'A'}:
                    tiebreak_p1_score = int(p1_str)
                    tiebreak_p2_score = int(p2_str)
            except (ValueError, TypeError):
                pass
        
        # Check if decisive tiebreak (2-2 sets in best-of-5, or 1-1 in best-of-3)
        p1_sets = pd.to_numeric(row.get('P1SetsWon', 0), errors='coerce')
        p2_sets = pd.to_numeric(row.get('P2SetsWon', 0), errors='coerce')
        if pd.isna(p1_sets): p1_sets = 0
        if pd.isna(p2_sets): p2_sets = 0
        
        is_decisive_tiebreak = is_tiebreak and (p1_sets == p2_sets) and (p1_sets >= 1)
        
        # Score differential in tiebreak (normalized to [-1, 1])
        if is_tiebreak:
            score_diff = tiebreak_p1_score - tiebreak_p2_score
            # Normalize: ±7 points = ±1.0
            tiebreak_score_diff = np.clip(score_diff / 7.0, -1.0, 1.0)
            
            # Proximity to winning tiebreak (need 7+ with 2-point lead in super tiebreak at Wimbledon 2019)
            # For Wimbledon 2019 final set tiebreak: first to 7 points wins (no 2-point margin needed)
            # Value from 0 (not close) to 1 (at match/set point)
            
            # Calculate how many points each player needs to win
            # At Wimbledon 2019: first to 7 wins (simplified rule for 12-12 tiebreak)
            p1_needs = max(0, 7 - tiebreak_p1_score)
            p2_needs = max(0, 7 - tiebreak_p2_score)
            
            # Exponential proximity: use factor 1.0 for steeper curve
            # This makes being close to winning much more impactful
            p1_proximity = np.exp(-p1_needs * 1.0)  # ranges from exp(-6)≈0.0025 to exp(0)=1.0
            p2_proximity = np.exp(-p2_needs * 1.0)
            tiebreak_win_proximity = p1_proximity - p2_proximity  # [-1, 1]
            
            # Score differential (normalized to [-1, 1])
            tiebreak_score_diff = score_diff / 7.0
            
            # Late stage indicator
            total_tb_points = tiebreak_p1_score + tiebreak_p2_score
            is_tiebreak_late_stage = float(total_tb_points >= 10)
        else:
            tiebreak_score_diff = 0.0
            tiebreak_win_proximity = 0.0
            is_tiebreak_late_stage = 0.0
        
        return pd.Series({
            'is_tiebreak': float(is_tiebreak),
            'is_decisive_tiebreak': float(is_decisive_tiebreak),
            'tiebreak_score_diff': tiebreak_score_diff,
            'tiebreak_win_proximity': tiebreak_win_proximity,
            'is_tiebreak_late_stage': is_tiebreak_late_stage,
        })
    
    tiebreak_features = df.apply(extract_tiebreak_features, axis=1)
    df['is_tiebreak'] = tiebreak_features['is_tiebreak']
    df['is_decisive_tiebreak'] = tiebreak_features['is_decisive_tiebreak']
    df['tiebreak_score_diff'] = tiebreak_features['tiebreak_score_diff']
    df['tiebreak_win_proximity'] = tiebreak_features['tiebreak_win_proximity']
    df['is_tiebreak_late_stage'] = tiebreak_features['is_tiebreak_late_stage']

    # Calculate point type indicators for PLOTTING ONLY (not used in model training)
    # These provide clear visual markers on plots independent of point_importance
    def extract_point_types(row):
        """Extract binary indicators for break point, set point, match point."""
        # Helper function to convert score to numeric
        def score_to_numeric(score_raw):
            score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'A': 4}
            score_str = str(score_raw).strip().upper()
            return score_map.get(score_str, 0)
        
        # Get basic information
        server = row.get('PointServer', 1)
        p1_score_raw = row.get('P1Score', 0)
        p2_score_raw = row.get('P2Score', 0)
        p1_games = pd.to_numeric(row.get('P1GamesWon', 0), errors='coerce')
        p2_games = pd.to_numeric(row.get('P2GamesWon', 0), errors='coerce')
        p1_sets = pd.to_numeric(row.get('P1SetsWon', 0), errors='coerce')
        p2_sets = pd.to_numeric(row.get('P2SetsWon', 0), errors='coerce')
        
        if pd.isna(p1_games): p1_games = 0
        if pd.isna(p2_games): p2_games = 0
        if pd.isna(p1_sets): p1_sets = 0
        if pd.isna(p2_sets): p2_sets = 0
        
        # Detect tiebreak
        is_games_tied = (p1_games == p2_games) and (p1_games >= 6)
        
        # Parse scores
        try:
            p1_str = str(p1_score_raw).strip()
            p2_str = str(p2_score_raw).strip()
            tennis_scores = {'0', '15', '30', '40', 'AD', 'A'}
            
            # Check if tiebreak (numeric scores)
            if p1_str not in tennis_scores and p2_str not in tennis_scores:
                p1_score = int(p1_str)
                p2_score = int(p2_str)
                is_tb = True
            else:
                p1_score = score_to_numeric(p1_score_raw)
                p2_score = score_to_numeric(p2_score_raw)
                is_tb = False
        except (ValueError, TypeError):
            p1_score = score_to_numeric(p1_score_raw)
            p2_score = score_to_numeric(p2_score_raw)
            is_tb = False
        
        # Determine match format
        set_no = pd.to_numeric(row.get('SetNo', 1), errors='coerce')
        if pd.isna(set_no): set_no = 1
        is_best_of_5 = (set_no >= 4) or (p1_sets + p2_sets >= 3)
        sets_to_win = 3 if is_best_of_5 else 2
        
        # Initialize
        is_break_point = 0.0
        is_set_point = 0.0
        is_match_point = 0.0
        
        # BREAK POINT: Receiver can win game
        # Normal game: receiver at 40 or advantage, ahead in score
        if not is_tb and not is_games_tied:
            if server == 1:  # P2 is receiving
                if p2_score >= 3 and p2_score > p1_score:
                    is_break_point = 1.0
            else:  # P1 is receiving
                if p1_score >= 3 and p1_score > p2_score:
                    is_break_point = 1.0
        
        # SET POINT: Player can win set
        # Check if player is close to winning set
        p1_can_win_set = (p1_games >= 5 and p1_games >= p2_games + 1)
        p2_can_win_set = (p2_games >= 5 and p2_games >= p1_games + 1)
        
        if is_tb or is_games_tied:
            # Tiebreak set point: score >= 6 with 1+ lead
            if p1_score >= 6 and p1_score >= p2_score + 1:
                is_set_point = 1.0
            elif p2_score >= 6 and p2_score >= p1_score + 1:
                is_set_point = 1.0
        else:
            # Normal game set point
            if p1_can_win_set and p1_score >= 3 and p1_score > p2_score:
                is_set_point = 1.0
            elif p2_can_win_set and p2_score >= 3 and p2_score > p1_score:
                is_set_point = 1.0
        
        # MATCH POINT: Player can win match
        p1_one_set_away = (p1_sets == sets_to_win - 1)
        p2_one_set_away = (p2_sets == sets_to_win - 1)
        
        if is_tb or is_games_tied:
            # Tiebreak match point
            if p1_one_set_away and p1_score >= 6 and p1_score >= p2_score + 1:
                is_match_point = 1.0
            elif p2_one_set_away and p2_score >= 6 and p2_score >= p1_score + 1:
                is_match_point = 1.0
        else:
            # Normal game match point
            if p1_one_set_away and p1_can_win_set and p1_score >= 3 and p1_score > p2_score:
                is_match_point = 1.0
            elif p2_one_set_away and p2_can_win_set and p2_score >= 3 and p2_score > p1_score:
                is_match_point = 1.0
        
        return pd.Series({
            'is_break_point': is_break_point,
            'is_set_point': is_set_point,
            'is_match_point': is_match_point
        })
    
    point_types = df.apply(extract_point_types, axis=1)
    df['is_break_point'] = point_types['is_break_point']
    df['is_set_point'] = point_types['is_set_point']
    df['is_match_point'] = point_types['is_match_point']

    # Momentum difference - normalize per MATCH to preserve match-level context
    # Z-score normalization balances momentum across entire match duration
    # Reduced in decisive tiebreaks where current score matters more than past momentum
    if 'P1Momentum' in df.columns and 'P2Momentum' in df.columns:
        p1_mom = pd.to_numeric(df['P1Momentum'], errors='coerce').fillna(0)
        p2_mom = pd.to_numeric(df['P2Momentum'], errors='coerce').fillna(0)
        raw_diff = p1_mom - p2_mom
        
        # Z-score normalization per match
        match_groups = df.groupby(MATCH_COL, sort=False).ngroup()
        mean_per_match = raw_diff.groupby(match_groups).transform('mean')
        std_per_match = raw_diff.groupby(match_groups).transform('std').replace(0, 1)
        df['Momentum_Diff'] = (raw_diff - mean_per_match) / std_per_match
        df['Momentum_Diff'] = df['Momentum_Diff'].fillna(0.0).clip(-3.0, 3.0)
        
        # Reduce momentum weight in decisive tiebreaks (score matters more than history)
        if 'is_decisive_tiebreak' in df.columns:
            df.loc[df['is_decisive_tiebreak'] == 1.0, 'Momentum_Diff'] *= 0.3
    else:
        df['Momentum_Diff'] = 0.0

    # Score difference (convert tennis scores to numeric first)
    if 'P1Score' in df.columns and 'P2Score' in df.columns:
        p1_score_numeric = df['P1Score'].apply(score_to_numeric)
        p2_score_numeric = df['P2Score'].apply(score_to_numeric)
        df['Score_Diff'] = p1_score_numeric - p2_score_numeric
    else:
        df['Score_Diff'] = 0.0

    # Clip score differential to keep model from overfitting absolute values
    df['Score_Diff'] = pd.to_numeric(df['Score_Diff'], errors='coerce').fillna(0.0)
    df['Score_Diff'] = df['Score_Diff'].clip(lower=-2.0, upper=2.0)

    # Game won difference
    if 'P1GamesWon' in df.columns and 'P2GamesWon' in df.columns:
        df['Game_Diff'] = pd.to_numeric(df['P1GamesWon'], errors='coerce').fillna(0) - pd.to_numeric(df['P2GamesWon'], errors='coerce').fillna(0)
    else:
        df['Game_Diff'] = 0.0

    df['Game_Diff'] = pd.to_numeric(df['Game_Diff'], errors='coerce').fillna(0.0).clip(lower=-2.0, upper=2.0)
    
    # Current set games difference (captures in-set dominance independent of past sets)
    # Reduced amplification from 2.5× to 1.5× to limit overreaction to current set
    # This balances current set performance with overall match context
    if 'P1GamesWon' in df.columns and 'P2GamesWon' in df.columns:
        p1_games = pd.to_numeric(df['P1GamesWon'], errors='coerce').fillna(0)
        p2_games = pd.to_numeric(df['P2GamesWon'], errors='coerce').fillna(0)
        df['CurrentSetGamesDiff'] = ((p1_games - p2_games) * 1.5).clip(lower=-10.0, upper=10.0)
    else:
        df['CurrentSetGamesDiff'] = 0.0

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
    
    # Set progression features derived from per-set winners (0=no winner yet)
    # IMPORTANT: Calculate these BEFORE point_importance so match point detection works
    match_groups = df[MATCH_COL]
    if 'SetWinner' in df.columns:
        set_winners = pd.to_numeric(df['SetWinner'], errors='coerce').fillna(0.0).astype(int)
    else:
        set_winners = pd.Series(0, index=df.index)

    p1_set_wins = ((set_winners == 1).astype(int)
                   .groupby(match_groups)
                   .cumsum()
                   .astype(float))
    p2_set_wins = ((set_winners == 2).astype(int)
                   .groupby(match_groups)
                   .cumsum()
                   .astype(float))

    df['P1SetsWon'] = p1_set_wins
    df['P2SetsWon'] = p2_set_wins
    
    # Calculate point importance for sample weighting
    # NOW we have P1SetsWon and P2SetsWon available for match point detection
    df['point_importance'] = df.apply(calculate_point_importance, axis=1)

    df['SetsWonDiff_raw'] = df['P1SetsWon'] - df['P2SetsWon']
    
    # SetsWonDiff with NON-LINEAR step function:
    # Goal: When sets tied (0-0, 1-1, 2-2) → P~0.50, rely on current set
    #       +1 set advantage → P~0.65 (+0.15 boost)
    #       +2 sets → P~0.80 (+0.30 boost)
    #       +3 sets → P~0.95 (+0.45 boost, match almost won)
    # 
    # Implementation: Use step multipliers that grow non-linearly
    set_progress = pd.to_numeric(df['SetNo'], errors='coerce').fillna(0.5)
    abs_set_diff = df['SetsWonDiff_raw'].abs()
    
    # Step multipliers: 0 sets=0.0, 1 set=1.0, 2 sets=2.0, 3 sets=3.0
    # Then scale: 1 set → ×0.8, 2 sets → ×1.6, 3 sets → ×2.4
    step_multiplier = abs_set_diff * 0.8  # Linear in number of sets
    
    # Apply sign and scale by match progress
    df['SetsWonDiff'] = (df['SetsWonDiff_raw'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) 
                         * step_multiplier * set_progress)
    
    # Binary indicator: does P1 have set advantage? (+1 for advantage, 0 for tied, -1 for disadvantage)
    # SCALED BY 4.0 to make it strongly influential over cumulative features
    # When sets are tied, current game/score situation should still matter via other features
    df['SetsWonAdvantage'] = df['SetsWonDiff_raw'].apply(lambda x: 4.0 if x > 0 else (-4.0 if x < 0 else 0.0))

    def compute_set_win_prior(row) -> float:
        """
        Heuristic prior for P1 match win prob driven by set state.
        Constraints:
          - Tied in sets → keep close to 0.50 (±0.15 max, before ranking boost)
          - +1 set lead  → ~0.65 baseline
          - +2 set lead  → ~0.80 baseline
        Adjust with current-set games edge and (optionally) ranking gap if present.
        """
        set_diff = row.get('SetsWonDiff_raw', 0.0)
        current_edge = row.get('CurrentSetGamesDiff', 0.0)
        sets_played = row.get('P1SetsWon', 0.0) + row.get('P2SetsWon', 0.0)

        # Baseline anchored on set difference (symmetrical for P2)
        if set_diff >= 2:
            base = 0.80
        elif set_diff == 1:
            base = 0.65
        elif set_diff == 0:
            base = 0.50
        elif set_diff == -1:
            base = 0.35
        else:  # set_diff <= -2
            base = 0.20

        # Current set momentum nudges the prior; cap shift to keep ties near 50-50
        # tanh keeps adjustment bounded for blowouts
        edge = np.tanh(float(current_edge) / 4.0)
        if set_diff == 0:
            base += edge * 0.15  # at most ±0.15 when tied
        elif abs(set_diff) == 1:
            base += edge * 0.10  # smaller influence when already up/down a set
        else:
            base += edge * 0.05  # tiny tweak when two sets apart

        # Optional ranking prior: allow stronger player to start above 0.65 in set 1
        rank_boost = 0.0
        if 'P1Rank' in row and 'P2Rank' in row:
            try:
                p1_rank = float(row['P1Rank'])
                p2_rank = float(row['P2Rank'])
                if not np.isnan(p1_rank) and not np.isnan(p2_rank):
                    # Positive gap → P1 higher ranked. Bounded to ±0.12.
                    rank_gap = p2_rank - p1_rank
                    # Damp rank effect after first set (sets_played>0)
                    damp = 0.5 if sets_played >= 1 else 1.0
                    rank_boost = np.tanh(rank_gap / 400.0) * 0.12 * damp
            except Exception:
                rank_boost = 0.0

        prior = base + rank_boost
        
        # DAMPEN prior all'inizio del match: quando ci sono pochi game giocati,
        # il prior dovrebbe convergere verso 0.5 (neutrale)
        # Questo evita che la rete ignori Game_Diff all'inizio
        total_games = row.get('P1GamesWon', 0.0) + row.get('P2GamesWon', 0.0)
        if total_games <= 6:  # primi 6 game del match
            # Blend con 0.5: più game giocati, meno blending
            blend_to_neutral = (6 - total_games) / 6.0  # 1.0 a inizio, 0.0 dopo 6 game
            prior = prior * (1 - blend_to_neutral) + 0.5 * blend_to_neutral
        
        return float(np.clip(prior, 0.05, 0.95))

    df['SetWinProbPrior'] = df.apply(compute_set_win_prior, axis=1)
    df['SetWinProbEdge'] = (df['SetWinProbPrior'] - 0.5) * 2.0  # centered, in [-1,1]
    df['SetWinProbLogit'] = np.log(df['SetWinProbPrior'].clip(1e-6, 1 - 1e-6) / (1.0 - df['SetWinProbPrior'].clip(1e-6, 1 - 1e-6)))

    # When sets are tied, damp serve/return probabilities toward 0.5 to reduce volatility.
    # BUT: preserve current game situation (40-0, break points, etc.)
    # Only apply dampening when BOTH sets tied AND current game is not critical
    if all(col in df.columns for col in ["P_srv_win_long", "P_srv_lose_long", "P_srv_win_short", "P_srv_lose_short"]):
        tie_mask = df['SetsWonDiff_raw'] == 0
        
        # Identify critical game situations where dampening should be reduced
        # (match points, break points for set, 40-0 situations, etc.)
        critical_game = df.get('point_importance', pd.Series(1.0, index=df.index)) > 3.0
        
        if tie_mask.any():
            # Base dampening: 50% when sets tied
            # But reduce to 20% in critical game situations
            damp = np.where(critical_game & tie_mask, 0.20, 0.50)
            
            for col in ["P_srv_win_long", "P_srv_lose_long", "P_srv_win_short", "P_srv_lose_short"]:
                df.loc[tie_mask, col] = df.loc[tie_mask, col] * (1.0 - damp[tie_mask]) + 0.5 * damp[tie_mask]
    
    # Indicator for tied decisive set (2-2 in best-of-5, or 1-1 in best-of-3 in final set)
    # Criteria: (1) sets are tied AND (2) at least 2 sets have been completed (P1+P2 >= 2)
    # Use SetNo_original to check if we're late in the match (SetNo >= 4 means at least set 4)
    sets_completed = df['P1SetsWon'] + df['P2SetsWon']
    in_late_set = pd.to_numeric(df.get('SetNo_original', df['SetNo']), errors='coerce').fillna(0) >= 4
    df['is_decider_tied'] = ((df['SetsWonDiff_raw'] == 0) & (sets_completed >= 2) & in_late_set).astype(float)
    
    # Dampen cumulative features (momentum, long serve stats) in tied decisive situations
    # This reduces historical bias when the match is truly 50-50
    tied_mask = df['is_decider_tied'] == 1.0
    dampen_factor = 0.02  # reduce momentum/long-window influence to 2% in ties (very aggressive, from 5%)
    
    if 'momentum' in df.columns:
        df.loc[tied_mask, 'momentum'] = df.loc[tied_mask, 'momentum'] * dampen_factor
    
    for col in ['P_srv_win_long', 'P_srv_lose_long']:
        if col in df.columns:
            # Blend long-window with short-window when tied (rely more on recent form)
            short_col = col.replace('long', 'short')
            if short_col in df.columns:
                df.loc[tied_mask, col] = (
                    df.loc[tied_mask, col] * dampen_factor + 
                    df.loc[tied_mask, short_col] * (1 - dampen_factor)
                )
    
    # DECISIVE TIEBREAK DAMPENING: removed - handled by neural network features

    # Early match indicator: primi 2-3 game quando le statistiche cumulative non sono affidabili
    # In questa fase, Game_Diff e break counts sono più informativi del momentum/storia
    df['is_early_match'] = 0.0
    total_games_played = df['P1GamesWon'] + df['P2GamesWon'] if 'P1GamesWon' in df.columns and 'P2GamesWon' in df.columns else 0
    early_mask = (df['SetNo'] == 1) & (total_games_played <= 4)
    df.loc[early_mask, 'is_early_match'] = 1.0
    
    # Set situation indicators - binari e CHIARI per la rete
    # Non usiamo SetsWonDiff con segno perché la rete si confonde
    # Usiamo indicatori separati: chi è avanti? Di quanto?
    if 'P1SetsWon' in df.columns and 'P2SetsWon' in df.columns:
        p1_sets = df['P1SetsWon']
        p2_sets = df['P2SetsWon']
        
        # Indicatori binari: 1.0 se vera, 0.0 altrimenti
        df['is_p1_ahead_sets'] = (p1_sets > p2_sets).astype(float)
        df['is_p2_ahead_sets'] = (p2_sets > p1_sets).astype(float)
        df['is_sets_tied'] = (p1_sets == p2_sets).astype(float)
        
        # Magnitudine del vantaggio (sempre positiva): 0, 1, o 2
        df['sets_magnitude'] = np.abs(p1_sets - p2_sets).astype(float)
    else:
        df['is_p1_ahead_sets'] = 0.0
        df['is_p2_ahead_sets'] = 0.0
        df['is_sets_tied'] = 1.0  # Default: pari
        df['sets_magnitude'] = 0.0

    # Add MatchFinished feature: 1 if match is over after this point, 0 otherwise
    # This helps the model learn that when the match ends, probability should be 0 or 1
    df['MatchFinished'] = 0.0
    
    # Get the last point index for each match
    last_point_idx = df.groupby(MATCH_COL).tail(1).index
    df.loc[last_point_idx, 'MatchFinished'] = 1.0
    
    # Add DistanceToMatchEnd: non-linear feature that increases as match nears completion
    # This helps the model understand urgency and criticality of late-match points
    # Formula: based on sets won and games won, with exponential scaling
    
    # Calculate how close each player is to winning the match
    # For best-of-5: need 3 sets (or 2 sets if leading decisively)
    # For best-of-3: need 2 sets
    
    # Determine match format (best-of-3 or best-of-5) based on SetNo
    if 'SetNo_full_max' in df.columns:
        max_set = df['SetNo_full_max']
    else:
        max_set = df.groupby(MATCH_COL)['SetNo_original'].transform('max') if 'SetNo_original' in df.columns else 3
    is_best_of_5 = max_set >= 4
    
    # Calculate sets needed to win
    sets_to_win = 3.0  # default best-of-5
    if not isinstance(is_best_of_5, (int, float, bool)):
        # It's a series, apply conditional
        sets_to_win = is_best_of_5.apply(lambda x: 3.0 if x else 2.0)
    elif not is_best_of_5:
        sets_to_win = 2.0
    
    # Calculate proximity to match end for each player
    # Higher value = closer to winning
    p1_sets = df['P1SetsWon'] if 'P1SetsWon' in df.columns else 0
    p2_sets = df['P2SetsWon'] if 'P2SetsWon' in df.columns else 0
    p1_games = pd.to_numeric(df['P1GamesWon'], errors='coerce').fillna(0)
    p2_games = pd.to_numeric(df['P2GamesWon'], errors='coerce').fillna(0)
    
    # Normalize sets won to [0, 1] based on how many needed
    if isinstance(sets_to_win, (int, float)):
        p1_set_progress = p1_sets / sets_to_win
        p2_set_progress = p2_sets / sets_to_win
    else:
        p1_set_progress = p1_sets / sets_to_win
        p2_set_progress = p2_sets / sets_to_win
    
    # Normalize games won to [0, 1] (need 6 games to win set, or 7 in tiebreak)
    p1_game_progress = np.minimum(p1_games / 6.0, 1.0)
    p2_game_progress = np.minimum(p2_games / 6.0, 1.0)
    
    # Combined progress: weighted sum of set and game progress
    # Sets are much more important than games
    p1_progress = 0.8 * p1_set_progress + 0.2 * p1_game_progress
    p2_progress = 0.8 * p2_set_progress + 0.2 * p2_game_progress
    
    # Maximum progress (whoever is closer to winning)
    max_progress = np.maximum(p1_progress, p2_progress)
    
    # Apply non-linear transformation: exponential increase near the end
    # Use x^3 to make it increase sharply as match nears completion
    # Scale to [0, 10] range for better model interpretation
    df['DistanceToMatchEnd'] = np.power(max_progress, 3.0) * 10.0
    
    # Clip to reasonable range
    df['DistanceToMatchEnd'] = df['DistanceToMatchEnd'].clip(0.0, 10.0)

    # Normalize coarse progress indicators within each match to reduce raw-score dominance
    # Save original SetNo before normalization (needed for filtering/analysis)
    if 'SetNo' in df.columns:
        df['SetNo_original'] = df['SetNo'].copy()
    
    for col in ['GameNo', 'PointNumber', 'SetNo']:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            override_max = df.get(f"{col}_full_max")
            if override_max is not None:
                max_per_match = pd.to_numeric(override_max, errors='coerce').fillna(0.0).replace(0, 1)
            else:
                max_per_match = numeric.groupby(match_groups).transform('max').replace(0, 1)
            df[col] = (numeric / max_per_match).clip(0.0, 1.0)

    return df


def add_rolling_serve_return_features(
    df: pd.DataFrame,
    long_window: int,
    short_window: int,
    weight_serve_return: bool = False,
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
      
    Args:
        weight_serve_return: If True, weight serve wins less (0.6x) and return wins more (1.5x)
                            to reflect that holding serve is expected while breaking is significant.
                            Use this for point-by-point reconstruction to reduce over-reaction to holds.
    """
    df = df.copy()

    # Ensure server / winner columns are numeric and drop rows with missing values
    df[SERVER_COL] = pd.to_numeric(df[SERVER_COL], errors='coerce')
    df[POINT_WINNER_COL] = pd.to_numeric(df[POINT_WINNER_COL], errors='coerce')
    df = df.dropna(subset=[SERVER_COL, POINT_WINNER_COL])
    df[SERVER_COL] = df[SERVER_COL].astype(int)
    df[POINT_WINNER_COL] = df[POINT_WINNER_COL].astype(int)

    # Determine contextual weights based on match situation
    if weight_serve_return and 'SetNo' in df.columns:
        set_no = pd.to_numeric(df['SetNo'], errors='coerce').fillna(1).astype(int)
        # Context-dependent weights - MORE AGGRESSIVE to counter normalization:
        # Sets 1-3: hold=0.4x (much less), break=2.0x (much more)
        # Set 4: hold=0.6x, break=1.7x
        # Set 5: hold=0.8x, break=1.3x (both critical but still differentiated)
        serve_weight = np.where(set_no >= 5, 0.8,
                               np.where(set_no >= 4, 0.6, 0.4))
        return_weight = np.where(set_no >= 5, 1.3,
                                np.where(set_no >= 4, 1.7, 2.0))
    else:
        serve_weight = 1.0
        return_weight = 1.0

    # Base indicator columns: per side, did they serve+win or receive+win?
    # Apply contextual weights if enabled
    for side in (1, 2):
        srv_win_col = f"s{side}_srv_win"
        rcv_win_col = f"s{side}_rcv_win"

        # Weighted indicators: serve wins count less, return wins count more
        if weight_serve_return:
            df[srv_win_col] = np.where(
                (df[SERVER_COL] == side) & (df[POINT_WINNER_COL] == side),
                serve_weight,  # Contextual weight for serve wins
                0.0
            )
            df[rcv_win_col] = np.where(
                (df[SERVER_COL] != side) & (df[POINT_WINNER_COL] == side),
                return_weight,  # Contextual weight for return wins
                0.0
            )
        else:
            # Standard binary indicators
            df[srv_win_col] = (
                (df[SERVER_COL] == side) &
                (df[POINT_WINNER_COL] == side)
            ).astype(int)

            df[rcv_win_col] = (
                (df[SERVER_COL] != side) &
                (df[POINT_WINNER_COL] == side)
            ).astype(int)

        # Rolling sums for both long and short windows
        # Accumulate across entire match to capture player form evolution
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


def add_break_hold_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiungi features che distinguono break (vincere in risposta) da hold (vincere al servizio).
    
    Features aggiunte:
    - P1BreaksWon: numero di break vinti da P1 (game vinti quando P2 serviva)
    - P2BreaksWon: numero di break vinti da P2 (game vinti quando P1 serviva)
    - BreaksDiff: P1BreaksWon - P2BreaksWon (più importante di Game_Diff)
    - P1HoldsWon: numero di hold di P1 (game vinti quando P1 serviva)
    - P2HoldsWon: numero di hold di P2
    - HoldRate_P1: percentuale di servizi tenuti da P1 (resilienza)
    - HoldRate_P2: percentuale di servizi tenuti da P2
    - BreakRate_P1: percentuale di break ottenuti da P1 (aggressività)
    - BreakRate_P2: percentuale di break ottenuti da P2
    - weighted_game_diff: Game_Diff dove i break contano 2x
    """
    df = df.copy()
    
    # Identifichiamo chi ha vinto ogni game completato
    # Un game è completato quando GameNo cambia
    df['game_completed'] = (df['GameNo'] != df['GameNo'].shift(1))
    
    # Il vincitore del game è chi ha vinto l'ultimo punto del game precedente
    # Otteniamo questo guardando GAME_WINNER_COL della riga PRECEDENTE
    df['prev_game_winner'] = df[GAME_WINNER_COL].shift(1)
    df['prev_server'] = df[SERVER_COL].shift(1)
    
    # Un break avviene quando il winner del game NON era il server
    # Hold avviene quando il winner del game ERA il server
    df['was_break'] = (df['game_completed']) & (df['prev_game_winner'] != df['prev_server'])
    df['was_hold'] = (df['game_completed']) & (df['prev_game_winner'] == df['prev_server'])
    
    # Conta break e hold per match, cumulativamente
    # P1 break = P1 vince quando P2 serviva (prev_game_winner=1, prev_server=2)
    # P2 break = P2 vince quando P1 serviva (prev_game_winner=2, prev_server=1)
    df['p1_break_this_game'] = (df['was_break']) & (df['prev_game_winner'] == 1)
    df['p2_break_this_game'] = (df['was_break']) & (df['prev_game_winner'] == 2)
    df['p1_hold_this_game'] = (df['was_hold']) & (df['prev_game_winner'] == 1)
    df['p2_hold_this_game'] = (df['was_hold']) & (df['prev_game_winner'] == 2)
    
    # Cumulative sum per match
    df['P1BreaksWon'] = df.groupby(MATCH_COL)['p1_break_this_game'].cumsum().astype(float)
    df['P2BreaksWon'] = df.groupby(MATCH_COL)['p2_break_this_game'].cumsum().astype(float)
    df['P1HoldsWon'] = df.groupby(MATCH_COL)['p1_hold_this_game'].cumsum().astype(float)
    df['P2HoldsWon'] = df.groupby(MATCH_COL)['p2_hold_this_game'].cumsum().astype(float)
    
    # Break difference (più importante di game diff)
    df['BreaksDiff'] = df['P1BreaksWon'] - df['P2BreaksWon']
    
    # Serve opportunities: quante volte ogni giocatore ha servito
    # Approssimato come numero di game completati diviso 2 (si alternano)
    df['total_games_completed'] = df.groupby(MATCH_COL)['game_completed'].cumsum().astype(float)
    df['P1_serve_opportunities'] = (df['total_games_completed'] / 2.0).apply(np.floor)
    df['P2_serve_opportunities'] = (df['total_games_completed'] / 2.0).apply(np.ceil)
    
    # Hold rates (resilienza): % di servizi tenuti
    # +1 smoothing per evitare divisione per zero
    df['HoldRate_P1'] = (df['P1HoldsWon'] + 1.0) / (df['P1_serve_opportunities'] + 2.0)
    df['HoldRate_P2'] = (df['P2HoldsWon'] + 1.0) / (df['P2_serve_opportunities'] + 2.0)
    
    # Break rates (aggressività): % di break ottenuti quando avversario serviva
    df['BreakRate_P1'] = (df['P1BreaksWon'] + 1.0) / (df['P2_serve_opportunities'] + 2.0)
    df['BreakRate_P2'] = (df['P2BreaksWon'] + 1.0) / (df['P1_serve_opportunities'] + 2.0)
    
    # Weighted game diff: i break contano 2x rispetto agli hold
    # Questo dà più peso ai momenti cruciali
    if 'Game_Diff' in df.columns:
        # Game_Diff normale + extra weight per break
        df['weighted_game_diff'] = df['Game_Diff'] + df['BreaksDiff'] * 0.5
    else:
        df['weighted_game_diff'] = df['BreaksDiff']
    
    # Cleanup temporary columns
    df = df.drop(columns=['game_completed', 'prev_game_winner', 'prev_server', 
                          'was_break', 'was_hold', 'p1_break_this_game', 'p2_break_this_game',
                          'p1_hold_this_game', 'p2_hold_this_game', 'total_games_completed',
                          'P1_serve_opportunities', 'P2_serve_opportunities'])
    
    return df


def add_leverage_and_momentum(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Compute leverage and weighted momentum with the given alpha.
    
    Momentum is calculated using point importance as weights, so that
    critical points (break points, set points) contribute more to momentum.
    
    NOTE: This must be called AFTER add_additional_features() which computes point_importance.
    """
    df = df.copy()

    server_wins = (df[POINT_WINNER_COL] == df[SERVER_COL])

    leverage_raw = df["P_srv_win_long"] - df["P_srv_lose_long"]
    leverage_raw = leverage_raw.clip(lower=0.0)

    df["leverage"] = np.where(server_wins, leverage_raw, 0.0)

    # Further damp leverage when sets are tied to avoid extreme swings.
    # BUT: preserve leverage in critical game situations (break/set/match points)
    if "SetsWonDiff_raw" in df.columns:
        tie_mask = df["SetsWonDiff_raw"] == 0
        if tie_mask.any():
            critical_game = df.get('point_importance', pd.Series(1.0, index=df.index)) > 3.0
            # 35% leverage when tied + normal, 70% when tied + critical
            leverage_factor = np.where(critical_game, 0.70, 0.35)
            df.loc[tie_mask, "leverage"] = df.loc[tie_mask, "leverage"] * leverage_factor[tie_mask]

    # Check if point_importance exists (it should be computed before this)
    if "point_importance" not in df.columns:
        # Fallback: use uniform weights
        df["point_importance"] = 1.0
    
    # Weight leverage by point importance for momentum calculation
    # Critical points have more impact on momentum
    df["weighted_leverage"] = df["leverage"] * df["point_importance"]
    
    # Compute weighted momentum using EWMA
    # Accumulate across entire match to capture psychological momentum
    # pandas.Series.ewm enforces 0 < alpha <= 1, while our momentum helper
    # supports alpha > 1 (for faster decay). Use the custom implementation
    # when alpha is outside the pandas bounds.
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    # TEMPORARY FIX: Always use pandas ewm (alpha <= 1.0) to avoid blocking
    # The custom implementation has a bug that causes infinite loop
    if alpha > 1.0:
        print(f"[WARNING] alpha={alpha} > 1.0, clamping to 1.0 to avoid blocking")
        alpha = 1.0
    
    df["momentum"] = (
        df.groupby(MATCH_COL)["weighted_leverage"]
          .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    )
    
    # Clean up temporary column
    df = df.drop(columns=["weighted_leverage"])

    return df


def build_dataset(df: pd.DataFrame):
    """
        Build feature matrix X and target y for training.

        Features (22 total):
            - P_srv_win_long
            - P_srv_lose_long
            - P_srv_win_short   (real-time window)
            - P_srv_lose_short  (real-time window)
            - PointServer (Srv)
            - momentum (EWMA of leverage)
            - Momentum_Diff: P1Momentum - P2Momentum (rolling z-score, window=50)
            - Score_Diff: P1Score - P2Score
            - Game_Diff: P1GamesWon - P2GamesWon
            - CurrentSetGamesDiff: games difference in current set (in-set performance)
            - SrvScr: cumulative points won when p1 served in game
            - RcvScr: cumulative points won when p1 received in game
            - SetNo (St)
            - GameNo (Gm)
            - PointNumber (Pt)
            - point_importance: critical point indicator (1.0-7.0)
            - SetsWonDiff: set difference scaled by match progress (non-linear weighting)
            - SetsWonAdvantage: +1 when P1 leads in sets, -1 when behind, 0 when tied
            - SetWinProbPrior: calibrated prior for P1 match win (0.05-0.95)
            - SetWinProbEdge: centered prior in [-1,1] to stabilize training
            - SetWinProbLogit: log-odds of the calibrated prior
            - is_decider_tied: 1.0 when sets are level in the final set, 0.0 otherwise

        Target:
            - p1_wins_match
        """
    df = df.copy()
    feature_cols = MATCH_FEATURE_COLUMNS.copy()
    
    # Filtra solo le colonne che esistono nel DataFrame
    # Le break features sono opzionali (solo per NN, non per BDT)
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_feature_cols) < len(feature_cols):
        missing_cols = set(feature_cols) - set(available_feature_cols)
        # Solo avvisa per colonne non-break
        non_break_missing = [c for c in missing_cols if not c.startswith('break') and not c.endswith('_breaks')]
        if non_break_missing:
            print(f"[build_dataset] Warning: Missing columns {non_break_missing}")
    
    feature_cols = available_feature_cols

    # Determine sets required to win for each match (best-of-3 vs best-of-5)
    # Defaults to best-of-5 (3 sets) if original set count is unavailable.
    sets_to_win_series = pd.Series(3.0, index=df.index)
    if 'SetNo_full_max' in df.columns:
        max_set_per_match = df['SetNo_full_max']
        sets_to_win_series = pd.Series(np.where(max_set_per_match >= 4, 3.0, 2.0), index=df.index)
    elif 'SetNo_original' in df.columns:
        max_set_per_match = df.groupby(MATCH_COL)['SetNo_original'].transform('max')
        sets_to_win_series = pd.Series(np.where(max_set_per_match >= 4, 3.0, 2.0), index=df.index)

    def _clamp(val: float, lo: float, hi: float) -> float:
        return float(np.minimum(np.maximum(val, lo), hi))

    def _set_state_anchor(row, sets_needed: float) -> float:
        """
        Heuristic prior for P1 match win tied to set state.
        Encodes domain knowledge:
          - Tied sets (except final decider) → stay ~0.50
          - +1/-1 set → cap near 0.70/0.30
          - 2-2 (or 1-1 in best-of-3) can drift slightly with game edge
        """
        p1_sets = float(row.get('P1SetsWon', 0.0))
        p2_sets = float(row.get('P2SetsWon', 0.0))
        set_diff = p1_sets - p2_sets
        sets_played = p1_sets + p2_sets
        games_edge = float(row.get('CurrentSetGamesDiff', 0.0))

        in_decider_tied = (set_diff == 0) and (sets_played >= sets_needed - 1.0)

        if set_diff == 0:
            if in_decider_tied:
                # Allow small drift based on current-set games but stay near coin flip
                anchor = 0.5 + np.tanh(games_edge / 3.0) * 0.12
                return _clamp(anchor, 0.40, 0.60)
            return 0.50

        if set_diff == 1:
            anchor = 0.65 + np.tanh(games_edge / 4.0) * 0.03
            return _clamp(anchor, 0.30, 0.70)  # enforce <=0.70 with single-set lead

        if set_diff == -1:
            anchor = 0.35 - np.tanh(games_edge / 4.0) * 0.03
            return _clamp(anchor, 0.30, 0.70)

        if set_diff >= 2:
            # Match nearly over in favor of P1
            return 0.90

        # set_diff <= -2: P1 in deep trouble
        return 0.10

    def _anchor_strength(row, sets_needed: float) -> float:
        """
        How strongly to blend toward the set-based anchor.
        Early in match → strong; near finish → weaker.
        """
        p1_sets = float(row.get('P1SetsWon', 0.0))
        p2_sets = float(row.get('P2SetsWon', 0.0))
        set_diff = abs(p1_sets - p2_sets)
        sets_played = p1_sets + p2_sets
        distance = float(row.get('DistanceToMatchEnd', 0.0)) if 'DistanceToMatchEnd' in row else 0.0
        in_decider_tied = (set_diff == 0) and (sets_played >= sets_needed - 1.0)

        # Base strength by set situation
        if set_diff == 0:
            base = 0.80
        elif set_diff == 1:
            base = 0.70
        else:
            base = 0.50

        if in_decider_tied:
            base = min(base, 0.45)  # allow more flexibility in final-set ties

        # Fade anchor as we near the end of the match
        progress = _clamp(distance / 10.0, 0.0, 1.0)
        strength = base * (1.0 - 0.6 * progress)

        if 'MatchFinished' in row and float(row.get('MatchFinished', 0.0)) == 1.0:
            strength = 0.0

        return _clamp(strength, 0.0, 0.85)

    # Ensure all feature columns are numeric
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X_all = df[feature_cols].values.astype(float)

    # Hard labels (0/1)
    y_hard = df["p1_wins_match"].values.astype(float)

    # Soft labels blend hard outcome with set-based anchor so the model learns
    # desired probability shapes directly (rather than post-hoc guardrails).
    y_soft_list = []
    for idx, row in df.iterrows():
        sets_needed = sets_to_win_series.at[idx] if isinstance(sets_to_win_series, pd.Series) else 3.0
        anchor = _set_state_anchor(row, sets_needed)
        strength = _anchor_strength(row, sets_needed)
        blended = float(row["p1_wins_match"]) * (1.0 - strength) + anchor * strength
        y_soft_list.append(blended)
    y_soft = np.array(y_soft_list, dtype=float)
    
    # Extract point importance for sample weighting
    if 'point_importance' in df.columns:
        weights_all = df['point_importance'].values.astype(float)
    else:
        weights_all = np.ones(len(df), dtype=float)
    
    # Add set proximity weight: closer to winning = higher weight
    # For each point, calculate how many sets each player needs to win the match
    if 'P1SetsWon' in df.columns and 'P2SetsWon' in df.columns:
        p1_sets = df['P1SetsWon'].values.astype(float)
        p2_sets = df['P2SetsWon'].values.astype(float)
        
        # Determine sets needed to win match (3 for best-of-5, 2 for best-of-3)
        sets_needed = sets_to_win_series.values if isinstance(sets_to_win_series, pd.Series) else np.full(len(df), 3.0)
        
        # Calculate how many more sets each player needs
        p1_sets_needed = sets_needed - p1_sets  # e.g., if 2-0, P1 needs 1 more set (for bo5)
        p2_sets_needed = sets_needed - p2_sets
        
        # Minimum sets needed by either player (closest to winning)
        min_sets_to_victory = np.minimum(p1_sets_needed, p2_sets_needed)
        
        # Weight based on proximity to match end:
        # 1 set away: 2.0x
        # 2 sets away: 1.5x
        # 3+ sets away: 1.0x
        set_proximity_weight = np.where(min_sets_to_victory <= 1.0, 2.0,
                                       np.where(min_sets_to_victory <= 2.0, 1.5, 1.0))
        
        weights_all = weights_all * set_proximity_weight
        # Questo print è troppo verbose durante point-by-point prediction
        # print(f"[build_dataset] Set proximity weights - 1 set: {np.sum(min_sets_to_victory <= 1.0)}, "
        #       f"2 sets: {np.sum((min_sets_to_victory > 1.0) & (min_sets_to_victory <= 2.0))}, "
        #       f"3+ sets: {np.sum(min_sets_to_victory > 2.0)}")
    
    # Boost weights based on match competitiveness
    # Competitive matches (4-5 sets) are underrepresented, so upweight them
    match_groups = df.groupby(MATCH_COL, sort=False).ngroup()
    
    # Count total sets per match (P1SetsWon + P2SetsWon at end of match)
    if 'P1SetsWon' in df.columns and 'P2SetsWon' in df.columns:
        total_sets_per_match = (df['P1SetsWon'] + df['P2SetsWon']).groupby(match_groups).transform('max')
        
        # Weight multiplier based on competitiveness:
        # 5-set matches (very competitive): 4.0× (increased to teach model balanced situations)
        # 4-set matches (competitive): 2.5×
        # 3-set matches: 1.0× (baseline)
        competitive_weight = np.where(total_sets_per_match >= 4.5, 4.0,  # 5-set
                                     np.where(total_sets_per_match >= 3.5, 2.5,  # 4-set
                                             1.0))  # 3-set or less
        weights_all = weights_all * competitive_weight
    
    # Additional boost for tied-decider situations (2-2 in final set)
    if 'is_decider_tied' in df.columns:
        tied_boost = 2.0  # additional 2× for tied final-set points
        is_tied = df['is_decider_tied'].values.astype(float)
        weights_all = weights_all * (1.0 + is_tied * (tied_boost - 1.0))

    mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[mask]
    y_soft_masked = y_soft[mask]
    y_hard_masked = y_hard[mask]
    sample_weights = weights_all[mask]

    # CRITICAL FIX: Use HARD labels (0/1) for classification, not soft blended labels
    # Soft labels were confusing the network and causing wrong predictions
    # Return hard labels as the primary target
    return X, y_hard_masked, mask, sample_weights, y_hard_masked


def build_dataset_point_level(df: pd.DataFrame):
    """
    Build feature matrix X and target y for POINT-LEVEL prediction.
    
    Target: 1 if P1 wins the current point, 0 if P2 wins
    
    This eliminates look-ahead bias since the model predicts individual points
    rather than the match outcome. Features mirror the match-level model, including
    set-advantage and calibrated set-win priors/logits.
    """
    df = df.copy()
    feature_cols = POINT_FEATURE_COLUMNS.copy()

    # Ensure all feature columns are numeric
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X_all = df[feature_cols].values.astype(float)
    
    # Target: who wins THIS POINT (not the match)
    if 'PointWinner' in df.columns:
        y_all = (df['PointWinner'] == 1).astype(int).values
    else:
        raise ValueError("PointWinner column required for point-level prediction")
    
    # Extract point importance for sample weighting
    if 'point_importance' in df.columns:
        weights_all = df['point_importance'].values.astype(float)
    else:
        weights_all = np.ones(len(df), dtype=float)
    
    # Apply same competitive match weighting as match-level model
    match_groups = df.groupby(MATCH_COL, sort=False).ngroup()
    
    if 'P1SetsWon' in df.columns and 'P2SetsWon' in df.columns:
        total_sets_per_match = (df['P1SetsWon'] + df['P2SetsWon']).groupby(match_groups).transform('max')
        competitive_weight = np.where(total_sets_per_match >= 4.5, 4.0,
                                     np.where(total_sets_per_match >= 3.5, 2.5,
                                             1.0))
        weights_all = weights_all * competitive_weight
    
    if 'is_decider_tied' in df.columns:
        tied_boost = 2.0
        is_tied = df['is_decider_tied'].values.astype(float)
        weights_all = weights_all * (1.0 + is_tied * (tied_boost - 1.0))

    mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[mask]
    y = y_all[mask]
    sample_weights = weights_all[mask]

    return X, y, mask, sample_weights


def build_clean_features_nn(df: pd.DataFrame):
    """
    Costruisce features PULITE per la rete neurale, eliminando duplicati e 
    mantenendo solo quelle essenziali per imparare le regole del tennis.
    
    Features progettate per far capire alla rete:
    1. Come funziona un GAME (punti 0-15-30-40-deuce)
    2. Come funziona un SET (vincere 6+ game con vantaggio di 2)
    3. Come funziona un MATCH (vincere 2/3 set per donne, 3/5 per uomini)
    4. Come funziona il TIE-BREAK (punteggio numerico, primo a 7 con vantaggio di 2)
    5. Riconoscere PUNTI CRITICI (break/set/match point)
    
    Features (25 totali - RITORNO ALLE ORIGINALI):
    CORE SCORING (6):
        - P1_points: punti nel game corrente (0-4, dove 4=vantaggio)
        - P2_points: punti nel game corrente (0-4)
        - P1_games: game vinti nel set corrente (0-13)
        - P2_games: game vinti nel set corrente (0-13)
        - P1_sets: set vinti (0-3)
        - P2_sets: set vinti (0-3)
    
    CONTEXT (4):
        - point_server: chi sta servendo (1 o 2)
        - set_number: numero del set corrente (1-5)
        - game_number: numero del game corrente (1-40)
        - point_number: numero del punto nel match (1-500)
    
    TIE-BREAK (6):
        - is_tiebreak: 1 se siamo in tie-break, 0 altrimenti
        - is_decisive_tiebreak: 1 se tie-break del set decisivo
        - tb_p1_points: punti di P1 nel tie-break (0-20)
        - tb_p2_points: punti di P2 nel tie-break (0-20)
        - tb_p1_needs_to_win: quanti punti mancano a P1 per vincere (0-7)
        - tb_p2_needs_to_win: quanti punti mancano a P2 per vincere (0-7)
        - tb_points_diff: (P1_TB - P2_TB)/7 NORMALIZZATA (range -1 a +1)
        - tb_p1_is_leading: 1 se P1 avanti nel TB, 0 altrimenti (BINARIA)
        - tb_p1_one_point_away: 1 se P1 a un punto dal vincere TB
        - tb_p2_one_point_away: 1 se P2 a un punto dal vincere TB
    
    MATCH FORMAT (3):
        - is_best_of_5: 1 se match al meglio di 5 set, 0 se best-of-3
        - sets_to_win: quanti set servono per vincere (2 o 3)
        - is_final_set: 1 se siamo nell'ultimo set possibile
    
    PERFORMANCE (6):
        - P_srv_win_long: probabilità di vincere punto al servizio (finestra lunga)
        - P_srv_lose_long: probabilità quando si riceve (finestra lunga)
        - P_srv_win_short: probabilità al servizio (finestra corta, real-time)
        - P_srv_lose_short: probabilità in ricezione (finestra corta)
        - p1_momentum: momentum di P1 (media pesata della leverage)
        - p2_momentum: momentum di P2
    
    CRITICAL POINTS (6) - ESPLICITI:
        - is_p1_break_point: P1 in risposta può vincere il game
        - is_p2_break_point: P2 in risposta può vincere il game
        - is_p1_set_point: P1 può vincere il set
        - is_p2_set_point: P2 può vincere il set
        - is_p1_match_point: P1 può vincere il match
        - is_p2_match_point: P2 può vincere il match
    
    Target:
        - p1_wins_match: 1 se P1 vince il match, 0 altrimenti
    """
    df = df.copy()
    
    # Helper per convertire punteggio tennis a numerico
    def score_to_numeric(score):
        if pd.isna(score):
            return 0
        score_str = str(score).strip().upper()
        score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'A': 4}
        # Se è già numerico (tie-break), restituiscilo
        try:
            return int(score_str)
        except ValueError:
            return score_map.get(score_str, 0)
    
    # 1. CORE SCORING - rappresentazione esplicita del punteggio
    # Usa colonne esistenti o crea con valori di default
    if 'P1Score' in df.columns:
        df['P1_points'] = df['P1Score'].apply(score_to_numeric).astype(float)
    else:
        df['P1_points'] = 0.0
    
    if 'P2Score' in df.columns:
        df['P2_points'] = df['P2Score'].apply(score_to_numeric).astype(float)
    else:
        df['P2_points'] = 0.0
    
    df['P1_games'] = pd.to_numeric(df['P1GamesWon'], errors='coerce').fillna(0).astype(float) if 'P1GamesWon' in df.columns else 0.0
    df['P2_games'] = pd.to_numeric(df['P2GamesWon'], errors='coerce').fillna(0).astype(float) if 'P2GamesWon' in df.columns else 0.0
    df['P1_sets'] = pd.to_numeric(df['P1SetsWon'], errors='coerce').fillna(0).astype(float) if 'P1SetsWon' in df.columns else 0.0
    df['P2_sets'] = pd.to_numeric(df['P2SetsWon'], errors='coerce').fillna(0).astype(float) if 'P2SetsWon' in df.columns else 0.0
    
    # 2. CONTEXT
    df['point_server'] = pd.to_numeric(df[SERVER_COL], errors='coerce').fillna(1).astype(float) if SERVER_COL in df.columns else 1.0
    df['set_number'] = pd.to_numeric(df['SetNo'], errors='coerce').fillna(1).astype(float) if 'SetNo' in df.columns else 1.0
    df['game_number'] = pd.to_numeric(df['GameNo'], errors='coerce').fillna(1).astype(float) if 'GameNo' in df.columns else 1.0
    df['point_number'] = pd.to_numeric(df['PointNumber'], errors='coerce').fillna(1).astype(float) if 'PointNumber' in df.columns else 1.0
    
    # match_progress: progresso del match (0-1) per dare contesto temporale
    # Calcola per ogni match quanti punti totali ci sono, poi normalizza
    max_points_per_match = df.groupby(MATCH_COL)['point_number'].transform('max')
    df['match_progress'] = (df['point_number'] / max_points_per_match).clip(0, 1).astype(float)
    
    # 3. TIE-BREAK features
    # IMPORTANTE: Regola Wimbledon 2019+
    # - Set 1-4: tie-break al 6-6
    # - Set 5 (decisivo): tie-break al 12-12
    
    # Determina se siamo nel SET FINALE (non solo "potenzialmente finale")
    # Set finale = set 5 in best-of-5, set 3 in best-of-3
    # SOLO in questo set si usa la regola 12-12
    is_final_set_bo5 = (df.get('is_best_of_5', 1.0) == 1.0) & (df['set_number'] == 5)
    is_final_set_bo3 = (df.get('is_best_of_5', 1.0) == 0.0) & (df['set_number'] == 3)
    is_final_set_actual = is_final_set_bo5 | is_final_set_bo3
    
    # Determina soglia per tie-break: 12-12 nel set finale, 6-6 altrimenti
    tb_threshold = np.where(is_final_set_actual, 12, 6)
    is_games_at_tb_threshold = (df['P1_games'] == df['P2_games']) & (df['P1_games'] >= tb_threshold)
    
    # Determina se i punteggi sono numerici (tie-break) o tennis standard
    if 'P1Score' in df.columns:
        p1_score_raw = df['P1Score'].astype(str).str.strip().str.upper()
    else:
        p1_score_raw = pd.Series('0', index=df.index)
    tennis_scores = {'0', '15', '30', '40', 'AD', 'A'}
    is_numeric_score = ~p1_score_raw.isin(tennis_scores)
    
    df['is_tiebreak'] = (is_games_at_tb_threshold & is_numeric_score).astype(float)
    
    # Tie-break decisivo: SOLO tie-break nel SET FINALE (12-12 al set 5)
    # Non tutti i tie-break con set pari, solo quello del set 5/3
    df['is_decisive_tiebreak'] = (df['is_tiebreak'] == 1.0) & is_final_set_actual
    
    # Punteggio nel tie-break (quando in tie-break, i punti sono già numerici)
    df['tb_p1_points'] = np.where(df['is_tiebreak'] == 1.0, df['P1_points'], 0.0)
    df['tb_p2_points'] = np.where(df['is_tiebreak'] == 1.0, df['P2_points'], 0.0)
    
    # Quanto manca per vincere il tie-break (serve 7+ con vantaggio di 2)
    # Semplificato: se < 6, mancano (7-punti); se >= 6, serve vantaggio di 2
    def points_needed_to_win_tb(my_points, opp_points):
        my_points = np.maximum(my_points, 0)
        opp_points = np.maximum(opp_points, 0)
        
        # Se sotto 6, manca (7 - punti)
        needs = 7.0 - my_points
        
        # Se sopra 6, serve vantaggio di 2
        # Esempio: 7-6 → serve ancora 1 punto, 6-7 → serve 2 punti (pareggiare + vantaggio)
        needs = np.where(
            (my_points >= 6) & (opp_points >= 6),
            np.maximum(0, opp_points - my_points + 2),  # deve superare di 2
            needs
        )
        
        return np.clip(needs, 0, 7)
    
    df['tb_p1_needs_to_win'] = np.where(
        df['is_tiebreak'] == 1.0,
        points_needed_to_win_tb(df['tb_p1_points'].values, df['tb_p2_points'].values),
        0.0
    )
    df['tb_p2_needs_to_win'] = np.where(
        df['is_tiebreak'] == 1.0,
        points_needed_to_win_tb(df['tb_p2_points'].values, df['tb_p1_points'].values),
        0.0
    )
    
    # Differenza punti nel tie-break (NORMALIZZATA per facilitare apprendimento)
    raw_diff = df['tb_p1_points'] - df['tb_p2_points']
    df['tb_points_diff'] = np.where(
        df['is_tiebreak'] == 1.0,
        raw_diff / 7.0,  # Normalizza: range tipico -7 a +7 diventa -1 a +1
        0.0
    )
    
    # Chi è in vantaggio nel tie-break (BINARIA, più facile da imparare)
    df['tb_p1_is_leading'] = np.where(
        df['is_tiebreak'] == 1.0,
        (raw_diff > 0).astype(float),
        0.0
    )
    
    # Chi è a un punto dal vincere il tie-break (feature CRITICA)
    df['tb_p1_one_point_away'] = np.where(
        df['is_tiebreak'] == 1.0,
        (df['tb_p1_needs_to_win'] <= 1.0).astype(float),
        0.0
    )
    df['tb_p2_one_point_away'] = np.where(
        df['is_tiebreak'] == 1.0,
        (df['tb_p2_needs_to_win'] <= 1.0).astype(float),
        0.0
    )
    
    # 4. MATCH FORMAT
    # Determina formato dal numero massimo di set nel match
    # Determine match format: best-of-5 or best-of-3
    # CRITICAL: This must be determined from match METADATA, not from sets played so far!
    # 
    # For training: we have complete match, so we can look at final set count
    # For prediction: we need to infer from tournament/gender or use explicit flag
    
    # Check if we already have is_best_of_5 as a column (set by user or tournament metadata)
    if 'is_best_of_5' not in df.columns:
        # Infer from data: if ANY match has 4+ sets, it's best-of-5
        # Group by match and get the maximum set number seen IN THIS DATAFRAME
        if 'SetNo_original' in df.columns:
            max_set_per_match = df.groupby(MATCH_COL)['SetNo_original'].transform('max')
        else:
            max_set_per_match = df.groupby(MATCH_COL)['set_number'].transform('max')
        
        # Propagate the format to ALL points in that match
        # If max_set >= 4 anywhere in the match, it's best-of-5
        match_format = df.groupby(MATCH_COL).apply(
            lambda g: (g['SetNo_original'].max() if 'SetNo_original' in g else g['set_number'].max()) >= 4
        )
        df['is_best_of_5'] = df[MATCH_COL].map(match_format).fillna(1.0).astype(float)
    else:
        df['is_best_of_5'] = pd.to_numeric(df['is_best_of_5'], errors='coerce').fillna(1.0).astype(float)
    
    df['sets_to_win'] = np.where(df['is_best_of_5'] == 1.0, 3.0, 2.0)
    
    # Set finale: siamo nel set che potrebbe decidere il match
    # Best-of-5: set 3, 4, o 5 con almeno 2 set giocati
    # Best-of-3: set 2 o 3 con almeno 1 set giocato
    is_potentially_final = (
        ((df['is_best_of_5'] == 1.0) & (df['set_number'] >= 3) & (sets_played >= 2)) |
        ((df['is_best_of_5'] == 0.0) & (df['set_number'] >= 2) & (sets_played >= 1))
    )
    df['is_final_set'] = is_potentially_final.astype(float)
    
    # 5. PERFORMANCE features - prendiamo quelle già calcolate
    performance_cols = [
        'P_srv_win_long', 'P_srv_lose_long', 
        'P_srv_win_short', 'P_srv_lose_short'
    ]
    for col in performance_cols:
        if col not in df.columns:
            df[col] = 0.5  # default neutro
    
    # Momentum
    if 'P1Momentum' in df.columns and 'P2Momentum' in df.columns:
        df['p1_momentum'] = pd.to_numeric(df['P1Momentum'], errors='coerce').fillna(0).astype(float)
        df['p2_momentum'] = pd.to_numeric(df['P2Momentum'], errors='coerce').fillna(0).astype(float)
    else:
        df['p1_momentum'] = 0.0
        df['p2_momentum'] = 0.0
    
    # 6. CRITICAL POINTS features - ESPLICITE
    # Usa le features già calcolate da add_additional_features
    # Se non esistono, creale basandoti sulle regole del tennis
    
    # Break point: ricevitore può vincere il game
    if 'is_break_point' not in df.columns:
        # Calcola manualmente: ricevitore a 40 o advantage
        df['is_break_point'] = 0.0
    else:
        df['is_break_point'] = pd.to_numeric(df['is_break_point'], errors='coerce').fillna(0).astype(float)
    
    # Set point: giocatore può vincere il set
    if 'is_set_point' not in df.columns:
        df['is_set_point'] = 0.0
    else:
        df['is_set_point'] = pd.to_numeric(df['is_set_point'], errors='coerce').fillna(0).astype(float)
    
    # Match point: giocatore può vincere il match
    if 'is_match_point' not in df.columns:
        df['is_match_point'] = 0.0
    else:
        df['is_match_point'] = pd.to_numeric(df['is_match_point'], errors='coerce').fillna(0).astype(float)
    
    # Separa per giocatore guardando il punteggio e chi serve
    # P1 break point = P1 in risposta può vincere il game (server=2, P1 a 40+)
    # P2 break point = P2 in risposta può vincere il game (server=1, P2 a 40+)
    server = df['point_server']
    
    # Score helper per determinare chi è avanti
    p1_ahead_in_game = df['P1_points'] > df['P2_points']
    p2_ahead_in_game = df['P2_points'] > df['P1_points']
    p1_can_win_game = (df['P1_points'] >= 3) & p1_ahead_in_game
    p2_can_win_game = (df['P2_points'] >= 3) & p2_ahead_in_game
    
    # Break point: ricevitore può vincere il game
    df['is_p1_break_point'] = ((server == 2) & p1_can_win_game).astype(float)
    df['is_p2_break_point'] = ((server == 1) & p2_can_win_game).astype(float)
    
    # Set point: giocatore può vincere il set (5+ game, avanti di 1+)
    p1_can_win_set = (df['P1_games'] >= 5) & (df['P1_games'] >= df['P2_games'] + 1)
    p2_can_win_set = (df['P2_games'] >= 5) & (df['P2_games'] >= df['P1_games'] + 1)
    
    df['is_p1_set_point'] = (p1_can_win_set & p1_can_win_game).astype(float)
    df['is_p2_set_point'] = (p2_can_win_set & p2_can_win_game).astype(float)
    
    # Match point: giocatore a un set dalla vittoria E può vincere il set
    p1_one_set_away = df['P1_sets'] >= (df['sets_to_win'] - 1)
    p2_one_set_away = df['P2_sets'] >= (df['sets_to_win'] - 1)
    
    df['is_p1_match_point'] = (p1_one_set_away & df['is_p1_set_point'] == 1.0).astype(float)
    df['is_p2_match_point'] = (p2_one_set_away & df['is_p2_set_point'] == 1.0).astype(float)
    
    # Lista delle feature pulite in ordine canonico
    CLEAN_FEATURE_COLUMNS = [
        # Core scoring (6)
        'P1_points', 'P2_points',
        'P1_games', 'P2_games',
        'P1_sets', 'P2_sets',
        # Context (4)
        'point_server', 'set_number', 'game_number', 'point_number',
        # COMMENTATO: match_progress - causava problemi con match points
        # Tie-break (6)
        'is_tiebreak', 'is_decisive_tiebreak',
        'tb_p1_points', 'tb_p2_points',
        'tb_p1_needs_to_win', 'tb_p2_needs_to_win',
        # COMMENTATO: tb_points_diff, tb_p1_is_leading, tb_p1_one_point_away, tb_p2_one_point_away
        # Match format (3)
        'is_best_of_5', 'sets_to_win', 'is_final_set',
        # Performance (6)
        'P_srv_win_long', 'P_srv_lose_long',
        'P_srv_win_short', 'P_srv_lose_short',
        'p1_momentum', 'p2_momentum',
        # COMMENTATO: Critical points (6) - is_p1_break_point, is_p2_break_point, etc.
    ]
    
    # Converti tutte le colonne a float
    for col in CLEAN_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
    
    # Costruisci matrice delle features
    X_all = df[CLEAN_FEATURE_COLUMNS].values.astype(float)
    
    # Target
    if 'p1_wins_match' not in df.columns:
        raise ValueError("p1_wins_match column required - call add_match_labels first")
    y_all = df['p1_wins_match'].values.astype(int)
    
    # Sample weights - usa importanza per dare più peso ai punti critici
    if 'point_importance' in df.columns:
        weights_all = df['point_importance'].values.astype(float)
        # Cap più alto per dare più peso ai punti critici (match points, etc.)
        weights_all = np.clip(weights_all, 1.0, 6.0)  # Cap a 6 per punti molto critici
    else:
        weights_all = np.ones(len(df), dtype=float)
    
    # Bonus per match competitivi (andati a 3+ set)
    match_groups = df.groupby(MATCH_COL, sort=False).ngroup()
    total_sets = (df['P1_sets'] + df['P2_sets']).groupby(match_groups).transform('max')
    competitive_weight = np.where(total_sets >= 3.5, 1.5, 1.0)
    weights_all = weights_all * competitive_weight
    
    # Maschera per rimuovere NaN
    mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)
    X = X_all[mask]
    y = y_all[mask]
    sample_weights = weights_all[mask]
    
    print(f"[build_clean_features_nn] Built {X.shape[0]} samples with {X.shape[1]} features")
    print(f"[build_clean_features_nn] Features (25 total - ORIGINALI):")
    print(f"  - Core scoring: 6")
    print(f"  - Context: 4") 
    print(f"  - Tie-break: 6")
    print(f"  - Match format: 3")
    print(f"  - Performance: 6")
    print(f"  - Critical points: 6 (break/set/match point per giocatore)")
    
    return X, y, mask, sample_weights, CLEAN_FEATURE_COLUMNS