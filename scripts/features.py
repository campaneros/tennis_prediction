import numpy as np
import pandas as pd

from .data_loader import (
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
    "P1SetsWon",  # NEW: Number of sets won by P1
    "P2SetsWon",  # NEW: Number of sets won by P2
    "SetsWonDiff",
    "SetsWonAdvantage",
    "SetWinProbPrior",
    "SetWinProbEdge",
    "SetWinProbLogit",
    "is_decider_tied",
    "DistanceToMatchEnd",
    "MatchFinished",
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
        
        # Get scores
        p1_score = score_to_numeric(row.get('P1Score', 0))
        p2_score = score_to_numeric(row.get('P2Score', 0))
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
        is_set_point_situation = False
        set_point_player = None
        
        if p1_games >= 5 and p1_games >= p2_games + 1:
            # P1 is ahead and could win set
            is_set_point_situation = True
            set_point_player = 1
        elif p2_games >= 5 and p2_games >= p1_games + 1:
            # P2 is ahead and could win set
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
        # Check if either player is one point away from winning the match
        is_match_point = False
        match_point_player = 0
        
        # Best-of-5: need 3 sets to win
        # Best-of-3: need 2 sets to win
        # Determine format based on max sets played (if 5th set exists, it's best-of-5)
        set_no = pd.to_numeric(row.get('SetNo', 1), errors='coerce')
        if pd.isna(set_no):
            set_no = 1
        
        # Check if player could win match by winning this point
        if p1_sets == 2:  # P1 has won 2 sets
            # In best-of-5: need 1 more set; in best-of-3: match point
            if set_no <= 3:  # Best-of-3
                if p1_games >= 5 and p1_games >= p2_games + 1:  # Could win set this game
                    if (server == 1 and p1_score >= 3 and p1_score > p2_score) or \
                       (server == 2 and p1_score >= 3 and p1_score > p2_score):
                        is_match_point = True
                        match_point_player = 1
            elif p1_sets > p2_sets:  # In best-of-5, P1 leading in sets
                if p1_games >= 5 and p1_games >= p2_games + 1:  # Could win set
                    if (server == 1 and p1_score >= 3 and p1_score > p2_score) or \
                       (server == 2 and p1_score >= 3 and p1_score > p2_score):
                        is_match_point = True
                        match_point_player = 1
        
        if p2_sets == 2:  # P2 has won 2 sets
            if set_no <= 3:  # Best-of-3
                if p2_games >= 5 and p2_games >= p1_games + 1:  # Could win set this game
                    if (server == 1 and p2_score >= 3 and p2_score > p1_score) or \
                       (server == 2 and p2_score >= 3 and p2_score > p1_score):
                        is_match_point = True
                        match_point_player = 2
            elif p2_sets > p1_sets:  # In best-of-5, P2 leading in sets
                if p2_games >= 5 and p2_games >= p1_games + 1:  # Could win set
                    if (server == 1 and p2_score >= 3 and p2_score > p1_score) or \
                       (server == 2 and p2_score >= 3 and p2_score > p1_score):
                        is_match_point = True
                        match_point_player = 2
        
        # Match point: maximum importance
        if is_match_point:
            #print(f"Match point detected for Player {match_point_player}")
            return 7.0
        
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
        
        # Tiebreak detection: both players at 6 games: +2.5
        if p1_games == 6 and p2_games == 6:
            weight += 2.5
        
        # Game differential weight: closer games are more critical
        # 5-4 or 4-5 is more critical than 5-0 or 5-1
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
        if p1_sets >= 1 or p2_sets >= 1:  # Not first set
            if set_diff == 0:  # Tied in sets (1-1, 2-2)
                weight += 1.2
            elif set_diff == 1 and (p1_sets >= 1 and p2_sets >= 1):  # 2-1 situation
                weight += 0.8
        
        # Critical games: score is 30-30 or higher (tight game): +0.5
        if p1_score >= 2 and p2_score >= 2 and not (p1_score >= 3 and p2_score >= 3):
            weight += 0.5
        
        return min(weight, 7.0)  # Cap at 7.0

    # Momentum difference - normalize per MATCH to preserve match-level context
    # Z-score normalization balances momentum across entire match duration
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

    # Ensure server / winner columns are numeric and drop rows with missing values
    df[SERVER_COL] = pd.to_numeric(df[SERVER_COL], errors='coerce')
    df[POINT_WINNER_COL] = pd.to_numeric(df[POINT_WINNER_COL], errors='coerce')
    df = df.dropna(subset=[SERVER_COL, POINT_WINNER_COL])
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

    # Return both soft labels (for training) and hard labels (for evaluation)
    return X, y_soft_masked, mask, sample_weights, y_hard_masked


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
