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
      - point_importance: weight/importance of each point
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
        p1_sets = pd.to_numeric(row.get('P1Sets', 0), errors='coerce')
        p2_sets = pd.to_numeric(row.get('P2Sets', 0), errors='coerce')
        if pd.isna(p1_sets):
            p1_sets = 0
        if pd.isna(p2_sets):
            p2_sets = 0
        set_diff = abs(p1_sets - p2_sets)
        
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
    
    # Calculate point importance for sample weighting
    df['point_importance'] = df.apply(calculate_point_importance, axis=1)

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

    # Check if point_importance exists (it should be computed before this)
    if "point_importance" not in df.columns:
        # Fallback: use uniform weights
        df["point_importance"] = 1.0
    
    # Weight leverage by point importance for momentum calculation
    # Critical points have more impact on momentum
    df["weighted_leverage"] = df["leverage"] * df["point_importance"]
    
    # Compute weighted momentum using EWMA
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
      - point_importance: critical point indicator (1.0-7.0)

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
        "point_importance",
    ]

    # Ensure all feature columns are numeric
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X_all = df[feature_cols].values.astype(float)
    y_all = df["p1_wins_match"].values
    
    # Extract point importance for sample weighting
    if 'point_importance' in df.columns:
        weights_all = df['point_importance'].values.astype(float)
    else:
        weights_all = np.ones(len(df), dtype=float)

    mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[mask]
    y = y_all[mask]
    sample_weights = weights_all[mask]

    return X, y, mask, sample_weights
