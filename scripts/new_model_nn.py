"""
New Neural Network Model with Multi-Task Learning and Distance Features

Approach:
1. Multi-task: predicts match, set, and game outcomes simultaneously
2. Distance-to-victory features: explicit mathematical features showing how close each player is
3. Custom loss: penalizes logical inconsistencies
4. Simple architecture: [128, 64] with moderate regularization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import json
import os

from .features import (
    add_additional_features, add_match_labels, 
    MATCH_COL, SERVER_COL
)
from .data_loader import load_points_multiple
from .config import load_config


def calculate_distance_features(df):
    """
    Calculate explicit mathematical features showing distance to victory.
    These features encode the exact rules of tennis scoring.
    
    Returns 8 features per point:
    - p1_points_to_win_game: how many points P1 needs to win current game
    - p2_points_to_win_game: how many points P2 needs to win current game
    - p1_games_to_win_set: how many games P1 needs to win current set
    - p2_games_to_win_set: how many games P2 needs to win current set
    - p1_sets_to_win_match: how many sets P1 needs to win match
    - p2_sets_to_win_match: how many sets P2 needs to win match
    - p1_can_close_match_this_game: 1 if P1 wins match by winning this game
    - p2_can_close_match_this_game: 1 if P2 wins match by winning this game
    """
    
    # Parse scores
    def score_to_numeric(score_str):
        score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'A': 4}
        score_clean = str(score_str).strip().upper()
        return score_map.get(score_clean, 0)
    
    if 'P1Score' in df.columns:
        p1_pts = df['P1Score'].apply(score_to_numeric)
    else:
        p1_pts = 0
    
    if 'P2Score' in df.columns:
        p2_pts = df['P2Score'].apply(score_to_numeric)
    else:
        p2_pts = 0
    
    p1_games = pd.to_numeric(df.get('P1GamesWon', 0), errors='coerce').fillna(0)
    p2_games = pd.to_numeric(df.get('P2GamesWon', 0), errors='coerce').fillna(0)
    p1_sets = pd.to_numeric(df.get('P1SetsWon', 0), errors='coerce').fillna(0)
    p2_sets = pd.to_numeric(df.get('P2SetsWon', 0), errors='coerce').fillna(0)
    
    # 1. Points to win game
    # Normal game: need to reach 4 points (40) with 2-point lead
    # At 3-3 (deuce): need 2 consecutive points
    def points_to_win_game(my_pts, opp_pts):
        my_pts = np.array(my_pts)
        opp_pts = np.array(opp_pts)
        
        # If already at 4+ and ahead, need 0 more
        needs = np.where((my_pts >= 4) & (my_pts > opp_pts), 0, 999)
        
        # If at deuce or behind, calculate
        # At 3-3: need 2 points
        needs = np.where((my_pts == 3) & (opp_pts == 3), 2, needs)
        # At 4-3 (my advantage): need 1 point
        needs = np.where((my_pts == 4) & (opp_pts == 3), 1, needs)
        # At 3-4 (their advantage): need 3 points (get to deuce, then win 2)
        needs = np.where((my_pts == 3) & (opp_pts == 4), 3, needs)
        
        # Before deuce
        needs = np.where((my_pts < 3) & (needs == 999), 4 - my_pts, needs)
        needs = np.where((my_pts == 3) & (opp_pts < 3) & (needs == 999), 1, needs)
        
        return np.clip(needs, 0, 5).astype(float)
    
    df['p1_points_to_win_game'] = points_to_win_game(p1_pts, p2_pts) / 5.0
    df['p2_points_to_win_game'] = points_to_win_game(p2_pts, p1_pts) / 5.0
    
    # 2. Games to win set
    # Need 6+ games with 2-game lead (or 7-6 via tiebreak)
    # Special rule: final set (5th) uses 12-12 tiebreak threshold instead of 6-6
    def games_to_win_set(my_games, opp_games, set_num, max_set):
        my_games = np.array(my_games)
        opp_games = np.array(opp_games)
        set_num = np.array(set_num)
        max_set = np.array(max_set)
        
        # Determine tiebreak threshold: 6 for normal sets, 12 for final set
        is_final_set = (set_num >= 3) & (max_set >= 4)  # Set 3+ in best-of-5
        tb_threshold = np.where(is_final_set, 12, 6)
        
        # Already won
        needs = np.where((my_games >= tb_threshold) & (my_games >= opp_games + 2), 0, 999)
        
        # At tiebreak threshold: tiebreak determines winner (model as needing 1 "super-game")
        needs = np.where((my_games == tb_threshold) & (opp_games == tb_threshold), 1, needs)
        
        # Leading by 1 at threshold: need 1 game
        needs = np.where((my_games == tb_threshold) & (opp_games == tb_threshold - 1), 1, needs)
        
        # At one below threshold: need to get to threshold with lead
        needs = np.where((my_games == tb_threshold - 1) & (opp_games == tb_threshold - 1), 2, needs)  # best case: win 2
        needs = np.where((my_games < tb_threshold) & (needs == 999), tb_threshold - my_games, needs)
        
        return np.clip(needs, 0, 13).astype(float)
    
    set_num = df['set_number'].values
    if 'SetNo_original' in df.columns:
        max_set_temp = df.groupby(MATCH_COL)['SetNo_original'].transform('max').values
    else:
        max_set_temp = df.groupby(MATCH_COL)['set_number'].transform('max').values
    
    df['p1_games_to_win_set'] = games_to_win_set(p1_games, p2_games, set_num, max_set_temp) / 13.0
    df['p2_games_to_win_set'] = games_to_win_set(p2_games, p1_games, set_num, max_set_temp) / 13.0
    
    # 3. Sets to win match
    # Best-of-5: need 3 sets, best-of-3: need 2 sets
    if 'SetNo_original' in df.columns:
        max_set = df.groupby(MATCH_COL)['SetNo_original'].transform('max')
    else:
        max_set = df.groupby(MATCH_COL)['set_number'].transform('max')
    
    sets_to_win = np.where(max_set >= 4, 3, 2)
    
    df['p1_sets_to_win_match'] = (np.clip(sets_to_win - p1_sets, 0, 3) / 3.0).astype(float)
    df['p2_sets_to_win_match'] = (np.clip(sets_to_win - p2_sets, 0, 3) / 3.0).astype(float)
    
    # 4. Can close match this game?
    # True if: (1) one set away from winning AND (2) can win this set by winning this game
    # Since features are normalized, 1 set away = 1/3 ≈ 0.333
    p1_one_set_away = np.isclose(df['p1_sets_to_win_match'], 1/3.0, atol=0.01)
    p2_one_set_away = np.isclose(df['p2_sets_to_win_match'], 1/3.0, atol=0.01)
    
    # Since features are normalized, 1 game away = 1/13 ≈ 0.077
    p1_can_win_set_this_game = (df['p1_games_to_win_set'] <= 1/13.0 + 0.01) & (df['p1_games_to_win_set'] > 0)
    p2_can_win_set_this_game = (df['p2_games_to_win_set'] <= 1/13.0 + 0.01) & (df['p2_games_to_win_set'] > 0)
    
    df['p1_can_close_match_this_game'] = (p1_one_set_away & p1_can_win_set_this_game).astype(float)
    df['p2_can_close_match_this_game'] = (p2_one_set_away & p2_can_win_set_this_game).astype(float)
    
    # 5. CRITICAL POINT FEATURES - These encode the LOGIC of tennis
    # These are TRUE if winning THIS EXACT POINT wins the game/set/match
    
    # Match point: if I win THIS point, I win the match
    # Conditions: (1) one set away, (2) one game away from set, (3) one point away from game
    # For tiebreaks: also check tiebreak points
    is_tiebreak = df.get('is_tiebreak', 0) > 0
    tb_p1 = pd.to_numeric(df.get('tb_p1_points', 0), errors='coerce').fillna(0)
    tb_p2 = pd.to_numeric(df.get('tb_p2_points', 0), errors='coerce').fillna(0)
    
    # Determine sets needed to win match from match FORMAT (not from sets played!)
    # Use is_best_of_5 feature or infer from tournament structure
    if 'is_best_of_5' in df.columns:
        is_bo5 = df['is_best_of_5'].astype(bool)
    else:
        # Fallback: infer from gender or tournament (males usually best-of-5 in Grand Slams)
        # For safety, check if match went to 4+ sets
        match_went_to_4_sets = df.groupby(MATCH_COL)['set_number'].transform('max') >= 4
        is_bo5 = match_went_to_4_sets
    
    sets_needed = np.where(is_bo5, 3, 2)  # Best-of-5 needs 3 sets, best-of-3 needs 2
    
    # One set away means: current sets == sets_needed - 1
    # For best-of-5: need to be at 2 sets (2-0, 2-1, or 2-2 situation where you're at 2)
    # For best-of-3: need to be at 1 set (1-0 or 1-1 situation where you're at 1)
    p1_one_set_away_direct = (p1_sets == sets_needed - 1)
    p2_one_set_away_direct = (p2_sets == sets_needed - 1)
    
    # Game point: winning this point wins the current game
    p1_game_point = np.zeros(len(df), dtype=float)
    p2_game_point = np.zeros(len(df), dtype=float)
    
    # In regular games
    p1_game_point = np.where(~is_tiebreak & (p1_pts == 4) & (p1_pts > p2_pts), 1.0, p1_game_point)
    p1_game_point = np.where(~is_tiebreak & (p1_pts == 3) & (p2_pts < 3), 1.0, p1_game_point)
    p2_game_point = np.where(~is_tiebreak & (p2_pts == 4) & (p2_pts > p1_pts), 1.0, p2_game_point)
    p2_game_point = np.where(~is_tiebreak & (p2_pts == 3) & (p1_pts < 3), 1.0, p2_game_point)
    
    # In tiebreaks: need 7+ points with 2-point lead (or 1-point lead if already at 6+)
    p1_game_point = np.where(is_tiebreak & (tb_p1 >= 6) & (tb_p1 >= tb_p2 + 1), 1.0, p1_game_point)
    p2_game_point = np.where(is_tiebreak & (tb_p2 >= 6) & (tb_p2 >= tb_p1 + 1), 1.0, p2_game_point)
    
    df['is_game_point_p1'] = p1_game_point
    df['is_game_point_p2'] = p2_game_point
    
    # Set point: winning this point wins the current set
    # This is TRUE only if: on game point AND winning THIS game wins the set
    # 
    # Rules for winning set by winning this game:
    # 1. If at 5-X (X<=3): winning goes to 6-X, wins set
    # 2. If at 5-4: winning goes to 6-4, wins set
    # 3. If at 6-5: winning goes to 7-5, wins set  
    # 4. If at 5-5: winning goes to 6-5, does NOT win set (need one more)
    # 5. If in tiebreak: winning tiebreak wins the set
    # 6. For final set with 12-12 rule: need to consider extended games (6-6, 7-7, ..., 12-12)
    
    # Simplified logic: can win set this game if:
    # - (my_games == 5 AND opp_games <= 4) OR
    # - (my_games >= 6 AND my_games == opp_games + 1) OR  (6-5, 7-6, 8-7, etc.)
    # - in_tiebreak
    
    p1_can_win_set_this_game_direct = (
        ((p1_games == 5) & (p2_games <= 4)) |  # 5-0, 5-1, 5-2, 5-3, 5-4
        ((p1_games >= 6) & (p1_games == p2_games + 1)) |  # 6-5, 7-6, 8-7, etc.
        is_tiebreak  # Tiebreak always determines set
    )
    p2_can_win_set_this_game_direct = (
        ((p2_games == 5) & (p1_games <= 4)) |
        ((p2_games >= 6) & (p2_games == p1_games + 1)) |
        is_tiebreak
    )
    
    df['is_set_point_p1'] = (p1_game_point * p1_can_win_set_this_game_direct.astype(float))
    df['is_set_point_p2'] = (p2_game_point * p2_can_win_set_this_game_direct.astype(float))
    
    # Match point: winning this point wins the match
    # This is true when: (1) on set point AND (2) one set away from match
    df['is_match_point_p1'] = (df['is_set_point_p1'].values * p1_one_set_away_direct.astype(float))
    df['is_match_point_p2'] = (df['is_set_point_p2'].values * p2_one_set_away_direct.astype(float))
    
    # Break point: winning this point breaks opponent's serve
    # Only matters when opponent is serving
    is_p2_serving = (df.get('point_server', 2) == 2).values
    is_p1_serving = (df.get('point_server', 1) == 1).values
    
    df['is_break_point_p1'] = (p1_game_point * is_p2_serving.astype(float))
    df['is_break_point_p2'] = (p2_game_point * is_p1_serving.astype(float))
    
    return df


def build_new_features(df):
    """
    Build clean feature set with distance-to-victory features and critical point indicators.
    
    Features (31 total):
    - Core scoring (6): P1_points, P2_points, P1_games, P2_games, P1_sets, P2_sets
    - Context (3): point_server, set_number, is_best_of_5
    - Tie-break (4): is_tiebreak, tb_p1_points, tb_p2_points, is_decisive_tiebreak
    - Distance features (8): points/games/sets to win, can_close_match
    - Critical points (6): is_match_point, is_set_point, is_break_point (for both players)
    - Performance (2): P_srv_win, P_srv_lose
    - Game point (2): is_game_point_p1, is_game_point_p2
    """
    
    df = df.copy()
    
    # Ensure necessary features exist
    if 'P1SetsWon' not in df.columns or 'P2SetsWon' not in df.columns:
        df = add_additional_features(df)
    
    # Score parsing
    def score_to_numeric(score_str):
        score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'A': 4}
        score_clean = str(score_str).strip().upper()
        return score_map.get(score_clean, 0)
    
    # 1. Core scoring
    df['P1_points'] = df['P1Score'].apply(score_to_numeric).astype(float) if 'P1Score' in df.columns else 0.0
    df['P2_points'] = df['P2Score'].apply(score_to_numeric).astype(float) if 'P2Score' in df.columns else 0.0
    df['P1_games'] = pd.to_numeric(df.get('P1GamesWon', 0), errors='coerce').fillna(0).astype(float)
    df['P2_games'] = pd.to_numeric(df.get('P2GamesWon', 0), errors='coerce').fillna(0).astype(float)
    df['P1_sets'] = pd.to_numeric(df.get('P1SetsWon', 0), errors='coerce').fillna(0).astype(float)
    df['P2_sets'] = pd.to_numeric(df.get('P2SetsWon', 0), errors='coerce').fillna(0).astype(float)
    
    # 2. Context
    df['point_server'] = pd.to_numeric(df.get(SERVER_COL, 1), errors='coerce').fillna(1).astype(float)
    df['set_number'] = pd.to_numeric(df.get('SetNo', 1), errors='coerce').fillna(1).astype(float)
    
    # is_best_of_5: MUST be constant for entire match
    # For prediction: should already be set by add_additional_features
    # For safety, enforce it here if missing
    if 'is_best_of_5' not in df.columns or df['is_best_of_5'].isna().any():
        # Default: assume best-of-5 for Grand Slams (Wimbledon, etc.)
        # This is SAFE because if match actually goes to 4+ sets, we'll know it's bo5
        # If it ends in 2-3 sets, both bo3 and bo5 would be valid
        print("[build_new_features] WARNING: is_best_of_5 not found, defaulting to best-of-5 (Grand Slam)")
        df['is_best_of_5'] = 1.0
    else:
        # Make sure it's constant per match (use the value from any point in that match)
        df['is_best_of_5'] = df.groupby(MATCH_COL)['is_best_of_5'].transform('first').astype(float)
    
    # 3. Tie-break (with 12-12 rule for final set)
    sets_tied = (df['P1_sets'] == df['P2_sets'])
    sets_played = df['P1_sets'] + df['P2_sets']
    is_final_set = (df['set_number'] >= 3) & sets_tied & (sets_played >= 2)
    
    tb_threshold = np.where(is_final_set, 12, 6)
    is_games_at_threshold = (df['P1_games'] == df['P2_games']) & (df['P1_games'] >= tb_threshold)
    
    if 'P1Score' in df.columns:
        p1_score_raw = df['P1Score'].astype(str).str.strip().str.upper()
    else:
        p1_score_raw = pd.Series('0', index=df.index)
    tennis_scores = {'0', '15', '30', '40', 'AD', 'A'}
    is_numeric_score = ~p1_score_raw.isin(tennis_scores)
    
    df['is_tiebreak'] = (is_games_at_threshold & is_numeric_score).astype(float)
    df['tb_p1_points'] = np.where(df['is_tiebreak'] == 1.0, df['P1_points'], 0.0)
    df['tb_p2_points'] = np.where(df['is_tiebreak'] == 1.0, df['P2_points'], 0.0)
    df['is_decisive_tiebreak'] = (df['is_tiebreak'] == 1.0) & is_final_set
    
    # 4. Distance features (8 new features - KEY INNOVATION)
    df = calculate_distance_features(df)
    
    # 5. Performance (simplified)
    if 'P_srv_win_long' in df.columns and 'P_srv_lose_long' in df.columns:
        df['P_srv_win'] = pd.to_numeric(df['P_srv_win_long'], errors='coerce').fillna(0.5).astype(float)
        df['P_srv_lose'] = pd.to_numeric(df['P_srv_lose_long'], errors='coerce').fillna(0.5).astype(float)
    else:
        df['P_srv_win'] = 0.5
        df['P_srv_lose'] = 0.5
    
    FEATURE_COLUMNS = [
        # Core (6)
        'P1_points', 'P2_points', 'P1_games', 'P2_games', 'P1_sets', 'P2_sets',
        # Context (3)
        'point_server', 'set_number', 'is_best_of_5',
        # Tie-break (4)
        'is_tiebreak', 'tb_p1_points', 'tb_p2_points', 'is_decisive_tiebreak',
        # Distance features (8)
        'p1_points_to_win_game', 'p2_points_to_win_game',
        'p1_games_to_win_set', 'p2_games_to_win_set',
        'p1_sets_to_win_match', 'p2_sets_to_win_match',
        'p1_can_close_match_this_game', 'p2_can_close_match_this_game',
        # Critical points (8): game/set/match/break points for both players
        'is_game_point_p1', 'is_game_point_p2',
        'is_set_point_p1', 'is_set_point_p2',
        'is_match_point_p1', 'is_match_point_p2',
        'is_break_point_p1', 'is_break_point_p2',
        # Performance (2)
        'P_srv_win', 'P_srv_lose',
    ]
    
    # Convert to float
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
    
    X_all = df[FEATURE_COLUMNS].values.astype(float)
    
    # Targets: match, set, game
    if 'p1_wins_match' not in df.columns:
        raise ValueError("p1_wins_match column required")
    
    y_match = df['p1_wins_match'].astype(float).values
    
    # Create set and game targets
    # P1 wins current set if they end up with P1SetsWon > P2SetsWon at end of this set
    # P1 wins current game if they win the next point when at game point
    
    # For simplicity, use proxies:
    # - Set target: did P1 end with more sets at end of match?
    # - Game target: did P1 score increase after this point?
    
    # Better approach: calculate from future data
    df['p1_wins_set'] = 0.0
    df['p1_wins_game'] = 0.0
    
    for match_id, group in df.groupby(MATCH_COL):
        # Set outcome: look at final sets at end of each set
        group = group.copy()
        group['set_change'] = group['P1_sets'].diff().fillna(0)
        # When P1_sets increases, P1 won that set
        mask = group['set_change'] > 0
        if mask.any():
            # Mark all points in that set
            for idx in group[mask].index:
                set_num = group.loc[idx, 'set_number']
                same_set = (group['set_number'] == set_num)
                df.loc[group[same_set].index, 'p1_wins_set'] = 1.0
        
        # Similar for P2
        group['set_change_p2'] = group['P2_sets'].diff().fillna(0)
        mask_p2 = group['set_change_p2'] > 0
        if mask_p2.any():
            for idx in group[mask_p2].index:
                set_num = group.loc[idx, 'set_number']
                same_set = (group['set_number'] == set_num)
                df.loc[group[same_set].index, 'p1_wins_set'] = 0.0
        
        # Game outcome: check if P1GamesWon increases
        group['game_change'] = group['P1_games'].diff().fillna(0)
        df.loc[group[group['game_change'] > 0].index, 'p1_wins_game'] = 1.0
        
        group['game_change_p2'] = group['P2_games'].diff().fillna(0)
        df.loc[group[group['game_change_p2'] > 0].index, 'p1_wins_game'] = 0.0
    
    y_set = df['p1_wins_set'].values
    y_game = df['p1_wins_game'].values
    
    # Sample weights: increase weight for critical points
    # Match points and set points should have much higher weight
    if 'leverage' in df.columns:
        leverage = pd.to_numeric(df['leverage'], errors='coerce').fillna(1.0)
    else:
        leverage = 1.0
    
    # Amplify weights for critical points
    critical_weight = np.ones(len(df))
    critical_weight = np.where(
        (df['is_match_point_p1'] > 0) | (df['is_match_point_p2'] > 0), 
        25.0,  # Match points get 25x weight
        critical_weight
    )
    critical_weight = np.where(
        ((df['is_set_point_p1'] > 0) | (df['is_set_point_p2'] > 0)) & (critical_weight == 1.0),
        8.0,  # Set points get 8x weight (if not already match point)
        critical_weight
    )
    critical_weight = np.where(
        ((df['is_break_point_p1'] > 0) | (df['is_break_point_p2'] > 0)) & (critical_weight == 1.0),
        3.0,  # Break points get 3x weight
        critical_weight
    )
    
    weights = np.clip(leverage * critical_weight, 0.5, 30.0)
    
    # Mask for valid samples
    mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_match)
    
    X = X_all[mask]
    y_match_clean = y_match[mask]
    y_set_clean = y_set[mask]
    y_game_clean = y_game[mask]
    weights_clean = weights[mask] if hasattr(weights, '__len__') else np.ones(X.shape[0])
    
    # Only print statistics once for full dataset (not during incremental builds)
    # Check if this is likely an incremental build (small dataset with many missing features)
    is_incremental_build = len(df) < 50 or df[MATCH_COL].nunique() < len(df) / 50
    
    if not is_incremental_build:
        print(f"[build_new_features] Built {X.shape[0]} samples with {X.shape[1]} features")
        print(f"  Features: 6 core + 3 context + 4 tiebreak + 8 distance + 8 critical + 2 performance = 31 total")
        print(f"  Critical point samples: {int((df['is_match_point_p1'] + df['is_match_point_p2']).sum())} match points")
        print(f"                          {int((df['is_set_point_p1'] + df['is_set_point_p2']).sum())} set points")
        print(f"                          {int((df['is_break_point_p1'] + df['is_break_point_p2']).sum())} break points")
    
    return X, y_match_clean, y_set_clean, y_game_clean, weights_clean, FEATURE_COLUMNS


class MultiTaskTennisNN(nn.Module):
    """
    Neural network with 3 outputs:
    1. P(P1 wins match)
    2. P(P1 wins current set)
    3. P(P1 wins current game)
    
    Forces the network to learn tennis hierarchy.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.4):
        super().__init__()
        
        # Shared trunk
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Three separate heads
        self.head_match = nn.Linear(prev_dim, 1)
        self.head_set = nn.Linear(prev_dim, 1)
        self.head_game = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared(x)
        
        match_logit = self.head_match(shared_features)
        set_logit = self.head_set(shared_features)
        game_logit = self.head_game(shared_features)
        
        return {
            'match': torch.sigmoid(match_logit),
            'set': torch.sigmoid(set_logit),
            'game': torch.sigmoid(game_logit),
        }


def custom_loss(pred_dict, y_match, y_set, y_game, weights, temperature=3.0, 
                is_match_point_p1=None, is_match_point_p2=None):
    """
    Custom loss with four components:
    1. Multi-task BCE for match, set, game
    2. Consistency penalty (set outcome should influence match outcome)
    3. Temperature scaling for calibration
    4. Match point penalty (force high prob when player has match point)
    """
    
    # Apply temperature scaling with numerical stability
    eps = 1e-7
    pred_match_clipped = torch.clamp(pred_dict['match'], eps, 1 - eps)
    
    # Temperature scaling: p_calibrated = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
    p_temp = torch.pow(pred_match_clipped, 1.0/temperature)
    one_minus_p_temp = torch.pow(1 - pred_match_clipped, 1.0/temperature)
    pred_match_cal = p_temp / (p_temp + one_minus_p_temp + eps)
    
    # BCE losses
    bce = nn.BCELoss(reduction='none')
    
    loss_match = bce(pred_match_cal.squeeze(), y_match)
    loss_set = bce(torch.clamp(pred_dict['set'], eps, 1-eps).squeeze(), y_set)
    loss_game = bce(torch.clamp(pred_dict['game'], eps, 1-eps).squeeze(), y_game)
    
    # Weighted average
    total_loss = (
        1.0 * (loss_match * weights).mean() +
        0.5 * (loss_set * weights).mean() +
        0.3 * (loss_game * weights).mean()
    )
    
    # Consistency penalty: if set prob is high, match prob should be high
    # Only applies when P1 is ahead in sets
    # This is a soft constraint to encourage logical consistency
    consistency_loss = torch.relu(pred_dict['set'] - pred_dict['match'] - 0.2).mean()
    
    total_loss = total_loss + 0.1 * consistency_loss
    
    # Match point penalty: enforce tennis rules
    # If P1 has match point, P1 probability should be > 0.85
    # If P2 has match point, P1 probability should be < 0.15
    match_point_penalty = torch.tensor(0.0, device=pred_match_cal.device)
    
    if is_match_point_p1 is not None and is_match_point_p2 is not None:
        # P1 match points: penalize if pred < 0.85
        p1_mp_mask = is_match_point_p1 > 0.5
        if p1_mp_mask.any():
            p1_mp_penalty = torch.relu(0.85 - pred_match_cal[p1_mp_mask]).mean()
            match_point_penalty = match_point_penalty + p1_mp_penalty
        
        # P2 match points: penalize if pred > 0.15
        p2_mp_mask = is_match_point_p2 > 0.5
        if p2_mp_mask.any():
            p2_mp_penalty = torch.relu(pred_match_cal[p2_mp_mask] - 0.15).mean()
            match_point_penalty = match_point_penalty + p2_mp_penalty
    
    total_loss = total_loss + 0.5 * match_point_penalty
    
    return total_loss, {
        'match': loss_match.mean().item(),
        'set': loss_set.mean().item(),
        'game': loss_game.mean().item(),
        'consistency': consistency_loss.item(),
        'match_point_penalty': match_point_penalty.item(),
    }


def train_new_model(file_paths, model_out, gender="male", 
                    hidden_dims=[128, 64], dropout=0.4, temperature=8.0,
                    epochs=200, batch_size=1024, learning_rate=0.001,
                    early_stopping_patience=40):
    """
    Train multi-task neural network with distance features.
    """
    
    print("="*60)
    print("TRAINING NEW MULTI-TASK NEURAL NETWORK")
    print("="*60)
    print(f"Architecture: {hidden_dims}")
    print(f"Dropout: {dropout}, Temperature: {temperature}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_points_multiple(file_paths)
    
    # Gender filter (same logic as model_nn.py)
    if gender != "both":
        print(f"[train_new_model] Gender filter: {gender}")
        extracted = df['match_id'].str.extract(r'-(\d+)$')[0]
        valid_mask = extracted.notna()
        
        if valid_mask.any():
            df_temp = df[valid_mask].copy()
            df_temp['match_num'] = extracted[valid_mask].astype(int)
            
            if gender == "male":
                df_temp = df_temp[df_temp['match_num'] < 2000].copy()
            else:
                df_temp = df_temp[df_temp['match_num'] >= 2000].copy()
            
            df_temp = df_temp.drop(columns=['match_num'])
            df = df_temp
    
    # Exclude test match 1701
    df = df[~df['match_id'].astype(str).str.contains('1701', na=False)]
    
    df = add_additional_features(df)
    df = add_match_labels(df)
    
    print(f"Loaded {len(df)} points from {df[MATCH_COL].nunique()} matches")
    
    # Build features
    print("\nBuilding features...")
    X, y_match, y_set, y_game, weights, feature_names = build_new_features(df)
    
    # Normalize
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/val/test split in one step
    indices = np.arange(len(X_scaled))
    
    # First split: 85% train, 15% temp (will split into val and test)
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=(y_match > 0.5).astype(int)
    )
    
    # Second split: 50-50 split of temp into val and test
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42
    )
    
    # Apply splits to all arrays
    X_train, X_val, X_test = X_scaled[idx_train], X_scaled[idx_val], X_scaled[idx_test]
    y_match_train, y_match_val, y_match_test = y_match[idx_train], y_match[idx_val], y_match[idx_test]
    y_set_train, y_set_val, y_set_test = y_set[idx_train], y_set[idx_val], y_set[idx_test]
    y_game_train, y_game_val, y_game_test = y_game[idx_train], y_game[idx_val], y_game[idx_test]
    weights_train, weights_val, weights_test = weights[idx_train], weights[idx_val], weights[idx_test]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_match_train),
        torch.FloatTensor(y_set_train),
        torch.FloatTensor(y_game_train),
        torch.FloatTensor(weights_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskTennisNN(X_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y_match, batch_y_set, batch_y_game, batch_w in train_loader:
            batch_X = batch_X.to(device)
            batch_y_match = batch_y_match.to(device)
            batch_y_set = batch_y_set.to(device)
            batch_y_game = batch_y_game.to(device)
            batch_w = batch_w.to(device)
            
            # Extract match point features (indices 23, 24 in FEATURE_COLUMNS)
            # is_match_point_p1 is at index 23, is_match_point_p2 at index 24
            is_mp_p1 = batch_X[:, 23]
            is_mp_p2 = batch_X[:, 24]
            
            optimizer.zero_grad()
            pred_dict = model(batch_X)
            
            loss, loss_components = custom_loss(
                pred_dict, batch_y_match, batch_y_set, batch_y_game,
                batch_w, temperature, is_mp_p1, is_mp_p2
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_match_val_t = torch.FloatTensor(y_match_val).to(device)
            y_set_val_t = torch.FloatTensor(y_set_val).to(device)
            y_game_val_t = torch.FloatTensor(y_game_val).to(device)
            weights_val_t = torch.FloatTensor(weights_val).to(device)
            
            # Extract match point features for validation
            is_mp_p1_val = X_val_t[:, 23]
            is_mp_p2_val = X_val_t[:, 24]
            
            pred_val = model(X_val_t)
            val_loss, val_components = custom_loss(
                pred_val, y_match_val_t, y_set_val_t, y_game_val_t,
                weights_val_t, temperature, is_mp_p1_val, is_mp_p2_val
            )
            val_loss = val_loss.item()
            
            # Accuracy
            pred_match_np = pred_val['match'].cpu().numpy().squeeze()
            val_acc = ((pred_match_np > 0.5) == y_match_val).mean()
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.3f}")
            print(f"  Val components: match={val_components['match']:.4f}, "
                  f"set={val_components['set']:.4f}, game={val_components['game']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"\nLoaded best model with Val Loss={best_val_loss:.4f}")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_match_test_t = torch.FloatTensor(y_match_test).to(device)
        
        pred_test = model(X_test_t)
        pred_match_test = pred_test['match'].cpu().numpy().squeeze()
        
        test_acc = ((pred_match_test > 0.5) == y_match_test).mean()
        
        # ROC AUC
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(y_match_test, pred_match_test)
    
    print(f"\nTest accuracy: {test_acc:.3f}")
    print(f"Test ROC AUC:  {test_auc:.3f}")
    
    # Save model
    model_data = {
        'model_type': 'multi_task_nn',
        'state_dict': {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
        'scaler_center': scaler.center_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'feature_names': feature_names,
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'temperature': temperature,
        'input_dim': X_train.shape[1],
        'test_accuracy': test_acc,
        'test_auc': test_auc,
    }
    
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    with open(model_out, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"\nModel saved to: {model_out}")
    
    return model, scaler, feature_names


if __name__ == "__main__":
    import sys
    
    # Example usage
    files = [
        "data/2015-wimbledon-points.csv",
        "data/2016-wimbledon-points.csv",
        "data/2017-wimbledon-points.csv",
        "data/2018-wimbledon-points.csv",
        "data/2019-wimbledon-points.csv",
    ]
    
    model, scaler, features = train_new_model(
        files,
        "models/new_nn_multitask.json",
        gender="male",
        hidden_dims=[128, 64],
        dropout=0.4,
        temperature=3.0,
        epochs=200,
        batch_size=1024,
    )
