import numpy as np
import pandas as pd
import os

from scripts.features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    add_additional_features,
    build_dataset,
)
from scripts.model import _default_model
from scripts.config import load_config


def get_config_params(config_path=None):
    """Load configuration parameters from config file."""
    if config_path is None:
        # Try to find config.json in the configs directory
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.json")
    
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 0.3))
    
    return long_window, short_window, alpha


def make_realistic_df():
    """
    Build a more realistic synthetic dataset with:
      - Multiple matches (4 matches)
      - Multiple games per match
      - Realistic point progression
      - All necessary columns for feature engineering
    """
    rows = []
    match_id = 0

    # Create 4 matches: 2 won by P1, 2 won by P2
    for match_idx in range(4):
        match_id += 1
        match_winner = 1 if match_idx % 2 == 0 else 2
        
        # Create 3 games per match
        for game_no in range(1, 4):
            game_winner = match_winner if game_no <= 2 else (3 - match_winner)
            
            # Create 8-12 points per game
            num_points = 8 + (game_no * match_idx) % 5
            for point_idx in range(num_points):
                server = 1 if (game_no + point_idx) % 2 == 0 else 2
                winner = game_winner if point_idx < num_points // 2 else (3 - game_winner)
                
                # Generate tennis scores
                p1_score_val = min(point_idx % 5, 3)
                p2_score_val = min((num_points - point_idx) % 5, 3)
                score_map = {0: "0", 1: "15", 2: "30", 3: "40"}
                p1_score = score_map[p1_score_val]
                p2_score = score_map[p2_score_val]
                
                rows.append({
                    "match_id": match_id,
                    "SetNo": 1,
                    "GameNo": game_no,
                    "PointNumber": point_idx + 1,
                    "PointServer": server,
                    "PointWinner": winner,
                    "GameWinner": game_winner,
                    "P1Score": p1_score,
                    "P2Score": p2_score,
                    "P1GamesWon": game_no - 1 if match_winner == 1 else game_no - 2,
                    "P2GamesWon": game_no - 1 if match_winner == 2 else game_no - 2,
                    "P1Momentum": 0.5 + 0.1 * (point_idx % 5),
                    "P2Momentum": 0.5 - 0.1 * (point_idx % 5),
                })

    return pd.DataFrame(rows)


def test_feature_pipeline_comprehensive():
    """Test the complete feature engineering pipeline using config file."""
    # Load parameters from config
    long_window, short_window, alpha = get_config_params()
    print(f"📋 Config: long_window={long_window}, short_window={short_window}, alpha={alpha}")
    
    df = make_realistic_df()
    
    # Test 1: Match labels
    df = add_match_labels(df)
    assert "p1_wins_match" in df.columns
    assert "match_winner" in df.columns
    assert df["p1_wins_match"].nunique() == 2  # Should have both 0 and 1
    print(f"✓ Match labels: {df['p1_wins_match'].sum()} P1 wins, {(1-df['p1_wins_match']).sum()} P2 wins")
    
    # Test 2: Rolling serve/return features (using config parameters)
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    assert "P_srv_win_long" in df.columns
    assert "P_srv_lose_long" in df.columns
    assert "P_srv_win_short" in df.columns
    assert "P_srv_lose_short" in df.columns
    assert not df["P_srv_win_long"].isna().all()
    print(f"✓ Rolling features: long_window={long_window}, short_window={short_window}")
    
    # Test 3: Leverage and momentum (using config alpha)
    df = add_leverage_and_momentum(df, alpha=alpha)
    assert "leverage" in df.columns
    assert "momentum" in df.columns
    assert not df["momentum"].isna().all()
    print(f"✓ Leverage and momentum: alpha={alpha}")
    
    # Test 4: Additional features
    df = add_additional_features(df)
    assert "Momentum_Diff" in df.columns
    assert "Score_Diff" in df.columns
    assert "Game_Diff" in df.columns
    assert "SrvScr" in df.columns
    assert "RcvScr" in df.columns

    # New normalization/clipping checks
    assert df["Score_Diff"].min() >= -2.0 and df["Score_Diff"].max() <= 2.0, "Score_Diff should be clipped to [-2, 2]"
    assert df["Game_Diff"].min() >= -3.0 and df["Game_Diff"].max() <= 3.0, "Game_Diff should be clipped to [-3, 3]"
    for col in ["SetNo", "GameNo", "PointNumber"]:
        assert df[col].between(0.0, 1.0).all(), f"{col} should be normalized between 0 and 1"

    print(f"✓ Additional features: clipping + normalization verified")
    
    # Test 5: Build dataset
    X, y, mask, sample_weights = build_dataset(df)
    assert len(sample_weights) == len(y), "Sample weights should match y length"
    assert X.shape[0] == y.shape[0], "X and y should have same number of rows"
    assert X.shape[1] == 15, f"Expected 15 features, got {X.shape[1]}"
    assert X.shape[0] > 0, "Should have at least some valid samples"
    assert y.sum() > 0 and y.sum() < len(y), "Should have both positive and negative classes"
    assert not np.isnan(X).any(), "X should not contain NaN values"
    print(f"✓ Dataset: X shape={X.shape}, y distribution: {y.sum()}/{len(y)} P1 wins")
    
    # Test 6: Model training
    model = _default_model()
    model.fit(X, y)
    print(f"✓ Model training completed")
    
    # Test 7: Model prediction
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    assert y_pred.shape == y.shape, "Predictions should match target shape"
    assert y_proba.shape == (len(y), 2), "Probabilities should be (n_samples, 2)"
    assert np.all((y_proba >= 0) & (y_proba <= 1)), "Probabilities should be between 0 and 1"
    assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities should sum to 1"
    print(f"✓ Model predictions: accuracy={np.mean(y_pred == y):.3f}")
    
    # Test 8: Feature importance
    feature_importance = model.feature_importances_
    assert len(feature_importance) == 15, "Should have importance for all 15 features"
    assert np.all(feature_importance >= 0), "Feature importance should be non-negative"
    print(f"✓ Feature importance computed")
    
    print("\n✅ All tests passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    df = make_realistic_df()
    
    # Test with very small windows
    df_copy = df.copy()
    df_copy = add_match_labels(df_copy)
    df_copy = add_rolling_serve_return_features(df_copy, long_window=2, short_window=1)
    df_copy = add_leverage_and_momentum(df_copy, alpha=0.5)
    df_copy = add_additional_features(df_copy)
    X, y, _, sample_weights = build_dataset(df_copy)
    assert X.shape[0] > 0, "Should work with small windows"
    print(f"✓ Edge case: small windows (2, 1) - {X.shape[0]} samples")
    
    # Test with different alpha values
    for alpha_val in [0.1, 0.3, 0.5, 0.9]:
        df_copy = df.copy()
        df_copy = add_match_labels(df_copy)
        df_copy = add_rolling_serve_return_features(df_copy, long_window=10, short_window=3)
        df_copy = add_leverage_and_momentum(df_copy, alpha=alpha_val)
        df_copy = add_additional_features(df_copy)
        X, y, _, sample_weights = build_dataset(df_copy)
        assert not np.isnan(X).any(), f"Should not have NaN with alpha={alpha_val}"
        assert len(sample_weights) == len(y), "Sample weights should match y"
    print(f"✓ Edge case: tested alpha values [0.1, 0.3, 0.5, 0.9]")
    
    print("\n✅ Edge case tests passed!")
