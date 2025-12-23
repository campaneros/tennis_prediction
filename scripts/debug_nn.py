"""
Script di debug per capire perché la NN predice male.
Controlla target, features, e predizioni su esempi semplici.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.data_loader import load_points_multiple
from scripts.features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_additional_features,
    add_leverage_and_momentum,
    build_dataset,
)


def debug_features():
    """Stampa esempi di features per verificare che siano corrette."""

    repo_root = Path(__file__).resolve().parent.parent

    # Carica un file
    data_file = repo_root / "data/2019-wimbledon-points.csv"
    df = load_points_multiple([str(data_file)])

    # Filtra un match
    match_id = '2019-wimbledon-1701'
    df = df[df['match_id'] == match_id].copy()
    
    print(f"\n=== DEBUG MATCH {match_id} ===")
    print(f"Total points: {len(df)}")
    
    # Add labels
    df = add_match_labels(df)
    
    print(f"\nMatch winner: {df['match_winner'].iloc[0]}")
    print(f"p1_wins_match values: {df['p1_wins_match'].unique()}")
    print(f"Target distribution: P1 wins = {df['p1_wins_match'].mean():.2f}")
    
    # Build features
    df = add_rolling_serve_return_features(df, long_window=20, short_window=5)
    
    print(f"\n=== After rolling features ===")
    print(f"P_srv_win_long NaN: {df['P_srv_win_long'].isna().sum()}/{len(df)}")
    print(f"P_srv_lose_long NaN: {df['P_srv_lose_long'].isna().sum()}/{len(df)}")
    
    df = add_additional_features(df)
    
    print(f"\n=== After additional features ===")
    if 'point_importance' in df.columns:
        print(f"point_importance NaN: {df['point_importance'].isna().sum()}/{len(df)}")
        print(f"point_importance range: {df['point_importance'].min():.2f} - {df['point_importance'].max():.2f}")
    else:
        print("point_importance NOT CREATED!")
    
    print(f"\n=== Calling add_leverage_and_momentum ===")
    try:
        df = add_leverage_and_momentum(df, alpha=1.2)
        print(f"SUCCESS!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n=== After leverage and momentum ===")
    if 'leverage' in df.columns:
        print(f"leverage NaN: {df['leverage'].isna().sum()}/{len(df)}")
        print(f"leverage range: {df['leverage'].min():.3f} - {df['leverage'].max():.3f}")
    if 'momentum' in df.columns:
        print(f"momentum NaN: {df['momentum'].isna().sum()}/{len(df)}")
        if df['momentum'].notna().any():
            print(f"momentum range: {df['momentum'].min():.3f} - {df['momentum'].max():.3f}")
    
    # Check for NaN before build_dataset
    print(f"\n=== NaN CHECK BEFORE build_dataset ===")
    for col in ['P_srv_win_long', 'P_srv_lose_long', 'momentum', 'Game_Diff', 'P1SetsWon']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            print(f"{col:20s}: {nan_count}/{len(df)} NaN values")
    
    X, y, mask, weights, _ = build_dataset(df)
    df_valid = df[mask].copy()
    
    print(f"\nValid points: {len(df_valid)}")
    print(f"Target mean (y): {y.mean():.3f} (should be close to p1_wins_match)")
    
    # Controlla alcune situazioni chiave
    print("\n=== SITUAZIONI CHIAVE ===")
    
    # Punto quando P1 è avanti nei set
    idx_p1_ahead = df_valid[df_valid['P1SetsWon'] > df_valid['P2SetsWon']].index
    if len(idx_p1_ahead) > 0:
        sample_idx = idx_p1_ahead[0]
        print(f"\nPunto quando P1 è avanti nei set (index {sample_idx}):")
        print(f"  P1SetsWon: {df_valid.loc[sample_idx, 'P1SetsWon']}")
        print(f"  P2SetsWon: {df_valid.loc[sample_idx, 'P2SetsWon']}")
        print(f"  Game_Diff: {df_valid.loc[sample_idx, 'Game_Diff']}")
        print(f"  Target (p1_wins_match): {df_valid.loc[sample_idx, 'p1_wins_match']}")
        print(f"  Prediction should be > 0.5")
    
    # Punto quando P2 è avanti nei set
    idx_p2_ahead = df_valid[df_valid['P2SetsWon'] > df_valid['P1SetsWon']].index
    if len(idx_p2_ahead) > 0:
        sample_idx = idx_p2_ahead[0]
        print(f"\nPunto quando P2 è avanti nei set (index {sample_idx}):")
        print(f"  P1SetsWon: {df_valid.loc[sample_idx, 'P1SetsWon']}")
        print(f"  P2SetsWon: {df_valid.loc[sample_idx, 'P2SetsWon']}")
        print(f"  Game_Diff: {df_valid.loc[sample_idx, 'Game_Diff']}")
        print(f"  Target (p1_wins_match): {df_valid.loc[sample_idx, 'p1_wins_match']}")
        print(f"  Prediction should be < 0.5")
    
    # Stampa statistiche delle features principali
    print("\n=== STATISTICHE FEATURES ===")
    key_features = ['Game_Diff', 'Score_Diff', 'P1SetsWon', 'P2SetsWon', 
                    'SetsWonDiff', 'SetWinProbPrior', 'CurrentSetGamesDiff']
    
    for feat in key_features:
        if feat in df_valid.columns:
            print(f"{feat:20s}: mean={df_valid[feat].mean():7.3f}, "
                  f"std={df_valid[feat].std():7.3f}, "
                  f"min={df_valid[feat].min():7.3f}, "
                  f"max={df_valid[feat].max():7.3f}")
    
    # Verifica correlazione con target
    print("\n=== CORRELAZIONE CON TARGET ===")
    X_df = pd.DataFrame(X, columns=range(X.shape[1]))
    X_df['target'] = y
    
    for feat in key_features:
        if feat in df_valid.columns:
            feat_idx = list(df_valid.columns).index(feat) if feat in df_valid.columns else None
            if feat_idx is not None and feat_idx < X.shape[1]:
                corr = np.corrcoef(X[:, feat_idx], y)[0, 1]
                print(f"{feat:20s}: correlation = {corr:7.3f}")


if __name__ == "__main__":
    debug_features()
