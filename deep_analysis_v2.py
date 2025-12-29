"""
Deep analysis of the original model (v2) predictions.
Compare predictions with actual match events to find WHERE it fails.
"""
import pandas as pd
import numpy as np
import torch
from scripts.pretrain_tennis_rules import TennisRulesNet

print('='*80)
print('DEEP ANALYSIS: ORIGINAL MODEL vs ACTUAL MATCH EVENTS')
print('='*80)

# Load original predictions
df_pred = pd.read_csv('plots/match_2019-wimbledon-1701_probabilities.csv')

# Load actual match data with events
from scripts.data_loader import load_points_multiple
df_raw = load_points_multiple(['data/2019-wimbledon-points.csv'])
df_raw = df_raw[df_raw['match_id'] == '2019-wimbledon-1701'].copy()

print(f"\nMatch: 2019-wimbledon-1701 (Wimbledon 2019 Final)")
print(f"Total points: {len(df_raw)}")

# Parse score information
print("\n=== MATCH STRUCTURE ===")
if 'SetNo' in df_raw.columns:
    print(f"Sets played: {df_raw['SetNo'].max()}")
    for set_num in sorted(df_raw['SetNo'].unique()):
        set_data = df_raw[df_raw['SetNo'] == set_num]
        if 'SetWinner' in set_data.columns:
            winner = set_data['SetWinner'].iloc[-1]
            print(f"  Set {int(set_num)}: Winner = Player {int(winner)}")

# Analyze predictions at key moments
print("\n=== PREDICTIONS AT KEY MOMENTS ===")

# Start of match (should be ~0.5)
print("\n1. START OF MATCH (first 5 points):")
print(f"   Expected: ~0.50 (no info yet)")
print(f"   Actual: {df_pred['prob_p1'].iloc[:5].mean():.3f}")
print(f"   Range: [{df_pred['prob_p1'].iloc[:5].min():.3f}, {df_pred['prob_p1'].iloc[:5].max():.3f}]")

# After first set
if 'SetNo' in df_raw.columns:
    set1_end = df_raw[df_raw['SetNo'] == 1].index[-1]
    if set1_end < len(df_pred):
        set1_winner = df_raw.loc[set1_end, 'SetWinner'] if 'SetWinner' in df_raw.columns else None
        print(f"\n2. AFTER SET 1 (point {set1_end}):")
        if set1_winner:
            print(f"   Set winner: Player {int(set1_winner)}")
            expected = 0.65 if set1_winner == 1 else 0.35
            print(f"   Expected: ~{expected:.2f} (set winner has advantage)")
        print(f"   Actual: {df_pred.iloc[set1_end+1]['prob_p1']:.3f}")
        
    # After set 2
    set2_end = df_raw[df_raw['SetNo'] == 2].index[-1] if 2 in df_raw['SetNo'].values else None
    if set2_end and set2_end < len(df_pred):
        set2_winner = df_raw.loc[set2_end, 'SetWinner'] if 'SetWinner' in df_raw.columns else None
        print(f"\n3. AFTER SET 2 (point {set2_end}):")
        if set2_winner:
            print(f"   Set winner: Player {int(set2_winner)}")
            # Calculate set score
            p1_sets = (df_raw.loc[:set2_end, 'SetWinner'] == 1).sum()
            p2_sets = (df_raw.loc[:set2_end, 'SetWinner'] == 2).sum()
            print(f"   Set score: {int(p1_sets)}-{int(p2_sets)}")
            if p1_sets > p2_sets:
                expected = 0.75
            elif p2_sets > p1_sets:
                expected = 0.25
            else:
                expected = 0.50
            print(f"   Expected: ~{expected:.2f}")
        print(f"   Actual: {df_pred.iloc[set2_end+1]['prob_p1']:.3f}")

# End of match
print(f"\n4. END OF MATCH (last 5 points):")
if 'PointWinner' in df_raw.columns:
    final_winner = df_raw['PointWinner'].mode()[0]  # Most common winner (crude estimate)
    print(f"   Match winner (estimated): Player {int(final_winner)}")
    expected = 0.95 if final_winner == 1 else 0.05
    print(f"   Expected: ~{expected:.2f}")
print(f"   Actual: {df_pred['prob_p1'].iloc[-5:].mean():.3f}")
print(f"   Range: [{df_pred['prob_p1'].iloc[-5:].min():.3f}, {df_pred['prob_p1'].iloc[-5:].max():.3f}]")

# Analyze correlation with score
print("\n=== CORRELATION WITH SCORE ===")
if 'P1Score' in df_raw.columns and 'P2Score' in df_raw.columns:
    # When P1 is ahead in current game
    df_raw['score_advantage'] = 0
    # Simple heuristic: higher score = advantage
    score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}
    try:
        df_raw['p1_score_val'] = df_raw['P1Score'].map(score_map)
        df_raw['p2_score_val'] = df_raw['P2Score'].map(score_map)
        
        p1_ahead = df_raw['p1_score_val'] > df_raw['p2_score_val']
        p2_ahead = df_raw['p2_score_val'] > df_raw['p1_score_val']
        
        print(f"\nWhen P1 ahead in score:")
        print(f"  Mean probability: {df_pred.loc[p1_ahead, 'prob_p1'].mean():.3f}")
        print(f"  Expected: >0.50")
        
        print(f"\nWhen P2 ahead in score:")
        print(f"  Mean probability: {df_pred.loc[p2_ahead, 'prob_p1'].mean():.3f}")
        print(f"  Expected: <0.50")
    except:
        print("  (Could not parse scores)")

print("\n=== DIAGNOSIS ===")
issues = []

# Check if model responds to set wins
if 'SetNo' in df_raw.columns and len(df_raw['SetNo'].unique()) > 1:
    set1_end_prob = df_pred.iloc[set1_end]['prob_p1'] if set1_end < len(df_pred) else 0.5
    if abs(set1_end_prob - 0.5) < 0.1:
        issues.append("Model doesn't respond to set wins (prob stays ~0.5)")

# Check if model is confident at end
if df_pred['prob_p1'].iloc[-10:].mean() < 0.7 and df_pred['prob_p1'].iloc[-10:].mean() > 0.3:
    issues.append("Model not confident at end of match (should be >0.8 or <0.2)")

# Check if model has extreme values when it shouldn't
if (df_pred['prob_p1'].iloc[:20] > 0.7).any() or (df_pred['prob_p1'].iloc[:20] < 0.3).any():
    issues.append("Model too confident at start of match")

if issues:
    print("\nISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\nNo obvious issues found. Model seems reasonable.")

print("\n" + "="*80)
