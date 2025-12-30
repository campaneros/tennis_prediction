#!/usr/bin/env python3
"""
Analizza il bias P1 nel dataset di training e testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from scripts.data_loader import load_points_multiple
    from scripts.features import add_match_labels
except ImportError:
    from data_loader import load_points_multiple
    from features import add_match_labels

import pandas as pd
import numpy as np
import glob

print("=" * 80)
print("ANALISI BIAS P1 NEL DATASET")
print("=" * 80)
print()

# Load all training data
print("[1/2] Loading all Grand Slam data...")
files = sorted(glob.glob("data/*-matches.csv"))
print(f"  Found {len(files)} match files")

# Sample: load only matches file to get winners
all_matches = []
for f in files:
    try:
        df = pd.read_csv(f)
        if 'match_id' in df.columns and 'winner' in df.columns:
            all_matches.append(df[['match_id', 'winner']].drop_duplicates())
    except:
        pass

if all_matches:
    matches_df = pd.concat(all_matches, ignore_index=True)
    print(f"  Loaded {len(matches_df)} unique matches")
    
    # In our format, winner is 1 or 2
    # P1 wins if winner == 1
    if 'winner' in matches_df.columns:
        p1_wins = (matches_df['winner'] == 1).sum()
        p2_wins = (matches_df['winner'] == 2).sum()
        total = p1_wins + p2_wins
        
        if total > 0:
            p1_win_rate = p1_wins / total
            print()
            print("DATASET STATISTICS:")
            print(f"  P1 wins: {p1_wins}/{total} ({p1_win_rate:.1%})")
            print(f"  P2 wins: {p2_wins}/{total} ({100-p1_win_rate*100:.1%})")
            print()
            
            if p1_win_rate > 0.55:
                print(f"⚠ STRONG P1 BIAS: P1 wins {p1_win_rate:.1%} of matches!")
                print("  This explains why model predicts >0.80 for P1")
            elif p1_win_rate < 0.45:
                print(f"⚠ STRONG P2 BIAS: P2 wins {100-p1_win_rate*100:.1%} of matches!")
            else:
                print(f"✓ Dataset is balanced: {p1_win_rate:.1%} P1 wins")

# Check specific match
print()
print("=" * 80)
print("[2/2] Checking Wimbledon 2019 Final")
print("=" * 80)
print()

try:
    from scripts.data_loader import load_points_single
    df_test = load_points_single("data/2019-wimbledon-points.csv")
    
    # Find match 1701
    match_1701 = df_test[df_test['match_id'] == '2019-wimbledon-1701']
    
    if len(match_1701) > 0:
        # Check who is P1 and P2
        # Typically, player names might be in server info or we need to check match metadata
        print(f"Match 2019-wimbledon-1701:")
        print(f"  Points: {len(match_1701)}")
        
        # Check final score
        last_row = match_1701.iloc[-1]
        p1_sets = last_row.get('P1SetsWon', last_row.get('P1_sets', 0))
        p2_sets = last_row.get('P2SetsWon', last_row.get('P2_sets', 0))
        
        print(f"  Final set score: P1={p1_sets}, P2={p2_sets}")
        
        if p1_sets > p2_sets:
            print(f"  P1 won the match")
        else:
            print(f"  P2 won the match")
        
        print()
        print("Since this is Djokovic (P1) vs Federer (P2):")
        print("  P1 (Djokovic) won 3-2")
        print()
        print("If the model was trained on data where P1 represents")
        print("the 'first listed player' or 'higher seed', it might have")
        print("learned that P1 wins more often.")
        
except Exception as e:
    print(f"Error loading test match: {e}")

print()
print("=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print()
print("1. Check if P1/P2 assignment is based on:")
print("   - Seeding (higher seed = P1) → creates bias")
print("   - Ranking (higher rank = P1) → creates bias")  
print("   - Random or alphabetical → no bias")
print()
print("2. If there's P1 bias in training data:")
print("   - Data augmentation: swap P1/P2 to balance")
print("   - Remove ranking/seeding features")
print("   - Use only match state features")
print()
print("3. The high baseline probability (0.80) suggests the model")
print("   learned 'P1 usually wins' rather than understanding")
print("   the match dynamics.")
print("=" * 80)
