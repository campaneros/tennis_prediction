#!/usr/bin/env python3
"""
Test model v4 on 2019 Wimbledon final and compare with v2.
This will verify if the fixes resolved the saturation problem.
"""
import pandas as pd
import numpy as np
import sys

print('='*80)
print('TESTING MODEL V4 vs V2')
print('Match: 2019 Wimbledon Final (Djokovic vs Federer)')
print('='*80)

# Check if v4 exists
import os
if not os.path.exists('models/complete_model_v4.pth'):
    print("\n❌ Model v4 not found! Run fine-tuning first.")
    sys.exit(1)

print("\n[1/3] Generating predictions with model v2...")
os.system('venv/bin/python tennisctl.py predict --model ./models/complete_model_v2.pth --match-id 2019-wimbledon-1701 --files data/2019-wimbledon-points.csv --plot-dir plots/v2 --point-by-point > /dev/null 2>&1')

print("\n[2/3] Generating predictions with model v4...")
os.system('venv/bin/python tennisctl.py predict --model ./models/complete_model_v4.pth --match-id 2019-wimbledon-1701 --files data/2019-wimbledon-points.csv --plot-dir plots/v4 --point-by-point > /dev/null 2>&1')

# Load predictions
print("\n[3/3] Comparing results...")
print("\n" + "="*80)

try:
    df_v2 = pd.read_csv('plots/v2/match_2019-wimbledon-1701_probabilities.csv')
    df_v4 = pd.read_csv('plots/v4/match_2019-wimbledon-1701_probabilities.csv')
    
    # Load raw match data to get set scores
    df_raw = pd.read_csv('data/2019-wimbledon-points.csv')
    match_points = df_raw[df_raw['match_id'] == '2019-wimbledon-1701'].copy()
    
    # Calculate set scores
    match_points['P1_sets_won'] = 0
    match_points['P2_sets_won'] = 0
    
    for set_num in sorted(match_points['SetNo'].unique()):
        set_data = match_points[match_points['SetNo'] == set_num]
        last_point = set_data.iloc[-1]
        
        p1_games = last_point['P1GamesWon']
        p2_games = last_point['P2GamesWon']
        set_winner = 1 if p1_games > p2_games else 2
        
        match_points.loc[match_points['SetNo'] > set_num, 'P1_sets_won'] += (1 if set_winner == 1 else 0)
        match_points.loc[match_points['SetNo'] > set_num, 'P2_sets_won'] += (1 if set_winner == 2 else 0)
    
    print("COMPARISON: Model v2 vs v4")
    print("="*80)
    
    # Start
    print(f"\n1. Match START (first 5 points):")
    print(f"   v2: {df_v2['prob_p1'].iloc[:5].mean():.3f}  (expected ~0.50)")
    print(f"   v4: {df_v4['prob_p1'].iloc[:5].mean():.3f}  (expected ~0.50)")
    
    # After each set
    for set_num in range(1, 6):
        set_data = match_points[match_points['SetNo'] == set_num]
        if len(set_data) == 0:
            continue
        
        set_end_idx = set_data.index[-1]
        
        if set_end_idx < len(df_v2) - 1:
            prob_v2 = df_v2.iloc[set_end_idx + 1]['prob_p1']
            prob_v4 = df_v4.iloc[set_end_idx + 1]['prob_p1']
            
            p1_sets = int(match_points.loc[set_end_idx, 'P1_sets_won'])
            p2_sets = int(match_points.loc[set_end_idx, 'P2_sets_won'])
            
            if match_points.loc[set_end_idx, 'P1GamesWon'] > match_points.loc[set_end_idx, 'P2GamesWon']:
                p1_sets += 1
            else:
                p2_sets += 1
            
            print(f"\n{set_num+1}. After SET {set_num} (Djokovic {p1_sets}-{p2_sets} Federer):")
            print(f"   v2: {prob_v2:.3f}")
            print(f"   v4: {prob_v4:.3f}")
            
            # Check if v4 is better
            if p1_sets > p2_sets:
                expected = 0.65
                if abs(prob_v4 - expected) < abs(prob_v2 - expected):
                    print(f"   ✓ v4 is CLOSER to expected ~{expected:.2f}")
            elif p2_sets > p1_sets:
                expected = 0.35
                if abs(prob_v4 - expected) < abs(prob_v2 - expected):
                    print(f"   ✓ v4 is CLOSER to expected ~{expected:.2f}")
            elif p1_sets == p2_sets:
                expected = 0.50
                if abs(prob_v4 - expected) < abs(prob_v2 - expected):
                    print(f"   ✓ v4 is CLOSER to expected ~{expected:.2f}")
    
    # End
    print(f"\n6. Match END (Djokovic won 3-2):")
    print(f"   v2: {df_v2.iloc[-1]['prob_p1']:.3f}  (expected ~0.95)")
    print(f"   v4: {df_v4.iloc[-1]['prob_p1']:.3f}  (expected ~0.95)")
    
    if abs(df_v4.iloc[-1]['prob_p1'] - 0.95) < abs(df_v2.iloc[-1]['prob_p1'] - 0.95):
        print(f"   ✓ v4 is CLOSER to expected 0.95")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"\nModel v2:")
    print(f"  Mean: {df_v2['prob_p1'].mean():.3f}")
    print(f"  Std:  {df_v2['prob_p1'].std():.3f}")
    print(f"  Range: [{df_v2['prob_p1'].min():.3f}, {df_v2['prob_p1'].max():.3f}]")
    
    print(f"\nModel v4:")
    print(f"  Mean: {df_v4['prob_p1'].mean():.3f}")
    print(f"  Std:  {df_v4['prob_p1'].std():.3f}")
    print(f"  Range: [{df_v4['prob_p1'].min():.3f}, {df_v4['prob_p1'].max():.3f}]")
    
    # Check variability
    if df_v4['prob_p1'].std() > df_v2['prob_p1'].std():
        print(f"\n✓ v4 has HIGHER variability ({df_v4['prob_p1'].std():.3f} vs {df_v2['prob_p1'].std():.3f})")
        print("  This suggests v4 is more responsive to match state")
    
    # Overall assessment
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    v2_stuck = df_v2['prob_p1'].std() < 0.05
    v4_stuck = df_v4['prob_p1'].std() < 0.05
    
    if v2_stuck and not v4_stuck:
        print("✅ SUCCESS: v4 is NO LONGER STUCK!")
        print("   v2 was saturated (std < 0.05)")
        print("   v4 responds to match dynamics")
    elif v4_stuck:
        print("❌ FAILURE: v4 is still stuck")
        print("   Need more aggressive fixes")
    else:
        print("✓ Both models show variability")
        print("  Need to check if v4 improvements are meaningful")
        
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Make sure predictions were generated correctly")
