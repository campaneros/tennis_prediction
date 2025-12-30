#!/usr/bin/env python3
"""
Balance synthetic data by swapping P1 and P2 for each match.
This creates a mirror copy where P2 becomes P1, ensuring 50-50 balance.
"""
import pandas as pd
import numpy as np

print('='*80)
print('BALANCING SYNTHETIC DATA')
print('='*80)

# Load original synthetic data
print('\n[1/3] Loading original synthetic data...')
df = pd.read_csv('data/synthetic_training_30k_v4.csv')
print(f'  Original points: {len(df):,}')
print(f'  Original P1 win rate: {df["p1_wins_match"].mean():.3f}')

# Create swapped version
print('\n[2/3] Creating swapped version (P1 ↔ P2)...')
df_swapped = df.copy()

# Swap all P1/P2 columns
df_swapped['p1_points_temp'] = df['p2_points']
df_swapped['p2_points'] = df['p1_points']
df_swapped['p1_points'] = df_swapped['p1_points_temp']
df_swapped.drop('p1_points_temp', axis=1, inplace=True)

df_swapped['p1_games_temp'] = df['p2_games']
df_swapped['p2_games'] = df['p1_games']
df_swapped['p1_games'] = df_swapped['p1_games_temp']
df_swapped.drop('p1_games_temp', axis=1, inplace=True)

df_swapped['p1_sets_temp'] = df['p2_sets']
df_swapped['p2_sets'] = df['p1_sets']
df_swapped['p1_sets'] = df_swapped['p1_sets_temp']
df_swapped.drop('p1_sets_temp', axis=1, inplace=True)

# Swap skills
df_swapped['p1_skill_temp'] = df['p2_skill']
df_swapped['p2_skill'] = df['p1_skill']
df_swapped['p1_skill'] = df_swapped['p1_skill_temp']
df_swapped.drop('p1_skill_temp', axis=1, inplace=True)

# Invert winner
df_swapped['p1_wins_match'] = 1 - df['p1_wins_match']

# Invert server (1→2, 2→1)
df_swapped['server'] = 3 - df['server']

# Invert point_winner (1→2, 2→1)
df_swapped['point_winner'] = 3 - df['point_winner']

# Update match IDs to distinguish swapped matches
df_swapped['match_id'] = df_swapped['match_id'].str.replace('synthetic_', 'synthetic_swap_')

# Combine original + swapped
print('\n[3/3] Combining datasets...')
df_balanced = pd.concat([df, df_swapped], ignore_index=True)

print(f'  Combined points: {len(df_balanced):,}')
print(f'  Balanced P1 win rate: {df_balanced["p1_wins_match"].mean():.3f}')

# Verify balance by checking conditions
print('\n  Verification:')
p1_ahead = df_balanced[df_balanced['p1_sets'] > df_balanced['p2_sets']]
print(f'    When P1 ahead in sets: P1 wins {p1_ahead["p1_wins_match"].mean():.3f}')

p2_ahead = df_balanced[df_balanced['p2_sets'] > df_balanced['p1_sets']]
print(f'    When P2 ahead in sets: P1 wins {p2_ahead["p1_wins_match"].mean():.3f}')

tied = df_balanced[df_balanced['p1_sets'] == df_balanced['p2_sets']]
print(f'    When tied: P1 wins {tied["p1_wins_match"].mean():.3f}')

# Save
output_path = 'data/synthetic_training_30k_v4_balanced.csv'
df_balanced.to_csv(output_path, index=False)
print(f'\n✓ Saved to: {output_path}')
print(f'  Size: {len(df_balanced):,} points from 60k matches (30k original + 30k swapped)')

if abs(df_balanced["p1_wins_match"].mean() - 0.5) < 0.01:
    print('\n✅ SUCCESS: Data is now perfectly balanced!')
else:
    print(f'\n⚠️  Warning: Still some imbalance ({df_balanced["p1_wins_match"].mean():.3f})')
