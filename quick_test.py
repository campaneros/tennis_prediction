#!/usr/bin/env python3
"""Quick test of simulator"""
import sys
sys.path.insert(0, 'scripts')

from tennis_simulator import generate_training_dataset

print('Testing simulator with 10 matches...')
df = generate_training_dataset(n_matches=10, output_path=None, seed=42)
print(f'✓ Generated {len(df)} points')
print(f'  Average: {len(df)/10:.1f} points/match')
print(f'  Columns: {len(df.columns)}')
print(f'  P1 wins: {df.groupby("match_id")["p1_wins_match"].first().sum():.0f}/10')
print('\n✓ Simulator test PASSED')
