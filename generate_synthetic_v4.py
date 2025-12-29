#!/usr/bin/env python3
"""Generate synthetic training data"""
import sys
sys.path.insert(0, 'scripts')

from tennis_simulator import TennisSimulator
import pandas as pd
import numpy as np

n_matches = 30000
output_path = 'data/synthetic_training_30k_v4.csv'

simulator = TennisSimulator(best_of_5=True, seed=42)
all_matches = []

print(f"Generating {n_matches} synthetic tennis matches...")

for match_id in range(1, n_matches + 1):
    if match_id % 1000 == 0:
        print(f"  Generated {match_id}/{n_matches} matches...")
    
    # Vary skill levels to create diverse matches
    p1_skill = np.random.uniform(0.50, 0.65)
    p2_skill = np.random.uniform(0.50, 0.65)
    
    match_df = simulator.simulate_match(p1_skill, p2_skill)
    match_df['match_id'] = f"synthetic_{match_id:06d}"
    match_df['p1_skill'] = p1_skill
    match_df['p2_skill'] = p2_skill
    
    all_matches.append(match_df)

# Combine all matches
full_df = pd.concat(all_matches, ignore_index=True)

print(f"Generated {len(full_df)} total points from {n_matches} matches")
print(f"Average points per match: {len(full_df) / n_matches:.1f}")

full_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# Check balance
p1_wins = full_df['p1_wins_match'].mean()
print(f"\nP1 win rate: {p1_wins:.3f} (should be ~0.50 for balanced data)")
