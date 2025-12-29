"""
Analyze the distribution of training data.
Check if there's a severe class imbalance that caused the bias.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print('='*80)
print('ANALYZING TRAINING DATA DISTRIBUTION')
print('='*80)

# Load real data
data_dir = Path('data')
all_points = []

# Load all matches
for csv_file in sorted(data_dir.glob('*-points.csv')):
    try:
        df = pd.read_csv(csv_file)
        all_points.append(df)
        print(f"Loaded: {csv_file.name:40s} - {len(df):6d} points")
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

df_all = pd.concat(all_points, ignore_index=True)
print(f"\nTotal points: {len(df_all):,}")

# Check P1 win rate
matches_file = list(data_dir.glob('*-matches.csv'))
print(f"\nFound {len(matches_file)} match files")

all_matches = []
for match_file in sorted(matches_file):
    try:
        df_match = pd.read_csv(match_file)
        all_matches.append(df_match)
    except:
        pass

if all_matches:
    df_matches = pd.concat(all_matches, ignore_index=True)
    print(f"Total matches: {len(df_matches):,}")
    
    # Check who wins
    # In the data format, the match winner should be determinable
    # Let's look at a sample
    print("\nSample match data:")
    print(df_matches.head(3)[['player1', 'player2']].to_string())
    
    # Check if there's a match_winner column
    if 'match_winner' in df_matches.columns:
        print(f"\nMatch winners:")
        print(df_matches['match_winner'].value_counts())
    
    # Check set scores
    set_cols = [c for c in df_matches.columns if 'set' in c.lower()]
    print(f"\nSet-related columns: {set_cols[:10]}")

# Analyze point-level data
print("\n" + "="*80)
print("POINT-LEVEL ANALYSIS")
print("="*80)

# Check column names
print(f"Columns ({len(df_all.columns)}): {list(df_all.columns[:20])}")

# Look for winner information
winner_cols = [c for c in df_all.columns if 'winner' in c.lower() or 'won' in c.lower()]
print(f"\nWinner-related columns: {winner_cols}")

if 'PointWinner' in df_all.columns:
    point_winners = df_all['PointWinner'].value_counts()
    print(f"\nPoint winner distribution:")
    print(point_winners)
    print(f"\nPlayer 1 win rate: {point_winners.get(1, 0) / len(df_all) * 100:.2f}%")
    print(f"Player 2 win rate: {point_winners.get(2, 0) / len(df_all) * 100:.2f}%")

# Check set wins at point level
if 'P1SetsWon' in df_all.columns and 'P2SetsWon' in df_all.columns:
    print("\n" + "="*80)
    print("SET SCORE DISTRIBUTION")
    print("="*80)
    
    # Look at set score at each point
    set_scores = df_all[['P1SetsWon', 'P2SetsWon']].value_counts().sort_index()
    print("\nSet scores (P1 sets, P2 sets) - count:")
    for (p1s, p2s), count in set_scores.head(20).items():
        pct = count / len(df_all) * 100
        print(f"  {p1s}-{p2s}: {count:7d} points ({pct:5.2f}%)")
    
    # Check if P1 wins more often
    p1_winning_sets = (df_all['P1SetsWon'] > df_all['P2SetsWon']).sum()
    p2_winning_sets = (df_all['P2SetsWon'] > df_all['P1SetsWon']).sum()
    tied_sets = (df_all['P1SetsWon'] == df_all['P2SetsWon']).sum()
    
    print(f"\n Points where P1 ahead in sets: {p1_winning_sets:7d} ({p1_winning_sets/len(df_all)*100:5.2f}%)")
    print(f"Points where P2 ahead in sets: {p2_winning_sets:7d} ({p2_winning_sets/len(df_all)*100:5.2f}%)")
    print(f"Points where sets tied:         {tied_sets:7d} ({tied_sets/len(df_all)*100:5.2f}%)")
    
    if p1_winning_sets > p2_winning_sets * 1.5:
        print(f"\n⚠️  SEVERE IMBALANCE: P1 leads in sets {p1_winning_sets/p2_winning_sets:.2f}x more often!")
        print("This could explain model's bias toward P1.")

# Check match outcomes at point level
# The last point of each match should show who won
if 'match_id' in df_all.columns and 'P1SetsWon' in df_all.columns:
    print("\n" + "="*80)
    print("MATCH OUTCOMES")
    print("="*80)
    
    # Group by match and get last point
    last_points = df_all.groupby('match_id').tail(1)
    print(f"Total matches (from last points): {len(last_points)}")
    
    p1_match_wins = (last_points['P1SetsWon'] > last_points['P2SetsWon']).sum()
    p2_match_wins = (last_points['P2SetsWon'] > last_points['P1SetsWon']).sum()
    
    print(f"\nP1 won: {p1_match_wins} matches ({p1_match_wins/len(last_points)*100:.2f}%)")
    print(f"P2 won: {p2_match_wins} matches ({p2_match_wins/len(last_points)*100:.2f}%)")
    
    if p1_match_wins > p2_match_wins * 1.2:
        print(f"\n❌ CLASS IMBALANCE: P1 wins {p1_match_wins/p2_match_wins:.2f}x more often!")
        print("Model learned to always predict P1 wins.")
    elif abs(p1_match_wins - p2_match_wins) < len(last_points) * 0.05:
        print(f"\n✓ Balanced: P1/P2 match wins are roughly equal.")
        print("The bias is NOT due to training data imbalance.")
