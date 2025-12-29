import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('plots_v3/match_2019-wimbledon-1701_probabilities.csv')

print('='*80)
print('ANALISI DETTAGLIATA: PROBLEMI DEL MODELLO')
print('='*80)

# Parse match_id to get set/game info if available
# Typically match_id format includes point info

print(f"\nTotal points: {len(df)}")
print(f"Probability range: [{df['prob_p1'].min():.4f}, {df['prob_p1'].max():.4f}]")

# Find key moments
print("\n=== KEY MOMENTS ANALYSIS ===")

# Look at first 20 points
print("\nFirst 20 points (should start ~0.5):")
print(df[['prob_p1', 'prob_p2']].head(20).to_string())

# Look at probability changes
df['prob_change'] = df['prob_p1'].diff().abs()
big_changes = df[df['prob_change'] > 0.1].copy()

print(f"\nPoints with >10% probability change: {len(big_changes)}")
if len(big_changes) > 0:
    print("\nBiggest changes:")
    print(big_changes[['prob_p1', 'prob_change']].nlargest(10, 'prob_change'))

# Statistics
print("\n=== STATISTICAL ISSUES ===")
print(f"Mean probability: {df['prob_p1'].mean():.4f} (should vary around 0.5)")
print(f"Std deviation: {df['prob_p1'].std():.4f}")
print(f"Points with prob > 0.8: {(df['prob_p1'] > 0.8).sum()}")
print(f"Points with prob < 0.2: {(df['prob_p1'] < 0.2).sum()}")

# Check if probabilities sum to 1
df['prob_sum'] = df['prob_p1'] + df['prob_p2']
print(f"\nP1 + P2 sum: min={df['prob_sum'].min():.4f}, max={df['prob_sum'].max():.4f}")
print("(Should be exactly 1.0)")

# Look at end of match (should be close to 1.0 or 0.0)
print("\n=== END OF MATCH ===")
print("Last 10 points (winner should have high probability):")
print(df[['prob_p1', 'prob_p2']].tail(10).to_string())

print("\n=== DIAGNOSIS ===")
if df['prob_p1'].mean() > 0.8:
    print("ERROR: Model is heavily biased toward P1!")
elif df['prob_p1'].mean() < 0.2:
    print("ERROR: Model is heavily biased toward P2!")
elif df['prob_p1'].std() < 0.1:
    print("ERROR: Probabilities don't vary enough (flat predictions)!")
elif (df['prob_p1'] > 0.95).sum() > 100:
    print("ERROR: Too many extreme probabilities (overconfident)!")
else:
    print("UNKNOWN ISSUE: Need to inspect actual match events")
