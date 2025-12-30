"""
Re-analyze the 2019 Wimbledon final knowing that:
- P1 = Djokovic (WINNER)
- P2 = Federer (LOSER)
- Final score: Djokovic won 3-2 in sets
"""
import pandas as pd
import numpy as np
from pathlib import Path

print('='*80)
print('RE-ANALYSIS: 2019 WIMBLEDON FINAL')
print('P1 = Novak Djokovic (WINNER)')
print('P2 = Roger Federer (LOSER)')
print('='*80)

# Load raw data
df_raw = pd.read_csv('data/2019-wimbledon-points.csv')
match_points = df_raw[df_raw['match_id'] == '2019-wimbledon-1701'].copy()

print(f"\nTotal points: {len(match_points)}")

# Calculate sets won through the match
match_points['P1_sets_won'] = 0
match_points['P2_sets_won'] = 0

for set_num in sorted(match_points['SetNo'].unique()):
    set_data = match_points[match_points['SetNo'] == set_num]
    last_point = set_data.iloc[-1]
    
    # Who won this set?
    p1_games = last_point['P1GamesWon']
    p2_games = last_point['P2GamesWon']
    set_winner = 1 if p1_games > p2_games else 2
    
    # Update sets won for all subsequent points
    match_points.loc[match_points['SetNo'] > set_num, 'P1_sets_won'] += (1 if set_winner == 1 else 0)
    match_points.loc[match_points['SetNo'] > set_num, 'P2_sets_won'] += (1 if set_winner == 2 else 0)
    
    print(f"Set {int(set_num)}: P1 {int(p1_games)}-{int(p2_games)} P2 → Winner: P{set_winner}")

# Final score
final_p1_sets = match_points['P1_sets_won'].iloc[-1] + (1 if match_points.iloc[-1]['P1GamesWon'] > match_points.iloc[-1]['P2GamesWon'] else 0)
final_p2_sets = match_points['P2_sets_won'].iloc[-1] + (1 if match_points.iloc[-1]['P2GamesWon'] > match_points.iloc[-1]['P1GamesWon'] else 0)

print(f"\nFinal: Djokovic {int(final_p1_sets)}-{int(final_p2_sets)} Federer")

# Load predictions
pred_file = Path('plots/v4/match_2019-wimbledon-1701_probabilities.csv')
if pred_file.exists():
    df_pred = pd.read_csv(pred_file)
    
    print("\n" + "="*80)
    print("MODEL PREDICTIONS ANALYSIS")
    print("="*80)
    
    # Start of match
    print("\n1. MATCH START (first 5 points):")
    print(f"   Expected: ~0.50 (no information)")
    print(f"   Model avg: {df_pred['prob_p1'].iloc[:5].mean():.3f}")
    
    # Key moments
    for set_num in range(1, 6):
        set_data = match_points[match_points['SetNo'] == set_num]
        if len(set_data) == 0:
            continue
        
        set_end_idx = set_data.index[-1]
        
        # Get prediction at end of set (or next point)
        if set_end_idx < len(df_pred) - 1:
            prob_after_set = df_pred.iloc[set_end_idx + 1]['prob_p1']
        else:
            prob_after_set = df_pred.iloc[-1]['prob_p1']
        
        # Calculate what was the set score after this set
        p1_sets = int(match_points.loc[set_end_idx, 'P1_sets_won'])
        p2_sets = int(match_points.loc[set_end_idx, 'P2_sets_won'])
        
        # Add the set that just finished
        if match_points.loc[set_end_idx, 'P1GamesWon'] > match_points.loc[set_end_idx, 'P2GamesWon']:
            p1_sets += 1
        else:
            p2_sets += 1
        
        print(f"\n{set_num+1}. AFTER SET {set_num}:")
        print(f"   Set score: Djokovic {p1_sets}-{p2_sets} Federer")
        
        # Expected probability based on set score
        if p1_sets == 2 and p2_sets == 0:
            expected = "~0.80 (Djokovic 2 sets up)"
        elif p1_sets == 2 and p2_sets == 1:
            expected = "~0.65-0.70 (Djokovic leads)"
        elif p1_sets == 1 and p2_sets == 1:
            expected = "~0.50 (tied)"
        elif p1_sets == 2 and p2_sets == 2:
            expected = "~0.50 (tied, final set)"
        elif p1_sets == 1 and p2_sets == 0:
            expected = "~0.60-0.65 (Djokovic 1 set up)"
        elif p1_sets == 0 and p2_sets == 1:
            expected = "~0.35-0.40 (Federer 1 set up)"
        elif p1_sets == 1 and p2_sets == 2:
            expected = "~0.30-0.35 (Federer leads)"
        else:
            expected = "~???"
        
        print(f"   Expected: {expected}")
        print(f"   Model: {prob_after_set:.3f}")
        
        # Check if model is reasonable
        if p1_sets > p2_sets and prob_after_set < 0.50:
            print(f"   ❌ ERROR: Djokovic leads but model gives <0.50")
        elif p2_sets > p1_sets and prob_after_set > 0.50:
            print(f"   ⚠️  WARNING: Federer leads but model gives >0.50")
        elif abs(p1_sets - p2_sets) >= 2 and 0.40 < prob_after_set < 0.60:
            print(f"   ❌ ERROR: 2-set difference but model says ~50-50")
    
    # End of match
    print(f"\n6. END OF MATCH (final prediction):")
    print(f"   Djokovic WON 3-2")
    print(f"   Expected: ~0.95+ (Djokovic certain winner)")
    print(f"   Model: {df_pred.iloc[-1]['prob_p1']:.3f}")
    
    if df_pred.iloc[-1]['prob_p1'] < 0.85:
        print(f"   ❌ CRITICAL ERROR: Match is over but model only {df_pred.iloc[-1]['prob_p1']:.3f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Check if model correctly identifies winner
    final_prob = df_pred.iloc[-1]['prob_p1']
    if final_prob > 0.50:
        model_pick = "Djokovic"
        confidence = final_prob
    else:
        model_pick = "Federer"
        confidence = 1 - final_prob
    
    print(f"Model's pick: {model_pick} (confidence: {confidence:.3f})")
    print(f"Actual winner: Djokovic")
    
    if model_pick == "Djokovic":
        print(f"\n✓ Model correctly identified the winner!")
        if confidence > 0.90:
            print(f"✓ With very high confidence ({confidence:.3f})")
        elif confidence > 0.75:
            print(f"⚠️ But confidence is moderate ({confidence:.3f}), should be >0.90")
        else:
            print(f"❌ But confidence is too low ({confidence:.3f}), should be >0.90")
    else:
        print(f"\n❌ Model picked the WRONG winner!")

else:
    print(f"\n⚠️ Prediction file not found: {pred_file}")
    print("Run: python tennisctl.py predict --model ./models/complete_model_v2.pth --match-id 2019-wimbledon-1701 --files data/2019-wimbledon-points.csv --plot-dir plots --point-by-point")
