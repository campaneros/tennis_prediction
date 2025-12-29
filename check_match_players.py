"""
Check who won the 2019 Wimbledon final.
"""
import pandas as pd

print('='*80)
print('2019 WIMBLEDON FINAL - PLAYER IDENTIFICATION')
print('='*80)

# Load match data
df_matches = pd.read_csv('data/2019-wimbledon-matches.csv')
match = df_matches[df_matches['match_id'] == '2019-wimbledon-1701']

if len(match) > 0:
    print("\nMatch info:")
    print(f"Player 1: {match['player1'].values[0]}")
    print(f"Player 2: {match['player2'].values[0]}")
    
    if 'winner' in match.columns:
        print(f"Winner: {match['winner'].values[0]}")
    
    # Check score
    score_cols = [c for c in match.columns if 'score' in c.lower() or 'set' in c.lower()]
    if score_cols:
        print(f"\nScore columns: {score_cols[:10]}")
        for col in score_cols[:10]:
            print(f"  {col}: {match[col].values[0]}")
else:
    print("Match not found in matches file")

# Load point data to check final score
print("\n" + "="*80)
print("POINT DATA ANALYSIS")
print("="*80)

df_points = pd.read_csv('data/2019-wimbledon-points.csv')
match_points = df_points[df_points['match_id'] == '2019-wimbledon-1701']

if len(match_points) > 0:
    print(f"\nTotal points in match: {len(match_points)}")
    
    # Get final score (last point)
    last_point = match_points.iloc[-1]
    
    print(f"\nFinal score (sets):")
    print(f"  P1 sets: {last_point['P1SetsWon']}")
    print(f"  P2 sets: {last_point['P2SetsWon']}")
    
    if last_point['P1SetsWon'] > last_point['P2SetsWon']:
        winner = "Player 1"
        winner_sets = int(last_point['P1SetsWon'])
        loser_sets = int(last_point['P2SetsWon'])
    else:
        winner = "Player 2"
        winner_sets = int(last_point['P2SetsWon'])
        loser_sets = int(last_point['P1SetsWon'])
    
    print(f"\n**{winner} won the match {winner_sets}-{loser_sets}**")
    
    # Check set-by-set scores
    print("\nSet-by-set progression:")
    for set_num in sorted(match_points['SetNo'].unique()):
        set_points = match_points[match_points['SetNo'] == set_num]
        last_in_set = set_points.iloc[-1]
        print(f"  Set {int(set_num)}: P1 {int(last_in_set['P1GamesWon'])}-{int(last_in_set['P2GamesWon'])} P2")
    
    # Now check who actually is player 1 and 2
    if len(match) > 0:
        p1_name = match['player1'].values[0]
        p2_name = match['player2'].values[0]
        
        print(f"\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print(f"Player 1 (P1): {p1_name}")
        print(f"Player 2 (P2): {p2_name}")
        print(f"Winner: {winner}")
        
        if winner == "Player 1":
            print(f"\n✓ {p1_name} won {winner_sets}-{loser_sets}")
        else:
            print(f"\n✓ {p2_name} won {winner_sets}-{loser_sets}")
        
        # Famous match context
        print("\nHistorical context:")
        print("The 2019 Wimbledon final was Novak Djokovic vs Roger Federer.")
        print("Djokovic won 7-6(5), 1-6, 7-6(4), 4-6, 13-12(3)")
        print("It was the longest Wimbledon final in history (4h 57min).")
else:
    print("No point data found for this match")
