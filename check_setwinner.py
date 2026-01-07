import pandas as pd
from compute_counterfactual import recalculate_match_state_with_changed_point

points_df = pd.read_csv('data/2019-wimbledon-points.csv')
match_data = points_df[points_df['match_id'] == '2019-wimbledon-1701'].copy()

cf_data, _, _ = recalculate_match_state_with_changed_point(match_data, 88)

print('Punto 87 reale SetWinner:', match_data.iloc[87]['SetWinner'])
print('Punto 88 reale SetWinner:', match_data.iloc[88]['SetWinner'])
print('Punto 88 CF SetWinner:', cf_data.iloc[88]['SetWinner'])
print('Punto 88 CF Games:', int(cf_data.iloc[88]['P1GamesWon']), '-', int(cf_data.iloc[88]['P2GamesWon']))
