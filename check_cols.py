import sys
sys.path.insert(0, '.')
from scripts.point_predictors import prepare_dataframe

df = prepare_dataframe(['data/2019-wimbledon-points.csv'])
dm = df[df['match_id']=='2019-wimbledon-1701'].reset_index(drop=True)

print("Colonne disponibili nel dataframe:")
print([c for c in dm.columns if 'Set' in c or 'Game' in c or 'Score' in c])

print("\nUltimi 3 punti del match:")
print(dm.tail(3)[['SetNo','GameNo','P1GamesWon','P2GamesWon','P1Score','P2Score']].to_string())

print("\nColonne P1SetsWon e P2SetsWon esistono?")
print(f"P1SetsWon: {'P1SetsWon' in dm.columns}")
print(f"P2SetsWon: {'P2SetsWon' in dm.columns}")

if 'P1SetsWon' in dm.columns:
    print("\nUltimi 3 punti con info set:")
    print(dm.tail(3)[['P1GamesWon','P2GamesWon','P1SetsWon','P2SetsWon','P1Score','P2Score']].to_string())
