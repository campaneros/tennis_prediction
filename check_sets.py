import sys
sys.path.insert(0, '.')
from scripts.point_predictors import prepare_dataframe

df = prepare_dataframe(['data/2019-wimbledon-points.csv'])
dm = df[df['match_id']=='2019-wimbledon-1701'].reset_index(drop=True)

print('Has P1SetsWon:', 'P1SetsWon' in dm.columns)
print('Has P2SetsWon:', 'P2SetsWon' in dm.columns)

if 'P1SetsWon' in dm.columns:
    print("\nDati ai punti 198-200:")
    print(dm.iloc[198:201][['SetNo','P1GamesWon','P2GamesWon','P1SetsWon','P2SetsWon','P1Score','P2Score']].to_string())
else:
    print("Colonne mancanti! Colonne disponibili:")
    print([c for c in dm.columns if 'Set' in c or 'set' in c])
