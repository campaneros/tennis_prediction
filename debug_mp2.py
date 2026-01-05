import sys
sys.path.insert(0, '.')
from scripts.point_predictors import prepare_dataframe

df = prepare_dataframe(['data/2019-wimbledon-points.csv'])
dm = df[df['match_id']=='2019-wimbledon-1701'].reset_index(drop=True)

print("Punti 197-201:")
print(dm.iloc[197:202][['SetNo','P1GamesWon','P2GamesWon','P1SetsWon','P2SetsWon','P1Score','P2Score']].to_string())
