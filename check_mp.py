import sys
sys.path.insert(0, '.')
import pandas as pd
from scripts.point_predictors import prepare_dataframe

df = prepare_dataframe(['data/2019-wimbledon-points.csv'])
dm = df[df['match_id']=='2019-wimbledon-1701'].reset_index(drop=True)

# Penultimo punto (indice 423)
row = dm.iloc[423]

print("Penultimo punto (dovrebbe essere match point):")
print(f"Index: 423")
print(f"SetNo: {row.get('SetNo')}")
print(f"P1GamesWon: {row.get('P1GamesWon')}")
print(f"P2GamesWon: {row.get('P2GamesWon')}")
print(f"P1Score: {row.get('P1Score')}")
print(f"P2Score: {row.get('P2Score')}")
print(f"P1SetsWon: {row.get('P1SetsWon')}")
print(f"P2SetsWon: {row.get('P2SetsWon')}")
print(f"PointWinner: {row.get('PointWinner')}")
