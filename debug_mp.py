import sys
sys.path.insert(0, '.')
from scripts.point_predictors import prepare_dataframe
import pandas as pd

df = prepare_dataframe(['data/2019-wimbledon-points.csv'])
dm = df[df['match_id']=='2019-wimbledon-1701'].reset_index(drop=True)

# Controlla punto 199 (riga 199 nel df)
row = dm.iloc[199]

print("Punto 199 - dati row:")
print(f"SetNo: {row.get('SetNo')}")
print(f"P1GamesWon: {row.get('P1GamesWon')}")
print(f"P2GamesWon: {row.get('P2GamesWon')}")
print(f"P1Score: {row.get('P1Score')}")
print(f"P2Score: {row.get('P2Score')}")
print(f"P1SetsWon: {row.get('P1SetsWon', 'MISSING')}")
print(f"P2SetsWon: {row.get('P2SetsWon', 'MISSING')}")
print(f"SetNo_original: {row.get('SetNo_original', 'MISSING')}")
print(f"SetNo_full_max: {row.get('SetNo_full_max', 'MISSING')}")
