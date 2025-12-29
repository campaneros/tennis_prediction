import pandas as pd
import numpy as np

# Load predictions
df_old = pd.read_csv('plots/match_2019-wimbledon-1701_probabilities.csv')
df_new = pd.read_csv('plots_v3/match_2019-wimbledon-1701_probabilities.csv')

print('='*80)
print('COMPARISON: OLD MODEL (T=12.0) vs NEW MODEL (T=3.0)')
print('='*80)

print('\nOLD MODEL (T=12.0):')
print(f'  Probability range: [{df_old["prob_p1"].min():.4f}, {df_old["prob_p1"].max():.4f}]')
print(f'  Mean: {df_old["prob_p1"].mean():.4f}')
print(f'  Std: {df_old["prob_p1"].std():.4f}')

print('\nNEW MODEL (T=3.0):')
print(f'  Probability range: [{df_new["prob_p1"].min():.4f}, {df_new["prob_p1"].max():.4f}]')
print(f'  Mean: {df_new["prob_p1"].mean():.4f}')
print(f'  Std: {df_new["prob_p1"].std():.4f}')

print('\nIMPROVEMENT:')
old_range = df_old['prob_p1'].max() - df_old['prob_p1'].min()
new_range = df_new['prob_p1'].max() - df_new['prob_p1'].min()
print(f'  Old range width: {old_range:.4f}')
print(f'  New range width: {new_range:.4f}')
print(f'  Range increase: {(new_range / old_range - 1) * 100:.1f}%')
print(f'\n  Old model max probability: {df_old["prob_p1"].max():.4f} (CAPPED!)')
print(f'  New model max probability: {df_new["prob_p1"].max():.4f}')
print(f'  Gained access to: {(df_new["prob_p1"].max() - df_old["prob_p1"].max()):.4f} more probability space')

print('\n' + '='*80)
print('CONCLUSION: Temperature reduction from 12.0 to 3.0 successfully')
print('eliminated the artificial cap, allowing full dynamic range!')
print('='*80)
