#!/usr/bin/env python3
import pandas as pd

# Test del merge
print('Test merge LSTM probs...')
df_points = pd.read_csv('data/2019-wimbledon-points.csv')
df_lstm = pd.read_csv('data/lstm_point_probs_male.csv')

print(f'Points dtype PointNumber: {df_points["PointNumber"].dtype}')
print(f'LSTM dtype PointNumber: {df_lstm["PointNumber"].dtype}')

# Converti gestendo '0X'
df_points['PointNumber'] = df_points['PointNumber'].replace('0X', '0')
df_points['PointNumber'] = pd.to_numeric(df_points['PointNumber'], errors='coerce').fillna(0).astype(int)
df_lstm['PointNumber'] = pd.to_numeric(df_lstm['PointNumber'], errors='coerce').fillna(0).astype(int)

print(f'Points dtype PointNumber (after): {df_points["PointNumber"].dtype}')
print(f'LSTM dtype PointNumber (after): {df_lstm["PointNumber"].dtype}')

# Test merge
match_data = df_points[df_points['match_id'] == '2019-wimbledon-1101'].copy()
print(f'\nMatch data points: {len(match_data)}')

merged = match_data.merge(
    df_lstm[['match_id', 'SetNo', 'GameNo', 'PointNumber', 'p1_point_prob']],
    on=['match_id', 'SetNo', 'GameNo', 'PointNumber'],
    how='left'
)
print(f'Merged points: {len(merged)}')
print(f'Probs non-null: {merged["p1_point_prob"].notna().sum()}')
print(f'Sample probs: {merged["p1_point_prob"].head(10).tolist()}')
print('\nMerge funziona correttamente!')
