#!/usr/bin/env python3
import pandas as pd
import pickle
from compute_counterfactual import recalculate_match_state_with_changed_point, create_tennis_features

# Carica dati
points_df = pd.read_csv('data/2019-wimbledon-points.csv')
match_data = points_df[points_df['match_id'] == '2019-wimbledon-1701'].copy()

# Carica modello
with open('models/tennis_bdt_male.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']

# Carica LSTM
lstm_df = pd.read_csv('data/lstm_point_probs_male.csv')
lstm_df = lstm_df[lstm_df['match_id'] == '2019-wimbledon-1701'].copy()

i = 88

print(f"\n=== PUNTO {i} ===")
print(f"Score REALE prima del punto {i}:")
if i > 0:
    prev = match_data.iloc[i-1]
    print(f"  Games: {prev['P1GamesWon']}-{prev['P2GamesWon']}")
    print(f"  Points: {prev['P1Score']}-{prev['P2Score']}")

print(f"\nPunto {i} reale:")
curr = match_data.iloc[i]
print(f"  Winner: {curr['PointWinner']}")
print(f"  Games DOPO: {curr['P1GamesWon']}-{curr['P2GamesWon']}")
print(f"  Points DOPO: {curr['P1Score']}-{curr['P2Score']}")

# Calcola counterfactual
cf_data, match_won, winner = recalculate_match_state_with_changed_point(match_data, i)
cf_row = cf_data.iloc[-1]

print(f"\nPunto {i} COUNTERFACTUAL:")
print(f"  Winner CF: {cf_row['PointWinner']}")
print(f"  Games DOPO CF: {cf_row['P1GamesWon']}-{cf_row['P2GamesWon']}")
print(f"  Points DOPO CF: {cf_row['P1Score']}-{cf_row['P2Score']}")

# Calcola feature reali PRIMA del punto i (usando dati fino a i-1)
if i == 0:
    real_hist = match_data.iloc[:1].copy()
    real_lstm = lstm_df.iloc[:1].copy()
else:
    real_hist = match_data.iloc[:i].copy()
    real_lstm = lstm_df.iloc[:i].copy()

X_real = create_tennis_features(real_hist, real_lstm, n_features=51)
print(f"\nFeature REALI (prima del punto {i}):")
print(f"  N features: {len(X_real)}")
if len(X_real) > 0:
    feat_real = X_real[-1]
    print(f"  p1_games (idx 4): {feat_real[4]}")
    print(f"  p2_games (idx 5): {feat_real[5]}")
    print(f"  p1_point_val (idx 9): {feat_real[9]}")
    print(f"  p2_point_val (idx 10): {feat_real[10]}")
    
    # Predici
    prob_real = model.predict_proba(X_real[-1:, :])
    print(f"  P1 prob: {prob_real[0, 1]:.4f}")

# Calcola feature CF DOPO il punto i con vincitore invertito
if i == 0:
    cf_hist = cf_data.iloc[:1].copy()
    cf_lstm = lstm_df.iloc[:1].copy()
else:
    cf_hist = cf_data.iloc[:i+1].copy()
    cf_lstm = lstm_df.iloc[:i+1].copy()

X_cf = create_tennis_features(cf_hist, cf_lstm, n_features=51)
print(f"\nFeature CF (dopo il punto {i} con esito opposto):")
print(f"  N features: {len(X_cf)}")
if len(X_cf) > 0:
    feat_cf = X_cf[-1]
    print(f"  p1_games (idx 4): {feat_cf[4]}")
    print(f"  p2_games (idx 5): {feat_cf[5]}")
    print(f"  p1_point_val (idx 9): {feat_cf[9]}")
    print(f"  p2_point_val (idx 10): {feat_cf[10]}")
    print(f"  in_tiebreak (idx 48): {feat_cf[48]}")
    print(f"  tiebreak_point_diff (idx 49): {feat_cf[49]}")
    
    # Predici
    prob_cf = model.predict_proba(X_cf[-1:, :])
    print(f"  P1 prob CF: {prob_cf[0, 1]:.4f}")
