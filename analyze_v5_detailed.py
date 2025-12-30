#!/usr/bin/env python3
"""
Analizza il comportamento del modello v5 sul match di Wimbledon 2019.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from scripts.data_loader import load_points_single
    from scripts.new_model_nn import build_new_features
    from scripts.pretrain_tennis_rules import TennisRulesNet
except ImportError:
    from data_loader import load_points_single
    from new_model_nn import build_new_features
    from pretrain_tennis_rules import TennisRulesNet

import torch
import pandas as pd
import numpy as np

print("=" * 80)
print("ANALISI MODELLO V5 - Wimbledon 2019 Final")
print("=" * 80)
print()

# Load match data
print("[1/3] Loading match data...")
df = load_points_single("data/2019-wimbledon-points.csv")
df = df[df['match_id'] == '2019-wimbledon-1701'].copy()
# Add dummy label for feature extraction
df['p1_wins_match'] = 1  # Djokovic won
print(f"  Loaded {len(df)} points")
print()

# Load model
print("[2/3] Loading model v5...")
checkpoint = torch.load("models/complete_model_v5.pth", map_location='cpu', weights_only=False)
model = TennisRulesNet(input_size=31, hidden_sizes=[128, 64], dropout=0.4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
temperature = checkpoint.get('temperature', 3.0)
print(f"  Model loaded (temperature={temperature})")
print()

# Build features and predict
print("[3/3] Computing predictions...")
X, y_match, y_set, y_game, weights, match_ids_list = build_new_features(df)

with torch.no_grad():
    X_tensor = torch.FloatTensor(X)
    outputs = model(X_tensor)
    logits = outputs['match']
    
    # Apply temperature
    probs_p1 = torch.pow(torch.sigmoid(logits), 1/temperature)
    probs_p2 = 1 - probs_p1
    probs_p1 = probs_p1 / (probs_p1 + probs_p2)
    probs = probs_p1.numpy()

# Add predictions to dataframe
df = df.iloc[:len(probs)].copy()
df['pred_prob_p1'] = probs

print(f"  Computed {len(probs)} predictions")
print()

# Analyze results
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print("OVERALL STATISTICS:")
print(f"  Range: [{float(probs.min()):.4f}, {float(probs.max()):.4f}]")
print(f"  Mean: {float(probs.mean()):.4f}")
print(f"  Std: {float(probs.std()):.4f}")
print(f"  Final: {float(probs[-1]):.4f}")
print()

# Analyze by set
print("BY SET:")
for set_no in sorted(df['SetNo'].unique()):
    df_set = df[df['SetNo'] == set_no]
    probs_set = df_set['pred_prob_p1'].values
    
    # Final games in set
    last_row = df_set.iloc[-1]
    p1_games = last_row['P1GamesWon']
    p2_games = last_row['P2GamesWon']
    winner = "Djokovic" if p1_games > p2_games else "Federer"
    
    print(f"  Set {set_no}: {int(p1_games)}-{int(p2_games)} ({winner})")
    print(f"    Prob range: [{probs_set.min():.4f}, {probs_set.max():.4f}]")
    print(f"    Prob mean: {probs_set.mean():.4f}")
    print(f"    Prob final: {probs_set[-1]:.4f}")

print()

# Check correlation with match state
print("CORRELATION WITH MATCH STATE:")
df['sets_diff'] = df['P1SetsWon'] - df['P2SetsWon']
df['games_diff'] = df['P1GamesWon'] - df['P2GamesWon']

corr_sets = df['pred_prob_p1'].corr(df['sets_diff'])
corr_games = df['pred_prob_p1'].corr(df['games_diff'])

print(f"  Correlation with sets difference: {corr_sets:.4f}")
print(f"  Correlation with games difference: {corr_games:.4f}")

if corr_sets < 0:
    print("  ⚠ NEGATIVE correlation with sets! Model is INVERTED")
if corr_games < 0:
    print("  ⚠ NEGATIVE correlation with games! Model is INVERTED")

print()

# Group by sets ahead/behind
print("PROBABILITY BY MATCH SITUATION:")
for diff in sorted(df['sets_diff'].unique()):
    df_diff = df[df['sets_diff'] == diff]
    mean_prob = df_diff['pred_prob_p1'].mean()
    n = len(df_diff)
    print(f"  Djokovic {int(diff):+d} sets: prob={mean_prob:.4f} (n={n})")

print()

# Final situation
print("FINAL SITUATION:")
last = df.iloc[-1]
print(f"  Sets: Djokovic {int(last['P1SetsWon'])} - Federer {int(last['P2SetsWon'])}")
print(f"  Games (final set): {int(last['P1GamesWon'])}-{int(last['P2GamesWon'])}")
print(f"  Predicted prob Djokovic: {last['pred_prob_p1']:.4f}")
print(f"  ACTUAL WINNER: Djokovic (3-2 sets)")
print()

if last['pred_prob_p1'] < 0.75:
    print("⚠ PROBLEM: Final probability is LOW (<0.75) despite Djokovic winning!")
    print()
    print("Possible issues:")
    print("  1. Early stopping too aggressive (stopped at epoch 8)")
    print("  2. Fine-tuning destroyed pre-training knowledge")
    print("  3. Pre-training still has hidden bias")
    print("  4. Model architecture not suitable for this task")

print()
print("=" * 80)
