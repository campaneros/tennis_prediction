#!/usr/bin/env python3
"""
Test modello v5 SENZA data leakage (p1_wins_match = 0.5 unknown).
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
print("TEST: p1_wins_match = 1 (LEAK) vs p1_wins_match = 0 (UNKNOWN)")
print("=" * 80)
print()

# Load match data
df = load_points_single("data/2019-wimbledon-points.csv")
df = df[df['match_id'] == '2019-wimbledon-1701'].copy()

# Load model v5
checkpoint = torch.load("models/complete_model_v5.pth", map_location='cpu', weights_only=False)
model = TennisRulesNet(input_size=31, hidden_sizes=[128, 64], dropout=0.4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
temperature = checkpoint.get('temperature', 3.0)

print(f"Testing with temperature: {temperature}\n")

# Test 1: With CORRECT winner (data leakage)
print("[TEST 1] p1_wins_match = 1 (Djokovic wins - TRUE)")
df1 = df.copy()
df1['p1_wins_match'] = 1  # Djokovic actually won

X1, _, _, _, _, _ = build_new_features(df1)

with torch.no_grad():
    X_tensor = torch.FloatTensor(X1)
    outputs = model(X_tensor)
    logits = outputs['match']
    probs_p1 = torch.pow(torch.sigmoid(logits), 1/temperature)
    probs_p2 = 1 - probs_p1
    probs_p1 = probs_p1 / (probs_p1 + probs_p2)
    probs1 = probs_p1.numpy()

print(f"  Range: [{float(probs1.min()):.4f}, {float(probs1.max()):.4f}]")
print(f"  Mean: {float(probs1.mean()):.4f}")
print(f"  Final: {float(probs1[-1]):.4f}")
print()

# Test 2: With WRONG winner (opposite)
print("[TEST 2] p1_wins_match = 0 (Federer wins - FALSE)")
df2 = df.copy()
df2['p1_wins_match'] = 0  # Pretend Federer won

X2, _, _, _, _, _ = build_new_features(df2)

with torch.no_grad():
    X_tensor = torch.FloatTensor(X2)
    outputs = model(X_tensor)
    logits = outputs['match']
    probs_p1 = torch.pow(torch.sigmoid(logits), 1/temperature)
    probs_p2 = 1 - probs_p1
    probs_p1 = probs_p1 / (probs_p1 + probs_p2)
    probs2 = probs_p1.numpy()

print(f"  Range: [{float(probs2.min()):.4f}, {float(probs2.max()):.4f}]")
print(f"  Mean: {float(probs2.mean()):.4f}")
print(f"  Final: {float(probs2[-1]):.4f}")
print()

# Test 3: Check if features are different
print("[FEATURE COMPARISON]")
features_same = np.allclose(X1, X2)
print(f"  Features identical: {features_same}")

if not features_same:
    diff_mask = ~np.isclose(X1, X2).all(axis=1)
    n_different = diff_mask.sum()
    print(f"  Different samples: {n_different}/{len(X1)}")
    
    # Find which features differ
    feature_diff = ~np.isclose(X1, X2)
    different_features = feature_diff.any(axis=0)
    print(f"  Different feature columns: {different_features.sum()}/31")
else:
    print("  Features are IDENTICAL - p1_wins_match is NOT used as input")
    print("  Predictions should be IDENTICAL too...")
    
    if np.allclose(probs1, probs2):
        print("  ✓ Predictions ARE identical (as expected)")
    else:
        print("  ✗ Predictions DIFFER (unexpected - bug in model?)")

print()
print("=" * 80)
print("CONCLUSION:")
print("=" * 80)

if features_same:
    print("p1_wins_match does NOT affect model input (no data leakage from that)")
    print()
    print("But why is probability always >0.80?")
    print("Possible causes:")
    print("  1. Model learned that P1 (first player) wins more often")
    print("  2. Features like P_srv_win/P_srv_lose contain skill information")
    print("  3. Model is too confident from training data")
    print("  4. Temperature calibration needs adjustment")
else:
    print("⚠ WARNING: p1_wins_match DOES affect features (data leakage!)")
    print("This explains why probability is always high")

print("=" * 80)
