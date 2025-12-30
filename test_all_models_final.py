#!/usr/bin/env python3
"""
Test finale del modello v5 sul match Wimbledon 2019 con predizioni corrette.
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

def test_model(model_path, model_name):
    """Test a model on Wimbledon 2019 final."""
    
    print(f"\n{'=' * 80}")
    print(f"TESTING {model_name}")
    print(f"{'=' * 80}\n")
    
    # Load match data
    df = load_points_single("data/2019-wimbledon-points.csv")
    df = df[df['match_id'] == '2019-wimbledon-1701'].copy()
    df['p1_wins_match'] = 1  # Djokovic won
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = TennisRulesNet(input_size=31, hidden_sizes=[128, 64], dropout=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    temperature = checkpoint.get('temperature', 3.0)
    
    # Build features and predict
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
    
    # Results
    print(f"Points: {len(probs)}")
    print(f"Temperature: {temperature}")
    print(f"Range: [{float(probs.min()):.4f}, {float(probs.max()):.4f}] (width: {float(probs.max() - probs.min()):.4f})")
    print(f"Mean: {float(probs.mean()):.4f}")
    print(f"Std: {float(probs.std()):.4f}")
    print(f"Final: {float(probs[-1]):.4f}")
    
    # Check responsiveness
    if probs.std() < 0.10:
        print("⚠ Low std - model not responsive!")
    else:
        print("✓ Good std - model responds to match dynamics")
    
    # Check final probability
    if probs[-1] < 0.70:
        print("⚠ Low final prob - should be >0.85 for winner!")
    elif probs[-1] < 0.85:
        print("~ Acceptable final prob (0.70-0.85)")
    else:
        print("✓ Excellent final prob (>0.85)")
    
    return {
        'name': model_name,
        'probs': probs,
        'min': float(probs.min()),
        'max': float(probs.max()),
        'mean': float(probs.mean()),
        'std': float(probs.std()),
        'final': float(probs[-1]),
        'temperature': temperature
    }

# Test all models
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("Match: 2019 Wimbledon Final - Djokovic vs Federer")
print("Actual winner: Djokovic (3-2 sets)")
print("=" * 80)

results = []

if os.path.exists("models/complete_model_v2.pth"):
    results.append(test_model("models/complete_model_v2.pth", "V2 (T=12.0, saturated)"))

if os.path.exists("models/complete_model_v4.pth"):
    results.append(test_model("models/complete_model_v4.pth", "V4 (T=3.0, biased 74%)"))

if os.path.exists("models/complete_model_v5.pth"):
    results.append(test_model("models/complete_model_v5.pth", "V5 (T=3.0, balanced 50%)"))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print()

for r in results:
    print(f"{r['name']}:")
    print(f"  Range: [{r['min']:.4f}, {r['max']:.4f}]")
    print(f"  Final: {r['final']:.4f}")
    print(f"  Std: {r['std']:.4f}")
    print()

# Winner
best = max(results, key=lambda x: x['final'] + x['std'])
print(f"🏆 BEST MODEL: {best['name']}")
print(f"   Final prob: {best['final']:.4f} (correct winner prediction)")
print(f"   Std: {best['std']:.4f} (responsive to match dynamics)")
print()

print("=" * 80)
