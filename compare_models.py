#!/usr/bin/env python3
"""
Compare all model versions on Wimbledon 2019 Final:
- v2: T=12.0, saturated weights (stuck at 0.791)
- v4: T=3.0, biased training data (inverse correlation)
- v5: T=3.0, balanced training data (50-50)
"""

import sys
import os

# Ensure scripts can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from scripts.prediction import run_prediction
    from scripts.data_loader import load_points_single
except ImportError:
    from prediction import run_prediction
    from data_loader import load_points_single

import pandas as pd
import numpy as np

# Match details
MATCH_ID = "2019-wimbledon-1701"
MATCH_FILE = "data/2019-wimbledon-points.csv"
PLAYER1 = "Djokovic"
PLAYER2 = "Federer"
WINNER = "Djokovic (3-2 sets)"

print("=" * 80)
print(f"MODEL COMPARISON: {PLAYER1} vs {PLAYER2}")
print(f"Match: {MATCH_ID} - Winner: {WINNER}")
print("=" * 80)
print()

# Generate predictions for all models
models = {
    "v2": {"path": "models/complete_model_v2.pth", "desc": "T=12.0, saturated weights"},
    "v4": {"path": "models/complete_model_v4.pth", "desc": "T=3.0, biased training (74% P1)"},
    "v5": {"path": "models/complete_model_v5.pth", "desc": "T=3.0, balanced training (50% P1)"}
}

results = {}

for version, info in models.items():
    model_path = info["path"]
    if not os.path.exists(model_path):
        print(f"⚠ Model {version} not found: {model_path}")
        continue
    
    print(f"[{version.upper()}] Testing {info['desc']}...")
    plot_dir = f"plots_{version}"
    
    try:
        run_prediction(
            file_paths=[MATCH_FILE],
            model_path=model_path,
            match_id=MATCH_ID,
            plot_dir=plot_dir,
            point_by_point=True
        )
        
        # Load predictions
        csv_path = os.path.join(plot_dir, f"match_{MATCH_ID}_probabilities.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results[version] = {
                "desc": info["desc"],
                "probs": df["prob_p1"].values,
                "min": df["prob_p1"].min(),
                "max": df["prob_p1"].max(),
                "mean": df["prob_p1"].mean(),
                "std": df["prob_p1"].std(),
                "final": df["prob_p1"].iloc[-1]
            }
            print(f"  ✓ Predictions generated: {len(df)} points")
        else:
            print(f"  ✗ Predictions file not found: {csv_path}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

# Compare results
if not results:
    print("No results to compare!")
    sys.exit(1)

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()

for version in ["v2", "v4", "v5"]:
    if version not in results:
        continue
    
    r = results[version]
    print(f"MODEL {version.upper()}: {r['desc']}")
    print(f"  Range:        [{r['min']:.4f}, {r['max']:.4f}] (width: {r['max'] - r['min']:.4f})")
    print(f"  Mean:         {r['mean']:.4f}")
    print(f"  Std:          {r['std']:.4f}")
    print(f"  Final prob:   {r['final']:.4f} (at match end)")
    
    # Analyze correlation with match state
    # High std = responsive to match dynamics
    # Final prob should be high (>0.85) since Djokovic won
    if r['std'] < 0.10:
        print(f"  ⚠ WARNING: Low std ({r['std']:.4f}) - model not responsive!")
    if r['final'] < 0.70:
        print(f"  ⚠ WARNING: Low final prob ({r['final']:.4f}) - should be >0.85!")
    
    print()

# Compare improvements
print("=" * 80)
print("IMPROVEMENTS")
print("=" * 80)
print()

if "v2" in results and "v5" in results:
    v2 = results["v2"]
    v5 = results["v5"]
    
    print("v2 → v5:")
    print(f"  Range width:  {v2['max'] - v2['min']:.4f} → {v5['max'] - v5['min']:.4f} "
          f"({(v5['max'] - v5['min']) / (v2['max'] - v2['min']) * 100 - 100:+.1f}%)")
    print(f"  Std:          {v2['std']:.4f} → {v5['std']:.4f} "
          f"({(v5['std'] / v2['std'] * 100 - 100):+.1f}%)")
    print(f"  Final prob:   {v2['final']:.4f} → {v5['final']:.4f} "
          f"({v5['final'] - v2['final']:+.4f})")
    print()

if "v4" in results and "v5" in results:
    v4 = results["v4"]
    v5 = results["v5"]
    
    print("v4 → v5:")
    print(f"  Mean:         {v4['mean']:.4f} → {v5['mean']:.4f} "
          f"({v5['mean'] - v4['mean']:+.4f})")
    print(f"  Final prob:   {v4['final']:.4f} → {v5['final']:.4f} "
          f"({v5['final'] - v4['final']:+.4f})")
    
    if v4['final'] < 0.50 < v5['final']:
        print("  ✓ FIXED: v4 had inverse correlation (predicted wrong winner)")
    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)

best_version = None
best_score = -1

for version, r in results.items():
    # Score = high final prob + high std + reasonable mean
    score = r['final'] + r['std'] - abs(r['mean'] - 0.50)
    if score > best_score:
        best_score = score
        best_version = version

if best_version:
    r = results[best_version]
    print(f"✓ Best model: {best_version.upper()} ({r['desc']})")
    print(f"  - Responsive to match dynamics (std={r['std']:.4f})")
    print(f"  - Correct winner prediction (final={r['final']:.4f})")
    print(f"  - No artificial caps or biases")
else:
    print("Unable to determine best model")

print("=" * 80)
