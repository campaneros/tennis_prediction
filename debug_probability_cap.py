"""
Debug script to understand where the 0.8 cap on probabilities comes from.
Analyzes:
1. Raw model outputs (before temperature)
2. Temperature-scaled outputs
3. Impact of loss constraints
4. Distribution of predictions on match points
"""

import torch
import numpy as np
import pandas as pd
from scripts.data_loader import load_points_multiple
from scripts.features import add_additional_features, add_match_labels
from scripts.new_model_nn import build_new_features
from scripts.model import load_model

# Load the model (PyTorch checkpoint)
print("="*80)
print("LOADING MODEL")
print("="*80)
model_path = './models/complete_model_v2.pth'

# Load PyTorch checkpoint
from scripts.pretrain_tennis_rules import TennisRulesNet

checkpoint = torch.load(model_path, map_location='cpu')
print(f"Checkpoint keys: {checkpoint.keys()}")

model = TennisRulesNet(
    input_size=checkpoint['input_size'],
    hidden_sizes=checkpoint['hidden_sizes'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check if scaler is in checkpoint
if 'scaler_center' in checkpoint:
    model.scaler_center = checkpoint['scaler_center']
    model.scaler_scale = checkpoint['scaler_scale']
else:
    print("  Warning: No scaler in checkpoint, will compute from data")
    model.scaler_center = None
    model.scaler_scale = None

model.temperature = checkpoint['temperature']

print(f"Model loaded from: {model_path}")
print(f"  Architecture: {checkpoint['hidden_sizes']}")
print(f"  Temperature: {model.temperature}")
print(f"  Dropout: {checkpoint['dropout']}")

# Load test match data
print("\n" + "="*80)
print("LOADING TEST MATCH DATA")
print("="*80)
df = load_points_multiple(['data/2019-wimbledon-points.csv'])
df = df[df['match_id'] == '2019-wimbledon-1701']
print(f"Loaded {len(df)} points from match 2019-wimbledon-1701")

# Add features
df = add_additional_features(df)
df = add_match_labels(df)

# Build features
X, y_match, y_set, y_game, weights, feature_names = build_new_features(df)
print(f"Built features: {X.shape}")

# Normalize
if model.scaler_center is not None:
    X_scaled = (X - model.scaler_center) / model.scaler_scale
else:
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

# Identify critical points
is_mp_p1 = X[:, 23]  # Match point P1
is_mp_p2 = X[:, 24]  # Match point P2
is_sp_p1 = X[:, 21]  # Set point P1
is_sp_p2 = X[:, 22]  # Set point P2

match_points = (is_mp_p1 > 0.5) | (is_mp_p2 > 0.5)
set_points = (is_sp_p1 > 0.5) | (is_sp_p2 > 0.5)

print(f"\nCritical points in match:")
print(f"  Match points: {match_points.sum()}")
print(f"  Set points (non-MP): {(set_points & ~match_points).sum()}")

# Make predictions with model
print("\n" + "="*80)
print("ANALYZING MODEL OUTPUTS")
print("="*80)

with torch.no_grad():
    model.eval()
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Get raw output from model
    output = model(X_tensor)
    
    if isinstance(output, dict):
        # Multi-task model
        prob_raw = output['match'].squeeze().numpy()
        print(f"Model output type: dict (multi-task)")
    elif isinstance(output, tuple):
        # Old format: (match_logits, set_logits, game_logits)
        prob_raw = torch.sigmoid(output[0]).squeeze().numpy()
        print(f"Model output type: tuple (old format)")
    else:
        prob_raw = output.squeeze().numpy()
        print(f"Model output type: tensor")
    
    print(f"\nRaw probabilities (after sigmoid, before any post-processing):")
    print(f"  Min: {prob_raw.min():.4f}")
    print(f"  Max: {prob_raw.max():.4f}")
    print(f"  Mean: {prob_raw.mean():.4f}")
    print(f"  Std: {prob_raw.std():.4f}")
    
    # Check if temperature is applied during inference
    if hasattr(model, 'temperature'):
        temp = model.temperature
        print(f"\nModel has temperature attribute: {temp}")
        
        # Manually apply temperature scaling to see effect
        eps = 1e-7
        prob_clipped = np.clip(prob_raw, eps, 1-eps)
        
        # Temperature scaling: p_cal = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        p_temp = np.power(prob_clipped, 1.0/temp)
        one_minus_p_temp = np.power(1 - prob_clipped, 1.0/temp)
        prob_calibrated = p_temp / (p_temp + one_minus_p_temp + eps)
        
        print(f"\nTemperature-calibrated probabilities (T={temp}):")
        print(f"  Min: {prob_calibrated.min():.4f}")
        print(f"  Max: {prob_calibrated.max():.4f}")
        print(f"  Mean: {prob_calibrated.mean():.4f}")
        print(f"  Std: {prob_calibrated.std():.4f}")
        
        # Compare
        print(f"\nEffect of temperature scaling:")
        print(f"  Max difference: {np.abs(prob_raw - prob_calibrated).max():.4f}")
        print(f"  Mean difference: {np.abs(prob_raw - prob_calibrated).mean():.4f}")

# Analyze predictions on critical points
print("\n" + "="*80)
print("PREDICTIONS ON CRITICAL POINTS")
print("="*80)

if match_points.sum() > 0:
    mp_probs = prob_raw[match_points]
    print(f"\nMatch points (n={match_points.sum()}):")
    print(f"  Min: {mp_probs.min():.4f}")
    print(f"  Max: {mp_probs.max():.4f}")
    print(f"  Mean: {mp_probs.mean():.4f}")
    print(f"  Values > 0.85: {(mp_probs > 0.85).sum()}")
    print(f"  Values > 0.90: {(mp_probs > 0.90).sum()}")
    print(f"  Values < 0.80: {(mp_probs < 0.80).sum()}")
    
    # Show distribution
    bins = [0, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    hist, _ = np.histogram(mp_probs, bins=bins)
    print(f"\n  Distribution:")
    for i in range(len(bins)-1):
        print(f"    {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]}")

sp_only = set_points & ~match_points
if sp_only.sum() > 0:
    sp_probs = prob_raw[sp_only]
    print(f"\nSet points (non-match, n={sp_only.sum()}):")
    print(f"  Min: {sp_probs.min():.4f}")
    print(f"  Max: {sp_probs.max():.4f}")
    print(f"  Mean: {sp_probs.mean():.4f}")
    print(f"  Values > 0.80: {(sp_probs > 0.80).sum()}")
    print(f"  Values > 0.75: {(sp_probs > 0.75).sum()}")
    print(f"  Values < 0.65: {(sp_probs < 0.65).sum()}")

# Regular points
regular = ~match_points & ~set_points
if regular.sum() > 0:
    reg_probs = prob_raw[regular]
    print(f"\nRegular points (n={regular.sum()}):")
    print(f"  Min: {reg_probs.min():.4f}")
    print(f"  Max: {reg_probs.max():.4f}")
    print(f"  Mean: {reg_probs.mean():.4f}")
    print(f"  Values > 0.80: {(reg_probs > 0.80).sum()}")
    print(f"  Values > 0.90: {(reg_probs > 0.90).sum()}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nPossible causes of probability capping:")
print("1. Temperature scaling pulling predictions toward 0.5")
print("2. Loss constraints during training (match point >= 0.85, set point ~0.70)")
print("3. Sample weights not being high enough for critical points")
print("4. Network architecture limiting expressiveness")
print("\nRecommended next steps:")
if prob_raw.max() < 0.85:
    print("  → Raw outputs already capped! Problem is in TRAINING, not inference")
    print("  → Likely cause: loss constraints (match_point_penalty, set_point_penalty)")
elif hasattr(model, 'temperature') and model.temperature > 5:
    print("  → Temperature scaling is significant (T={model.temperature})")
    print("  → Consider reducing temperature or removing it for inference")
else:
    print("  → Check if temperature scaling is applied in predict_with_model()")
