"""
Quick test: what happens if we just change the temperature during inference
without retraining? This will show us the isolated effect of temperature.
"""

import torch
import numpy as np
import pandas as pd
from scripts.data_loader import load_points_multiple
from scripts.features import add_additional_features, add_match_labels
from scripts.new_model_nn import build_new_features
from scripts.pretrain_tennis_rules import TennisRulesNet
from sklearn.preprocessing import RobustScaler

print("="*80)
print("TEST: EFFECT OF TEMPERATURE ON EXISTING MODEL")
print("="*80)

# Load existing model
model_path = './models/complete_model_v2.pth'
checkpoint = torch.load(model_path, map_location='cpu')

model = TennisRulesNet(
    input_size=checkpoint['input_size'],
    hidden_sizes=checkpoint['hidden_sizes'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nOriginal model temperature: {checkpoint['temperature']}")

# Load test data
df = load_points_multiple(['data/2019-wimbledon-points.csv'])
df = df[df['match_id'] == '2019-wimbledon-1701']
df = add_additional_features(df)
df = add_match_labels(df)

X, y_match, y_set, y_game, weights, feature_names = build_new_features(df)

# Normalize
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Test different temperatures
temperatures = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0]

with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    output = model(X_tensor)
    
    if isinstance(output, tuple):
        prob_raw = torch.sigmoid(output[0]).squeeze()
    else:
        prob_raw = output['match'].squeeze()
    
    print(f"\nRaw output (no temperature):")
    print(f"  Min: {prob_raw.min():.4f}, Max: {prob_raw.max():.4f}, Mean: {prob_raw.mean():.4f}")
    
    for temp in temperatures:
        # Apply temperature scaling
        eps = 1e-7
        prob_clipped = torch.clamp(prob_raw, eps, 1-eps)
        p_temp = torch.pow(prob_clipped, 1.0/temp)
        one_minus_p_temp = torch.pow(1 - prob_clipped, 1.0/temp)
        prob_cal = p_temp / (p_temp + one_minus_p_temp + eps)
        
        print(f"\nTemperature = {temp}:")
        print(f"  Min: {prob_cal.min():.4f}, Max: {prob_cal.max():.4f}, Mean: {prob_cal.mean():.4f}")
        print(f"  Range width: {(prob_cal.max() - prob_cal.min()).item():.4f}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("- Lower temperature (1-3): predictions closer to raw output")
print("- Higher temperature (12+): predictions pushed toward 0.5")
print("- T=3.0 is a good balance: allows confidence without extreme values")
