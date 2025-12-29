"""
Quick test of the pre-trained model with T=3.0 to see if temperature reduction works.
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
print("TESTING PRE-TRAINED MODEL (T=3.0)")
print("="*80)

# Load pre-trained model
model_path = './models/tennis_rules_pretrained_v3.pth'
checkpoint = torch.load(model_path, map_location='cpu')

model = TennisRulesNet(
    input_size=checkpoint['input_size'],
    hidden_sizes=checkpoint['hidden_sizes'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel info:")
print(f"  Temperature: {checkpoint['temperature']}")
print(f"  Architecture: {checkpoint['hidden_sizes']}")
print(f"  Trained on: {checkpoint.get('n_training_matches', '?')} synthetic matches")

# Load test data
df = load_points_multiple(['data/2019-wimbledon-points.csv'])
df = df[df['match_id'] == '2019-wimbledon-1701']
print(f"\nLoaded {len(df)} points from match 2019-wimbledon-1701")

df = add_additional_features(df)
df = add_match_labels(df)

X, y_match, y_set, y_game, weights, feature_names = build_new_features(df)

# Normalize
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Identify critical points
is_mp_p1 = X[:, 23]
is_mp_p2 = X[:, 24]
match_points = (is_mp_p1 > 0.5) | (is_mp_p2 > 0.5)

print(f"\nCritical points: {match_points.sum()} match points")

# Make predictions
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    output = model(X_tensor)
    
    if isinstance(output, dict):
        prob_raw = output['match'].squeeze().numpy()
    elif isinstance(output, tuple):
        prob_raw = torch.sigmoid(output[0]).squeeze().numpy()
    else:
        prob_raw = output.squeeze().numpy()

print(f"\nRaw probabilities (no temperature):")
print(f"  Min: {prob_raw.min():.4f}, Max: {prob_raw.max():.4f}, Mean: {prob_raw.mean():.4f}")

# Apply temperature scaling T=3.0
temp = checkpoint['temperature']
eps = 1e-7
prob_clipped = np.clip(prob_raw, eps, 1-eps)
p_temp = np.power(prob_clipped, 1.0/temp)
one_minus_p_temp = np.power(1 - prob_clipped, 1.0/temp)
prob_cal = p_temp / (p_temp + one_minus_p_temp + eps)

print(f"\nTemperature-calibrated (T={temp}):")
print(f"  Min: {prob_cal.min():.4f}, Max: {prob_cal.max():.4f}, Mean: {prob_cal.mean():.4f}")

if match_points.sum() > 0:
    mp_probs = prob_cal[match_points]
    print(f"\nMatch points (n={match_points.sum()}):")
    print(f"  Min: {mp_probs.min():.4f}, Max: {mp_probs.max():.4f}, Mean: {mp_probs.mean():.4f}")

print("\n" + "="*80)
print("COMPARISON WITH OLD MODEL (T=12.0):")
print("="*80)
print(f"Old model (T=12.0): range [0.21, 0.79]")
print(f"New model (T=3.0):  range [{prob_cal.min():.2f}, {prob_cal.max():.2f}]")
print(f"\nImprovement: {(prob_cal.max() - prob_cal.min()) / (0.79 - 0.21) * 100:.1f}% wider range")
