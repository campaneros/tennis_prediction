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

# Dynamic calibration: flatten early sets, sharpen endgame (especially match/set points)
set_numbers = df['SetNo'].to_numpy()
max_set = set_numbers.max()
# Progress from 0 (start match) to 1 (end match)
match_progress = np.linspace(0, 1, len(prob_cal))
set_progress = (set_numbers - 1) / max(max_set - 1, 1)
progress = np.maximum(match_progress, set_progress)

logits = np.log(prob_cal / (1 - prob_cal + eps))

# Start flatter (0.75×), finish sharper (1.3×); interpolate in between
phase_factor = np.interp(progress, [0.0, 0.25, 0.65, 1.0], [0.75, 0.9, 1.15, 1.3])

is_set_point = (df.get('is_set_point', 0).to_numpy().astype(float) > 0.5)
clutch_factor = 1.0 + 0.45 * match_points + 0.2 * is_set_point

adj_logits = logits * phase_factor * clutch_factor
prob_final = 1 / (1 + np.exp(-adj_logits))

print(f"\nDynamic-calibrated (early flattened, clutch boosted):")
print(f"  Min: {prob_final.min():.4f}, Max: {prob_final.max():.4f}, Mean: {prob_final.mean():.4f}")

if match_points.sum() > 0:
    mp_probs = prob_final[match_points]
    print(f"  Match points (n={match_points.sum()}): min {mp_probs.min():.4f}, max {mp_probs.max():.4f}, mean {mp_probs.mean():.4f}")
if is_set_point.sum() > 0:
    sp_probs = prob_final[is_set_point]
    print(f"  Set points   (n={is_set_point.sum()}): min {sp_probs.min():.4f}, max {sp_probs.max():.4f}, mean {sp_probs.mean():.4f}")

print("\n" + "="*80)
print("COMPARISON WITH OLD MODEL (T=12.0):")
print("="*80)
print(f"Old model (T=12.0): range [0.21, 0.79]")
print(f"New model (T=3.0):  range [{prob_cal.min():.2f}, {prob_cal.max():.2f}]")
print(f"Dynamic model:      range [{prob_final.min():.2f}, {prob_final.max():.2f}]")
print(f"\nImprovement: {(prob_final.max() - prob_final.min()) / (0.79 - 0.21) * 100:.1f}% wider range")
