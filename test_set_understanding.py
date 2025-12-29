"""
Test: Does the model understand set score?
Feed it synthetic data with different set scores and see how it responds.
"""
import torch
import numpy as np
from scripts.pretrain_tennis_rules import TennisRulesNet

print('='*80)
print('TEST: DOES MODEL UNDERSTAND SET SCORES?')
print('='*80)

# Load model
checkpoint = torch.load('models/complete_model_v2.pth', map_location='cpu')
model = TennisRulesNet(
    input_size=checkpoint['input_size'],
    hidden_sizes=checkpoint['hidden_sizes'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel: {checkpoint['hidden_sizes']}, T={checkpoint['temperature']}")
print(f"Input features: {checkpoint['input_size']}")

# Create synthetic test points with controlled set scores
# Feature order should match FEATURE_COLUMNS in new_model_nn.py
# Let's create a baseline point (0-0, 0-0 in games, 0-0 in sets)

def create_test_point(p1_sets=0, p2_sets=0, p1_games=0, p2_games=0):
    """Create a synthetic point with specific score."""
    # 31 features total - we'll set most to 0 and key ones to specific values
    # Based on FEATURE_COLUMNS order:
    # 0-3: P1_points, P2_points, P1_games, P2_games
    # 4-5: P1_sets, P2_sets
    # ... rest are various contextual features
    
    features = np.zeros(31)
    features[0] = 0  # P1_points
    features[1] = 0  # P2_points  
    features[2] = p1_games  # P1_games
    features[3] = p2_games  # P2_games
    features[4] = p1_sets  # P1_sets
    features[5] = p2_sets  # P2_sets
    # features[6] = 1  # P1_serving (let's say P1 serves)
    
    return features

print("\n=== TEST SCENARIOS ===")

scenarios = [
    ("Match start (0-0 sets)", 0, 0),
    ("P1 leads 1-0 sets", 1, 0),
    ("P1 leads 2-0 sets", 2, 0),
    ("P1 up 2-1 sets", 2, 1),
    ("P1 leads 3-1 sets (closing)", 3, 1),
    ("Tied 1-1 sets", 1, 1),
    ("Tied 2-2 sets (5th set)", 2, 2),
    ("P2 leads 1-0 sets", 0, 1),
    ("P2 leads 2-0 sets", 0, 2),
]

results = []
for desc, p1_sets, p2_sets in scenarios:
    point = create_test_point(p1_sets, p2_sets)
    with torch.no_grad():
        X = torch.FloatTensor(point).unsqueeze(0)
        output = model(X)
        if isinstance(output, dict):
            prob = output['match'][0].item()
        elif isinstance(output, tuple):
            prob = torch.sigmoid(output[0])[0].item()
        else:
            prob = output[0].item()
    
    results.append((desc, p1_sets, p2_sets, prob))
    
    # Apply temperature scaling like in inference
    temp = checkpoint['temperature']
    eps = 1e-7
    prob_clipped = np.clip(prob, eps, 1-eps)
    p_temp = np.power(prob_clipped, 1.0/temp)
    one_minus_p_temp = np.power(1 - prob_clipped, 1.0/temp)
    prob_cal = p_temp / (p_temp + one_minus_p_temp + eps)
    
    expected = "~0.50"
    if p1_sets > p2_sets:
        if p1_sets == 2 and p2_sets == 0:
            expected = "~0.75-0.80"
        elif p1_sets == 3:
            expected = "~0.90+"
        else:
            expected = f"~{0.50 + 0.10 * (p1_sets - p2_sets):.2f}"
    elif p2_sets > p1_sets:
        if p2_sets == 2 and p1_sets == 0:
            expected = "~0.20-0.25"
        elif p2_sets == 3:
            expected = "~0.10-"
        else:
            expected = f"~{0.50 - 0.10 * (p2_sets - p1_sets):.2f}"
    
    print(f"\n{desc}:")
    print(f"  Set score: {p1_sets}-{p2_sets}")
    print(f"  Raw output: {prob:.4f}")
    print(f"  After T={temp}: {prob_cal:.4f}")
    print(f"  Expected: {expected}")
    
    if p1_sets > p2_sets and prob_cal < 0.55:
        print(f"  ⚠️  ERROR: P1 leads but prob={prob_cal:.3f} (should be >0.55)")
    elif p2_sets > p1_sets and prob_cal > 0.45:
        print(f"  ⚠️  ERROR: P2 leads but prob={prob_cal:.3f} (should be <0.45)")
    elif p1_sets == 3 and prob_cal < 0.85:
        print(f"  ⚠️  CRITICAL: P1 one set away but prob={prob_cal:.3f} (should be >0.85)")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

# Check if model responds to set differences
probs_p1_ahead = [r[3] for r in results if r[1] > r[2]]
probs_p2_ahead = [r[3] for r in results if r[2] > r[1]]
probs_tied = [r[3] for r in results if r[1] == r[2]]

if probs_p1_ahead and probs_p2_ahead:
    print(f"\nWhen P1 ahead in sets: mean prob = {np.mean(probs_p1_ahead):.3f}")
    print(f"When P2 ahead in sets: mean prob = {np.mean(probs_p2_ahead):.3f}")
    print(f"When tied: mean prob = {np.mean(probs_tied):.3f}")
    
    if np.mean(probs_p1_ahead) - np.mean(probs_p2_ahead) < 0.2:
        print("\n❌ MODEL DOES NOT UNDERSTAND SET SCORES!")
        print("   Difference too small. Model is not using P1_sets/P2_sets features effectively.")
    else:
        print("\n✓ Model responds to set scores (but may need stronger signal)")
