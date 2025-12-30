"""Test if model v4 understands set scores"""
import torch
import numpy as np
from scripts.pretrain_tennis_rules import TennisRulesNet

print('='*80)
print('TEST: MODEL V4 - DOES IT UNDERSTAND SET SCORES?')
print('='*80)

# Load model v4
checkpoint = torch.load('models/complete_model_v4.pth', map_location='cpu')
model = TennisRulesNet(
    input_size=checkpoint['input_size'],
    hidden_sizes=checkpoint['hidden_sizes'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel v4: {checkpoint['hidden_sizes']}, T={checkpoint['temperature']}")
print(f"Input features: {checkpoint['input_size']}")

def create_test_point(p1_sets=0, p2_sets=0, p1_games=0, p2_games=0):
    features = np.zeros(31)
    features[0] = 0  # P1_points
    features[1] = 0  # P2_points  
    features[2] = p1_games  # P1_games
    features[3] = p2_games  # P2_games
    features[4] = p1_sets  # P1_sets
    features[5] = p2_sets  # P2_sets
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
        prob = output['match'][0].item()
    
    # Apply temperature
    temp = checkpoint['temperature']
    eps = 1e-7
    prob_clipped = np.clip(prob, eps, 1-eps)
    p_temp = np.power(prob_clipped, 1.0/temp)
    one_minus_p_temp = np.power(1 - prob_clipped, 1.0/temp)
    prob_cal = p_temp / (p_temp + one_minus_p_temp + eps)
    
    results.append((desc, p1_sets, p2_sets, prob, prob_cal))
    
    expected = "~0.50"
    if p1_sets > p2_sets:
        if p1_sets == 3:
            expected = "~0.90+"
        elif p1_sets == 2 and p2_sets == 0:
            expected = "~0.75-0.80"
        else:
            expected = f"~{0.50 + 0.10 * (p1_sets - p2_sets):.2f}"
    elif p2_sets > p1_sets:
        if p2_sets == 3:
            expected = "~0.10-"
        elif p2_sets == 2 and p1_sets == 0:
            expected = "~0.20-0.25"
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

probs_p1_ahead = [r[4] for r in results if r[1] > r[2]]
probs_p2_ahead = [r[4] for r in results if r[2] > r[1]]
probs_tied = [r[4] for r in results if r[1] == r[2]]

print(f"\nWhen P1 ahead in sets: mean prob = {np.mean(probs_p1_ahead):.3f}")
print(f"When P2 ahead in sets: mean prob = {np.mean(probs_p2_ahead):.3f}")
print(f"When tied: mean prob = {np.mean(probs_tied):.3f}")

if np.mean(probs_p1_ahead) - np.mean(probs_p2_ahead) < 0.2:
    print("\n❌ MODEL V4 DOES NOT UNDERSTAND SET SCORES!")
    print("   Difference too small. Features P1_sets/P2_sets not being used.")
else:
    print("\n✓ Model v4 responds to set scores")

# Check raw outputs
raw_outputs = [r[3] for r in results]
print(f"\nRaw output statistics:")
print(f"  Mean: {np.mean(raw_outputs):.3f}")
print(f"  Std: {np.std(raw_outputs):.3f}")
print(f"  Range: [{np.min(raw_outputs):.3f}, {np.max(raw_outputs):.3f}]")

if np.std(raw_outputs) < 0.05:
    print("  ❌ Raw outputs are SATURATED (std < 0.05)")
else:
    print("  ✓ Raw outputs show variability")
