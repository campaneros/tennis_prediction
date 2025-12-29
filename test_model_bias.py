"""
Find out WHY model always predicts 1.0 for most inputs.
Check if there's a systematic bias in the model.
"""
import torch
import numpy as np
from scripts.pretrain_tennis_rules import TennisRulesNet

print('='*80)
print('INVESTIGATING MODEL BIAS')
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

print(f"\nModel architecture:")
print(f"  Input: {checkpoint['input_size']} features")
print(f"  Hidden: {checkpoint['hidden_sizes']}")
print(f"  Dropout: {checkpoint['dropout']}")
print(f"  Temperature: {checkpoint['temperature']}")

# Test 1: All zeros
print("\n" + "="*80)
print("TEST 1: All zeros (no information)")
print("="*80)
X = torch.zeros(1, 31)
with torch.no_grad():
    output = model(X)
    if isinstance(output, dict):
        prob = output['match'][0].item()
        # Compute intermediate activations
        x = torch.relu(model.bn1(model.fc1(X)))
        x = model.dropout1(x)
        x = torch.relu(model.bn2(model.fc2(x)))
        x = model.dropout2(x)
        logit_match = model.match_head(x)[0].item()
    else:
        prob = torch.sigmoid(output)[0].item() if isinstance(output, tuple) else output[0].item()
        logit_match = None

print(f"Raw output (after sigmoid): {prob:.6f}")
if logit_match is not None:
    print(f"Logit before sigmoid: {logit_match:.6f}")
    print(f"Sigmoid(logit): {torch.sigmoid(torch.tensor(logit_match)).item():.6f}")

# Test 2: Check which features matter
print("\n" + "="*80)
print("TEST 2: Feature importance via perturbation")
print("="*80)

baseline = torch.zeros(1, 31)
with torch.no_grad():
    baseline_output = model(baseline)
    if isinstance(baseline_output, dict):
        baseline_prob = baseline_output['match'][0].item()
    else:
        baseline_prob = torch.sigmoid(baseline_output)[0].item() if isinstance(baseline_output, tuple) else baseline_output[0].item()

print(f"Baseline (all zeros): {baseline_prob:.4f}")

# Check features individually
feature_names = [
    'P1_points', 'P2_points', 'P1_games', 'P2_games', 'P1_sets', 'P2_sets',
    'P1_serving', 'is_tiebreak', 'fifth_set', 'p1_distance_to_game', 
    'p2_distance_to_game', 'p1_distance_to_set', 'p2_distance_to_set',
    'p1_sets_to_win_match', 'p2_sets_to_win_match', 'p1_games_to_win_set',
    'p2_games_to_win_set', 'is_break_point', 'is_set_point', 'is_match_point',
    'p1_break_point_opp', 'p2_break_point_opp', 'p1_set_point_opp',
    'p2_set_point_opp', 'p1_match_point_opp', 'p2_match_point_opp',
    'game_in_set', 'total_points', 'p1_win_pct', 'p2_win_pct', 'momentum'
]

print("\nTop 10 most impactful features (setting to 1.0):")
impacts = []
for i, name in enumerate(feature_names):
    test_input = baseline.clone()
    test_input[0, i] = 1.0
    with torch.no_grad():
        output = model(test_input)
        if isinstance(output, dict):
            prob = output['match'][0].item()
        else:
            prob = torch.sigmoid(output)[0].item() if isinstance(output, tuple) else output[0].item()
    
    impact = prob - baseline_prob
    impacts.append((name, impact, prob))

# Sort by absolute impact
impacts.sort(key=lambda x: abs(x[1]), reverse=True)

for i, (name, impact, prob) in enumerate(impacts[:10]):
    print(f"{i+1:2d}. {name:25s}: impact={impact:+.4f}, prob={prob:.4f}")

# Test 3: Specific problematic case
print("\n" + "="*80)
print("TEST 3: Realistic match scenario")
print("="*80)

# P2 leads 2-0 in sets, 5-4 in games (about to win match)
test_case = torch.zeros(1, 31)
test_case[0, 2] = 4   # P1_games
test_case[0, 3] = 5   # P2_games  
test_case[0, 4] = 0   # P1_sets
test_case[0, 5] = 2   # P2_sets
test_case[0, 6] = 0   # P1_serving (P2 serves)
test_case[0, 8] = 0   # not fifth_set
test_case[0, 13] = 3  # p1_sets_to_win_match
test_case[0, 14] = 1  # p2_sets_to_win_match (one set away!)

with torch.no_grad():
    output = model(test_case)
    if isinstance(output, dict):
        prob = output['match'][0].item()
    else:
        prob = torch.sigmoid(output)[0].item() if isinstance(output, tuple) else output[0].item()

print(f"P2 leads 2-0 sets, 5-4 games (serving for match)")
print(f"Model prediction: {prob:.4f}")
print(f"Expected: ~0.10-0.20 (P2 about to win)")

if prob > 0.5:
    print("❌ MODEL IS BROKEN: P2 is clearly winning but model says P1 favored!")

# Test 4: Check model weights
print("\n" + "="*80)
print("TEST 4: Inspect first layer weights")  
print("="*80)

first_layer = model.fc1.weight.data.numpy()  # Shape: [128, 31]
print(f"First layer weight matrix shape: {first_layer.shape}")

# Check if certain input features have abnormally large weights
mean_weights = np.mean(np.abs(first_layer), axis=0)  # Average magnitude per input feature
print(f"\nMean absolute weight per input feature:")
for i in range(min(10, len(mean_weights))):
    print(f"  Feature {i} ({feature_names[i] if i < len(feature_names) else 'unknown'}): {mean_weights[i]:.4f}")

# Check bias
bias = model.fc1.bias.data.numpy()
print(f"\nFirst layer bias statistics:")
print(f"  Mean: {np.mean(bias):.4f}")
print(f"  Std: {np.std(bias):.4f}")
print(f"  Min: {np.min(bias):.4f}")
print(f"  Max: {np.max(bias):.4f}")

# Final layer
final_bias = model.match_head.bias.data.numpy()[0]
final_weight = model.match_head.weight.data.numpy()[0]
print(f"\nFinal layer (match head) bias: {final_bias:.4f}")
print(f"This bias alone gives sigmoid(bias) = {torch.sigmoid(torch.tensor(final_bias)).item():.4f}")
print(f"Match head weight stats: mean={np.mean(final_weight):.4f}, std={np.std(final_weight):.4f}")
print(f"Match head weight max={np.max(final_weight):.4f}, min={np.min(final_weight):.4f}")
