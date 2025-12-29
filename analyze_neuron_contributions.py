"""
Find which neurons in layer 2 are causing the explosion.
"""
import torch
import numpy as np
from scripts.pretrain_tennis_rules import TennisRulesNet

print('='*80)
print('NEURON-LEVEL ANALYSIS')
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

# Test with all zeros
X = torch.zeros(1, 31)

with torch.no_grad():
    # Get layer 2 output
    x = torch.relu(model.bn1(model.fc1(X)))
    x = model.dropout1(x)
    x = torch.relu(model.bn2(model.fc2(x)))
    layer2_output = model.dropout2(x)  # Shape: (1, 64)
    
    print(f"Layer 2 output shape: {layer2_output.shape}")
    print(f"Layer 2 activations:")
    print(f"  Mean: {layer2_output.mean().item():.4f}")
    print(f"  Std: {layer2_output.std().item():.4f}")
    print(f"  Range: [{layer2_output.min().item():.4f}, {layer2_output.max().item():.4f}]")
    
    # Get match head weights
    match_weights = model.match_head.weight.data.squeeze()  # Shape: (64,)
    match_bias = model.match_head.bias.data.item()
    
    print(f"\nMatch head weights shape: {match_weights.shape}")
    print(f"Match head weights stats:")
    print(f"  Mean: {match_weights.mean().item():.4f}")
    print(f"  Std: {match_weights.std().item():.4f}")
    print(f"  Range: [{match_weights.min().item():.4f}, {match_weights.max().item():.4f}]")
    
    # Compute contribution of each neuron
    contributions = layer2_output.squeeze() * match_weights
    
    print(f"\nNeuron contributions to final logit:")
    print(f"  Contribution sum: {contributions.sum().item():.4f}")
    print(f"  Bias: {match_bias:.4f}")
    print(f"  Total logit: {contributions.sum().item() + match_bias:.4f}")
    
    # Find top contributing neurons
    contrib_sorted, indices = torch.sort(contributions.abs(), descending=True)
    
    print(f"\nTop 10 neurons by absolute contribution:")
    for i in range(10):
        idx = indices[i].item()
        act = layer2_output[0, idx].item()
        weight = match_weights[idx].item()
        contrib = contributions[idx].item()
        print(f"  Neuron {idx:2d}: activation={act:6.3f}, weight={weight:7.3f}, contribution={contrib:7.3f}")
    
    # Check if a few neurons dominate
    top5_contrib = contributions[indices[:5]].abs().sum().item()
    total_contrib = contributions.abs().sum().item()
    
    print(f"\nTop 5 neurons contribute: {top5_contrib:.2f} out of {total_contrib:.2f} ({top5_contrib/total_contrib*100:.1f}%)")
    
    # Test: what if we zero out the top contributor?
    print("\n" + "="*80)
    print("EXPERIMENT: Zero out top contributing neuron")
    print("="*80)
    
    layer2_modified = layer2_output.clone()
    top_neuron = indices[0].item()
    layer2_modified[0, top_neuron] = 0.0
    
    logit_original = model.match_head(layer2_output)[0].item()
    logit_modified = model.match_head(layer2_modified)[0].item()
    
    print(f"Original logit: {logit_original:.4f}")
    print(f"Modified logit (neuron {top_neuron} zeroed): {logit_modified:.4f}")
    print(f"Difference: {logit_original - logit_modified:.4f}")
    print(f"Sigmoid(original): {torch.sigmoid(torch.tensor(logit_original)).item():.6f}")
    print(f"Sigmoid(modified): {torch.sigmoid(torch.tensor(logit_modified)).item():.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The match head has learned extreme weights that cause saturation.")
print("This makes the model insensitive to input changes.")
print("")
print("Possible causes:")
print("1. Training data had severe imbalance in pre-training")
print("2. Learning rate too high caused weight explosion")
print("3. No weight regularization (L2 penalty)")
print("4. Match point/set point penalties in loss function forced extreme confidencè")
