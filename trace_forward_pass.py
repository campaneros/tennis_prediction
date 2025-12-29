"""
Deep dive into model internals to find the explosion point.
"""
import torch
import numpy as np
from scripts.pretrain_tennis_rules import TennisRulesNet

print('='*80)
print('TRACING MODEL FORWARD PASS')
print('='*80)

# Load model
checkpoint = torch.load('models/complete_model_v2.pth', map_location='cpu')
model = TennisRulesNet(
    input_size=checkpoint['input_size'],
    hidden_sizes=checkpoint['hidden_sizes'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # IMPORTANT: eval mode

print(f"Model in eval mode: {not model.training}")

# Test with all zeros
X = torch.zeros(1, 31)

print("\n" + "="*80)
print("FORWARD PASS STEP BY STEP (all zeros input)")
print("="*80)

with torch.no_grad():
    # Layer 1
    print("\n1. First linear layer (fc1):")
    z1 = model.fc1(X)
    print(f"   Output shape: {z1.shape}")
    print(f"   Output stats: mean={z1.mean().item():.4f}, std={z1.std().item():.4f}")
    print(f"   Output range: [{z1.min().item():.4f}, {z1.max().item():.4f}]")
    
    # Batch norm 1
    print("\n2. First batch norm (bn1):")
    bn1_out = model.bn1(z1)
    print(f"   Output shape: {bn1_out.shape}")
    print(f"   Output stats: mean={bn1_out.mean().item():.4f}, std={bn1_out.std().item():.4f}")
    print(f"   Output range: [{bn1_out.min().item():.4f}, {bn1_out.max().item():.4f}]")
    
    # Check batch norm running stats
    print(f"\n   BN1 running_mean: mean={model.bn1.running_mean.mean().item():.4f}, std={model.bn1.running_mean.std().item():.4f}")
    print(f"   BN1 running_var: mean={model.bn1.running_var.mean().item():.4f}, std={model.bn1.running_var.std().item():.4f}")
    print(f"   BN1 running_var range: [{model.bn1.running_var.min().item():.6f}, {model.bn1.running_var.max().item():.6f}]")
    
    # Check if any variance is too small (causes explosion)
    if (model.bn1.running_var < 0.001).any():
        print(f"   ⚠️  WARNING: {(model.bn1.running_var < 0.001).sum()} neurons have variance < 0.001")
        print(f"   This causes explosion during normalization!")
    
    # ReLU 1
    print("\n3. First ReLU:")
    relu1_out = torch.relu(bn1_out)
    print(f"   Output range: [{relu1_out.min().item():.4f}, {relu1_out.max().item():.4f}]")
    print(f"   Fraction active: {(relu1_out > 0).float().mean().item():.3f}")
    
    # Dropout 1 (should be no-op in eval mode)
    print("\n4. First dropout (should be identity in eval mode):")
    drop1_out = model.dropout1(relu1_out)
    print(f"   Changed: {not torch.allclose(drop1_out, relu1_out)}")
    
    # Layer 2
    print("\n5. Second linear layer (fc2):")
    z2 = model.fc2(drop1_out)
    print(f"   Output shape: {z2.shape}")
    print(f"   Output stats: mean={z2.mean().item():.4f}, std={z2.std().item():.4f}")
    print(f"   Output range: [{z2.min().item():.4f}, {z2.max().item():.4f}]")
    
    # Batch norm 2
    print("\n6. Second batch norm (bn2):")
    bn2_out = model.bn2(z2)
    print(f"   Output shape: {bn2_out.shape}")
    print(f"   Output stats: mean={bn2_out.mean().item():.4f}, std={bn2_out.std().item():.4f}")
    print(f"   Output range: [{bn2_out.min().item():.4f}, {bn2_out.max().item():.4f}]")
    
    # Check batch norm 2 running stats
    print(f"\n   BN2 running_mean: mean={model.bn2.running_mean.mean().item():.4f}")
    print(f"   BN2 running_var: mean={model.bn2.running_var.mean().item():.4f}")
    print(f"   BN2 running_var range: [{model.bn2.running_var.min().item():.6f}, {model.bn2.running_var.max().item():.6f}]")
    
    if (model.bn2.running_var < 0.001).any():
        print(f"   ⚠️  WARNING: {(model.bn2.running_var < 0.001).sum()} neurons have variance < 0.001")
    
    # ReLU 2
    print("\n7. Second ReLU:")
    relu2_out = torch.relu(bn2_out)
    print(f"   Output range: [{relu2_out.min().item():.4f}, {relu2_out.max().item():.4f}]")
    print(f"   Fraction active: {(relu2_out > 0).float().mean().item():.3f}")
    
    # Dropout 2
    print("\n8. Second dropout:")
    drop2_out = model.dropout2(relu2_out)
    
    # Match head
    print("\n9. Match head (final layer):")
    match_logits = model.match_head(drop2_out)
    print(f"   Logits: {match_logits[0].item():.4f}")
    print(f"   Sigmoid(logits): {torch.sigmoid(match_logits)[0].item():.6f}")
    
    # Check match head weights
    print(f"\n   Match head weight stats:")
    print(f"     mean={model.match_head.weight.mean().item():.4f}")
    print(f"     std={model.match_head.weight.std().item():.4f}")
    print(f"     range=[{model.match_head.weight.min().item():.4f}, {model.match_head.weight.max().item():.4f}]")
    print(f"   Match head bias: {model.match_head.bias.item():.4f}")
    
    # Manual calculation
    manual_logit = (drop2_out @ model.match_head.weight.T).item() + model.match_head.bias.item()
    print(f"   Manual calculation check: {manual_logit:.4f}")
    
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Find the problem layer
if bn1_out.abs().max() > 10:
    print("❌ PROBLEM: First batch norm (bn1) produces huge values!")
    print("   This is causing the explosion.")
elif relu2_out.abs().max() > 10:
    print("❌ PROBLEM: Second layer output is too large!")
elif match_logits.abs().max() > 10:
    print("❌ PROBLEM: Match head produces huge logits!")
    if drop2_out.abs().max() < 5:
        print("   The input to match head is reasonable, so the problem is in match head weights.")
else:
    print("❓ Problem unclear from this analysis.")
