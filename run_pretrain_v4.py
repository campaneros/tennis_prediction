#!/usr/bin/env python3
"""Pre-train model on synthetic data with fixed hyperparameters"""
import sys
sys.path.insert(0, 'scripts')

import torch
import pandas as pd
from pretrain_tennis_rules import TennisRulesNet, compute_tennis_features, compute_labels, custom_loss_pretrain
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

print('='*80)
print('PRE-TRAINING ON SYNTHETIC DATA - V4')
print('='*80)

# Load synthetic data
print('\n[1/5] Loading synthetic data...')
df = pd.read_csv('data/synthetic_training_30k_v4.csv')
print(f'  Loaded {len(df):,} points from 30k matches')

# Compute features
print('\n[2/5] Computing features...')
X = compute_tennis_features(df)
print(f'  Feature matrix shape: {X.shape}')

# Compute labels
print('\n[3/5] Computing labels...')
y_match, y_set, y_game = compute_labels(df)
print(f'  Match label distribution: {y_match.mean():.3f}')

# Compute sample weights (all equal for now)
weights = torch.ones(len(X))

# Create dataset
device = 'cpu'  # Use CPU
print(f'\n[4/5] Setting up training (device: {device})...')

X_tensor = torch.FloatTensor(X).to(device)
y_match_tensor = torch.FloatTensor(y_match).to(device)
y_set_tensor = torch.FloatTensor(y_set).to(device)
y_game_tensor = torch.FloatTensor(y_game).to(device)
weights_tensor = weights.to(device)

dataset = TensorDataset(X_tensor, y_match_tensor, y_set_tensor, y_game_tensor, weights_tensor)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Initialize model
model = TennisRulesNet(input_size=31, hidden_sizes=[128, 64], dropout=0.4)
model.to(device)

# Optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training
print(f'\n[5/5] Training for 20 epochs...')
print(f'  Temperature: 3.0')
print(f'  Weight decay: 0.01')
print(f'  Gradient clipping: max_norm=1.0')
print(f'  Match point penalty weight: 0.5 (was 5.0)')
print(f'  Set point penalty weight: 0.3 (was 2.0)')

temperature = 3.0
epochs = 20

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y_match, batch_y_set, batch_y_game, batch_weights in dataloader:
        optimizer.zero_grad()
        
        pred = model(batch_X)
        
        # Custom loss (no match/set point indicators in synthetic data)
        loss = custom_loss_pretrain(
            pred, batch_y_match, batch_y_set, batch_y_game,
            batch_weights, temperature=temperature
        )
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_loss = train_loss / len(dataloader)
    print(f'  Epoch {epoch+1:2d}/{epochs}: loss={avg_loss:.4f}')

# Save model
save_path = 'models/tennis_rules_pretrained_v4.pth'
print(f'\n[SAVE] Saving model to {save_path}...')

torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': 31,
    'hidden_sizes': [128, 64],
    'dropout': 0.4,
    'temperature': temperature,
    'epoch': epochs,
    'train_loss': avg_loss,
}, save_path)

print('✓ Pre-training complete!')
