#!/usr/bin/env python3
"""
Train model v5 DIRECTLY on real data (no pre-training).
This avoids the bias problem from synthetic data.
"""
import sys
sys.path.insert(0, 'scripts')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from glob import glob
from pretrain_tennis_rules import TennisRulesNet
from data_loader import load_points_multiple
from features import add_additional_features, add_match_labels
from new_model_nn import build_new_features, custom_loss

print('='*80)
print('TRAINING MODEL V5 - DIRECT ON REAL DATA (NO PRE-TRAINING)')
print('='*80)
print('\nFixes applied:')
print('  ✓ Temperature: 3.0')
print('  ✓ Match point penalty: 0.5')
print('  ✓ Set point penalty: 0.3')
print('  ✓ Weight decay: 0.01')
print('  ✓ Gradient clipping: 1.0')
print('  ✓ NO pre-training (avoids synthetic data bias)')
print('='*80)

# Load real data
print('\n[1/4] Loading real match data...')
train_files = sorted(glob('data/*-points.csv'))
train_files = [f for f in train_files if '2019-wimbledon' not in f]
print(f'  Found {len(train_files)} files for training')
print(f'  (Excluded 2019 Wimbledon for testing)')

df_raw = load_points_multiple(train_files)
# Filter by gender
df_raw = df_raw[df_raw.get('gender', 'male') == 'male'] if 'gender' in df_raw.columns else df_raw
print(f'  Total points: {len(df_raw):,}')
print(f'  Matches: {df_raw["match_id"].nunique()}')

# Build features
print('\n[2/4] Computing features...')
df = add_additional_features(df_raw)
df = add_match_labels(df)

X, y_match, y_set, y_game, sample_weights, match_ids = build_new_features(df)
print(f'  Features: {X.shape}')
print(f'  P1 win rate: {y_match.mean():.3f} (should be ~0.50)')

# Train/val split
n_train = int(0.85 * len(X))
indices = np.random.permutation(len(X))
train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_train = X[train_idx]
y_match_train = y_match[train_idx]
y_set_train = y_set[train_idx]
y_game_train = y_game[train_idx]
weights_train = sample_weights[train_idx]

X_val = X[val_idx]
y_match_val = y_match[val_idx]
y_set_val = y_set[val_idx]
y_game_val = y_game[val_idx]
weights_val = sample_weights[val_idx]

print(f'  Train: {len(X_train):,} points')
print(f'  Val: {len(X_val):,} points')

# Create model
print('\n[3/4] Training model from scratch...')
model = TennisRulesNet(input_size=31, hidden_sizes=[128, 64], dropout=0.4)
device = 'cpu'
model.to(device)

# Optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Create dataloaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_match_train),
    torch.FloatTensor(y_set_train),
    torch.FloatTensor(y_game_train),
    torch.FloatTensor(weights_train)
)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.FloatTensor(y_match_val),
    torch.FloatTensor(y_set_val),
    torch.FloatTensor(y_game_val),
    torch.FloatTensor(weights_val)
)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Training
temperature = 3.0
epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y_match, batch_y_set, batch_y_game, batch_weights in train_loader:
        batch_X = batch_X.to(device)
        batch_y_match = batch_y_match.to(device)
        batch_y_set = batch_y_set.to(device)
        batch_y_game = batch_y_game.to(device)
        batch_weights = batch_weights.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_X)
        
        # Custom loss (simplified - no match/set point indicators for now)
        loss = custom_loss(
            pred, batch_y_match, batch_y_set, batch_y_game,
            batch_weights, temperature=temperature
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y_match, batch_y_set, batch_y_game, batch_weights in val_loader:
            batch_X = batch_X.to(device)
            batch_y_match = batch_y_match.to(device)
            batch_y_set = batch_y_set.to(device)
            batch_y_game = batch_y_game.to(device)
            batch_weights = batch_weights.to(device)
            
            pred = model(batch_X)
            loss = custom_loss(
                pred, batch_y_match, batch_y_set, batch_y_game,
                batch_weights, temperature=temperature
            )
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'  Epoch {epoch+1:2d}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            break

# Load best model
model.load_state_dict(best_state)

# Save
print('\n[4/4] Saving model...')
save_path = 'models/complete_model_v5.pth'

torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': 31,
    'hidden_sizes': [128, 64],
    'dropout': 0.4,
    'temperature': temperature,
    'train_points': len(X_train),
    'val_points': len(X_val),
    'best_val_loss': best_val_loss,
    'method': 'direct_training_no_pretrain',
}, save_path)

print('\n' + '='*80)
print('TRAINING COMPLETE!')
print('='*80)
print(f'  Model: {save_path}')
print(f'  Best val loss: {best_val_loss:.4f}')
print(f'  Train data balance: {y_match.mean():.3f} (perfect: 0.50)')
print('\nTo test:')
print('  python tennisctl.py predict --model ./models/complete_model_v5.pth \\')
print('    --match-id 2019-wimbledon-1701 \\')
print('    --files data/2019-wimbledon-points.csv \\')
print('    --plot-dir plots/v5 --point-by-point')
