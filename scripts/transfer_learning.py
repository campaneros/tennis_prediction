"""
Complete Model Training with Transfer Learning

This module fine-tunes a pre-trained tennis rules model on real match data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
from pathlib import Path
from typing import List, Optional

from .data_loader import load_points_multiple, MATCH_COL
from .features import add_additional_features, add_match_labels
from .new_model_nn import (
    calculate_distance_features,
    build_new_features,
    custom_loss
)
from .pretrain_tennis_rules import TennisRulesNet


def load_pretrained_model(pretrained_path: str, device='cuda') -> TennisRulesNet:
    """Load pre-trained model from checkpoint."""
    print(f"Loading pre-trained model from {pretrained_path}...")
    
    # Force CPU if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("  Warning: CUDA not available, using CPU")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    model = TennisRulesNet(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['hidden_sizes'],
        dropout=checkpoint['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"  ✓ Loaded model pre-trained on {checkpoint['n_training_matches']} synthetic matches")
    print(f"  ✓ Architecture: {checkpoint['hidden_sizes']}")
    print(f"  ✓ Temperature: {checkpoint['temperature']}")
    
    return model, checkpoint


def fine_tune_on_real_data(
    files: List[str],
    pretrained_path: str,
    output_path: str,
    gender: str = 'male',
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 0.0001,  # Lower LR for fine-tuning
    temperature: float = 3.0,
    freeze_layers: bool = False,
    device: str = 'cuda'
):
    """
    Fine-tune pre-trained model on real tennis data.
    
    Args:
        files: List of CSV files with real match data
        pretrained_path: Path to pre-trained model checkpoint
        output_path: Where to save fine-tuned model
        gender: 'male' or 'female'
        epochs: Number of fine-tuning epochs
        batch_size: Batch size
        learning_rate: Learning rate (lower than pre-training)
        temperature: Temperature for calibration
        freeze_layers: If True, freeze first layer (only train heads)
        device: 'cuda' or 'cpu'
    """
    print("="*80)
    print("PHASE 2: FINE-TUNING ON REAL MATCH DATA")
    print("="*80)
    
    # Load pre-trained model
    print("\n[1/5] Loading pre-trained model...")
    model, pretrain_info = load_pretrained_model(pretrained_path, device)
    
    # Optionally freeze early layers
    if freeze_layers:
        print("  → Freezing first hidden layer (fc1, bn1)")
        model.fc1.requires_grad_(False)
        model.bn1.requires_grad_(False)
    
    # Load real data
    print("\n[2/5] Loading real match data...")
    df = load_points_multiple(files)
    
    # Gender filter (same logic as new_model_nn.py)
    if gender != "both":
        print(f"  Gender filter: {gender}")
        extracted = df['match_id'].str.extract(r'-(\d+)$')[0]
        valid_mask = extracted.notna()
        
        if valid_mask.any():
            df_temp = df[valid_mask].copy()
            df_temp['match_num'] = extracted[valid_mask].astype(int)
            
            if gender == "male":
                df_temp = df_temp[df_temp['match_num'] < 2000].copy()
            else:
                df_temp = df_temp[df_temp['match_num'] >= 2000].copy()
            
            df_temp = df_temp.drop(columns=['match_num'])
            df = df_temp
    
    # Exclude test match 1701
    df = df[~df['match_id'].astype(str).str.contains('1701', na=False)]
    
    # Add features
    df = add_additional_features(df)
    df = add_match_labels(df)
    
    print(f"  Total points loaded: {len(df)}")
    print(f"  Matches: {df[MATCH_COL].nunique()}")
    
    # Build features (using existing pipeline)
    print("\n[3/5] Computing features from real data...")
    # build_new_features will call calculate_distance_features internally and return match_ids
    X_train, y_train_match, y_train_set, y_train_game, sample_weights, match_ids = build_new_features(df)
    
    print(f"  Features shape: {X_train.shape}")
    print(f"  Matches: {len(np.unique(match_ids))}")
    print(f"  Match label distribution: {y_train_match.mean():.3f}")
    
    # Split by match (80/20 train/val)
    unique_matches = np.unique(match_ids)
    n_train_matches = int(0.8 * len(unique_matches))
    train_match_ids = unique_matches[:n_train_matches]
    val_match_ids = unique_matches[n_train_matches:]
    
    train_mask = np.isin(match_ids, train_match_ids)
    val_mask = np.isin(match_ids, val_match_ids)
    
    X_train_split = X_train[train_mask]
    y_match_train = y_train_match[train_mask]
    y_set_train = y_train_set[train_mask]
    y_game_train = y_train_game[train_mask]
    weights_train = sample_weights[train_mask]
    
    X_val = X_train[val_mask]
    y_match_val = y_train_match[val_mask]
    y_set_val = y_train_set[val_mask]
    y_game_val = y_train_game[val_mask]
    weights_val = sample_weights[val_mask]
    
    print(f"  Train: {len(X_train_split)} points from {len(train_match_ids)} matches")
    print(f"  Val: {len(X_val)} points from {len(val_match_ids)} matches")
    
    # Convert to tensors
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device_obj}")
    
    X_train_tensor = torch.FloatTensor(X_train_split).to(device_obj)
    y_match_tensor = torch.FloatTensor(y_match_train).to(device_obj)
    y_set_tensor = torch.FloatTensor(y_set_train).to(device_obj)
    y_game_tensor = torch.FloatTensor(y_game_train).to(device_obj)
    weights_tensor = torch.FloatTensor(weights_train).to(device_obj)
    
    X_val_tensor = torch.FloatTensor(X_val).to(device_obj)
    y_match_val_tensor = torch.FloatTensor(y_match_val).to(device_obj)
    y_set_val_tensor = torch.FloatTensor(y_set_val).to(device_obj)
    y_game_val_tensor = torch.FloatTensor(y_game_val).to(device_obj)
    weights_val_tensor = torch.FloatTensor(weights_val).to(device_obj)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_match_tensor, y_set_tensor, 
                                  y_game_tensor, weights_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_match_val_tensor, y_set_val_tensor,
                                y_game_val_tensor, weights_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer (lower learning rate for fine-tuning)
    print(f"\n[4/5] Fine-tuning for {epochs} epochs...")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, weight_decay=0.01)
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(1, epochs + 1):
        # Training
        train_loss = 0.0
        n_train_batches = 0
        
        for batch_X, batch_y_match, batch_y_set, batch_y_game, batch_weights in train_loader:
            # Extract critical point features
            batch_is_mp_p1 = batch_X[:, 26]
            batch_is_mp_p2 = batch_X[:, 27]
            batch_is_sp_p1 = batch_X[:, 24]
            batch_is_sp_p2 = batch_X[:, 25]
            batch_p1_sets = batch_X[:, 4]  # P1SetsWon feature
            batch_p2_sets = batch_X[:, 5]  # P2SetsWon feature
            
            # Forward pass
            pred = model(batch_X)
            
            # Use the same custom loss as training (returns tuple: loss, metrics_dict)
            loss_result = custom_loss(
                pred, batch_y_match, batch_y_set, batch_y_game,
                batch_weights, temperature,
                batch_is_mp_p1, batch_is_mp_p2,
                batch_is_sp_p1, batch_is_sp_p2,
                batch_p1_sets, batch_p2_sets
            )
            
            # Extract the loss tensor from the result
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = train_loss / n_train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y_match, batch_y_set, batch_y_game, batch_weights in val_loader:
                batch_is_mp_p1 = batch_X[:, 26]
                batch_is_mp_p2 = batch_X[:, 27]
                batch_is_sp_p1 = batch_X[:, 24]
                batch_is_sp_p2 = batch_X[:, 25]
                batch_p1_sets = batch_X[:, 4]
                batch_p2_sets = batch_X[:, 5]
                
                pred = model(batch_X)
                loss_result = custom_loss(
                    pred, batch_y_match, batch_y_set, batch_y_game,
                    batch_weights, temperature,
                    batch_is_mp_p1, batch_is_mp_p2,
                    batch_is_sp_p1, batch_is_sp_p2,
                    batch_p1_sets, batch_p2_sets
                )
                
                # Extract the loss tensor from the result
                loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
                
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss / n_val_batches
        model.train()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Save fine-tuned model
    print(f"\n[5/5] Saving fine-tuned model...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save in same format as original model
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': 31,
        'hidden_sizes': [128, 64],
        'dropout': 0.4,
        'temperature': temperature,
        'pretrained_from': pretrained_path,
        'pretrain_matches': pretrain_info['n_training_matches'],
        'finetune_points': len(X_train_split),
        'finetune_matches': len(train_match_ids),
        'best_val_loss': best_val_loss,
    }
    
    torch.save(model_info, output_path)
    
    print(f"\n{'='*80}")
    print("Fine-tuning summary:")
    print(f"  - Pre-trained on: {pretrain_info['n_training_matches']} synthetic matches")
    print(f"  - Fine-tuned on: {len(train_match_ids)} real matches ({len(X_train_split)} points)")
    print(f"  - Best validation loss: {best_val_loss:.4f}")
    print(f"  - Model saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return model, model_info


if __name__ == "__main__":
    # Example usage
    import glob
    
    # Fine-tune on real Wimbledon data
    files = glob.glob("data/*-wimbledon-points.csv")
    
    fine_tune_on_real_data(
        files=files,
        pretrained_path="models/tennis_rules_pretrained.pth",
        output_path="models/nn_model_transfer.pth",
        gender='male',
        epochs=30,
        batch_size=1024,
        learning_rate=0.0001,
        temperature=12.0,
        freeze_layers=False
    )
