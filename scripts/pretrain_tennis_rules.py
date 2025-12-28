"""
Pre-training Neural Network on Tennis Rules

This module trains a neural network on synthetic tennis matches to learn
the fundamental scoring rules before fine-tuning on real data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
from pathlib import Path
from typing import Tuple, Dict

from .tennis_simulator import generate_training_dataset


def compute_tennis_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute features for synthetic tennis data.
    
    Features (31 total):
    - 6 core: p1_points, p2_points, p1_games, p2_games, p1_sets, p2_sets (normalized)
    - 4 context: set_number (normalized), server_p1 (one-hot), server_p2 (one-hot), is_best_of_5
    - 4 tiebreak: is_tiebreak, is_decisive_tiebreak, tb_p1_points, tb_p2_points
    - 8 distance: points_to_game_p1/p2, games_to_set_p1/p2, sets_to_match_p1/p2, total_dist_p1/p2
    - 8 critical: is_game_point_p1/p2, is_set_point_p1/p2, is_match_point_p1/p2, is_break_point_p1/p2
    - 1 performance: p1_performance (not used in synthetic, set to 0)
    
    Total: 6+4+4+8+8+1 = 31 features
    
    IMPORTANT: Features are SYMMETRIC - for every P1 feature there's a P2 feature.
    The model learns that P(P2 wins) = 1 - P(P1 wins), no separate P2 probability feature needed.
    
    Labels (computed separately):
    - y_match: 1.0 if P1 wins match, 0.0 if P2 wins
    - y_set: 1.0 if P1 wins set, 0.0 if P2 wins
    - y_game: 1.0 if P1 wins game, 0.0 if P2 wins
    """
    n_points = len(df)
    features = np.zeros((n_points, 31), dtype=np.float32)
    
    # Determine match format from data
    if 'is_best_of_5' not in df.columns:
        # Infer from max set number
        max_sets = df.groupby('match_id')['set_number'].transform('max')
        df['is_best_of_5'] = (max_sets >= 4).astype(float)
    
    is_best_of_5 = df['is_best_of_5'].values
    sets_to_win = np.where(is_best_of_5 == 1, 3, 2)
    
    # Extract raw values
    p1_points = df['p1_points'].values
    p2_points = df['p2_points'].values
    p1_games = df['p1_games'].values
    p2_games = df['p2_games'].values
    p1_sets = df['p1_sets'].values
    p2_sets = df['p2_sets'].values
    set_number = df['set_number'].values
    server = df['server'].values
    
    # Detect tiebreaks
    is_tiebreak = ((p1_points > 3) | (p2_points > 3)).astype(float)
    
    # Decisive tiebreak (12-12 in final set)
    is_final_set_bo5 = ((is_best_of_5 == 1) & (set_number == 5)).astype(float)
    is_final_set_bo3 = ((is_best_of_5 == 0) & (set_number == 3)).astype(float)
    is_final_set = (is_final_set_bo5 + is_final_set_bo3 > 0).astype(float)
    tb_threshold = np.where(is_final_set > 0, 12, 6)
    is_decisive_tiebreak = (is_tiebreak * is_final_set * (p1_games == tb_threshold).astype(float) * (p2_games == tb_threshold).astype(float))
    
    # === 6 CORE FEATURES (normalized) ===
    features[:, 0] = p1_points / 5.0  # Max ~10 in long tiebreaks
    features[:, 1] = p2_points / 5.0
    features[:, 2] = p1_games / 13.0  # Max 13 in decisive tiebreak
    features[:, 3] = p2_games / 13.0
    features[:, 4] = p1_sets / 3.0    # Max 3 in bo5
    features[:, 5] = p2_sets / 3.0
    
    # === 3 CONTEXT FEATURES ===
    features[:, 6] = set_number / 5.0  # Max 5
    features[:, 7] = (server == 1).astype(float)  # Server one-hot
    features[:, 8] = (server == 2).astype(float)
    features[:, 9] = is_best_of_5
    
    # === 4 TIEBREAK FEATURES ===
    features[:, 10] = is_tiebreak
    features[:, 11] = is_decisive_tiebreak.astype(float)
    features[:, 12] = np.where(is_tiebreak, p1_points / 15.0, 0)  # TB points normalized
    features[:, 13] = np.where(is_tiebreak, p2_points / 15.0, 0)
    
    # === 8 DISTANCE FEATURES ===
    # Points to win game
    p1_needs_points = np.where(is_tiebreak, 
                                np.maximum(7 - p1_points, np.maximum(1, p2_points + 2 - p1_points)),
                                np.maximum(4 - p1_points, np.maximum(1, p2_points + 2 - p1_points)))
    p2_needs_points = np.where(is_tiebreak,
                                np.maximum(7 - p2_points, np.maximum(1, p1_points + 2 - p2_points)),
                                np.maximum(4 - p2_points, np.maximum(1, p1_points + 2 - p2_points)))
    features[:, 14] = p1_needs_points / 5.0
    features[:, 15] = p2_needs_points / 5.0
    
    # Games to win set
    p1_needs_games = np.maximum(6 - p1_games, np.maximum(1, p2_games + 2 - p1_games))
    p2_needs_games = np.maximum(6 - p2_games, np.maximum(1, p1_games + 2 - p2_games))
    features[:, 16] = p1_needs_games / 13.0
    features[:, 17] = p2_needs_games / 13.0
    
    # Sets to win match
    p1_needs_sets = sets_to_win - p1_sets
    p2_needs_sets = sets_to_win - p2_sets
    features[:, 18] = p1_needs_sets / 3.0
    features[:, 19] = p2_needs_sets / 3.0
    
    # Total distance (sum)
    features[:, 20] = (p1_needs_points + p1_needs_games + p1_needs_sets) / 21.0
    features[:, 21] = (p2_needs_points + p2_needs_games + p2_needs_sets) / 21.0
    
    # === 8 CRITICAL POINT FEATURES (TRUE/FALSE logic) ===
    # Game points
    p1_can_win_game = (p1_needs_points == 1).astype(float)
    p2_can_win_game = (p2_needs_points == 1).astype(float)
    features[:, 22] = p1_can_win_game
    features[:, 23] = p2_can_win_game
    
    # Set points (winning this point wins the set)
    p1_can_win_set_this_game = (
        ((p1_games == 5) & (p2_games <= 4)) |
        ((p1_games >= 6) & (p1_games == p2_games + 1)) |
        (is_tiebreak > 0)
    ).astype(float)
    p2_can_win_set_this_game = (
        ((p2_games == 5) & (p1_games <= 4)) |
        ((p2_games >= 6) & (p2_games == p1_games + 1)) |
        (is_tiebreak > 0)
    ).astype(float)
    features[:, 24] = p1_can_win_set_this_game * p1_can_win_game
    features[:, 25] = p2_can_win_set_this_game * p2_can_win_game
    
    # Match points (winning this point wins the match)
    p1_can_win_match_this_set = (p1_needs_sets == 1).astype(float)
    p2_can_win_match_this_set = (p2_needs_sets == 1).astype(float)
    features[:, 26] = p1_can_win_match_this_set * p1_can_win_set_this_game * p1_can_win_game
    features[:, 27] = p2_can_win_match_this_set * p2_can_win_set_this_game * p2_can_win_game
    
    # Break points (receiving player can break serve)
    is_p1_serving = (server == 1).astype(float)
    is_p2_serving = (server == 2).astype(float)
    features[:, 28] = is_p2_serving * p1_can_win_game  # P1 breaks P2's serve
    features[:, 29] = is_p1_serving * p2_can_win_game  # P2 breaks P1's serve
    
    # === 2 PERFORMANCE FEATURES (not applicable for synthetic) ===
    # In reality, we only use 1 performance feature (p1_performance) at index 30
    # The original model has p1 and p2 performance, but for synthetic data both are 0
    # Total: 6+3+1+4+8+8+1 = 31 features (indices 0-30)
    features[:, 30] = 0  # p1_performance (always 0 in synthetic data)
    
    return features  # Shape: (n_points, 31)


def compute_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deterministic labels for synthetic data (VECTORIZED for speed).
    
    Returns:
        match_labels, set_labels, game_labels (all perfect ground truth)
    """
    print("  Computing labels (vectorized)...")
    
    # Match outcome is known (already in df)
    match_labels = df['p1_wins_match'].values.astype(np.float32)
    
    # Set outcome: who wins this set
    # Use groupby with transform to get final games for each set
    df_temp = df.copy()
    df_temp['final_p1_games'] = df_temp.groupby(['match_id', 'set_number'])['p1_games'].transform('last')
    df_temp['final_p2_games'] = df_temp.groupby(['match_id', 'set_number'])['p2_games'].transform('last')
    set_winner = (df_temp['final_p1_games'] > df_temp['final_p2_games']).astype(np.float32).values
    
    # Game outcome: who wins current game
    # Create game identifier
    df_temp['game_id'] = df_temp['p1_games'] + df_temp['p2_games']
    df_temp['final_p1_points'] = df_temp.groupby(['match_id', 'set_number', 'game_id'])['p1_points'].transform('last')
    df_temp['final_p2_points'] = df_temp.groupby(['match_id', 'set_number', 'game_id'])['p2_points'].transform('last')
    game_winner = (df_temp['final_p1_points'] > df_temp['final_p2_points']).astype(np.float32).values
    
    return match_labels, set_winner, game_winner


class TennisRulesNet(nn.Module):
    """Neural network for learning tennis rules."""
    
    def __init__(self, input_size=31, hidden_sizes=[128, 64], dropout=0.4):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout)
        
        # Task-specific heads
        self.match_head = nn.Linear(hidden_sizes[1], 1)
        self.set_head = nn.Linear(hidden_sizes[1], 1)
        self.game_head = nn.Linear(hidden_sizes[1], 1)
    
    def forward(self, x):
        # Shared feature extraction
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Multi-task outputs
        match_logits = self.match_head(x)
        set_logits = self.set_head(x)
        game_logits = self.game_head(x)
        
        return match_logits, set_logits, game_logits


def custom_loss_pretrain(pred, y_match, y_set, y_game, weights, temperature=10.0,
                        is_mp_p1=None, is_mp_p2=None, is_sp_p1=None, is_sp_p2=None):
    """
    Custom loss for pre-training on synthetic data.
    
    Components:
    1. Multi-task BCE (match, set, game)
    2. Consistency penalty (set→match hierarchy)
    3. Temperature scaling for calibration
    4. Match point penalty (force high confidence at match points)
    5. Set point penalty (force moderate confidence at set points)
    """
    device = pred[0].device
    
    # 1. Multi-task BCE
    match_prob = torch.sigmoid(pred[0] / temperature)
    set_prob = torch.sigmoid(pred[1] / temperature)
    game_prob = torch.sigmoid(pred[2] / temperature)
    
    loss_match = nn.BCELoss(weight=weights)(match_prob.squeeze(), y_match)
    loss_set = nn.BCELoss(weight=weights)(set_prob.squeeze(), y_set)
    loss_game = nn.BCELoss(weight=weights)(game_prob.squeeze(), y_game)
    
    total_loss = loss_match + 0.5 * loss_set + 0.3 * loss_game
    
    # 2. Consistency penalty (set must agree with match in direction)
    consistency_penalty = torch.mean(weights * torch.abs(match_prob.squeeze() - set_prob.squeeze()))
    total_loss += 0.1 * consistency_penalty
    
    # 3. Match point penalty (enforce >0.85 when player has match point)
    if is_mp_p1 is not None and is_mp_p2 is not None:
        mp_p1_mask = (is_mp_p1 > 0.5).float()
        mp_p2_mask = (is_mp_p2 > 0.5).float()
        
        # P1 match points should have prob >0.85
        mp_p1_penalty = mp_p1_mask * torch.relu(0.85 - match_prob.squeeze())
        # P2 match points should have prob <0.15
        mp_p2_penalty = mp_p2_mask * torch.relu(match_prob.squeeze() - 0.15)
        
        match_point_loss = torch.mean(mp_p1_penalty + mp_p2_penalty)
        total_loss += 5.0 * match_point_loss
    
    # 4. Set point penalty (enforce ~0.70 for P1, ~0.30 for P2)
    if is_sp_p1 is not None and is_sp_p2 is not None:
        sp_p1_mask = (is_sp_p1 > 0.5).float()
        sp_p2_mask = (is_sp_p2 > 0.5).float()
        
        # P1 set points: push toward 0.70 (not as high as match points)
        target_p1 = 0.70
        sp_p1_penalty = sp_p1_mask * torch.abs(match_prob.squeeze() - target_p1)
        
        # P2 set points: push toward 0.30
        target_p2 = 0.30
        sp_p2_penalty = sp_p2_mask * torch.abs(match_prob.squeeze() - target_p2)
        
        set_point_loss = torch.mean(sp_p1_penalty + sp_p2_penalty)
        total_loss += 2.0 * set_point_loss
    
    return total_loss


def pretrain_tennis_rules(n_matches=50000, epochs=50, batch_size=2048, 
                         temperature=3.0, output_path="models/tennis_rules_pretrained.pth",
                         device='cuda'):
    """
    Pre-train neural network on synthetic tennis matches.
    
    Args:
        n_matches: Number of synthetic matches to generate
        epochs: Training epochs
        batch_size: Batch size
        temperature: Temperature for calibration
        output_path: Where to save pre-trained weights
        device: 'cuda' or 'cpu'
    """
    print("="*80)
    print("PHASE 1: PRE-TRAINING ON SYNTHETIC TENNIS MATCHES")
    print("="*80)
    
    # Generate synthetic data
    print(f"\n[1/4] Generating {n_matches} synthetic matches...")
    df = generate_training_dataset(n_matches=n_matches, best_of_5=True, seed=42)
    
    # Compute features
    print(f"[2/4] Computing features...")
    print(f"  Processing {len(df)} points from {n_matches} matches...")
    X = compute_tennis_features(df)
    print(f"  ✓ Features computed: {X.shape}")
    
    y_match, y_set, y_game = compute_labels(df)
    print(f"  ✓ Labels computed")
    
    print(f"  Total points: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Match label distribution: {y_match.mean():.3f}")
    
    # Sample weights (emphasize critical points)
    print(f"[3/4] Computing sample weights...")
    weights = np.ones(len(X), dtype=np.float32)
    
    # Extract critical features
    is_mp_p1 = X[:, 26]  # Match point P1
    is_mp_p2 = X[:, 27]  # Match point P2
    is_sp_p1 = X[:, 24]  # Set point P1
    is_sp_p2 = X[:, 25]  # Set point P2
    is_bp_p1 = X[:, 28]  # Break point P1
    is_bp_p2 = X[:, 29]  # Break point P2
    is_decisive_tb = X[:, 11]  # Decisive tiebreak
    
    is_match_point = (is_mp_p1 > 0.5) | (is_mp_p2 > 0.5)
    is_set_point = (is_sp_p1 > 0.5) | (is_sp_p2 > 0.5)
    is_break_point = (is_bp_p1 > 0.5) | (is_bp_p2 > 0.5)
    
    weights = np.where(is_decisive_tb > 0.5, 15.0, weights)
    weights = np.where(is_match_point, 25.0, weights)
    weights = np.where(is_set_point & (weights <= 1.0), 2.0, weights)
    weights = np.where(is_break_point & (weights <= 1.0), 1.5, weights)
    
    print(f"  Match points: {is_match_point.sum()}")
    print(f"  Set points: {is_set_point.sum()}")
    print(f"  Break points: {is_break_point.sum()}")
    print(f"  Decisive tiebreaks: {(is_decisive_tb > 0.5).sum()}")
    
    # Convert to tensors
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"  WARNING: CUDA requested but not available, using CPU instead")
    print(f"  Using device: {device_obj}")
    
    X_tensor = torch.FloatTensor(X).to(device_obj)
    y_match_tensor = torch.FloatTensor(y_match).to(device_obj)
    y_set_tensor = torch.FloatTensor(y_set).to(device_obj)
    y_game_tensor = torch.FloatTensor(y_game).to(device_obj)
    weights_tensor = torch.FloatTensor(weights).to(device_obj)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_match_tensor, y_set_tensor, y_game_tensor, weights_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TennisRulesNet(input_size=31, hidden_sizes=[128, 64], dropout=0.4).to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n[4/4] Training for {epochs} epochs...")
    print(f"  Total batches per epoch: {len(dataloader)}")
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_y_match, batch_y_set, batch_y_game, batch_weights in dataloader:
            # Extract critical point features
            batch_is_mp_p1 = batch_X[:, 26]
            batch_is_mp_p2 = batch_X[:, 27]
            batch_is_sp_p1 = batch_X[:, 24]
            batch_is_sp_p2 = batch_X[:, 25]
            
            # Forward pass
            pred = model(batch_X)
            
            # Compute loss
            loss = custom_loss_pretrain(
                pred, batch_y_match, batch_y_set, batch_y_game,
                batch_weights, temperature,
                batch_is_mp_p1, batch_is_mp_p2,
                batch_is_sp_p1, batch_is_sp_p2
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Print progress every epoch (not just every 5)
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Save model
    print(f"\n[✓] Pre-training complete!")
    print(f"    Saving pre-trained weights to {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save both state dict and architecture info
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': 31,
        'hidden_sizes': [128, 64],
        'dropout': 0.4,
        'temperature': temperature,
        'n_training_points': len(X),
        'n_training_matches': n_matches,
    }
    
    torch.save(checkpoint, output_path)
    
    print(f"\n{'='*80}")
    print("Pre-training summary:")
    print(f"  - Trained on {n_matches} synthetic matches ({len(X)} points)")
    print(f"  - Final loss: {avg_loss:.4f}")
    print(f"  - Model saved to: {output_path}")
    print(f"  - Ready for Phase 2: Fine-tuning on real data")
    print(f"{'='*80}\n")
    
    return model, checkpoint


if __name__ == "__main__":
    # Run pre-training
    pretrain_tennis_rules(
        n_matches=50000,
        epochs=50,
        batch_size=2048,
        temperature=10.0,
        output_path="models/tennis_rules_pretrained.pth"
    )
