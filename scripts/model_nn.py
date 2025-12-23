"""
Neural Network model for tennis match prediction.

This module implements a PyTorch-based neural network as an alternative to XGBoost.
The network can learn complex non-linear relationships between features like hold/break
context and match situations.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from .data_loader import load_points_multiple
from .features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_additional_features,
    add_leverage_and_momentum,
    build_dataset,
    MATCH_FEATURE_COLUMNS,
)
from .config import load_config


class TennisDataset(Dataset):
    """PyTorch Dataset for tennis match data."""
    
    def __init__(self, X, y, weights=None):
        # Store as numpy arrays to avoid large tensor creation
        self.X = X if isinstance(X, np.ndarray) else np.array(X)
        self.y = y if isinstance(y, np.ndarray) else np.array(y)
        self.weights = weights if weights is None or isinstance(weights, np.ndarray) else np.array(weights)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor([self.y[idx]])
        if self.weights is not None:
            w = torch.FloatTensor([self.weights[idx]])
            return x, y, w
        return x, y


class TennisNN(nn.Module):
    """
    Neural Network for tennis match prediction.
    
    Architecture:
    - Input: 31 features (same as XGBoost)
    - Hidden layers with BatchNorm and Dropout for regularization
    - Output: P(P1 wins match)
    
    The network learns to automatically weight features based on context,
    potentially handling hold/break situations better than tree-based models.
    """
    
    def __init__(self, input_dim=31, hidden_dims=[128, 64, 32], dropout=0.3):
        super(TennisNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_nn_model(file_paths, model_out, config_path=None, gender="male",
                   hidden_dims=[128, 64, 32], dropout=0.3, 
                   epochs=100, batch_size=512, learning_rate=0.001,
                   weight_exponent=0.5, use_weighted_features=True):
    """
    Train a neural network model for tennis match prediction.
    
    Args:
        file_paths: List of CSV files with point-by-point data
        model_out: Path to save trained model
        config_path: Path to config file
        gender: Filter by gender ("male", "female", "both")
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate for regularization
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        weight_exponent: Exponent for sample weighting (weights = importance^exponent)
        use_weighted_features: If True, use weighted serve/return features
    
    Returns:
        Trained model, test accuracy, test ROC AUC
    """
    print("[train_nn] Loading and processing data...")
    
    if not file_paths:
        raise ValueError("train_nn_model: no input files provided")
    
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))
    
    # Load data
    df = load_points_multiple(file_paths)
    
    # Gender filter
    if gender != "both":
        print(f"[train_nn] Gender filter: {gender}")
        extracted = df['match_id'].str.extract(r'-(\d+)$')[0]
        valid_mask = extracted.notna()
        
        if valid_mask.any():
            df_temp = df[valid_mask].copy()
            df_temp['match_num'] = extracted[valid_mask].astype(int)
            
            original_count = len(df)
            if gender == "male":
                df_temp = df_temp[df_temp['match_num'] < 2000].copy()
            else:
                df_temp = df_temp[df_temp['match_num'] >= 2000].copy()
            
            df_temp = df_temp.drop(columns=['match_num'])
            df = df_temp
    
    # Exclude test match (1701)
    test_match_mask = df['match_id'].astype(str).str.endswith('-1701')
    test_points = df[test_match_mask]
    df = df[~test_match_mask]
    
    print(f"[train_nn] Filtered {len(test_points)} points:")
    print(f"[train_nn]   - Excluded women's matches (>=2000): {original_count - len(df) - len(test_points)}")
    print(f"[train_nn]   - Match 1701 (test set): {len(test_points)}")
    print(f"[train_nn] Training on {len(df)} points from men's matches (best-of-5)")
    
    # Add labels
    df = add_match_labels(df)
    
    # Build features with optional weighting
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window,
                                          weight_serve_return=use_weighted_features)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)
    
    X, y, mask, sample_weights, _ = build_dataset(df)
    print(f"[train_nn] dataset shape: {X.shape} positives (P1 wins): {int(np.sum(y))}")
    print(f"[train_nn] long_window={long_window}, short_window={short_window}, alpha={alpha:.2f}")
    
    # Apply weight exponent
    if weight_exponent != 1.0:
        sample_weights = np.power(sample_weights, weight_exponent)
        print(f"[train_nn] weight exponent: {weight_exponent}")
    
    print(f"[train_nn] sample weights - mean: {np.mean(sample_weights):.2f}, max: {np.max(sample_weights):.2f}")
    
    # Train/test split (no stratify - we're working with point-level data, not match-level)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Create datasets
    train_dataset = TennisDataset(X_train, y_train, w_train)
    test_dataset = TennisDataset(X_test, y_test)
    
    # Use num_workers=0 to avoid multiprocessing issues on macOS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    device = torch.device('cpu')  # Force CPU on macOS to avoid MPS issues
    print(f"[train_nn] Using device: {device}")
    
    model = TennisNN(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    print(f"[train_nn] Model parameters: {sum(p.numel() for p in model.parameters())}")
    criterion = nn.BCELoss(reduction='none')  # We'll apply weights manually
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"[train_nn] Training for {epochs} epochs...")
    print(f"[train_nn] Architecture: {X.shape[1]} -> {' -> '.join(map(str, hidden_dims))} -> 1")
    
    best_auc = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y, batch_w in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            batch_w = batch_w.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            weighted_loss = (loss * batch_w).mean()
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += weighted_loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(batch_y.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))
        
        scheduler.step(train_loss / len(train_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"[train_nn] Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_acc:.3f}, Val AUC={val_auc:.3f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_labels.extend(batch_y.numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    test_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
    test_auc = roc_auc_score(test_labels, test_preds)
    
    print(f"[train_nn] Test accuracy: {test_acc:.3f}")
    print(f"[train_nn] Test ROC AUC:  {test_auc:.3f}")
    
    # Save model
    model_data = {
        'state_dict': {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
        'input_dim': X.shape[1],
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'feature_names': MATCH_FEATURE_COLUMNS,
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'use_weighted_features': use_weighted_features,
    }
    
    with open(model_out, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"[train_nn] Model saved to: {model_out}")
    
    return model, test_acc, test_auc


def load_nn_model(model_path):
    """
    Load a trained neural network model from JSON.
    
    Args:
        model_path: Path to saved model JSON
    
    Returns:
        Loaded PyTorch model in eval mode
    """
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Reconstruct model
    model = TennisNN(
        input_dim=model_data['input_dim'],
        hidden_dims=model_data['hidden_dims'],
        dropout=model_data['dropout']
    )
    
    # Load weights
    state_dict = {k: torch.FloatTensor(v) for k, v in model_data['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def predict_nn(model, X):
    """
    Make predictions with neural network model.
    
    Args:
        model: Trained TennisNN model
        X: Feature matrix (numpy array or single row)
    
    Returns:
        Predictions as numpy array
    """
    model.eval()
    
    # Handle single prediction
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        preds = model(X_tensor).numpy().flatten()
    
    return preds
