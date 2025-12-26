"""
Neural Network model for tennis match prediction.

This module implements a PyTorch-based neural network as an alternative to XGBoost.
The network can learn complex non-linear relationships between features like hold/break
context and match situations.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler

from .data_loader import load_points_multiple
from .features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_additional_features,
    add_break_hold_features,
    add_leverage_and_momentum,
    build_dataset,
    build_clean_features_nn,  # Nuova funzione per features pulite
    MATCH_FEATURE_COLUMNS,
)
from .config import load_config


def add_break_features(df):
    """
    Aggiunge features che distinguono tra HOLD (servizio tenuto) e BREAK (servizio perso).
    Solo i break dovrebbero pesare molto sulla probabilità di vittoria.
    NON usare per BDT - solo per NN!
    
    Args:
        df: DataFrame con colonne 'PointServer', 'P1GamesWon', 'P2GamesWon'
    
    Returns:
        df con nuove colonne per break tracking
    """
    df = df.copy()
    
    # Identifica quando inizia un nuovo game
    df['game_id'] = (df['P1GamesWon'] + df['P2GamesWon']).astype(str) + '_' + df['match_id'].astype(str)
    df['new_game'] = (df['game_id'] != df.groupby('match_id')['game_id'].shift(1))
    
    # Chi serviva nel game precedente
    df['prev_server'] = df.groupby('match_id')['PointServer'].shift(1)
    
    # Identifica i break: nuovo game E il server è cambiato in modo "sbagliato"
    # Se P1 serviva e ora P2 ha vinto il game (P2GamesWon è aumentato) → break per P2
    # Se P2 serviva e ora P1 ha vinto il game (P1GamesWon è aumentato) → break per P1
    
    df['P1_recent_breaks'] = 0.0
    df['P2_recent_breaks'] = 0.0
    
    for match_id in df['match_id'].unique():
        mask = df['match_id'] == match_id
        match_df = df[mask].copy()
        
        p1_breaks = []
        p2_breaks = []
        
        prev_p1_games = 0
        prev_p2_games = 0
        prev_server = None
        
        for idx, row in match_df.iterrows():
            p1_games = row['P1GamesWon']
            p2_games = row['P2GamesWon']
            
            # Check se c'è stato un break
            if prev_server is not None:
                if p1_games > prev_p1_games and prev_server == 2:
                    # P1 ha vinto un game mentre P2 serviva → BREAK per P1
                    p1_breaks.append(idx)
                elif p2_games > prev_p2_games and prev_server == 1:
                    # P2 ha vinto un game mentre P1 serviva → BREAK per P2
                    p2_breaks.append(idx)
            
            prev_p1_games = p1_games
            prev_p2_games = p2_games
            prev_server = row['PointServer']
        
        # Conta break recenti (ultimi 3 game)
        for i, (idx, row) in enumerate(match_df.iterrows()):
            # Conta break di P1 negli ultimi 3 game
            recent_p1_breaks = sum(1 for b_idx in p1_breaks if b_idx <= idx and i - match_df.index.get_loc(b_idx) <= 15)
            recent_p2_breaks = sum(1 for b_idx in p2_breaks if b_idx <= idx and i - match_df.index.get_loc(b_idx) <= 15)
            
            df.loc[idx, 'P1_recent_breaks'] = min(recent_p1_breaks, 3)  # Cap a 3
            df.loc[idx, 'P2_recent_breaks'] = min(recent_p2_breaks, 3)
    
    df['break_advantage'] = df['P1_recent_breaks'] - df['P2_recent_breaks']
    
    # Cleanup
    df = df.drop(columns=['game_id', 'new_game', 'prev_server'])
    
    return df


def prepare_nn_data(X, y, sample_weights, df=None, remove_duplicates=True, normalize=True, weight_cap=6.0):
    """
    Prepara i dati specificamente per la rete neurale.
    NON usare questa funzione per il BDT - solo per NN!
    
    Args:
        X: Feature matrix (può già includere break features se df aveva le colonne)
        y: Target labels
        sample_weights: Sample importance weights
        df: DataFrame originale (non più usato - le break features sono già in X)
        remove_duplicates: Se True, rimuove punti duplicati
        normalize: Se True, normalizza le features con StandardScaler
        weight_cap: Cap massimo per i sample weights (riduce influenza outliers)
    
    Returns:
        X_processed, y_processed, weights_processed, scaler
    """
    X_proc = X.copy()
    y_proc = y.copy()
    w_proc = sample_weights.copy()
    
    # Le break features sono ora già incluse in X se df le aveva
    # Non serve più calcolarle qui
    
    # 1. PULIZIA FEATURES: riduzione rimossa - era troppo aggressiva
    # Manteniamo Game_Diff e CurrentSetGamesDiff al valore originale
    
    # 2. RIDUZIONE PESO: Cap sui sample weights per ridurre influenza di punti estremi
    if weight_cap is not None:
        original_max = np.max(w_proc)
        w_proc = np.minimum(w_proc, weight_cap)
        if original_max > weight_cap:
            print(f"[prepare_nn_data] Capped weights: {original_max:.2f} -> {weight_cap:.2f}")
    
    # 2. RIMOZIONE DUPLICATI: usa 3 decimali per non essere troppo aggressiva
    if remove_duplicates:
        # Hash delle features per trovare duplicati
        # Arrotondiamo a 3 decimali per trovare punti "abbastanza simili"
        X_rounded = np.round(X_proc, decimals=3)
        unique_indices = []
        seen_hashes = set()
        
        for i in range(len(X_rounded)):
            row_hash = hash(X_rounded[i].tobytes())
            if row_hash not in seen_hashes:
                seen_hashes.add(row_hash)
                unique_indices.append(i)
        
        n_removed = len(X_proc) - len(unique_indices)
        if n_removed > 0:
            X_proc = X_proc[unique_indices]
            y_proc = y_proc[unique_indices]
            w_proc = w_proc[unique_indices]
            print(f"[prepare_nn_data] Rimossi {n_removed} duplicati ({100*n_removed/len(X):.1f}%)")
    
    # 4. NORMALIZZAZIONE: RobustScaler invece di StandardScaler
    # RobustScaler usa mediana e IQR invece di mean/std
    # Più robusto agli outlier e preserva meglio le relazioni tra features
    scaler = None
    if normalize:
        scaler = RobustScaler()
        X_proc = scaler.fit_transform(X_proc)
        print(f"[prepare_nn_data] Features normalizzate con RobustScaler (mediana/IQR)")
    
    return X_proc, y_proc, w_proc, scaler


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
    
    def __init__(self, input_dim=31, hidden_dims=[512, 256, 128, 64], dropout=0.3, temperature=1.0):
        super(TennisNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.temperature = temperature  # Per calibrazione probabilità
        
        # NO BatchNorm - features raw hanno già scale corrette
        # Usiamo solo inizializzazione Kaiming per ReLU
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            # Kaiming initialization per ReLU
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer con inizializzazione Xavier (per sigmoid)
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        # NO normalization - raw features mantengono semantica
        features = self.network(x)
        logits = self.output_layer(features)
        # Temperature scaling per calibrare le probabilità
        return torch.sigmoid(logits / self.temperature)


def train_nn_model(file_paths, model_out, config_path=None, gender="male",
                   hidden_dims=[128, 64], dropout=0.4, 
                   epochs=200, batch_size=1024, learning_rate=0.001,
                   weight_exponent=0.5, use_weighted_features=True,
                   temperature=4.0, early_stopping_patience=40,
                   use_clean_features=True):  # NUOVO: usa features pulite per default
    """
    Train a neural network model for tennis match prediction.
    
    Args:
        file_paths: List of CSV files with point-by-point data
        model_out: Path to save trained model
        config_path: Path to config file
        gender: Filter by gender ("male", "female", "both")
        hidden_dims: List of hidden layer dimensions (default [256, 128, 64])
        dropout: Dropout rate for regularization (default 0.3)
        epochs: Maximum number of training epochs (default 200)
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        weight_exponent: Exponent for sample weighting (weights = importance^exponent)
        use_weighted_features: If True, use weighted serve/return features
        temperature: Temperature scaling per calibrazione probabilità (default 1.0)
        early_stopping_patience: Stop se validation loss non migliora per N epochs (default 20)
        use_clean_features: Se True, usa features pulite (25) invece di complete (45)
    
    Returns:
        Trained model, test accuracy, test ROC AUC
        
    Miglioramenti specifici per NN (NON influenzano BDT):
        - Normalizzazione features con StandardScaler
        - Rimozione duplicati moderata (3 decimali)
        - Cap su sample weights (max 6.0)
        - Temperature scaling per calibrazione
        - Early stopping basato su validation loss
        - Rete più profonda [256, 128, 64]
        - L2 regularization (weight_decay)
        - Break features opzionali
        - Features pulite (NEW): set minimo di features essenziali per imparare tennis
    """
    print("[train_nn] Loading and processing data...")
    print(f"[train_nn] Using {'CLEAN' if use_clean_features else 'FULL'} feature set")
    
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
    
    # Build features
    if use_clean_features:
        # USA FEATURES PULITE: solo le essenziali per imparare tennis
        print("[train_nn] Building CLEAN feature set (25 features)")
        df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window,
                                              weight_serve_return=use_weighted_features)
        df = add_additional_features(df)  # Crea P1SetsWon, P2SetsWon, etc.
        df = add_leverage_and_momentum(df, alpha=alpha)
        X, y, mask, sample_weights, feature_names = build_clean_features_nn(df)
    else:
        # USA FEATURES COMPLETE: tutte le features originali
        print("[train_nn] Building FULL feature set (45 features)")
        df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window,
                                              weight_serve_return=use_weighted_features)
        df = add_additional_features(df)
        df = add_break_hold_features(df)  # CRITICAL: distingue break vs hold
        df = add_leverage_and_momentum(df, alpha=alpha)
        X, y, mask, sample_weights, _ = build_dataset(df)
        feature_names = MATCH_FEATURE_COLUMNS
    
    print(f"[train_nn] dataset shape: {X.shape} positives (P1 wins): {int(np.sum(y))}")
    print(f"[train_nn] long_window={long_window}, short_window={short_window}, alpha={alpha:.2f}")
    
    # Apply weight exponent
    if weight_exponent != 1.0:
        sample_weights = np.power(sample_weights, weight_exponent)
        print(f"[train_nn] weight exponent: {weight_exponent}")
    
    print(f"[train_nn] PRIMA pre-processing NN: weights mean={np.mean(sample_weights):.2f}, max={np.max(sample_weights):.2f}")
    
    # ===== PREPARAZIONE SPECIFICA PER RETE NEURALE =====
    # Normalizzazione con StandardScaler per stabilizzare training
    # IMPORTANTE: preserve feature semantics usando RobustScaler invece di StandardScaler
    # RobustScaler usa mediana e IQR invece di mean/std, più robusto agli outlier
    X, y, sample_weights, scaler = prepare_nn_data(
        X, y, sample_weights,
        remove_duplicates=True,    # Rimuove punti duplicati (3 decimali)
        normalize=True,            # NORMALIZZAZIONE ABILITATA - con RobustScaler
        weight_cap=6.0             # Cap sui weights
    )
    print(f"[train_nn] DOPO pre-processing NN: weights mean={np.mean(sample_weights):.2f}, max={np.max(sample_weights):.2f}")
    print(f"[train_nn] Dataset finale: {X.shape[0]} punti, {X.shape[1]} features")
    print(f"[train_nn] Features normalizzate con RobustScaler (mediana/IQR)")
    
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
    
    model = TennisNN(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout=dropout, temperature=temperature).to(device)
    print(f"[train_nn] Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"[train_nn] Temperature scaling: {temperature}")
    
    # Calcola pesi delle classi per bilanciare
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"[train_nn] Class balance: {n_pos:.0f} positives, {n_neg:.0f} negatives, pos_weight={pos_weight:.2f}")
    
    # Use BCEWithLogitsLoss con pos_weight per bilanciare le classi
    criterion = nn.BCELoss(reduction='none')  # We'll apply weights manually
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # FORTE L2 reg
    
    # Learning rate scheduler più aggressivo
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=8)
    
    print(f"[train_nn] Training for max {epochs} epochs (early stopping patience={early_stopping_patience})...")
    print(f"[train_nn] Architecture: {X.shape[1]} -> {' -> '.join(map(str, hidden_dims))} -> 1")
    print(f"[train_nn] Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    best_val_loss = float('inf')
    best_auc = 0
    best_model_state = None
    patience_counter = 0
    
    # Training loop con early stopping
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y, batch_w in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Label smoothing MODERATO per bilanciare: 
            # 0 → 0.10, 1 → 0.90
            smoothed_y = batch_y * 0.80 + 0.10
            
            loss = criterion(outputs, smoothed_y)
            weighted_loss = (loss * batch_w).mean()
            weighted_loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += weighted_loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                # Calcola validation loss
                loss = criterion(outputs, batch_y).mean()
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(batch_y.cpu().numpy().flatten())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels).astype(int)
        val_loss = val_loss / len(test_loader)
        
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"[train_nn] Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, Val AUC={val_auc:.3f}")
        
        # Early stopping basato su validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"[train_nn] Early stopping at epoch {epoch+1} (patience={early_stopping_patience})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[train_nn] Loaded best model with Val AUC={best_auc:.3f}")
    
    # Final evaluation
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_labels.extend(batch_y.numpy().flatten())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels).astype(int)
    
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
        'temperature': temperature,
        'feature_names': feature_names,  # usa i nomi corretti
        'use_clean_features': use_clean_features,  # salva quale tipo di features
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'use_weighted_features': use_weighted_features,
        # Salva parametri dello scaler per normalizzazione in predizione
        # RobustScaler usa center_ e scale_ invece di mean_ e scale_
        'scaler_center': scaler.center_.tolist() if scaler is not None else None,
        'scaler_scale': scaler.scale_.tolist() if scaler is not None else None,
        'scaler_type': 'RobustScaler' if scaler is not None else None,
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    
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
        tuple: (model, scaler_params) where scaler_params is dict with 'mean' and 'scale' or None
    """
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Reconstruct model
    model = TennisNN(
        input_dim=model_data['input_dim'],
        hidden_dims=model_data['hidden_dims'],
        dropout=model_data['dropout'],
        temperature=model_data.get('temperature', 1.0)  # Backward compatibility
    )
    
    # Load weights
    state_dict = {k: torch.FloatTensor(v) for k, v in model_data['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load scaler parameters and reconstruct scaler object
    model.scaler = None
    if 'scaler_center' in model_data and model_data['scaler_center'] is not None:
        scaler = RobustScaler()
        scaler.center_ = np.array(model_data['scaler_center'])
        scaler.scale_ = np.array(model_data['scaler_scale'])
        model.scaler = scaler
    elif 'scaler_mean' in model_data and model_data['scaler_mean'] is not None:
        # Backward compatibility con vecchi modelli StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_data['scaler_mean'])
        scaler.scale_ = np.array(model_data['scaler_scale'])
        model.scaler = scaler
    
    # Store whether model uses clean features (for prediction pipeline)
    model.use_clean_features = model_data.get('use_clean_features', False)
    
    return model


def predict_nn(model, X):
    """
    Make predictions with neural network model.
    Applica normalizzazione se il modello è stato allenato con scaler.
    
    Args:
        model: Trained TennisNN model (con attributo scaler opzionale)
        X: Feature matrix (numpy array or single row) - valori grezzi
    
    Returns:
        P(P1 wins) predictions as numpy array
    """
    model.eval()
    
    # Handle single prediction
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # Check feature dimension mismatch
    if hasattr(model, 'input_dim') and X.shape[1] != model.input_dim:
        raise ValueError(
            f"Feature dimension mismatch: model expects {model.input_dim} features, "
            f"but input has {X.shape[1]} features. "
            f"This usually means the model was trained with different features. "
            f"Please retrain the model with: python tennisctl.py train-nn --files data/*.csv --model-out models/nn_model.json"
        )
    
    # Applica normalizzazione se presente
    if hasattr(model, 'scaler') and model.scaler is not None:
        X = model.scaler.transform(X)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        preds = model(X_tensor).numpy().flatten()
    
    return preds
