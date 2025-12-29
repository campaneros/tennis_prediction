import os
import json
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from .data_loader import load_points_multiple
from .features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    add_additional_features,
    build_dataset,
)
from .config import load_config

def _default_model():
    # Regressor with logistic link so we can train on soft labels directly.
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def _predict_proba_model(model, X_batch):
    """Return P1 win probability for either classifier or regressor."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_batch)[:, 1]
    preds = model.predict(X_batch)
    return np.clip(preds, 0.0, 1.0)


def train_model(file_paths, model_out, config_path=None, gender="male"):
    """
    Train the XGBoost model on one or more CSV files and save it to 'model_out'.

    Feature parameters (long_window, short_window, momentum_alpha) are loaded
    from the JSON config.
    
    Args:
        file_paths: List of CSV files to train on
        model_out: Path to save the trained model
        config_path: Path to config JSON file
        gender: Filter by gender - "male" (match_id<2000), "female" (match_id>=2000), or "both" (all)
    """
    if not file_paths:
        raise ValueError("train_model: no input files provided")

    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))

    df = load_points_multiple(file_paths)
    
    # Filter out rows with missing match_id
    df = df.dropna(subset=['match_id'])
    
    # Extract match number from match_id
    extracted = df['match_id'].str.extract(r'-(\d+)$')[0]
    # Drop rows where extraction failed (NaN)
    valid_mask = extracted.notna()
    df = df[valid_mask].copy()
    extracted = extracted[valid_mask]
    df['match_num'] = extracted.astype(int)
    
    original_count = len(df)
    
    # Filter by gender
    if gender == "male":
        # Men's matches: match_id < 2000 (best-of-5 sets)
        df = df[df['match_num'] < 2000].copy()
        gender_label = "men's matches (best-of-5)"
    elif gender == "female":
        # Women's matches: match_id >= 2000 (best-of-3 sets)
        df = df[df['match_num'] >= 2000].copy()
        gender_label = "women's matches (best-of-3)"
    else:  # both
        # Use all matches
        gender_label = "all matches (mixed)"
    
    # Exclude match 1701 to avoid test set leakage (only for male or both)
    if gender in ["male", "both"]:
        test_match_filter = ~df['match_id'].str.contains('1701')
        excluded_1701 = (~test_match_filter).sum()
        df = df[test_match_filter].copy()
    else:
        excluded_1701 = 0
    
    df = df.drop(columns=['match_num'])
    filtered_count = original_count - len(df)
    
    if filtered_count > 0:
        print(f"[train] Gender filter: {gender}")
        print(f"[train] Filtered {filtered_count} points:")
        if gender == "male":
            print(f"[train]   - Excluded women's matches (>=2000): {filtered_count - excluded_1701}")
        elif gender == "female":
            print(f"[train]   - Excluded men's matches (<2000): {filtered_count}")
        else:
            print(f"[train]   - Excluded none (using all)")
        if excluded_1701 > 0:
            print(f"[train]   - Match 1701 (test set): {excluded_1701}")
        print(f"[train] Training on {len(df)} points from {gender_label}")
    else:
        print(f"[train] Training on {len(df)} points from {gender_label}")
    
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)

    X, y_soft, _, sample_weights, y_hard = build_dataset(df)
    y_hard = y_hard.astype(int)
    train_cfg = cfg.get("training", {})
    weight_exp = float(train_cfg.get("sample_weight_exponent", 1.0))
    adjusted_weights = np.power(sample_weights, weight_exp)

    print("[train] dataset shape:", X.shape, "positives (P1 wins):", int(y_hard.sum()))
    print(f"[train] long_window={long_window}, short_window={short_window}, alpha={alpha}")
    print(f"[train] weight exponent: {weight_exp}")
    print(f"[train] sample weights - mean: {adjusted_weights.mean():.2f}, max: {adjusted_weights.max():.2f}")

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y_hard
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train_soft, y_test_soft = y_soft[idx_train], y_soft[idx_test]
    y_train_hard, y_test_hard = y_hard[idx_train], y_hard[idx_test]
    w_train, w_test = adjusted_weights[idx_train], adjusted_weights[idx_test]

    model = _default_model()
    model.fit(X_train, y_train_soft, sample_weight=w_train)

    y_proba = _predict_proba_model(model, X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test_hard, y_pred)
    try:
        auc = roc_auc_score(y_test_hard, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"[train] Test accuracy: {acc:.3f}")
    print(f"[train] Test ROC AUC:  {auc:.3f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    # Use get_booster().save_model() to avoid sklearn wrapper issues
    model.get_booster().save_model(model_out)
    print(f"[train] Model saved to: {model_out}")

def load_model(model_path: str):
    """
    Load either XGBoost or Neural Network model based on file extension.
    
    - .json with 'state_dict' key: Neural Network (PyTorch)
    - .json without 'state_dict': XGBoost (regressor or classifier)
    
    Returns the loaded model in appropriate format.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check if it's a neural network model
    if model_path.endswith('.json'):
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            # Neural network models have 'state_dict' key
            if 'state_dict' in model_data:
                from .model_nn import load_nn_model
                print(f"[load_model] Loading Neural Network model from {model_path}")
                return load_nn_model(model_path)
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to XGBoost loading
    
    # Try loading as XGBoost model
    last_error = None
    for cls in (XGBRegressor, XGBClassifier):
        try:
            model = cls()
            model.load_model(model_path)
            print(f"[load_model] Loaded XGBoost {cls.__name__} from {model_path}")
            return model
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to load model {model_path}: {last_error}")


def predict_with_model(model, X):
    """
    Make predictions with either XGBoost or Neural Network model.
    
    Args:
        model: Trained model (XGBoost or PyTorch)
        X: Feature matrix (numpy array)
    
    Returns:
        P(P1 wins) predictions as numpy array
    """
    # Check if it's a PyTorch model
    import torch.nn as nn
    if isinstance(model, nn.Module):
        # PyTorch neural network (TennisNN, TennisRulesNet, or MultiTaskTennisNN)
        with torch.no_grad():
            model.eval()
            X_tensor = torch.FloatTensor(X)
            output = model(X_tensor)
            
            # Handle different output formats
            if isinstance(output, dict):
                # Multi-task model returns dict with 'match', 'set', 'game' keys
                # Already has sigmoid applied, but might need temperature recalibration
                probs = output['match'].squeeze()
                
                # If model has temperature attribute, apply inverse then re-apply
                # This is needed because forward() applies sigmoid but training used temperature
                if hasattr(model, 'temperature') and model.temperature != 1.0:
                    # Convert back to logits approximately: logit = log(p/(1-p))
                    eps = 1e-7
                    probs_clipped = torch.clamp(probs, eps, 1-eps)
                    logits = torch.log(probs_clipped / (1 - probs_clipped))
                    # Apply temperature scaling: divide logits by temperature
                    calibrated_logits = logits / model.temperature
                    # Apply sigmoid to get calibrated probabilities
                    probs = torch.sigmoid(calibrated_logits)
                
                probs = probs.cpu().numpy()
            elif isinstance(output, tuple):
                # Old format: (match_logits, set_logits, game_logits)
                probs = torch.sigmoid(output[0]).squeeze().cpu().numpy()
            else:
                # Single output
                probs = output.squeeze().cpu().numpy()
            
            return probs if probs.ndim > 0 else np.array([probs])
    elif hasattr(model, 'network'):  # TennisNN has 'network' attribute
        from .model_nn import predict_nn
        return predict_nn(model, X)
    else:
        # XGBoost model
        return _predict_proba_model(model, X)
