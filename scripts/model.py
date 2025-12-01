import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from .data_loader import load_points_multiple, WINDOW

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
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def train_model(file_paths, model_out: str, config_path=None):
    """
    Train the XGBoost model on one or more CSV files and save it to 'model_out'.

    Feature parameters (long_window, short_window, momentum_alpha) are loaded
    from the JSON config.
    """
    if not file_paths:
        raise ValueError("train_model: no input files provided")

    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))

    df = load_points_multiple(file_paths)
    
    # Filter out women's matches (match_id >= 2000) - they are best-of-3, not best-of-5
    # This ensures model learns from men's tennis where 5-set matches are possible
    df['match_num'] = df['match_id'].str.extract(r'-(\d+)$')[0].astype(int)
    original_count = len(df)
    df = df[df['match_num'] < 2000].copy()
    
    # Also exclude match 1701 to avoid test set leakage
    test_match_filter = ~df['match_id'].str.contains('1701')
    excluded_1701 = (~test_match_filter).sum()
    df = df[test_match_filter].copy()
    
    df = df.drop(columns=['match_num'])
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"[train] Filtered {filtered_count} points total:")
        print(f"[train]   - Women's matches (best-of-3): {filtered_count - excluded_1701}")
        print(f"[train]   - Match 1701 (test set): {excluded_1701}")
        print(f"[train] Training on {len(df)} points from men's matches")
    
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)

    X, _, _, sample_weights, y = build_dataset(df)
    y = y.astype(int)
    train_cfg = cfg.get("training", {})
    weight_exp = float(train_cfg.get("sample_weight_exponent", 1.0))
    adjusted_weights = np.power(sample_weights, weight_exp)

    print("[train] dataset shape:", X.shape, "positives (P1 wins):", int(y.sum()))
    print(f"[train] long_window={long_window}, short_window={short_window}, alpha={alpha}")
    print(f"[train] weight exponent: {weight_exp}")
    print(f"[train] sample weights - mean: {adjusted_weights.mean():.2f}, max: {adjusted_weights.max():.2f}")

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    w_train, w_test = adjusted_weights[idx_train], adjusted_weights[idx_test]

    model = _default_model()
    model.fit(X_train, y_train, sample_weight=w_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"[train] Test accuracy: {acc:.3f}")
    print(f"[train] Test ROC AUC:  {auc:.3f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    model.save_model(model_out)
    print(f"[train] Model saved to: {model_out}")

def load_model(model_path: str) -> XGBClassifier:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    return model
