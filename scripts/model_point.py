"""
Point-level prediction model: predicts who wins the NEXT POINT instead of who wins the match.
This eliminates look-ahead bias since the model doesn't know the match outcome.
"""
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from .data_loader import load_points_multiple
from .features import (
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    add_additional_features,
    build_dataset_point_level,
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


def train_point_model(file_paths, model_out: str, config_path: str | None = None):
    """
    Train XGBoost model to predict POINT winner (not match winner).
    
    Uses same features as match-level model but with different target:
    - Target: 1 if P1 wins the current point, 0 if P2 wins
    - No look-ahead bias: model doesn't know who wins the match
    """
    if not file_paths:
        raise ValueError("train_point_model: no input files provided")

    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 0.35))

    df = load_points_multiple(file_paths)
    
    # Filter women's matches (best-of-3 only)
    df['match_num'] = df['match_id'].str.extract(r'-(\d+)$')[0].astype(int)
    original_count = len(df)
    df = df[df['match_num'] < 2000].copy()
    df = df.drop(columns=['match_num'])
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"[train-point] Filtered {filtered_count} points from women's matches")
        print(f"[train-point] Training on {len(df)} points from men's matches")
    
    # Build features (no match labels needed for point prediction)
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)

    # Build dataset with POINT winner as target
    X, y, _, sample_weights = build_dataset_point_level(df)
    
    train_cfg = cfg.get("training", {})
    weight_exp = float(train_cfg.get("sample_weight_exponent", 0.5))
    adjusted_weights = np.power(sample_weights, weight_exp)

    print("[train-point] dataset shape:", X.shape, "P1 wins point:", int(y.sum()), f"({y.mean()*100:.1f}%)")
    print(f"[train-point] long_window={long_window}, short_window={short_window}, alpha={alpha}")
    print(f"[train-point] weight exponent: {weight_exp}")
    print(f"[train-point] sample weights - mean: {adjusted_weights.mean():.2f}, max: {adjusted_weights.max():.2f}")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, adjusted_weights, test_size=0.2, random_state=42, stratify=y
    )

    model = _default_model()
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)

    print(f"[train-point] Test accuracy: {acc:.3f}")
    print(f"[train-point] Test ROC AUC:  {roc:.3f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    model.save_model(model_out)
    print(f"[train-point] Model saved to: {model_out}")

    return model
