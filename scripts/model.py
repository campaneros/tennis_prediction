import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from .data_loader import load_points_multiple, WINDOW
from .features import add_match_labels, add_rolling_serve_return_features, build_dataset


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


def train_model(file_paths, model_out: str):
    if not file_paths:
        raise ValueError("train_model: no input files provided")

    df = load_points_multiple(file_paths)
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, window=WINDOW)

    X, y, _ = build_dataset(df)
    print("[train] dataset shape:", X.shape, "positives (P1 wins):", int(y.sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = _default_model()
    model.fit(X_train, y_train)

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
