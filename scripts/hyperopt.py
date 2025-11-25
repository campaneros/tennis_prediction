import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .data_loader import load_points_multiple, WINDOW
from .features import add_match_labels, add_rolling_serve_return_features, build_dataset
from .plotting import plot_hyperopt_results, plot_confusion_matrix_and_roc


def run_hyperopt(file_paths, n_iter: int, plot_dir: str, model_out: str):
    """
    Run RandomizedSearchCV hyperparameter optimisation on the given files.

    - Uses 5-fold CV.
    - Tracks mean accuracy, precision, recall, F1, and AUC for each model.
    - Saves the best model to 'model_out'.
    - Writes a CSV with metrics for all tested models.
    - For the best model, builds a 5-fold CV confusion matrix and ROC curve
      with label convention: 0 = Player 1 wins, 1 = Player 2 wins.
    """
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Load and build features
    # ------------------------------------------------------------------
    df = load_points_multiple(file_paths)
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, window=WINDOW)
    X, y, _ = build_dataset(df)

    print("[hyperopt] dataset shape:", X.shape, "positives (P1 wins):", int(y.sum()))

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # Hyperparameter space + scoring metrics
    # ------------------------------------------------------------------
    param_distributions = {
        "n_estimators":     [200, 400, 600, 800],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0.0, 0.1, 0.3, 0.5],
    }

    # Multiple metrics; refit on best AUC
    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="roc_auc",          # best model chosen by mean AUC
        cv=5,                     # 5-fold CV
        verbose=1,
        random_state=42,
        n_jobs=-1,
        return_train_score=False,
    )

    search.fit(X, y)

    print("[hyperopt] Best parameters:")
    print(search.best_params_)
    print(f"[hyperopt] Best CV ROC AUC: {search.best_score_:.3f}")

    # ------------------------------------------------------------------
    # Save tuned model
    # ------------------------------------------------------------------
    best_model = search.best_estimator_
    best_model.save_model(model_out)
    print(f"[hyperopt] Tuned model saved to: {model_out}")

    # ------------------------------------------------------------------
    # Save per-model CV metrics (for all hyperparameter points)
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(search.cv_results_)

    metric_cols = [
        "mean_test_roc_auc",
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
    ]
    # Map from scoring keys to cv_results_ column names
    rename_map = {
        "mean_test_roc_auc": "AUC",
        "mean_test_accuracy": "Accuracy",
        "mean_test_precision": "Precision",
        "mean_test_recall": "Recall",
        "mean_test_f1": "F1",
    }

    # Extract parameters and metrics
    param_cols = [c for c in results_df.columns if c.startswith("param_")]
    metrics_df = results_df[param_cols + metric_cols].copy()
    metrics_df = metrics_df.rename(columns=rename_map)

    metrics_csv = os.path.join(plot_dir, "hyperopt_cv_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[hyperopt] Per-model CV metrics written to: {metrics_csv}")

    # ------------------------------------------------------------------
    # Hyperparameter diagnostic plots (AUC vs params, etc.)
    # ------------------------------------------------------------------
    plot_hyperopt_results(search.cv_results_, plot_dir, prefix="hyperopt")
    print(f"[hyperopt] Hyperparameter plots written in: {plot_dir}")

    # ------------------------------------------------------------------
    # 5-fold CV evaluation for the best model: confusion matrix + ROC
    # ------------------------------------------------------------------
    print("[hyperopt] Building 5-fold CV confusion matrix and ROC for best model...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated predicted probabilities for the "P1 wins" class (label 1)
    y_proba_p1 = cross_val_predict(
        best_model, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    # ------------------------------------------------------------------
    # For the confusion matrix and ROC we use the paper's convention:
    #   0 = Player 1 wins (our y = 1)
    #   1 = Player 2 wins (our y = 0)
    #
    # So we flip the labels:
    #   y_cm = 1 - y
    #   p(P2 wins) = 1 - p(P1 wins)
    # ------------------------------------------------------------------
    y_cm = 1 - y
    y_proba_p2 = 1.0 - y_proba_p1

    # Hard predictions with threshold 0.5 on "P2 wins"
    y_pred_cm = (y_proba_p2 >= 0.5).astype(int)

    cm = confusion_matrix(y_cm, y_pred_cm, labels=[0, 1])

    # Global metrics (using the 0/1 coding P1/P2)
    auc_value = roc_auc_score(y_cm, y_proba_p2)
    acc_value = accuracy_score(y_cm, y_pred_cm)
    prec_value = precision_score(y_cm, y_pred_cm, zero_division=0)
    rec_value = recall_score(y_cm, y_pred_cm, zero_division=0)
    f1_value = f1_score(y_cm, y_pred_cm, zero_division=0)

    print("[hyperopt] Best-model CV metrics (5-fold, P1=0, P2=1):")
    print(f"  Accuracy : {acc_value:.3f}")
    print(f"  Precision: {prec_value:.3f}")
    print(f"  Recall   : {rec_value:.3f}")
    print(f"  F1 score : {f1_value:.3f}")
    print(f"  AUC      : {auc_value:.3f}")

    # Plot confusion matrix + ROC curve in a single figure
    plot_confusion_matrix_and_roc(
        cm=cm,
        y_true=y_cm,
        y_score=y_proba_p2,
        auc_value=auc_value,
        acc_value=acc_value,
        prec_value=prec_value,
        rec_value=rec_value,
        f1_value=f1_value,
        plot_dir=plot_dir,
        filename_prefix="best_model_cv",
    )
    print(f"[hyperopt] Confusion matrix + ROC figure saved in: {plot_dir}")
