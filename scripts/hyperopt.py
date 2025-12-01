import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, KFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .data_loader import load_points_multiple, WINDOW
from .config import load_config
from .features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    add_additional_features,
    build_dataset,
)
from .plotting import plot_hyperopt_results, plot_confusion_matrix_and_roc

def run_hyperopt(file_paths, n_iter: int, plot_dir: str, model_out: str, config_path: str | None = None, search_type: str = "random"):
    """
    Run GridSearchCV or RandomizedSearchCV hyperparameter optimisation on the given files.

    - Uses 5-fold CV.
    - Tracks mean Accuracy, Precision, Recall, F1, and AUC for each model.
    - Saves the best model to 'model_out'.
    - Writes a CSV with CV metrics for all tested models.
    - For *every* model in the hyperparameter scan, computes a 5-fold CV
      confusion matrix + ROC curve and saves a figure.

      Label convention for confusion/ROC:
        0 = Player 1 wins
        1 = Player 2 (rival) wins
    """
    # Create separate directories for grid and random search
    search_plot_dir = os.path.join(plot_dir, search_type)
    os.makedirs(search_plot_dir, exist_ok=True)
    
    # Modify model output path to include search type
    model_dir = os.path.dirname(model_out) or "."
    model_name = os.path.basename(model_out)
    name_parts = os.path.splitext(model_name)
    search_model_out = os.path.join(model_dir, f"{name_parts[0]}_{search_type}{name_parts[1]}")
    os.makedirs(model_dir, exist_ok=True)
    
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    print(f"{fcfg}")
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))
    print(f"[hyperopt] long_window={long_window}, short_window={short_window}, alpha={alpha}")
    print(f"[hyperopt] Search type: {search_type}")
    print(f"[hyperopt] Output directory: {search_plot_dir}")
    print(f"[hyperopt] Model output: {search_model_out}")


    # ------------------------------------------------------------------
    # Load and build features
    # ------------------------------------------------------------------
    df = load_points_multiple(file_paths)
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)
    X, y_soft, _, sample_weights, y_hard = build_dataset(df)
    print("[hyperopt] dataset shape:", X.shape, "positives (P1 wins, soft):", float(y_soft.sum()))
    print(f"[hyperopt] sample weights - mean: {sample_weights.mean():.2f}, max: {sample_weights.max():.2f}")


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
        "n_estimators":     [200, 400, 600, 800, 1000],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.0001, 0.01, 0.05, 0.1, 0.2, 0.25],
        "subsample":        [0.4, 0.6, 0.8, 1.0],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma":            [0.0, 0.1, 0.3, 0.5, 0.7],
    }

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    if search_type == "grid":
        print(f"[hyperopt] Running GridSearchCV (all {np.prod([len(v) for v in param_distributions.values()])} combinations)")
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_distributions,
            scoring=scoring,
            refit="roc_auc",
            cv=5,
            verbose=1,
            n_jobs=-1,
            return_train_score=False,
        )
    else:
        print(f"[hyperopt] Running RandomizedSearchCV ({n_iter} iterations)")
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            refit="roc_auc",
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            return_train_score=False,
        )

    search.fit(X, y, sample_weight=sample_weights)

    print("[hyperopt] Best parameters:")
    print(search.best_params_)
    print(f"[hyperopt] Best CV ROC AUC: {search.best_score_:.3f}")

    # ------------------------------------------------------------------
    # Save tuned "best" model
    # ------------------------------------------------------------------
    best_model = search.best_estimator_
    best_model.save_model(search_model_out)
    print(f"[hyperopt] Tuned model saved to: {search_model_out}")

    # ------------------------------------------------------------------
    # Create models directory for saving all models
    # ------------------------------------------------------------------
    models_dir = os.path.join(model_dir, search_type)
    os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save per-model CV metrics (from RandomizedSearchCV) for all models
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(search.cv_results_)

    metric_cols = [
        "mean_test_roc_auc",
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
    ]
    rename_map = {
        "mean_test_roc_auc": "AUC",
        "mean_test_accuracy": "Accuracy",
        "mean_test_precision": "Precision",
        "mean_test_recall": "Recall",
        "mean_test_f1": "F1",
    }

    param_cols = [c for c in results_df.columns if c.startswith("param_")]
    metrics_df = results_df[param_cols + metric_cols].copy()
    metrics_df = metrics_df.rename(columns=rename_map)

    metrics_csv = os.path.join(search_plot_dir, "hyperopt_cv_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[hyperopt] Per-model CV metrics written to: {metrics_csv}")

    # ------------------------------------------------------------------
    # AUC vs hyperparameters plots, histogram, etc.
    # ------------------------------------------------------------------
    plot_hyperopt_results(search.cv_results_, search_plot_dir, prefix=f"hyperopt_{search_type}")
    print(f"[hyperopt] Hyperparameter plots written in: {search_plot_dir}")

    # ------------------------------------------------------------------
    # For *every* model: 5-fold CV confusion matrix + ROC curve
    # ------------------------------------------------------------------
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cm = 1 - y_hard  # 0 = P1 wins, 1 = P2 wins

    all_params = search.cv_results_["params"]
    print("[hyperopt] Computing confusion matrix + ROC for all models...")
    for idx, params in enumerate(all_params):
        model_i = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model_i.set_params(**params)

        # Cross-validated probabilities for "P1 wins" (class 1 in y)
        # Use soft labels for training; KFold avoids stratification issues with soft labels
        y_proba_p1 = cross_val_predict(
            model_i,
            X,
            y_soft,
            cv=cv,
            method="predict_proba",
            fit_params={"sample_weight": sample_weights},
        )[:, 1]

        # Flip to P2 probability + predictions for (0=P1,1=P2) convention
        y_proba_p2 = 1.0 - y_proba_p1
        y_pred_cm = (y_proba_p2 >= 0.5).astype(int)

        cm = confusion_matrix(y_cm, y_pred_cm, labels=[0, 1])

        auc_value = roc_auc_score(y_cm, y_proba_p2)
        acc_value = accuracy_score(y_cm, y_pred_cm)
        prec_value = precision_score(y_cm, y_pred_cm, zero_division=0)
        rec_value = recall_score(y_cm, y_pred_cm, zero_division=0)
        f1_value = f1_score(y_cm, y_pred_cm, zero_division=0)

        prefix = f"model_{idx:03d}"
        print(
            f"[hyperopt] Model {idx:03d}: "
            f"AUC={auc_value:.3f}, Acc={acc_value:.3f}, "
            f"Prec={prec_value:.3f}, Rec={rec_value:.3f}, F1={f1_value:.3f}"
        )
        
        # Save each model to models/{search_type}/model_{idx}.json
        model_path = os.path.join(models_dir, f"model_{idx:03d}.json")
        model_i.fit(X, y_soft, sample_weight=sample_weights)
        model_i.save_model(model_path)

        plot_confusion_matrix_and_roc(
            cm=cm,
            y_true=y_cm,
            y_score=y_proba_p2,
            auc_value=auc_value,
            acc_value=acc_value,
            prec_value=prec_value,
            rec_value=rec_value,
            f1_value=f1_value,
            plot_dir=search_plot_dir,
            filename_prefix=prefix,
        )

    print(f"[hyperopt] All models saved in: {models_dir}")
    print(f"[hyperopt] Confusion + ROC figures saved in: {search_plot_dir}")
