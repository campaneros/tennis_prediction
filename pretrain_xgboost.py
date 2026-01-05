#!/usr/bin/env python3
"""
Pre-train XGBoost on synthetic tennis data (like v5 neural network pre-training).

This teaches XGBoost the basic rules and dynamics of tennis before fine-tuning on real data.

Usage:
    python pretrain_xgboost.py --synthetic-file data/synthetic_tennis_12M.csv --model-out models/xgboost_pretrained.json
    python train_xgboost.py --files data/2017*points.csv --model-out models/xgboost_finetuned.json --pretrained models/xgboost_pretrained.json
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    from scripts.data_loader import load_points_multiple
    from scripts.features import (
        add_match_labels,
        add_rolling_serve_return_features,
        add_leverage_and_momentum,
        add_additional_features,
        build_dataset,
    )
    from scripts.config import load_config
except ImportError:
    from data_loader import load_points_multiple
    from features import (
        add_match_labels,
        add_rolling_serve_return_features,
        add_leverage_and_momentum,
        add_additional_features,
        build_dataset,
    )
    from config import load_config


def pretrain_xgboost(data_files, model_out: str, config_path: str = None, 
                     n_estimators: int = 400, max_depth: int = 4):
    """
    Pre-train XGBoost on real tennis data from 2011-2017.
    
    Args:
        data_files: List of paths to data CSV files
        model_out: Output path for pre-trained model
        config_path: Config file path
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
    """
    print("=" * 80)
    print("XGBoost Pre-training on Real Tennis Data")
    print("=" * 80)
    print(f"Data files: {len(data_files)} files")
    print(f"Output model: {model_out}")
    print(f"n_estimators: {n_estimators}, max_depth: {max_depth}")
    print("=" * 80)
    print()
    
    # Load config
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 0.35))
    
    # Load data
    print("[pretrain] Loading data...")
    df = load_points_multiple(data_files)
    print(f"[pretrain] Loaded {len(df)} points from {len(data_files)} files")
    
    # Build features
    print("[pretrain] Building features...")
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)
    
    X, y_soft, _, sample_weights, y_hard = build_dataset(df)
    y_hard = y_hard.astype(int)
    
    print(f"[pretrain] Dataset shape: {X.shape}, positives: {int(y_hard.sum())}")
    print(f"[pretrain] P1 win rate: {y_hard.mean():.4f}")
    
    # Reduce sample weights for pre-training (smoother learning)
    train_cfg = cfg.get("training", {})
    weight_exp = 0.3  # Lower than fine-tuning (0.5) for gentler pre-training
    adjusted_weights = np.power(sample_weights, weight_exp)
    print(f"[pretrain] Sample weights - mean: {adjusted_weights.mean():.2f}, max: {adjusted_weights.max():.2f}")
    
    # Create XGBoost model
    print("[pretrain] Initializing XGBoost model...")
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    
    # Train
    print("[pretrain] Training XGBoost on real data...")
    model.fit(X, y_soft, sample_weight=adjusted_weights, verbose=True)
    
    # Save model using booster
    print(f"[pretrain] Saving pre-trained model to {model_out}...")
    model.get_booster().save_model(model_out)
    
    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred_proba = model.predict(X)
    y_pred_hard = (y_pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_hard, y_pred_hard)
    auc = roc_auc_score(y_hard, y_pred_proba)
    
    print()
    print("=" * 80)
    print(f"✓ Pre-training completed!")
    print(f"  Training accuracy: {acc:.4f}")
    print(f"  Training ROC AUC: {auc:.4f}")
    print(f"  Model saved: {model_out}")
    print("=" * 80)
    print()
    print("Next step: Fine-tune on real data:")
    print(f"  python train_xgboost.py --files data/*points.csv --model-out models/xgb_finetuned.json --pretrained {model_out}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train XGBoost on synthetic tennis data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing CSV files (e.g., data/)"
    )
    
    parser.add_argument(
        "--data-files",
        type=str,
        nargs="+",
        default=None,
        help="Specific data files to use (e.g., data/2011-ausopen-points.csv). If not specified, uses all 2011-2017 files."
    )
    
    parser.add_argument(
        "--model-out",
        required=True,
        help="Path to save pre-trained model (e.g., models/xgboost_pretrained.json)"
    )
    
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: configs/config.json)"
    )
    
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of boosting rounds (default: 400)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth (default: 4)"
    )
    
    args = parser.parse_args()
    
    # If no specific files provided, use all 2011-2017 data
    if args.data_files is None:
        import glob
        args.data_files = sorted(glob.glob(f"{args.data_dir}/201[1-7]-*-points.csv"))
        print(f"Using {len(args.data_files)} files from 2011-2017 for pre-training")
    
    if not args.data_files:
        print("ERROR: No data files found!")
        print()
        print("Please provide data files:")
        print("  python pretrain_xgboost.py --data-files data/2011-*-points.csv --model-out models/xgb_pretrained.json")
        sys.exit(1)
    
    # Create output directory
    out_dir = os.path.dirname(args.model_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    try:
        pretrain_xgboost(
            args.data_files,
            args.model_out,
            config_path=args.config,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Pre-training failed:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
