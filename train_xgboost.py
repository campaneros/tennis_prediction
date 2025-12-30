#!/usr/bin/env python3
"""
Standalone script to train XGBoost model for tennis match prediction.
Can be run independently of neural network training.

Usage:
    python train_xgboost.py --help
    python train_xgboost.py --files data/*matches.csv --model-out models/xgboost_model.json
    python train_xgboost.py --files data/2017*matches.csv data/2018*matches.csv --model-out models/xgb_2017_2018.json --gender male
"""

import argparse
import os
import sys

# Handle imports for both standalone and module execution
try:
    from scripts.model import train_model
    from scripts.config import load_config
except ImportError:
    from model import train_model
    from config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for tennis match prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on all available matches
  python train_xgboost.py --files data/*matches.csv --model-out models/xgb_all.json
  
  # Train on specific years
  python train_xgboost.py --files data/2017*matches.csv data/2018*matches.csv --model-out models/xgb_2017_2018.json
  
  # Train on male matches only
  python train_xgboost.py --files data/*matches.csv --model-out models/xgb_male.json --gender male
  
  # Train on female matches only
  python train_xgboost.py --files data/*matches.csv --model-out models/xgb_female.json --gender female
  
  # Use custom config
  python train_xgboost.py --files data/*matches.csv --model-out models/xgb_custom.json --config configs/custom_config.json
        """
    )
    
    parser.add_argument(
        "--files", 
        nargs="+", 
        required=True,
        help="List of CSV point-by-point match files (e.g., data/*matches.csv)"
    )
    
    parser.add_argument(
        "--model-out", 
        required=True,
        help="Path to save trained XGBoost model (e.g., models/xgboost_model.json)"
    )
    
    parser.add_argument(
        "--config", 
        default=None,
        help="Path to JSON config file with hyperparameters (default: configs/config.json)"
    )
    
    parser.add_argument(
        "--gender", 
        choices=["male", "female", "both"], 
        default="male",
        help="Filter dataset by gender: 'male' (match_id<2000), 'female' (match_id>=2000), 'both' (all matches). Default: male"
    )
    
    args = parser.parse_args()
    
    # Verify input files exist
    missing_files = [f for f in args.files if not os.path.exists(f)]
    if missing_files:
        print(f"ERROR: The following files do not exist:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.model_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")
    
    # Load config if provided
    config_path = args.config
    if config_path and not os.path.exists(config_path):
        print(f"WARNING: Config file not found: {config_path}")
        print("Using default config from configs/config.json")
        config_path = None
    
    print("=" * 80)
    print("XGBoost Tennis Match Prediction - Training")
    print("=" * 80)
    print(f"Input files: {len(args.files)} CSV files")
    for f in args.files[:5]:  # Show first 5 files
        print(f"  - {f}")
    if len(args.files) > 5:
        print(f"  ... and {len(args.files) - 5} more")
    print(f"Output model: {args.model_out}")
    print(f"Gender filter: {args.gender}")
    print(f"Config: {config_path or 'default (configs/config.json)'}")
    print("=" * 80)
    print()
    
    # Train the model
    try:
        print("Starting training...")
        train_model(
            file_paths=args.files,
            model_out=args.model_out,
            config_path=config_path,
            gender=args.gender
        )
        print()
        print("=" * 80)
        print(f"✓ Training completed successfully!")
        print(f"✓ Model saved to: {args.model_out}")
        print("=" * 80)
        print()
        print("To use this model for predictions:")
        print(f"  python tennisctl.py predict --model {args.model_out} --match-file <match_file.csv>")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Training failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
