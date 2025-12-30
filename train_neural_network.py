#!/usr/bin/env python3
"""
Standalone script to train Neural Network model for tennis match prediction.
Can be run independently of XGBoost training.

Supports multiple training modes:
1. Standard NN training (45 features)
2. Clean features NN training (25 features) 
3. Multi-task NN training (23 features, predicts match/set/game)
4. Transfer learning (pre-train on synthetic + fine-tune on real)

Usage:
    python train_neural_network.py --help
    python train_neural_network.py --files data/*matches.csv --model-out models/nn_model.pth
    python train_neural_network.py --files data/*matches.csv --model-out models/nn_clean.pth --clean-features
    python train_neural_network.py --files data/*matches.csv --model-out models/nn_multitask.pth --new-model
"""

import argparse
import os
import sys

# Handle imports for both standalone and module execution
try:
    from scripts.model_nn import train_nn_model
    from scripts.new_model_nn import train_new_model
    from scripts.pretrain_tennis_rules import pretrain_tennis_rules
    from scripts.transfer_learning import fine_tune_on_real_data
except ImportError:
    from model_nn import train_nn_model
    from new_model_nn import train_new_model
    from pretrain_tennis_rules import pretrain_tennis_rules
    from transfer_learning import fine_tune_on_real_data


def main():
    parser = argparse.ArgumentParser(
        description="Train Neural Network model for tennis match prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  1. Standard NN (45 features):
     python train_neural_network.py --files data/*matches.csv --model-out models/nn_standard.pth
  
  2. Clean features NN (25 features, better for learning rules):
     python train_neural_network.py --files data/*matches.csv --model-out models/nn_clean.pth --clean-features
  
  3. Multi-task NN (23 features, predicts match/set/game):
     python train_neural_network.py --files data/*matches.csv --model-out models/nn_multitask.pth --new-model
  
  4. Transfer learning (pre-train on synthetic, then fine-tune):
     # Step 1: Pre-train on synthetic matches
     python train_neural_network.py --mode pretrain --model-out models/pretrained.pth --n-synthetic 50000
     
     # Step 2: Fine-tune on real data
     python train_neural_network.py --mode finetune --files data/*matches.csv --pretrained models/pretrained.pth --model-out models/complete.pth

Hyperparameters:
  --epochs: Number of training epochs (default: 200 for standard, 50 for pretrain, 30 for finetune)
  --batch-size: Batch size (default: 1024 for standard, 2048 for pretrain)
  --temperature: Temperature for calibration (default: 3.0, use 3.0-5.0 for synthetic data)
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["train", "pretrain", "finetune"],
        default="train",
        help="Training mode: 'train' (standard), 'pretrain' (synthetic data), 'finetune' (transfer learning). Default: train"
    )
    
    # Input/output
    parser.add_argument(
        "--files", 
        nargs="+",
        help="List of CSV point-by-point match files (required for 'train' and 'finetune' modes)"
    )
    
    parser.add_argument(
        "--model-out", 
        required=True,
        help="Path to save trained model (e.g., models/nn_model.pth)"
    )
    
    parser.add_argument(
        "--pretrained",
        help="Path to pre-trained model checkpoint (required for 'finetune' mode)"
    )
    
    # Model type
    parser.add_argument(
        "--clean-features", 
        action="store_true",
        help="Use clean feature set (25 features) instead of full set (45). Better for learning tennis rules."
    )
    
    parser.add_argument(
        "--new-model", 
        action="store_true",
        help="Use new multi-task model with distance features (23 features). Predicts match/set/game simultaneously."
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", 
        type=int,
        help="Number of epochs (default: 200 for train, 50 for pretrain, 30 for finetune)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int,
        help="Batch size (default: 1024 for train/finetune, 2048 for pretrain)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=3.0,
        help="Temperature for calibration (default: 3.0, use 3.0-5.0 for synthetic data)"
    )
    
    # Pretrain specific
    parser.add_argument(
        "--n-synthetic", 
        type=int, 
        default=50000,
        help="Number of synthetic matches to generate for pre-training (default: 50000)"
    )
    
    # Data filtering
    parser.add_argument(
        "--gender", 
        choices=["male", "female", "both"], 
        default="male",
        help="Filter dataset by gender (default: male)"
    )
    
    parser.add_argument(
        "--device", 
        choices=["cuda", "cpu"], 
        default="cuda",
        help="Device to use for training (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Set default epochs and batch_size based on mode
    if args.epochs is None:
        if args.mode == "pretrain":
            args.epochs = 50
        elif args.mode == "finetune":
            args.epochs = 30
        else:
            args.epochs = 200
    
    if args.batch_size is None:
        if args.mode == "pretrain":
            args.batch_size = 2048
        else:
            args.batch_size = 1024
    
    # Validation
    if args.mode in ["train", "finetune"] and not args.files:
        parser.error(f"--files is required for mode '{args.mode}'")
    
    if args.mode == "finetune" and not args.pretrained:
        parser.error("--pretrained is required for mode 'finetune'")
    
    if args.mode in ["train", "finetune"] and args.files:
        missing_files = [f for f in args.files if not os.path.exists(f)]
        if missing_files:
            print(f"ERROR: The following files do not exist:")
            for f in missing_files:
                print(f"  - {f}")
            sys.exit(1)
    
    if args.mode == "finetune" and not os.path.exists(args.pretrained):
        print(f"ERROR: Pre-trained model not found: {args.pretrained}")
        sys.exit(1)
    
    # Create output directory
    out_dir = os.path.dirname(args.model_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")
    
    # Print configuration
    print("=" * 80)
    print(f"Neural Network Tennis Prediction - {args.mode.upper()} Mode")
    print("=" * 80)
    
    if args.files:
        print(f"Input files: {len(args.files)} CSV files")
        for f in args.files[:5]:
            print(f"  - {f}")
        if len(args.files) > 5:
            print(f"  ... and {len(args.files) - 5} more")
    
    if args.pretrained:
        print(f"Pre-trained model: {args.pretrained}")
    
    print(f"Output model: {args.model_out}")
    
    if args.new_model:
        print(f"Model type: Multi-task (23 features)")
    elif args.clean_features:
        print(f"Model type: Clean features (25 features)")
    else:
        print(f"Model type: Standard (45 features)")
    
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {args.device}")
    
    if args.mode == "pretrain":
        print(f"Synthetic matches: {args.n_synthetic}")
    else:
        print(f"Gender filter: {args.gender}")
    
    print("=" * 80)
    print()
    
    # Train the model
    try:
        if args.mode == "pretrain":
            print("Starting pre-training on synthetic data...")
            pretrain_tennis_rules(
                n_matches=args.n_synthetic,
                epochs=args.epochs,
                batch_size=args.batch_size,
                temperature=args.temperature,
                output_path=args.model_out,
                device=args.device
            )
            
        elif args.mode == "finetune":
            print("Starting fine-tuning on real data...")
            fine_tune_on_real_data(
                pretrained_path=args.pretrained,
                real_data_files=args.files,
                output_path=args.model_out,
                gender=args.gender,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            
        else:  # train mode
            print("Starting training...")
            if args.new_model:
                train_new_model(
                    file_paths=args.files,
                    model_out=args.model_out,
                    gender=args.gender,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    temperature=args.temperature
                )
            else:
                train_nn_model(
                    file_paths=args.files,
                    model_out=args.model_out,
                    gender=args.gender,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    clean_features=args.clean_features
                )
        
        print()
        print("=" * 80)
        print(f"✓ Training completed successfully!")
        print(f"✓ Model saved to: {args.model_out}")
        print("=" * 80)
        print()
        
        if args.mode == "pretrain":
            print("Next step: Fine-tune on real data")
            print(f"  python train_neural_network.py --mode finetune --files data/*matches.csv --pretrained {args.model_out} --model-out models/complete.pth")
        else:
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
