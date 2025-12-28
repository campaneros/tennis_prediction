#!/usr/bin/env python3
import argparse

from .model import train_model
from .model_nn import train_nn_model
from .new_model_nn import train_new_model
from .model_point import train_point_model
from .prediction import run_prediction
from .hyperopt import run_hyperopt
from .plotting import plot_match_probabilities
from .pretrain_tennis_rules import pretrain_tennis_rules
from .transfer_learning import fine_tune_on_real_data
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(
        description="Tennis Counterfactual CLI Tool"
    )
    subparsers = parser.add_subparsers(dest="command")

# TRAIN
    train_p = subparsers.add_parser("train", help="Train a model (match-level)")
    train_p.add_argument("--files", nargs="+", required=True,
                         help="List of CSV point-by-point files")
    train_p.add_argument("--model-out", required=True,
                         help="Path to save trained model (JSON)")
    train_p.add_argument("--config", default=None,
                         help="Path to JSON config file (default: config.json)")
    train_p.add_argument("--gender", choices=["male", "female", "both"], default="male",
                         help="Filter dataset by gender: 'male' (match_id<2000), 'female' (match_id>=2000), 'both' (all matches)")
    train_p.add_argument("--model-type", choices=["xgboost", "nn"], default="xgboost",
                         help="Model type: 'xgboost' (gradient boosting, default) or 'nn' (neural network)")
    train_p.add_argument("--epochs", type=int, default=200,
                         help="Number of epochs for neural network training (default: 200)")
    train_p.add_argument("--batch-size", type=int, default=1024,
                         help="Batch size for neural network training (default: 1024)")
    train_p.add_argument("--clean-features", action="store_true",
                         help="Use clean feature set (25 features) for neural network instead of full set (45). Recommended for better learning of tennis rules.")
    train_p.add_argument("--new-model", action="store_true",
                         help="Use new multi-task model with distance features (23 features). Predicts match, set, and game outcomes simultaneously to learn tennis hierarchy.")

    # TENNIS-TRAINING (pre-training on synthetic matches)
    tennis_train_p = subparsers.add_parser("tennis-training", 
                                            help="Pre-train model on synthetic tennis matches to learn scoring rules")
    tennis_train_p.add_argument("--n-matches", type=int, default=50000,
                                help="Number of synthetic matches to generate (default: 50000)")
    tennis_train_p.add_argument("--epochs", type=int, default=50,
                                help="Number of training epochs (default: 50)")
    tennis_train_p.add_argument("--batch-size", type=int, default=2048,
                                help="Batch size (default: 2048)")
    tennis_train_p.add_argument("--temperature", type=float, default=3.0,
                                help="Temperature for calibration (default: 3.0, use 3.0-5.0 for synthetic data)")
    tennis_train_p.add_argument("--model-out", required=True,
                                help="Path to save pre-trained model (e.g., models/tennis_rules_pretrained.pth)")
    tennis_train_p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                                help="Device to use for training (default: cuda)")

    # COMPLETE-MODEL (fine-tuning on real data)
    complete_p = subparsers.add_parser("complete-model",
                                       help="Fine-tune pre-trained model on real tennis data")
    complete_p.add_argument("--files", nargs="+", required=True,
                            help="List of CSV point-by-point files (real data)")
    complete_p.add_argument("--pretrained", required=True,
                            help="Path to pre-trained model checkpoint")
    complete_p.add_argument("--model-out", required=True,
                            help="Path to save fine-tuned model")
    complete_p.add_argument("--gender", choices=["male", "female", "both"], default="male",
                            help="Filter dataset by gender (default: male)")
    complete_p.add_argument("--epochs", type=int, default=30,
                            help="Number of fine-tuning epochs (default: 30)")
    complete_p.add_argument("--batch-size", type=int, default=1024,
                            help="Batch size (default: 1024)")
    complete_p.add_argument("--learning-rate", type=float, default=0.0001,
                            help="Learning rate for fine-tuning (default: 0.0001)")
    complete_p.add_argument("--temperature", type=float, default=12.0,
                            help="Temperature for calibration (default: 12.0)")
    complete_p.add_argument("--freeze-layers", action="store_true",
                            help="Freeze first layer during fine-tuning (only train task heads)")
    complete_p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                            help="Device to use for training (default: cuda)")


    # TRAIN-POINT (new)
    train_point_p = subparsers.add_parser("train-point", help="Train point-level model (predicts point winner)")
    train_point_p.add_argument("--files", nargs="+", required=True,
                              help="List of CSV point-by-point files")
    train_point_p.add_argument("--model-out", required=True,
                              help="Path to save trained model (JSON)")
    train_point_p.add_argument("--config", default=None,
                              help="Path to JSON config file (default: config.json)")

    # PREDICT
    pred_p = subparsers.add_parser("predict", help="Predict probabilities and plot")
    pred_p.add_argument("--files", nargs="+", required=True,
                        help="List of CSV point-by-point files")
    pred_p.add_argument("--model", required=True,
                        help="Path to trained model JSON")
    pred_p.add_argument("--match-id", required=True,
                        help="match_id to plot (string or numeric)")
    pred_p.add_argument("--plot-dir", default="plots",
                        help="Directory to store plots")
    pred_p.add_argument("--config", default=None,
                        help="Path to JSON config file (default: config.json)")
    pred_p.add_argument("--mode", choices=["importance", "semi-realistic", "realistic"], 
                        default="importance",
                        help="Counterfactual mode: 'importance' (fast, default), 'semi-realistic' (critical points only), 'realistic' (all points, slow)")
    pred_p.add_argument("--point-by-point", action="store_true",
                        help="Rebuild dataset point by point (slower but more accurate for counterfactuals). If not set, uses full match info (faster).")
    pred_p.add_argument("--gender", choices=["male", "female", "both"], default="male",
                        help="Filter dataset by gender: 'male' (match_id<2000), 'female' (match_id>=2000), 'both' (all matches)")

    # REPLOT
    replot_p = subparsers.add_parser("replot", help="Regenerate plot from saved probabilities CSV")
    replot_p.add_argument("--csv", required=True,
                          help="Path to saved probabilities CSV file")
    replot_p.add_argument("--plot-dir", default="plots",
                          help="Directory to store plots")

    # HYPEROPT
    hyp_p = subparsers.add_parser("hyperopt", help="Hyperparameter optimisation")
    hyp_p.add_argument("--files", nargs="+", required=True,
                       help="List of CSV point-by-point files")
    hyp_p.add_argument("--n-iter", type=int, default=30,
                       help="Number of RandomizedSearchCV iterations")
    hyp_p.add_argument("--plot-dir", default="hyperopt_plots",
                       help="Directory to store hyperopt plots")
    hyp_p.add_argument("--model-out", required=True,
                       help="Path to save tuned model JSON")
    hyp_p.add_argument("--search-type", choices=["grid", "random"], default="random",
                       help="Search strategy: 'grid' for exhaustive, 'random' for sampling (default: random)")
    hyp_p.add_argument("--config", default=None,
                       help="Path to JSON config file (default: config.json)")
    args = parser.parse_args()

    if args.command == "train":
        gender = getattr(args, 'gender', 'male')
        model_type = getattr(args, 'model_type', 'xgboost')
        
        if model_type == 'nn':
            epochs = getattr(args, 'epochs', 100)
            batch_size = getattr(args, 'batch_size', 512)
            use_clean_features = getattr(args, 'clean_features', False)
            use_new_model = getattr(args, 'new_model', False)
            
            if use_new_model:
                print("[CLI] Using NEW multi-task model with distance features")
                train_new_model(args.files, args.model_out, gender=gender, 
                              epochs=epochs, batch_size=batch_size)
            else:
                train_nn_model(args.files, args.model_out, config_path=args.config, 
                              gender=gender, epochs=epochs, batch_size=batch_size,
                              use_clean_features=use_clean_features)
        else:
            train_model(args.files, args.model_out, config_path=args.config, gender=gender)
    
    elif args.command == "train-point":
        train_point_model(args.files, args.model_out, config_path=args.config)

    elif args.command == "predict":
        mode = getattr(args, 'mode', 'importance')
        gender = getattr(args, 'gender', 'male')
        point_by_point = getattr(args, 'point_by_point', False)
        run_prediction(args.files, args.model, args.match_id, args.plot_dir, 
                      config_path=args.config, counterfactual_mode=mode, gender=gender,
                      point_by_point=point_by_point)

    elif args.command == "replot":
        # Regenerate plot from saved CSV
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            return
        
        df = pd.read_csv(args.csv)
        if 'counterfactual_computed' not in df.columns:
            df['counterfactual_computed'] = True
        match_id = df['match_id'].iloc[0]
        os.makedirs(args.plot_dir, exist_ok=True)
        
        has_alt = all(col in df.columns for col in ["prob_p1_alt","prob_p2_alt","prob_p1_lose_alt","prob_p2_lose_alt"])
        if has_alt:
            from .plotting import plot_match_probabilities_comparison
            plot_match_probabilities_comparison(df, str(match_id), args.plot_dir)
        else:
            plot_match_probabilities(df, str(match_id), args.plot_dir)
        print(f"[replot] Regenerated plot from {args.csv}")

    elif args.command == "tennis-training":
        # Pre-train on synthetic tennis matches
        print("\n" + "="*80)
        print("STARTING PHASE 1: PRE-TRAINING ON SYNTHETIC MATCHES")
        print("="*80 + "\n")
        
        pretrain_tennis_rules(
            n_matches=args.n_matches,
            epochs=args.epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            output_path=args.model_out,
            device=args.device
        )
        
        print("\n✓ Phase 1 complete! Pre-trained model saved.")
        print(f"  Next step: Run 'complete-model' with --pretrained {args.model_out}\n")
    
    elif args.command == "complete-model":
        # Fine-tune on real data
        print("\n" + "="*80)
        print("STARTING PHASE 2: FINE-TUNING ON REAL MATCHES")
        print("="*80 + "\n")
        
        fine_tune_on_real_data(
            files=args.files,
            pretrained_path=args.pretrained,
            output_path=args.model_out,
            gender=args.gender,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            freeze_layers=args.freeze_layers,
            device=args.device
        )
        
        print("\n✓ Phase 2 complete! Fine-tuned model ready for predictions.")
        print(f"  Use: python tennisctl.py predict --model {args.model_out} ...\n")

    elif args.command == "hyperopt":
        run_hyperopt(args.files, args.n_iter, args.plot_dir, args.model_out, config_path=args.config, search_type=args.search_type)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
