#!/usr/bin/env python3
import argparse

from .model import train_model
from .model_point import train_point_model
from .prediction import run_prediction
from .hyperopt import run_hyperopt
from .plotting import plot_match_probabilities
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
        train_model(args.files, args.model_out, config_path=args.config)
    
    elif args.command == "train-point":
        train_point_model(args.files, args.model_out, config_path=args.config)

    elif args.command == "predict":
        mode = getattr(args, 'mode', 'importance')
        run_prediction(args.files, args.model, args.match_id, args.plot_dir, 
                      config_path=args.config, counterfactual_mode=mode)

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

    elif args.command == "hyperopt":
        run_hyperopt(args.files, args.n_iter, args.plot_dir, args.model_out, config_path=args.config, search_type=args.search_type)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
