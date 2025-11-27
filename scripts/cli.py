#!/usr/bin/env python3
import argparse

from .model import train_model
from .prediction import run_prediction
from .hyperopt import run_hyperopt


def main():
    parser = argparse.ArgumentParser(
        description="Tennis Counterfactual CLI Tool"
    )
    subparsers = parser.add_subparsers(dest="command")

# TRAIN
    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("--files", nargs="+", required=True,
                         help="List of CSV point-by-point files")
    train_p.add_argument("--model-out", required=True,
                         help="Path to save trained model (JSON)")
    train_p.add_argument("--config", default=None,
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

    elif args.command == "predict":
        run_prediction(args.files, args.model, args.match_id, args.plot_dir, config_path=args.config)

    elif args.command == "hyperopt":
        run_hyperopt(args.files, args.n_iter, args.plot_dir, args.model_out, config_path=args.config, search_type=args.search_type)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
