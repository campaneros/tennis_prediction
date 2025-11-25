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

    # TRAIN -------------------------------------------------------------
    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("--files", nargs="+", required=True,
                         help="List of CSV point-by-point files")
    train_p.add_argument("--model-out", required=True,
                         help="Path to save trained model (JSON)")

    # PREDICT -----------------------------------------------------------
    pred_p = subparsers.add_parser("predict", help="Predict probabilities and plot")
    pred_p.add_argument("--files", nargs="+", required=True,
                        help="List of CSV point-by-point files")
    pred_p.add_argument("--model", required=True,
                        help="Path to trained model JSON")
    pred_p.add_argument("--match-id", required=True,
                        help="match_id to plot (string or numeric)")
    pred_p.add_argument("--plot-dir", default="plots",
                        help="Directory to store plots")

    # HYPEROPT ----------------------------------------------------------
    hyp_p = subparsers.add_parser("hyperopt", help="Hyperparameter optimisation")
    hyp_p.add_argument("--files", nargs="+", required=True,
                       help="List of CSV point-by-point files")
    hyp_p.add_argument("--n-iter", type=int, default=30,
                       help="Number of RandomizedSearchCV iterations")
    hyp_p.add_argument("--plot-dir", default="hyperopt_plots",
                       help="Directory to store hyperopt plots")
    hyp_p.add_argument("--model-out", required=True,
                       help="Path to save tuned model JSON")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.files, args.model_out)

    elif args.command == "predict":
        run_prediction(args.files, args.model, args.match_id, args.plot_dir)

    elif args.command == "hyperopt":
        run_hyperopt(args.files, args.n_iter, args.plot_dir, args.model_out)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
