#  Tennis Counterfactual Analysis — Usage Guide

This package provides a command-line tool `tennisctl` to train a model, run hyperparameter optimisation, and generate match probability trajectories with counterfactual scenarios.

Below are **all the commands you need to run the code**.

---

## Model Features (26 total)

Serve/return and state:
- P_srv_win_long, P_srv_lose_long, P_srv_win_short, P_srv_lose_short
- PointServer, momentum, Momentum_Diff, Score_Diff, Game_Diff, CurrentSetGamesDiff
- SrvScr, RcvScr, SetNo, GameNo, PointNumber, point_importance

Set/decider context:
- P1SetsWon, P2SetsWon, SetsWonDiff, SetsWonAdvantage
- SetWinProbPrior, SetWinProbEdge, SetWinProbLogit
- is_decider_tied, DistanceToMatchEnd, MatchFinished

### Training target and constraints
- Soft labels blend the hard match outcome with a set-aware prior so the model learns: ~0.5 when sets are tied (slight drift at 2-2), capped ~0.70 when up one set, ~0.90 when up two sets.
- Model: `XGBRegressor` with `reg:logistic` and monotone constraints (+ for P1 set edge, – for P2 set edge, + for set priors) so tree splits respect tennis logic.

### Sample Weighting
Points are weighted during training based on:
```python
weight = point_importance^0.5 × competitive_multiplier × tied_boost
```
- **point_importance**: 1.0 (regular) to 7.0 (match point)
- **competitive_multiplier**: 5-set match ×4, 4-set ×2.5, 3-set ×1.0
- **tied_boost**: ×2 when sets are tied in decisive set

### Training Configuration
- **long_window**: 20 points
- **short_window**: 5 points
- **momentum_alpha**: 0.35 (faster decay, more reactive)
- **sample_weight_exponent**: 0.5

**Note**: Match 1701 is excluded from training to prevent test set leakage.

---

# 1. Setup

```
git clone --recursive https://github.com/campaneros/tennis_prediction.git
cd tennis_prediction
```
## Create and activate a virtual environment

```
make venv
source venv/bin/activate
make install
```

Test that the CLI works:
```
tennisctl --help
```

then 

```
pytest -v
```

# 2. Train a baseline model
```
tennisctl train \
  --files data/2021-wimbledon-points.csv \
  --model-out models/xgb_baseline.json \
  --config configs/config.json
```

# 3.  Hyperparameter optimisation (5-fold CV)
```
tennisctl hyperopt \
  --files data/2021-wimbledon-points.csv \
  --n-iter 30 \
  --plot-dir hyperopt_plots \
  --model-out models/xgb_tuned.json
  --config configs/config.json
```


This will:
	•	run RandomizedSearchCV with 5-fold CV
	•	compute Accuracy, Precision, Recall, F1, and AUC
	•	save the best model to models/xgb_tuned.json
	•	write diagnostic plots to hyperopt_plots/
	•	produce confusion matrix + ROC curve (best_model_cv_confusion_roc.png
	•	save a complete metrics table (hyperopt_cv_metrics.csv)

# 4. Predict match probabilities

```
tennisctl predict \
  --files data/2021-wimbledon-points.csv \
  --model models/xgb_tuned.json \
  --match-id <MATCH_ID> \
  --plot-dir plots
```





