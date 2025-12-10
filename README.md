#  Tennis Counterfactual Analysis — Usage Guide

This package provides a command-line tool `tennisctl` to train a model, run hyperparameter optimisation, and generate match probability trajectories with counterfactual scenarios.

Below are **all the commands you need to run the code**.

---

## Model Features (31 total)

Serve/return form
- `P_srv_win_long` / `P_srv_lose_long`: long-window share of P1 serve points won/lost (stabilised form).
- `P_srv_win_short` / `P_srv_lose_short`: short-window serve form for quick momentum shifts.
- `PointServer`: who serves this point (1 = P1, 2 = P2).

Score, momentum, and game state
- `momentum`: EWMA of leverage, higher when recent points tilt to P1.
- `Momentum_Diff`: raw P1Momentum − P2Momentum from feed.
- `Score_Diff`: normalized point-score differential (clipped to [-2, 2]).
- `Game_Diff`: normalized games differential in match (clipped to [-3, 3]).
- `CurrentSetGamesDiff`: games differential inside the current set.
- `SrvScr`: cumulative points P1 has won on serve within the current game.
- `RcvScr`: cumulative points P1 has won while returning in the current game.
- `SetNo`, `GameNo`, `PointNumber`: within-match/set progress indicators normalized to [0, 1].
- `point_importance`: heuristic criticality weight (1.0 regular → 7.0 match point).

Set progression and priors
- `P1SetsWon`, `P2SetsWon`: sets captured so far.
- `SetsWonDiff`: non-linear set differential scaled by match progress.
- `SetsWonAdvantage`: strong indicator (+4/-4) for having a set lead or deficit.
- `SetWinProbPrior`: calibrated prior P(P1 wins match) from set/game state (bounded 0.05–0.95).
- `SetWinProbEdge`: centered prior (prior − 0.5) * 2 for symmetric modeling.
- `SetWinProbLogit`: log-odds of the prior for smoother tree splits.

Decider and finish signals
- `is_decider_tied`: 1 when sets are level in the final set (true coin-flip spots).
- `DistanceToMatchEnd`: non-linear proximity to finishing (rises sharply late).
- `MatchFinished`: 1 on the last point of a match, else 0.

Tiebreak signals
- `is_tiebreak`: 1 when in a tiebreak (6-6+ games).
- `is_decisive_tiebreak`: 1 for final-set tiebreaks.
- `tiebreak_score_diff`: normalized tiebreak point differential in [-1, 1].
- `tiebreak_win_proximity`: exponential closeness to sealing the tiebreak (P1 minus P2).
- `is_tiebreak_late_stage`: 1 when the tiebreak has reached 10+ total points.

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



