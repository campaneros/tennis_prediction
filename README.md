#  Tennis Counterfactual Analysis — Usage Guide

This package provides a command-line tool `tennisctl` to train a model, run hyperparameter optimisation, and generate match probability trajectories with counterfactual scenarios.

Below are **all the commands you need to run the code**.

---

## Model Features

The model uses **18 features** to predict match outcomes:

### Feature Importance Ranking
1. **SetsWonDiff (21.8%)** - Set difference scaled by match progress (weight: 0.25)
2. **SetNo (17.1%)** - Set number normalized (1-5)
3. **Game_Diff (11.1%)** - Game difference in current set
4. **is_decider_tied (9.0%)** - Binary flag for tied decisive set (set 4+, sets 2-2)
5. **GameNo (4.8%)** - Game number within set
6. **PointNumber (4.8%)** - Point number within match
7. **CurrentSetGamesDiff (4.5%)** - In-set game difference amplified (×2.5)
8. **Momentum_Diff (3.7%)** - Momentum difference normalized per set (z-score)
9. **Score_Diff (3.2%)** - Point score difference in current game
10. **momentum (3.0%)** - Exponential weighted moving average (alpha=0.15)
11. **SrvScr (2.9%)** - Cumulative points won when P1 served in game
12. **RcvScr (2.9%)** - Cumulative points won when P1 received in game
13. **Server (2.7%)** - Binary: 1 if P1 serves, 0 if P2 serves
14. **point_importance (2.1%)** - Critical point indicator (1.0-7.0)
15. **P_srv_win_long (2.0%)** - P1 serve win rate (20-point window)
16. **P_srv_lose_long (1.9%)** - P1 return game win rate (20-point window)
17. **P_srv_lose_short (1.3%)** - P1 return win rate (5-point window)
18. **P_srv_win_short (1.2%)** - P1 serve win rate (5-point window)

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
- **momentum_alpha**: 0.15 (faster decay, more reactive)
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






