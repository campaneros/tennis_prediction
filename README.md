#  Tennis Counterfactual Analysis — Usage Guide

This package provides a command-line tool `tennisctl` to train a model, run hyperparameter optimisation, and generate match probability trajectories with counterfactual scenarios.

Below are **all the commands you need to run the code**.

---

## Model Features

The model uses **22 features** to predict match outcomes:

### Feature List & Descriptions

1. **P_srv_win_long** - P1 serve win probability (20-point rolling window, Bayesian smoothing)
2. **P_srv_lose_long** - P1 return game win probability (20-point rolling window)
3. **P_srv_win_short** - P1 serve win probability (5-point real-time window)
4. **P_srv_lose_short** - P1 return win probability (5-point real-time window)
5. **PointServer** - Binary: 1 if P1 serves, 2 if P2 serves
6. **momentum** - Exponential weighted moving average of leverage (alpha=0.35, weighted by point importance)
7. **Momentum_Diff** - P1 momentum minus P2 momentum (rolling z-score, window=50)
8. **Score_Diff** - Point score difference in current game (P1 - P2)
9. **Game_Diff** - Game difference in current set (P1 - P2)
10. **CurrentSetGamesDiff** - In-set game difference amplified ×1.5 for current set performance
11. **SrvScr** - Cumulative points won when P1 served in current game
12. **RcvScr** - Cumulative points won when P1 received in current game
13. **SetNo** - Current set number (1-5)
14. **GameNo** - Game number within current set
15. **PointNumber** - Point number within entire match
16. **point_importance** - Critical point indicator (1.0=normal, 7.0=match point)
17. **SetsWonDiff** - Set difference scaled by match progress with non-linear weighting
18. **SetsWonAdvantage** - Binary set advantage: +4.0 if P1 leads, -4.0 if P1 behind, 0.0 if tied
19. **SetWinProbPrior** - Calibrated prior for P1 match win based on set state (0.05-0.95)
20. **SetWinProbEdge** - Centered prior in [-1,1] to stabilize training
21. **SetWinProbLogit** - Log-odds of the calibrated prior
22. **is_decider_tied** - 1.0 when sets are tied in decisive set (2-2 in set 5), 0.0 otherwise

### Key Feature Engineering Strategies

#### Set Advantage Dominance (SetsWonAdvantage)
- **Weight**: ±4.0 (strongest signal when sets differ)
- **Purpose**: Force probabilities toward 0.5 when sets are tied, override historical bias
- When sets tied: probabilities reflect current game situation, not cumulative stats

#### Adaptive Dampening When Sets Tied
When `SetsWonDiff_raw == 0`:
- **Serve/return stats dampening**: 50% pull toward 0.5 for normal points, 20% for critical points (importance > 3.0)
- **Leverage reduction**: 35% for normal points, 70% for critical points
- **Momentum suppression**: 2% in tied decisive set to remove historical bias

This ensures:
- Tied sets + normal points → probabilities ≈ 0.5
- Tied sets + critical situations (40-0, break point) → current game dominance matters

#### Point Importance Weighting
Break points, set points, and match points get amplified impact:
```python
importance = 1.0  # baseline
importance += 2.0 if break_point
importance += 2.0 if set_point  
importance += 2.0 if match_point
importance *= 1.5 if deuce/advantage
```

### Sample Weighting
Points are weighted during training based on:
```python
weight = point_importance^0.5 × competitive_multiplier × tied_boost
```
- **point_importance**: 1.0 (regular) to 7.0 (match point)
- **competitive_multiplier**: 5-set match ×4.0, 4-set ×2.5, 3-set ×1.0
- **tied_boost**: ×2.0 when sets are tied in decisive set

### Training Configuration
- **long_window**: 20 points
- **short_window**: 5 points
- **momentum_alpha**: 0.35 (moderate decay, balances history and recent form)
- **sample_weight_exponent**: 0.5

**Note**: Match 1701 is excluded from training to prevent test set leakage. Women's matches (best-of-3) are also excluded to focus on men's Grand Slam format (best-of-5).

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






