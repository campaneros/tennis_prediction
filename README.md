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

## How the Model Learns Match Win Probability

### Training Approach: Point-by-Point Classification

The model uses **XGBoost Classifier** trained on every point of historical matches with a binary target:

```python
target = 1 if P1 wins the match, 0 if P2 wins
```

**Key Insight**: Every point in a match gets labeled with the final outcome, creating a massive dataset where the model learns patterns that correlate with eventual victory.

### Training Data Structure

From ~800k points across men's Grand Slam matches (2011-2018):
- Each row = one point in a match
- Features = game state at that exact moment (score, momentum, set count, etc.)
- Label = who ultimately won the match

Example:
```
Point 1 of match → P1 eventually wins → label = 1
Point 2 of match → P1 eventually wins → label = 1
...
Point 423 of match → P1 eventually wins → label = 1
```

### What the Model Learns

Through gradient boosting, XGBoost discovers patterns like:

1. **Set advantage is crucial**:
   - If `SetsWonAdvantage = +4.0` → Strong signal P1 wins
   - If `SetsWonAdvantage = 0.0` → Balanced, other features matter more

2. **Critical moments matter**:
   - Break points, set points carry higher weight (via sample weighting)
   - Model learns these are predictive of final outcome

3. **Momentum and form**:
   - Recent performance (short windows) vs historical (long windows)
   - Current set dominance vs match-wide statistics

4. **Match progression**:
   - Later sets (SetNo=4,5) in tied situations get special treatment
   - Model learns probability trajectories change differently early vs late

### Probability Calibration

The model outputs **P(P1 wins match | current game state)** via:

```python
model.predict_proba(features)[:, 1]  # Probability for class 1 (P1 wins)
```

XGBoost's internal calibration comes from:
- **Logistic loss function**: Naturally outputs probabilities
- **Stratified cross-validation**: Prevents overfitting to imbalanced scenarios
- **Sample weights**: Upweights critical/competitive points

### Why This Works

The approach leverages **temporal structure**:
- Early match points with tied sets → probability ≈ 0.5
- Mid-match after one player wins 2 sets → probability shifts toward ~0.65-0.70
- Late match at match point → probability ≈ 0.95+

The model learns these probability dynamics from observing hundreds of thousands of real match trajectories.

### Handling Bias

To prevent the model from over-relying on cumulative statistics when sets are tied:

1. **Feature dampening**: Serve/return stats pulled toward 0.5 when sets tied
2. **SetsWonAdvantage dominance**: Large weight (±4.0) forces reset when tied
3. **Competitive match upweighting**: 5-set matches get 4× weight to learn balanced scenarios
4. **Tied decisive set boost**: 2× weight when sets are 2-2 in set 5

This ensures probabilities stay realistic (near 0.5 when truly balanced) while still capturing who's dominating the current game/set.

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

### Understanding the Probability Curves

The prediction generates four probability curves showing current state and counterfactual scenarios:

1. **P1 wins match (current)** [Blue solid line]
   - Probability that Player 1 wins the match given the current game state
   - P1 and P2 are fixed identities throughout the entire match

2. **P2 wins match (current)** [Orange solid line]
   - Probability that Player 2 wins = 1 - P(P1 wins)
   - Always sums to 1.0 with the P1 probability

3. **P1 wins | if P1 loses current point** [Green dashed line]
   - **Counterfactual scenario**: What if P1 loses this specific point?
   - Shows how P1's win probability would change if P1 fails to win this point
   - Applies whether P1 is serving (would lose serve point) or receiving (would fail to win return point)

4. **P2 wins | if P2 loses current point** [Red dashed line]
   - **Counterfactual scenario**: What if P2 loses this specific point?
   - Shows how P2's win probability would change if P2 fails to win this point
   - Applies whether P2 is serving (would lose serve point) or receiving (would fail to win return point)

**Example: P2 has match point while serving (point index ~400)**
- **Current state**: P2 probability is very high (~0.95) [orange solid line]
- **If P2 loses this point**: P2 probability drops dramatically [red dashed line drops sharply]
- **Large gap** between solid orange and dashed red = extremely critical point for P2
- This visualizes how losing a match point devastates P2's chances

**Example: P1 is serving at 40-0 (game point)**
- **Current state**: P1 probability relatively high
- **If P1 loses this point**: P1 probability drops slightly [green dashed line drops a bit]
- **Small gap** = point is important but not catastrophic (P1 would still have 40-15)

**Key Insight - Point Importance Visualization**: 
- **Large gap between solid and dashed lines** → Critical point for that player
- **Small gap** → Less critical point
- The counterfactuals show **player-specific risk**, not just server risk
- At match point for P2: red dashed line plummets, green dashed line soars (P1 benefits from P2's loss)






