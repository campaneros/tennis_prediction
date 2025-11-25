#  Tennis Counterfactual Analysis — Usage Guide

This package provides a command-line tool `tennisctl` to train a model, run hyperparameter optimisation, and generate match probability trajectories with counterfactual scenarios.

Below are **all the commands you need to run the code**.

---

# 1. Setup

``
git clone --recursive https://github.com/campaneros/tennis_prediction.git
cd tennis_prediction
```
## Create and activate a virtual environment

```
make venv
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
  --model-out models/xgb_baseline.json
```

# 3.  Hyperparameter optimisation (5-fold CV)
```
tennisctl hyperopt \
  --files data/2021-wimbledon-points.csv \
  --n-iter 30 \
  --plot-dir hyperopt_plots \
  --model-out models/xgb_tuned.json
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






