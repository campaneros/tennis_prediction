"""
CPU-only baseline trainers for point-by-point match win probability.

Creates two models without touching existing code:
  - BDT: HistGradientBoostingRegressor (probabilistic, robust).
  - MLP: shallow PyTorch network (CPU).

CLI example:
python -m scripts.point_predictors --files data/train.csv --model-out models/bdt_point.pkl --model-type bdt
python -m scripts.point_predictors --files data/train.csv --model-out models/mlp_point.pt --model-type mlp
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:  # Optional; only needed for --model-type mlp
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # noqa: BLE001
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from .data_loader import MATCH_COL, SERVER_COL, POINT_WINNER_COL, load_points_multiple
from .features import (
    MATCH_FEATURE_COLUMNS,
    add_additional_features,
    add_leverage_and_momentum,
    add_match_labels,
    add_rolling_serve_return_features,
    build_dataset,
)


# Extra columns dedicated to pressure/phase awareness.
EXTRA_FEATURE_COLUMNS = [
    "points_in_game_frac",
    "points_in_set_frac",
    "points_in_match_frac",
    "server_point_run",
    "receiver_point_run",
    "server_recent_win_rate_5",
    "receiver_recent_win_rate_5",
    "game_pressure",
    "set_pressure",
    "match_point_proximity_p1",  # How close P1 is to match point (0-1)
    "match_point_proximity_p2",  # How close P2 is to match point (0-1)
    "set_point_situation",  # Critical set point indicator (0-1)
]


def _score_to_numeric(score) -> int:
    """Map tennis score to monotonic numeric scale; tolerant to noise."""
    mapping = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}
    if pd.isna(score):
        return 0
    s = str(score).strip().upper()
    return mapping.get(s, pd.to_numeric(score, errors="coerce") if str(score).isdigit() else 0)


def _running_streak(series: pd.Series) -> pd.Series:
    """Return running streak lengths for True values."""
    out = np.zeros(len(series), dtype=int)
    streak = 0
    for i, val in enumerate(series.astype(bool).tolist()):
        streak = streak + 1 if val else 0
        out[i] = streak
    return pd.Series(out, index=series.index, dtype=float)


def add_extra_point_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject lightweight pressure/phase features without touching core pipeline."""
    df = df.copy()

    # Ordered counters per granularity
    order_match = df.groupby(MATCH_COL).cumcount()
    order_set = df.groupby([MATCH_COL, "SetNo"]).cumcount()
    order_game = df.groupby([MATCH_COL, "SetNo", "GameNo"]).cumcount()

    # Group keys as strings to avoid expensive tuple conversions
    set_key = df[MATCH_COL].astype(str) + "_s" + df["SetNo"].astype(str)
    game_key = df[MATCH_COL].astype(str) + "_s" + df["SetNo"].astype(str) + "_g" + df["GameNo"].astype(str)

    df["points_in_match_frac"] = order_match / order_match.groupby(df[MATCH_COL]).transform("max").replace(0, 1)
    df["points_in_set_frac"] = order_set / order_set.groupby(set_key).transform("max").replace(0, 1)
    df["points_in_game_frac"] = order_game / order_game.groupby(game_key).transform("max").replace(0, 1)

    # Server/receiver streaks and short-form
    server_wins = (df[SERVER_COL] == df[POINT_WINNER_COL]).astype(int)
    receiver_wins = 1 - server_wins

    df["server_point_run"] = server_wins.groupby(df[MATCH_COL]).transform(_running_streak)
    df["receiver_point_run"] = receiver_wins.groupby(df[MATCH_COL]).transform(_running_streak)

    df["server_recent_win_rate_5"] = (
        server_wins.groupby(df[MATCH_COL]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df["receiver_recent_win_rate_5"] = (
        receiver_wins.groupby(df[MATCH_COL]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Pressure proxies
    p1_score = df.get("P1Score", 0).apply(_score_to_numeric)
    p2_score = df.get("P2Score", 0).apply(_score_to_numeric)

    # How close the current game is to being over (0-1)
    points_needed_p1 = (4 - p1_score).clip(lower=0)
    points_needed_p2 = (4 - p2_score).clip(lower=0)
    close_to_game_end = 1.0 - (np.minimum(points_needed_p1, points_needed_p2) / 4.0)
    df["game_pressure"] = close_to_game_end.clip(0.0, 1.0)

    # Set pressure: fewer games needed -> higher pressure
    p1_games = pd.to_numeric(df.get("P1GamesWon", 0), errors="coerce").fillna(0)
    p2_games = pd.to_numeric(df.get("P2GamesWon", 0), errors="coerce").fillna(0)
    games_to_win_set_p1 = (6 - p1_games).clip(lower=0)
    games_to_win_set_p2 = (6 - p2_games).clip(lower=0)
    df["set_pressure"] = 1.0 - (np.minimum(games_to_win_set_p1, games_to_win_set_p2) / 6.0)

    # Match point proximity: quanto ogni giocatore è vicino al match point
    # Usa P1SetsWon e P2SetsWon create da add_additional_features
    p1_sets = pd.to_numeric(df.get("P1SetsWon", 0), errors="coerce").fillna(0)
    p2_sets = pd.to_numeric(df.get("P2SetsWon", 0), errors="coerce").fillna(0)
    
    # Determina sets_to_win basato su match_id (maschi=best of 5, femmine=best of 3)
    import re
    match_ids = df[MATCH_COL].astype(str)
    is_male = match_ids.apply(lambda mid: 1000 <= int(re.search(r'-(\d+)$', mid).group(1)) < 2000 if re.search(r'-(\d+)$', mid) else False)
    sets_to_win = np.where(is_male, 3, 2)
    
    # Proximity to match point: quanto manca per avere match point
    # 1.0 = ha già abbastanza set, serve solo chiudere questo set
    # 0.0 = molto lontano dal match point
    p1_sets_needed = sets_to_win - p1_sets
    p2_sets_needed = sets_to_win - p2_sets
    
    df["match_point_proximity_p1"] = np.clip(1.0 - p1_sets_needed / sets_to_win, 0.0, 1.0)
    df["match_point_proximity_p2"] = np.clip(1.0 - p2_sets_needed / sets_to_win, 0.0, 1.0)
    
    # Set point situation: siamo in una situazione di set point?
    # Alta quando un giocatore può chiudere il set in questo game
    can_close_set_p1 = ((p1_games == 5) & (p2_games <= 4)) | ((p1_games == 6) & (p2_games == 5)) | ((p1_games >= 6) & (p2_games >= 6))
    can_close_set_p2 = ((p2_games == 5) & (p1_games <= 4)) | ((p2_games == 6) & (p1_games == 5)) | ((p2_games >= 6) & (p1_games >= 6))
    
    set_point_weight = np.where(can_close_set_p1 | can_close_set_p2, 1.0, 0.0)
    # Amplifica se è anche vicino al match point
    set_point_weight *= np.maximum(df["match_point_proximity_p1"], df["match_point_proximity_p2"])
    df["set_point_situation"] = set_point_weight

    for col in EXTRA_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


@dataclass
class DatasetBundle:
    X: np.ndarray
    y_soft: np.ndarray
    weights: np.ndarray
    y_hard: np.ndarray


def build_enhanced_dataset(df: pd.DataFrame) -> DatasetBundle:
    """Reuse base builder, then append extra columns for richer training."""
    X_base, y_soft, mask, weights, y_hard = build_dataset(df)
    extras = df.loc[mask, EXTRA_FEATURE_COLUMNS].to_numpy(dtype=float)
    X = np.hstack([X_base, extras])
    return DatasetBundle(X=X, y_soft=y_soft, weights=weights, y_hard=y_hard.astype(int))


def prepare_dataframe(paths: Sequence[str]) -> pd.DataFrame:
    """Full feature pipeline with new extras."""
    df = load_points_multiple(paths)
    df = df.dropna(subset=[MATCH_COL, SERVER_COL, POINT_WINNER_COL])
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=20, short_window=5)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=0.35)
    df = add_extra_point_features(df)
    return df


def train_bdt(bundle: DatasetBundle, random_state: int = 42):
    """Train CPU-only boosted trees."""
    idx = np.arange(len(bundle.X))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=random_state, stratify=bundle.y_hard)

    X_train, X_val = bundle.X[train_idx], bundle.X[val_idx]
    y_train_soft, y_val_soft = bundle.y_soft[train_idx], bundle.y_soft[val_idx]
    y_train_hard, y_val_hard = bundle.y_hard[train_idx], bundle.y_hard[val_idx]
    w_train, w_val = bundle.weights[train_idx], bundle.weights[val_idx]

    # Use classifier to get calibrated probabilities with supported loss.
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train_hard, sample_weight=w_train)

    val_pred = model.predict_proba(X_val)[:, 1].clip(0.0, 1.0)
    metrics = {
        "logloss": float(log_loss(y_val_hard, val_pred, sample_weight=w_val)),
        "brier": float(brier_score_loss(y_val_hard, val_pred, sample_weight=w_val)),
        "roc_auc": float(roc_auc_score(y_val_hard, val_pred)),
    }
    return model, metrics


class MLP(nn.Module):  # type: ignore[misc]
    def __init__(self, dim_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(bundle: DatasetBundle, random_state: int = 42, device: str = "cpu"):
    if torch is None:
        raise ImportError("PyTorch non installato: usa --model-type bdt o installa torch per la MLP.")

    torch.manual_seed(random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(bundle.X)

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_scaled, bundle.y_soft, bundle.weights, test_size=0.15, random_state=random_state, stratify=bundle.y_hard
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(w_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    model = MLP(dim_in=bundle.X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    def focal_bce(logits, targets, weights, gamma=1.5):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        mod = torch.pow(1.0 - prob, gamma)
        return (bce * mod * weights).mean()

    best_loss = float("inf")
    patience, bad = 8, 0
    for epoch in range(60):
        model.train()
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = focal_bce(logits, yb, wb)
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            preds = []
            ys = []
            ws = []
            for xb, yb, wb in val_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                logits = model(xb)
                val_losses.append(focal_bce(logits, yb, wb).item())
                preds.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(yb.cpu().numpy())
                ws.append(wb.cpu().numpy())
            val_loss = float(np.mean(val_losses))
            preds = np.concatenate(preds)
            ys = np.concatenate(ys)
            ws = np.concatenate(ws)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            bad = 0
            best_state = {
                "model": model.state_dict(),
                "scaler": scaler,
                "metrics": {
                    "logloss": float(log_loss(ys, preds, sample_weight=ws)),
                    "brier": float(brier_score_loss(ys, preds, sample_weight=ws)),
                    "roc_auc": float(roc_auc_score((ys >= 0.5).astype(int), preds)),
                },
            }
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state["model"])
    return model, scaler, best_state["metrics"]


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description="Train CPU BDT/MLP for point-wise match win probability.")
    p.add_argument("--files", nargs="+", required=True, help="Input CSV files with point-level data.")
    p.add_argument("--model-out", required=True, help="Output path for the trained model.")
    p.add_argument("--model-type", choices=["bdt", "mlp"], default="bdt", help="Model to train.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    df = prepare_dataframe(args.files)
    bundle = build_enhanced_dataset(df)

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    if args.model_type == "bdt":
        model, metrics = train_bdt(bundle, random_state=args.seed)
        joblib.dump({"model": model, "features": MATCH_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS}, args.model_out)
        print(f"[bdt] saved to {args.model_out}")
    else:
        model, scaler, metrics = train_mlp(bundle, random_state=args.seed)
        torch.save(
            {"state_dict": model.state_dict(), "scaler": scaler, "features": MATCH_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS},
            args.model_out,
        )
        print(f"[mlp] saved to {args.model_out}")

    print("[metrics]", metrics)


if __name__ == "__main__":
    main()
