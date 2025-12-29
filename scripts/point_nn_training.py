"""
Training and inference utilities for the point-level neural network.
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import MATCH_COL, load_points_multiple
from .point_nn_features import PointNNFeatureBuilder
from .point_nn_model import build_model
from .point_nn_prior import compute_rule_prior


def _filter_gender(df, gender: str):
    """Filter dataset by gender using match numeric suffix convention."""
    df = df.copy()
    if gender == "both":
        return df
    match_num = df[MATCH_COL].astype(str).str.extract(r"-(\d+)$")[0].astype(int)
    if gender == "male":
        return df[match_num < 2000].copy()
    if gender == "female":
        return df[match_num >= 2000].copy()
    return df


def _split_by_match(match_ids: np.ndarray, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    uniq = np.unique(match_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_val = int(len(uniq) * val_frac)
    n_test = int(len(uniq) * test_frac)
    val_ids = set(uniq[:n_val])
    test_ids = set(uniq[n_val : n_val + n_test])
    train_ids = set(uniq[n_val + n_test :])

    def mask(ids_set):
        return np.array([m in ids_set for m in match_ids])

    return mask(train_ids), mask(val_ids), mask(test_ids)


def _make_loader(X_num, X_cat, y, w, mask, batch_size, shuffle: bool):
    dataset = TensorDataset(
        torch.tensor(X_num[mask], dtype=torch.float32),
        torch.tensor(X_cat[mask], dtype=torch.long),
        torch.tensor(y[mask], dtype=torch.float32),
        torch.tensor(w[mask], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate(model, loader, device):
    model.eval()
    logits_all = []
    y_all = []
    with torch.no_grad():
        for x_num, x_cat, y, _ in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            logits = model(x_num, x_cat)
            logits_all.append(logits.cpu())
            y_all.append(y.cpu())
    if not logits_all:
        return {"logloss": float("nan"), "brier": float("nan"), "acc": float("nan")}
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    probs = torch.sigmoid(logits)
    logloss = F.binary_cross_entropy(probs, y, reduction="mean").item()
    brier = torch.mean((probs - y) ** 2).item()
    acc = ((probs > 0.5) == (y > 0.5)).float().mean().item()
    return {"logloss": logloss, "brier": brier, "acc": acc}


def train_point_nn(
    file_paths: List[str],
    model_out: str,
    gender: str = "male",
    device: str = "cuda",
    epochs: int = 25,
    batch_size: int = 2048,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    long_window: int = 24,
    short_window: int = 8,
    history_points: int = 12,
    seed: int = 42,
    dropout: float = 0.15,
    weight_exp: float = 0.6,
):
    if not file_paths:
        raise ValueError("No input files provided for training")

    df = load_points_multiple(file_paths)
    df = _filter_gender(df, gender)
    df = df.sort_values([MATCH_COL, "SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)

    builder = PointNNFeatureBuilder(
        long_window=long_window,
        short_window=short_window,
        history_points=history_points,
    )
    X_num, X_cat, y, sample_weight, meta = builder.fit_transform(df)
    sample_weight = np.power(sample_weight, weight_exp)
    match_ids = meta[MATCH_COL].astype(str).values

    train_mask, val_mask, test_mask = _split_by_match(match_ids, val_frac=0.1, test_frac=0.1, seed=seed)

    train_loader = _make_loader(X_num, X_cat, y, sample_weight, train_mask, batch_size, shuffle=True)
    val_loader = _make_loader(X_num, X_cat, y, sample_weight, val_mask, batch_size, shuffle=False)
    test_loader = _make_loader(X_num, X_cat, y, sample_weight, test_mask, batch_size, shuffle=False)

    model = build_model(num_numeric=X_num.shape[1], cat_maps=builder.cat_maps, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_num, x_cat, target, weights in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            target = target.to(device)
            weights = weights.to(device)

            logits = model(x_num, x_cat)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(target)

        val_metrics = _evaluate(model, val_loader, device)
        avg_train_loss = running_loss / max(1, len(train_loader.dataset))

        print(
            f"[epoch {epoch:03d}] train_loss={avg_train_loss:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"val_brier={val_metrics['brier']:.4f} "
            f"val_acc={val_metrics['acc']:.3f}"
        )

        if val_metrics["logloss"] < best_val:
            best_val = val_metrics["logloss"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    final_val = _evaluate(model, val_loader, device)
    final_test = _evaluate(model, test_loader, device)
    print(
        f"[final] val_logloss={final_val['logloss']:.4f} val_brier={final_val['brier']:.4f} "
        f"test_logloss={final_test['logloss']:.4f} test_brier={final_test['brier']:.4f}"
    )

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "feature_state": builder.state_dict(),
            "config": {
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "batch_size": batch_size,
                "gender": gender,
                "long_window": long_window,
                "short_window": short_window,
                "history_points": history_points,
                "seed": seed,
            },
        },
        model_out,
    )
    print(f"[train-point-nn] Saved model to {model_out}")
    return model


def pretrain_point_nn(
    n_matches: int,
    model_out: str,
    device: str = "cuda",
    epochs: int = 15,
    batch_size: int = 4096,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    long_window: int = 24,
    short_window: int = 8,
    history_points: int = 12,
    seed: int = 123,
    best_of_5_prob: float = 0.6,
    weight_exp: float = 0.6,
):
    """Pre-train on synthetic data to teach rules before fine-tuning on real CSV."""
    from .point_synthetic_generator import generate_synthetic_matches

    df = generate_synthetic_matches(
        n_matches=n_matches,
        best_of_5_prob=best_of_5_prob,
        seed=seed,
    )
    builder = PointNNFeatureBuilder(
        long_window=long_window,
        short_window=short_window,
        history_points=history_points,
    )
    X_num, X_cat, y, sample_weight, meta = builder.fit_transform(df)
    sample_weight = np.power(sample_weight, weight_exp)
    match_ids = meta[MATCH_COL].astype(str).values

    train_mask, val_mask, test_mask = _split_by_match(match_ids, val_frac=0.1, test_frac=0.1, seed=seed)

    train_loader = _make_loader(X_num, X_cat, y, sample_weight, train_mask, batch_size, shuffle=True)
    val_loader = _make_loader(X_num, X_cat, y, sample_weight, val_mask, batch_size, shuffle=False)
    test_loader = _make_loader(X_num, X_cat, y, sample_weight, test_mask, batch_size, shuffle=False)

    model = build_model(num_numeric=X_num.shape[1], cat_maps=builder.cat_maps, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_num, x_cat, target, weights in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            target = target.to(device)
            weights = weights.to(device)
            logits = model(x_num, x_cat)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = (loss * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(target)

        val_metrics = _evaluate(model, val_loader, device)
        avg_train_loss = running_loss / max(1, len(train_loader.dataset))
        print(
            f"[pretrain epoch {epoch:03d}] train_loss={avg_train_loss:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"val_brier={val_metrics['brier']:.4f} "
            f"val_acc={val_metrics['acc']:.3f}"
        )
        if val_metrics["logloss"] < best_val:
            best_val = val_metrics["logloss"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    final_val = _evaluate(model, val_loader, device)
    final_test = _evaluate(model, test_loader, device)
    print(
        f"[pretrain final] val_logloss={final_val['logloss']:.4f} val_brier={final_val['brier']:.4f} "
        f"test_logloss={final_test['logloss']:.4f} test_brier={final_test['brier']:.4f}"
    )

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "feature_state": builder.state_dict(),
            "config": {
                "dropout": 0.1,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "batch_size": batch_size,
                "long_window": long_window,
                "short_window": short_window,
                "history_points": history_points,
                "seed": seed,
                "best_of_5_prob": best_of_5_prob,
                "stage": "pretrain_synth",
            },
        },
        model_out,
    )
    print(f"[pretrain-point-nn] Saved checkpoint to {model_out}")
    return model


def finetune_point_nn(
    file_paths: List[str],
    pretrained_path: str,
    model_out: str,
    gender: str = "male",
    device: str = "cuda",
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    weight_exp: float = 0.6,
):
    """Fine-tune a pre-trained point NN on real CSV data."""
    model, builder, ckpt = _load_checkpoint(pretrained_path, device=device)

    df = load_points_multiple(file_paths)
    df = _filter_gender(df, gender)
    df = df.sort_values([MATCH_COL, "SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)

    X_num, X_cat, y, sample_weight, meta = builder.transform(df)
    sample_weight = np.power(sample_weight, weight_exp)
    match_ids = meta[MATCH_COL].astype(str).values

    train_mask, val_mask, test_mask = _split_by_match(match_ids, val_frac=0.1, test_frac=0.1, seed=seed)
    train_loader = _make_loader(X_num, X_cat, y, sample_weight, train_mask, batch_size, shuffle=True)
    val_loader = _make_loader(X_num, X_cat, y, sample_weight, val_mask, batch_size, shuffle=False)
    test_loader = _make_loader(X_num, X_cat, y, sample_weight, test_mask, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_num, x_cat, target, weights in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            target = target.to(device)
            weights = weights.to(device)
            logits = model(x_num, x_cat)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = (loss * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(target)

        val_metrics = _evaluate(model, val_loader, device)
        avg_train_loss = running_loss / max(1, len(train_loader.dataset))
        print(
            f"[finetune epoch {epoch:03d}] train_loss={avg_train_loss:.4f} "
            f"val_logloss={val_metrics['logloss']:.4f} "
            f"val_brier={val_metrics['brier']:.4f} "
            f"val_acc={val_metrics['acc']:.3f}"
        )
        if val_metrics["logloss"] < best_val:
            best_val = val_metrics["logloss"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    final_val = _evaluate(model, val_loader, device)
    final_test = _evaluate(model, test_loader, device)
    print(
        f"[finetune final] val_logloss={final_val['logloss']:.4f} val_brier={final_val['brier']:.4f} "
        f"test_logloss={final_test['logloss']:.4f} test_brier={final_test['brier']:.4f}"
    )

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "feature_state": builder.state_dict(),
            "config": {
                **ckpt.get("config", {}),
                "finetune_epochs": epochs,
                "finetune_lr": lr,
                "finetune_weight_decay": weight_decay,
                "finetune_seed": seed,
                "stage": "finetune_real",
                "gender": gender,
            },
        },
        model_out,
    )
    print(f"[finetune-point-nn] Saved fine-tuned model to {model_out}")
    return model


def _load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    builder = PointNNFeatureBuilder.load_state_dict(ckpt["feature_state"])
    model = build_model(
        num_numeric=len(builder.numeric_features),
        cat_maps=builder.cat_maps,
        dropout=ckpt.get("config", {}).get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, builder, ckpt


def predict_point_nn(
    file_paths: List[str],
    model_path: str,
    output_path: str | None = None,
    gender: str = "male",
    device: str = "cpu",
    batch_size: int = 4096,
    plot_dir: str | None = None,
    match_id: str | None = None,
    temperature: float = 1.0,
    smooth_window: int = 5,
    rule_blend: float = 0.35,
):
    if not file_paths:
        raise ValueError("No input files provided for prediction")

    model, builder, ckpt = _load_checkpoint(model_path, device=device)
    df = load_points_multiple(file_paths)
    df = _filter_gender(df, gender)
    df = df.sort_values([MATCH_COL, "SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)

    X_num, X_cat, y, _, meta = builder.transform(df)
    dataset = TensorDataset(
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    preds = []
    targets = []
    with torch.no_grad():
        for x_num, x_cat, target in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            logits = model(x_num, x_cat)
            if temperature and temperature > 0:
                logits = logits / float(temperature)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
            targets.append(target.numpy())

    prob_raw_model = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Compute rule prior on pre-point features and blend
    features_df = builder._prepare_features(df)  # uses pre-point state
    rule_prior = compute_rule_prior(features_df)
    prob_raw = (1.0 - rule_blend) * prob_raw_model + rule_blend * rule_prior

    # Hard clamps in critical states to enforce tennis logic
    if "match_point_p1" in features_df.columns:
        mp1 = features_df["match_point_p1"].to_numpy() > 0
        mp2 = features_df["match_point_p2"].to_numpy() > 0
        prob_raw[mp1] = np.maximum(prob_raw[mp1], 0.97)
        prob_raw[mp2] = np.minimum(prob_raw[mp2], 0.03)
    if "set_point_p1" in features_df.columns:
        sp1 = features_df["set_point_p1"].to_numpy() > 0
        sp2 = features_df["set_point_p2"].to_numpy() > 0
        prob_raw[sp1] = np.maximum(prob_raw[sp1], 0.9)
        prob_raw[sp2] = np.minimum(prob_raw[sp2], 0.1)

    # Optional smoothing to reduce point-to-point oscillations
    prob_smooth = prob_raw
    if smooth_window and smooth_window > 1:
        prob_smooth = (
            pd.Series(prob_raw)
            .groupby(df[MATCH_COL].astype(str))
            .transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())
            .to_numpy()
        )

    # Align output to the filtered feature dataframe (drop rows with missing server/winner)
    out_df = features_df.copy()
    out_df["prob_p1_point_model_raw"] = prob_raw_model
    out_df["prob_p1_point_prior"] = rule_prior
    out_df["prob_p1_point_raw"] = prob_raw
    out_df["prob_p1_point"] = prob_smooth
    out_df["target_p1_point"] = targets

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out_df.to_csv(output_path, index=False)
        print(f"[predict-point-nn] Saved predictions to {output_path}")

    if plot_dir and match_id is not None:
        from .point_nn_plotting import plot_point_probabilities

        plot_point_probabilities(out_df, str(match_id), builder, plot_dir)

    return out_df, ckpt
