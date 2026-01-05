"""
Predizione point-by-point della probabilità di vittoria match per un singolo match_id,
con plot di P1/P2. Non tocca i file esistenti; usa i modelli salvati da
run_point_predictors(_gender). Funziona solo su CPU.

Esempio:
python run_point_predict_match.py --files data/2019-wimbledon-points.csv --model models/bdt_male.pkl --match-id 2019-wimbledon-1234 --gender male --out-png plots/pred_1234.png --out-csv predictions_1234.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402


def _load_point_module():
    """Carica scripts.point_predictors evitando scripts/__init__.py."""
    repo_root = Path(__file__).resolve().parent
    pkg_path = repo_root / "scripts"

    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = [str(pkg_path)]
    sys.modules.setdefault("scripts", scripts_pkg)

    target_path = pkg_path / "point_predictors.py"
    spec = importlib.util.spec_from_file_location("scripts.point_predictors", target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossibile caricare {target_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.point_predictors"] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description="Predici probabilità match point-by-point per un match_id e genera plot.")
    p.add_argument("--files", nargs="+", required=True, help="CSV dei punti.")
    p.add_argument("--model", required=True, help="Modello addestrato (.pkl BDT o .pt MLP).")
    p.add_argument("--match-id", required=True, help="match_id da predire (es. 2019-wimbledon-1234).")
    p.add_argument("--gender", choices=["male", "female", "both"], default="both", help="Filtro match_id come nel training.")
    p.add_argument("--out-png", default="plots/prediction.png", help="Path immagine plot (png).")
    p.add_argument("--out-html", default=None, help="Opzionale: salva plot interattivo (zoom) in HTML.")
    p.add_argument("--out-csv", default=None, help="Opzionale: salva CSV con probabilità.")
    return p.parse_args(argv)


def _filter_by_gender(df: pd.DataFrame, gender: str) -> pd.DataFrame:
    """Applica lo stesso filtro dei runner di training."""
    import re

    num = df["match_id"].astype(str).apply(lambda s: int(re.search(r"-(\d+)$", s).group(1)) if re.search(r"-(\d+)$", s) else None)
    df = df[num.notnull()].copy()
    num = num[num.notnull()].astype(int)

    if gender == "male":
        mask = (num >= 1000) & (num < 2000)
    elif gender == "female":
        mask = (num >= 2000) & (num <= 3000)
    else:
        mask = num == num

    return df[mask.values]


def _load_model(path: Path):
    """Carica modello BDT (joblib) o MLP (torch)."""
    if path.suffix == ".pt":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        return "mlp", ckpt
    else:
        import joblib

        ckpt = joblib.load(path)
    return "bdt", ckpt


def _numeric_score(val):
    """Convert tennis score to numeric (AD=4); fallback to int for tiebreak."""
    mapping = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}
    if pd.isna(val):
        return 0, False
    s = str(val).strip().upper()
    if s in mapping:
        return mapping[s], False
    try:
        return int(s), True  # tiebreak numeric
    except Exception:  # noqa: BLE001
        return 0, False


def _match_point_holder(row) -> int:
    """Return 1 if P1 ha match point, 2 se P2, 0 altrimenti."""
    p1_score, is_tb1 = _numeric_score(row.get("P1Score"))
    p2_score, is_tb2 = _numeric_score(row.get("P2Score"))
    is_tb = is_tb1 or is_tb2
    p1_games = pd.to_numeric(row.get("P1GamesWon", 0), errors="coerce")
    p2_games = pd.to_numeric(row.get("P2GamesWon", 0), errors="coerce")
    p1_sets = pd.to_numeric(row.get("P1SetsWon", 0), errors="coerce")
    p2_sets = pd.to_numeric(row.get("P2SetsWon", 0), errors="coerce")
    
    # Determina se è best of 5 (maschi) o best of 3 (femmine) dal match_id
    import re
    match_id = str(row.get("match_id", ""))
    match = re.search(r"-(\d+)$", match_id)
    is_male = False
    if match:
        num = int(match.group(1))
        is_male = 1000 <= num < 2000  # Maschi: 1000-1999, Femmine: 2000-2999
    
    sets_to_win = 3 if is_male else 2
    p1_one_set_away = p1_sets == sets_to_win - 1
    p2_one_set_away = p2_sets == sets_to_win - 1

    # IMPORTANTE: Se entrambi i punteggi sono 0, è l'inizio di un nuovo game, non può essere match point
    if p1_score == 0 and p2_score == 0:
        return 0

    # Can win current set? DEVE essere realmente set point (uno dal vincere il set)
    if is_tb:
        # Nel tiebreak, serve almeno 6 punti e 2 di vantaggio per vincere
        p1_set_point = p1_score >= 6 and p1_score >= p2_score + 2
        p2_set_point = p2_score >= 6 and p2_score >= p1_score + 2
        p1_game_point = p1_set_point  # Nel TB, set point = game point
        p2_game_point = p2_set_point
    else:
        # Normale game: set point solo se:
        # - 5-4 o meglio (può chiudere 6-4)
        # - oppure 6-5 (può chiudere 7-5)
        # - NON 5-3 perché serve ancora vincere 2 game
        # - NON 7-6 o peggio perché il set è già finito
        p1_can_close_set = (p1_games == 5 and p2_games <= 4) or (p1_games == 6 and p2_games == 5)
        p2_can_close_set = (p2_games == 5 and p1_games <= 4) or (p2_games == 6 and p1_games == 5)
        
        # Game point: serve avere 40-x o vantaggio
        p1_game_point = p1_score >= 3 and p1_score > p2_score
        p2_game_point = p2_score >= 3 and p2_score > p1_score
        
        p1_set_point = p1_can_close_set
        p2_set_point = p2_can_close_set

    # Match point = un set dalla vittoria + può chiudere il set + ha game point
    if p1_one_set_away and p1_set_point and p1_game_point:
        return 1
    if p2_one_set_away and p2_set_point and p2_game_point:
        return 2
    return 0


def _boost_match_points(prob: np.ndarray, df_match: pd.DataFrame, strength: float = 2.5) -> np.ndarray:
    """Increase/decrease odds when unambiguous match point is detected."""
    adjusted = prob.copy()
    for idx, row in df_match.iterrows():
        holder = _match_point_holder(row)
        if holder == 0:
            continue
        i = df_match.index.get_loc(idx)
        p = adjusted[i]
        odds = p / (1.0 - p + 1e-9)
        factor = np.exp(strength)
        if holder == 1:
            odds *= factor
        else:
            odds /= factor
        adjusted[i] = np.clip(odds / (1.0 + odds), 0.02, 0.98)
    return adjusted


def _anchor_start(prob: np.ndarray, n: int = 5) -> np.ndarray:
    """Blend primi n punti verso 0.5 per evitare partenze sbilanciate."""
    anchored = prob.copy()
    n = min(n, len(prob))
    for i in range(n):
        w = 1.0 - i / max(n - 1, 1)  # decresce da 1 a 0
        anchored[i] = 0.5 * w + prob[i] * (1.0 - w)
    return anchored


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    mod = _load_point_module()

    df = mod.prepare_dataframe(args.files)
    df = _filter_by_gender(df, args.gender)
    df_match = df[df["match_id"] == args.match_id].copy()
    if df_match.empty:
        raise ValueError(f"match_id {args.match_id} non trovato dopo filtro genere={args.gender}")

    # Carica modello
    model_type, artifact = _load_model(Path(args.model))
    feature_cols = artifact.get("features", mod.MATCH_FEATURE_COLUMNS + mod.EXTRA_FEATURE_COLUMNS)

    X = df_match[feature_cols].to_numpy(dtype=float)

    if model_type == "bdt":
        model = artifact["model"]
        prob_p1 = model.predict_proba(X)[:, 1]
    else:
        import torch
        model = mod.MLP(dim_in=X.shape[1])
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        scaler = artifact["scaler"]
        X_scaled = scaler.transform(X)
        with torch.no_grad():
            logits = model(torch.tensor(X_scaled, dtype=torch.float32))
            prob_p1 = torch.sigmoid(logits).cpu().numpy()

    # Post-process: SOLO smoothing leggero per ridurre rumore
    prob_p1 = prob_p1.clip(0.01, 0.99)
    
    # Smoothing molto leggero (solo per ridurre rumore, non per cambiare trend)
    if len(prob_p1) >= 5:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        prob_p1 = np.convolve(prob_p1, kernel, mode="same")
        prob_p1 = np.clip(prob_p1, 0.01, 0.99)
    
    prob_p2 = 1.0 - prob_p1

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(prob_p1, label="P1 win prob", color="tab:blue")
    plt.plot(prob_p2, label="P2 win prob", color="tab:orange")
    plt.ylim(0, 1)
    plt.xlabel("Point index (ordinato)")
    plt.ylabel("Probabilità vittoria match")
    plt.title(f"Probabilità match – {args.match_id}")
    plt.legend()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[plot] salvato in {args.out_png}")

    if args.out_html:
        out_html = Path(args.out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prob_p1, mode="lines", name="P1 win prob", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(y=prob_p2, mode="lines", name="P2 win prob", line=dict(color="#ff7f0e")))
        fig.update_layout(
            title=f"Probabilità match – {args.match_id}",
            xaxis_title="Point index (ordinato)",
            yaxis_title="Probabilità vittoria match",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified",
        )
        fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
        print(f"[html] salvato in {out_html}")

    if args.out_csv:
        out_df = df_match[["match_id", "SetNo", "GameNo", "PointNumber", mod.SERVER_COL, mod.POINT_WINNER_COL]].copy()
        out_df["prob_p1_win_match"] = prob_p1
        out_df["prob_p2_win_match"] = prob_p2
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"[csv] salvato in {args.out_csv}")


if __name__ == "__main__":
    main()
