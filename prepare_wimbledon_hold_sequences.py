#!/usr/bin/env python3
"""
Prepara sequenze punto-per-punto dai file wimbledon-points per un modello LSTM
che stima la probabilità di tenere il servizio (hold) in ogni game.

Non modifica file esistenti: salva nuovi artifact sotto data/.
"""

import glob
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


POINT_MAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}
MALE_THRESHOLD = 2000  # match_id ...-NNNN < 2000 -> maschi, >=2000 -> femmine
MALE_THRESHOLD = 2000  # match_id ...-NNNN < 2000 -> maschi, >=2000 -> femmine


def _numeric_point(score: str) -> int:
    return POINT_MAP.get(str(score).strip(), 0)


def _extract_match_year(match_id: str) -> int:
    try:
        return int(match_id.split("-")[0])
    except Exception:
        return -1


def _match_gender(match_id: str) -> str:
    try:
        num = int(match_id.split("-")[-1])
        return "male" if num < MALE_THRESHOLD else "female"
    except Exception:
        return "unknown"


def _add_set_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni riga aggiunge quanti set ciascun giocatore aveva vinto prima del set corrente.
    """
    out_parts = []
    for _, match_df in df.groupby("match_id", sort=False):
        set_winners = (
            match_df.groupby("SetNo")["SetWinner"]
            .apply(lambda s: s.replace(0, method="ffill").fillna(0).iloc[-1])
        )
        p1_so_far, p2_so_far = 0, 0
        set_to_counts: Dict[int, Tuple[int, int]] = {}
        for set_no in sorted(set_winners.index):
            set_to_counts[set_no] = (p1_so_far, p2_so_far)
            winner = set_winners.loc[set_no]
            if winner == 1:
                p1_so_far += 1
            elif winner == 2:
                p2_so_far += 1
        part = match_df.copy()
        part["P1SetsWonBefore"] = part["SetNo"].map(lambda s: set_to_counts.get(s, (0, 0))[0])
        part["P2SetsWonBefore"] = part["SetNo"].map(lambda s: set_to_counts.get(s, (0, 0))[1])
        out_parts.append(part)
    return pd.concat(out_parts, ignore_index=True)


def _extract_match_year(match_id: str) -> int:
    try:
        return int(match_id.split("-")[0])
    except Exception:
        return -1


def _match_gender(match_id: str) -> str:
    try:
        num = int(match_id.split("-")[-1])
        return "male" if num < MALE_THRESHOLD else "female"
    except Exception:
        return "unknown"


def _load_points(min_year: int = 2019, max_year: int = None, gender: str = "both") -> pd.DataFrame:
    files = sorted(glob.glob("data/*wimbledon-points*.csv"))
    if not files:
        raise FileNotFoundError("Nessun file data/*wimbledon-points*.csv trovato.")
    dfs = []
    for fpath in files:
        df = pd.read_csv(fpath)
        df["source_file"] = os.path.basename(fpath)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"], inplace=True)
    df_all = df_all.reset_index(drop=True)

    # Filtri anno e genere
    df_all["match_year"] = df_all["match_id"].apply(_extract_match_year)
    if min_year is not None:
        df_all = df_all[df_all["match_year"] >= min_year]
    if max_year is not None:
        df_all = df_all[df_all["match_year"] <= max_year]
    if gender != "both":
        df_all["match_gender"] = df_all["match_id"].apply(_match_gender)
        df_all = df_all[df_all["match_gender"] == gender]

    if df_all.empty:
        raise ValueError("Dopo i filtri (anno/genere) non restano dati. Controlla i parametri.")

    # Pulisci le colonne usate nelle feature: rimpiazza NaN con 0 o "0"
    numeric_cols = [
        "P1GamesWon",
        "P2GamesWon",
        "P1BreakPoint",
        "P2BreakPoint",
        "P1Momentum",
        "P2Momentum",
        "GameWinner",
        "PointServer",
    ]
    score_cols = ["P1Score", "P2Score"]
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna(0)
    for col in score_cols:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna("0")
    df_all.replace([float("inf"), float("-inf")], 0, inplace=True)
    return df_all


def _build_sequences(df: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
    """
    Crea una lista di sequenze, una per game, con label = server che tiene il servizio.
    """
    feature_names = [
        "p1_sets_before",
        "p2_sets_before",
        "p1_games",
        "p2_games",
        "game_diff",
        "p1_point",
        "p2_point",
        "point_diff",
        "server_is_p1",
        "server_is_p2",
        "p1_break_point",
        "p2_break_point",
        "p1_momentum",
        "p2_momentum",
    ]

    sequences: List[Dict] = []
    grouped = df.groupby(["match_id", "SetNo", "GameNo"], sort=False)
    for (match_id, set_no, game_no), g in grouped:
        g = g.sort_values("PointNumber")
        if g.empty:
            continue

        server_start = int(g.iloc[0]["PointServer"])
        if server_start not in (1, 2):
            continue  # skip se manca info sul server
        game_winner_series = g["GameWinner"].replace(0, np.nan).dropna()
        if game_winner_series.empty:
            continue  # nessun winner registrato
        game_winner = int(game_winner_series.iloc[-1])
        label_hold = 1 if game_winner == server_start else 0

        feats = []
        for _, row in g.iterrows():
            p1_point = _numeric_point(row["P1Score"])
            p2_point = _numeric_point(row["P2Score"])
            feat_row = [
                row["P1SetsWonBefore"],
                row["P2SetsWonBefore"],
                row["P1GamesWon"],
                row["P2GamesWon"],
                row["P1GamesWon"] - row["P2GamesWon"],
                p1_point,
                p2_point,
                p1_point - p2_point,
                int(row["PointServer"] == 1),
                int(row["PointServer"] == 2),
                row.get("P1BreakPoint", 0) or 0,
                row.get("P2BreakPoint", 0) or 0,
                row.get("P1Momentum", 0) or 0,
                row.get("P2Momentum", 0) or 0,
            ]
            if any(pd.isna(x) for x in feat_row):
                # Salta punti con valori non validi
                continue
            feats.append(feat_row)

        if not feats:
            continue  # skip game senza feature valide

        sequences.append(
            {
                "match_id": match_id,
                "set_no": int(set_no),
                "game_no": int(game_no),
                "server_start": server_start,
                "label_hold": label_hold,
                "features": np.asarray(feats, dtype=np.float32),
            }
        )
    return sequences, feature_names


def prepare_sequences(
    out_path: str = "data/wimbledon_hold_sequences.pkl",
    min_year: int = 2019,
    max_year: int = None,
    gender: str = "both",
) -> str:
    df = _load_points(min_year=min_year, max_year=max_year, gender=gender)
    df = _add_set_counts(df)
    sequences, feature_names = _build_sequences(df)
    payload = {
        "sequences": sequences,
        "feature_names": feature_names,
        "info": {
            "source": "wimbledon points",
            "num_sequences": len(sequences),
            "feature_size": len(feature_names),
            "filters": {"min_year": min_year, "max_year": max_year, "gender": gender},
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Salvato {len(sequences)} sequenze in {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepara sequenze wimbledon (con filtri anno/genere).")
    parser.add_argument("--out", default="data/wimbledon_hold_sequences.pkl", help="Percorso output pickle.")
    parser.add_argument("--min-year", type=int, default=2019, help="Anno minimo incluso (default: 2019).")
    parser.add_argument("--max-year", type=int, default=None, help="Anno massimo incluso.")
    parser.add_argument("--gender", choices=["male", "female", "both"], default="both", help="Filtra match per genere.")
    args = parser.parse_args()

    prepare_sequences(out_path=args.out, min_year=args.min_year, max_year=args.max_year, gender=args.gender)
