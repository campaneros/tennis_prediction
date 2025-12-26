import os
import pandas as pd

# Column configuration for point-by-point data
MATCH_COL = "match_id"
SET_COL = "SetNo"
GAME_COL = "GameNo"
POINT_COL = "PointNumber"

SERVER_COL = "PointServer"        # 1 or 2
POINT_WINNER_COL = "PointWinner"  # 1 or 2
GAME_WINNER_COL = "GameWinner"    # 1 or 2

# Rolling window length
WINDOW = 20


def load_points_single(path: str) -> pd.DataFrame:
    """
    Load a single CSV file with point-by-point data.
    Sort by (match, set, game, point) and add 'source_file' column.
    Filters out invalid rows (metadata with PointNumber = 0X, 0Y, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    
    # Filter out metadata rows with PointNumber = "0X", "0Y", etc.
    # Keep only valid numeric PointNumber or standard values
    if POINT_COL in df.columns:
        # Remove rows where PointNumber contains letters (0X, 0Y, etc.)
        valid_mask = df[POINT_COL].astype(str).str.match(r'^\d+$', na=False)
        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            print(f"[load_points_single] Removed {n_removed} metadata rows from {path}")
        df = df[valid_mask].copy()
    
    df = df.sort_values([MATCH_COL, SET_COL, GAME_COL, POINT_COL]).reset_index(drop=True)
    df["source_file"] = path
    return df


def load_points_multiple(paths) -> pd.DataFrame:
    """
    Load and concatenate multiple point-by-point CSV files.
    """
    if not paths:
        raise ValueError("No input files provided")

    dfs = [load_points_single(p) for p in paths]
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all
