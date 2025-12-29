"""
Feature builder for point-by-point neural models.

The goal is to generate **pre-point** features only, avoiding any look-ahead
information from the current point outcome. All rolling stats and scoreboard
signals are computed using shifted values so they represent the state before
the point is played.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_loader import MATCH_COL, POINT_WINNER_COL, SERVER_COL
from .features import add_rolling_serve_return_features


_SCORE_MAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


def _score_series_to_numeric(series: pd.Series) -> pd.Series:
    """Convert score strings to numeric, keeping tiebreak numeric values."""
    s = series.astype(str).str.upper().str.strip()
    mapped = s.map(_SCORE_MAP)
    numeric = pd.to_numeric(s, errors="coerce")
    out = mapped.where(mapped.notna(), numeric)
    return out.fillna(0).astype(float)


def _set_cumsum_before_point(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Cumulative sets won *before* the current point for each match."""
    set_winner = pd.to_numeric(df.get("SetWinner", 0), errors="coerce").fillna(0).astype(int)
    p1_sets = ((set_winner == 1).astype(int).groupby(df[MATCH_COL]).cumsum()).shift(1).fillna(0)
    p2_sets = ((set_winner == 2).astype(int).groupby(df[MATCH_COL]).cumsum()).shift(1).fillna(0)
    return p1_sets.astype(float), p2_sets.astype(float)


def _build_pre_point_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create scoreboard/state columns representing the situation **before**
    each point starts.
    """
    df = df.copy()
    # Shift within game for point-level score
    df["p1_score_pre"] = (
        _score_series_to_numeric(
            df.groupby([MATCH_COL, "SetNo", "GameNo"])["P1Score"].shift(1)
        )
    )
    df["p2_score_pre"] = (
        _score_series_to_numeric(
            df.groupby([MATCH_COL, "SetNo", "GameNo"])["P2Score"].shift(1)
        )
    )
    df["p1_score_pre"] = df["p1_score_pre"].fillna(0.0)
    df["p2_score_pre"] = df["p2_score_pre"].fillna(0.0)

    # Games/sets before the point (shift by match)
    df["p1_games_pre"] = (
        pd.to_numeric(df.get("P1GamesWon", 0), errors="coerce")
        .groupby(df[MATCH_COL])
        .shift(1)
        .fillna(0.0)
    )
    df["p2_games_pre"] = (
        pd.to_numeric(df.get("P2GamesWon", 0), errors="coerce")
        .groupby(df[MATCH_COL])
        .shift(1)
        .fillna(0.0)
    )

    df["p1_sets_pre"], df["p2_sets_pre"] = _set_cumsum_before_point(df)

    # Progress indicators (normalized by match maxima)
    df["point_no_pre"] = pd.to_numeric(df.get("PointNumber", 0), errors="coerce").fillna(0.0)
    df["game_no_pre"] = pd.to_numeric(df.get("GameNo", 0), errors="coerce").fillna(0.0)
    df["set_no_pre"] = pd.to_numeric(df.get("SetNo", 1), errors="coerce").fillna(1.0)

    max_point = df.groupby(MATCH_COL)["point_no_pre"].transform("max").replace(0, 1)
    max_game = df.groupby(MATCH_COL)["game_no_pre"].transform("max").replace(0, 1)
    max_set = df.groupby(MATCH_COL)["set_no_pre"].transform("max").replace(0, 1)

    df["point_no_norm"] = (df["point_no_pre"] / max_point).clip(0.0, 1.0)
    df["game_no_norm"] = (df["game_no_pre"] / max_game).clip(0.0, 1.0)
    df["set_no_norm"] = (df["set_no_pre"] / max_set).clip(0.0, 1.0)

    return df


def _detect_tiebreak(df: pd.DataFrame) -> pd.Series:
    """Tiebreak flag based on pre-point games."""
    return (df["p1_games_pre"] == df["p2_games_pre"]) & (df["p1_games_pre"] >= 6)


def _game_point(my_score: float, opp_score: float, is_tb: bool) -> bool:
    if is_tb:
        return (my_score >= 6) and ((my_score - opp_score) >= 1)
    if my_score < 3:
        return False
    if my_score == 3 and opp_score < 3:
        return True
    return (my_score >= 4) and ((my_score - opp_score) >= 1)


def _set_point(my_games: float, opp_games: float, game_point_flag: bool, is_tb: bool) -> bool:
    if not game_point_flag:
        return False
    if is_tb:
        return True
    after_my = my_games + 1.0
    after_opp = opp_games
    return ((after_my >= 6) and ((after_my - after_opp) >= 2)) or (after_my == 7 and after_opp == 6)


def _match_point(
    my_sets: float, opp_sets: float, set_point_flag: bool, sets_needed: float
) -> bool:
    if not set_point_flag:
        return False
    return my_sets >= (sets_needed - 1.0)


def _recent_form(df: pd.DataFrame, history_points: int) -> pd.DataFrame:
    """Rolling/EWMA signals for recent performance."""
    df = df.copy()
    outcome = (df[POINT_WINNER_COL] == 1).astype(float)
    shifted = outcome.groupby(df[MATCH_COL]).shift(1).fillna(0.5)

    df["p1_recent_short"] = (
        shifted.groupby(df[MATCH_COL])
        .transform(lambda x: x.rolling(history_points, min_periods=1).mean())
        .fillna(0.5)
    )
    df["p1_recent_ewm"] = (
        shifted.groupby(df[MATCH_COL]).transform(lambda x: x.ewm(alpha=0.35, adjust=False).mean())
    ).fillna(0.5)
    return df


def _serve_context(df: pd.DataFrame) -> pd.DataFrame:
    """Serve/return categorical features with safe defaults."""
    df = df.copy()
    for col in ["ServeIndicator", "ServeWidth", "ServeDepth", "ReturnDepth", "Serve_Direction"]:
        if col not in df.columns:
            df[col] = "UNK"
        df[col] = df[col].fillna("UNK").astype(str)
    df["serve_indicator_numeric"] = pd.to_numeric(df["ServeIndicator"], errors="coerce").fillna(0.0)
    return df


def _compute_point_flags(df: pd.DataFrame, sets_needed: pd.Series) -> pd.DataFrame:
    """Game/set/match/break point flags computed on pre-point state."""
    df = df.copy()
    is_tb = _detect_tiebreak(df)
    df["is_tiebreak"] = is_tb.astype(float)

    gp_p1 = []
    gp_p2 = []
    sp_p1 = []
    sp_p2 = []
    mp_p1 = []
    mp_p2 = []
    bp_p1 = []
    bp_p2 = []

    for i, row in df.iterrows():
        p1s = float(row["p1_score_pre"])
        p2s = float(row["p2_score_pre"])
        p1g = float(row["p1_games_pre"])
        p2g = float(row["p2_games_pre"])
        p1set = float(row["p1_sets_pre"])
        p2set = float(row["p2_sets_pre"])
        tb = bool(is_tb.loc[i])
        server = int(row.get(SERVER_COL, 1))
        sets_need = float(sets_needed.loc[i])

        gp1_flag = _game_point(p1s, p2s, tb)
        gp2_flag = _game_point(p2s, p1s, tb)
        sp1_flag = _set_point(p1g, p2g, gp1_flag, tb)
        sp2_flag = _set_point(p2g, p1g, gp2_flag, tb)
        mp1_flag = _match_point(p1set, p2set, sp1_flag, sets_need)
        mp2_flag = _match_point(p2set, p1set, sp2_flag, sets_need)
        bp1_flag = gp1_flag and server == 2
        bp2_flag = gp2_flag and server == 1

        gp_p1.append(float(gp1_flag))
        gp_p2.append(float(gp2_flag))
        sp_p1.append(float(sp1_flag))
        sp_p2.append(float(sp2_flag))
        mp_p1.append(float(mp1_flag))
        mp_p2.append(float(mp2_flag))
        bp_p1.append(float(bp1_flag))
        bp_p2.append(float(bp2_flag))

    df["game_point_p1"] = gp_p1
    df["game_point_p2"] = gp_p2
    df["set_point_p1"] = sp_p1
    df["set_point_p2"] = sp_p2
    df["match_point_p1"] = mp_p1
    df["match_point_p2"] = mp_p2
    df["break_point_p1"] = bp_p1
    df["break_point_p2"] = bp_p2
    return df


def _importance(df: pd.DataFrame) -> pd.Series:
    """Lightweight importance for sample weighting."""
    weight = 1.0 + 0.6 * (df["game_point_p1"] + df["game_point_p2"])
    weight += 0.5 * (df["break_point_p1"] + df["break_point_p2"])
    weight += 0.8 * (df["set_point_p1"] + df["set_point_p2"])
    weight += 1.0 * (df["match_point_p1"] + df["match_point_p2"])
    weight += 0.4 * df["is_tiebreak"]
    return np.clip(weight, 1.0, 7.0)


@dataclass
class FeatureState:
    numeric_features: List[str]
    categorical_features: List[str]
    num_mean: Dict[str, float]
    num_std: Dict[str, float]
    cat_maps: Dict[str, Dict[str, int]]
    long_window: int
    short_window: int
    history_points: int


class PointNNFeatureBuilder:
    """
    Build, normalize, and encode point-level features for neural models.
    """

    def __init__(
        self,
        long_window: int = 24,
        short_window: int = 8,
        history_points: int = 12,
    ):
        self.long_window = int(long_window)
        self.short_window = int(short_window)
        self.history_points = int(history_points)

        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = [
            "ServeIndicator",
            "ServeWidth",
            "ServeDepth",
            "ReturnDepth",
            "Serve_Direction",
        ]
        self.num_mean: Dict[str, float] = {}
        self.num_std: Dict[str, float] = {}
        self.cat_maps: Dict[str, Dict[str, int]] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all model features without normalization/encoding."""
        df = df.copy()
        # Fill missing server info forward within match to avoid NaN cast issues
        if SERVER_COL in df.columns:
            df[SERVER_COL] = (
                pd.to_numeric(df[SERVER_COL], errors="coerce")
                .groupby(df[MATCH_COL])
                .ffill()
                .fillna(1)
                .astype(int)
            )
        df = _build_pre_point_state(df)

        # Determine set format per match (best-of-5 when max SetNo >= 4)
        max_set = df.groupby(MATCH_COL)["set_no_pre"].transform("max")
        sets_needed = np.where(max_set >= 4, 3.0, 2.0)
        sets_needed_series = pd.Series(sets_needed, index=df.index)

        df = _compute_point_flags(df, sets_needed_series)
        df["point_importance"] = _importance(df)

        # Rolling serve/return + leverage-like signals (uses only past points via shift)
        df = add_rolling_serve_return_features(
            df,
            long_window=self.long_window,
            short_window=self.short_window,
            weight_serve_return=True,
        )

        df = _recent_form(df, self.history_points)
        df = _serve_context(df)

        # Derived numeric signals
        df["server_is_p1"] = (df[SERVER_COL] == 1).astype(float)
        df["score_diff_pre"] = (df["p1_score_pre"] - df["p2_score_pre"]).clip(-5, 5)
        df["games_diff_pre"] = (df["p1_games_pre"] - df["p2_games_pre"]).clip(-6, 6)
        df["sets_diff_pre"] = (df["p1_sets_pre"] - df["p2_sets_pre"]).clip(-3, 3)

        # Momentum proxy from rolling serve probabilities
        df["leverage_proxy"] = (df["P_srv_win_long"] - df["P_srv_lose_long"]).clip(0.0, 1.0)

        # Target
        df["p1_wins_point"] = (df[POINT_WINNER_COL] == 1).astype(float)
        return df

    def _fit_numeric(self, features: pd.DataFrame):
        stats = features[self.numeric_features].agg(["mean", "std"])
        self.num_mean = stats.loc["mean"].to_dict()
        std = stats.loc["std"].replace(0, 1e-6)
        self.num_std = std.to_dict()

    def _encode_categorical(self, features: pd.DataFrame, fit: bool) -> np.ndarray:
        encoded_cols = []
        for col in self.categorical_features:
            series = features[col].fillna("UNK").astype(str)
            if fit or col not in self.cat_maps:
                uniques = sorted(series.unique().tolist())
                mapping = {v: i for i, v in enumerate(uniques)}
                mapping["<UNK>"] = len(mapping)
                self.cat_maps[col] = mapping
            mapping = self.cat_maps[col]
            encoded = series.map(lambda v: mapping.get(v, mapping["<UNK>"])).astype(np.int64)
            encoded_cols.append(encoded.values)
        if not encoded_cols:
            return np.zeros((len(features), 0), dtype=np.int64)
        return np.stack(encoded_cols, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        features = self._prepare_features(df)

        # Define numeric features once all columns exist
        self.numeric_features = [
            "p1_score_pre",
            "p2_score_pre",
            "score_diff_pre",
            "p1_games_pre",
            "p2_games_pre",
            "games_diff_pre",
            "p1_sets_pre",
            "p2_sets_pre",
            "sets_diff_pre",
            "point_no_norm",
            "game_no_norm",
            "set_no_norm",
            "server_is_p1",
            "P_srv_win_long",
            "P_srv_lose_long",
            "P_srv_win_short",
            "P_srv_lose_short",
            "leverage_proxy",
            "p1_recent_short",
            "p1_recent_ewm",
            "is_tiebreak",
            "game_point_p1",
            "game_point_p2",
            "break_point_p1",
            "break_point_p2",
            "set_point_p1",
            "set_point_p2",
            "match_point_p1",
            "match_point_p2",
            "point_importance",
        ]

        self._fit_numeric(features)
        X_num = (features[self.numeric_features] - pd.Series(self.num_mean)).div(pd.Series(self.num_std)).fillna(0.0)
        X_cat = self._encode_categorical(features, fit=True)

        y = features["p1_wins_point"].astype(np.float32).values
        sample_weight = (features["point_importance"].values.astype(np.float32)) ** 0.5

        return (
            X_num.values.astype(np.float32),
            X_cat,
            y,
            sample_weight,
            features[[MATCH_COL]].copy(),
        )

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        features = self._prepare_features(df)
        X_num = (features[self.numeric_features] - pd.Series(self.num_mean)).div(pd.Series(self.num_std)).fillna(0.0)
        X_cat = self._encode_categorical(features, fit=False)
        y = features["p1_wins_point"].astype(np.float32).values
        sample_weight = (features["point_importance"].values.astype(np.float32)) ** 0.5
        return (
            X_num.values.astype(np.float32),
            X_cat,
            y,
            sample_weight,
            features[[MATCH_COL]].copy(),
        )

    def to_state(self) -> FeatureState:
        return FeatureState(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            num_mean=self.num_mean,
            num_std=self.num_std,
            cat_maps=self.cat_maps,
            long_window=self.long_window,
            short_window=self.short_window,
            history_points=self.history_points,
        )

    @classmethod
    def from_state(cls, state: FeatureState) -> "PointNNFeatureBuilder":
        builder = cls(
            long_window=state.long_window,
            short_window=state.short_window,
            history_points=state.history_points,
        )
        builder.numeric_features = list(state.numeric_features)
        builder.categorical_features = list(state.categorical_features)
        builder.num_mean = dict(state.num_mean)
        builder.num_std = dict(state.num_std)
        builder.cat_maps = {k: dict(v) for k, v in state.cat_maps.items()}
        return builder

    def state_dict(self) -> dict:
        return asdict(self.to_state())

    @classmethod
    def load_state_dict(cls, state_dict: dict) -> "PointNNFeatureBuilder":
        state = FeatureState(**state_dict)
        return cls.from_state(state)
