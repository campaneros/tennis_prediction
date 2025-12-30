"""
Rule-based prior for point-win probability.

Uses pre-point flags (break/set/match point, tiebreak) and server form to
produce a conservative prior, then can be blended with the model output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def compute_rule_prior(features_df: pd.DataFrame, blend_strength: float = 0.35) -> np.ndarray:
    """
    Build a prior using:
      - server advantage from P_srv_win_long
      - score context (games/sets diff)
      - tie-break flag
      - break/set/match point flags
    blend_strength controls how strongly we pull toward the prior when later blended.
    """
    df = features_df.copy()

    base = df["P_srv_win_long"].to_numpy().astype(float)

    # Context columns
    is_bp_p1 = df.get("break_point_p1", pd.Series(0.0)).to_numpy().astype(float)
    is_bp_p2 = df.get("break_point_p2", pd.Series(0.0)).to_numpy().astype(float)

    # Set point and match point flags
    is_sp_p1 = df.get("set_point_p1", pd.Series(0.0)).to_numpy().astype(float)
    is_sp_p2 = df.get("set_point_p2", pd.Series(0.0)).to_numpy().astype(float)
    is_mp_p1 = df.get("match_point_p1", pd.Series(0.0)).to_numpy().astype(float)
    is_mp_p2 = df.get("match_point_p2", pd.Series(0.0)).to_numpy().astype(float)

    # Tiebreak: reduce raw server dominance
    is_tb = df.get("is_tiebreak", pd.Series(0.0)).to_numpy().astype(float)

    # Set/game context
    sets_diff = df.get("sets_diff_pre", pd.Series(0.0)).to_numpy().astype(float)
    games_diff = df.get("games_diff_pre", pd.Series(0.0)).to_numpy().astype(float)

    # Apply effects in log-odds space for stability
    logit = _safe_logit(base)

    # Progress in match: flatten early sets/games, mild sharpening late
    point_prog = df.get("point_no_norm", pd.Series(0.0, index=df.index)).to_numpy().astype(float)
    set_prog = df.get("set_no_norm", pd.Series(0.0, index=df.index)).to_numpy().astype(float)
    progress = np.maximum(point_prog, set_prog)
    # Overall scale ramps from very gentle (0.5) to moderate (1.0) to avoid early skew
    progress_scale = np.interp(progress, [0.0, 0.25, 0.7, 1.0], [0.5, 0.7, 0.9, 1.0])

    # Context swings (sets/games) are also damped early, strengthened late
    sets_scale = np.interp(progress, [0.0, 0.3, 0.7, 1.0], [0.3, 0.5, 0.8, 0.95])
    games_scale = np.interp(progress, [0.0, 0.3, 0.7, 1.0], [0.2, 0.45, 0.7, 0.85])

    # Tiebreak dampening: shrink toward 0.5
    logit = logit * (1.0 - 0.35 * is_tb)

    # Sets advantage: meaningful swing (win prob correlates with set lead)
    logit += 1.2 * sets_diff * sets_scale
    # Games advantage: smaller swing
    logit += 0.35 * games_diff * games_scale

    # Break point: nudge toward receiver
    logit -= 0.8 * is_bp_p1  # P1 break point -> hurts server (P2 serve)
    logit += 0.8 * is_bp_p2  # P2 break point -> hurts server (P1 serve)

    # Set point: boost the player on set point (stronger)
    sp_boost = np.interp(progress, [0.0, 0.5, 1.0], [0.7, 0.9, 1.1])
    logit += sp_boost * is_sp_p1
    logit -= sp_boost * is_sp_p2

    # Match point: very strong boost
    mp_boost = np.interp(progress, [0.0, 0.5, 1.0], [1.2, 1.6, 2.0])
    logit += mp_boost * is_mp_p1
    logit -= mp_boost * is_mp_p2

    # Apply overall progress scaling last to temper early volatility
    logit = logit * progress_scale

    prior = _safe_sigmoid(logit)

    # Blend strength is used downstream; here we just return the prior
    return prior
