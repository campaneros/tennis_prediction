"""
Synthetic generator for point-level tennis rallies respecting scoring rules.

Produces a DataFrame compatible with the minimal columns expected by the
point-level feature builder:
  - match_id, SetNo, GameNo, PointNumber
  - PointServer, PointWinner
  - P1Score, P2Score (tennis strings or tiebreak integers)
  - P1GamesWon, P2GamesWon
  - SetWinner
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _simulate_point(p_srv_win: float) -> int:
    """Return 1 if server (player 1) wins, else 2."""
    return 1 if random.random() < p_srv_win else 2


def _score_str(points: int) -> str:
    mapping = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}
    return mapping.get(points, "0")


def _play_game(start_server: int, p1_srv: float, p2_srv: float) -> Tuple[int, List[Dict]]:
    """Simulate a standard game, return winner and per-point records."""
    points = []
    p1_points = p2_points = 0
    server = start_server
    point_no = 1
    while True:
        p_srv_win = p1_srv if server == 1 else p2_srv
        winner = _simulate_point(p_srv_win)
        if server == 2:  # receiving side is p1 if server=2
            winner = 2 if winner == 1 else 1
        if winner == 1:
            p1_points += 1
        else:
            p2_points += 1

        points.append(
            {
                "PointNumber": None,  # filled by outer loop
                "PointServer": server,
                "PointWinner": winner,
                "P1Score": _score_str(p1_points),
                "P2Score": _score_str(p2_points),
            }
        )

        # Check win condition
        if p1_points >= 4 or p2_points >= 4:
            if abs(p1_points - p2_points) >= 2:
                game_winner = 1 if p1_points > p2_points else 2
                return game_winner, points

        point_no += 1
        server = server  # server stays same during game


def _play_tiebreak(start_server: int, p1_srv: float, p2_srv: float) -> Tuple[int, List[Dict]]:
    """Simulate a first-to-7 (2-point margin) tiebreak (simplified)."""
    points = []
    p1_tb = p2_tb = 0
    point_no = 0
    server = start_server
    # Server switches after the first point, then every two points
    def server_for_point(idx: int, current_server: int) -> int:
        if idx == 0:
            return current_server
        block = (idx - 1) // 2
        if block % 2 == 0:
            return 2 if current_server == 1 else 1
        else:
            return current_server

    while True:
        srv = server_for_point(point_no, start_server)
        p_srv_win = p1_srv if srv == 1 else p2_srv
        winner = _simulate_point(p_srv_win)
        if srv == 2:
            winner = 2 if winner == 1 else 1
        if winner == 1:
            p1_tb += 1
        else:
            p2_tb += 1

        points.append(
            {
                "PointNumber": None,
                "PointServer": srv,
                "PointWinner": winner,
                "P1Score": str(p1_tb),
                "P2Score": str(p2_tb),
            }
        )

        if (p1_tb >= 7 or p2_tb >= 7) and abs(p1_tb - p2_tb) >= 2:
            tb_winner = 1 if p1_tb > p2_tb else 2
            return tb_winner, points

        point_no += 1


def _sets_needed(best_of_5_prob: float) -> Tuple[int, bool]:
    """Decide match format: True for best-of-5."""
    is_bo5 = random.random() < best_of_5_prob
    return (3 if is_bo5 else 2), is_bo5


def generate_synthetic_matches(
    n_matches: int = 2000,
    best_of_5_prob: float = 0.6,
    p1_srv_mean: float = 0.65,
    p2_srv_mean: float = 0.62,
    srv_std: float = 0.05,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Simulate matches with randomized serve strengths; covers standard games and tiebreaks.

    Args:
        n_matches: number of matches to generate.
        best_of_5_prob: probability a match is best-of-5 (else best-of-3).
        p1_srv_mean, p2_srv_mean: mean serve win prob for players.
        srv_std: standard deviation for per-match serve skill variability.
        seed: RNG seed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    records: List[Dict] = []
    for m in range(n_matches):
        match_id = f"synthetic-{m:05d}"
        sets_to_win, is_bo5 = _sets_needed(best_of_5_prob)

        p1_sets = p2_sets = 0
        server = 1 if m % 2 == 0 else 2
        # Draw per-match serve skills
        p1_srv = float(np.clip(np.random.normal(p1_srv_mean, srv_std), 0.45, 0.8))
        p2_srv = float(np.clip(np.random.normal(p2_srv_mean, srv_std), 0.45, 0.8))

        set_no = 1
        point_counter = 1
        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            p1_games = p2_games = 0
            game_no = 1
            while True:
                # Tiebreak check
                is_tb = (p1_games == 6 and p2_games == 6)
                if is_tb:
                    game_winner, points = _play_tiebreak(server, p1_srv, p2_srv)
                else:
                    game_winner, points = _play_game(server, p1_srv, p2_srv)

                # Set point numbers and add contextual columns
                for idx, p in enumerate(points):
                    p["match_id"] = match_id
                    p["SetNo"] = set_no
                    p["GameNo"] = game_no
                    p["PointNumber"] = point_counter
                    p["P1GamesWon"] = p1_games
                    p["P2GamesWon"] = p2_games
                    p["SetWinner"] = 0
                    records.append(p)
                    point_counter += 1

                if game_winner == 1:
                    p1_games += 1
                else:
                    p2_games += 1

                # Toggle server for next game
                server = 2 if server == 1 else 1

                # Set finished?
                if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
                    set_winner = 1 if p1_games > p2_games else 2
                    if set_winner == 1:
                        p1_sets += 1
                    else:
                        p2_sets += 1
                    records[-1]["SetWinner"] = set_winner
                    break
                if is_tb:
                    set_winner = game_winner
                    if set_winner == 1:
                        p1_sets += 1
                    else:
                        p2_sets += 1
                    records[-1]["SetWinner"] = set_winner
                    break

                game_no += 1

            set_no += 1

    return pd.DataFrame.from_records(records)
