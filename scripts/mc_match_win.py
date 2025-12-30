"""
Monte Carlo match-win probability using the point-level NN.

Approach (fast, approximate):
- First compute point-win probabilities on the REAL sequence (no simulation).
- For each prefix i, run Monte Carlo on the REMAINING points using those
  per-point probabilities as-is (no state-dependent recomputation).
- Advance tennis scoring with sampled winners; repeat for many rollouts.
- Fraction of rollouts where P1 wins -> match win probability at point i.

Notes:
- This ignores feedback of simulated outcomes on future point probabilities
  (they stay fixed to the original model outputs), but is fast and captures
  score/state through the scoring engine.
- Scoring:
  * Games: to 4 points with 2-point lead (deuce/AD).
  * Set: to 6 with 2-game lead, or 7-6 via tiebreak at 6-6 (7 with 2-pt lead).
  * Best-of-5 if SetNo max >= 4, else best-of-3.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from scripts.data_loader import load_points_multiple, MATCH_COL
from scripts.point_nn_training import _load_checkpoint  # noqa: protected access
from scripts.point_nn_features import SERVER_COL


def tennis_advance_point(state: dict, winner: int, sets_to_win: int) -> dict:
    """
    Advance tennis scoring state by one point.
    state keys: server, p1_sets, p2_sets, p1_games, p2_games, p1_pts, p2_pts, set_no
    winner: 1 or 2
    Returns updated state.
    """
    p1_pts = state["p1_pts"]
    p2_pts = state["p2_pts"]
    p1_games = state["p1_games"]
    p2_games = state["p2_games"]
    p1_sets = state["p1_sets"]
    p2_sets = state["p2_sets"]
    server = state["server"]
    set_no = state["set_no"]

    def point_to_next(pts: int) -> int:
        # Simple count; game win check uses >=4 and 2-point lead
        return pts + 1

    # Update point score
    if winner == 1:
        p1_pts = point_to_next(p1_pts)
    else:
        p2_pts = point_to_next(p2_pts)

    # Check game win
    game_won = False
    game_winner = 0
    if p1_pts >= 4 or p2_pts >= 4:
        if abs(p1_pts - p2_pts) >= 2 and (p1_pts >= 4 or p2_pts >= 4):
            game_won = True
            game_winner = 1 if p1_pts > p2_pts else 2

    if game_won:
        if game_winner == 1:
            p1_games += 1
        else:
            p2_games += 1
        # reset points
        p1_pts = 0
        p2_pts = 0
        # toggle server for next game (simplified)
        server = 2 if server == 1 else 1

    # Check set win (tiebreak at 6-6, first to 7 with 2-pt lead)
    set_won = False
    if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
        set_won = True
        set_winner = 1 if p1_games > p2_games else 2
    elif p1_games == 7 and p2_games == 6:
        set_won = True
        set_winner = 1
    elif p2_games == 7 and p1_games == 6:
        set_won = True
        set_winner = 2

    if set_won:
        if set_winner == 1:
            p1_sets += 1
        else:
            p2_sets += 1
        set_no += 1
        p1_games = 0
        p2_games = 0
        p1_pts = 0
        p2_pts = 0
        # server carries over as is (simplified)

    match_done = (p1_sets >= sets_to_win) or (p2_sets >= sets_to_win)

    return dict(
        server=server,
        p1_sets=p1_sets,
        p2_sets=p2_sets,
        p1_games=p1_games,
        p2_games=p2_games,
        p1_pts=p1_pts,
        p2_pts=p2_pts,
        set_no=set_no,
        match_done=match_done,
        last_winner=winner,
    )


def simulate_match_win_prob_fixed_ps(
    df_match: pd.DataFrame,
    point_probs: np.ndarray,
    n_sims: int = 256,
) -> np.ndarray:
    """
    Monte Carlo using fixed per-point probabilities (no recompute during rollout).
    """
    max_set = int(df_match["SetNo"].max())
    sets_to_win = 3 if max_set >= 4 else 2
    probs = np.zeros(len(df_match))

    # Pre-extract servers for real sequence; use them as fallback pattern
    servers = df_match["PointServer"].to_numpy().astype(int)

    for idx in range(len(df_match)):
        # Starting state from real history up to idx
        pref = df_match.iloc[: idx + 1]
        last = pref.iloc[-1]
        state = dict(
            server=int(last.get("PointServer", servers[0] if len(servers) > 0 else 1)),
            p1_sets=int(last.get("P1SetsWon", 0)),
            p2_sets=int(last.get("P2SetsWon", 0)),
            p1_games=int(last.get("P1GamesWon", 0)),
            p2_games=int(last.get("P2GamesWon", 0)),
            p1_pts=0,
            p2_pts=0,
            set_no=int(last.get("SetNo", 1)),
            match_done=False,
        )

        wins = 0
        for s in range(n_sims):
            sim_state = state.copy()
            ptr = idx
            steps = 0
            # run until match ends; cap steps to avoid infinite loops
            while not sim_state["match_done"] and steps < 2000:
                p = float(point_probs[min(ptr, len(point_probs) - 1)])
                winner = 1 if np.random.rand() < p else 2
                sim_state = tennis_advance_point(sim_state, winner, sets_to_win)
                ptr += 1
                steps += 1
            if sim_state["p1_sets"] >= sets_to_win:
                wins += 1

        probs[idx] = wins / float(max(1, n_sims))

    return probs


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo match-win prob from point NN")
    parser.add_argument("--files", nargs="+", required=True, help="CSV files with point data")
    parser.add_argument("--model", required=True, help="Path to point NN checkpoint (pth)")
    parser.add_argument("--match-id", required=True, help="Match ID to process")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sims", type=int, default=256, help="Monte Carlo rollouts per point")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("--plot-dir", default=None, help="Optional directory to save PNG/HTML plots")
    args = parser.parse_args()

    model, builder, ckpt = _load_checkpoint(args.model, device=args.device)
    df = load_points_multiple(args.files)
    df = df[df[MATCH_COL] == args.match_id].copy()
    df = df.sort_values(["SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)

    # Compute point-level probabilities on the real sequence
    feats_df = builder._prepare_features(df)  # type: ignore
    X_num, X_cat, _, _, _ = builder.transform(feats_df)  # type: ignore

    import torch

    with torch.no_grad():
        logits = model(
            torch.tensor(X_num, dtype=torch.float32, device=args.device),
            torch.tensor(X_cat, dtype=torch.long, device=args.device),
        )
        point_probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    probs = simulate_match_win_prob_fixed_ps(df, point_probs, n_sims=args.sims)
    df_out = df.copy()
    df_out["prob_p1_match_mc"] = probs
    df_out["prob_p2_match_mc"] = 1.0 - probs

    if args.output:
        pd.DataFrame(df_out).to_csv(args.output, index=False)
        print(f"[mc-match] Saved to {args.output}")
    else:
        print(df_out[["SetNo", "GameNo", "PointNumber", "prob_p1_match_mc"]].tail())

    if args.plot_dir:
        import os
        os.makedirs(args.plot_dir, exist_ok=True)
        base = os.path.join(args.plot_dir, f"match_{args.match_id}_mc_match_prob")

        # PNG via matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(df_out.index, df_out["prob_p1_match_mc"], label="P1 wins match")
            plt.plot(df_out.index, df_out["prob_p2_match_mc"], label="P2 wins match")
            plt.xlabel("Point index in match")
            plt.ylabel("Match win probability")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(base + ".png", dpi=150)
            plt.close()
            print(f"[mc-match] Saved plot PNG to {base}.png")
        except Exception as e:
            print(f"[mc-match] Could not save PNG: {e}")

        # HTML via plotly
        try:
            import plotly.express as px
            fig = px.line(
                df_out.reset_index(),
                x=df_out.index,
                y=["prob_p1_match_mc", "prob_p2_match_mc"],
                title=f"Match win probability (MC) — {args.match_id}",
                labels={"x": "Point index in match", "value": "Match win probability", "variable": "player"},
            )
            fig.write_html(base + ".html", include_plotlyjs="cdn")
            print(f"[mc-match] Saved plot HTML to {base}.html")
        except Exception as e:
            print(f"[mc-match] Could not save HTML: {e}")


if __name__ == "__main__":
    main()
