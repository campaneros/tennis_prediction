"""
Monte Carlo match-win probability with feature recomputation at each simulated point.

For each prefix of the real match:
- Use the real history up to that point.
- Simulate the remaining points many times.
- At every simulated point, rebuild point-level features via the PointNNFeatureBuilder
  so rolling serve/return and recent-form features evolve with simulated outcomes.
- Aggregate the fraction of simulated futures where P1 wins the match.

Scoring assumptions:
- Standard tennis scoring with deuce/advantage.
- Tiebreak at 6-6: first to 7 with 2-point lead.
- Best-of-5 if max SetNo in real match >= 4, else best-of-3.

Output:
- CSV with columns prob_p1_match_mc and prob_p2_match_mc.
- Optional PNG/HTML plots showing both players' match-win probabilities over points.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import torch

from scripts.data_loader import load_points_multiple, MATCH_COL
from scripts.point_nn_training import _load_checkpoint  # noqa: protected access
from scripts.point_nn_features import SERVER_COL


SCORE_TO_PTS = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}
PTS_TO_SCORE = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}


def score_str_to_pts(s: str) -> int:
    return SCORE_TO_PTS.get(str(s).strip().upper(), 0)


def score_pts_to_str(v: int) -> str:
    v = max(0, min(4, int(v)))
    return PTS_TO_SCORE.get(v, "0")


def tennis_advance_point(state: dict, winner: int, sets_to_win: int) -> dict:
    """
    Advance tennis scoring by one point with deuce/AD and 7-point tiebreak at 6-6.
    """
    p1_pts = state["p1_pts"]
    p2_pts = state["p2_pts"]
    p1_games = state["p1_games"]
    p2_games = state["p2_games"]
    p1_sets = state["p1_sets"]
    p2_sets = state["p2_sets"]
    server = state["server"]
    set_no = state["set_no"]

    # Tiebreak detection
    in_tiebreak = (p1_games == 6 and p2_games == 6)

    if in_tiebreak:
        # First to 7 with 2-point lead
        if winner == 1:
            p1_pts += 1
        else:
            p2_pts += 1

        tb_won = False
        if (p1_pts >= 7 or p2_pts >= 7) and abs(p1_pts - p2_pts) >= 2:
            tb_won = True
            set_winner = 1 if p1_pts > p2_pts else 2

        if tb_won:
            if set_winner == 1:
                p1_sets += 1
            else:
                p2_sets += 1
            set_no += 1
            p1_games = 0
            p2_games = 0
            p1_pts = 0
            p2_pts = 0
        # Server alternation every 2 points in tiebreak (approx: just flip each point)
        server = 2 if server == 1 else 1
    else:
        # Normal game
        if winner == 1:
            p1_pts += 1
        else:
            p2_pts += 1

        game_won = False
        if (p1_pts >= 4 or p2_pts >= 4) and abs(p1_pts - p2_pts) >= 2:
            game_won = True
            game_winner = 1 if p1_pts > p2_pts else 2

        if game_won:
            if game_winner == 1:
                p1_games += 1
            else:
                p2_games += 1
            p1_pts = 0
            p2_pts = 0
            server = 2 if server == 1 else 1

            # Set win check
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

    match_done = (p1_sets >= sets_to_win) or (p2_sets >= sets_to_win)

    state.update(
        dict(
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
    )
    return state


def build_row_from_state(state: dict, point_no: int, game_no: int, winner: int | None) -> dict:
    """
    Build a minimal row with required columns for the feature builder.
    winner can be None (placeholder) before sampling.
    """
    return dict(
        PointServer=state["server"],
        PointWinner=winner if winner is not None else 0,
        P1Score=score_pts_to_str(state["p1_pts"]),
        P2Score=score_pts_to_str(state["p2_pts"]),
        P1GamesWon=state["p1_games"],
        P2GamesWon=state["p2_games"],
        P1SetsWon=state["p1_sets"],
        P2SetsWon=state["p2_sets"],
        SetNo=state["set_no"],
        GameNo=game_no,
        PointNumber=point_no,
        ServeIndicator="UNK",
        ServeWidth="UNK",
        ServeDepth="UNK",
        ReturnDepth="UNK",
        Serve_Direction="UNK",
        is_best_of_5=1.0 if state.get("sets_to_win", 3) == 3 else 0.0,
        **{MATCH_COL: state.get("match_id", "sim")},
    )


def derive_state_from_row(row: pd.Series) -> dict:
    return dict(
        server=int(row.get("PointServer", 1)),
        p1_sets=int(row.get("P1SetsWon", 0)),
        p2_sets=int(row.get("P2SetsWon", 0)),
        p1_games=int(row.get("P1GamesWon", 0)),
        p2_games=int(row.get("P2GamesWon", 0)),
        p1_pts=score_str_to_pts(row.get("P1Score", "0")),
        p2_pts=score_str_to_pts(row.get("P2Score", "0")),
        set_no=int(row.get("SetNo", 1)),
        match_done=False,
        last_winner=int(row.get("PointWinner", 0)),
        match_id=row.get(MATCH_COL, "sim"),
    )


def simulate_match_win_prob_full(
    df_match: pd.DataFrame,
    model,
    builder,
    device: str = "cpu",
    n_sims: int = 64,
    feature_window: int = 150,
) -> np.ndarray:
    """
    Monte Carlo with feature recomputation every simulated point.
    """
    max_set = int(df_match["SetNo"].max())
    sets_to_win = 3 if max_set >= 4 else 2

    probs = np.zeros(len(df_match))

    for idx in range(len(df_match)):
        prefix = df_match.iloc[: idx + 1].copy()
        last_row = prefix.iloc[-1]
        base_state = derive_state_from_row(last_row)
        base_state["sets_to_win"] = sets_to_win

        wins = 0
        for _ in range(n_sims):
            # Copy prefix history
            history = [r._asdict() if hasattr(r, "_asdict") else r.to_dict() for _, r in prefix.iterrows()]
            state = base_state.copy()
            point_no = int(last_row.get("PointNumber", idx + 1)) + 1
            game_no = int(last_row.get("GameNo", 1))

            # Continue until match ends
            steps = 0
            while not state["match_done"] and steps < 2000:
                # Append placeholder row for current point
                history.append(build_row_from_state(state, point_no, game_no, winner=None))

                # Keep only recent history to limit computation
                if feature_window and len(history) > feature_window:
                    history = history[-feature_window:]

                sim_df = pd.DataFrame(history)

                # Build features and get prob for last row
                feats = builder._prepare_features(sim_df)  # type: ignore
                X_num, X_cat, _, _, _ = builder.transform(feats)  # type: ignore
                with torch.no_grad():
                    logits = model(
                        torch.tensor(X_num[-1:], dtype=torch.float32, device=device),
                        torch.tensor(X_cat[-1:], dtype=torch.long, device=device),
                    )
                    p1_point = float(torch.sigmoid(logits).cpu().numpy().reshape(-1)[0])

                # Sample winner and update last row
                winner = 1 if np.random.rand() < p1_point else 2
                history[-1]["PointWinner"] = winner

                # Advance state
                state = tennis_advance_point(state, winner, sets_to_win)

                # Increment counters
                point_no += 1
                if state["p1_pts"] == 0 and state["p2_pts"] == 0:
                    game_no += 1

                steps += 1

            if state["p1_sets"] >= sets_to_win:
                wins += 1

        probs[idx] = wins / float(max(1, n_sims))

    return probs


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo match-win prob with full feature recomputation")
    parser.add_argument("--files", nargs="+", required=True, help="CSV files with point data")
    parser.add_argument("--model", required=True, help="Path to point NN checkpoint (pth)")
    parser.add_argument("--match-id", required=True, help="Match ID to process")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sims", type=int, default=64, help="Monte Carlo rollouts per point")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("--plot-dir", default=None, help="Optional directory to save PNG/HTML plots")
    parser.add_argument("--max-points", type=int, default=None, help="Optional limit on number of points (for quick tests)")
    parser.add_argument("--feature-window", type=int, default=150, help="History window size for feature recomputation")
    args = parser.parse_args()

    model, builder, _ = _load_checkpoint(args.model, device=args.device)
    df = load_points_multiple(args.files)
    df = df[df[MATCH_COL] == args.match_id].copy()
    df = df.sort_values(["SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)
    if args.max_points:
        df = df.iloc[: args.max_points].reset_index(drop=True)
    df["is_best_of_5"] = 1.0  # assume bo5 for Wimbledon

    probs = simulate_match_win_prob_full(
        df, model, builder, device=args.device, n_sims=args.sims, feature_window=args.feature_window
    )
    df_out = df.copy()
    df_out["prob_p1_match_mc"] = probs
    df_out["prob_p2_match_mc"] = 1.0 - probs

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"[mc-match-full] Saved to {args.output}")
    else:
        print(df_out[["SetNo", "GameNo", "PointNumber", "prob_p1_match_mc"]].tail())

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        base = os.path.join(args.plot_dir, f"match_{args.match_id}_mc_full_match_prob")
        # PNG
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
            print(f"[mc-match-full] Saved plot PNG to {base}.png")
        except Exception as e:
            print(f"[mc-match-full] Could not save PNG: {e}")
        # HTML
        try:
            import plotly.express as px
            fig = px.line(
                df_out.reset_index(),
                x=df_out.index,
                y=["prob_p1_match_mc", "prob_p2_match_mc"],
                title=f"Match win probability (MC full) — {args.match_id}",
                labels={"x": "Point index in match", "value": "Match win probability", "variable": "player"},
            )
            fig.write_html(base + ".html", include_plotlyjs="cdn")
            print(f"[mc-match-full] Saved plot HTML to {base}.html")
        except Exception as e:
            print(f"[mc-match-full] Could not save HTML: {e}")


if __name__ == "__main__":
    main()
