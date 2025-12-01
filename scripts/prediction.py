import os
import numpy as np

from .data_loader import load_points_multiple, MATCH_COL, SERVER_COL, WINDOW
from .config import load_config
from .features import (
    MATCH_FEATURE_COLUMNS,
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    add_additional_features,
    build_dataset,
)
from .model import load_model
from .plotting import plot_match_probabilities


def simulate_score_after_point_loss(row, server_wins: bool):
    """
    Simulate what the actual tennis score would be if the server wins/loses the point.
    Returns a modified row with updated scores and games.
    
    This properly implements tennis scoring rules:
    - 0, 15, 30, 40, deuce, advantage, game
    - Games accumulate to win sets
    - Realistic game/set transitions
    """
    new_row = row.copy()
    
    # Get current state
    p1_score = str(row.get('P1Score', '0')).strip()
    p2_score = str(row.get('P2Score', '0')).strip()
    server = int(row.get('PointServer', 1))
    p1_games = int(row.get('P1GamesWon', 0))
    p2_games = int(row.get('P2GamesWon', 0))
    
    # Score progression map
    score_progression = {'0': '15', '15': '30', '30': '40', '40': 'WIN', 'AD': 'WIN'}
    
    # Determine who wins the point
    if server_wins:
        winner = server
    else:
        winner = 2 if server == 1 else 1
    
    # Update scores based on winner
    game_won = False
    
    # Handle deuce/advantage situations
    if p1_score == '40' and p2_score == '40':
        # Deuce
        if winner == 1:
            new_row['P1Score'] = 'AD'
            new_row['P2Score'] = '40'
        else:
            new_row['P1Score'] = '40'
            new_row['P2Score'] = 'AD'
    elif p1_score == 'AD':
        if winner == 1:
            game_won = True
            new_row['P1GamesWon'] = p1_games + 1
        else:
            new_row['P1Score'] = '40'
            new_row['P2Score'] = '40'
    elif p2_score == 'AD':
        if winner == 2:
            game_won = True
            new_row['P2GamesWon'] = p2_games + 1
        else:
            new_row['P1Score'] = '40'
            new_row['P2Score'] = '40'
    else:
        # Normal scoring
        if winner == 1:
            next_score = score_progression.get(p1_score, '0')
            if next_score == 'WIN':
                game_won = True
                new_row['P1GamesWon'] = p1_games + 1
            else:
                new_row['P1Score'] = next_score
        else:
            next_score = score_progression.get(p2_score, '0')
            if next_score == 'WIN':
                game_won = True
                new_row['P2GamesWon'] = p2_games + 1
            else:
                new_row['P2Score'] = next_score
    
    # If game was won, reset point scores
    if game_won:
        new_row['P1Score'] = '0'
        new_row['P2Score'] = '0'
    
    return new_row


_FEATURE_INDEX = {name: idx for idx, name in enumerate(MATCH_FEATURE_COLUMNS)}


def advance_game_state_simple(row_features, server_wins_point: bool, point_importance: float = 1.0, eff_window: float = 20.0):
    """
    Update features to simulate counterfactual scenario where server loses/wins the point.
    
    Takes into account point_importance to amplify changes for critical points like:
    - Break points (importance ~2-5)
    - Set points (importance ~3-5)
    - Tiebreaks (importance ~3-5)

    Current feature order (22 features):
    [0] P_srv_win_long (dampened when tied)
    [1] P_srv_lose_long (dampened when tied)
    [2] P_srv_win_short
    [3] P_srv_lose_short
    [4] PointServer
    [5] momentum (dampened when tied)
    [6] Momentum_Diff
    [7] Score_Diff
    [8] Game_Diff
    [9] CurrentSetGamesDiff
    [10] SrvScr
    [11] RcvScr
    [12] SetNo
    [13] GameNo
    [14] PointNumber
    [15] point_importance
    [16] SetsWonDiff (scaled by match progress)
    [17] SetsWonAdvantage
    [18] SetWinProbPrior
    [19] SetWinProbEdge
    [20] SetWinProbLogit
    [21] is_decider_tied (1.0 in tied final set)

    For the counterfactual, we update:
    - P_srv_win/lose (long and short) using Bayesian update, amplified by importance
    - SrvScr/RcvScr based on outcome
    - Score_Diff changes (tennis scoring)
    - Game_Diff changes for game-deciding points
    - Momentum changes based on leverage and importance
    - point_importance stays the same (inherent to the game situation)
    """
    x = row_features.copy()
    
    # Extract long window probabilities
    idx = _FEATURE_INDEX
    P_win_long = float(x[idx["P_srv_win_long"]])
    P_lose_long = float(x[idx["P_srv_lose_long"]])
    P_win_short = float(x[idx["P_srv_win_short"]])
    P_lose_short = float(x[idx["P_srv_lose_short"]])
    server = int(x[idx[SERVER_COL]])
    current_momentum = float(x[idx["momentum"]])
    score_diff = float(x[idx["Score_Diff"]])
    game_diff = float(x[idx["Game_Diff"]])
    
    # Normalize probabilities
    def normalize_probs(p_win, p_lose):
        if p_win < 0.0:
            p_win = 0.0
        if p_lose < 0.0:
            p_lose = 0.0
        S = p_win + p_lose
        if S <= 0.0:
            return 0.5, 0.5
        return p_win / S, p_lose / S
    
    P_win_long, P_lose_long = normalize_probs(P_win_long, P_lose_long)
    P_win_short, P_lose_short = normalize_probs(P_win_short, P_lose_short)
    
    # Scale updates by importance (critical points have larger impact)
    # importance ranges from 1.0 (normal) to 7.0 (critical break/set point)
    # 
    # NEW APPROACH: Suppress normal points instead of amplifying critical ones
    # Use a sigmoid-like function that keeps normal points very low
    # and only starts growing significantly after importance > 2.5
    
    if point_importance <= 2.0:
        # Very conservative for truly normal points
        importance_factor = point_importance * 0.02  # 1.0->0.02, 2.0->0.04
    elif point_importance <= 3.0:
        # Moderate transition zone
        # Linear interpolation from 0.04 at 2.0 to 0.25 at 3.0
        importance_factor = 0.04 + (point_importance - 2.0) * 0.21
    else:
        # Above 3.0, use linear scaling
        # From 0.25 at 3.0 to 1.5 at 7.0
        importance_factor = 0.25 + (point_importance - 3.0) * 0.3125
        importance_factor = min(importance_factor, 1.5)
    
    # Bayesian update for long window
    alpha_long = P_win_long * eff_window
    beta_long = P_lose_long * eff_window
    
    # Update strength proportional to importance_factor
    # For normal points (importance_factor ~ 0.02), we add very few pseudo-observations
    # For critical points (importance_factor ~ 1.5), we add many
    # Use importance_factor^2.0 to further suppress normal point changes in probabilities
    bayesian_factor = np.power(importance_factor, 2.0)
    update_strength = 0.05 + bayesian_factor * 1.5  # Range: ~0.05 (normal) to 3.4 (critical)
    
    if server_wins_point:
        alpha_long_new = alpha_long + update_strength
        beta_long_new = beta_long
    else:
        alpha_long_new = alpha_long
        beta_long_new = beta_long + update_strength
    
    total_long = alpha_long_new + beta_long_new
    x[idx["P_srv_win_long"]] = alpha_long_new / total_long
    x[idx["P_srv_lose_long"]] = beta_long_new / total_long
    
    # Bayesian update for short window (more sensitive) - also amplified
    short_window = 5.0
    alpha_short = P_win_short * short_window
    beta_short = P_lose_short * short_window
    
    if server_wins_point:
        alpha_short_new = alpha_short + update_strength
        beta_short_new = beta_short
    else:
        alpha_short_new = alpha_short
        beta_short_new = beta_short + update_strength
    
    total_short = alpha_short_new + beta_short_new
    x[idx["P_srv_win_short"]] = alpha_short_new / total_short
    x[idx["P_srv_lose_short"]] = beta_short_new / total_short
    
    # Update momentum based on leverage and importance
    # Critical points have much larger momentum impact
    raw_leverage = max(0.0, x[idx["P_srv_win_long"]] - x[idx["P_srv_lose_long"]])
    
    # Cap leverage for normal points to prevent outlier changes
    # For normal points (importance <= 1.5), cap leverage at 0.15
    # For critical points, use full leverage
    if point_importance <= 1.5:
        leverage = min(raw_leverage, 0.15)
    elif point_importance <= 2.5:
        # Smooth transition: linearly interpolate cap from 0.15 to full leverage
        max_lev = 0.15 + (point_importance - 1.5) * (raw_leverage - 0.15)
        leverage = min(raw_leverage, max_lev)
    else:
        leverage = raw_leverage
    
    # Use importance_factor^2.5 to aggressively suppress momentum changes for normal points
    # 0.002^2.5 = 0.000003, 0.01^2.5 = 0.000003, 1.5^2.5 = 2.76
    momentum_factor = np.power(importance_factor, 2.5)
    momentum_change = 0.15 * leverage * momentum_factor  # Further reduced base from 0.2 to 0.15
    
    if server_wins_point:
        # Winning critical point increases momentum significantly
        x[idx["momentum"]] = current_momentum + momentum_change
    else:
        # Losing critical point decreases momentum significantly
        x[idx["momentum"]] = current_momentum - momentum_change
    
    # Clip momentum to reasonable range
    x[idx["momentum"]] = np.clip(x[idx["momentum"]], -3.0, 3.0)
    
    # Update Game_Diff and Score_Diff using smooth scaling
    # Use importance_factor^2.0 for game/score to suppress normal point changes
    # 0.002^2.0 = 0.000004, 1.5^2.0 = 2.25
    game_change = np.power(importance_factor, 2.0) * 1.0  # Reduced from 1.5 and squared
    score_change = np.power(importance_factor, 2.0) * 1.5  # Reduced from 2.5 and squared
    
    # Apply game differential change
    if server == 1:
        if server_wins_point:
            x[idx["Game_Diff"]] = game_diff + game_change
        else:
            x[idx["Game_Diff"]] = game_diff - game_change
    else:
        if server_wins_point:
            x[idx["Game_Diff"]] = game_diff - game_change
        else:
            x[idx["Game_Diff"]] = game_diff + game_change
    x[idx["Game_Diff"]] = np.clip(x[idx["Game_Diff"]], -3.0, 3.0)
    
    # Apply score differential change
    if server == 1:
        if server_wins_point:
            x[idx["Score_Diff"]] = score_diff + score_change
        else:
            x[idx["Score_Diff"]] = score_diff - score_change
    else:
        if server_wins_point:
            x[idx["Score_Diff"]] = score_diff - score_change
        else:
            x[idx["Score_Diff"]] = score_diff + score_change
    x[idx["Score_Diff"]] = np.clip(x[idx["Score_Diff"]], -2.0, 2.0)
    
    # Update SrvScr/RcvScr based on who served and won
    if server_wins_point:
        if server == 1:
            x[idx["SrvScr"]] += 1  # SrvScr
        # If server=2, P1 metrics don't change
    else:
        # Server lost
        if server == 2:
            x[idx["RcvScr"]] += 1  # RcvScr (P1 received and won)
        # If server=1 lost, P1 metrics don't improve
    
    return x


def compute_counterfactual_with_importance(X, model, point_importances):
    """
    Fast counterfactual using point_importance to scale feature changes.
    Uses the graduated scaling approach based on importance levels.
    """
    n = X.shape[0]
    prob_p1 = np.zeros(n)
    prob_p2 = np.zeros(n)
    prob_p1_lose_srv = np.zeros(n)
    prob_p2_lose_srv = np.zeros(n)

    for i in range(n):
        x = X[i]
        importance = point_importances[i] if i < len(point_importances) else 1.0

        p1_now = float(model.predict_proba(x.reshape(1, -1))[:, 1])
        prob_p1[i] = p1_now
        prob_p2[i] = 1.0 - p1_now

        x_lose = advance_game_state_simple(x, server_wins_point=False, point_importance=importance)
        p1_lose = float(model.predict_proba(x_lose.reshape(1, -1))[:, 1])
        prob_p1_lose_srv[i] = p1_lose
        prob_p2_lose_srv[i] = 1.0 - p1_lose

    # Statistics
    diffs = np.abs(prob_p1 - prob_p1_lose_srv)
    print(f"[counterfactual-importance] Mean change: {np.mean(diffs):.4f}, Max: {np.max(diffs):.4f}")
    print(f"[counterfactual-importance] Points with >10% change: {np.sum(diffs > 0.10)}/{n}")

    return prob_p1, prob_p2, prob_p1_lose_srv, prob_p2_lose_srv


def compute_counterfactual_point_by_point(df_valid, df_raw_with_labels, model, config_path=None, 
                                          match_id=None, importance_threshold=None, mode="realistic"):
    """
    Compute counterfactual by modifying dataset point by point.
    
    For each point (or critical points only):
    1. Create a FRESH copy of the raw dataset
    2. Modify ONLY that specific point's score
    3. Rebuild ALL features from scratch
    4. Get model prediction
    
    Args:
        df_valid: DataFrame with all features computed
        df_raw_with_labels: Raw DataFrame for rebuilding features
        model: Trained model
        config_path: Config file path
        match_id: If provided, only process points from this match
        importance_threshold: If provided, only process points above this importance
        mode: "realistic" (all points) or "semi-realistic" (critical only)
    
    Returns:
        prob_p1, prob_p2, prob_p1_lose_srv, prob_p2_lose_srv
    """
    from .features import (
        add_rolling_serve_return_features,
        add_leverage_and_momentum,
        add_additional_features,
        build_dataset,
    )
    from .config import load_config
    
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))
    
    n = len(df_valid)
    prob_p1 = np.zeros(n)
    prob_p2 = np.zeros(n)
    prob_p1_lose_srv = np.zeros(n)
    prob_p2_lose_srv = np.zeros(n)
    
    # Get current features
    X_current, _, _, _, _ = build_dataset(df_valid)
    
    # Compute current probabilities
    for i in range(len(X_current)):
        p1_now = float(model.predict_proba(X_current[i].reshape(1, -1))[:, 1])
        prob_p1[i] = p1_now
        prob_p2[i] = 1.0 - p1_now
    
    # Determine which points to simulate
    point_importances = df_valid['point_importance'].values if 'point_importance' in df_valid.columns else np.ones(n)
    
    if importance_threshold is not None:
        # Semi-realistic: only critical points
        simulate_mask = point_importances > importance_threshold
        n_simulate = np.sum(simulate_mask)
        print(f"[{mode}] Simulating {n_simulate}/{n} points with importance > {importance_threshold}")
    else:
        # Realistic: all points
        simulate_mask = np.ones(n, dtype=bool)
        n_simulate = n
        print(f"[{mode}] Simulating all {n_simulate} points")
    
    # Process each point that needs simulation
    simulated_count = 0
    for i, idx in enumerate(df_valid.index):
        if not simulate_mask[i]:
            # Non-critical point: alternative scenario identical to current dataset.
            # Use the same probabilities so curves overlap exactly.
            prob_p1_lose_srv[i] = prob_p1[i]
            prob_p2_lose_srv[i] = prob_p2[i]
            continue
        
        # Create a FRESH copy of raw dataset for THIS point only
        cf_df = df_raw_with_labels.copy()
        
        # Simulate score change for THIS specific point only
        current_row = df_raw_with_labels.loc[idx]
        modified_row = simulate_score_after_point_loss(current_row, server_wins=False)
        
        # Update ONLY this point's scores in the copy
        for col in ['P1Score', 'P2Score', 'P1GamesWon', 'P2GamesWon']:
            if col in modified_row.index:
                cf_df.at[idx, col] = modified_row[col]
        
        # Rebuild ALL features from this modified dataset
        cf_df = add_rolling_serve_return_features(cf_df, long_window=long_window, short_window=short_window)
        cf_df = add_additional_features(cf_df)
        cf_df = add_leverage_and_momentum(cf_df, alpha=alpha)
        
        X_cf, _, mask_cf, _, _ = build_dataset(cf_df)
        
        # Find this point in the rebuilt dataset
        try:
            cf_idx = list(cf_df.index).index(idx)
            if cf_idx < len(X_cf) and cf_idx < len(mask_cf) and mask_cf[cf_idx]:
                p1_lose = float(model.predict_proba(X_cf[cf_idx].reshape(1, -1))[:, 1])
                prob_p1_lose_srv[i] = p1_lose
                prob_p2_lose_srv[i] = 1.0 - p1_lose
            else:
                # Fallback
                prob_p1_lose_srv[i] = prob_p1[i]
                prob_p2_lose_srv[i] = prob_p2[i]
        except (ValueError, IndexError):
            # Fallback if index not found
            prob_p1_lose_srv[i] = prob_p1[i]
            prob_p2_lose_srv[i] = prob_p2[i]
        
        simulated_count += 1
        if simulated_count % 50 == 0:
            print(f"[{mode}] Processed {simulated_count}/{n_simulate} points...")
    
    # Statistics
    diffs = np.abs(prob_p1 - prob_p1_lose_srv)
    print(f"[{mode}] Mean change: {np.mean(diffs):.4f}, Max: {np.max(diffs):.4f}")
    print(f"[{mode}] Points with >10% change: {np.sum(diffs > 0.10)}/{n}")
    
    if importance_threshold is not None:
        simulated_diffs = diffs[simulate_mask]
        if len(simulated_diffs) > 0:
            print(f"[{mode}] Simulated points >10% change: {np.sum(simulated_diffs > 0.10)}/{n_simulate}")

    return prob_p1, prob_p2, prob_p1_lose_srv, prob_p2_lose_srv




def run_prediction(file_paths, model_path: str, match_id: str, plot_dir: str, config_path: str | None = None, 
                   counterfactual_mode: str = "importance"):
    """
    End-to-end prediction + plotting for a given set of files and one match_id.

    - Loads data
    - Rebuilds features (including momentum and point importance)
    - Loads trained model
    - Computes current and counterfactual probabilities
    - Produces probability trajectory plots
    
    Args:
        file_paths: Paths to data files
        model_path: Path to trained model
        match_id: Match ID to analyze
        plot_dir: Directory for plots
        config_path: Path to config file
        counterfactual_mode: "importance" (default), "semi-realistic", or "realistic"
            - importance: Fast scaling using point_importance (always generated)
            - semi-realistic: Dataset modification for critical points only (importance > 3.5)
            - realistic: Dataset modification for ALL points (slow!)
    """
    os.makedirs(plot_dir, exist_ok=True)
    cfg = load_config(config_path)
    fcfg = cfg.get("features", {})
    long_window = int(fcfg.get("long_window", 20))
    short_window = int(fcfg.get("short_window", 5))
    alpha = float(fcfg.get("momentum_alpha", 1.2))

    df = load_points_multiple(file_paths)
    df = add_match_labels(df)
    
    # Keep a copy of raw data with labels for counterfactual simulation
    df_raw_with_labels = df.copy()
    
    df = add_rolling_serve_return_features(df, long_window=long_window, short_window=short_window)
    df = add_additional_features(df)
    df = add_leverage_and_momentum(df, alpha=alpha)

    X, y, mask, sample_weights, _ = build_dataset(df)
    df_valid = df[mask].copy()

    model = load_model(model_path)

    # ALWAYS compute importance-based counterfactual (fast)
    print("\n=== COUNTERFACTUAL: Point Importance Scaling ===")
    prob_p1, prob_p2, prob_p1_lose, prob_p2_lose = compute_counterfactual_with_importance(
        X, model, sample_weights
    )

    df_valid["prob_p1"] = prob_p1
    df_valid["prob_p2"] = prob_p2
    df_valid["prob_p1_lose_srv"] = prob_p1_lose
    df_valid["prob_p2_lose_srv"] = prob_p2_lose

    # Filter to match for additional simulations
    match_id_str = str(match_id)
    df_valid[MATCH_COL] = df_valid[MATCH_COL].astype(str)
    match_df = df_valid[df_valid[MATCH_COL] == match_id_str]
    
    # Save probabilities to CSV for later plot regeneration
    # Add suffix based on counterfactual mode
    mode_suffix = "" if counterfactual_mode == "importance" else f"_{counterfactual_mode}"
    probs_file = os.path.join(plot_dir, f"match_{match_id_str}_probabilities{mode_suffix}.csv")
    if not match_df.empty:
        # Save essential columns for plotting
        cols_to_save = [MATCH_COL, 'prob_p1', 'prob_p2', 'prob_p1_lose_srv', 'prob_p2_lose_srv']
        if 'point_importance' in match_df.columns:
            cols_to_save.append('point_importance')
        # If we have semi-realistic/realistic alternative probabilities, save them too
        for col in ["prob_p1_alt", "prob_p2_alt", "prob_p1_lose_alt", "prob_p2_lose_alt"]:
            if col in match_df.columns:
                cols_to_save.append(col)
        match_df[cols_to_save].to_csv(probs_file, index=False)
        print(f"[prediction] Saved probabilities to: {probs_file}")
    
    # Optionally compute semi-realistic or realistic counterfactuals
    if counterfactual_mode in ["semi-realistic", "realistic"] and not match_df.empty:
        print(f"\n=== ADDITIONAL MODE: {counterfactual_mode.upper()} ===")
        
        if counterfactual_mode == "semi-realistic":
            # Only critical points in this match
            threshold = 3.5
            prob_p1_2, prob_p2_2, prob_p1_lose_2, prob_p2_lose_2 = compute_counterfactual_point_by_point(
                match_df, df_raw_with_labels, model, config_path,
                match_id=match_id_str, importance_threshold=threshold, mode="semi-realistic"
            )
        else:  # realistic
            # ALL points in this match
            prob_p1_2, prob_p2_2, prob_p1_lose_2, prob_p2_lose_2 = compute_counterfactual_point_by_point(
                match_df, df_raw_with_labels, model, config_path,
                match_id=match_id_str, importance_threshold=None, mode="realistic"
            )
        
        # Add to match dataframe
        match_df = match_df.copy()
        match_df["prob_p1_alt"] = prob_p1_2
        match_df["prob_p2_alt"] = prob_p2_2
        match_df["prob_p1_lose_alt"] = prob_p1_lose_2
        match_df["prob_p2_lose_alt"] = prob_p2_lose_2
        
        # Update df_valid with match results
        for col in ["prob_p1_alt", "prob_p2_alt", "prob_p1_lose_alt", "prob_p2_lose_alt"]:
            df_valid.loc[match_df.index, col] = match_df[col]
    
    # Print statistics
    if not match_df.empty:
        print(f"\n=== MATCH {match_id_str} STATISTICS ===")
        print(f"  Points in match: {len(match_df)}")
        
        if 'point_importance' in match_df.columns:
            match_importances = match_df['point_importance'].values
            print(f"  Point importance: mean={np.mean(match_importances):.2f}, max={np.max(match_importances):.2f}")
            print(f"  Critical points (>3.5): {np.sum(match_importances > 3.5)}")
        
        # Stats for importance method
        match_diffs = np.abs(match_df["prob_p1"] - match_df["prob_p1_lose_srv"])
        print(f"\n  Importance method:")
        print(f"    Mean change: {np.mean(match_diffs):.4f}, Max: {np.max(match_diffs):.4f}")
        print(f"    Points with >10% change: {np.sum(match_diffs > 0.10)}/{len(match_diffs)}")
        
        # Stats for alternative method if computed
        if f"prob_p1_lose_alt" in match_df.columns:
            match_diffs_alt = np.abs(match_df["prob_p1_alt"] - match_df["prob_p1_lose_alt"])
            print(f"\n  {counterfactual_mode.capitalize()} method:")
            print(f"    Mean change: {np.mean(match_diffs_alt):.4f}, Max: {np.max(match_diffs_alt):.4f}")
            print(f"    Points with >10% change: {np.sum(match_diffs_alt > 0.10)}/{len(match_diffs_alt)}")

    # Generate plots
    print(f"\n=== Generating plots (mode={counterfactual_mode}) ===")
    
    if counterfactual_mode in ["semi-realistic", "realistic"] and "prob_p1_alt" in df_valid.columns:
        # Generate comparison plot
        from .plotting import plot_match_probabilities_comparison
        plot_match_probabilities_comparison(df_valid, match_id_str, plot_dir, counterfactual_mode)
    else:
        # Generate standard plot
        plot_match_probabilities(df_valid, match_id_str, plot_dir)
