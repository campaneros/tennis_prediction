import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import plotly.graph_objects as go

from .data_loader import MATCH_COL


def _save_interactive_plot(traces, verticals, title, filepath):
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(go.Scatter(
            x=trace["x"],
            y=trace["y"],
            mode="lines",
            name=trace["name"],
            line=dict(color=trace.get("color"), dash=trace.get("dash", "solid"), width=trace.get("width", 2))
        ))

    legend_styles = {}
    for vline in verticals:
        fig.add_vline(
            x=vline["x"],
            line=dict(color=vline.get("color", "gray"), dash=vline.get("dash", "dash"), width=vline.get("width", 2))
        )

        label = vline.get("label")
        if label and label not in legend_styles:
            legend_styles[label] = {
                "color": vline.get("color", "gray"),
                "dash": vline.get("dash", "dash"),
                "width": vline.get("width", 2)
            }

    for label, style in legend_styles.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=label,
            line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
            showlegend=True,
            hoverinfo="skip"
        ))
    fig.update_layout(
        title=dict(text=title, y=0.96, x=0.0, xanchor="left"),
        xaxis_title="Point index in match",
        yaxis_title="Match win probability",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0
        ),
        margin=dict(t=100, b=120)
    )
    fig.write_html(filepath)


def plot_match_probabilities(df_valid: pd.DataFrame, match_id_to_plot: str, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)

    dfm = df_valid[df_valid[MATCH_COL] == match_id_to_plot].reset_index(drop=True)

    if dfm.empty:
        print(f"[plot] No rows for match_id={match_id_to_plot}")
        return

    max_dev_now = float(np.max(np.abs(dfm["prob_p1"] + dfm["prob_p2"] - 1.0)))
    max_dev_lose = float(np.max(np.abs(dfm["prob_p1_lose_srv"] + dfm["prob_p2_lose_srv"] - 1.0)))
    print(f"[plot] Max deviation P1+P2=1 (current): {max_dev_now:.2e}")
    print(f"[plot] Max deviation P1+P2=1 (lose):    {max_dev_lose:.2e}")

    x = np.arange(len(dfm))

    # Precompute critical point markers to reuse across both plots
    vertical_cache = []
    if 'point_importance' in dfm.columns:
        break_points = dfm[dfm['point_importance'] > 2.0]
        for idx, row in break_points.iterrows():
            importance = row['point_importance']
            if importance >= 6.5:
                label = "Decisive Point"
                color = 'darkred'
                alpha = 0.6
                linestyle = '-'
            elif importance >= 6.0:
                label = "Set/Break Point"
                color = 'darkorange'
                alpha = 0.5
                linestyle = '--'
            elif importance >= 5.0:
                label = "Critical Point"
                color = 'purple'
                alpha = 0.4
                linestyle = ':'
            elif importance >= 3.5:
                label = "Important Point"
                color = 'brown'
                alpha = 0.3
                linestyle = '-.'
            else:
                label = "Break Point"
                color = 'gray'
                alpha = 0.2
                linestyle = ':'
            vertical_cache.append({
                "idx": idx,
                "label": label,
                "color": color,
                "alpha": alpha,
                "linestyle": linestyle,
            })

    vertical_cache = []
    if 'point_importance' in dfm.columns:
        critical_points = dfm[dfm['point_importance'] > 3.5]
        for idx, row in critical_points.iterrows():
            importance = row['point_importance']
            if importance >= 6.5:
                label, color, alpha, linestyle = "Decisive", 'darkred', 0.5, '-'
            elif importance >= 6.0:
                label, color, alpha, linestyle = "Set/Break", 'darkorange', 0.4, '--'
            else:
                label, color, alpha, linestyle = "Critical", 'purple', 0.3, ':'
            vertical_cache.append({"idx": idx, "label": label, "color": color, "alpha": alpha, "linestyle": linestyle})

    plt.figure(figsize=(14, 7))

    plt.plot(x, dfm["prob_p1"], label="P1 wins match (current)", linewidth=2)
    plt.plot(x, dfm["prob_p2"], label="P2 wins match (current)", linewidth=2)

    plt.plot(
        x, dfm["prob_p1_lose_srv"],
        "--", label="P1 wins | server loses point", linewidth=1.5,
    )
    plt.plot(
        x, dfm["prob_p2_lose_srv"],
        "--", label="P2 wins | server loses point", linewidth=1.5,
    )

    interactive_traces = [
        {"x": x, "y": dfm["prob_p1"], "name": "P1 wins match (current)", "color": "blue"},
        {"x": x, "y": dfm["prob_p2"], "name": "P2 wins match (current)", "color": "orange"},
        {"x": x, "y": dfm["prob_p1_lose_srv"], "name": "P1 wins | server loses point", "color": "green", "dash": "dash"},
        {"x": x, "y": dfm["prob_p2_lose_srv"], "name": "P2 wins | server loses point", "color": "red", "dash": "dash"},
    ]
    vertical_lines = []

    # Add vertical lines and annotations for critical points
    if 'point_importance' in dfm.columns:
        break_points = dfm[dfm['point_importance'] > 2.0]

        added_labels = set()
        dash_map = {'-': 'solid', '--': 'dash', ':': 'dot', '-.': 'dashdot'}
        shown_labels = set()

        for idx, row in break_points.iterrows():
            point_idx = idx
            importance = row['point_importance']

            if importance >= 6.5:
                label = "Decisive Point"
                color = 'darkred'
                alpha = 0.6
                linestyle = '-'
                linewidth = 2.5
            elif importance >= 6.0:
                label = "Set/Break Point"
                color = 'darkorange'
                alpha = 0.5
                linestyle = '--'
                linewidth = 2.0
            elif importance >= 5.0:
                label = "Critical Point"
                color = 'purple'
                alpha = 0.4
                linestyle = ':'
                linewidth = 2.0
            elif importance >= 3.5:
                label = "Important Point"
                color = 'brown'
                alpha = 0.3
                linestyle = '-.'
                linewidth = 1.5
            else:  # > 2.0
                label = "Break Point"
                color = 'gray'
                alpha = 0.2
                linestyle = ':'
                linewidth = 1.0

            if label not in added_labels:
                plt.axvline(x=point_idx, color=color, alpha=alpha, linestyle=linestyle,
                            linewidth=linewidth, label=label)
                added_labels.add(label)
            else:
                plt.axvline(x=point_idx, color=color, alpha=alpha, linestyle=linestyle,
                            linewidth=linewidth)

            show_label = label not in shown_labels
            vertical_lines.append({
                "x": point_idx,
                "color": color,
                "dash": dash_map.get(linestyle, 'dash'),
                "width": linewidth,
                "label": label if show_label else None,
            })
            if show_label:
                shown_labels.add(label)

    plt.xlabel("Point index in match")
    plt.ylabel("Match win probability")
    plt.ylim(0.0, 1.0)
    plt.title(f"Match win probabilities and counterfactual (server loses)\nmatch_id={match_id_to_plot}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    fname = os.path.join(plot_dir, f"match_{match_id_to_plot}_probabilities.png")
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"[plot] Saved match probability plot to: {fname}")

    html_name = os.path.join(plot_dir, f"match_{match_id_to_plot}_probabilities.html")
    _save_interactive_plot(interactive_traces, vertical_lines,
                           f"Match probabilities - {match_id_to_plot}", html_name)
    print(f"[plot] Saved interactive plot to: {html_name}")


def plot_match_probabilities_comparison(df_valid: pd.DataFrame, match_id_to_plot: str, plot_dir: str):
    """
    Generate 2 separate plots comparing the two counterfactual methods:
    1. Importance-based counterfactual (fast, all points)
    2. Hybrid simulation (critical points >3.5 get score simulation)
    """
    os.makedirs(plot_dir, exist_ok=True)

    dfm = df_valid[df_valid[MATCH_COL] == match_id_to_plot].reset_index(drop=True)

    if dfm.empty:
        print(f"[plot] No rows for match_id={match_id_to_plot}")
        return

    x = np.arange(len(dfm))
    
    # === PLOT 1: Importance-based counterfactual ===
    plt.figure(figsize=(14, 7))

    plt.plot(x, dfm["prob_p1_imp"], label="P1 wins match (current)", linewidth=2, color='blue')
    plt.plot(x, dfm["prob_p2_imp"], label="P2 wins match (current)", linewidth=2, color='orange')

    plt.plot(
        x, dfm["prob_p1_lose_imp"],
        "--", label="P1 wins | server loses point", linewidth=1.5, color='green'
    )
    plt.plot(
        x, dfm["prob_p2_lose_imp"],
        "--", label="P2 wins | server loses point", linewidth=1.5, color='red'
    )

    # Add vertical lines for critical points
    if 'point_importance' in dfm.columns:
        critical_points = dfm[dfm['point_importance'] > 5.0]
        added_labels = set()
        
        for idx, row in critical_points.iterrows():
            importance = row['point_importance']
            
            if importance >= 6.5:
                label = "Decisive Point"
                color = 'darkred'
                alpha = 0.5
                linestyle = '-'
            elif importance >= 6.0:
                label = "Set/Break Point"
                color = 'darkorange'
                alpha = 0.4
                linestyle = '--'
            else:
                label = "Critical Point"
                color = 'purple'
                alpha = 0.3
                linestyle = ':'
            
            if label not in added_labels:
                plt.axvline(x=idx, color=color, alpha=alpha, linestyle=linestyle, 
                           linewidth=2, label=label)
                added_labels.add(label)
            else:
                plt.axvline(x=idx, color=color, alpha=alpha, linestyle=linestyle, 
                           linewidth=2)

    plt.xlabel("Point index in match")
    plt.ylabel("Match win probability")
    plt.ylim(0.0, 1.0)
    plt.title(f"METHOD 1: Importance-based counterfactual\nmatch_id={match_id_to_plot}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    fname1 = os.path.join(plot_dir, f"match_{match_id_to_plot}_importance.png")
    plt.savefig(fname1, dpi=150)
    plt.close()
    print(f"[plot] Saved importance-based plot to: {fname1}")

    # === PLOT 2: Hybrid simulation counterfactual ===
    plt.figure(figsize=(14, 7))

    plt.plot(x, dfm["prob_p1_sim"], label="P1 wins match (current)", linewidth=2, color='blue')
    plt.plot(x, dfm["prob_p2_sim"], label="P2 wins match (current)", linewidth=2, color='orange')

    plt.plot(
        x, dfm["prob_p1_lose_sim"],
        "--", label="P1 wins | server loses point", linewidth=1.5, color='green'
    )
    plt.plot(
        x, dfm["prob_p2_lose_sim"],
        "--", label="P2 wins | server loses point", linewidth=1.5, color='red'
    )

    # Highlight critical points that get score simulation (>3.5)
    if 'point_importance' in dfm.columns:
        simulated_points = dfm[dfm['point_importance'] > 3.5]
        added_labels = set()
        
        for idx, row in simulated_points.iterrows():
            importance = row['point_importance']
            
            if importance >= 6.5:
                label = "Decisive (simulated)"
                color = 'darkred'
                alpha = 0.6
                linestyle = '-'
            elif importance >= 6.0:
                label = "Set/Break (simulated)"
                color = 'darkorange'
                alpha = 0.5
                linestyle = '--'
            else:
                label = "Critical (simulated)"
                color = 'purple'
                alpha = 0.4
                linestyle = ':'
            
            if label not in added_labels:
                plt.axvline(x=idx, color=color, alpha=alpha, linestyle=linestyle, 
                           linewidth=2, label=label)
                added_labels.add(label)
            else:
                plt.axvline(x=idx, color=color, alpha=alpha, linestyle=linestyle, 
                           linewidth=2)

    plt.xlabel("Point index in match")
    plt.ylabel("Match win probability")
    plt.ylim(0.0, 1.0)
    plt.title(f"METHOD 2: Hybrid simulation (score sim for importance >3.5)\nmatch_id={match_id_to_plot}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    fname2 = os.path.join(plot_dir, f"match_{match_id_to_plot}_simulation.png")
    plt.savefig(fname2, dpi=150)
    plt.close()
    print(f"[plot] Saved simulation-based plot to: {fname2}")


def plot_hyperopt_results(cv_results, plot_dir: str, prefix: str = "hyperopt"):
    os.makedirs(plot_dir, exist_ok=True)

    results_df = pd.DataFrame(cv_results)
    mean_scores = results_df["mean_test_roc_auc"]

    # Histogram of AUC scores
    plt.figure(figsize=(6, 4))
    plt.hist(mean_scores, bins=10)
    plt.xlabel("Mean CV ROC AUC")
    plt.ylabel("Count")
    plt.title("Distribution of CV ROC AUC over hyperparameters")
    plt.tight_layout()
    hist_name = os.path.join(plot_dir, f"{prefix}_score_hist.png")
    plt.savefig(hist_name, dpi=150)
    plt.close()
    print(f"[hyperopt plot] Saved: {hist_name}")

    params_to_plot = [
        "param_n_estimators",
        "param_max_depth",
        "param_learning_rate",
        "param_subsample",
        "param_colsample_bytree",
    ]

    for param in params_to_plot:
        if param not in results_df.columns:
            continue

        x = results_df[param].astype(float)
        y = mean_scores

        plt.figure(figsize=(6, 4))
        plt.scatter(x, y)
        plt.xlabel(param.replace("param_", ""))
        plt.ylabel("Mean CV ROC AUC")
        plt.title(f"ROC AUC vs {param.replace('param_', '')}")
        plt.tight_layout()
        fname = os.path.join(plot_dir, f"{prefix}_score_vs_{param.replace('param_', '')}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[hyperopt plot] Saved: {fname}")


def plot_confusion_matrix_and_roc(
    cm,
    y_true,
    y_score,
    auc_value,
    acc_value,
    prec_value,
    rec_value,
    f1_value,
    plot_dir: str,
    filename_prefix: str = "best_model_cv",
):
    """
    Plot confusion matrix and ROC curve side by side, similar to the
    figure you showed.

    Convention:
        0 = Player 1 wins
        1 = Player 2 (rival) wins

    cm: 2x2 confusion matrix with labels [0,1]
    y_true: array-like, true labels in {0,1}
    y_score: array-like, predicted probability of class 1 (P2 wins)
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Confusion matrix values
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # --------------------------------------------------------------
    # Left: confusion matrix
    # --------------------------------------------------------------
    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    ax.set_title("Confusion Matrix (5-Fold CV)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    # Annotate cells with counts
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm[i, j]}",
                ha="center", va="center",
                color="black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------
    # Right: ROC curve
    # --------------------------------------------------------------
    ax2 = axes[1]
    fpr, tpr, _ = roc_curve(y_true, y_score)

    ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})", linewidth=2)
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1)  # diagonal
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")

    # Optionally display metrics in the ROC panel
    text_str = (
        f"Acc  = {acc_value:.3f}\n"
        f"Prec = {prec_value:.3f}\n"
        f"Rec  = {rec_value:.3f}\n"
        f"F1   = {f1_value:.3f}"
    )
    ax2.text(
        0.65, 0.25,
        text_str,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout()
    fname = os.path.join(plot_dir, f"{filename_prefix}_confusion_roc.png")
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"[plot] Saved confusion matrix + ROC figure to: {fname}")


def plot_match_probabilities_comparison(df_valid: pd.DataFrame, match_id_to_plot: str, plot_dir: str, mode: str = "semi-realistic"):
    """
    Generate 2 plots comparing different counterfactual methods:
    1. Importance-based counterfactual (always computed)
    2. Dataset simulation counterfactual (semi-realistic or realistic)
    
    Args:
        df_valid: DataFrame with predictions
        match_id_to_plot: Match ID to plot
        plot_dir: Output directory
        mode: "semi-realistic" or "realistic"
    """
    os.makedirs(plot_dir, exist_ok=True)

    dfm = df_valid[df_valid[MATCH_COL] == match_id_to_plot].reset_index(drop=True)

    if dfm.empty:
        print(f"[plot] No rows for match_id={match_id_to_plot}")
        return

    x = np.arange(len(dfm))

    # Precompute critical point metadata for reuse
    vertical_cache = []
    if 'point_importance' in dfm.columns:
        break_points = dfm[dfm['point_importance'] > 2.0]
        for idx, row in break_points.iterrows():
            importance = row['point_importance']
            if importance >= 6.5:
                label = "Decisive Point"
                color = 'darkred'
                alpha = 0.6
                linestyle = '-'
            elif importance >= 6.0:
                label = "Set/Break Point"
                color = 'darkorange'
                alpha = 0.5
                linestyle = '--'
            elif importance >= 5.0:
                label = "Critical Point"
                color = 'purple'
                alpha = 0.4
                linestyle = ':'
            elif importance >= 3.5:
                label = "Important Point"
                color = 'brown'
                alpha = 0.3
                linestyle = '-.'
            else:
                label = "Break Point"
                color = 'gray'
                alpha = 0.2
                linestyle = ':'
            vertical_cache.append({
                "idx": idx,
                "label": label,
                "color": color,
                "alpha": alpha,
                "linestyle": linestyle,
            })
    
    # Check which columns are available
    has_importance = 'prob_p1' in dfm.columns and 'prob_p1_lose_srv' in dfm.columns
    has_alt = 'prob_p1_alt' in dfm.columns and 'prob_p1_lose_alt' in dfm.columns
    
    if not has_importance:
        print(f"[plot] Missing importance-based probabilities")
        return
    
    # PLOT 1: Importance-based counterfactual (ALWAYS generated)
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.plot(x, dfm["prob_p1"], label="P1 wins match (current)", linewidth=2, color='blue')
    ax1.plot(x, dfm["prob_p2"], label="P2 wins match (current)", linewidth=2, color='orange')
    ax1.plot(x, dfm["prob_p1_lose_srv"], "--", label="P1 wins | server loses (importance)", 
             linewidth=1.5, color='green')
    ax1.plot(x, dfm["prob_p2_lose_srv"], "--", label="P2 wins | server loses (importance)", 
             linewidth=1.5, color='red')
    
    # Add critical points markers
    vertical_lines1 = []
    if vertical_cache:
        added_labels = set()
        dash_map = {'-': 'solid', '--': 'dash', ':': 'dot'}
        for v in vertical_cache:
            show_label = v['label'] not in added_labels
            if show_label:
                ax1.axvline(x=v['idx'], color=v['color'], alpha=v['alpha'], linestyle=v['linestyle'], linewidth=2, label=v['label'])
                added_labels.add(v['label'])
            else:
                ax1.axvline(x=v['idx'], color=v['color'], alpha=v['alpha'], linestyle=v['linestyle'], linewidth=2)
            vertical_lines1.append({
                "x": v['idx'],
                "color": v['color'],
                "dash": dash_map.get(v['linestyle'], 'dash'),
                "label": v['label'] if show_label else None
            })
    
    ax1.set_xlabel("Point index in match")
    ax1.set_ylabel("Match win probability")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f"Counterfactual: Point Importance Scaling\nmatch_id={match_id_to_plot}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    fig1.tight_layout()
    
    fname1 = os.path.join(plot_dir, f"match_{match_id_to_plot}_importance.png")
    fig1.savefig(fname1, dpi=150)
    plt.close(fig1)
    print(f"[plot] Saved importance-based plot to: {fname1}")

    traces1 = [
        {"x": x, "y": dfm["prob_p1"], "name": "P1 wins match (current)", "color": "blue"},
        {"x": x, "y": dfm["prob_p2"], "name": "P2 wins match (current)", "color": "orange"},
        {"x": x, "y": dfm["prob_p1_lose_srv"], "name": "P1 wins | server loses (importance)", "color": "green", "dash": "dash"},
        {"x": x, "y": dfm["prob_p2_lose_srv"], "name": "P2 wins | server loses (importance)", "color": "red", "dash": "dash"},
    ]
    html1 = os.path.join(plot_dir, f"match_{match_id_to_plot}_importance.html")
    _save_interactive_plot(traces1, vertical_lines1, f"Match probabilities (importance) - {match_id_to_plot}", html1)
    print(f"[plot] Saved interactive importance plot to: {html1}")
    
    # PLOT 2: Simulation-based counterfactual (if available)
    if has_alt:
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        
        ax2.plot(x, dfm["prob_p1_alt"], label="P1 wins match (current)", linewidth=2, color='blue')
        ax2.plot(x, dfm["prob_p2_alt"], label="P2 wins match (current)", linewidth=2, color='orange')
        ax2.plot(x, dfm["prob_p1_lose_alt"], "--", label=f"P1 wins | server loses ({mode})", 
                 linewidth=1.5, color='green')
        ax2.plot(x, dfm["prob_p2_lose_alt"], "--", label=f"P2 wins | server loses ({mode})", 
                 linewidth=1.5, color='red')
        
        # Add critical points markers
        vertical_lines2 = []
        if vertical_cache:
            added_labels = set()
            dash_map = {'-': 'solid', '--': 'dash', ':': 'dot'}
            for v in vertical_cache:
                show_label = v['label'] not in added_labels
                if show_label:
                    ax2.axvline(x=v['idx'], color=v['color'], alpha=v['alpha'], linestyle=v['linestyle'], linewidth=2, label=v['label'])
                    added_labels.add(v['label'])
                else:
                    ax2.axvline(x=v['idx'], color=v['color'], alpha=v['alpha'], linestyle=v['linestyle'], linewidth=2)
                vertical_lines2.append({
                    "x": v['idx'],
                    "color": v['color'],
                    "dash": dash_map.get(v['linestyle'], 'dash'),
                    "label": v['label'] if show_label else None
                })
        
        ax2.set_xlabel("Point index in match")
        ax2.set_ylabel("Match win probability")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title(f"Counterfactual: Dataset Simulation ({mode.title()})\nmatch_id={match_id_to_plot}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
        fig2.tight_layout()
        
        fname2 = os.path.join(plot_dir, f"match_{match_id_to_plot}_{mode}.png")
        fig2.savefig(fname2, dpi=150)
        plt.close(fig2)
        print(f"[plot] Saved {mode} plot to: {fname2}")

        traces2 = [
            {"x": x, "y": dfm["prob_p1_alt"], "name": "P1 wins match (current)", "color": "blue"},
            {"x": x, "y": dfm["prob_p2_alt"], "name": "P2 wins match (current)", "color": "orange"},
            {"x": x, "y": dfm["prob_p1_lose_alt"], "name": f"P1 wins | server loses ({mode})", "color": "green", "dash": "dash"},
            {"x": x, "y": dfm["prob_p2_lose_alt"], "name": f"P2 wins | server loses ({mode})", "color": "red", "dash": "dash"},
        ]
        html2 = os.path.join(plot_dir, f"match_{match_id_to_plot}_{mode}.html")
        _save_interactive_plot(traces2, vertical_lines2, f"Match probabilities ({mode}) - {match_id_to_plot}", html2)
        print(f"[plot] Saved interactive {mode} plot to: {html2}")
    else:
        print(f"[plot] No alternative counterfactual data available for {mode} plot")
