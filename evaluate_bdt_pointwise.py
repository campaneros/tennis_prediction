#!/usr/bin/env python3
"""
Valuta un modello BDT che predice la probabilita' di vittoria finale punto per punto.

Metriche calcolate:
- Point-level: Accuracy, AUC, Log Loss, Brier Score
- Match-level: metriche su ultima osservazione del match e su media delle probabilita'

Lo split train/test e' fatto per match per evitare leakage.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve


def load_preprocessed_csv(filepath):
    df = pd.read_csv(filepath)
    if 'match_winner' not in df.columns or 'match_id' not in df.columns:
        raise ValueError("Il CSV deve contenere le colonne 'match_id' e 'match_winner'.")

    match_ids = df['match_id'].values
    y = df['match_winner'].values
    feature_cols = [c for c in df.columns if c not in ['match_id', 'match_winner', 'point_index']]
    X = df[feature_cols].values
    return X, y, match_ids, feature_cols, df


def split_by_match(match_ids, test_size=0.2, seed=42):
    unique_matches = np.unique(match_ids)
    train_matches, test_matches = train_test_split(
        unique_matches, test_size=test_size, random_state=seed
    )
    test_mask = np.isin(match_ids, test_matches)
    return test_mask, train_matches, test_matches


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_prob)


def evaluate_point_level(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': safe_auc(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
    }
    return metrics


def summarize_match_level(df_test, prob_col='y_prob'):
    # Ordina per punto per garantire che "last" sia davvero l'ultimo punto del match.
    if 'point_index' in df_test.columns:
        df_ordered = df_test.sort_values(['match_id', 'point_index'])
    else:
        df_ordered = df_test.copy()
    # Usa l'ultima riga per match come proxy della stima finale.
    last_rows = df_ordered.groupby('match_id', sort=False).tail(1)
    last_metrics = evaluate_point_level(last_rows['match_winner'].values,
                                        last_rows[prob_col].values)

    # Media delle probabilita' per match (stima aggregata del percorso).
    mean_probs = df_test.groupby('match_id', sort=False)[prob_col].mean()
    winners = df_test.groupby('match_id', sort=False)['match_winner'].first()
    mean_metrics = evaluate_point_level(winners.values, mean_probs.values)

    return last_metrics, mean_metrics, last_rows, mean_probs, winners


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * len(title))
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    if metrics['auc'] is None:
        print("AUC       : N/A (una sola classe nel set)")
    else:
        print(f"AUC       : {metrics['auc']:.4f}")
    print(f"Log Loss  : {metrics['log_loss']:.4f}")
    print(f"Brier     : {metrics['brier']:.4f}")


def save_plot(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot salvato: {png_path}")
    print(f"Plot salvato: {pdf_path}")


def plot_roc(y_true, y_prob, output_dir, suffix):
    if len(np.unique(y_true)) < 2:
        print(f"Skip ROC ({suffix}): una sola classe nel set.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, color='darkorange')
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = f'ROC Curve ({suffix})'
    if auc_val is not None:
        title += f' - AUC {auc_val:.4f}'
    plt.title(title)
    plt.grid(alpha=0.3)
    save_plot(fig, output_dir, f'roc_curve_{suffix}')


def plot_pr(y_true, y_prob, output_dir, suffix):
    if len(np.unique(y_true)) < 2:
        print(f"Skip PR ({suffix}): una sola classe nel set.")
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, color='steelblue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title = f'Precision-Recall ({suffix})'
    if ap is not None:
        title += f' - AP {ap:.4f}'
    plt.title(title)
    plt.grid(alpha=0.3)
    save_plot(fig, output_dir, f'precision_recall_{suffix}')


def plot_calibration(y_true, y_prob, output_dir, suffix, n_bins=10):
    if len(np.unique(y_true)) < 2:
        print(f"Skip calibration ({suffix}): una sola classe nel set.")
        return
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    fig = plt.figure(figsize=(8, 6))
    plt.plot(mean_pred, frac_pos, marker='o', lw=1.5, color='seagreen', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve ({suffix})')
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot(fig, output_dir, f'calibration_{suffix}')


def plot_calibration_multi_bins(y_true, y_prob, output_dir, suffix, bins_list):
    if len(np.unique(y_true)) < 2:
        print(f"Skip calibration multi-bins ({suffix}): una sola classe nel set.")
        return
    fig = plt.figure(figsize=(8, 6))
    for n_bins in bins_list:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        plt.plot(mean_pred, frac_pos, marker='o', lw=1.0, label=f'{n_bins} bins')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Reliability Diagram ({suffix})')
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot(fig, output_dir, f'reliability_{suffix}')


def plot_prob_hist(y_prob, output_dir, suffix):
    fig = plt.figure(figsize=(8, 6))
    plt.hist(y_prob, bins=30, color='slateblue', alpha=0.8)
    plt.xlabel('Predicted Probability (P1 wins)')
    plt.ylabel('Count')
    plt.title(f'Predicted Probability Histogram ({suffix})')
    plt.grid(alpha=0.3)
    save_plot(fig, output_dir, f'prob_hist_{suffix}')


def plot_curves_by_set(df_test, output_dir, prob_col='y_prob'):
    if 'set_number' not in df_test.columns:
        print("Skip per-set curves: colonna 'set_number' non trovata.")
        return
    for set_no in sorted(df_test['set_number'].unique()):
        df_set = df_test[df_test['set_number'] == set_no]
        y_true = df_set['match_winner'].values
        y_prob = df_set[prob_col].values
        if len(y_true) < 10:
            print(f"Skip set {set_no}: pochi punti ({len(y_true)}).")
            continue
        suffix = f"set_{int(set_no)}"
        plot_roc(y_true, y_prob, output_dir, suffix)
        plot_pr(y_true, y_prob, output_dir, suffix)
        plot_calibration(y_true, y_prob, output_dir, suffix)


def evaluate_and_plot_phase(df_phase, output_dir, phase_name):
    y_true = df_phase['match_winner'].values
    y_prob = df_phase['y_prob'].values
    if len(y_true) < 10:
        print(f"Skip fase {phase_name}: pochi punti ({len(y_true)}).")
        return
    metrics = evaluate_point_level(y_true, y_prob)
    print_metrics(f"Metriche fase: {phase_name}", metrics)
    plot_roc(y_true, y_prob, output_dir, phase_name)
    plot_pr(y_true, y_prob, output_dir, phase_name)
    plot_calibration(y_true, y_prob, output_dir, phase_name)
    plot_calibration_multi_bins(y_true, y_prob, output_dir, phase_name, [5, 10, 15, 20])
    plot_prob_hist(y_prob, output_dir, phase_name)


def add_match_phase_quartiles(df_test):
    if 'point_index' in df_test.columns:
        df_ordered = df_test.sort_values(['match_id', 'point_index'])
    else:
        df_ordered = df_test.copy()
    df_ordered['point_rank'] = df_ordered.groupby('match_id').cumcount()
    df_ordered['match_points'] = df_ordered.groupby('match_id')['match_id'].transform('count')
    denom = (df_ordered['match_points'] - 1).replace(0, 1)
    df_ordered['match_pos'] = df_ordered['point_rank'] / denom
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ['q1_0_25', 'q2_25_50', 'q3_50_75', 'q4_75_100']
    df_ordered['match_phase_quartile'] = pd.cut(
        df_ordered['match_pos'], bins=bins, labels=labels, include_lowest=True
    )
    return df_ordered


def add_critical_phase_flags(df_test):
    df_out = df_test.copy()
    required = [
        'p1_match_point', 'p2_match_point',
        'p1_set_point', 'p2_set_point',
        'p1_break_point', 'p2_break_point',
    ]
    for col in required:
        if col not in df_out.columns:
            df_out[col] = 0
    df_out['is_match_point'] = (df_out['p1_match_point'] == 1) | (df_out['p2_match_point'] == 1)
    df_out['is_set_point'] = (df_out['p1_set_point'] == 1) | (df_out['p2_set_point'] == 1)
    df_out['is_break_point'] = (df_out['p1_break_point'] == 1) | (df_out['p2_break_point'] == 1)
    if 'match_criticality_score' in df_out.columns:
        threshold = df_out['match_criticality_score'].quantile(0.75)
        df_out['is_high_criticality'] = df_out['match_criticality_score'] >= threshold
    else:
        df_out['is_high_criticality'] = False
    df_out['is_critical'] = (
        df_out['is_match_point'] |
        df_out['is_set_point'] |
        df_out['is_break_point'] |
        df_out['is_high_criticality']
    )
    return df_out


def main():
    parser = argparse.ArgumentParser(description='Valuta BDT point-by-point')
    parser.add_argument('--model', type=str, default='models/tennis_bdt_male.pkl',
                        help='Path al modello addestrato')
    parser.add_argument('--data', type=str, default='data/tennis_features_preprocessed_male.csv',
                        help='CSV con feature preprocessate')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Quota di match nel test set')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed per lo split dei match')
    parser.add_argument('--save-preds', type=str, default='',
                        help='CSV di output per le predizioni point-by-point (solo test)')
    parser.add_argument('--plots-dir', type=str, default='plots_bdt_eval',
                        help='Directory dove salvare i plot')
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Modello non trovato: {args.model}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dati non trovati: {args.data}")

    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']

    X, y, match_ids, feature_cols, df = load_preprocessed_csv(args.data)
    test_mask, train_matches, test_matches = split_by_match(
        match_ids, test_size=args.test_size, seed=args.seed
    )

    X_test = X[test_mask]
    y_test = y[test_mask]

    y_prob = model.predict_proba(X_test)[:, 1]
    df_test = df.loc[test_mask].copy()
    df_test['y_prob'] = y_prob

    print("=" * 70)
    print("VALUTAZIONE BDT - PREDIZIONE PUNTO PER PUNTO")
    print("=" * 70)
    print(f"Totale match: {len(np.unique(match_ids))}")
    print(f"Match test  : {len(test_matches)}")
    print(f"Punti test  : {len(y_test)}")
    print(f"Features    : {len(feature_cols)}")

    point_metrics = evaluate_point_level(y_test, y_prob)
    print_metrics("Metriche Point-Level", point_metrics)

    last_metrics, mean_metrics, last_rows, mean_probs, winners = summarize_match_level(df_test)
    print_metrics("Metriche Match-Level (ultimo punto)", last_metrics)
    print_metrics("Metriche Match-Level (media punti)", mean_metrics)

    # Plot point-level
    plot_roc(y_test, y_prob, args.plots_dir, 'point')
    plot_pr(y_test, y_prob, args.plots_dir, 'point')
    plot_calibration(y_test, y_prob, args.plots_dir, 'point')
    plot_calibration_multi_bins(y_test, y_prob, args.plots_dir, 'point', [5, 10, 15, 20])
    plot_prob_hist(y_prob, args.plots_dir, 'point')

    # Plot match-level (ultimo punto)
    plot_roc(last_rows['match_winner'].values, last_rows['y_prob'].values, args.plots_dir, 'match_last')
    plot_pr(last_rows['match_winner'].values, last_rows['y_prob'].values, args.plots_dir, 'match_last')
    plot_calibration(last_rows['match_winner'].values, last_rows['y_prob'].values, args.plots_dir, 'match_last')
    plot_calibration_multi_bins(last_rows['match_winner'].values, last_rows['y_prob'].values,
                                args.plots_dir, 'match_last', [5, 10, 15, 20])
    plot_prob_hist(last_rows['y_prob'].values, args.plots_dir, 'match_last')

    # Plot match-level (media prob)
    plot_roc(winners.values, mean_probs.values, args.plots_dir, 'match_mean')
    plot_pr(winners.values, mean_probs.values, args.plots_dir, 'match_mean')
    plot_calibration(winners.values, mean_probs.values, args.plots_dir, 'match_mean')
    plot_calibration_multi_bins(winners.values, mean_probs.values, args.plots_dir, 'match_mean', [5, 10, 15, 20])
    plot_prob_hist(mean_probs.values, args.plots_dir, 'match_mean')

    # Curve per set (point-level), usando la feature set_number
    plot_curves_by_set(df_test, args.plots_dir, prob_col='y_prob')

    # Metriche e plot per fase: set
    if 'set_number' in df_test.columns:
        for set_no in sorted(df_test['set_number'].unique()):
            df_set = df_test[df_test['set_number'] == set_no]
            evaluate_and_plot_phase(df_set, args.plots_dir, f'phase_set_{int(set_no)}')
    else:
        print("Skip fasi per set: colonna 'set_number' non trovata.")

    # Metriche e plot per fase: quartili temporali del match
    df_phase = add_match_phase_quartiles(df_test)
    for phase_label in df_phase['match_phase_quartile'].dropna().unique():
        df_q = df_phase[df_phase['match_phase_quartile'] == phase_label]
        evaluate_and_plot_phase(df_q, args.plots_dir, f'phase_{phase_label}')

    # Metriche e plot per criticita'
    df_crit = add_critical_phase_flags(df_test)
    evaluate_and_plot_phase(df_crit[df_crit['is_critical']], args.plots_dir, 'phase_critical')
    evaluate_and_plot_phase(df_crit[~df_crit['is_critical']], args.plots_dir, 'phase_non_critical')
    evaluate_and_plot_phase(df_crit[df_crit['is_match_point']], args.plots_dir, 'phase_match_point')
    evaluate_and_plot_phase(df_crit[df_crit['is_set_point']], args.plots_dir, 'phase_set_point')
    evaluate_and_plot_phase(df_crit[df_crit['is_break_point']], args.plots_dir, 'phase_break_point')

    # Metriche e plot per tie-break
    if 'in_tiebreak' in df_test.columns:
        evaluate_and_plot_phase(df_test[df_test['in_tiebreak'] == 1], args.plots_dir, 'phase_tiebreak')
        evaluate_and_plot_phase(df_test[df_test['in_tiebreak'] == 0], args.plots_dir, 'phase_no_tiebreak')
    else:
        print("Skip fasi tie-break: colonna 'in_tiebreak' non trovata.")

    if args.save_preds:
        out_path = Path(args.save_preds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ['match_id', 'match_winner']
        if 'point_index' in df_test.columns:
            cols.append('point_index')
        df_test[cols + ['y_prob']].to_csv(out_path, index=False)
        print(f"\nPredizioni salvate in: {out_path}")


if __name__ == '__main__':
    main()
