#!/usr/bin/env python3
"""
Script per valutare il modello BDT con metriche appropriate per probabilità punto-per-punto.

Metriche implementate:
1. Match-level accuracy: valuta solo la predizione finale di ogni match
2. Calibration plot: verifica che le probabilità siano calibrate
3. Brier Score: errore quadratico medio delle probabilità
4. Time-weighted metrics: più peso ai punti critici/finali
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from pathlib import Path
import argparse
import sys


def load_preprocessed_csv(filepath):
    """Carica i dati preprocessati da CSV"""
    print(f"\n📂 Caricamento dati da: {filepath}")
    df = pd.read_csv(filepath)
    
    # Estrai match_id e point_index se disponibili
    match_ids = df['match_id'].values if 'match_id' in df.columns else None
    point_indices = df['point_index'].values if 'point_index' in df.columns else None
    
    # Rimuovi colonne non-feature
    cols_to_drop = ['match_id', 'point_index']
    if 'match_winner' in df.columns:
        y = df['match_winner'].values
        cols_to_drop.append('match_winner')
    else:
        raise ValueError("Colonna 'match_winner' non trovata nel CSV")
    
    # Rimuovi le colonne metadata
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).values
    
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  P1 wins: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  P2 wins: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
    
    return X, y, match_ids, point_indices


def match_level_accuracy(match_ids, y_true, y_prob, threshold=0.5):
    """
    Calcola l'accuracy a livello di match, considerando solo l'ultima predizione.
    
    Args:
        match_ids: ID dei match per ogni punto
        y_true: Vincitore reale (1=P1, 0=P2)
        y_prob: Probabilità predetta che P1 vinca
        threshold: Soglia per classificare (default 0.5)
    
    Returns:
        accuracy: frazione di match predetti correttamente
        results_df: DataFrame con dettagli per ogni match
    """
    print("\n" + "="*70)
    print("📊 METRICA 1: MATCH-LEVEL ACCURACY")
    print("="*70)
    print("Valuta solo l'ultima predizione di ogni match")
    
    # Raggruppa per match
    df = pd.DataFrame({
        'match_id': match_ids,
        'y_true': y_true,
        'y_prob': y_prob
    })
    
    # Per ogni match, prendi l'ultima predizione
    last_predictions = df.groupby('match_id').last().reset_index()
    
    # Calcola predizioni finali
    last_predictions['y_pred'] = (last_predictions['y_prob'] >= threshold).astype(int)
    last_predictions['correct'] = (last_predictions['y_pred'] == last_predictions['y_true']).astype(int)
    
    accuracy = last_predictions['correct'].mean()
    
    # Statistiche per vincitore
    p1_matches = last_predictions[last_predictions['y_true'] == 1]
    p2_matches = last_predictions[last_predictions['y_true'] == 0]
    
    p1_accuracy = p1_matches['correct'].mean() if len(p1_matches) > 0 else 0
    p2_accuracy = p2_matches['correct'].mean() if len(p2_matches) > 0 else 0
    
    print(f"\n✓ Accuracy complessiva: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Match totali valutati: {len(last_predictions)}")
    print(f"  Match corretti: {last_predictions['correct'].sum()}")
    print(f"  Match sbagliati: {(~last_predictions['correct'].astype(bool)).sum()}")
    print(f"\n📈 Accuracy per vincitore:")
    print(f"  P1 wins (y=1): {p1_accuracy:.4f} su {len(p1_matches)} match")
    print(f"  P2 wins (y=0): {p2_accuracy:.4f} su {len(p2_matches)} match")
    
    # Statistiche sulle probabilità finali
    print(f"\n📊 Statistiche probabilità finali:")
    print(f"  Media: {last_predictions['y_prob'].mean():.4f}")
    print(f"  Std: {last_predictions['y_prob'].std():.4f}")
    print(f"  Min: {last_predictions['y_prob'].min():.4f}")
    print(f"  Max: {last_predictions['y_prob'].max():.4f}")
    
    return accuracy, last_predictions


def calibration_plot(y_true, y_prob, n_bins=10, output_path='plots/calibration_plot.png'):
    """
    Genera un calibration plot per verificare se le probabilità sono calibrate.
    
    Args:
        y_true: Vincitore reale (1=P1, 0=P2)
        y_prob: Probabilità predetta che P1 vinca
        n_bins: Numero di bin per raggruppare le probabilità
        output_path: Path per salvare il plot
    
    Returns:
        calibration_error: Errore di calibrazione medio
    """
    print("\n" + "="*70)
    print("📊 METRICA 2: CALIBRATION PLOT")
    print("="*70)
    print("Verifica se le probabilità predette corrispondono alle frequenze osservate")
    
    # Crea i bin
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calcola frazione di vittorie per ogni bin
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    empirical_probs = []
    predicted_probs = []
    counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            empirical_probs.append(y_true[mask].mean())
            predicted_probs.append(y_prob[mask].mean())
            counts.append(mask.sum())
        else:
            empirical_probs.append(np.nan)
            predicted_probs.append(bin_centers[i])
            counts.append(0)
    
    empirical_probs = np.array(empirical_probs)
    predicted_probs = np.array(predicted_probs)
    counts = np.array(counts)
    
    # Calcola calibration error (solo per bin non vuoti)
    valid_mask = ~np.isnan(empirical_probs) & (counts > 0)
    if valid_mask.sum() > 0:
        calibration_error = np.mean(np.abs(empirical_probs[valid_mask] - predicted_probs[valid_mask]))
    else:
        calibration_error = np.nan
    
    print(f"\n✓ Expected Calibration Error (ECE): {calibration_error:.4f}")
    print(f"\n📊 Calibrazione per bin:")
    for i in range(n_bins):
        if counts[i] > 0:
            print(f"  Bin {i+1} [{bins[i]:.2f}-{bins[i+1]:.2f}]: "
                  f"Predetto={predicted_probs[i]:.3f}, "
                  f"Osservato={empirical_probs[i]:.3f}, "
                  f"Count={counts[i]}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calibration curve
    valid_idx = ~np.isnan(empirical_probs)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax1.plot(predicted_probs[valid_idx], empirical_probs[valid_idx], 
             'o-', markersize=8, linewidth=2, label='Model calibration')
    ax1.set_xlabel('Predicted Probability (P1 wins)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Empirical Frequency (P1 wins)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Calibration Plot (ECE={calibration_error:.4f})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram dei count per bin
    ax2.bar(range(n_bins), counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Probability Bin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(n_bins)], 
                         rotation=45, ha='right')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Calibration plot salvato: {output_path}")
    plt.close()
    
    return calibration_error


def brier_score_analysis(y_true, y_prob):
    """
    Analizza il Brier Score (errore quadratico medio delle probabilità).
    
    Args:
        y_true: Vincitore reale (1=P1, 0=P2)
        y_prob: Probabilità predetta che P1 vinca
    
    Returns:
        brier: Brier score
    """
    print("\n" + "="*70)
    print("📊 METRICA 3: BRIER SCORE")
    print("="*70)
    print("Misura l'errore quadratico medio delle probabilità")
    
    brier = brier_score_loss(y_true, y_prob)
    
    # Brier score per vincitore
    p1_wins = y_true == 1
    p2_wins = y_true == 0
    
    brier_p1 = brier_score_loss(y_true[p1_wins], y_prob[p1_wins]) if p1_wins.sum() > 0 else np.nan
    brier_p2 = brier_score_loss(y_true[p2_wins], y_prob[p2_wins]) if p2_wins.sum() > 0 else np.nan
    
    print(f"\n✓ Brier Score complessivo: {brier:.6f}")
    print(f"  (0 = perfetto, 1 = pessimo)")
    print(f"\n📈 Brier Score per vincitore:")
    print(f"  Match vinti da P1: {brier_p1:.6f} su {p1_wins.sum()} punti")
    print(f"  Match vinti da P2: {brier_p2:.6f} su {p2_wins.sum()} punti")
    
    # Confronto con baseline
    baseline_prob = y_true.mean()  # Sempre predire la probabilità media
    brier_baseline = brier_score_loss(y_true, np.full_like(y_prob, baseline_prob))
    brier_skill_score = 1 - (brier / brier_baseline)
    
    print(f"\n📊 Confronto con baseline:")
    print(f"  Brier Score baseline (sempre {baseline_prob:.3f}): {brier_baseline:.6f}")
    print(f"  Brier Skill Score: {brier_skill_score:.4f}")
    print(f"  (>0 = migliore del baseline, <0 = peggiore)")
    
    return brier


def time_weighted_metrics(match_ids, point_indices, y_true, y_prob, output_path='plots/time_weighted_metrics.png'):
    """
    Calcola metriche pesate temporalmente, dando più importanza ai punti finali.
    
    Args:
        match_ids: ID dei match per ogni punto
        point_indices: Indice del punto nel match
        y_true: Vincitore reale (1=P1, 0=P2)
        y_prob: Probabilità predetta che P1 vinca
        output_path: Path per salvare il plot
    
    Returns:
        metrics_by_phase: DataFrame con metriche per fase del match
    """
    print("\n" + "="*70)
    print("📊 METRICA 4: TIME-WEIGHTED METRICS")
    print("="*70)
    print("Valuta accuratezza in diverse fasi del match")
    
    df = pd.DataFrame({
        'match_id': match_ids,
        'point_index': point_indices,
        'y_true': y_true,
        'y_prob': y_prob
    })
    
    # Calcola la posizione relativa nel match (0-1)
    # Usa la posizione ordinale (0, 1, 2, ...) invece del point_index
    df['ordinal_position'] = df.groupby('match_id').cumcount()
    df['match_length'] = df.groupby('match_id')['ordinal_position'].transform('max') + 1
    df['relative_position'] = df['ordinal_position'] / (df['match_length'] - 1)
    df['relative_position'] = df['relative_position'].fillna(0)  # Per match di 1 solo punto
    
    # Dividi in fasi
    phases = [
        ('Inizio (0-25%)', 0.0, 0.25),
        ('Metà (25-50%)', 0.25, 0.50),
        ('Fine (50-75%)', 0.50, 0.75),
        ('Finale (75-100%)', 0.75, 1.0)
    ]
    
    results = []
    
    for phase_name, start, end in phases:
        mask = (df['relative_position'] >= start) & (df['relative_position'] < end)
        if mask.sum() > 0:
            phase_data = df[mask]
            y_pred = (phase_data['y_prob'] >= 0.5).astype(int)
            accuracy = (y_pred == phase_data['y_true']).mean()
            brier = brier_score_loss(phase_data['y_true'], phase_data['y_prob'])
            avg_prob = phase_data['y_prob'].mean()
            std_prob = phase_data['y_prob'].std()
            
            results.append({
                'Phase': phase_name,
                'Start': start,
                'End': end,
                'Count': mask.sum(),
                'Accuracy': accuracy,
                'Brier': brier,
                'Avg_Prob': avg_prob,
                'Std_Prob': std_prob
            })
    
    metrics_df = pd.DataFrame(results)
    
    print("\n📊 Metriche per fase del match:")
    print(metrics_df.to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy per fase
    ax1 = axes[0, 0]
    ax1.bar(range(len(metrics_df)), metrics_df['Accuracy'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Match Phase', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Match Phase', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(metrics_df)))
    ax1.set_xticklabels(metrics_df['Phase'], rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.grid(alpha=0.3, axis='y')
    for i, acc in enumerate(metrics_df['Accuracy']):
        ax1.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Brier Score per fase
    ax2 = axes[0, 1]
    ax2.bar(range(len(metrics_df)), metrics_df['Brier'], color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Match Phase', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Brier Score', fontsize=12, fontweight='bold')
    ax2.set_title('Brier Score by Match Phase (lower is better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(metrics_df)))
    ax2.set_xticklabels(metrics_df['Phase'], rotation=45, ha='right')
    ax2.grid(alpha=0.3, axis='y')
    for i, brier in enumerate(metrics_df['Brier']):
        ax2.text(i, brier + 0.005, f'{brier:.4f}', ha='center', fontweight='bold')
    
    # Numero di punti per fase
    ax3 = axes[1, 0]
    ax3.bar(range(len(metrics_df)), metrics_df['Count'], color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Match Phase', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax3.set_title('Point Distribution by Match Phase', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(metrics_df)))
    ax3.set_xticklabels(metrics_df['Phase'], rotation=45, ha='right')
    ax3.grid(alpha=0.3, axis='y')
    
    # Probabilità media e std per fase
    ax4 = axes[1, 1]
    x = range(len(metrics_df))
    ax4.errorbar(x, metrics_df['Avg_Prob'], yerr=metrics_df['Std_Prob'], 
                 fmt='o-', markersize=10, linewidth=2, capsize=5, capthick=2,
                 color='purple', ecolor='purple', alpha=0.7)
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Neutral (0.5)')
    ax4.set_xlabel('Match Phase', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Probability (P1 wins)', fontsize=12, fontweight='bold')
    ax4.set_title('Probability Evolution During Match', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(metrics_df)))
    ax4.set_xticklabels(metrics_df['Phase'], rotation=45, ha='right')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Time-weighted metrics plot salvato: {output_path}")
    plt.close()
    
    return metrics_df


def evaluate_model_proper(model_path, data_csv, output_dir='plots', test_size=0.2):
    """Funzione principale per valutare il modello con metriche appropriate"""
    print("=" * 70)
    print(f"🎾 VALUTAZIONE MODELLO TENNIS BDT - METRICHE APPROPRIATE")
    print("=" * 70)
    
    # Crea directory output se non esiste
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Carica il modello
    print(f"\n🔧 Caricamento modello: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'
    print(f"  ✓ Modello caricato con {n_features} features")
    
    # Carica i dati
    X, y, match_ids, point_indices = load_preprocessed_csv(data_csv)
    
    if match_ids is None or point_indices is None:
        print("\n❌ ERRORE: Il CSV deve contenere 'match_id' e 'point_index'")
        sys.exit(1)
    
    # Split a livello di MATCH (non di punti) per evitare che match siano divisi tra train/test
    unique_matches = np.unique(match_ids)
    unique_y = []
    for mid in unique_matches:
        unique_y.append(y[match_ids == mid][0])  # Prendi y del primo punto (è uguale per tutto il match)
    unique_y = np.array(unique_y)
    
    train_matches, test_matches = train_test_split(
        unique_matches, test_size=test_size, random_state=42, stratify=unique_y
    )
    
    # Ora seleziona i punti basandoti sui match
    train_mask = np.isin(match_ids, train_matches)
    test_mask = np.isin(match_ids, test_matches)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    match_train, match_test = match_ids[train_mask], match_ids[test_mask]
    idx_train, idx_test = point_indices[train_mask], point_indices[test_mask]
    
    print(f"\n📊 Split dataset (a livello di MATCH):")
    print(f"  Train: {len(y_train)} punti in {len(train_matches)} match")
    print(f"  Test: {len(y_test)} punti in {len(test_matches)} match")
    print(f"  Match completi garantiti in train e test")
    
    # Predizioni sul test set
    print("\n🔮 Calcolo predizioni sul test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilità P1 wins
    
    print(f"\n📊 Statistiche base predizioni:")
    print(f"  Prob media: {y_pred_proba.mean():.4f}")
    print(f"  Prob std: {y_pred_proba.std():.4f}")
    print(f"  Prob min: {y_pred_proba.min():.4f}")
    print(f"  Prob max: {y_pred_proba.max():.4f}")
    
    # 1. Match-level accuracy
    match_acc, match_results = match_level_accuracy(
        match_test, y_test, y_pred_proba,
        threshold=0.5
    )
    
    # 2. Calibration plot
    calib_error = calibration_plot(
        y_test, y_pred_proba,
        n_bins=10,
        output_path=f'{output_dir}/calibration_plot.png'
    )
    
    # 3. Brier Score
    brier = brier_score_analysis(y_test, y_pred_proba)
    
    # 4. Time-weighted metrics
    time_metrics = time_weighted_metrics(
        match_test, idx_test, y_test, y_pred_proba,
        output_path=f'{output_dir}/time_weighted_metrics.png'
    )
    
    # Summary report
    print("\n" + "=" * 70)
    print("📋 SUMMARY REPORT")
    print("=" * 70)
    print(f"Match-level Accuracy: {match_acc:.4f} ({match_acc*100:.2f}%)")
    print(f"Expected Calibration Error: {calib_error:.4f}")
    print(f"Brier Score: {brier:.6f}")
    print(f"\nAccuracy by match phase:")
    for _, row in time_metrics.iterrows():
        print(f"  {row['Phase']:20s}: {row['Accuracy']:.4f}")
    
    # Salva report testuale
    report_path = f'{output_dir}/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TENNIS BDT MODEL EVALUATION - PROPER METRICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_csv}\n")
        f.write(f"Test size: {test_size}\n\n")
        f.write(f"Match-level Accuracy: {match_acc:.4f} ({match_acc*100:.2f}%)\n")
        f.write(f"Expected Calibration Error: {calib_error:.4f}\n")
        f.write(f"Brier Score: {brier:.6f}\n\n")
        f.write("Accuracy by match phase:\n")
        for _, row in time_metrics.iterrows():
            f.write(f"  {row['Phase']:20s}: {row['Accuracy']:.4f}\n")
    
    print(f"\n✓ Report testuale salvato: {report_path}")
    print("\n" + "="*70)
    print("✅ Valutazione completata!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate tennis BDT model with proper metrics for point-by-point predictions'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model pickle file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to preprocessed data CSV with match_id and point_index')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots and reports')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    # Verifica file
    if not Path(args.model).exists():
        print(f"❌ Errore: Modello non trovato: {args.model}")
        sys.exit(1)
    
    if not Path(args.data).exists():
        print(f"❌ Errore: Dati non trovati: {args.data}")
        sys.exit(1)
    
    # Esegui valutazione
    evaluate_model_proper(args.model, args.data, args.output_dir, args.test_size)
