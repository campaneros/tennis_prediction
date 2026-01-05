#!/usr/bin/env python3
"""
Script per valutare il modello BDT:
- ROC Curve con AUC
- Confusion Matrix
- Metriche di classificazione (precision, recall, F1)

NOTA: Richiede che i dati preprocessati siano stati salvati durante il training.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    confusion_matrix, 
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import argparse
import sys
from pathlib import Path


def load_preprocessed_csv(filepath):
    """Carica i dati preprocessati da CSV"""
    print(f"\n📂 Caricamento dati da: {filepath}")
    df = pd.read_csv(filepath)
    
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
    
    return X, y


def evaluate_model(model_path, data_csv, output_dir='plots', test_size=0.2):
    """Funzione principale per valutare il modello"""
    print("=" * 70)
    print(f"🎾 VALUTAZIONE MODELLO TENNIS BDT")
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
    X, y = load_preprocessed_csv(data_csv)
    
    # Split in train/test (stesso seed del training per coerenza)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📊 Split dataset:")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Test: {len(y_test)} samples")
    
    # Predizioni sul test set
    print("\n🔮 Calcolo predizioni sul test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilità P1 wins
    y_pred = model.predict(X_test)
    
    # Statistiche base
    print(f"\n📊 Statistiche predizioni:")
    print(f"  Accuracy: {(y_pred == y_test).mean():.4f}")
    print(f"  Prob media: {y_pred_proba.mean():.4f}")
    print(f"  Prob std: {y_pred_proba.std():.4f}")
    print(f"  Prob min: {y_pred_proba.min():.4f}")
    print(f"  Prob max: {y_pred_proba.max():.4f}")


def plot_roc_curve(y_true, y_prob, output_path='plots/roc_curve.png'):
    """Genera e salva la ROC curve"""
    print("\n📊 Generazione ROC Curve...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ ROC curve salvata: {output_path}")
    print(f"  📈 AUC Score: {roc_auc:.4f}")
    plt.close()
    
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, output_path='plots/confusion_matrix.png'):
    """Genera e salva la confusion matrix"""
    print("\n📊 Generazione Confusion Matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizza per mostrare percentuali
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix con valori assoluti
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                cbar_kws={'label': 'Count'}, square=True,
                xticklabels=['P2 Wins', 'P1 Wins'],
                yticklabels=['P2 Wins', 'P1 Wins'])
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Confusion matrix con percentuali
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                cbar_kws={'label': 'Percentage'}, square=True,
                xticklabels=['P2 Wins', 'P1 Wins'],
                yticklabels=['P2 Wins', 'P1 Wins'])
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Confusion matrix salvata: {output_path}")
    plt.close()
    
    return cm


def plot_precision_recall_curve(y_true, y_prob, output_path='plots/precision_recall_curve.png'):
    """Genera e salva la Precision-Recall curve"""
    print("\n📊 Generazione Precision-Recall Curve...")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ PR curve salvata: {output_path}")
    print(f"  📈 Average Precision: {avg_precision:.4f}")
    plt.close()
    
    return avg_precision


def print_classification_report(y_true, y_pred):
    """Stampa il classification report"""
    print("\n📋 Classification Report:")
    print("=" * 60)
    report = classification_report(y_true, y_pred, 
                                   target_names=['P2 Wins', 'P1 Wins'],
                                   digits=4)
    print(report)
    print("=" * 60)


def evaluate_model(model_path, data_csv, output_dir='plots', test_size=0.2):
    """Funzione principale per valutare il modello"""
    print("=" * 70)
    print(f"🎾 VALUTAZIONE MODELLO TENNIS BDT")
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
    X, y = load_preprocessed_csv(data_csv)
    
    # Split in train/test (stesso seed del training per coerenza)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📊 Split dataset:")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Test: {len(y_test)} samples")
    
    # Predizioni sul test set
    print("\n🔮 Calcolo predizioni sul test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilità P1 wins
    y_pred = model.predict(X_test)
    
    # Statistiche base
    print(f"\n📊 Statistiche predizioni:")
    print(f"  Accuracy: {(y_pred == y_test).mean():.4f}")
    print(f"  Prob media: {y_pred_proba.mean():.4f}")
    print(f"  Prob std: {y_pred_proba.std():.4f}")
    print(f"  Prob min: {y_pred_proba.min():.4f}")
    print(f"  Prob max: {y_pred_proba.max():.4f}")
    
    # Genera i plot
    roc_auc = plot_roc_curve(y_test, y_pred_proba, 
                             output_path=f'{output_dir}/roc_curve.png')
    
    cm = plot_confusion_matrix(y_test, y_pred,
                               output_path=f'{output_dir}/confusion_matrix.png')
    
    avg_precision = plot_precision_recall_curve(y_test, y_pred_proba,
                                                output_path=f'{output_dir}/precision_recall.png')
    
    # Classification report
    print_classification_report(y_test, y_pred)
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    print(f"  AUC-ROC: {roc_auc:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Accuracy: {(y_pred == y_test).mean():.4f}")
    print(f"  Test samples: {len(y_test):,}")
    print("=" * 70)
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'accuracy': (y_pred == y_test).mean(),
        'confusion_matrix': cm
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Tennis BDT Model')
    parser.add_argument('--model', type=str, default='models/tennis_bdt_male.pkl',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/tennis_features_preprocessed_male.csv',
                       help='Path to preprocessed data CSV file')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    # Verifica che i file esistano
    if not Path(args.model).exists():
        print(f"❌ Errore: Model file non trovato: {args.model}")
        sys.exit(1)
    
    if not Path(args.data).exists():
        print(f"❌ Errore: Data file non trovato: {args.data}")
        print(f"\n💡 Suggerimento: Esegui prima il training con:")
        print(f"   ./venv/bin/python train_tennis_bdt.py")
        sys.exit(1)
    
    # Esegui valutazione
    results = evaluate_model(args.model, args.data, args.output, args.test_size)
    
    print("\n✅ Valutazione completata!")
