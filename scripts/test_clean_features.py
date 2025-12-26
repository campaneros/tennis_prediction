#!/usr/bin/env python3
"""
Test script per verificare le nuove features pulite per la rete neurale.
Questo script:
1. Carica alcuni dati di esempio
2. Costruisce le features pulite
3. Verifica che le features del tie-break siano corrette
4. Stampa alcune statistiche
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scripts.data_loader import load_points_multiple
from scripts.features import (
    add_match_labels,
    add_rolling_serve_return_features,
    add_leverage_and_momentum,
    build_clean_features_nn,
)

def test_clean_features():
    """Testa le nuove features pulite."""
    
    print("=" * 80)
    print("TEST FEATURES PULITE PER NEURAL NETWORK")
    print("=" * 80)
    
    # 1. Carica dati
    print("\n[1] Caricamento dati di test...")
    data_files = [
        'data/2019-wimbledon-matches.csv',
        'data/2019-wimbledon-points.csv',
    ]
    
    # Controlla se i file esistono
    available_files = [f for f in data_files if os.path.exists(f)]
    if not available_files:
        print("ERROR: Nessun file dati trovato!")
        return False
    
    df = load_points_multiple(available_files)
    print(f"   Caricati {len(df)} punti da {len(available_files)} file")
    
    # Filtra solo match 1701 (Wimbledon 2019 Final)
    df = df[df['match_id'].astype(str).str.endswith('-1701')]
    print(f"   Match 1701 (Wimbledon 2019 Final): {len(df)} punti")
    
    if len(df) == 0:
        print("ERROR: Match 1701 non trovato!")
        return False
    
    # 2. Prepara features
    print("\n[2] Costruzione features...")
    df = add_match_labels(df)
    df = add_rolling_serve_return_features(df, long_window=20, short_window=5)
    df = add_leverage_and_momentum(df, alpha=1.2)
    
    # 3. Costruisci features pulite
    print("\n[3] Costruzione features PULITE...")
    X, y, mask, weights, feature_names = build_clean_features_nn(df)
    
    print(f"\n   Shape: {X.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Target: {np.sum(y)} positivi su {len(y)} ({100*np.sum(y)/len(y):.1f}%)")
    
    # 4. Verifica features del tie-break
    print("\n[4] Verifica features TIE-BREAK...")
    df_masked = df[mask].copy()
    
    # Aggiungi colonne delle features al dataframe per analisi
    for i, col in enumerate(feature_names):
        df_masked[f'feat_{col}'] = X[:, i]
    
    # Trova punti in tie-break
    tb_mask = df_masked['feat_is_tiebreak'] == 1.0
    n_tb_points = tb_mask.sum()
    print(f"   Punti in tie-break: {n_tb_points}")
    
    if n_tb_points > 0:
        tb_df = df_masked[tb_mask].copy()
        
        # Verifica tie-break decisivo
        decisive_tb = tb_df['feat_is_decisive_tiebreak'] == 1.0
        print(f"   Tie-break decisivi: {decisive_tb.sum()}")
        
        # Mostra alcuni esempi di punteggi nel tie-break
        print("\n   Esempi di punteggi nel tie-break:")
        sample_cols = ['P1Score', 'P2Score', 'feat_tb_p1_points', 'feat_tb_p2_points',
                      'feat_tb_p1_needs_to_win', 'feat_tb_p2_needs_to_win']
        print(tb_df[sample_cols].head(10).to_string())
        
        # Verifica logica: se P1 ha 7+ punti e vantaggio di 2, needs_to_win dovrebbe essere 0
        p1_winning = (tb_df['feat_tb_p1_points'] >= 7) & \
                     (tb_df['feat_tb_p1_points'] >= tb_df['feat_tb_p2_points'] + 2)
        if p1_winning.any():
            print(f"\n   P1 winning position: {p1_winning.sum()} punti")
            winning_needs = tb_df[p1_winning]['feat_tb_p1_needs_to_win'].unique()
            print(f"   P1 needs_to_win in winning position: {winning_needs}")
            if not all(n == 0 for n in winning_needs):
                print("   WARNING: P1 needs_to_win dovrebbe essere 0!")
    
    # 5. Verifica features del formato match
    print("\n[5] Verifica MATCH FORMAT...")
    print(f"   Best-of-5: {df_masked['feat_is_best_of_5'].iloc[0]}")
    print(f"   Sets to win: {df_masked['feat_sets_to_win'].iloc[0]}")
    
    # 6. Verifica evoluzione del punteggio
    print("\n[6] Verifica EVOLUZIONE PUNTEGGIO...")
    # Prendi ultimi 20 punti
    last_points = df_masked.tail(20)
    
    print("\n   Ultimi 20 punti del match:")
    score_cols = ['feat_P1_sets', 'feat_P2_sets', 'feat_P1_games', 'feat_P2_games',
                  'feat_P1_points', 'feat_P2_points']
    print(last_points[score_cols].to_string())
    
    # 7. Verifica features performance
    print("\n[7] Verifica FEATURES PERFORMANCE...")
    perf_cols = ['feat_P_srv_win_long', 'feat_P_srv_lose_long',
                 'feat_p1_momentum', 'feat_p2_momentum']
    print("\n   Statistiche performance features:")
    for col in perf_cols:
        vals = df_masked[col].values
        print(f"   {col:25s}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_clean_features()
    sys.exit(0 if success else 1)
