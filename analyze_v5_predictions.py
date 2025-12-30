#!/usr/bin/env python3
"""
Analizza le predizioni del modello v5 per capire perché la probabilità finale è bassa.
"""

import pandas as pd
import numpy as np

# Load v5 predictions
df = pd.read_csv('plots_v5/match_2019-wimbledon-1701_probabilities.csv')

print("=" * 80)
print("ANALISI DETTAGLIATA MODELLO V5")
print("=" * 80)
print()

print("STATISTICHE GLOBALI:")
print(f"  Punti totali: {len(df)}")
print(f"  Range: [{df['prob_p1'].min():.4f}, {df['prob_p1'].max():.4f}]")
print(f"  Mean: {df['prob_p1'].mean():.4f}")
print(f"  Std: {df['prob_p1'].std():.4f}")
print()

# Analyze by set
print("ANALISI PER SET:")
for set_num in sorted(df['SetNo'].unique()):
    df_set = df[df['SetNo'] == set_num]
    print(f"\nSet {int(set_num)}:")
    print(f"  Punti: {len(df_set)}")
    print(f"  Range: [{df_set['prob_p1'].min():.4f}, {df_set['prob_p1'].max():.4f}]")
    print(f"  Mean: {df_set['prob_p1'].mean():.4f}")
    print(f"  Std: {df_set['prob_p1'].std():.4f}")
    
    # Final score of the set
    last_row = df_set.iloc[-1]
    p1_games = last_row['P1GamesWon']
    p2_games = last_row['P2GamesWon']
    winner = "P1 (Djokovic)" if p1_games > p2_games else "P2 (Federer)"
    print(f"  Risultato: {int(p1_games)}-{int(p2_games)} (vince {winner})")
    print(f"  Prob finale set: {last_row['prob_p1']:.4f}")

print()
print("=" * 80)

# Check correlation with match state
# When P1 is ahead in sets, prob should increase
print("\nCORRELAZIONE CON STATO MATCH:")

# Create features: sets won, games in current set
df['P1SetsAhead'] = df['P1SetsWon'] - df['P2SetsWon']
df['P1GamesAhead'] = df['P1GamesWon'] - df['P2GamesWon']

# Correlation
corr_sets = df['prob_p1'].corr(df['P1SetsAhead'])
corr_games = df['prob_p1'].corr(df['P1GamesAhead'])

print(f"  Correlazione prob_p1 vs SetsAhead: {corr_sets:.4f}")
print(f"  Correlazione prob_p1 vs GamesAhead: {corr_games:.4f}")

if corr_sets < 0:
    print("  ⚠ PROBLEMA: Correlazione negativa con i set! (dovrebbe essere positiva)")
if corr_games < 0:
    print("  ⚠ PROBLEMA: Correlazione negativa con i game! (dovrebbe essere positiva)")

print()

# Group by sets ahead
print("PROB MEDIA PER SITUAZIONE SET:")
for ahead in sorted(df['P1SetsAhead'].unique()):
    df_ahead = df[df['P1SetsAhead'] == ahead]
    print(f"  P1 {int(ahead):+d} set: {df_ahead['prob_p1'].mean():.4f} (n={len(df_ahead)})")

print()

# Check final situation
print("SITUAZIONE FINALE:")
last = df.iloc[-1]
print(f"  Set score: {int(last['P1SetsWon'])}-{int(last['P2SetsWon'])}")
print(f"  Game score: {int(last['P1GamesWon'])}-{int(last['P2GamesWon'])}")
print(f"  Point score: {int(last['P1Score'])}-{int(last['P2Score'])}")
print(f"  Prob P1: {last['prob_p1']:.4f}")
print(f"  VINCITORE REALE: Djokovic (P1) con 3-2 set")
print()

if last['prob_p1'] < 0.70:
    print("⚠ PROBLEMA: La probabilità finale è bassa (<0.70) nonostante P1 abbia vinto!")
    print("   Possibili cause:")
    print("   1. Il modello non ha imparato bene la gerarchia del tennis")
    print("   2. Il fine-tuning è troppo breve (early stopping epoch 8)")
    print("   3. I dati sintetici hanno ancora qualche bias nascosto")

print("=" * 80)
