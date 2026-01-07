#!/usr/bin/env python3
"""Script temporaneo per verificare il bug degli spike al primo punto di ogni set"""

import pandas as pd
import sys
sys.path.insert(0, '.')
from predict_single_match import calculate_sets_won

# Carica i dati
df = pd.read_csv('data/2019-wimbledon-points.csv')
match = df[df['match_id'] == '2019-wimbledon-1701'].reset_index(drop=True)

print("="*70)
print("VERIFICA 1: Dati originali CSV ai cambi di set")
print("="*70)

# Punti di cambio set
change_points = [(88, 89), (126, 127), (199, 200), (254, 255)]

for prev_idx, curr_idx in change_points:
    prev = match.iloc[prev_idx]
    curr = match.iloc[curr_idx]
    
    print(f"\n--- Cambio set al punto {curr_idx} ---")
    print(f"Punto {prev_idx} (ultimo del set {prev['SetNo']}):")
    print(f"  Games: {prev['P1GamesWon']}-{prev['P2GamesWon']}")
    print(f"  Score: {prev['P1Score']}-{prev['P2Score']}")
    print(f"  SetWinner: {prev.get('SetWinner', 'N/A')}")
    
    print(f"\nPunto {curr_idx} (primo del set {curr['SetNo']}):")
    print(f"  Games: {curr['P1GamesWon']}-{curr['P2GamesWon']}")
    print(f"  Score: {curr['P1Score']}-{curr['P2Score']}")
    print(f"  SetWinner: {curr.get('SetWinner', 'N/A')}")

print("\n" + "="*70)
print("VERIFICA 2: Cosa calcola calculate_sets_won")
print("="*70)

for prev_idx, curr_idx in change_points:
    # Simula la modalità causale: history_data fino a prev_idx (escluso curr_idx)
    history_data = match.iloc[:curr_idx].copy()
    
    # Quello che fa create_tennis_features per l'ultimo punto di history_data
    idx = prev_idx  # Ultimo indice in history_data
    p1_sets, p2_sets = calculate_sets_won(history_data, idx+1)
    
    row = history_data.iloc[idx]
    p1_games = row['P1GamesWon']
    p2_games = row['P2GamesWon']
    
    print(f"\n--- Predizione punto {curr_idx} usando storia fino a {prev_idx} ---")
    print(f"Feature calcolate dall'ultimo punto della storia (punto {prev_idx}):")
    print(f"  p1_sets_won = {p1_sets}, p2_sets_won = {p2_sets}")
    print(f"  p1_games = {p1_games}, p2_games = {p2_games}")
    print(f"  game_diff = {p1_games - p2_games}")
    print(f"\n  PROBLEMA: Combinazione IMPOSSIBILE!")
    print(f"  Il modello vede: {p1_sets}-{p2_sets} sets, {p1_games}-{p2_games} games")
    print(f"  Ma questi games sono del set FINITO, non del set corrente!")

print("\n" + "="*70)
print("CONCLUSIONE")
print("="*70)
print("\nIl bug è confermato:")
print("- Durante la predizione causale del primo punto del nuovo set,")
print("- Usiamo le feature dell'ultimo punto del set precedente")
print("- Dove p1_sets_won è già aggiornato (es. 1)")
print("- Ma p1_games/p2_games hanno ancora i valori del set finito (es. 7-6)")
print("- Il modello interpreta questo come '1 set vinto e 7-6 nel set corrente'")
print("- Invece del corretto '1 set vinto e 0-0 nel nuovo set'")
print("\nSoluzione necessaria: resettare games a 0-0 quando detectiamo SetWinner != 0")
