#!/usr/bin/env python3
"""
Script per fare predizioni punto per punto su una singola partita di tennis.
Genera plot interattivi che mostrano l'evoluzione della probabilità di vittoria.
"""

import argparse
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


def get_match_gender(match_id):
    """
    Determina il genere del match dall'ID.
    Match ID < 2000 = maschile
    Match ID >= 2000 = femminile
    """
    # Estrai il numero dal match_id (formato: YYYY-tournament-NNNN)
    parts = match_id.split('-')
    if len(parts) >= 3:
        match_num = int(parts[-1])
        return 'male' if match_num < 2000 else 'female'
    return 'male'  # default


def parse_point_score(p1_score, p2_score):
    """
    Converte il punteggio del punto in valori numerici.
    Nel tie-break gestisce numeri diretti (0, 1, 2, ...).
    """
    score_map = {
        '0': 0,
        '15': 1,
        '30': 2,
        '40': 3,
        'AD': 4,
        'A': 4
    }
    
    p1_str = str(p1_score).strip()
    p2_str = str(p2_score).strip()
    
    # Prova prima la mappa standard
    if p1_str in score_map:
        p1_val = score_map[p1_str]
    else:
        # Nel tie-break sono numeri diretti
        try:
            p1_val = int(p1_str)
        except:
            p1_val = 0
    
    if p2_str in score_map:
        p2_val = score_map[p2_str]
    else:
        # Nel tie-break sono numeri diretti
        try:
            p2_val = int(p2_str)
        except:
            p2_val = 0
    
    return p1_val, p2_val


def calculate_sets_won(match_data, current_idx):
    """
    Calcola il numero di set vinti da ciascun giocatore fino al punto corrente.
    """
    if current_idx == 0:
        return 0, 0
    
    prev_rows = match_data.iloc[:current_idx]
    
    p1_sets = 0
    p2_sets = 0
    
    completed_sets_data = prev_rows[prev_rows['SetWinner'] != 0]
    
    if not completed_sets_data.empty:
        for set_no in completed_sets_data['SetNo'].unique():
            set_data = completed_sets_data[completed_sets_data['SetNo'] == set_no]
            if not set_data.empty:
                winner = set_data.iloc[-1]['SetWinner']
                if winner == 1:
                    p1_sets += 1
                elif winner == 2:
                    p2_sets += 1
    
    return p1_sets, p2_sets


def create_tennis_features(df, lstm_probs_df=None, n_features=None):
    """
    Crea le stesse feature usate durante il training.
    Deve essere identico alla funzione in train_tennis_bdt.py
    
    Args:
        df: DataFrame con i dati del match
        lstm_probs_df: DataFrame opzionale con le probabilità LSTM per punto
        n_features: Numero di feature attese dal modello (per backward compatibility)
                   Se None, genera tutte le feature disponibili
                   Se 44, salta le feature LSTM (vecchio modello senza set_diff/sets_won)
                   Se 47, include LSTM ma senza set_diff/sets_won
                   Se 48, nuovo modello senza LSTM (con set_diff/sets_won/tiebreak_point_diff)
                   Se 51, nuovo modello completo (con tutto)
    """
    features = []
    
    # Merge LSTM probabilities se fornite E se il modello le supporta
    # Include LSTM se: n_features is None, oppure n_features in [47, 51]
    include_lstm = lstm_probs_df is not None and (n_features is None or n_features in [47, 51])
    
    if include_lstm:
        # Converti PointNumber in int per entrambi i dataframe per evitare errori di merge
        # Gestisci il caso speciale '0X' che indica il primo punto
        df['PointNumber'] = df['PointNumber'].replace('0X', '0')
        df['PointNumber'] = pd.to_numeric(df['PointNumber'], errors='coerce').fillna(0).astype(int)
        lstm_probs_df['PointNumber'] = pd.to_numeric(lstm_probs_df['PointNumber'], errors='coerce').fillna(0).astype(int)
        
        # Merge basato su match_id, SetNo, GameNo, PointNumber
        df = df.merge(
            lstm_probs_df[['match_id', 'SetNo', 'GameNo', 'PointNumber', 'p1_point_prob']],
            on=['match_id', 'SetNo', 'GameNo', 'PointNumber'],
            how='left'
        )
        # Riempi valori mancanti con 0.5 (neutro)
        df['p1_point_prob'] = df['p1_point_prob'].fillna(0.5)
    
    # Riempi i valori NaN con 0 per le colonne numeriche
    numeric_cols = ['P1Momentum', 'P2Momentum', 'P1BreakPoint', 'P2BreakPoint', 
                    'P1PointsWon', 'P2PointsWon', 'P1Ace', 'P2Ace', 'P1Winner', 
                    'P2Winner', 'P1UnfErr', 'P2UnfErr', 'P1DoubleFault', 'P2DoubleFault',
                    'P1BreakPointWon', 'P2BreakPointWon']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    for idx, row in df.iterrows():
        # Feature base: punteggio set
        p1_sets_won, p2_sets_won = calculate_sets_won(df, idx)
        
        # Punteggio game nel set corrente
        p1_games = row['P1GamesWon']
        p2_games = row['P2GamesWon']
        
        # Punteggio point nel game corrente
        p1_point_val, p2_point_val = parse_point_score(row['P1Score'], row['P2Score'])
        
        # Chi sta servendo
        server = row['PointServer']
        p1_serving = 1 if server == 1 else 0
        p2_serving = 1 if server == 2 else 0
        
        # Break point situation
        p1_break_point = row.get('P1BreakPoint', 0)
        p2_break_point = row.get('P2BreakPoint', 0)
        
        # Statistiche cumulative
        p1_points_won = row.get('P1PointsWon', 0)
        p2_points_won = row.get('P2PointsWon', 0)
        total_points = p1_points_won + p2_points_won if (p1_points_won + p2_points_won) > 0 else 1
        
        p1_ace = row.get('P1Ace', 0)
        p2_ace = row.get('P2Ace', 0)
        p1_winner = row.get('P1Winner', 0)
        p2_winner = row.get('P2Winner', 0)
        p1_unforced = row.get('P1UnfErr', 0)
        p2_unforced = row.get('P2UnfErr', 0)
        p1_double_fault = row.get('P1DoubleFault', 0)
        p2_double_fault = row.get('P2DoubleFault', 0)
        
        # Momentum
        p1_momentum = row.get('P1Momentum', 0)
        p2_momentum = row.get('P2Momentum', 0)
        
        # LSTM Point Probability - Momentum del match dal modello LSTM (solo se richiesto)
        if include_lstm:
            lstm_point_prob = row.get('p1_point_prob', 0.5)  # Default neutro
            lstm_momentum_centered = lstm_point_prob - 0.5  # Centra tra -0.5 e +0.5
            lstm_momentum_raw = lstm_point_prob  # Mantieni anche il valore raw
        
        # Break point statistics
        p1_bp_won = row.get('P1BreakPointWon', 0)
        p2_bp_won = row.get('P2BreakPointWon', 0)
        
        # Feature derivate: differenze nel punteggio
        set_diff = p1_sets_won - p2_sets_won
        game_diff = p1_games - p2_games
        point_diff = p1_point_val - p2_point_val
        points_won_diff = p1_points_won - p2_points_won
        momentum_diff = p1_momentum - p2_momentum
        
        # Percentuale punti vinti
        p1_points_pct = p1_points_won / total_points
        p2_points_pct = p2_points_won / total_points
        
        # Percentuale statistiche
        total_ace = p1_ace + p2_ace + 1
        total_winner = p1_winner + p2_winner + 1
        total_unforced = p1_unforced + p2_unforced + 1
        
        p1_ace_pct = p1_ace / total_ace
        p2_ace_pct = p2_ace / total_ace
        p1_winner_pct = p1_winner / total_winner
        p2_winner_pct = p2_winner / total_winner
        p1_unforced_pct = p1_unforced / total_unforced
        p2_unforced_pct = p2_unforced / total_unforced
        
        # Feature combinate: situazione critica
        decisive_set = 1 if p1_sets_won == 2 and p2_sets_won == 2 else 0
        close_to_winning_p1 = 1 if (p1_sets_won == 2 and p2_sets_won < 2 and p1_games >= 5) else 0
        close_to_winning_p2 = 1 if (p2_sets_won == 2 and p1_sets_won < 2 and p2_games >= 5) else 0
        
        # FEATURE CHIAVE: Distanza dalla vittoria finale
        p1_sets_to_win_match = max(0, 3 - p1_sets_won)
        p2_sets_to_win_match = max(0, 3 - p2_sets_won)
        
        p1_games_to_win_set = max(0, 6 - p1_games) if p1_games < 6 else 0
        p2_games_to_win_set = max(0, 6 - p2_games) if p2_games < 6 else 0
        
        p1_points_to_win_game = max(0, 4 - p1_point_val) if p1_point_val < 4 else 0
        p2_points_to_win_game = max(0, 4 - p2_point_val) if p2_point_val < 4 else 0
        
        # Match point detection MIGLIORATO
        # Un match point esiste quando vincere questo punto porta alla vittoria del match
        p1_match_point = 0
        p2_match_point = 0
        
        # TIE-BREAK logic
        # Nel 5° set (2-2): tie-break solo a 12-12 (regola Wimbledon dal 2019)
        # Negli altri set: tie-break a 6-6
        in_fifth_set = (p1_sets_won == 2 and p2_sets_won == 2)
        in_tiebreak = ((p1_games == 6 and p2_games == 6 and not in_fifth_set) or 
                       (p1_games == 12 and p2_games == 12 and in_fifth_set))
        
        # Feature TIE-BREAK: vantaggio nel tie-break
        tiebreak_point_diff = 0
        if in_tiebreak:
            tiebreak_point_diff = p1_point_val - p2_point_val
        
        # Verifica se un giocatore può vincere il match vincendo questo punto
        # Condizione: avere già 2 set vinti
        p1_can_win_match = (p1_sets_won == 2)  # P1 ha 2 set
        p2_can_win_match = (p2_sets_won == 2)  # P2 ha 2 set
        
        if p1_can_win_match:
            if in_tiebreak:
                # Nel tie-break (solo set 1-4): match point se ho almeno 6 punti E almeno 1 punto di vantaggio
                if p1_point_val >= 6 and p1_point_val > p2_point_val:
                    p1_match_point = 1
            else:
                # Game normale: match point se questo punto mi fa vincere il game che mi fa vincere il set
                # Nel 5° set: devo essere avanti di almeno 1 game E avere 40+ per vincere
                if in_fifth_set:
                    # Nel 5° set serve vantaggio di almeno 1 game
                    if p1_games >= 5 and p1_games > p2_games:
                        if p1_point_val >= 3 and p1_point_val > p2_point_val:
                            p1_match_point = 1
                else:
                    # Set normali (1-4): posso vincere a 6
                    if p1_games >= 5 and p1_games > p2_games:
                        if p1_point_val >= 3 and p1_point_val > p2_point_val:
                            p1_match_point = 1
        
        if p2_can_win_match:
            if in_tiebreak:
                # Nel tie-break (solo set 1-4): match point se ho almeno 6 punti E almeno 1 punto di vantaggio
                if p2_point_val >= 6 and p2_point_val > p1_point_val:
                    p2_match_point = 1
            else:
                # Game normale: match point se questo punto mi fa vincere il game che mi fa vincere il set
                # Nel 5° set: devo essere avanti di almeno 1 game E avere 40+ per vincere
                if in_fifth_set:
                    # Nel 5° set serve vantaggio di almeno 1 game
                    if p2_games >= 5 and p2_games > p1_games:
                        if p2_point_val >= 3 and p2_point_val > p1_point_val:
                            p2_match_point = 1
                else:
                    # Set normali (1-4): posso vincere a 6
                    if p2_games >= 5 and p2_games > p1_games:
                        if p2_point_val >= 3 and p2_point_val > p1_point_val:
                            p2_match_point = 1
        
        # Set point detection
        p1_set_point = 0
        p2_set_point = 0
        if p1_games >= 5 and p1_games > p2_games and p1_point_val == 3 and p2_point_val < 3:
            p1_set_point = 1
        if p2_games >= 5 and p2_games > p1_games and p2_point_val == 3 and p1_point_val < 3:
            p2_set_point = 1
        
        game_criticality = 0
        if p1_games >= 5 or p2_games >= 5:
            game_criticality = 1
        if p1_games == p2_games and p1_games >= 5:
            game_criticality = 2
        
        # Info sul set corrente
        set_number = row['SetNo']
        
        # FEATURE DI CRITICITÀ
        point_criticality = min(p1_point_val, p2_point_val)
        if p1_point_val == 3 or p2_point_val == 3:
            point_criticality += 2
        if p1_point_val >= 3 and p2_point_val >= 3:
            point_criticality += 3
        
        p1_distance_to_set = max(0, 6 - p1_games) if p1_games < 6 else 0
        p2_distance_to_set = max(0, 6 - p2_games) if p2_games < 6 else 0
        min_distance_to_set = min(p1_distance_to_set, p2_distance_to_set)
        
        p1_must_win_set = 1 if (p2_sets_won == 2 and p1_sets_won < 2) else 0
        p2_must_win_set = 1 if (p1_sets_won == 2 and p2_sets_won < 2) else 0
        
        game_pressure = 0
        if abs(p1_games - p2_games) <= 1 and (p1_games >= 4 or p2_games >= 4):
            game_pressure = 1
        if p1_games >= 5 and p2_games >= 5:
            game_pressure = 2
        
        p1_can_win_match_this_set = 1 if p1_sets_won == 2 else 0
        p2_can_win_match_this_set = 1 if p2_sets_won == 2 else 0
        match_on_the_line = p1_can_win_match_this_set or p2_can_win_match_this_set
        
        momentum_shift_potential = 0
        if p1_break_point or p2_break_point:
            momentum_shift_potential += 2
        if p1_set_point or p2_set_point:
            momentum_shift_potential += 3
        if p1_match_point or p2_match_point:
            momentum_shift_potential += 5
        
        match_criticality_score = 0
        match_criticality_score += set_number * 0.5
        match_criticality_score += game_criticality * 2
        match_criticality_score += point_criticality
        match_criticality_score += (1.0 / (min_distance_to_set + 1)) * 3
        match_criticality_score += (p1_must_win_set + p2_must_win_set) * 5
        match_criticality_score += game_pressure * 2
        match_criticality_score += match_on_the_line * 10
        
        # Power features
        p1_has_match_point_advantage = p1_match_point * 80  # Peso 80x
        p2_has_match_point_advantage = p2_match_point * 80  # Peso 80x
        p1_has_set_point_advantage = p1_set_point * 25  # Peso 25x
        p2_has_set_point_advantage = p2_set_point * 25  # Peso 25x
        
        p1_advantage_score = 0
        p2_advantage_score = 0
        
        if p1_match_point:
            p1_advantage_score += 100  # Match point = +100
        if p2_match_point:
            p2_advantage_score += 100
        if p1_set_point:
            p1_advantage_score += 40  # Set point = +40
        if p2_set_point:
            p2_advantage_score += 40
        if p1_break_point:
            p2_advantage_score += 3
        if p2_break_point:
            p1_advantage_score += 3
        if p1_serving:
            p1_advantage_score += 1
        if p2_serving:
            p2_advantage_score += 1
        
        advantage_diff = p1_advantage_score - p2_advantage_score
        
        # Ultra feature: match situation dominance
        match_situation_score = 0.0
        
        # MATCH POINT = 200
        if p1_match_point:
            match_situation_score += 200
        if p2_match_point:
            match_situation_score -= 200
        
        if p1_set_point and p1_can_win_match_this_set:
            match_situation_score += 100
        elif p1_set_point:
            match_situation_score += 40
        
        if p2_set_point and p2_can_win_match_this_set:
            match_situation_score -= 100
        elif p2_set_point:
            match_situation_score -= 40
        
        if match_on_the_line:
            if p1_break_point:
                match_situation_score -= 15
            if p2_break_point:
                match_situation_score += 15
        
        # TIE-BREAK logic
        in_tiebreak = (p1_games == 6 and p2_games == 6)
        if in_tiebreak and match_on_the_line:
            match_situation_score += point_diff * 30
            if p1_point_val >= 6:
                match_situation_score += 50
            if p2_point_val >= 6:
                match_situation_score -= 50
        elif in_tiebreak:
            match_situation_score += point_diff * 15
            if p1_point_val >= 6:
                match_situation_score += 25
            if p2_point_val >= 6:
                match_situation_score -= 25
        
        if match_on_the_line and not in_tiebreak:
            match_situation_score += game_diff * 8
        
        # Costruisci il feature vector
        feature_vec = [
            set_number, p1_sets_won, p2_sets_won, set_diff,
            p1_games, p2_games, game_diff,
            p1_games_to_win_set, p2_games_to_win_set,
            p1_point_val, p2_point_val, point_diff,
            p1_points_to_win_game, p2_points_to_win_game,
            p1_serving, p2_serving,
            p1_break_point, p2_break_point,
            decisive_set, close_to_winning_p1, close_to_winning_p2,
            p1_match_point, p2_match_point,
            p1_set_point, p2_set_point,
            game_criticality,
            point_criticality, p1_distance_to_set, p2_distance_to_set,
            min_distance_to_set, p1_must_win_set, p2_must_win_set,
            game_pressure, p1_can_win_match_this_set, p2_can_win_match_this_set,
            match_on_the_line, momentum_shift_potential, match_criticality_score,
            p1_has_match_point_advantage, p2_has_match_point_advantage,
            p1_has_set_point_advantage, p2_has_set_point_advantage,
            p1_advantage_score, p2_advantage_score, advantage_diff,
            match_situation_score,
            in_tiebreak,
            tiebreak_point_diff,  # Nuovo: vantaggio punti nel tie-break
        ]
        
        # LSTM momentum features (solo se richiesto)
        if include_lstm:
            feature_vec.extend([
                lstm_point_prob,
                lstm_momentum_centered,
                lstm_momentum_raw,
            ])
        
        features.append(feature_vec)
    
    return np.array(features)


def load_model(model_path):
    """
    Carica il modello salvato.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data['feature_names']


def predict_match(match_data, model, lstm_probs_df=None):
    """
    Fa predizioni punto per punto su una partita.
    
    Args:
        match_data: DataFrame con i dati del match
        model: Modello addestrato
        lstm_probs_df: DataFrame con probabilità LSTM
    """
    # Determina quante feature si aspetta il modello
    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    
    # Crea features (con LSTM probs se fornite E se il modello le supporta)
    X = create_tennis_features(match_data, lstm_probs_df, n_features=n_features)
    
    # Predici probabilità
    probabilities = model.predict_proba(X)
    
    # probabilities[:, 1] è la probabilità che P1 vinca
    p1_probs = probabilities[:, 1]
    p2_probs = 1 - p1_probs
    
    # CORREZIONE: quando il match finisce, forza probabilità a 1.0 per il vincitore
    # Verifica per ogni punto se il match è già finito (guardando i game vinti dopo il punto)
    for i in range(len(match_data)):
        row = match_data.iloc[i]
        
        # Se questo punto porta un giocatore a 13 o più game nel 5° set (2-2),
        # oppure a 7+ game in un tie-break, oppure a vincere un set che porta a 3 set vinti,
        # il match è finito
        p1_games = row['P1GamesWon']
        p2_games = row['P2GamesWon']
        p1_sets, p2_sets = calculate_sets_won(match_data, i)
        
        # Situazione 5° set: match finito se qualcuno arriva a 13+ (o vantaggio di 2 su 12+)
        in_fifth = (p1_sets == 2 and p2_sets == 2)
        if in_fifth and (p1_games >= 13 or p2_games >= 13):
            if p1_games > p2_games:
                p1_probs[i] = 1.0
                p2_probs[i] = 0.0
            else:
                p1_probs[i] = 0.0
                p2_probs[i] = 1.0
    
    return p1_probs, p2_probs


def get_score_string(row, p1_sets, p2_sets):
    """
    Crea una stringa che descrive il punteggio attuale.
    """
    score = f"Sets: {p1_sets}-{p2_sets} | Games: {row['P1GamesWon']}-{row['P2GamesWon']} | Points: {row['P1Score']}-{row['P2Score']}"
    return score


def plot_probabilities_matplotlib(match_data, p1_probs, p2_probs, output_path):
    """
    Crea un plot con matplotlib che mostra le probabilità nel tempo.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    points = np.arange(len(p1_probs))
    
    # Plot delle probabilità
    ax.plot(points, p1_probs, label='P1 Probability', color='blue', linewidth=2)
    ax.plot(points, p2_probs, label='P2 Probability', color='red', linewidth=2)
    
    # Linea al 50%
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Evidenzia i cambi di set
    set_changes = match_data[match_data['SetWinner'] != 0].index
    for idx in set_changes:
        if idx < len(points):
            ax.axvline(x=idx, color='green', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Point Number', fontsize=12)
    ax.set_ylabel('Win Probability', fontsize=12)
    ax.set_title(f'Match Win Probability Evolution - {match_data["match_id"].iloc[0]}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot salvato in: {output_path}")
    plt.close()


def plot_probabilities_plotly(match_data, p1_probs, p2_probs, output_path):
    """
    Crea un plot interattivo con Plotly stile seconda foto con molti dettagli.
    """
    points = np.arange(len(p1_probs))
    
    # Calcola i set vinti per ogni punto
    p1_sets_list = []
    p2_sets_list = []
    for idx in range(len(match_data)):
        p1_sets, p2_sets = calculate_sets_won(match_data, idx)
        p1_sets_list.append(p1_sets)
        p2_sets_list.append(p2_sets)
    
    # Crea hover text
    hover_texts = []
    for idx, row in match_data.iterrows():
        i = idx if isinstance(idx, int) else list(match_data.index).index(idx)
        p1_sets = p1_sets_list[i]
        p2_sets = p2_sets_list[i]
        score_str = get_score_string(row, p1_sets, p2_sets)
        hover_text = f"{score_str}<br>P1: {p1_probs[i]:.1%}<br>P2: {p2_probs[i]:.1%}"
        hover_texts.append(hover_text)
    
    # Crea il plot
    fig = go.Figure()
    
    # Linee tratteggiate per mostrare gli esiti alternativi punto per punto
    # P1 vince il punto corrente (tratteggiata blu)
    p1_wins_point = []
    for idx, row in match_data.iterrows():
        i = idx if isinstance(idx, int) else list(match_data.index).index(idx)
        winner = row['PointWinner']
        p1_wins_point.append(1 if winner == 1 else 0)
    
    # Crea linee tratteggiate per P1 e P2 quando vincono punti
    for i in range(len(points) - 1):
        if p1_wins_point[i] == 1:
            # P1 vince questo punto - linea tratteggiata blu
            fig.add_trace(go.Scatter(
                x=[points[i], points[i+1]],
                y=[p1_probs[i], p1_probs[i+1]],
                mode='lines',
                line=dict(color='blue', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            # P2 vince questo punto - linea tratteggiata rossa
            fig.add_trace(go.Scatter(
                x=[points[i], points[i+1]],
                y=[p2_probs[i], p2_probs[i+1]],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Linea principale P1 (match win)
    fig.add_trace(go.Scatter(
        x=points,
        y=p1_probs,
        mode='lines',
        name='P1 wins match (current)',
        line=dict(color='blue', width=2),
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    # Linea principale P2 (match win)
    fig.add_trace(go.Scatter(
        x=points,
        y=p2_probs,
        mode='lines',
        name='P2 wins match (current)',
        line=dict(color='red', width=2),
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    # Evidenzia punti critici - Match points (con logica corretta per 5° set)
    for idx, row in match_data.iterrows():
        i = idx if isinstance(idx, int) else list(match_data.index).index(idx)
        p1_sets = p1_sets_list[i]
        p2_sets = p2_sets_list[i]
        p1_games = row['P1GamesWon']
        p2_games = row['P2GamesWon']
        
        # Controlla se siamo nel 5° set (no tie-break)
        in_fifth_set = (p1_sets == 2 and p2_sets == 2)
        in_tiebreak = (p1_games == 6 and p2_games == 6 and not in_fifth_set)
        
        p1_score = str(row['P1Score'])
        p2_score = str(row['P2Score'])
        p1_point_val, p2_point_val = parse_point_score(p1_score, p2_score)
        
        # Match point per P1 (solo se ha 2 set vinti)
        if p1_sets == 2:
            if in_tiebreak:
                # Tie-break: match point se almeno 6 punti e almeno 1 di vantaggio
                if p1_point_val >= 6 and p1_point_val > p2_point_val:
                    fig.add_vline(x=i, line_dash="dot", line_color="darkblue", opacity=0.6, line_width=1)
            else:
                # Game normale - controlla 5° set
                if in_fifth_set:
                    # 5° set: serve vantaggio di almeno 1 game
                    if p1_games >= 5 and p1_games > p2_games and p1_point_val >= 3 and p1_point_val > p2_point_val:
                        fig.add_vline(x=i, line_dash="dot", line_color="darkblue", opacity=0.6, line_width=1)
                else:
                    # Set normali 1-4
                    if p1_games >= 5 and p1_games > p2_games and p1_point_val >= 3 and p1_point_val > p2_point_val:
                        fig.add_vline(x=i, line_dash="dot", line_color="darkblue", opacity=0.6, line_width=1)
        
        # Match point per P2 (solo se ha 2 set vinti)
        if p2_sets == 2:
            if in_tiebreak:
                # Tie-break: match point se almeno 6 punti e almeno 1 di vantaggio
                if p2_point_val >= 6 and p2_point_val > p1_point_val:
                    fig.add_vline(x=i, line_dash="dot", line_color="darkred", opacity=0.6, line_width=1)
            else:
                # Game normale - controlla 5° set
                if in_fifth_set:
                    # 5° set: serve vantaggio di almeno 1 game
                    if p2_games >= 5 and p2_games > p1_games and p2_point_val >= 3 and p2_point_val > p1_point_val:
                        fig.add_vline(x=i, line_dash="dot", line_color="darkred", opacity=0.6, line_width=1)
                else:
                    # Set normali 1-4
                    if p2_games >= 5 and p2_games > p1_games and p2_point_val >= 3 and p2_point_val > p1_point_val:
                        fig.add_vline(x=i, line_dash="dot", line_color="darkred", opacity=0.6, line_width=1)
    
    # Linea al 50%
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Evidenzia i cambi di set con linee solide
    set_changes = match_data[match_data['SetWinner'] != 0].index
    for idx in set_changes:
        i = idx if isinstance(idx, int) else list(match_data.index).index(idx)
        if i < len(points):
            fig.add_vline(x=i, line_dash="solid", line_color="brown", opacity=0.4, line_width=2)
    
    # Break points
    for idx, row in match_data.iterrows():
        i = idx if isinstance(idx, int) else list(match_data.index).index(idx)
        if row.get('P1BreakPoint', 0) > 0:
            fig.add_vline(x=i, line_dash="dot", line_color="orange", opacity=0.3, line_width=0.5)
        if row.get('P2BreakPoint', 0) > 0:
            fig.add_vline(x=i, line_dash="dot", line_color="orange", opacity=0.3, line_width=0.5)
    
    # Layout
    fig.update_layout(
        title=f'Match probabilities - {match_data["match_id"].iloc[0]}',
        xaxis_title='Point index in match',
        yaxis_title='Match win probability',
        hovermode='x unified',
        template='plotly_white',
        width=1600,
        height=800,
        font=dict(size=12)
    )
    
    fig.update_yaxes(range=[0, 1])
    
    fig.write_html(output_path)
    print(f"Plot interattivo salvato in: {output_path}")


def save_predictions_csv(match_data, p1_probs, p2_probs, output_path):
    """
    Salva le predizioni in un file CSV.
    """
    # Calcola i set vinti per ogni punto
    p1_sets_list = []
    p2_sets_list = []
    for idx in range(len(match_data)):
        p1_sets, p2_sets = calculate_sets_won(match_data, idx)
        p1_sets_list.append(p1_sets)
        p2_sets_list.append(p2_sets)
    
    # Crea DataFrame
    results = pd.DataFrame({
        'match_id': match_data['match_id'].values,
        'point_number': range(len(match_data)),
        'set_no': match_data['SetNo'].values,
        'p1_sets_won': p1_sets_list,
        'p2_sets_won': p2_sets_list,
        'p1_games': match_data['P1GamesWon'].values,
        'p2_games': match_data['P2GamesWon'].values,
        'p1_score': match_data['P1Score'].values,
        'p2_score': match_data['P2Score'].values,
        'point_winner': match_data['PointWinner'].values,
        'p1_win_prob': p1_probs,
        'p2_win_prob': p2_probs
    })
    
    results.to_csv(output_path, index=False)
    print(f"Predizioni salvate in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Predici probabilità di vittoria punto per punto per una partita di tennis'
    )
    parser.add_argument('--data', required=True, help='Path al file CSV con i dati della partita')
    parser.add_argument('--match-id', required=True, help='ID del match da analizzare')
    parser.add_argument('--model', help='Path al modello (se non specificato, auto-rileva in base al gender)')
    parser.add_argument('--gender', choices=['male', 'female'], help='Genere (se non specificato, auto-rileva dal match-id)')
    parser.add_argument('--lstm-probs', type=str, help='File CSV con le probabilità LSTM per punto (opzionale)')
    parser.add_argument('--output-png', default='match_prediction.png', help='Path output PNG')
    parser.add_argument('--output-html', default='match_prediction.html', help='Path output HTML')
    parser.add_argument('--output-csv', default='match_predictions.csv', help='Path output CSV')
    
    args = parser.parse_args()
    
    # Auto-rileva il genere se non specificato
    if args.gender is None:
        args.gender = get_match_gender(args.match_id)
        print(f"Genere auto-rilevato: {args.gender}")
    
    # Auto-imposta il path del modello se non specificato
    if args.model is None:
        args.model = f'models/tennis_bdt_{args.gender}.pkl'
    
    # Auto-imposta il file LSTM se non specificato
    if args.lstm_probs is None:
        lstm_default = f'data/lstm_point_probs_{args.gender}.csv'
        if os.path.exists(lstm_default):
            args.lstm_probs = lstm_default
            print(f"File LSTM auto-rilevato: {args.lstm_probs}")
    
    print("="*60)
    print(f"Tennis Match Prediction - Singola Partita ({args.gender.upper()})")
    print("="*60)
    
    # Carica il modello
    print(f"\nCaricamento modello da: {args.model}")
    model, feature_names = load_model(args.model)
    n_model_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(feature_names)
    print(f"Modello caricato con {n_model_features} features")
    
    # Avviso se il modello è vecchio e ci sono probabilità LSTM
    if n_model_features == 44 and args.lstm_probs:
        print(f"\n⚠️  ATTENZIONE: Modello vecchio (44 features) rilevato.")
        print(f"   Le probabilità LSTM verranno ignorate.")
        print(f"   Per usare le probabilità LSTM, ri-addestra il modello con:")
        print(f"   python train_tennis_bdt.py --gender {args.gender} --force-reprocess --lstm-probs {args.lstm_probs}\n")
    
    # Carica i dati
    print(f"\nCaricamento dati da: {args.data}")
    df = pd.read_csv(args.data)
    
    # Carica le probabilità LSTM se fornite
    lstm_probs_df = None
    if args.lstm_probs and os.path.exists(args.lstm_probs):
        print(f"Caricamento probabilità LSTM da: {args.lstm_probs}")
        lstm_probs_df = pd.read_csv(args.lstm_probs)
        print(f"  Caricate {len(lstm_probs_df)} probabilità LSTM")
    
    # Filtra per match_id
    match_data = df[df['match_id'] == args.match_id].copy()
    
    if len(match_data) == 0:
        print(f"ERRORE: Match {args.match_id} non trovato nel file!")
        return
    
    print(f"Match trovato: {args.match_id}")
    print(f"  Numero di punti: {len(match_data)}")
    
    # Reset index
    match_data = match_data.reset_index(drop=True)
    
    # Fa le predizioni
    print("\nCalculating predictions...")
    p1_probs, p2_probs = predict_match(match_data, model, lstm_probs_df)
    
    print(f"\nRisultati:")
    print(f"  P1 probabilità iniziale: {p1_probs[0]:.1%}")
    print(f"  P1 probabilità finale: {p1_probs[-1]:.1%}")
    print(f"  P2 probabilità iniziale: {p2_probs[0]:.1%}")
    print(f"  P2 probabilità finale: {p2_probs[-1]:.1%}")
    
    # Determina il vincitore previsto
    final_winner = "P1" if p1_probs[-1] > 0.5 else "P2"
    print(f"  Vincitore previsto: {final_winner}")
    
    # Crea le directory di output se non esistono
    os.makedirs(os.path.dirname(args.output_png) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_html) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    
    # Genera i plot
    print("\nGenerazione plot...")
    plot_probabilities_matplotlib(match_data, p1_probs, p2_probs, args.output_png)
    plot_probabilities_plotly(match_data, p1_probs, p2_probs, args.output_html)
    
    # Salva CSV
    save_predictions_csv(match_data, p1_probs, p2_probs, args.output_csv)
    
    print("\n" + "="*60)
    print("Predizione completata!")
    print("="*60)


if __name__ == '__main__':
    main()
