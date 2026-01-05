#!/usr/bin/env python3
"""
Training script per BDT (Boosted Decision Tree) per predire probabilità di vittoria finale 
di un match di tennis punto per punto.

Il modello comprende le regole del tennis attraverso feature specifiche che catturano:
- Set score (numero di set vinti)
- Game score (giochi vinti per set)
- Point score (punteggio nel game corrente)
- Momentum e statistiche cumulative
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import pickle
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
    Converte il punteggio del punto in valori numerici per catturare la situazione nel game.
    
    Regole tennis:
    - 0, 15, 30, 40
    - Deuce quando entrambi 40
    - Advantage quando uno ha un punto in più dal deuce
    - Nel tie-break: numeri diretti (0, 1, 2, 3, ...)
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
    
    # Guarda i set completati fino a questo punto
    prev_rows = match_data.iloc[:current_idx]
    
    # Conta i set vinti guardando tutti i SetWinner != 0
    p1_sets = 0
    p2_sets = 0
    
    # Ottieni tutti i set completati (quelli con SetWinner != 0)
    completed_sets_data = prev_rows[prev_rows['SetWinner'] != 0]
    
    if not completed_sets_data.empty:
        # Per ogni set number, prendi l'ultimo valore di SetWinner
        for set_no in completed_sets_data['SetNo'].unique():
            set_data = completed_sets_data[completed_sets_data['SetNo'] == set_no]
            if not set_data.empty:
                winner = set_data.iloc[-1]['SetWinner']
                if winner == 1:
                    p1_sets += 1
                elif winner == 2:
                    p2_sets += 1
    
    return p1_sets, p2_sets


def create_tennis_features(df, lstm_probs_df=None):
    """
    Crea feature specifiche per il tennis che catturano lo stato della partita.
    
    Feature chiave:
    1. Set score (quanti set ha vinto ogni giocatore)
    2. Game score nel set corrente
    3. Point score nel game corrente (convertito in numerico)
    4. Se è un game di servizio o break point
    5. Statistiche cumulative (ace, winner, errori)
    6. Momentum
    7. LSTM point probability (probabilità del momento della partita)
    
    Args:
        df: DataFrame con i dati del match
        lstm_probs_df: DataFrame opzionale con le probabilità LSTM per punto
                      (colonne: match_id, SetNo, GameNo, PointNumber, p1_point_prob)
    """
    features = []
    
    # Merge LSTM probabilities se fornite
    if lstm_probs_df is not None:
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
        
        # LSTM Point Probability - Momentum del match dal modello LSTM
        # Questa probabilità cattura il "momento" della partita
        # Se tutte le prob sono alte, sottraiamo 0.5 per centrare
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
        # Set decisivo (2-2 nei set)
        decisive_set = 1 if p1_sets_won == 2 and p2_sets_won == 2 else 0
        
        # Game vicino alla vittoria
        close_to_winning_p1 = 1 if (p1_sets_won == 2 and p2_sets_won < 2 and p1_games >= 5) else 0
        close_to_winning_p2 = 1 if (p2_sets_won == 2 and p1_sets_won < 2 and p2_games >= 5) else 0
        
        # FEATURE CHIAVE: Distanza dalla vittoria finale
        # Per vincere serve vincere 3 set. Calcola quanti set servono ancora
        p1_sets_to_win_match = max(0, 3 - p1_sets_won)
        p2_sets_to_win_match = max(0, 3 - p2_sets_won)
        
        # Quanto manca a vincere il set corrente (approssimazione)
        p1_games_to_win_set = max(0, 6 - p1_games) if p1_games < 6 else 0
        p2_games_to_win_set = max(0, 6 - p2_games) if p2_games < 6 else 0
        
        # Quanto manca a vincere il game corrente
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
        
        # Feature TIE-BREAK: vantaggio nel tie-break (molto importante!)
        tiebreak_point_diff = 0
        if in_tiebreak:
            tiebreak_point_diff = p1_point_val - p2_point_val  # Può essere negativo
        
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
        
        # Set point detection (nel set corrente)
        p1_set_point = 0
        p2_set_point = 0
        if p1_games >= 5 and p1_games > p2_games and p1_point_val == 3 and p2_point_val < 3:
            p1_set_point = 1
        if p2_games >= 5 and p2_games > p1_games and p2_point_val == 3 and p1_point_val < 3:
            p2_set_point = 1
        
        # Situazione critica del game
        game_criticality = 0
        if p1_games >= 5 or p2_games >= 5:
            game_criticality = 1
        if p1_games == p2_games and p1_games >= 5:
            game_criticality = 2
        
        # Info sul set corrente (meno correlato con l'outcome finale)
        set_number = row['SetNo']  # Quale set stiamo giocando (1-5)
        
        # FEATURE DI CRITICITÀ: quanto è importante questo momento
        
        # 1. Criticità del punteggio nel game (0-15 non critico, 30-40 molto critico)
        point_criticality = min(p1_point_val, p2_point_val)  # Più entrambi sono avanti, più è critico
        if p1_point_val == 3 or p2_point_val == 3:  # Qualcuno ha 40
            point_criticality += 2
        if p1_point_val >= 3 and p2_point_val >= 3:  # Deuce o advantage
            point_criticality += 3
        
        # 2. Distanza dal vincere il set (chi è più vicino?)
        p1_distance_to_set = max(0, 6 - p1_games) if p1_games < 6 else 0
        p2_distance_to_set = max(0, 6 - p2_games) if p2_games < 6 else 0
        min_distance_to_set = min(p1_distance_to_set, p2_distance_to_set)  # Il più vicino
        
        # 3. Situazione "must-win" - se perdi questo set perdi il match
        p1_must_win_set = 1 if (p2_sets_won == 2 and p1_sets_won < 2) else 0
        p2_must_win_set = 1 if (p1_sets_won == 2 and p2_sets_won < 2) else 0
        
        # 4. Pressione del game - giochi cruciali tipo 4-5, 5-5, 5-6
        game_pressure = 0
        if abs(p1_games - p2_games) <= 1 and (p1_games >= 4 or p2_games >= 4):
            game_pressure = 1
        if p1_games >= 5 and p2_games >= 5:  # Entrambi vicini a vincere
            game_pressure = 2
        
        # 5. Chi può vincere il match vincendo questo set?
        p1_can_win_match_this_set = 1 if p1_sets_won == 2 else 0
        p2_can_win_match_this_set = 1 if p2_sets_won == 2 else 0
        match_on_the_line = p1_can_win_match_this_set or p2_can_win_match_this_set
        
        # 6. Momentum shift potential - quanto può cambiare la situazione
        momentum_shift_potential = 0
        if p1_break_point or p2_break_point:
            momentum_shift_potential += 2
        if p1_set_point or p2_set_point:
            momentum_shift_potential += 3
        if p1_match_point or p2_match_point:
            momentum_shift_potential += 5
        
        # 7. Overall match criticality score
        match_criticality_score = 0
        match_criticality_score += set_number * 0.5  # Set più tardi = più critico
        match_criticality_score += game_criticality * 2
        match_criticality_score += point_criticality
        match_criticality_score += (1.0 / (min_distance_to_set + 1)) * 3  # Più vicini al set, più critico
        match_criticality_score += (p1_must_win_set + p2_must_win_set) * 5
        match_criticality_score += game_pressure * 2
        match_criticality_score += match_on_the_line * 10  # MOLTO più peso se il match è in gioco
        
        # 8. Feature POTENZIATE per match point e set point
        # Match point deve avere impatto molto forte
        p1_has_match_point_advantage = p1_match_point * 80  # Peso 80x
        p2_has_match_point_advantage = p2_match_point * 80  # Peso 80x
        p1_has_set_point_advantage = p1_set_point * 25  # Peso 25x
        p2_has_set_point_advantage = p2_set_point * 25  # Peso 25x
        
        # 9. Combined advantage score - quanto è favorito ogni giocatore
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
            p2_advantage_score += 3  # Break point per l'altro = +3
        if p2_break_point:
            p1_advantage_score += 3
        if p1_serving:
            p1_advantage_score += 1  # Servizio = +1
        if p2_serving:
            p2_advantage_score += 1
        
        advantage_diff = p1_advantage_score - p2_advantage_score
        
        # 10. ULTRA feature: situazione match dominance
        # Questa feature codifica direttamente quanto un giocatore domina la situazione
        match_situation_score = 0.0
        
        # MATCH POINT = 150 (aumentato per probabilità molto alta)
        if p1_match_point:
            match_situation_score += 150
        if p2_match_point:
            match_situation_score -= 150
        
        # Set point: peso azzerato per evitare picchi troppo alti
        # if p1_set_point and p1_can_win_match_this_set:
        #     match_situation_score += 10
        # elif p1_set_point:
        #     match_situation_score += 5
        # 
        # if p2_set_point and p2_can_win_match_this_set:
        #     match_situation_score -= 10
        # elif p2_set_point:
        #     match_situation_score -= 5
        
        # Set point che può chiudere il match = 100
        if p1_set_point and p1_can_win_match_this_set:
            match_situation_score += 100  # Era 30, ora 100
        elif p1_set_point:
            match_situation_score += 40  # Era 15, ora 40
        
        if p2_set_point and p2_can_win_match_this_set:
            match_situation_score -= 100
        elif p2_set_point:
            match_situation_score -= 40
        
        # Break point quando il match è in gioco = 15
        if match_on_the_line:
            if p1_break_point:
                match_situation_score -= 15  # Era 5, ora 15
            if p2_break_point:
                match_situation_score += 15
        
        # TIE-BREAK: se siamo a 6-6 nei game, ogni punto conta MOLTO
        in_tiebreak = (p1_games == 6 and p2_games == 6)
        if in_tiebreak and match_on_the_line:
            # Nel tie-break del set decisivo, la differenza punti conta MOLTISSIMO
            match_situation_score += point_diff * 30  # Ogni punto di differenza = 30
            # Se sei a 6 punti nel tiebreak, sei vicino a vincere
            if p1_point_val >= 6:
                match_situation_score += 50
            if p2_point_val >= 6:
                match_situation_score -= 50
        elif in_tiebreak:
            # Tie-break normale (non decisivo)
            match_situation_score += point_diff * 15
            if p1_point_val >= 6:
                match_situation_score += 25
            if p2_point_val >= 6:
                match_situation_score -= 25
        
        # Vantaggio nel game score quando il match è vicino
        if match_on_the_line and not in_tiebreak:
            match_situation_score += game_diff * 8  # Era 3, ora 8
        
        # Costruisci il feature vector
        # SOLO stato istantaneo + criticità - NO statistiche cumulative
        feature_vec = [
            # Set info
            set_number,
            p1_sets_won,
            p2_sets_won,
            set_diff / 2.0,  # Scalato per evitare salti dopo aver vinto un set
            
            # Game score
            p1_games,
            p2_games,
            game_diff,
            p1_games_to_win_set,
            p2_games_to_win_set,
            
            # Point score
            p1_point_val,
            p2_point_val,
            point_diff,
            p1_points_to_win_game,
            p2_points_to_win_game,
            
            # Serving
            p1_serving,
            p2_serving,
            
            # Break points
            p1_break_point,
            p2_break_point,
            
            # Situazioni critiche
            decisive_set,
            close_to_winning_p1,
            close_to_winning_p2,
            p1_match_point,
            p2_match_point,
            p1_set_point,
            p2_set_point,
            game_criticality,
            
            # Feature di criticità
            point_criticality,
            p1_distance_to_set,
            p2_distance_to_set,
            min_distance_to_set,
            p1_must_win_set,
            p2_must_win_set,
            game_pressure,
            p1_can_win_match_this_set,
            p2_can_win_match_this_set,
            match_on_the_line,
            momentum_shift_potential,
            match_criticality_score,
            
            # Power features per match/set point
            p1_has_match_point_advantage,
            p2_has_match_point_advantage,
            p1_has_set_point_advantage,
            p2_has_set_point_advantage,
            p1_advantage_score,
            p2_advantage_score,
            advantage_diff,
            match_situation_score,
            in_tiebreak,
            tiebreak_point_diff,  # Nuovo: vantaggio punti nel tie-break
            
            # LSTM momentum features
            lstm_point_prob,
            lstm_momentum_centered,
            lstm_momentum_raw,
        ]
        
        features.append(feature_vec)
    
    feature_names = [
        'set_number', 'p1_sets_won', 'p2_sets_won', 'set_diff',
        'p1_games', 'p2_games', 'game_diff',
        'p1_games_to_win_set', 'p2_games_to_win_set',
        'p1_point_val', 'p2_point_val', 'point_diff',
        'p1_points_to_win_game', 'p2_points_to_win_game',
        'p1_serving', 'p2_serving',
        'p1_break_point', 'p2_break_point',
        'decisive_set', 'close_to_winning_p1', 'close_to_winning_p2',
        'p1_match_point', 'p2_match_point',
        'p1_set_point', 'p2_set_point',
        'game_criticality',
        'point_criticality', 'p1_distance_to_set', 'p2_distance_to_set',
        'min_distance_to_set', 'p1_must_win_set', 'p2_must_win_set',
        'game_pressure', 'p1_can_win_match_this_set', 'p2_can_win_match_this_set',
        'match_on_the_line', 'momentum_shift_potential', 'match_criticality_score',
        'p1_has_match_point_advantage', 'p2_has_match_point_advantage',
        'p1_has_set_point_advantage', 'p2_has_set_point_advantage',
        'p1_advantage_score', 'p2_advantage_score', 'advantage_diff',
        'match_situation_score', 'in_tiebreak', 'tiebreak_point_diff',
        'lstm_point_prob', 'lstm_momentum_centered', 'lstm_momentum_raw'
    ]
    
    return np.array(features), feature_names


def determine_match_winner(df):
    """
    Determina chi ha vinto il match guardando il risultato finale.
    Per ogni punto, il target è 1 se P1 ha vinto il match, 0 se P2 ha vinto.
    """
    # Guarda l'ultimo punto per determinare il vincitore
    last_row = df.iloc[-1]
    
    # Conta i set vinti dal vincitore finale
    p1_sets_final, p2_sets_final = calculate_sets_won(df, len(df))
    
    # Aggiorna con l'ultimo set
    if last_row['SetWinner'] == 1:
        p1_sets_final += 1
    elif last_row['SetWinner'] == 2:
        p2_sets_final += 1
    
    # Il vincitore è chi ha vinto più set (best of 5)
    if p1_sets_final > p2_sets_final:
        return 1
    else:
        return 0


def swap_player_features(features):
    """
    Scambia le feature di P1 con quelle di P2 per data augmentation.
    Questo rende il modello simmetrico rispetto ai giocatori.
    
    Feature order (51 features con LSTM e nuove features):
    0: set_number (stays the same)
    1-3: p1_sets_won, p2_sets_won, set_diff
    4-8: p1_games, p2_games, game_diff, p1_games_to_win_set, p2_games_to_win_set
    9-13: p1_point_val, p2_point_val, point_diff, p1_points_to_win_game, p2_points_to_win_game
    14-15: p1_serving, p2_serving
    16-17: p1_break_point, p2_break_point
    18-25: decisive_set, close_to_winning_p1, close_to_winning_p2, p1_match_point, p2_match_point, p1_set_point, p2_set_point, game_criticality
    26-37: point_criticality, p1_distance_to_set, p2_distance_to_set, min_distance_to_set, p1_must_win_set, p2_must_win_set, game_pressure, p1_can_win_match_this_set, p2_can_win_match_this_set, match_on_the_line, momentum_shift_potential, match_criticality_score
    38-44: p1_has_match_point_advantage, p2_has_match_point_advantage, p1_has_set_point_advantage, p2_has_set_point_advantage, p1_advantage_score, p2_advantage_score, advantage_diff
    45: match_situation_score (inverted)
    46: in_tiebreak (stays the same)
    47: tiebreak_point_diff (inverted)
    48: lstm_point_prob (inverted: 1 - prob)
    49: lstm_momentum_centered (inverted)
    50: lstm_momentum_raw (inverted: 1 - raw)
    """
    swapped = np.copy(features)
    
    for i in range(len(features)):
        row = features[i].copy()
        
        # Swap pairs of P1/P2 features and invert differences
        # set_number stays the same (0)
        swapped[i][1], swapped[i][2] = row[2], row[1]  # sets_won
        swapped[i][3] = -row[3]  # set_diff inverted
        swapped[i][4], swapped[i][5] = row[5], row[4]  # games
        swapped[i][6] = -row[6]  # game_diff inverted
        swapped[i][7], swapped[i][8] = row[8], row[7]  # games_to_win_set
        swapped[i][9], swapped[i][10] = row[10], row[9]  # point_val
        swapped[i][11] = -row[11]  # point_diff inverted
        swapped[i][12], swapped[i][13] = row[13], row[12]  # points_to_win_game
        swapped[i][14], swapped[i][15] = row[15], row[14]  # serving
        swapped[i][16], swapped[i][17] = row[17], row[16]  # break_point
        # decisive_set stays the same (18)
        swapped[i][19], swapped[i][20] = row[20], row[19]  # close_to_winning
        swapped[i][21], swapped[i][22] = row[22], row[21]  # match_point
        swapped[i][23], swapped[i][24] = row[24], row[23]  # set_point
        # game_criticality stays the same (25)
        # point_criticality stays the same (26)
        swapped[i][27], swapped[i][28] = row[28], row[27]  # distance_to_set
        # min_distance_to_set stays the same (29)
        swapped[i][30], swapped[i][31] = row[31], row[30]  # must_win_set
        # game_pressure stays the same (32)
        swapped[i][33], swapped[i][34] = row[34], row[33]  # can_win_match_this_set
        # match_on_the_line stays the same (35)
        # momentum_shift_potential stays the same (36)
        # match_criticality_score stays the same (37)
        swapped[i][38], swapped[i][39] = row[39], row[38]  # has_match_point_advantage
        swapped[i][40], swapped[i][41] = row[41], row[40]  # has_set_point_advantage
        swapped[i][42], swapped[i][43] = row[43], row[42]  # advantage_score
        swapped[i][44] = -row[44]  # advantage_diff inverted
        swapped[i][45] = -row[45]  # match_situation_score inverted
        # in_tiebreak stays the same (46)
        swapped[i][47] = -row[47]  # tiebreak_point_diff inverted
        
        # LSTM features: invertire per P1 <-> P2 swap
        swapped[i][48] = 1.0 - row[48]  # lstm_point_prob inverted (1 - p1_prob = p2_prob)
        swapped[i][49] = -row[49]  # lstm_momentum_centered inverted
        swapped[i][50] = 1.0 - row[50]  # lstm_momentum_raw inverted
    
    return swapped


def load_and_prepare_data(data_dir='data', preprocessed_file='data/tennis_features_preprocessed.csv', force_reprocess=False, gender='male', lstm_probs_file=None):
    """
    Carica tutti i file *wimbledon-points.csv e prepara i dati per il training.
    Se il file preprocessato esiste, lo carica direttamente per velocizzare.
    
    Args:
        gender: 'male' o 'female' per filtrare i match appropriati
        lstm_probs_file: path al file CSV con le probabilità LSTM (opzionale)
                        es: 'data/lstm_point_probs_male.csv'
    """
    # Controlla se esiste già il file preprocessato
    if os.path.exists(preprocessed_file) and not force_reprocess:
        print(f"Caricamento dati preprocessati da: {preprocessed_file}")
        df_preprocessed = pd.read_csv(preprocessed_file)
        
        # Separa features, labels e match_ids
        match_ids = df_preprocessed['match_id'].values
        y = df_preprocessed['match_winner'].values
        
        # Rimuovi le colonne non-feature
        feature_cols = [col for col in df_preprocessed.columns 
                       if col not in ['match_id', 'match_winner', 'point_index']]
        X = df_preprocessed[feature_cols].values
        
        feature_names = feature_cols
        
        print(f"\nDati preprocessati caricati ({gender}):")
        print(f"  Totale punti: {len(X)}")
        print(f"  Features: {len(feature_names)}")
        print(f"  P1 vince: {np.sum(y)} ({100*np.mean(y):.1f}%)")
        print(f"  P2 vince: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")
        
        return X, y, match_ids, feature_names
    
    print(f"Caricamento e processamento dati da zero (genere: {gender})...")
    
    # Carica le probabilità LSTM se fornite
    lstm_probs_df = None
    if lstm_probs_file and os.path.exists(lstm_probs_file):
        print(f"\nCaricamento probabilità LSTM da: {lstm_probs_file}")
        lstm_probs_df = pd.read_csv(lstm_probs_file)
        print(f"  Caricate {len(lstm_probs_df)} probabilità LSTM")
    elif lstm_probs_file:
        print(f"\nWARNING: File LSTM non trovato: {lstm_probs_file}")
        print("  Procedo senza le probabilità LSTM (feature = 0.5)")
    
    # Trova tutti i file points
    point_files = glob.glob(os.path.join(data_dir, '*wimbledon-points.csv'))
    print(f"Trovati {len(point_files)} file di punti")
    
    all_features = []
    all_labels = []
    all_match_ids = []
    
    for file_path in point_files:
        print(f"Processando {os.path.basename(file_path)}...")
        df = pd.read_csv(file_path)
        
        # Raggruppa per match_id
        match_ids = df['match_id'].unique()
        
        for match_id in match_ids:
            # Filtra per genere
            match_gender = get_match_gender(match_id)
            if match_gender != gender:
                continue
            
            match_data = df[df['match_id'] == match_id].copy()
            
            # Salta match troppo corti o con dati mancanti
            if len(match_data) < 10:
                continue
            
            # Crea feature (con LSTM probs se disponibili)
            try:
                features, feature_names = create_tennis_features(match_data, lstm_probs_df)
                
                # Determina il vincitore
                winner = determine_match_winner(match_data)
                
                # Label per ogni punto: 1 se P1 vince il match, 0 se P2 vince
                labels = np.full(len(features), winner)
                
                all_features.append(features)
                all_labels.append(labels)
                all_match_ids.extend([match_id] * len(features))
                
                # DATA AUGMENTATION: Crea anche la versione con P1 e P2 scambiati
                # Questo rende le feature simmetriche e bilanciate
                features_swapped = swap_player_features(features)
                labels_swapped = np.full(len(features_swapped), 1 - winner)  # Inverti il vincitore
                
                all_features.append(features_swapped)
                all_labels.append(labels_swapped)
                all_match_ids.extend([match_id + '_swapped'] * len(features_swapped))
                
            except Exception as e:
                print(f"  Errore nel processare match {match_id}: {e}")
                continue
    
    # Concatena tutti i dati
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    match_ids = np.array(all_match_ids)
    
    print(f"\nDati preparati ({gender}):")
    print(f"  Totale punti: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  P1 vince: {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"  P2 vince: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")
    
    # Rimuovi eventuali NaN rimasti (sostituisci con 0)
    print("\nControllo e pulizia NaN...")
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        print(f"  Trovati {nan_count} valori NaN, sostituisco con 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        print("  Nessun NaN trovato")
    
    # Salva i dati preprocessati per uso futuro
    print(f"\nSalvataggio dati preprocessati in: {preprocessed_file}")
    df_preprocessed = pd.DataFrame(X, columns=feature_names)
    df_preprocessed['match_id'] = match_ids
    df_preprocessed['match_winner'] = y
    df_preprocessed['point_index'] = range(len(X))
    
    os.makedirs(os.path.dirname(preprocessed_file), exist_ok=True)
    df_preprocessed.to_csv(preprocessed_file, index=False)
    print("Dati preprocessati salvati con successo!")
    
    return X, y, match_ids, feature_names


def train_model(X, y, match_ids, feature_names):
    """
    Allena il modello BDT.
    
    Nota: Split per match per evitare data leakage (punti dello stesso match non devono
    essere sia in train che in test).
    """
    print("\n" + "="*60)
    print("Training del modello BDT")
    print("="*60)
    
    # Split basato sui match per evitare leakage
    unique_matches = np.unique(match_ids)
    train_matches, test_matches = train_test_split(
        unique_matches, test_size=0.2, random_state=42
    )
    
    train_mask = np.isin(match_ids, train_matches)
    test_mask = np.isin(match_ids, test_matches)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Calcola sample weights basati sulla criticità del punto
    # I punti con match point o set point sono MOLTO più importanti per l'apprendimento
    print("\nCalcolo sample weights per punti critici...")
    
    # Trova gli indici delle feature di criticità
    match_point_p1_idx = feature_names.index('p1_match_point')
    match_point_p2_idx = feature_names.index('p2_match_point')
    set_point_p1_idx = feature_names.index('p1_set_point')
    set_point_p2_idx = feature_names.index('p2_set_point')
    decisive_set_idx = feature_names.index('decisive_set')
    
    # Crea weight array (default = 1.0)
    sample_weights = np.ones(len(X_train))
    
    # Match point: peso 80x (aumentato per dare massima priorità)
    match_point_mask = (X_train[:, match_point_p1_idx] == 1) | (X_train[:, match_point_p2_idx] == 1)
    sample_weights[match_point_mask] = 80.0
    
    # Set point: pesi rimossi completamente per evitare picchi
    # decisive_set_point_mask = (
    #     (X_train[:, decisive_set_idx] == 1) & 
    #     ((X_train[:, set_point_p1_idx] == 1) | (X_train[:, set_point_p2_idx] == 1))
    # ) & ~match_point_mask
    # sample_weights[decisive_set_point_mask] = 1.5
    # 
    # normal_set_point_mask = (
    #     ((X_train[:, set_point_p1_idx] == 1) | (X_train[:, set_point_p2_idx] == 1)) &
    #     ~match_point_mask & ~decisive_set_point_mask
    # )
    # sample_weights[normal_set_point_mask] = 1.0
    
    # Statistiche sui pesi
    n_match_points = np.sum(match_point_mask)
    n_normal_points = len(sample_weights) - n_match_points
    
    print(f"  Match points: {n_match_points} (peso 80x)")
    print(f"  Punti normali: {n_normal_points} (peso 1x)")
    print(f"  Peso totale: {np.sum(sample_weights):.0f} (equivalente a {np.sum(sample_weights)/len(sample_weights):.2f}x dataset)")
    
    print(f"\nSplit dei dati:")
    print(f"  Training set: {len(X_train)} punti da {len(train_matches)} match")
    print(f"  Test set: {len(X_test)} punti da {len(test_matches)} match")
    
    # Crea e allena il modello GradientBoosting
    print("\nAllenamento del modello con sample weights...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.015,  # Ridotto da 0.03 per apprendimento più smooth
        max_depth=5,  # Ridotto da 6 per evitare overfitting su set points
        min_samples_split=200,  # Aumentato per split più conservativi
        min_samples_leaf=100,  # Aumentato per foglie più grandi
        subsample=0.8,
        max_features=0.7,
        random_state=42,
        verbose=1
    )
    
    # Usa sample_weight per dare più importanza ai punti critici
    # Il modello impara che sbagliare un match point costa 80x rispetto a un punto normale
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Valutazione
    print("\n" + "="*60)
    print("Valutazione del modello")
    print("="*60)
    
    # Training set
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_prob)
    train_logloss = log_loss(y_train, y_train_prob)
    
    print(f"\nTraining Set:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  AUC: {train_auc:.4f}")
    print(f"  Log Loss: {train_logloss:.4f}")
    
    # Test set
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_logloss = log_loss(y_test, y_test_prob)
    
    print(f"\nTest Set:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Log Loss: {test_logloss:.4f}")
    
    return model


def save_model(model, feature_names, output_path='models/tennis_bdt.pkl'):
    """
    Salva il modello e i nomi delle feature.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModello salvato in: {output_path}")


def main():
    """
    Main function per il training.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Train BDT per tennis prediction')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Forza il riprocessamento dei dati anche se esiste il file preprocessato')
    parser.add_argument('--gender', choices=['male', 'female'], default='male',
                        help='Genere per il quale allenare il modello (male: match_id < 2000, female: >= 2000)')
    parser.add_argument('--lstm-probs', type=str, default='data/lstm_point_probs_male.csv',
                        help='File CSV con le probabilità LSTM per punto (opzionale)')
    args = parser.parse_args()
    
    # Imposta i path basati sul genere
    preprocessed_file = f'data/tennis_features_preprocessed_{args.gender}.csv'
    model_file = f'models/tennis_bdt_{args.gender}.pkl'
    
    # Usa il file LSTM appropriato per il genere se non specificato
    lstm_probs_file = args.lstm_probs
    if args.lstm_probs == 'data/lstm_point_probs_male.csv' and args.gender == 'female':
        lstm_probs_file = 'data/lstm_point_probs_female.csv'
    
    print("="*60)
    print(f"Training BDT per Tennis Match Prediction - {args.gender.upper()}")
    print("="*60)
    
    # Carica e prepara i dati
    X, y, match_ids, feature_names = load_and_prepare_data(
        'data', 
        preprocessed_file=preprocessed_file,
        force_reprocess=args.force_reprocess,
        gender=args.gender,
        lstm_probs_file=lstm_probs_file
    )
    
    # Allena il modello
    model = train_model(X, y, match_ids, feature_names)
    
    # Feature importance
    print("\n" + "="*60)
    print("Feature Importance (Top 15)")
    print("="*60)
    
    feature_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (name, importance) in enumerate(feature_importance[:15], 1):
        print(f"{i:2d}. {name:25s}: {importance:.4f}")
    
    # Salva il modello
    save_model(model, feature_names, output_path=model_file)
    
    print("\n" + "="*60)
    print("Training completato!")
    print("="*60)


if __name__ == '__main__':
    main()
