#!/usr/bin/env python3
"""
Script per calcolare le probabilità counterfactual per i punti critici.
Legge il CSV delle predizioni prodotto da predict_single_match.py e
calcola cosa sarebbe successo se i punti critici fossero andati diversamente.
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_sets_won(df, up_to_point):
    """
    Calcola quanti set ha vinto ogni giocatore fino al punto up_to_point (escluso).
    """
    if up_to_point <= 0:
        return 0, 0
    
    prev_rows = df.iloc[:up_to_point]
    completed_sets_data = prev_rows[prev_rows['SetWinner'] != 0]
    
    p1_sets = 0
    p2_sets = 0
    
    for set_no in range(1, 6):
        set_data = completed_sets_data[completed_sets_data['SetNo'] == set_no]
        if len(set_data) > 0:
            set_winner = set_data.iloc[-1]['SetWinner']
            if set_winner == 1:
                p1_sets += 1
            elif set_winner == 2:
                p2_sets += 1
    
    return p1_sets, p2_sets


def recalculate_match_state_with_changed_point(match_data, point_idx_to_change):
    """
    Ricalcola lo stato del match se il punto point_idx_to_change avesse avuto vincitore opposto.
    
    Restituisce:
        cf_data: dati counterfactual
        match_won: True se il punto alternativo avrebbe fatto vincere il match
        winner: 1 o 2 se match_won è True, None altrimenti
    """
    # Prendi tutti i dati fino al punto prima di quello critico
    if point_idx_to_change == 0:
        # Primo punto del match - non può vincere il match
        cf_data = match_data.iloc[:1].copy()
        cf_data.at[cf_data.index[0], 'PointWinner'] = 2 if match_data.iloc[0]['PointWinner'] == 1 else 1
        return cf_data, False, None
    
    # Copia i dati fino al punto critico (incluso)
    cf_data = match_data.iloc[:point_idx_to_change+1].copy()
    
    # Inverti il vincitore solo del punto critico
    original_winner = match_data.iloc[point_idx_to_change]['PointWinner']
    cf_winner = 2 if original_winner == 1 else 1
    cf_data.at[cf_data.index[point_idx_to_change], 'PointWinner'] = cf_winner
    
    # Ora devo ricalcolare SOLO il punteggio del punto critico basandomi sul punto precedente
    prev_row = match_data.iloc[point_idx_to_change - 1]
    curr_idx = cf_data.index[point_idx_to_change]
    
    # Parti dal punteggio del punto precedente
    p1_score_prev = prev_row['P1Score']
    p2_score_prev = prev_row['P2Score']
    p1_games_prev = prev_row['P1GamesWon']
    p2_games_prev = prev_row['P2GamesWon']
    set_no = prev_row['SetNo']
    
    # Mappa punteggio tennis a numerico
    def score_to_num(s):
        s = str(s).strip()
        if s == '0': return 0
        elif s == '15': return 1
        elif s == '30': return 2
        elif s == '40': return 3
        elif s == 'AD': return 4
        else:
            try:
                return int(s)  # Tiebreak
            except:
                return 0
    
    def num_to_score(n):
        if n == 0: return '0'
        elif n == 1: return '15'
        elif n == 2: return '30'
        elif n == 3: return '40'
        elif n == 4: return 'AD'
        return str(n)
    
    p1_score = score_to_num(p1_score_prev)
    p2_score = score_to_num(p2_score_prev)
    p1_games = p1_games_prev
    p2_games = p2_games_prev
    
    # Controlla se siamo in tiebreak (games = 6-6)
    in_tiebreak = (p1_games == 6 and p2_games == 6)
    
    # Applica il nuovo vincitore
    game_won = False
    
    if not in_tiebreak:
        # Game normale
        if cf_winner == 1:
            if p1_score < 3:
                p1_score += 1
            elif p1_score == 3 and p2_score < 3:
                # P1 vince il game
                p1_games += 1
                p1_score = 0
                p2_score = 0
                game_won = True
            elif p1_score == 3 and p2_score == 3:
                # Deuce -> AD
                p1_score = 4
            elif p1_score == 4:
                # P1 vince da AD
                p1_games += 1
                p1_score = 0
                p2_score = 0
                game_won = True
            elif p2_score == 4:
                # Torna a deuce
                p1_score = 3
                p2_score = 3
        else:  # cf_winner == 2
            if p2_score < 3:
                p2_score += 1
            elif p2_score == 3 and p1_score < 3:
                # P2 vince il game
                p2_games += 1
                p1_score = 0
                p2_score = 0
                game_won = True
            elif p2_score == 3 and p1_score == 3:
                # Deuce -> AD
                p2_score = 4
            elif p2_score == 4:
                # P2 vince da AD
                p2_games += 1
                p1_score = 0
                p2_score = 0
                game_won = True
            elif p1_score == 4:
                # Torna a deuce
                p1_score = 3
                p2_score = 3
    else:
        # Tiebreak
        if cf_winner == 1:
            p1_score += 1
        else:
            p2_score += 1
        
        # Check vittoria tiebreak
        if (p1_score >= 7 or p2_score >= 7) and abs(p1_score - p2_score) >= 2:
            if p1_score > p2_score:
                p1_games = 7
                p2_games = 6
            else:
                p1_games = 6
                p2_games = 7
            p1_score = 0
            p2_score = 0
            game_won = True
    
    # Aggiorna il dataset counterfactual
    if not in_tiebreak:
        cf_data.at[curr_idx, 'P1Score'] = num_to_score(p1_score)
        cf_data.at[curr_idx, 'P2Score'] = num_to_score(p2_score)
    else:
        cf_data.at[curr_idx, 'P1Score'] = str(p1_score)
        cf_data.at[curr_idx, 'P2Score'] = str(p2_score)
    
    cf_data.at[curr_idx, 'P1GamesWon'] = p1_games
    cf_data.at[curr_idx, 'P2GamesWon'] = p2_games
    
    # IMPORTANTE: Se il game è stato vinto, devo controllare se ha vinto anche il set/match
    set_won = False
    match_won = False
    set_winner = None
    
    if game_won:
        # Controlla se il set è stato vinto
        if (p1_games >= 6 or p2_games >= 6):
            if abs(p1_games - p2_games) >= 2:
                set_won = True
                set_winner = 1 if p1_games > p2_games else 2
            elif p1_games == 7 and p2_games == 6:
                set_won = True
                set_winner = 1
            elif p1_games == 6 and p2_games == 7:
                set_won = True
                set_winner = 2
        
        # Se il set è stato vinto, controlla se è stato vinto il match
        if set_won:
            # Conta i set vinti prima di questo
            p1_sets_before, p2_sets_before = calculate_sets_won(match_data, point_idx_to_change - 1 if point_idx_to_change > 0 else 0)
            
            # Aggiungi il set appena vinto
            if set_winner == 1:
                p1_sets_total = p1_sets_before + 1
                p2_sets_total = p2_sets_before
            else:
                p1_sets_total = p1_sets_before
                p2_sets_total = p2_sets_before + 1
            
            # Match vinto a 3 set
            if p1_sets_total >= 3:
                match_won = True
                set_winner = 1
            elif p2_sets_total >= 3:
                match_won = True
                set_winner = 2
    
    return cf_data, match_won, set_winner if match_won else None


def create_tennis_features(df, lstm_probs_df=None, n_features=None):
    """
    Crea le feature tennis dal dataframe dei punti.
    Versione semplificata che importa dalla funzione originale.
    """
    # Importa la funzione originale
    from predict_single_match import create_tennis_features as original_create_features
    return original_create_features(df, lstm_probs_df, n_features)


def compute_counterfactual(predictions_csv, points_csv, model_path, output_csv, lstm_csv=None):
    """
    Calcola le probabilità counterfactual leggendo il CSV delle predizioni.
    
    Args:
        predictions_csv: CSV con le predizioni generate da predict_single_match.py
        points_csv: CSV originale con i dati dei punti
        model_path: Path del modello addestrato
        output_csv: Path per salvare il CSV con le counterfactual
        lstm_csv: CSV opzionale con le probabilità LSTM
    """
    print("=" * 70)
    print("🔄 CALCOLO COUNTERFACTUAL")
    print("=" * 70)
    
    # Carica predizioni
    print(f"\n📂 Caricamento predizioni da: {predictions_csv}")
    predictions_df = pd.read_csv(predictions_csv)
    
    # Carica dati punti
    print(f"📂 Caricamento dati punti da: {points_csv}")
    points_df = pd.read_csv(points_csv)
    
    # Filtra per match_id dalle predizioni
    match_id = predictions_df['match_id'].iloc[0]
    match_data = points_df[points_df['match_id'] == match_id].copy()
    print(f"   Match ID: {match_id}")
    print(f"   Punti totali: {len(match_data)}")
    
    # Carica modello
    print(f"\n🔧 Caricamento modello da: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    print(f"   Modello con {n_features} features")
    
    # Carica LSTM se disponibile
    lstm_probs_df = None
    if lstm_csv and Path(lstm_csv).exists():
        print(f"\n📂 Caricamento probabilità LSTM da: {lstm_csv}")
        lstm_probs_df = pd.read_csv(lstm_csv)
        lstm_probs_df = lstm_probs_df[lstm_probs_df['match_id'] == match_id].copy()
    
    # Inizializza array counterfactual
    n_points = len(match_data)
    p1_probs_cf = [None] * n_points
    p2_probs_cf = [None] * n_points
    critical_points = []
    
    print(f"\n🔄 Analisi punti critici...")
    
    # Per ogni punto, calcola la counterfactual e vedi se è critico
    for i in range(n_points):
        # Calcola counterfactual
        cf_data, match_won_cf, winner_cf = recalculate_match_state_with_changed_point(match_data, i)
        
        # Calcola le feature per vedere se nella realtà o nella CF c'è un punto critico
        history_data = match_data.iloc[:i+1].copy()
        history_lstm = lstm_probs_df.iloc[:i+1].copy() if lstm_probs_df is not None else None
        X_history = create_tennis_features(history_data, history_lstm, n_features=n_features)
        
        if len(X_history) == 0:
            continue
        
        # Feature della realtà
        features_real = X_history[-1, :]
        
        # Feature della counterfactual
        cf_lstm = lstm_probs_df.iloc[:i+1].copy() if lstm_probs_df is not None else None
        X_cf = create_tennis_features(cf_data, cf_lstm, n_features=n_features)
        
        if len(X_cf) == 0:
            continue
            
        features_cf = X_cf[-1, :]
        
        # Indici delle feature critiche:
        # 16: p1_break_point, 17: p2_break_point
        # 20: p1_match_point, 21: p2_match_point  
        # 22: p1_set_point, 23: p2_set_point
        
        is_critical_real = (features_real[16] == 1 or features_real[17] == 1 or 
                           features_real[20] == 1 or features_real[21] == 1 or
                           features_real[22] == 1 or features_real[23] == 1)
        
        is_critical_cf = (features_cf[16] == 1 or features_cf[17] == 1 or 
                         features_cf[20] == 1 or features_cf[21] == 1 or
                         features_cf[22] == 1 or features_cf[23] == 1)
        
        # Match vinto direttamente nella counterfactual è sempre critico
        is_critical = is_critical_real or is_critical_cf or match_won_cf
        
        if is_critical:
            critical_points.append(i)
            
            # Calcola probabilità counterfactual
            if match_won_cf:
                if winner_cf == 1:
                    p1_prob_cf = 1.0
                    p2_prob_cf = 0.0
                else:
                    p1_prob_cf = 0.0
                    p2_prob_cf = 1.0
            else:
                prob = model.predict_proba(X_cf[-1:, :])
                p1_prob_cf = prob[0, 1]
                p2_prob_cf = 1.0 - p1_prob_cf
            
            p1_probs_cf[i] = p1_prob_cf
            p2_probs_cf[i] = p2_prob_cf
        
        if (i + 1) % 50 == 0 or i == n_points - 1:
            print(f"  Analizzati {i+1}/{n_points} punti...")
    
    print(f"\n✓ Trovati {len(critical_points)} punti critici")
    
    # Aggiungi colonne counterfactual al dataframe delle predizioni
    predictions_df['is_critical_point'] = 0
    predictions_df['p1_win_prob_cf'] = None
    predictions_df['p2_win_prob_cf'] = None
    
    for i in critical_points:
        predictions_df.at[i, 'is_critical_point'] = 1
        predictions_df.at[i, 'p1_win_prob_cf'] = p1_probs_cf[i]
        predictions_df.at[i, 'p2_win_prob_cf'] = p2_probs_cf[i]
    
    # Salva
    predictions_df.to_csv(output_csv, index=False)
    print(f"\n✅ Counterfactual salvata in: {output_csv}")
    print(f"   Punti critici: {len(critical_points)}/{n_points}")
    print("=" * 70)
    
    return predictions_df, critical_points


def plot_counterfactual(predictions_df, output_html='counterfactual_plot.html'):
    """
    Genera un plot interattivo delle probabilità con counterfactual.
    """
    print(f"\n📊 Generazione plot counterfactual...")
    
    # Filtra solo i punti critici
    critical_df = predictions_df[predictions_df['is_critical_point'] == 1].copy()
    
    if len(critical_df) == 0:
        print("⚠️  Nessun punto critico da plottare")
        return
    
    fig = go.Figure()
    
    # Linea principale P1
    fig.add_trace(go.Scatter(
        x=predictions_df['point_number'],
        y=predictions_df['p1_win_prob'],
        mode='lines',
        name='P1 wins match (actual)',
        line=dict(color='blue', width=2)
    ))
    
    # Linea principale P2
    fig.add_trace(go.Scatter(
        x=predictions_df['point_number'],
        y=predictions_df['p2_win_prob'],
        mode='lines',
        name='P2 wins match (actual)',
        line=dict(color='red', width=2)
    ))
    
    # Punti critici - linee tratteggiate verso counterfactual
    for _, row in critical_df.iterrows():
        point_num = row['point_number']
        actual_p1 = row['p1_win_prob']
        actual_p2 = row['p2_win_prob']
        cf_p1 = row['p1_win_prob_cf']
        cf_p2 = row['p2_win_prob_cf']
        
        # Estrai informazioni sul punteggio
        set_no = int(row['set_no'])
        p1_sets = int(row['p1_sets_won'])
        p2_sets = int(row['p2_sets_won'])
        p1_games = int(row['p1_games'])
        p2_games = int(row['p2_games'])
        p1_score = str(row['p1_score'])
        p2_score = str(row['p2_score'])
        
        score_text = f"Set {set_no}: [{p1_sets}-{p2_sets}] Games: {p1_games}-{p2_games}, Points: {p1_score}-{p2_score}"
        
        # Linea tratteggiata da actual a counterfactual per P1
        fig.add_trace(go.Scatter(
            x=[point_num, point_num],
            y=[actual_p1, cf_p1],
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            showlegend=False,
            hovertext=f"Point {point_num}<br>{score_text}<br>P1: {actual_p1:.1%} → {cf_p1:.1%}",
            hoverinfo='text'
        ))
        
        # Linea tratteggiata da actual a counterfactual per P2
        fig.add_trace(go.Scatter(
            x=[point_num, point_num],
            y=[actual_p2, cf_p2],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False,
            hovertext=f"Point {point_num}<br>{score_text}<br>P2: {actual_p2:.1%} → {cf_p2:.1%}",
            hoverinfo='text'
        ))
        
        # Marker sul punto counterfactual P1
        fig.add_trace(go.Scatter(
            x=[point_num],
            y=[cf_p1],
            mode='markers',
            marker=dict(color='blue', size=8, symbol='circle-open'),
            showlegend=False,
            hovertext=f"CF Point {point_num}<br>{score_text}<br>P1: {cf_p1:.1%}",
            hoverinfo='text'
        ))
        
        # Marker sul punto counterfactual P2
        fig.add_trace(go.Scatter(
            x=[point_num],
            y=[cf_p2],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle-open'),
            showlegend=False,
            hovertext=f"CF Point {point_num}<br>{score_text}<br>P2: {cf_p2:.1%}",
            hoverinfo='text'
        ))
    
    # Linea al 50%
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'Match Probabilities with Counterfactual Analysis - {predictions_df["match_id"].iloc[0]}',
        xaxis_title='Point index in match',
        yaxis_title='Match win probability',
        hovermode='x unified',
        template='plotly_white',
        width=1600,
        height=800,
        font=dict(size=12)
    )
    
    fig.update_yaxes(range=[0, 1])
    
    fig.write_html(output_html)
    print(f"✅ Plot salvato in: {output_html}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute counterfactual probabilities for critical points')
    parser.add_argument('--predictions', type=str, required=True,
                       help='CSV file with predictions from predict_single_match.py')
    parser.add_argument('--points', type=str, required=True,
                       help='Original points CSV file')
    parser.add_argument('--model', type=str, default='models/tennis_bdt_male.pkl',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='match_predictions_with_cf.csv',
                       help='Output CSV with counterfactual probabilities')
    parser.add_argument('--lstm', type=str, default=None,
                       help='Optional LSTM probabilities CSV')
    parser.add_argument('--plot', type=str, default=None,
                       help='Generate interactive HTML plot (provide output path)')
    
    args = parser.parse_args()
    
    # Verifica file
    if not Path(args.predictions).exists():
        print(f"❌ Errore: File predizioni non trovato: {args.predictions}")
        sys.exit(1)
    
    if not Path(args.points).exists():
        print(f"❌ Errore: File punti non trovato: {args.points}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"❌ Errore: Modello non trovato: {args.model}")
        sys.exit(1)
    
    # Esegui counterfactual
    predictions_df, critical_points = compute_counterfactual(args.predictions, args.points, args.model, args.output, args.lstm)
    
    # Genera plot se richiesto
    if args.plot:
        plot_counterfactual(predictions_df, args.plot)
