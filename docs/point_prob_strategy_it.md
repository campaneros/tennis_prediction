# Predizione punto‑per‑punto della vittoria match (BDT / NN, CPU)

## Obiettivo
Stimare ad ogni punto la probabilità che P1 vinca il match, garantendo coerenza con l’andamento (break, set vinti, tie‑break, match point). Modelli CPU‑friendly:
- BDT (HistGradientBoosting) per robustezza e interpretabilità.
- MLP shallow (2 layer) per modellare non‑linearità morbide con costo basso.

## Feature consigliate (aggiuntive rispetto a `features.py`)
- `points_in_game_frac`, `points_in_set_frac`, `points_in_match_frac`: progressi normalizzati per catturare fase del gioco e pressione.
- `server_point_run` e `receiver_point_run`: streak correnti (proxy momentum intra‑game).
- `server_recent_win_rate_5`, `receiver_recent_win_rate_5`: forma recentissima (rolling 5 punti, shiftata per evitare leakage).
- `game_pressure`: punti rimanenti stimati per chiudere il game (0–1); alza importanza di 30‑40, 40‑AD.
- `set_pressure`: analogo per il set (usa giochi vinti e gap set‑point).
- `match_progress`: punti giocati / totale match (stima grossolana) per calibrare prior finale.
- Flag espliciti: `is_tiebreak`, `is_decisive_tiebreak`, `is_break_point`, `is_set_point`, `is_match_point` (già derivabili in `features.py`, ma esposti in matrice finale).

## Strategia modello
1) **Pre‑processing**  
   - Carica CSV → `add_match_labels` → `add_rolling_serve_return_features` → `add_additional_features` → `add_leverage_and_momentum`.  
   - Aggiungi feature extra sopra (tutte numeriche, fillna 0).  
   - Costruisci matrice `X` e target soft (blend outcome + ancora di set) per stabilità.
2) **BDT (HistGradientBoostingRegressor)**  
   - Perdita log‑cosh su logit (robusto a outlier).  
   - max_depth 4–6, 400–600 estimators, learning_rate 0.05, subsample 0.8.  
   - Sample weights = `point_importance` * boost set decisivi.  
   - Early stopping su 10% hold‑out, monitor logloss.
3) **NN (MLP) CPU**  
   - Input: standardizzazione per colonna.  
   - Architettura: `d -> 128 -> 64 -> 1` con GELU, dropout 0.1, sigmoid finale.  
   - Loss: BCE with logits + focal term leggero (gamma 1.5) per punti rari (match point).  
   - Scheduler cosine w/ warmup; batch 2048, AdamW lr 3e‑3, weight decay 1e‑4.  
   - Early stopping su logloss/BS, patience 8.  
4) **Coerenza temporale**  
   - Train su soft‑label ancorate allo stato set → probabilità non oscillano brutalmente.  
   - Post‑processing opzionale: media mobile corta (3‑5 punti) solo per plotting, non per valutazione.  
5) **Valutazione**  
   - Logloss/Brier per calibratura; ROC AUC; curva affidabilità per set e tie‑break; monotonia vs `DistanceToMatchEnd`.  
   - Slice importanti: tie‑break decisivi, match points, break points, inizio set.

## Uso rapido (vedi `scripts/point_predictors.py`)
```bash
python -m scripts.point_predictors --files data/train.csv --model-out models/bdt_point.pt --model-type bdt
python -m scripts.point_predictors --files data/train.csv --model-out models/mlp_point.pt --model-type mlp
```
Output: modello salvato + metriche base; il codice usa solo CPU.
