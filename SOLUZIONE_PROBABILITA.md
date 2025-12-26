# Soluzione al Problema delle Probabilità Errate

## Problema Identificato
Il modello mostrava probabilità sbagliate:
- Dopo che P1 vince il 1° set, la sua probabilità scendeva sotto P2 ❌
- Ai match point di P2, le probabilità non riflettevano la criticità ❌  
- Nel tie-break del 5° set, le probabilità erano piatte ❌

## Causa Root
1. **Features troppo minimali** (25): mancavano informazioni sul contesto
   - Non sapeva chi stava vincendo il set corrente
   - Non considerava la differenza di set vinti
   - Nessuna info su quanto mancava alla fine del match

2. **Parametri di training troppo conservativi**:
   - Label smoothing eccessivo: 0.25-0.75 (troppo piatto)
   - Temperature 10.0 (probabilità troppo smoothed)
   - Rete piccola [96, 48] (poca capacità di apprendere)

## Soluzione Implementata

### 1. Features Arricchite (da 25 a 28)
Aggiunte 3 features critiche di **contesto match**:
- `current_set_games_diff`: P1_games - P2_games (chi sta vincendo il set)
- `sets_diff`: P1_sets - P2_sets (chi ha vinto più set)
- `distance_to_end`: quanto manca alla fine (0-1, normalizzato)

### 2. Parametri Ottimizzati

| Parametro | Prima | Dopo | Motivazione |
|-----------|-------|------|-------------|
| **Architecture** | [96, 48] | [256, 128, 64] | Più capacità per pattern complessi |
| **Label smoothing** | 0.25-0.75 | 0.05-0.95 | Permette probabilità più estreme |
| **Temperature** | 10.0 | 1.5 | Riduce smoothing eccessivo |
| **Dropout** | 0.6 | 0.3 | Meno regolarizzazione |
| **Sample weights cap** | 4.0 | 6.0 | Più peso ai punti critici |
| **Epochs** | 150 | 200 | Più tempo per convergere |
| **Weight exponent** | 0.4 | 0.5 | Bilancio migliore |

### 3. Struttura Features Finale (28 totali)

```
Core Scoring (6):    P1/P2 punti, game, set
Context (4):         server, set#, game#, punto#
Tie-break (6):       flag, punti, distanza vittoria
Match Format (3):    best-of-5, sets_to_win, final_set
Performance (6):     prob srv/rcv long/short, momentum P1/P2
Match Context (3):   ⭐ NEW: games_diff, sets_diff, distance_to_end
```

## Vantaggi della Soluzione

✅ **Contesto esplicito**: la rete sa chi sta vincendo
✅ **Probabilità dinamiche**: può dare 0.9+ ai match point
✅ **Meno smoothing**: riflette meglio la criticità dei punti
✅ **Rete più capace**: [256,128,64] apprende pattern complessi
✅ **Features pulite**: solo 28 invece di 45, nessun duplicato

## Come Addestrare

```bash
# Esegui training con nuove features e parametri
./train_clean_nn.sh

# Output: models/nn_clean_optimized.json
```

## Test

```bash
# Testa sul match Wimbledon 2019 Final
python tennisctl.py predict \
  --model models/nn_clean_optimized.json \
  --match-id 2019-wimbledon-1701 \
  --files data/2019-wimbledon-points.csv \
  --plot-dir plots/clean_optimized \
  --point-by-point
```

## Risultati Attesi

- ✅ Probabilità > 0.7 per P1 dopo vittoria 1° set
- ✅ Probabilità > 0.85 ai match point
- ✅ Oscillazioni marcate nel tie-break decisivo (0.3-0.7)
- ✅ Risposta corretta a ogni game/set vinto
