# Transfer Learning per Previsione Tennis

Questo sistema utilizza un approccio di **transfer learning in due fasi** per addestrare una rete neurale a prevedere le probabilità di vittoria nel tennis:

## Fase 1: Pre-training su Partite Sintetiche (--tennis-training)

La rete impara le **regole fondamentali del tennis** su 50k-100k partite simulate perfettamente deterministiche:

```bash
python tennisctl.py tennis-training \
  --n-matches 50000 \
  --epochs 50 \
  --batch-size 2048 \
  --temperature 10.0 \
  --model-out models/tennis_rules_pretrained.pth \
  --device cuda
```

### Cosa impara in questa fase:

- **Regole di scoring**: 15-30-40, game, set, match
- **Tiebreak**: 6-6 per set normali, 12-12 per set finale
- **Critical points**: match point, set point, break point
- **Gerarchia tennis**: vincere game → vincere set → vincere match
- **Simmetria**: P(P1 vince) + P(P2 vince) = 1

### Caratteristiche delle partite sintetiche:

- ✅ Score perfettamente deterministico
- ✅ Nessun rumore, solo regole pure
- ✅ Variabilità: diversi livelli di skill (0.50-0.65)
- ✅ Copertura completa: tutti gli scenari possibili
- ✅ 31 features identiche al modello reale

### Parametri:

- `--n-matches`: Numero partite sintetiche (default: 50000)
  - 50k = ~5M punti (veloce, 20-30 minuti)
  - 100k = ~10M punti (migliore, 45-60 minuti)
- `--epochs`: Epoche training (default: 50)
- `--batch-size`: Dimensione batch (default: 2048)
- `--temperature`: Temperatura calibrazione (default: 10.0)
- `--model-out`: Dove salvare il modello pre-addestrato

---

## Fase 2: Fine-tuning su Dati Reali (--complete-model)

La rete **adatta le conoscenze** apprese ai pattern specifici dei dati reali:

```bash
python tennisctl.py complete-model \
  --files data/*-wimbledon-points.csv \
  --pretrained models/tennis_rules_pretrained.pth \
  --model-out models/nn_model_transfer.pth \
  --gender male \
  --epochs 30 \
  --batch-size 1024 \
  --learning-rate 0.0001 \
  --temperature 12.0 \
  --device cuda
```

### Cosa impara in questa fase:

- **Pattern specifici**: momenti psicologici, momentum
- **Variabilità reale**: rumore, errori, recuperi
- **Contestualizzazione**: serve/return, superficie, giocatori
- **Calibrazione fine**: adattamento delle probabilità

### Vantaggi del fine-tuning:

- ✅ **Learning rate basso** (0.0001 vs 0.001): non dimentica le regole
- ✅ **Early stopping**: evita overfitting sui dati reali
- ✅ **Validazione 80/20**: monitora generalizzazione
- ✅ **Opzione freeze**: può congelare primi layer (--freeze-layers)

### Parametri:

- `--files`: File CSV con dati reali (Wimbledon, etc.)
- `--pretrained`: Path al modello pre-addestrato (Fase 1)
- `--model-out`: Dove salvare il modello finale
- `--gender`: male/female/both
- `--epochs`: Epoche fine-tuning (default: 30)
- `--batch-size`: Dimensione batch (default: 1024)
- `--learning-rate`: Learning rate basso (default: 0.0001)
- `--temperature`: Temperatura calibrazione (default: 12.0)
- `--freeze-layers`: Congela primo layer (opzionale)

---

## Esempio Completo: Workflow Consigliato

### Step 1: Pre-training (Fase 1)

```bash
# Genera e addestra su 100k partite sintetiche
python tennisctl.py tennis-training \
  --n-matches 100000 \
  --epochs 50 \
  --batch-size 2048 \
  --model-out models/tennis_rules_100k.pth
```

**Output atteso:**
```
[1/4] Generating 100000 synthetic matches...
  Generated 100000/100000 matches...
  Total points: 10,234,567
  Match points: 125,432
  Set points: 534,211
  Break points: 2,341,098

[4/4] Training for 50 epochs...
  Epoch 50/50 - Loss: 0.0234

✓ Pre-training complete!
  Saved to: models/tennis_rules_100k.pth
```

### Step 2: Fine-tuning (Fase 2)

```bash
# Fine-tune sui dati Wimbledon
python tennisctl.py complete-model \
  --files data/2015-wimbledon-points.csv \
          data/2016-wimbledon-points.csv \
          data/2017-wimbledon-points.csv \
          data/2018-wimbledon-points.csv \
          data/2019-wimbledon-points.csv \
          data/2021-wimbledon-points.csv \
  --pretrained models/tennis_rules_100k.pth \
  --model-out models/wimbledon_transfer.pth \
  --gender male \
  --epochs 30 \
  --learning-rate 0.0001
```

**Output atteso:**
```
[1/5] Loading pre-trained model...
  ✓ Loaded model pre-trained on 100000 synthetic matches

[2/5] Loading real match data...
  Total points: 170,979

[3/5] Computing features from real data...
  Train: 136,783 points from 601 matches
  Val: 34,196 points from 151 matches

[4/5] Fine-tuning for 30 epochs...
  Epoch 30/30 - Train Loss: 0.1234, Val Loss: 0.1298

✓ Fine-tuning complete!
  Best validation loss: 0.1287
```

### Step 3: Predizione

```bash
# Usa il modello fine-tuned per predizioni
python tennisctl.py predict \
  --model models/wimbledon_transfer.pth \
  --files data/2019-wimbledon-points.csv \
  --match-id 2019-wimbledon-1701 \
  --plot-dir plots/transfer \
  --point-by-point
```

---

## Architettura del Sistema

### Componenti Principali:

1. **tennis_simulator.py**: Genera partite sintetiche deterministiche
   - Classe `TennisSimulator`: simula partite punto-per-punto
   - Regole perfette: scoring, tiebreak, set finale
   - Output: DataFrame con tutti i punti

2. **pretrain_tennis_rules.py**: Pre-training su dati sintetici
   - Classe `TennisRulesNet`: rete neurale multi-task
   - 31 features identiche al modello reale
   - Loss custom: BCE + consistency + temperature + penalties
   - Sample weights: match point 25x, set point 2x, break point 1.5x

3. **transfer_learning.py**: Fine-tuning su dati reali
   - Carica modello pre-addestrato
   - Usa pipeline features esistente (new_model_nn.py)
   - Learning rate basso per preservare conoscenze
   - Early stopping su validation set

4. **cli.py**: Interfaccia comandi
   - `tennis-training`: Fase 1 (pre-training)
   - `complete-model`: Fase 2 (fine-tuning)
   - `predict`: Predizioni con modello finale

### Features (31 totali):

```python
# 6 CORE: score normalizzato
p1_points/5, p2_points/5, p1_games/13, p2_games/13, p1_sets/3, p2_sets/3

# 3 CONTEXT: situazione partita
set_number/5, server_one_hot (2), is_best_of_5

# 4 TIEBREAK: rilevamento tiebreak
is_tiebreak, is_decisive_tiebreak, tb_p1_points/15, tb_p2_points/15

# 8 DISTANCE: distanza da vittoria
points_to_win_game (×2), games_to_win_set (×2), sets_to_win_match (×2), total_distance (×2)

# 8 CRITICAL: momenti critici
is_game_point (×2), is_set_point (×2), is_match_point (×2), is_break_point (×2)

# 2 PERFORMANCE: non usate in synth
p1_performance, p2_performance (sempre 0 in fase 1)
```

---

## Vantaggi del Transfer Learning

### ❌ Problema Originale:

- Rete non capiva regole tennis
- Probabilità schizzava dopo set vinti
- Non riconosceva match points
- Troppi oscillazioni punto-per-punto
- Confusione su tiebreak 6-6 vs 12-12

### ✅ Soluzione Transfer Learning:

1. **Fase 1 (Synthetic)**: impara SOLO regole pure
   - 100k partite perfette = nessun rumore
   - Critical points ben bilanciati (25x weight)
   - Loss penalties forzano logica tennis
   - Temperature alto (10.0) = smoothness

2. **Fase 2 (Real)**: aggiunge pattern reali
   - Learning rate basso = preserva conoscenze
   - Early stopping = no overfitting
   - Validation split = monitora generalizzazione
   - Temperature più alto (12.0) = ulteriore smoothing

### Risultati Attesi:

- ✅ Probabilità corrette ai match points (~0.85+)
- ✅ Probabilità moderate ai set points (~0.70)
- ✅ Oscillazioni ridotte (temperature 12.0)
- ✅ Tiebreak 12-12 riconosciuto correttamente
- ✅ Probabilità aumenta quando vinci un set (non diminuisce)

---

## Tuning Avanzato

### Se le oscillazioni sono ancora forti:

```bash
# Aumenta temperature in entrambe le fasi
python tennisctl.py tennis-training --temperature 12.0 ...
python tennisctl.py complete-model --temperature 15.0 ...
```

### Se la rete "dimentica" le regole durante fine-tuning:

```bash
# Congela primo layer (preserva features base)
python tennisctl.py complete-model --freeze-layers ...
```

### Se hai pochi dati reali:

```bash
# Più partite sintetiche per compensare
python tennisctl.py tennis-training --n-matches 200000 ...
```

### Se vuoi più epoche di pre-training:

```bash
# Training più lungo su regole
python tennisctl.py tennis-training --epochs 100 ...
```

---

## Confronto con Modello Originale

| Aspetto | Modello Originale | Transfer Learning |
|---------|-------------------|-------------------|
| Training | Diretto su dati reali | 2 fasi (synth + real) |
| Regole tennis | Implicate, deboli | Esplicite, forti |
| Oscillazioni | Forti | Ridotte |
| Match points | Non riconosciuti | Riconosciuti |
| Tiebreak finale | Confuso | Corretto |
| Generalizzazione | Media | Migliore |
| Tempo training | ~30 min | ~60 min totale |

---

## Debug e Troubleshooting

### Errore: "CUDA out of memory"

```bash
# Riduci batch size
python tennisctl.py tennis-training --batch-size 1024 ...
python tennisctl.py complete-model --batch-size 512 ...

# Oppure usa CPU
python tennisctl.py tennis-training --device cpu ...
```

### Loss non diminuisce in Fase 1:

- ✅ Normale se loss si stabilizza ~0.02-0.05
- ⚠️ Se loss > 0.1: aumenta epochs o learning rate

### Loss troppo bassa in Fase 2:

- ⚠️ Possibile overfitting sui dati reali
- ✅ Riduci epochs o aumenta learning rate

### Predizioni ancora sbagliate:

1. Verifica features: `print(X[0])` deve avere 31 valori normalizzati
2. Controlla temperature: deve essere consistente tra training e predict
3. Aumenta n_matches in Fase 1 (50k → 100k → 200k)

---

## File Generati

```
models/
  tennis_rules_pretrained.pth    # Fase 1: pre-training
  nn_model_transfer.pth          # Fase 2: fine-tuning
  
data/
  synthetic_tennis_matches_test.csv  # Test generatore (100 matches)
```

Formato checkpoint (.pth):
```python
{
  'model_state_dict': {...},        # Pesi rete
  'input_size': 31,                 # Features
  'hidden_sizes': [128, 64],        # Architettura
  'dropout': 0.4,
  'temperature': 12.0,
  'n_training_matches': 100000,     # Fase 1
  'finetune_matches': 752,          # Fase 2
  'pretrained_from': '...',         # Path Fase 1
  'best_val_loss': 0.1287
}
```

---

## Prossimi Passi

Dopo il training completo:

1. **Valida su test set**: usa match non visti in training
2. **Confronta con modello originale**: genera grafici side-by-side
3. **Analizza critical points**: verifica probabilità ai match/set points
4. **Calibrazione**: se necessario, ri-tuning temperature
5. **Esperimenti**: prova diverse configurazioni (freeze_layers, n_matches, etc.)

**Buon training! 🎾**
