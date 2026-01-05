# Integrazione Probabilità LSTM nel Modello BDT

## Sommario
Ho integrato con successo le probabilità LSTM dal file `data/lstm_point_probs_male.csv` nel modello BDT per catturare il "momento" della partita e migliorare le predizioni.

## Modifiche Implementate

### 1. **train_tennis_bdt.py**
- Aggiunta funzione `create_tennis_features()` con parametro opzionale `lstm_probs_df`
- Merge delle probabilità LSTM basato su: `match_id`, `SetNo`, `GameNo`, `PointNumber`
- Gestione del caso speciale '0X' (primo punto del match) → convertito a 0
- Gestione robusta dei tipi di dato con `pd.to_numeric()` ed `errors='coerce'`
- Aggiunta di **3 nuove feature**:
  - `lstm_point_prob`: Probabilità raw del momento (P1 vince il match)
  - `lstm_momentum_centered`: Probabilità centrata (-0.5 a +0.5) per catturare lo shift
  - `lstm_momentum_raw`: Probabilità raw non modificata
- Valori mancanti riempiti con 0.5 (neutro)
- Aggiornato `swap_player_features()` per invertire correttamente le probabilità quando si scambiano P1 e P2:
  - `lstm_point_prob`: invertito come `1.0 - prob`
  - `lstm_momentum_centered`: invertito come `-centered`
  - `lstm_momentum_raw`: invertito come `1.0 - raw`
- Aggiunto parametro `--lstm-probs` al comando CLI
- Auto-selezione del file LSTM in base al gender (male/female)

### 2. **predict_single_match.py**
- Sincronizzate tutte le modifiche con `train_tennis_bdt.py`
- Aggiunto parametro `--lstm-probs` alla CLI
- Auto-rilevamento del file LSTM se non specificato
- Supporto completo per le 3 nuove feature LSTM

### 3. **Feature Engineering**
Le nuove feature catturano:
- **Momento della partita**: Il modello LSTM ha appreso pattern temporali che il BDT può usare
- **Shift di momentum**: La versione centered (-0.5 a +0.5) aiuta il modello a capire cambi di direzione
- **Probabilità assoluta**: Il valore raw per punti dove la probabilità base è molto alta/bassa

## Utilizzo

### Training
```bash
# Training con LSTM probs (raccomandato)
python train_tennis_bdt.py --gender male --force-reprocess --lstm-probs data/lstm_point_probs_male.csv

# Training senza LSTM probs (feature = 0.5)
python train_tennis_bdt.py --gender male --force-reprocess
```

### Prediction
```bash
# Con LSTM probs (auto-detect)
python predict_single_match.py --data data/2019-wimbledon-points.csv --match-id 2019-wimbledon-1701

# Con LSTM probs esplicito
python predict_single_match.py --data data/2019-wimbledon-points.csv --match-id 2019-wimbledon-1701 --lstm-probs data/lstm_point_probs_male.csv
```

## Dettagli Tecnici

### Merge Strategy
- Join type: `LEFT` join (mantiene tutti i punti anche senza LSTM prob)
- Missing values: riempiti con 0.5 (neutro)
- Handling '0X': convertito a 0 per compatibilità

### Data Augmentation
Lo swap P1/P2 ora inverte correttamente le probabilità LSTM:
- Quando P1 e P2 vengono scambiati, le probabilità vengono invertite
- Questo mantiene la simmetria del modello

### Feature Count
- **Prima**: 44 feature
- **Dopo**: 47 feature (+3 LSTM features)

## Benefici Attesi

1. **Cattura del Momentum**: Le probabilità LSTM catturano pattern temporali e sequenziali
2. **Migliore Calibrazione**: Il modello può imparare quando "fidarsi" delle regole tennis vs momentum
3. **Predizioni più Accurate**: Combinazione di regole strutturate (BDT) + pattern temporali (LSTM)
4. **Robusto ai Missing Data**: Se mancano i dati LSTM, usa 0.5 (neutro) senza crashare

## Prossimi Passi

1. ✅ Completare il training con le nuove feature
2. Confrontare le metriche (Accuracy, AUC, Log Loss) con/senza LSTM
3. Analizzare feature importance per vedere quanto peso ha il modello dato alle feature LSTM
4. Testare su match specifici (es. 2019-wimbledon-1701) per vedere se il momentum migliora le predizioni nei momenti chiave

## Note
- Il file `data/lstm_point_probs_male.csv` contiene ~145k punti
- Le probabilità sono già calibrate (non serve ulteriore post-processing)
- La sottrazione di 0.5 serve solo per la feature "centered" che aiuta il modello a capire lo shift
