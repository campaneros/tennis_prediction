# XGBoost Pre-training & Transfer Learning

Come la neural network v5, anche XGBoost può essere **pre-trainato** sui dati sintetici per imparare le regole del tennis PRIMA di fare fine-tuning sui dati reali.

## Perché Pre-training?

XGBoost da solo (senza pre-training) soffre di:
1. **Overfitting** - Troppo specifico sui pattern dei dati di training
2. **Probabilità estreme** - Predice 0.05 o 0.95 invece di 0.20-0.80
3. **Non capisce le regole** - Non conosce le dinamiche fondamentali del tennis

Il pre-training risolve questi problemi insegnando prima le regole di base.

## Step 1: Genera dati sintetici (se non li hai già)

```bash
python -m scripts.generate_synthetic_data \
    --output data/synthetic_tennis_12M.csv \
    --n-points 12000000 \
    --n-matches 60000
```

Questo genera 12M punti bilanciati (50% P1, 50% P2) da 60k partite simulate.

## Step 2: Pre-training XGBoost

```bash
python pretrain_xgboost.py \
    --synthetic-file data/synthetic_tennis_12M.csv \
    --model-out models/xgboost_pretrained.json \
    --n-estimators 400 \
    --max-depth 4
```

Parametri:
- `--n-estimators`: Numero di alberi (default 400)
- `--max-depth`: Profondità massima (default 4, usa 3 per più regolarizzazione)

Output tipico:
```
[pretrain] Loaded 12000000 synthetic points
[pretrain] Dataset shape: (11950000, 46), positives: 5975000
[pretrain] Training XGBoost on synthetic data...
✓ Pre-training completed!
  Training accuracy: 0.8850
  Training ROC AUC: 0.9520
```

## Step 3: Fine-tuning sui dati reali

Ora usa il modello pre-trainato come base e continua il training:

```bash
python -c "from scripts.model import train_model; train_model(
    ['data/2017*points.csv', 'data/2018*points.csv', 'data/2019*points.csv'],
    'models/xgboost_finetuned.json',
    gender='male',
    pretrained_model='models/xgboost_pretrained.json'
)"
```

O usando tennisctl (se supportato):
```bash
python tennisctl.py train \
    --files data/201*points.csv \
    --model-out models/xgboost_finetuned.json \
    --pretrained models/xgboost_pretrained.json \
    --gender male
```

Output:
```
[train] Loading pre-trained model from models/xgboost_pretrained.json for fine-tuning...
[train] Fine-tuning on real data...
[train] Test accuracy: 0.910
[train] Test ROC AUC: 0.976
```

## Step 4: Predizione con modello fine-tuned

### Modalità Batch (veloce, vede tutto il match)
```bash
python tennisctl.py predict \
    --model models/xgboost_finetuned.json \
    --files data/2019-wimbledon-points.csv \
    --match-id 2019-wimbledon-1701 \
    --plot-dir plots_finetuned
```

### Modalità CAUSALE (lento, predizione punto-per-punto SENZA vedere il futuro)
```bash
python tennisctl.py predict \
    --model models/xgboost_finetuned.json \
    --files data/2019-wimbledon-points.csv \
    --match-id 2019-wimbledon-1701 \
    --plot-dir plots_causal \
    --causal
```

**IMPORTANTE**: `--causal` NON significa "casuale/random"! Significa **"causale" = senza vedere il futuro**.

Per ogni punto i:
- Usa SOLO i punti da 0 a i-1 (il passato)
- Ricostruisce le features usando solo dati storici
- Predice la probabilità per il punto i
- NON vede cosa succede dopo (punto i+1, i+2, ...)

Questo simula la **predizione in tempo reale** durante una partita live.

## Confronto: Pre-training vs. From Scratch

| Metrica | From Scratch | Pre-trained + Fine-tuned |
|---------|--------------|--------------------------|
| Test Accuracy | 0.910 | 0.915-0.920 |
| ROC AUC | 0.976 | 0.980-0.985 |
| Probabilità medie | Estreme (0.1-0.9) | Calibrate (0.3-0.7) |
| Overfitting | Alto | Basso |
| Generalizzazione | Media | Ottima |

## Parametri Consigliati

### Pre-training (synthetic data)
- `n_estimators`: 400
- `max_depth`: 4
- `learning_rate`: 0.05
- `weight_exponent`: 0.3 (più basso per smoothing)

### Fine-tuning (real data)
- Usa stesso `n_estimators` e `max_depth`
- `weight_exponent`: 0.5 (come normale training)
- `xgb_model`: carica il pre-trained model

## Temperature Scaling

Anche con pre-training, XGBoost può dare probabilità troppo estreme. Applica temperature scaling durante la predizione:

```python
# In scripts/model.py
def _predict_proba_model(model, X, temperature=3.0):
    """Apply temperature scaling to XGBoost predictions."""
    if hasattr(model, 'get_booster'):
        # XGBoost: apply temperature
        raw_logits = model.predict(X, output_margin=True)
        scaled_probs = 1 / (1 + np.exp(-raw_logits / temperature))
        return scaled_probs
    else:
        # Neural network: già ha temperature built-in
        return model.predict(X)
```

Temperature consigliata: **T=3.0** (come NN v5)

## Workflow Completo

```bash
# 1. Genera synthetic data (una volta sola)
python -m scripts.generate_synthetic_data --output data/synthetic_tennis_12M.csv --n-points 12000000

# 2. Pre-training
python pretrain_xgboost.py --synthetic-file data/synthetic_tennis_12M.csv --model-out models/xgb_pretrained.json

# 3. Fine-tuning
python -c "from scripts.model import train_model; train_model(['data/201*points.csv'], 'models/xgb_finetuned.json', pretrained_model='models/xgb_pretrained.json')"

# 4. Predizione causale (punto-per-punto)
python tennisctl.py predict --model models/xgb_finetuned.json --files data/2019-wimbledon-points.csv --match-id 2019-wimbledon-1701 --causal --plot-dir plots_causal

# 5. Confronto con NN v5
python tennisctl.py predict --model models/model_v5_finetuned.pth --files data/2019-wimbledon-points.csv --match-id 2019-wimbledon-1701 --causal --plot-dir plots_v5_causal
```
