# Guida al Training - XGBoost e Neural Network

Questa guida spiega come addestrare i modelli indipendentemente usando gli script dedicati.

## Training XGBoost

### Script Standalone
```bash
python train_xgboost.py --files data/*matches.csv --model-out models/xgboost_model.json
```

### Opzioni
- `--files`: File CSV dei match (richiesto)
- `--model-out`: Percorso dove salvare il modello (richiesto)
- `--gender`: Filtro per genere (`male`, `female`, `both`) - default: `male`
- `--config`: File di configurazione JSON personalizzato

### Esempi

#### Train su tutti i match
```bash
python train_xgboost.py --files data/*matches.csv --model-out models/xgb_all.json
```

#### Train su anni specifici
```bash
python train_xgboost.py \
  --files data/2017*matches.csv data/2018*matches.csv \
  --model-out models/xgb_2017_2018.json
```

#### Train solo match maschili
```bash
python train_xgboost.py \
  --files data/*matches.csv \
  --model-out models/xgb_male.json \
  --gender male
```

#### Train solo match femminili
```bash
python train_xgboost.py \
  --files data/*matches.csv \
  --model-out models/xgb_female.json \
  --gender female
```

#### Con configurazione personalizzata
```bash
python train_xgboost.py \
  --files data/*matches.csv \
  --model-out models/xgb_custom.json \
  --config configs/custom_config.json
```

---

## Training Neural Network

### 4 Modalità di Training

#### 1. Standard NN (45 features)
```bash
python train_neural_network.py \
  --files data/*matches.csv \
  --model-out models/nn_standard.pth
```

#### 2. Clean Features NN (25 features - migliore per imparare le regole)
```bash
python train_neural_network.py \
  --files data/*matches.csv \
  --model-out models/nn_clean.pth \
  --clean-features
```

#### 3. Multi-task NN (23 features - predice match/set/game)
```bash
python train_neural_network.py \
  --files data/*matches.csv \
  --model-out models/nn_multitask.pth \
  --new-model
```

#### 4. Transfer Learning (pre-training + fine-tuning)

**Step 1: Pre-training su dati sintetici**
```bash
python train_neural_network.py \
  --mode pretrain \
  --model-out models/pretrained.pth \
  --n-synthetic 50000 \
  --epochs 50
```

**Step 2: Fine-tuning su dati reali**
```bash
python train_neural_network.py \
  --mode finetune \
  --files data/*matches.csv \
  --pretrained models/pretrained.pth \
  --model-out models/complete.pth \
  --epochs 30
```

### Opzioni Comuni

#### Input/Output
- `--files`: File CSV dei match (richiesto per `train` e `finetune`)
- `--model-out`: Percorso dove salvare il modello (richiesto)
- `--pretrained`: Modello pre-allenato (richiesto per `finetune`)

#### Tipo di Modello
- `--clean-features`: Usa 25 features invece di 45
- `--new-model`: Usa modello multi-task (23 features)

#### Iperparametri
- `--epochs`: Numero di epoche (default: 200 per train, 50 per pretrain, 30 per finetune)
- `--batch-size`: Dimensione batch (default: 1024 per train/finetune, 2048 per pretrain)
- `--temperature`: Temperatura per calibrazione (default: 3.0)

#### Dati
- `--gender`: Filtro genere (`male`, `female`, `both`) - default: `male`
- `--device`: Device (`cuda`, `cpu`) - default: `cuda`
- `--n-synthetic`: Numero di match sintetici per pre-training (default: 50000)

### Esempi Avanzati

#### Training veloce per test
```bash
python train_neural_network.py \
  --files data/2017*matches.csv \
  --model-out models/nn_test.pth \
  --epochs 50 \
  --batch-size 2048
```

#### Training con temperatura personalizzata
```bash
python train_neural_network.py \
  --files data/*matches.csv \
  --model-out models/nn_temp5.pth \
  --temperature 5.0 \
  --epochs 200
```

#### Pre-training esteso
```bash
python train_neural_network.py \
  --mode pretrain \
  --model-out models/pretrained_large.pth \
  --n-synthetic 100000 \
  --epochs 100 \
  --temperature 3.0
```

#### Fine-tuning solo femminile
```bash
python train_neural_network.py \
  --mode finetune \
  --files data/*matches.csv \
  --pretrained models/pretrained.pth \
  --model-out models/complete_female.pth \
  --gender female
```

---

## Confronto Modelli

### XGBoost vs Neural Network

| Caratteristica | XGBoost | Neural Network |
|---------------|---------|----------------|
| **Features** | 45 (leverage, momentum, rolling stats) | 25-45 (selezionabili) |
| **Training time** | Veloce (~5 min) | Medio (~20 min) |
| **Interpretabilità** | Alta (feature importance) | Bassa (black box) |
| **Prestazioni** | Ottime su dati strutturati | Ottime con transfer learning |
| **Memoria** | Bassa | Media |
| **Transfer learning** | No | Sì (pre-train su sintetici) |

### Quando usare XGBoost
- Training veloce
- Debugging e analisi feature
- Baseline solido
- Risorse limitate

### Quando usare Neural Network
- Transfer learning (pre-train + fine-tune)
- Apprendimento gerarchico (match/set/game)
- Dataset piccoli (con pre-training)
- Features pulite (25) per imparare regole

---

## Workflow Completo

### 1. Baseline XGBoost
```bash
# Train veloce per baseline
python train_xgboost.py \
  --files data/*matches.csv \
  --model-out models/xgb_baseline.json
```

### 2. Neural Network Standard
```bash
# Train NN standard
python train_neural_network.py \
  --files data/*matches.csv \
  --model-out models/nn_standard.pth \
  --epochs 200
```

### 3. Transfer Learning (RACCOMANDATO)
```bash
# Step 1: Pre-training
python train_neural_network.py \
  --mode pretrain \
  --model-out models/pretrained_v5.pth \
  --n-synthetic 60000 \
  --epochs 50 \
  --temperature 3.0

# Step 2: Fine-tuning
python train_neural_network.py \
  --mode finetune \
  --files data/*matches.csv \
  --pretrained models/pretrained_v5.pth \
  --model-out models/complete_v5.pth \
  --epochs 30
```

### 4. Test e Confronto
```bash
# Testa tutti i modelli
python compare_models.py
```

---

## Uso tramite tennisctl.py

Gli script standalone sono equivalenti ai comandi `tennisctl.py`:

### XGBoost
```bash
# Equivalente
python train_xgboost.py --files data/*matches.csv --model-out models/xgb.json
python tennisctl.py train --files data/*matches.csv --model-out models/xgb.json --model-type xgboost
```

### Neural Network
```bash
# Equivalente
python train_neural_network.py --files data/*matches.csv --model-out models/nn.pth
python tennisctl.py train --files data/*matches.csv --model-out models/nn.pth --model-type nn
```

### Transfer Learning
```bash
# Pre-training
python train_neural_network.py --mode pretrain --model-out models/pre.pth --n-synthetic 50000
python tennisctl.py tennis-training --model-out models/pre.pth --n-matches 50000

# Fine-tuning
python train_neural_network.py --mode finetune --files data/*matches.csv --pretrained models/pre.pth --model-out models/complete.pth
python tennisctl.py complete-model --files data/*matches.csv --pretrained models/pre.pth --model-out models/complete.pth
```

---

## Troubleshooting

### Errore CUDA out of memory
```bash
# Riduci batch size
python train_neural_network.py ... --batch-size 512 --device cpu
```

### File non trovati
```bash
# Verifica i file esistano
ls data/*matches.csv
```

### Modello pre-trained mancante
```bash
# Verifica il path
ls models/pretrained.pth
```

### Import errors
```bash
# Attiva l'ambiente virtuale
source venv/bin/activate
pip install -r requirements.txt
```

---

## Best Practices

1. **Inizia con XGBoost**: Train veloce per baseline
2. **Usa transfer learning per NN**: Pre-train + fine-tune
3. **Testa su validation set**: Usa `compare_models.py`
4. **Monitora l'overfitting**: Guarda loss su train vs validation
5. **Salva checkpoint**: Modelli intermedi durante training
6. **Documenta esperimenti**: Versiona i modelli (v1, v2, v3...)

---

## Modelli Attuali

### XGBoost
- `models/xgb_with_weighted_leverage.json` - Baseline con leverage pesato
- `models/xgboost_model.json` - Modello standard

### Neural Network
- `models/complete_model_v2.pth` - Transfer learning v2 (T=12.0, saturato)
- `models/complete_model_v4.pth` - Transfer learning v4 (T=3.0, bias inverso)
- `models/complete_model_v5.pth` - Transfer learning v5 (T=3.0, dati bilanciati) - IN TRAINING

### Pre-trained
- `models/tennis_rules_pretrained_v4.pth` - Pre-train v4 (30k matches, biased)
- `models/tennis_rules_pretrained_v5.pth` - Pre-train v5 (60k matches, balanced) - IN TRAINING
