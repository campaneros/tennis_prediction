# Metriche di Valutazione per Modelli di Predizione Tennis

## Contesto del Problema

Il problema della predizione del vincitore di un match di tennis presenta caratteristiche uniche che richiedono metriche di valutazione specifiche, diverse dalle metriche standard utilizzate in problemi di classificazione binaria convenzionali.

### Peculiarità del Dominio

1. **Probabilità Tempo-Varianti**: Il modello produce una sequenza di predizioni {p₁, p₂, ..., pₙ} per ogni match, dove pᵢ rappresenta la probabilità stimata che il giocatore P1 vinca il match dopo aver osservato i primi i punti.

2. **Label Costante per Match**: Contrariamente ai problemi di classificazione sequenziale standard, il ground truth y ∈ {0,1} rimane costante per tutti i punti dello stesso match, rappresentando il vincitore finale.

3. **Dipendenza Temporale**: Le predizioni consecutive non sono indipendenti, essendo generate da stati successivi dello stesso processo stocastico (il match).

4. **Oscillazioni Naturali**: È statisticamente corretto che le probabilità oscillino durante il match in risposta agli eventi osservati, anche quando il vincitore finale è deterministico a posteriori.

---

## 1. Match-Level Accuracy

### Definizione Formale

Sia M = {m₁, m₂, ..., mₖ} l'insieme dei match nel test set. Per ogni match mᵢ, sia:
- nᵢ il numero di punti nel match
- p̂ᵢ,ₙᵢ la probabilità predetta dal modello dopo l'ultimo punto
- yᵢ ∈ {0,1} il vincitore reale

La Match-Level Accuracy (MLA) è definita come:

```
MLA = (1/k) Σᵢ₌₁ᵏ 𝟙[ŷᵢ = yᵢ]

dove ŷᵢ = 𝟙[p̂ᵢ,ₙᵢ ≥ τ]
```

con τ = 0.5 come soglia di decisione standard.

### Giustificazione Teorica

#### Perché è una metrica appropriata:

1. **Indipendenza tra Match**: A differenza dei punti individuali, match diversi possono essere considerati campioni indipendenti dalla stessa distribuzione generatrice, soddisfacendo l'assunzione i.i.d. necessaria per stime statistiche valide.

2. **Obiettivo Pratico**: In applicazioni reali (betting, analisi sportiva), l'interesse primario è predire correttamente il vincitore finale, non la correttezza di ogni predizione intermedia.

3. **Robustezza alle Oscillazioni**: Valutando solo l'ultimo punto, la metrica è insensibile alle oscillazioni naturali durante il match, che sono informative piuttosto che erronee.

4. **Interpretabilità**: L'accuracy ha una chiara interpretazione probabilistica come stima della probabilità P(Predizione Corretta) sul dominio dei match.

### Interpretazione dei Risultati

**Risultato Osservato**: MLA = 90.85%

Questo indica che il modello predice correttamente il vincitore finale in circa 9 match su 10, quando si considera la probabilità stimata all'ultimo punto del match.

#### Analisi per Classe:
- **P1 wins**: Accuracy = 99.66%
- **P2 wins**: Accuracy = 82.03%

Lo sbilanciamento suggerisce un possibile bias del modello verso P1, che potrebbe derivare da:
1. Asimmetrie nel dataset di training
2. Correlazione tra ordine dei giocatori e ranking/forza
3. Informazioni implicite nella designazione P1/P2

### Confronto con Baseline

Un classificatore naive che predice sempre la classe maggioritaria otterrebbe MLA ≈ 50% (dato il bilanciamento del dataset). Il nostro risultato di 90.85% rappresenta quindi un guadagno sostanziale rispetto alla baseline.

---

## 2. Calibration Plot

### Definizione Formale

Un modello probabilistico è **perfettamente calibrato** se:

```
P(Y=1 | p̂ = p) = p    ∀p ∈ [0,1]
```

Dove p̂ è la probabilità predetta dal modello e Y è il vero outcome.

### Expected Calibration Error (ECE)

L'ECE quantifica il grado di miscalibrazione:

```
ECE = Σⱼ₌₁ᴮ (nⱼ/N) |acc(Bⱼ) - conf(Bⱼ)|
```

dove:
- B è il numero di bin (tipicamente 10)
- Bⱼ è il j-esimo bin di probabilità
- nⱼ è il numero di campioni nel bin j
- N è il totale dei campioni
- acc(Bⱼ) = frazione empirica di outcome positivi nel bin j
- conf(Bⱼ) = probabilità media predetta nel bin j

### Giustificazione Teorica

#### Perché la calibrazione è cruciale:

1. **Interpretabilità Probabilistica**: Solo con un modello calibrato, la probabilità predetta p̂ = 0.7 può essere interpretata come "il modello assegna 70% di probabilità a questo outcome".

2. **Decision Making**: Per applicazioni che richiedono decisioni basate su soglie (betting odds, resource allocation), la calibrazione è essenziale per valutazioni risk-reward accurate.

3. **Affidabilità delle Incertezze**: La calibrazione garantisce che le incertezze espresse dal modello riflettano accuratamente la vera incertezza epistemica.

4. **Complementarietà con l'Accuracy**: Un modello può avere alta accuracy ma pessima calibrazione (es. predice sempre 0.51 o 0.49), rendendo necessarie entrambe le metriche.

### Costruzione del Plot

Il calibration plot visualizza:
- **Asse X**: Probabilità predette (raggruppate in bin)
- **Asse Y**: Frazione empirica di P1 vittorie
- **Linea ideale**: y = x (calibrazione perfetta)

La distanza tra la curva osservata e la linea ideale rappresenta visivamente l'errore di calibrazione.

### Interpretazione dei Risultati

**Risultato Osservato**: ECE = 0.0094

Questo valore estremamente basso indica una calibrazione quasi perfetta. In particolare:

```
Bin [0.00-0.10]: Predetto=0.064, Osservato=0.043  (diff = 0.021)
Bin [0.10-0.20]: Predetto=0.148, Osservato=0.136  (diff = 0.012)
...
Bin [0.90-1.00]: Predetto=0.941, Osservato=0.955  (diff = 0.014)
```

Tutti i bin mostrano discrepanze < 0.025, confermando che:
- Quando il modello predice 30%, P1 vince effettivamente ~30% delle volte
- Quando il modello predice 70%, P1 vince effettivamente ~70% delle volte

### Significato Statistico

Un ECE < 0.05 è generalmente considerato eccellente in letteratura. Il nostro valore di 0.0094 suggerisce che il modello ha imparato non solo a discriminare tra classi, ma anche a quantificare accuratamente la propria incertezza.

---

## 3. Brier Score

### Definizione Formale

Il Brier Score (BS) è una proper scoring rule definita come:

```
BS = (1/N) Σᵢ₌₁ᴺ (p̂ᵢ - yᵢ)²
```

dove:
- p̂ᵢ ∈ [0,1] è la probabilità predetta
- yᵢ ∈ {0,1} è l'outcome vero

Il Brier Score è equivalente al Mean Squared Error (MSE) per classificazione probabilistica.

### Proprietà Teoriche

#### 1. Proper Scoring Rule

Una scoring rule è *proper* se è minimizzata quando il modello riporta le sue vere credenze probabilistiche. Formalmente:

```
E[BS(p̂, Y)] ≥ E[BS(p*, Y)]
```

dove p* è la vera probabilità e p̂ è qualsiasi altra predizione.

Questo garantisce che il modello non è incentivato a "mentire" sulle sue probabilità.

#### 2. Decomposizione di Murphy

Il Brier Score può essere decomposto in tre componenti:

```
BS = Reliability - Resolution + Uncertainty
```

- **Reliability**: Misura la calibrazione (identico concetto dell'ECE)
- **Resolution**: Capacità di discriminare tra outcome diversi
- **Uncertainty**: Incertezza intrinseca dei dati (non controllabile dal modello)

### Brier Skill Score (BSS)

Per contestualizzare il BS, si calcola il BSS rispetto a un baseline:

```
BSS = 1 - (BS_model / BS_baseline)
```

dove BS_baseline è tipicamente il Brier Score di un modello che predice sempre la frequenza media.

### Interpretazione dei Risultati

**Risultato Osservato**: 
- BS = 0.1688
- BS_baseline = 0.2500
- BSS = 0.3249 (32.49% di miglioramento rispetto al baseline)

#### Analisi per Vincitore:
- Match vinti da P1: BS = 0.1638
- Match vinti da P2: BS = 0.1737

La differenza (0.0099) è relativamente piccola, suggerendo che il modello ha performance comparabili per entrambe le classi, nonostante l'accuracy asimmetrica.

### Confronto con Log Loss

Il Brier Score è preferibile alla log loss in questo contesto perché:
1. È meno sensibile a predizioni estreme (non diverge a 0 o 1)
2. È più interpretabile (scala 0-1)
3. Ha la decomposizione di Murphy che separa calibrazione e discriminazione

---

## 4. Time-Weighted Metrics

### Motivazione Teorica

La valutazione uniforme di tutti i punti ignora la struttura temporale del problema. È ragionevole che:
1. Predizioni all'inizio del match siano meno accurate (alta incertezza)
2. Predizioni alla fine del match siano più accurate (più informazione osservata)
3. Il valore informativo delle predizioni aumenti col tempo

### Formalizzazione

Dividiamo ogni match in fasi temporali basate sulla posizione relativa:

```
τᵢ = (posizione_ordinale_punto_i) / (lunghezza_totale_match - 1)
```

Definiamo 4 fasi:
- **Fase 1**: τ ∈ [0.00, 0.25)  "Inizio"
- **Fase 2**: τ ∈ [0.25, 0.50)  "Metà"  
- **Fase 3**: τ ∈ [0.50, 0.75)  "Fine"
- **Fase 4**: τ ∈ [0.75, 1.00]  "Finale"

Per ogni fase, calcoliamo:
1. **Accuracy_fase**: Frazione di predizioni corrette (con soglia 0.5)
2. **Brier_fase**: Brier Score medio
3. **Volatilità**: Deviazione standard delle probabilità predette

### Giustificazione Statistica

#### Information Accumulation

Formalizziamo il match come un processo di diffusione dell'informazione. Sia I(t) l'informazione mutua tra le osservazioni fino al tempo t e il vincitore finale:

```
I(t) = H(Y) - H(Y|X₁:ₜ)
```

Dove:
- H(Y) è l'entropia del vincitore (log 2 per match bilanciati)
- H(Y|X₁:ₜ) è l'entropia condizionale date le osservazioni fino al tempo t

Teorema: I(t) è monotonicamente non-decrescente in t.

Conseguenza: La capacità predittiva del modello dovrebbe migliorare monotonicamente con t, giustificando l'analisi per fase.

### Interpretazione dei Risultati

**Risultati Osservati**:

| Fase         | Punti  | Accuracy | Brier  | Avg Prob | Std Prob |
|-------------|--------|----------|--------|----------|----------|
| Inizio      | 34,316 | 58.75%   | 0.2437 | 0.5026   | 0.1605   |
| Metà        | 34,025 | 69.95%   | 0.2049 | 0.4948   | 0.2555   |
| Fine        | 34,153 | 77.13%   | 0.1562 | 0.5020   | 0.3067   |
| Finale      | 33,874 | 87.55%   | 0.0856 | 0.5028   | 0.3555   |

#### Analisi dei Pattern

1. **Accuracy Crescente**: 58.75% → 87.55%
   - Guadagno complessivo: +28.8 punti percentuali
   - Conferma l'accumulo di informazione nel tempo
   - Trend quasi lineare, suggerendo un flusso informativo costante

2. **Brier Score Decrescente**: 0.2437 → 0.0856
   - Riduzione del 64.9% nell'errore quadratico
   - Improvement più marcato tra fase 3 e 4 (-44.7%)
   - Indica convergenza verso certezza nelle fasi finali

3. **Volatilità Crescente**: 0.1605 → 0.3555
   - L'aumento della deviazione standard è controintuitivo se interpretato come "incertezza"
   - In realtà, riflette **confidence increasing**: il modello produce probabilità più estreme (vicino a 0 o 1)
   - Coerente con la distribuzione bimodale attesa in fasi avanzate

4. **Probabilità Media Stabile**: ~0.50 in tutte le fasi
   - Indica assenza di bias sistematico temporale
   - Il dataset di test è ben bilanciato in tutte le fasi
   - Le predizioni rimangono centrate anche con certezza crescente

### Distribuzione Temporale

La distribuzione uniforme (~25% per fase) è **non-triviale** e merita commento:

```
P(punto i è nella fase f) ≈ 0.25    ∀f
```

Questo implica che:
1. I match hanno distribuzioni di lunghezza sufficientemente omogenee
2. Non ci sono fasi "compresse" o "dilatate" sistematicamente
3. La discretizzazione in quartili è appropriata per il dataset

### Implicazioni per il Deployment

L'analisi time-weighted suggerisce strategie di utilizzo differenziate:

1. **Early-Match (0-25%)**:
   - Accuracy modesta (58%)
   - Usare predizioni con cautela
   - Ideale per identificare upset potenziali

2. **Mid-Match (25-50%)**:
   - Accuracy accettabile (70%)
   - Punto di equilibrio risk-reward
   - Utile per live betting adjustments

3. **Late-Match (50-75%)**:
   - Accuracy buona (77%)
   - Predizioni affidabili per decision-making
   - Momento critico per momentum analysis

4. **Final Phase (75-100%)**:
   - Accuracy eccellente (88%)
   - Alta confidenza, basso rischio
   - Predizioni quasi deterministiche

---

## Sintesi Comparativa delle Metriche

### Confronto con Metriche Tradizionali

#### ROC-AUC Point-wise (non utilizzata)

**Problema**: Calcolare ROC-AUC su tutti i punti individuali soffre di:

1. **Violazione di Indipendenza**:
   ```
   Cov(p̂ᵢ, p̂ᵢ₊₁) ≠ 0  per punti dello stesso match
   ```
   Le assunzioni di base per ROC curve (campioni i.i.d.) sono violate.

2. **Inflazione Artificiale**:
   - Match lunghi contribuiscono più di match corti
   - Un singolo match "facile" può dominare 100+ punti
   - L'AUC risultante non ha interpretazione probabilistica valida

3. **Insensibilità a Oscillazioni Corrette**:
   - Un modello che correttamente riflette momentum shifts sarebbe penalizzato
   - Confonde incertezza epistemica (buona) con rumore (cattivo)

#### Confusion Matrix Point-wise (non utilizzata)

**Problema**: Analogamente, una confusion matrix su tutti i punti:

1. **Pseudo-replication**: 
   - N_matches match diventano N_points "campioni", ma solo N_matches sono indipendenti
   - Standard errors sottostimati di un fattore ~√(avg_match_length)

2. **Interpretation Fallacy**:
   - Cosa significa un "False Positive" a metà match?
   - Un punto classificato come "P2 wins" in un match vinto da P1 potrebbe essere corretto dato lo stato del match

### Complementarietà delle 4 Metriche

Le nostre 4 metriche formano un framework completo:

| Metrica | Misura | Dominio | Interpretazione |
|---------|--------|---------|-----------------|
| MLA | Discriminazione | Match | "Quanto spesso indovino il vincitore?" |
| ECE | Calibrazione | Probabilità | "Posso fidarmi delle probabilità?" |
| Brier | Sharpness + Calibration | Punti | "Quanto sono precise le probabilità?" |
| Time-Weighted | Evolution | Temporale | "Come migliora la conoscenza?" |

Insieme, rispondono a:
1. **Cosa** predice il modello (MLA)
2. **Quanto bene** lo predice (Brier)
3. **Quanto onestamente** lo predice (ECE)
4. **Quando** lo predice bene (Time-Weighted)

---

## Conclusioni Metodologiche

### Validità della Valutazione

L'approccio proposto soddisfa i criteri di una valutazione statistica rigorosa:

1. **Split a Livello di Match**:
   ```python
   train_matches, test_matches = train_test_split(unique_matches, ...)
   ```
   Garantisce che match completi siano interamente in train o test, preservando l'indipendenza.

2. **Stratificazione per Outcome**:
   ```python
   stratify=unique_y
   ```
   Mantiene bilanciamento train/test, riducendo variance nelle stime.

3. **Seed Fisso**:
   ```python
   random_state=42
   ```
   Garantisce riproducibilità degli esperimenti.

### Limitazioni e Considerazioni

1. **Dominio Specifico**: Le metriche sono ottimizzate per tennis; adattamenti sarebbero necessari per altri sport.

2. **Temporal Smoothness**: Non valutiamo la "smoothness" delle traiettorie probabilistiche (salti improvvisi potrebbero essere problematici).

3. **Confidence Intervals**: Le metriche riportate sono stime puntuali; intervalli di confidenza via bootstrap potrebbero quantificare incertezza.

4. **Cross-Validation**: Un'analisi più robusta userebbe k-fold CV a livello di match, al costo di maggiore computational burden.

### Direzioni Future

1. **Scoring Rules Alternative**: Log loss, spherical score per robustezza
2. **Reliability Diagrams Dinamici**: Calibrazione condizionata sulla fase del match  
3. **Shapley Values Temporali**: Attribuzione dell'importanza di feature nel tempo
4. **Conformal Prediction**: Set di predizioni con garanzie statistiche di coverage

---

## Bibliografia di Riferimento

1. **Calibrazione**:
   - Guo, C. et al. (2017). "On Calibration of Modern Neural Networks". ICML.
   - Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning". ICML.

2. **Proper Scoring Rules**:
   - Gneiting, T. & Raftery, A.E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation". JASA.
   - Brier, G.W. (1950). "Verification of Forecasts Expressed in Terms of Probability". Monthly Weather Review.

3. **Time Series Evaluation**:
   - Tashman, L.J. (2000). "Out-of-sample tests of forecasting accuracy: an analysis and review". International Journal of Forecasting.
   - Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy". JBES.

4. **Sports Analytics**:
   - Kovalchik, S. (2016). "Searching for the GOAT of tennis win prediction". Journal of Quantitative Analysis in Sports.
   - Sipko, M. & Knottenbelt, W. (2015). "Machine Learning for the Prediction of Professional Tennis Matches". MEng thesis, Imperial College London.

---

## Appendice: Implementazione Tecnica

### Calcolo della Posizione Relativa

```python
# Posizione ordinale (0, 1, 2, ...) per ogni punto nel match
df['ordinal_position'] = df.groupby('match_id').cumcount()

# Lunghezza totale del match
df['match_length'] = df.groupby('match_id')['ordinal_position'].transform('max') + 1

# Posizione relativa normalizzata [0, 1]
df['relative_position'] = df['ordinal_position'] / (df['match_length'] - 1)
```

### Calcolo dell'ECE

```python
bins = np.linspace(0, 1, n_bins + 1)
bin_indices = np.digitize(y_prob, bins) - 1

empirical_probs = []
predicted_probs = []
counts = []

for i in range(n_bins):
    mask = bin_indices == i
    if mask.sum() > 0:
        empirical_probs.append(y_true[mask].mean())
        predicted_probs.append(y_prob[mask].mean())
        counts.append(mask.sum())

ECE = np.average(
    np.abs(empirical_probs - predicted_probs),
    weights=counts
)
```

### Calcolo del Brier Skill Score

```python
brier_model = brier_score_loss(y_true, y_prob)
baseline_prob = y_true.mean()
brier_baseline = brier_score_loss(
    y_true, 
    np.full_like(y_prob, baseline_prob)
)
brier_skill_score = 1 - (brier_model / brier_baseline)
```

---

**Documento preparato per valutazione accademica**  
**Contesto**: Tesi Magistrale in Statistica / Data Science  
**Livello**: Graduate / Dottorato  
**Ambito**: Apprendimento Statistico, Valutazione Modelli Probabilistici
