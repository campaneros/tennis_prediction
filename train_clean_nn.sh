#!/bin/bash
# Script per addestrare la rete neurale con features pulite e parametri conservativi

echo "=========================================="
echo "TRAINING NEURAL NETWORK - CLEAN FEATURES"
echo "=========================================="
echo ""
echo "Configurazione:"
echo "  - Features: CLEAN (25 features ORIGINALI, no aggiunte)"
echo "  - Architecture: [128, 64]"
echo "  - Dropout: 0.4"
echo "  - Epochs: 200 (early stopping: 40)"
echo "  - Batch size: 1024"
echo "  - Label smoothing: 0.10-0.90 (moderato)"
echo "  - Temperature: 4.0 (molto conservativa)"
echo "  - Sample weights: cap a 6.0"
echo ""
echo "Critical point features:"
echo "  - is_p1_break_point / is_p2_break_point"
echo "  - is_p1_set_point / is_p2_set_point"
echo "  - is_p1_match_point / is_p2_match_point"
echo ""

source venv/bin/activate

python tennisctl.py train \
  --model-type nn \
  --files data/2015-wimbledon-points.csv \
          data/2016-wimbledon-points.csv \
          data/2017-wimbledon-points.csv \
          data/2018-wimbledon-points.csv \
          data/2019-wimbledon-points.csv \
  --model-out models/nn_clean_optimized.json \
  --gender male \
  --epochs 200 \
  --batch-size 1024 \
  --clean-features

echo ""
echo "=========================================="
echo "Training completato!"
echo "Modello salvato in: models/nn_clean_optimized.json"
echo "=========================================="
echo ""
echo "Per testare il modello esegui:"
echo "  python tennisctl.py predict \\"
echo "    --model models/nn_clean_optimized.json \\"
echo "    --match-id 2019-wimbledon-1701 \\"
echo "    --files data/2019-wimbledon-points.csv \\"
echo "    --plot-dir plots/clean_optimized \\"
echo "    --point-by-point"
