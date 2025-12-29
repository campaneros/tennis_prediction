#!/bin/bash
# Test the newly trained model with reduced temperature

echo "Testing new model (temperature = 3.0)..."

if [ ! -f "models/complete_model_v3.pth" ]; then
    echo "ERROR: Model not found! Training may still be in progress."
    echo "Check with: ps aux | grep retrain_model"
    exit 1
fi

echo -e "\n=== Model info ==="
venv/bin/python -c "
import torch
checkpoint = torch.load('models/complete_model_v3.pth', map_location='cpu')
print(f'Temperature: {checkpoint[\"temperature\"]}')
print(f'Architecture: {checkpoint[\"hidden_sizes\"]}')
print(f'Input size: {checkpoint[\"input_size\"]}')
print(f'Fine-tuned on {checkpoint.get(\"finetune_points\", \"?\"))} points')
"

echo -e "\n=== Running prediction on test match ==="
venv/bin/python tennisctl.py predict \
  --model ./models/complete_model_v3.pth \
  --match-id 2019-wimbledon-1701 \
  --files data/2019-wimbledon-points.csv \
  --plot-dir plots_v3 \
  --point-by-point

echo -e "\n=== Check generated plot ==="
ls -lh plots_v3/match_2019-wimbledon-1701*png

echo -e "\nDone! Check plots_v3/ for the visualization."
