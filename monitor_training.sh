#!/bin/bash
# Monitor training progress

echo "Training process status:"
ps aux | grep retrain_model.py | grep -v grep || echo "  Not running"

echo -e "\n=== Latest training output (last 40 lines) ==="
tail -40 retrain_v3.log

echo -e "\n=== Check for model files ==="
ls -lh models/tennis_rules_pretrained_v3.pth models/complete_model_v3.pth 2>/dev/null || echo "  Models not yet created"
