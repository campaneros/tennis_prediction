"""
Retrain model from scratch without artificial probability constraints.
"""
import glob
import torch
from scripts.pretrain_tennis_rules import pretrain_tennis_rules
from scripts.transfer_learning import fine_tune_on_real_data

print("="*80)
print("STEP 1: PRE-TRAINING ON SYNTHETIC DATA")
print("="*80)

# Pre-train on synthetic data (without artificial constraints)
pretrain_tennis_rules(
    n_matches=30000,
    epochs=40,
    batch_size=2048,
    temperature=3.0,  # Reduced from 12.0 to avoid probability collapse
    output_path='models/tennis_rules_pretrained_v3.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("\n" + "="*80)
print("STEP 2: FINE-TUNING ON REAL DATA")
print("="*80)

# Get all data files (excluding 2019 which contains test match 1701)
files = glob.glob('data/*-points.csv')
files = [f for f in files if '2019' not in f]

print(f'\nTraining on {len(files)} files (excluding 2019 test data)')

fine_tune_on_real_data(
    files=files,
    pretrained_path='models/tennis_rules_pretrained_v3.pth',
    output_path='models/complete_model_v3.pth',
    gender='male',
    epochs=30,
    batch_size=1024,
    learning_rate=0.0001,
    temperature=3.0,  # Reduced from 12.0 to avoid probability collapse
    freeze_layers=False
)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Model saved to: models/complete_model_v3.pth")
