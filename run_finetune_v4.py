#!/usr/bin/env python3
"""Fine-tune pre-trained model on real data"""
import sys
sys.path.insert(0, 'scripts')

from transfer_learning import fine_tune_on_real_data
from glob import glob

print('='*80)
print('FINE-TUNING ON REAL DATA - V4')
print('='*80)

# Get all training files
train_files = sorted(glob('data/*-points.csv'))
print(f'\n[1/3] Found {len(train_files)} point files')

# Exclude test set (2019 Wimbledon)
train_files = [f for f in train_files if '2019-wimbledon' not in f]
print(f'  Using {len(train_files)} files for training')
print(f'  (Excluded 2019 Wimbledon for testing)')

# Fine-tune
print('\n[2/3] Fine-tuning...')
fine_tune_on_real_data(
    files=train_files,
    pretrained_path='models/tennis_rules_pretrained_v4.pth',
    output_path='models/complete_model_v4.pth',
    gender='male',
    epochs=30,
    batch_size=1024,
    learning_rate=0.0001,
    temperature=3.0
)

print('\n[3/3] Complete!')
print('='*80)
print('Model saved to: models/complete_model_v4.pth')
print('='*80)
