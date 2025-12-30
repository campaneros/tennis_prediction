#!/usr/bin/env python3
"""Fine-tune v5 pre-trained model on real data"""
import sys
sys.path.insert(0, 'scripts')

from transfer_learning import fine_tune_on_real_data
from glob import glob

print('='*80)
print('FINE-TUNING MODEL V5')
print('Using balanced pre-trained model')
print('='*80)

# Get training files
train_files = sorted(glob('data/*-points.csv'))
train_files = [f for f in train_files if '2019-wimbledon' not in f]
print(f'\nFound {len(train_files)} files for training')
print('(Excluded 2019 Wimbledon for testing)')

# Fine-tune
print('\nStarting fine-tuning...')
fine_tune_on_real_data(
    files=train_files,
    pretrained_path='models/tennis_rules_pretrained_v5.pth',
    output_path='models/complete_model_v5.pth',
    gender='male',
    epochs=30,
    batch_size=1024,
    learning_rate=0.0001,
    temperature=3.0
)

print('\n' + '='*80)
print('COMPLETE!')
print('='*80)
print('Model saved to: models/complete_model_v5.pth')
print('\nTo test:')
print('  python tennisctl.py predict --model ./models/complete_model_v5.pth \\')
print('    --match-id 2019-wimbledon-1701 \\')
print('    --files data/2019-wimbledon-points.csv \\')
print('    --plot-dir plots/v5 --point-by-point')
