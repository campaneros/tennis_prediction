"""
Complete training pipeline with fixed hyperparameters.
This will:
1. Generate 30k synthetic matches
2. Pre-train on synthetic data (T=3.0, low penalties, weight decay)
3. Fine-tune on real data (same settings)
4. Save as complete_model_v4.pth
"""
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from tennis_simulator import generate_training_dataset
import pretrain_tennis_rules
import transfer_learning
from glob import glob

print('='*80)
print('COMPLETE TRAINING PIPELINE - V4')
print('Fixes applied:')
print('  - Temperature: 3.0 (was 12.0)')
print('  - Match point penalty: 0.5 (was 5.0)')
print('  - Set point penalty: 0.3 (was 2.0)')
print('  - Weight decay: 0.01 (L2 regularization)')
print('  - Gradient clipping: max_norm=1.0')
print('='*80)

# Step 1: Generate synthetic data
print('\n[STEP 1/3] Generating 30,000 synthetic matches...')
synthetic_path = 'data/synthetic_training_30k_v4.csv'

df_synthetic = generate_training_dataset(
    n_matches=30000,
    output_path=synthetic_path,
    best_of_5=True,
    seed=42
)

print(f'✓ Generated {len(df_synthetic):,} points from 30k matches')
print(f'  Saved to: {synthetic_path}')

# Check label distribution
p1_win_rate = df_synthetic['p1_wins_match'].mean()
print(f'  P1 win rate in synthetic data: {p1_win_rate:.3f} (should be ~0.50)')

# Step 2: Pre-train on synthetic data
print('\n[STEP 2/3] Pre-training on synthetic data...')
pretrained_path = 'models/tennis_rules_pretrained_v4.pth'

model = pretrain_tennis_rules.pretrain_tennis_rules(
    n_matches=30000,
    epochs=20,
    batch_size=512,
    learning_rate=0.001,
    device='auto',
    save_path=pretrained_path,
    best_of_5=True,
    temperature=3.0,
    weight_decay=0.01
)

print(f'✓ Pre-training complete')
print(f'  Saved to: {pretrained_path}')

# Step 3: Fine-tune on real data
print('\n[STEP 3/3] Fine-tuning on real match data...')

# Get all points files
train_files = sorted(glob('data/*-points.csv'))
print(f'  Found {len(train_files)} point files')

# Exclude test sets (2019 Wimbledon for testing)
train_files = [f for f in train_files if '2019-wimbledon' not in f]
print(f'  Using {len(train_files)} files for training (excluded 2019 Wimbledon for test)')

final_model_path = 'models/complete_model_v4.pth'

fine_tuned_model = transfer_learning.fine_tune_on_real_data(
    files=train_files,
    pretrained_path=pretrained_path,
    output_path=final_model_path,
    gender='male',
    epochs=30,
    batch_size=1024,
    learning_rate=0.0001,
    temperature=3.0
)

print('\n' + '='*80)
print('TRAINING COMPLETE!')
print('='*80)
print(f'\nFinal model saved to: {final_model_path}')
print('\nTo test on 2019 Wimbledon final:')
print('  python tennisctl.py predict --model ./models/complete_model_v4.pth \\')
print('    --match-id 2019-wimbledon-1701 \\')
print('    --files data/2019-wimbledon-points.csv \\')
print('    --plot-dir plots --point-by-point')
print('\nExpected improvements:')
print('  - Probability should respond to set score changes')
print('  - Start at ~0.50, not stuck at 0.79')
print('  - End at 0.95+ when match ends, not 0.79')
print('  - Model should use P1_sets/P2_sets features effectively')
