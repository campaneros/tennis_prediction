#!/usr/bin/env python3
"""
Monitor training progress and test final model v5
"""
import time
import os
from pathlib import Path

print('='*80)
print('MONITORING TRAINING V5')
print('='*80)

# Check if still running
log_file = 'pretrain_v5_log.txt'
model_file = 'models/tennis_rules_pretrained_v5.pth'

print('\nWaiting for pre-training to complete...')
print('(This will take approximately 20-30 minutes)')
print()

last_size = 0
no_change_count = 0

while not os.path.exists(model_file):
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        if size > last_size:
            # Read last few lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Show last non-empty line
                    for line in reversed(lines[-5:]):
                        if line.strip():
                            print(f'\r  {line.strip()[:70]:<70}', end='', flush=True)
                            break
            last_size = size
            no_change_count = 0
        else:
            no_change_count += 1
            if no_change_count > 60:  # 5 minutes no change
                print('\n⚠️  Training seems stuck (no progress for 5 minutes)')
                break
    
    time.sleep(5)

if os.path.exists(model_file):
    print('\n\n✅ Pre-training complete!')
    print(f'   Model saved: {model_file}')
    
    # Show summary
    if os.path.exists(log_file):
        print('\n' + '='*80)
        print('TRAINING SUMMARY (last 20 lines):')
        print('='*80)
        os.system(f'tail -20 {log_file}')
    
    print('\n' + '='*80)
    print('NEXT STEP: Fine-tuning')
    print('='*80)
    print('Run: venv/bin/python run_finetune_v5.py')
else:
    print('\n❌ Training did not complete')
    print(f'   Check {log_file} for errors')
