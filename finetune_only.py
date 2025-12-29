"""
Fine-tune the pre-trained model on real data.
"""
from scripts.transfer_learning import fine_tune_on_real_data
import glob
import torch

files = glob.glob('data/*-points.csv')
files = [f for f in files if '2019' not in f]

print(f'Fine-tuning on {len(files)} files')

fine_tune_on_real_data(
    files=files,
    pretrained_path='models/tennis_rules_pretrained_v3.pth',
    output_path='models/complete_model_v3.pth',
    gender='male',
    epochs=30,
    batch_size=1024,
    learning_rate=0.0001,
    temperature=3.0,
    freeze_layers=False,
    device='cpu'  # Force CPU since CUDA not available
)

print('\n' + '='*80)
print('FINE-TUNING COMPLETE!')
print('='*80)
print('Model saved to: models/complete_model_v3.pth')
