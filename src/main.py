import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.utils import setup_logging, stratified_split_by_disease, FocalLoss

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config['logging']['log_dir'], config['logging']['log_file'])

    df = pd.read_excel(config['data']['data_file'])
    labels = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pneumothorax', 'Infiltration']

    train_df, val_df, test_df = stratified_split_by_disease(
        df,
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split'],
        labels
    )

    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train_split.csv', index=False)
    val_df.to_csv('data/val_split.csv', index=False)
    test_df.to_csv('data/test_split.csv', index=False)

    logging.info(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, pd.DataFrame(),
        config['data']['image_dir'],
        config['training']['batch_size'],
        use_metadata=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # Focal Loss với trọng số dựa trên tần suất lớp
    class_freq = train_df[labels].sum().values
    alpha = 1.0 / (class_freq / class_freq.sum())
    alpha = alpha / alpha.sum() * len(labels)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    logging.info('Starting training. AUC, ACC, F1 will be logged for validation.')
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        config['training']['num_epochs'], config['training']['patience'],
        'models',
        accum_steps=4
    )

    logging.info('Training completed. Run test.py to evaluate on val or test set.')
    print('Training completed. Run test.py to evaluate on val or test set.')

if __name__ == '__main__':
    main()