import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import logging
import pandas as pd
import yaml
import argparse
import time
from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import calculate_metrics, setup_logging, calculate_localization_metrics

def evaluate_model(config, split='test'):
    setup_logging(config['logging']['log_dir'], config['logging']['log_file'])

    df = pd.read_excel(config['data']['data_file'])
    labels = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pneumothorax', 'Infiltration']

    if split == 'test':
        split_df = pd.read_csv('data/test_split.csv')
    elif split == 'val':
        split_df = pd.read_csv('data/val_split.csv')
    else:
        raise ValueError(f"Split {split} not supported. Use 'test' or 'val'.")

    _, _, split_loader = get_dataloaders(
        pd.DataFrame(), pd.DataFrame(), split_df,
        config['data']['image_dir'], config['training']['batch_size'],
        use_metadata=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()

    eval_start_time = time.time()
    split_labels = []
    split_scores = []
    split_heatmaps = []

    with torch.no_grad():
        for batch in split_loader:
            if len(batch) == 3:
                images, labels, metadata = batch
                images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
                outputs, heatmap = model(images, metadata)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs, heatmap = model(images)
            outputs = torch.sigmoid(outputs)
            split_labels.append(labels.cpu().numpy())
            split_scores.append(outputs.cpu().numpy())
            split_heatmaps.append(heatmap.cpu().numpy())

    eval_time = time.time() - eval_start_time

    split_labels = np.vstack(split_labels)
    split_scores = np.vstack(split_scores)
    auc_scores, acc_scores, f1_scores = calculate_metrics(split_labels, split_scores, split_scores)

    # So sánh với benchmark (Table 17 trong ARXIV_V5_CHESTXRAY.pdf)
    diseases = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pneumothorax', 'Infiltration']
    benchmark_auc = [0.8100, 0.7585, 0.7003, 0.7993, 0.6614]  # Từ Table 17

    logging.info(f'\n{split.capitalize()} Results:')
    print(f'\n{split.capitalize()} Results:')
    for i, disease in enumerate(diseases):
        result = f'{disease}:\n' \
                 f'  AUC: {auc_scores[i]:.4f} (Benchmark: {benchmark_auc[i]})\n' \
                 f'  Accuracy: {acc_scores[i]:.4f}\n' \
                 f'  F1 Score: {f1_scores[i]:.4f}'
        if auc_scores[i] > benchmark_auc[i]:
            result += '\n  -> Surpassed benchmark!'
        logging.info(result)
        print(result + '\n')

    mean_auc = np.mean(auc_scores)
    mean_acc = np.mean(acc_scores)
    mean_f1 = np.mean(f1_scores)
    mean_result = f'Mean Results:\n' \
                  f'  Mean AUC: {mean_auc:.4f}\n' \
                  f'  Mean ACC: {mean_acc:.4f}\n' \
                  f'  Mean F1: {mean_f1:.4f}\n' \
                  f'Evaluation Time: {eval_time:.2f}s'
    logging.info(mean_result)
    print(mean_result)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test or val split.')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'], help='Split to evaluate (test or val)')
    args = parser.parse_args()

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    evaluate_model(config, split=args.split)

if __name__ == '__main__':
    main()