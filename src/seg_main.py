import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
import torch
import logging
from src.dataset import get_seg_dataloaders
from src.model import get_seg_model, get_model
from src.seg_train import train_seg_model
from src.utils import setup_logging, stratified_split_by_disease

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cls_model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)
    cls_model.load_state_dict(torch.load('models/best_model.pth'))
    cls_model.eval()

    # Sửa lỗi: Truyền tham số đúng thứ tự và không trùng lặp
    train_loader, val_loader, _ = get_seg_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=pd.DataFrame(),
        image_dir=config['data']['image_dir'],
        bbox_file='data/BBox_List_2017.csv',
        cls_model=cls_model,
        device=device,
        batch_size=config['training']['batch_size']
    )

    model = get_seg_model(
        model_name='chestxray_seg_model',
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    logging.info('Starting segmentation training. Dice and IoU scores will be logged for validation.')
    model = train_seg_model(
        model, train_loader, val_loader, device,
        config['training']['num_epochs'], config['training']['patience'],
        'models',
        accum_steps=4
    )

    logging.info('Training completed. Run seg_test.py to evaluate on val or test set.')
    print('Training completed. Run seg_test.py to evaluate on val or test set.')

if __name__ == '__main__':
    main()