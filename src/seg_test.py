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
from tqdm import tqdm
from src.dataset import get_seg_dataloaders
from src.model import get_seg_model, get_model
from src.utils import setup_logging
from PIL import Image, ImageDraw, ImageFont

def evaluate_seg_model(config, split='test'):
    setup_logging(config['logging']['log_dir'], config['logging']['log_file'])

    df = pd.read_excel(config['data']['data_file'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cls_model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)
    cls_model.load_state_dict(torch.load('models/best_model.pth'))
    cls_model.eval()

    if split == 'test':
        split_df = pd.read_csv('data/test_split.csv')
    elif split == 'val':
        split_df = pd.read_csv('data/val_split.csv')
    else:
        raise ValueError(f"Split {split} not supported. Use 'test' or 'val'.")

    # Đọc file BBox_List_2017.csv để lấy bbox
    bbox_df = pd.read_csv('data/BBox_List_2017.csv')
    bbox_mapping = {}
    diseases = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pneumothorax', 'Infiltration']
    for idx, row in bbox_df.iterrows():
        image_index = row['Image Index']
        label = row['Finding Label']
        if label in diseases:
            x = float(row['Bbox [x'])
            y = float(row['y'])
            w = float(row['w'])
            h = float(row['h]'])
            bbox = [x, y, w, h]
            if image_index not in bbox_mapping:
                bbox_mapping[image_index] = []
            bbox_mapping[image_index].append((diseases.index(label), bbox))

    _, _, split_loader = get_seg_dataloaders(
        train_df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        test_df=split_df,
        image_dir=config['data']['image_dir'],
        bbox_file='data/BBox_List_2017.csv',
        cls_model=cls_model,
        device=device,
        batch_size=config['training']['batch_size']
    )

    logging.info(f'Segmentation {split.capitalize()}: {len(split_loader.dataset)} images')
    print(f'Segmentation {split.capitalize()}: {len(split_loader.dataset)} images')

    model = get_seg_model(
        model_name='chestxray_seg_model',
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    model.load_state_dict(torch.load('models/best_seg_model.pth'))
    model.eval()

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    eval_start_time = time.time()
    split_seg_masks = []
    split_gt_masks = []

    num_samples_to_save = 5
    saved_samples = 0

    eval_progress = tqdm(split_loader, desc=f'Evaluating on {split.capitalize()} set', leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_progress):
            images, masks, heatmaps = batch
            images, masks, heatmaps = images.to(device), masks.to(device), heatmaps.to(device)
            seg_mask = model(images, heatmaps)
            
            split_seg_masks.append(seg_mask.cpu().numpy())
            split_gt_masks.append(masks.cpu().numpy())

            if saved_samples < num_samples_to_save:
                for i in range(images.size(0)):
                    if saved_samples >= num_samples_to_save:
                        break

                    image_idx = batch_idx * config['training']['batch_size'] + i
                    if image_idx >= len(split_loader.dataset):
                        break
                    image_info = split_loader.dataset.df.iloc[image_idx]
                    image_name = image_info['Image Index']
                    image_path = os.path.join(config['data']['image_dir'], image_name)
                    original_image = Image.open(image_path).convert('RGB')
                    original_image = original_image.resize((224, 224))

                    original_image_np = np.array(original_image)

                    heatmap = seg_mask[i].cpu().numpy()
                    gt_mask = masks[i].cpu().numpy()

                    for class_idx, disease in enumerate(diseases):
                        heatmap_class = heatmap[class_idx]
                        gt_mask_class = gt_mask[class_idx]

                        # Chuẩn hóa heatmap với ngưỡng để làm nổi bật vùng bệnh
                        heatmap_max = heatmap_class.max()
                        heatmap_min = heatmap_class.min()
                        if heatmap_max > heatmap_min:
                            heatmap_class = (heatmap_class - heatmap_min) / (heatmap_max - heatmap_min)
                        else:
                            heatmap_class = np.zeros_like(heatmap_class)
                        # Áp dụng ngưỡng để làm nổi bật vùng bệnh
                        threshold = 0.3  # Ngưỡng 0.3 để làm nổi bật các vùng có giá trị cao
                        heatmap_class = np.where(heatmap_class > threshold, heatmap_class, 0)
                        # Chuẩn hóa lại để các giá trị > threshold nằm trong khoảng [0, 255]
                        if heatmap_class.max() > 0:
                            heatmap_class = (heatmap_class / heatmap_class.max() * 255).astype(np.uint8)
                        else:
                            heatmap_class = heatmap_class.astype(np.uint8)

                        logging.info(f'Heatmap {disease} for {image_name}: min={heatmap_class.min()}, max={heatmap_class.max()}')

                        heatmap_colored = np.zeros((224, 224, 3), dtype=np.uint8)
                        for x in range(224):
                            for y in range(224):
                                value = heatmap_class[x, y]
                                if value < 85:
                                    r = 0
                                    g = int(value * 3)
                                    b = 255 - int(value * 3)
                                elif value < 170:
                                    r = int((value - 85) * 3)
                                    g = 255
                                    b = 0
                                else:
                                    r = 255
                                    g = 255 - int((value - 170) * 3)
                                    b = 0
                                heatmap_colored[x, y] = [r, g, b]

                        heatmap_image = Image.fromarray(heatmap_colored)
                        heatmap_image_np = np.array(heatmap_image)

                        alpha = 0.7
                        overlay_image_np = (alpha * heatmap_image_np + (1 - alpha) * original_image_np).astype(np.uint8)
                        overlay_image = Image.fromarray(overlay_image_np)

                        draw = ImageDraw.Draw(overlay_image)
                        # Lấy bbox trực tiếp từ bbox_mapping
                        if image_name in bbox_mapping:
                            for label_idx, bbox in bbox_mapping[image_name]:
                                if label_idx == class_idx:  # Chỉ vẽ bbox cho lớp bệnh hiện tại
                                    x, y, w, h = bbox
                                    # Scale tọa độ từ 1024x1024 xuống 224x224
                                    x_scale = 224 / 1024
                                    y_scale = 224 / 1024
                                    x, w = int(x * x_scale), int(w * x_scale)
                                    y, h = int(y * y_scale), int(h * y_scale)
                                    x_end = min(x + w, 224)
                                    y_end = min(y + h, 224)
                                    x = max(x, 0)
                                    y = max(y, 0)

                                    draw.rectangle([x, y, x_end, y_end], outline='green', width=2)

                                    try:
                                        font = ImageFont.truetype("arial.ttf", 12)
                                    except:
                                        font = ImageFont.load_default()
                                    text_position = (x, max(0, y - 15))
                                    draw.text((text_position[0] + 1, text_position[1] + 1), disease, font=font, fill='black')
                                    draw.text((text_position[0] - 1, text_position[1] - 1), disease, font=font, fill='black')
                                    draw.text((text_position[0] + 1, text_position[1] - 1), disease, font=font, fill='black')
                                    draw.text((text_position[0] - 1, text_position[1] + 1), disease, font=font, fill='black')
                                    draw.text(text_position, disease, font=font, fill='white')

                        try:
                            font = ImageFont.truetype("arial.ttf", 10)
                        except:
                            font = ImageFont.load_default()
                        image_name_text = f"Image: {image_name}"
                        image_name_position = (5, 5)
                        draw.text((image_name_position[0] + 1, image_name_position[1] + 1), image_name_text, font=font, fill='black')
                        draw.text((image_name_position[0] - 1, image_name_position[1] - 1), image_name_text, font=font, fill='black')
                        draw.text((image_name_position[0] + 1, image_name_position[1] - 1), image_name_text, font=font, fill='black')
                        draw.text((image_name_position[0] - 1, image_name_position[1] + 1), image_name_text, font=font, fill='black')
                        draw.text(image_name_position, image_name_text, font=font, fill='white')

                        output_filename = f"{image_name.split('.')[0]}_{disease}.png"
                        output_path = os.path.join(output_dir, output_filename)
                        overlay_image.save(output_path)

                    saved_samples += 1

    eval_time = time.time() - eval_start_time

    split_seg_masks = np.vstack(split_seg_masks)
    split_gt_masks = np.vstack(split_gt_masks)

    split_seg_masks = torch.tensor(split_seg_masks)
    split_gt_masks = torch.tensor(split_gt_masks)
    pred = (split_seg_masks > 0.5).float()

    intersection = (pred * split_gt_masks).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + split_gt_masks.sum(dim=(2, 3)) - intersection
    dice_scores = (2. * intersection + 1e-5) / (union + intersection + 1e-5)
    iou_scores = (intersection + 1e-5) / (union + 1e-5)
    dice_scores = dice_scores.mean(dim=0)
    iou_scores = iou_scores.mean(dim=0)

    correct_pixels = (pred == split_gt_masks).float().sum(dim=(2, 3))
    total_pixels = torch.prod(torch.tensor(split_gt_masks.shape[2:]))
    pixel_accuracy = correct_pixels / total_pixels
    pixel_accuracy = pixel_accuracy.mean(dim=0)

    diseases = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pneumothorax', 'Infiltration']
    logging.info(f'\n{split.capitalize()} Segmentation Results:')
    print(f'\n{split.capitalize()} Segmentation Results:')
    for i, disease in enumerate(diseases):
        result = f'{disease}:\n' \
                 f'  Dice Score: {dice_scores[i]:.4f}\n' \
                 f'  IoU Score: {iou_scores[i]:.4f}\n' \
                 f'  Pixel Accuracy: {pixel_accuracy[i]:.4f}'
        logging.info(result)
        print(result + '\n')

    mean_dice = torch.mean(dice_scores).item()
    mean_iou = torch.mean(iou_scores).item()
    mean_pixel_accuracy = torch.mean(pixel_accuracy).item()
    mean_result = f'Mean Results:\n' \
                  f'  Mean Dice: {mean_dice:.4f}\n' \
                  f'  Mean IoU: {mean_iou:.4f}\n' \
                  f'  Mean Pixel Accuracy: {mean_pixel_accuracy:.4f}\n' \
                  f'Evaluation Time: {eval_time:.2f}s'
    logging.info(mean_result)
    print(mean_result)

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model on test or val split.')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'], help='Split to evaluate (test or val)')
    args = parser.parse_args()

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    evaluate_seg_model(config, split=args.split)

if __name__ == '__main__':
    main()