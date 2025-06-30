# CFT: A Hybrid Machine Learning Method For Diagnosing The Five Common Lung Diseases Based On X-ray Images

# Chest X-ray Disease Classification and Segmentation

This project implements a deep learning pipeline for **multi-label classification** and **lesion segmentation** of chest X-ray images. It supports five thoracic diseases: **Cardiomegaly**, **Effusion**, **Atelectasis**, **Pneumothorax**, and **Infiltration**. The pipeline uses **EfficientNet-B3** as the backbone, **Feature Pyramid Network (FPN)** for multi-scale feature extraction, and a **Mini-Transformer** for enhanced classification. The segmentation model generates lesion masks guided by heatmaps from the classification output.

## Project Structure

- **src/**: Source code directory  
  - `dataset.py`: Defines classification and segmentation datasets, including augmentation and preprocessing.  
  - `model.py`: Implements classification and segmentation models using EfficientNet-B3, FPN, and Mini-Transformer.  
  - `train.py`: Classification training script using Focal Loss and gradient accumulation.  
  - `seg_train.py`: Segmentation training script using BCE Loss + Dice Loss.  
  - `utils.py`: Includes utility functions like Focal Loss, Dice Loss, stratified split, and evaluation metrics.  
- **main.py**: Entry point for classification training.  
- **seg_main.py**: Entry point for segmentation training.  
- **test.py**: Classification evaluation with AUC comparison.  
- **seg_test.py**: Segmentation evaluation and visualization.  
- **configs/config.yaml**: Configuration file for model, data, and training parameters.  
- **data/**: Contains data splits and bounding box annotations.  
- **models/**: Saved trained models.  
- **results/**: Visual results for segmentation.

## Features

- **Classification**:
  - Multi-label classification with Focal Loss.
  - Metadata integration (age, gender, view position).
  - Heatmap generation for weak lesion localization.

- **Segmentation**:
  - Disease-specific mask generation using bounding boxes and classification heatmaps.

- **Data Handling**:
  - Stratified patient-based data splitting for balanced train/val/test sets.

- **Evaluation**:
  - **Classification**: AUC, Accuracy, F1 Score — compared with `ARXIV_V5_CHESTXRAY.pdf` benchmarks.
  - **Segmentation**: Dice score, IoU, and pixel accuracy.

- **Training**:
  - Early stopping, cosine annealing scheduler, gradient accumulation.

- **Visualization**:
  - Saves color heatmaps and bounding box overlays on segmentation outputs.

## Requirements

- Python ≥ 3.8  
- torch  
- torchvision  
- pandas  
- numpy  
- scikit-learn  
- timm  
- pyyaml  
- tqdm  
- pillow  

Install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn timm pyyaml tqdm pillow
```

## Dataset

- **Inputs**:
  - Chest X-ray images in `config['data']['image_dir']`
  - Labels and metadata in Excel file at `config['data']['data_file']`
  - Bounding boxes in `data/BBox_List_2017.csv`

- **Preprocessing**:
  - Resize to 224x224, normalize with ImageNet mean/std.
  - Augmentations: random flip, rotation, color jitter, Gaussian blur, random erasing.
  - Normalize metadata: scale age to [0, 1], binary encode gender and view position.

## Configuration Highlights

- **Model**:
  - `name`: `chestxray_model` (classification) or `chestxray_seg_model` (segmentation)
  - `num_classes`: 5
  - `pretrained`: true (ImageNet)

- **Data**:
  - `data_file`: Path to label file
  - `image_dir`: Path to image folder
  - `train_split`, `val_split`, `test_split`: e.g., 0.7, 0.15, 0.15

- **Training**:
  - `batch_size`: 16
  - `num_epochs`: 50
  - `patience`: 5
  - `learning_rate`: 0.0001

## Sample Results 

| Disease         | AUC        | Accuracy   | F1 Score   |
|-----------------|------------|------------|------------|
| Cardiomegaly    | 0.8769     | 0.9418     | 0.4200     |
| Effusion        | 0.8781     | 0.8344     | 0.6488     |
| Atelectasis     | 0.8042     | 0.7848     | 0.5131     |
| Pneumothorax    | 0.8698     | 0.9287     | 0.4828     |
| Infiltration    | 0.7160     | 0.6883     | 0.5370     |
| **Average**     | **0.8290** | **0.8356** | **0.5203** |

## Notes

- Make sure `BBox_List_2017.csv` includes: `Image Index`, `Finding Label`, `Bbox [x`, `y`, `w`, `h]`
- Classification model must be trained before generating segmentation heatmaps
- `seg_test.py` assumes `arial.ttf` is available for drawing text. Falls back to default font if missing.


