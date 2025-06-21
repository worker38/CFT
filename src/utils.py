import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import logging
import os
import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Trọng số cho từng lớp
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets.long()]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def setup_logging(log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def stratified_split_by_disease(df, train_split, val_split, test_split, labels, random_state=42):
    patient_ids = df['Patient ID'].unique()
    train_pids, temp_pids = train_test_split(patient_ids, test_size=(val_split + test_split), random_state=random_state)
    val_pids, test_pids = train_test_split(temp_pids, test_size=test_split/(val_split + test_split), random_state=random_state)

    train_df = df[df['Patient ID'].isin(train_pids)]
    val_df = df[df['Patient ID'].isin(val_pids)]
    test_df = df[df['Patient ID'].isin(test_pids)]

    def get_label_distribution(df, labels):
        return df[labels].mean()

    print("Initial distribution:")
    print("Train:", get_label_distribution(train_df, labels))
    print("Val:", get_label_distribution(val_df, labels))
    print("Test:", get_label_distribution(test_df, labels))

    max_iter = 100
    for _ in range(max_iter):
        train_dist = get_label_distribution(train_df, labels)
        val_dist = get_label_distribution(val_df, labels)
        test_dist = get_label_distribution(test_df, labels)

        mean_dist = (train_dist + val_dist + test_dist) / 3
        train_dev = np.abs(train_dist - mean_dist).mean()
        val_dev = np.abs(val_dist - mean_dist).mean()
        test_dev = np.abs(test_dist - mean_dist).mean()

        if train_dev < 0.05 and val_dev < 0.05 and test_dev < 0.05:
            break

        for _ in range(10):
            pid = np.random.choice(patient_ids)
            current_set = 'train' if pid in train_pids else 'val' if pid in val_pids else 'test'
            new_set = np.random.choice(['train', 'val', 'test'])

            if current_set != new_set:
                if current_set == 'train':
                    train_pids = np.setdiff1d(train_pids, [pid])
                elif current_set == 'val':
                    val_pids = np.setdiff1d(val_pids, [pid])
                else:
                    test_pids = np.setdiff1d(test_pids, [pid])

                if new_set == 'train':
                    train_pids = np.append(train_pids, pid)
                elif new_set == 'val':
                    val_pids = np.append(val_pids, pid)
                else:
                    test_pids = np.append(test_pids, pid)

                train_df = df[df['Patient ID'].isin(train_pids)]
                val_df = df[df['Patient ID'].isin(val_pids)]
                test_df = df[df['Patient ID'].isin(test_pids)]

    print("\nFinal distribution:")
    print("Train:", get_label_distribution(train_df, labels))
    print("Val:", get_label_distribution(val_df, labels))
    print("Test:", get_label_distribution(test_df, labels))

    return train_df, val_df, test_df

def find_optimal_threshold(y_true, y_score):
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_thresholds = []
    for i in range(y_true.shape[1]):
        best_f1 = 0
        best_threshold = 0.5
        for t in thresholds:
            y_pred = (y_score[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        best_thresholds.append(best_threshold)
    return best_thresholds

def calculate_metrics(y_true, y_pred, y_score):
    auc_scores = []
    acc_scores = []
    f1_scores = []
    optimal_thresholds = find_optimal_threshold(y_true, y_score)

    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_score[:, i])
        acc = accuracy_score(y_true[:, i], (y_pred[:, i] > optimal_thresholds[i]).astype(int))
        f1 = f1_score(y_true[:, i], (y_pred[:, i] > optimal_thresholds[i]).astype(int))

        auc_scores.append(auc)
        acc_scores.append(acc)
        f1_scores.append(f1)

    return auc_scores, acc_scores, f1_scores

def calculate_localization_metrics(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    # Tính IoU và độ chính xác localization
    # (Cần ground truth bounding box từ file BBox_List_2017.csv)
    # Đây là placeholder, cần triển khai thực tế nếu có dữ liệu bounding box
    return 0.0, 0.0  # accuracy, average false positives

# Thêm hàm dice_loss
def dice_loss(pred, target, smooth=1e-5):
    """
    Tính Dice Loss cho bài toán segmentation.
    
    Args:
        pred (torch.Tensor): Mặt nạ dự đoán, shape [batch_size, num_classes, H, W]
        target (torch.Tensor): Mặt nạ ground truth, shape [batch_size, num_classes, H, W]
        smooth (float): Hằng số làm mượt để tránh chia cho 0
    
    Returns:
        torch.Tensor: Giá trị Dice Loss
    """
    pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
    target = target.contiguous().view(target.size(0), target.size(1), -1)  # [B, C, H*W]
    
    intersection = (pred * target).sum(dim=2)  # [B, C]
    union = pred.sum(dim=2) + target.sum(dim=2)  # [B, C]
    
    dice = (2. * intersection + smooth) / (union + smooth)  # [B, C]
    return 1 - dice.mean()  # Trung bình trên tất cả lớp và batch