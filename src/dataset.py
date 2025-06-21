import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import random

class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, use_metadata=False):
        self.df = df.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.labels = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pneumothorax', 'Infiltration']
        self.use_metadata = use_metadata
        if self.use_metadata:
            self.df.loc[:, 'Patient Age'] = self.df['Patient Age'].clip(0, 100) / 100.0
            self.df.loc[:, 'Patient Gender'] = self.df['Patient Gender'].map({'M': 1, 'F': 0})
            self.df.loc[:, 'View Position'] = self.df['View Position'].map({'PA': 1, 'AP': 0})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['Image Index'])
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx][self.labels].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        if self.use_metadata:
            metadata = self.df.iloc[idx][['Patient Age', 'Patient Gender', 'View Position']].values.astype(np.float32)
            return image, label, torch.tensor(metadata)
        return image, label

class ChestXraySegDataset(Dataset):
    def __init__(self, df, image_dir, bbox_df, cls_model=None, device='cpu', transform=None):
        self.df = df.copy()
        self.image_dir = image_dir
        self.bbox_df = bbox_df
        self.cls_model = cls_model
        self.device = device
        self.transform = transform
        self.labels = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'Pneumothorax', 'Infiltration']
        self.image_size = (224, 224)

        self.df = self.df[self.df['Image Index'].isin(self.bbox_df['Image Index'])]

        self.bbox_mapping = {}
        for idx, row in self.bbox_df.iterrows():
            image_index = row['Image Index']
            label = row['Finding Label']
            if label in self.labels:
                # Đọc trực tiếp từ 4 cột riêng: 'Bbox [x', 'y', 'w', 'h]'
                x = float(row['Bbox [x'])
                y = float(row['y'])
                w = float(row['w'])
                h = float(row['h]'])
                bbox = [x, y, w, h]
                if image_index not in self.bbox_mapping:
                    self.bbox_mapping[image_index] = []
                self.bbox_mapping[image_index].append((self.labels.index(label), bbox))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['Image Index'])
        image = Image.open(img_name).convert('RGB')
        
        mask = np.zeros((len(self.labels), self.image_size[0], self.image_size[1]), dtype=np.float32)
        image_index = self.df.iloc[idx]['Image Index']
        original_size = (1024, 1024)

        if image_index in self.bbox_mapping:
            for label_idx, bbox in self.bbox_mapping[image_index]:
                x_scale = self.image_size[0] / original_size[0]
                y_scale = self.image_size[1] / original_size[1]
                x, y, w, h = bbox
                x, w = int(x * x_scale), int(w * x_scale)
                y, h = int(y * y_scale), int(h * y_scale)

                x_end = min(x + w, self.image_size[0])
                y_end = min(y + h, self.image_size[1])
                x = max(x, 0)
                y = max(y, 0)

                mask[label_idx, y:y_end, x:x_end] = 1.0

        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.float32)

        heatmap = None
        if self.cls_model is not None:
            self.cls_model.eval()
            with torch.no_grad():
                image_tensor = image.unsqueeze(0).to(self.device)
                _, heatmap = self.cls_model(image_tensor)
                heatmap = heatmap.squeeze(0).cpu()

        return image, mask, heatmap

def cutmix(image, label, alpha=1.0):
    batch_size = image.size(0)
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(batch_size)
    target_a = label
    target_b = label[rand_index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
    return image, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform

def get_seg_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform

def get_dataloaders(train_df, val_df, test_df, image_dir, batch_size, num_workers=4, use_metadata=False):
    train_transform, val_test_transform = get_transforms()

    loaders = []
    for df_split, transform, name in [
        (train_df, train_transform, "train"),
        (val_df, val_test_transform, "val"),
        (test_df, val_test_transform, "test")
    ]:
        if len(df_split) > 0:
            dataset = ChestXrayDataset(df_split, image_dir, transform=transform, use_metadata=use_metadata)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers)
        else:
            loader = None
        loaders.append(loader)

    return tuple(loaders)

def get_seg_dataloaders(train_df, val_df, test_df, image_dir, bbox_file, cls_model=None, device='cpu', batch_size=16, num_workers=4):
    train_transform, val_test_transform = get_seg_transforms()
    bbox_df = pd.read_csv(bbox_file)

    # Chuẩn hóa tên cột: bỏ các cột không cần thiết (Unnamed: 6, Unnamed: 7, Unnamed: 8)
    bbox_df = bbox_df[['Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]']]

    loaders = []
    for df_split, transform, name in [
        (train_df, train_transform, "train"),
        (val_df, val_test_transform, "val"),
        (test_df, val_test_transform, "test")
    ]:
        if len(df_split) > 0:
            dataset = ChestXraySegDataset(df_split, image_dir, bbox_df=bbox_df, cls_model=cls_model, device=device, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers)
        else:
            loader = None
        loaders.append(loader)

    return tuple(loaders)