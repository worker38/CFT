import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import time
from tqdm import tqdm
from .utils import calculate_metrics, FocalLoss

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, model_dir, accum_steps=4):
    best_auc = 0
    trigger_times = 0
    os.makedirs(model_dir, exist_ok=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    total_train_time = 0.0
    total_val_time = 0.0
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Huấn luyện
        model.train()
        running_loss = 0.0
        train_start_time = time.time()

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        optimizer.zero_grad()
        for i, batch in enumerate(train_progress):
            if len(batch) == 3:
                images, labels, metadata = batch
                images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
                outputs, heatmap = model(images, metadata)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs, heatmap = model(images)

            loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps * images.size(0)
            train_progress.set_postfix(loss=loss.item())

        train_time = time.time() - train_start_time
        total_train_time += train_time
        epoch_loss = running_loss / len(train_loader.dataset)

        # Đánh giá trên tập validation
        model.eval()
        val_labels = []
        val_scores = []
        val_start_time = time.time()

        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch in val_progress:
                if len(batch) == 3:
                    images, labels, metadata = batch
                    images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
                    outputs, heatmap = model(images, metadata)
                else:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs, heatmap = model(images)
                outputs = torch.sigmoid(outputs)
                val_labels.append(labels.cpu().numpy())
                val_scores.append(outputs.cpu().numpy())

        val_time = time.time() - val_start_time
        total_val_time += val_time

        val_labels = np.vstack(val_labels)
        val_scores = np.vstack(val_scores)
        auc_scores, acc_scores, f1_scores = calculate_metrics(val_labels, val_scores, val_scores)
        mean_auc = np.mean(auc_scores)
        mean_acc = np.mean(acc_scores)
        mean_f1 = np.mean(f1_scores)

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        log_msg = f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, ' \
                  f'Val Mean AUC: {mean_auc:.4f}, Val Mean ACC: {mean_acc:.4f}, Val Mean F1: {mean_f1:.4f}\n' \
                  f'Val AUC per class: {", ".join([f"{auc:.4f}" for auc in auc_scores])}\n' \
                  f'Val ACC per class: {", ".join([f"{acc:.4f}" for acc in acc_scores])}\n' \
                  f'Val F1 per class: {", ".join([f"{f1:.4f}" for f1 in f1_scores])}\n' \
                  f'Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s, Total Epoch Time: {epoch_time:.2f}s'
        logging.info(log_msg)
        print(log_msg)

        if mean_auc > best_auc:
            best_auc = mean_auc
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logging.info('Early stopping!')
                print('Early stopping!')
                break

    total_time = sum(epoch_times)
    avg_epoch_time = total_time / len(epoch_times) if epoch_times else 0
    final_log = f'\nTraining Summary:\n' \
                f'Total Training Time: {total_train_time:.2f}s\n' \
                f'Total Validation Time: {total_val_time:.2f}s\n' \
                f'Total Time: {total_time:.2f}s\n' \
                f'Average Epoch Time: {avg_epoch_time:.2f}s'
    logging.info(final_log)
    print(final_log)

    return model