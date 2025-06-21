import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import time
from tqdm import tqdm
from src.utils import dice_loss

def train_seg_model(model, train_loader, val_loader, device, num_epochs, patience, model_dir, accum_steps=4):
    best_dice = 0
    trigger_times = 0
    os.makedirs(model_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    total_train_time = 0.0
    total_val_time = 0.0
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Huấn luyện
        model.train()
        running_seg_loss = 0.0
        train_start_time = time.time()

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        optimizer.zero_grad()
        for i, batch in enumerate(train_progress):
            images, masks, heatmaps = batch
            images, masks, heatmaps = images.to(device), masks.to(device), heatmaps.to(device)
            seg_mask = model(images, heatmaps)

            bce_loss = nn.functional.binary_cross_entropy(seg_mask, masks)
            dice = dice_loss(seg_mask, masks)
            loss = 0.5 * bce_loss + 0.5 * dice

            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_seg_loss += loss.item() * accum_steps * images.size(0)
            train_progress.set_postfix(seg_loss=loss.item())

        train_time = time.time() - train_start_time
        total_train_time += train_time
        epoch_seg_loss = running_seg_loss / len(train_loader.dataset)

        # Đánh giá trên tập validation
        model.eval()
        val_seg_losses = []
        val_dice_scores = []
        val_start_time = time.time()

        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch in val_progress:
                images, masks, heatmaps = batch
                images, masks, heatmaps = images.to(device), masks.to(device), heatmaps.to(device)
                seg_mask = model(images, heatmaps)

                bce_loss = nn.functional.binary_cross_entropy(seg_mask, masks)
                dice = dice_loss(seg_mask, masks)
                total_seg_loss = 0.5 * bce_loss + 0.5 * dice
                val_seg_losses.append(total_seg_loss.item())

                pred = (seg_mask > 0.5).float()
                intersection = (pred * masks).sum(dim=(2, 3))
                union = pred.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (2. * intersection + 1e-5) / (union + 1e-5)
                val_dice_scores.append(dice.mean(dim=0).cpu().numpy())

        val_time = time.time() - val_start_time
        total_val_time += val_time

        mean_seg_loss = np.mean(val_seg_losses)
        mean_dice_scores = np.mean(val_dice_scores, axis=0)
        mean_dice = np.mean(mean_dice_scores)

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        log_msg = f'Epoch {epoch+1}/{num_epochs}, Seg Loss: {epoch_seg_loss:.4f}, ' \
                  f'Val Seg Loss: {mean_seg_loss:.4f}, Val Mean Dice: {mean_dice:.4f}\n' \
                  f'Val Dice per class: {", ".join([f"{dice:.4f}" for dice in mean_dice_scores])}\n' \
                  f'Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s, Total Epoch Time: {epoch_time:.2f}s'
        logging.info(log_msg)
        print(log_msg)

        if mean_dice > best_dice:
            best_dice = mean_dice
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_seg_model.pth'))
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