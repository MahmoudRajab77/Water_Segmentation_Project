"""
Training script for pretrained U-Net (ResNet34 encoder).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_load import WaterDataset
from model import PretrainedUNet


class DiceBCELoss(nn.Module):
    """Combined Dice Loss and Binary Cross Entropy Loss"""
    
    def __init__(self, weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice loss
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_coef = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coef
        
        # Combined loss
        combined_loss = self.weight * bce_loss + (1 - self.weight) * dice_loss
        
        return combined_loss


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union for binary segmentation"""
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    intersection = np.logical_and(pred_flat, target_flat).sum()
    union = np.logical_or(pred_flat, target_flat).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False, ncols=80)
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            batch_iou = calculate_iou(pred_probs, masks)
        
        total_loss += loss.item()
        total_iou += batch_iou
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_iou:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_iou = 0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating', leave=False, ncols=80)
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred_probs = torch.sigmoid(outputs)
            batch_iou = calculate_iou(pred_probs, masks)
            
            pred_binary = (pred_probs > 0.5).float()
            all_preds.append(pred_binary.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            total_loss += loss.item()
            total_iou += batch_iou
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return avg_loss, avg_iou, precision, recall, f1


def plot_training_curves(train_losses, val_losses, val_ious):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ious, 'g-', label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def train_model(config):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n" + "="*50)
    print("LOADING DATASETS")
    print("="*50)
    
    n_bands = len(config.get('selected_bands', list(range(12))))
    
    train_dataset = WaterDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        split='train',
        selected_bands=config.get('selected_bands', None)
    )
    
    val_dataset = WaterDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        split='val',
        selected_bands=config.get('selected_bands', None)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "="*50)
    print("CREATING PRETRAINED MODEL (ResNet34)")
    print("="*50)
    
    model = PretrainedUNet(n_channels=n_bands, n_classes=1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    if config.get('loss', 'BCE') == 'DiceBCE':
        criterion = DiceBCELoss(weight=0.5)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    if config.get('optimizer', 'Adam') == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    if config.get('scheduler', 'OneCycle') == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000
        )
    
    # Training loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    val_ious = []
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1:2d}/{config['num_epochs']} ")
        print(f"{'='*60}")
        
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_iou, precision, recall, f1 = validate(
            model, val_loader, criterion, device
        )
        
        if config.get('scheduler', 'OneCycle') == 'CosineAnnealingWarmRestarts':
            scheduler.step()
        else:
            scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        print(f"\n RESULTS")
        print(f"{'─'*40}")
        print(f"Train   | Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"Val     | Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")
        print(f"{'─'*40}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': config
            }, 'best_model.pth')
            print(f" Best model saved! (IoU: {val_iou:.4f})")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Best validation IoU: {best_val_iou:.4f}")
    
    plot_training_curves(train_losses, val_losses, val_ious)
    
    return model
