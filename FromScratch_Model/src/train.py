"""
-----------------------------------------------------------{ Training script for water segmentation U-Net }---------------------------------------------------------------------------------
"""




# ----------------< Imports >-------------
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_load import WaterDataset
from model import UNet

from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt








#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
    Combined Dice Loss and Binary Cross Entropy Loss
    Best for imbalanced segmentation (water pixels usually less than non-water)
"""
class DiceBCELoss(nn.Module):
    
    def __init__(self, weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        """
        inputs: model outputs (logits) shape (B, 1, H, W)
        targets: ground truth masks shape (B, 1, H, W)
        """
        # 1. BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # 2. Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_coef = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coef
        
        # 3. Combined Loss
        combined_loss = self.weight * bce_loss + (1 - self.weight) * dice_loss
        
        return combined_loss

#----------------------------------------------------------------------------------------------------------------------------------------------

"""
    Calculate Intersection over Union for binary segmentation.
    
    Args:
        pred (torch.Tensor): Model predictions (after sigmoid), shape (B, 1, H, W)
        target (torch.Tensor): Ground truth masks, shape (B, 1, H, W)
        threshold (float): Threshold to convert probabilities to binary
    
    Returns:
        float: IoU score
"""
def calculate_iou(pred, target, threshold=0.5):
    
    # Convert predictions to binary (0 or 1)
    pred_binary = (pred > threshold).float()
    
    # Flatten the tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_flat, target_flat).sum()
    union = np.logical_or(pred_flat, target_flat).sum()
    
    # Handle case where union is 0
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


#------------------------------------------------------------------------------------------------

"""
    Train for one epoch with batch normalization.
"""
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    
    model.train()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False, ncols=80)
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)
        
        # ===== BATCH NORMALIZATION (PER BATCH) =====
        batch_mean = images.mean(dim=(0, 2, 3), keepdim=True)
        batch_std = images.std(dim=(0, 2, 3), keepdim=True) + 1e-8
        images = (images - batch_mean) / batch_std
        # ============================================
        
        # Add channel dimension to masks if needed
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
        
        masks = masks.float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Calculate IoU
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            batch_iou = calculate_iou(pred_probs, masks)
        
        # Update metrics
        total_loss += loss.item()
        total_iou += batch_iou
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_iou:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou


#----------------------------------------------------------------------------------------------------------------------

"""
    Main training function without validation.
    
    Args:
        config (dict): Configuration dictionary with training parameters
"""
def train_model(config):
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    
    train_dataset = WaterDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        split='train', 
        selected_bands=config.get('selected_bands', None)
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    
    # Create model
    print("\n" + "="*50)
    print("CREATING MODEL")
    print("="*50)
    
    n_bands = len(config.get('selected_bands', list(range(12))))
    model = UNet(n_channels=n_bands, n_classes=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
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
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate']
        )
    
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
    
    train_losses = []
    
    for epoch in range(config['num_epochs']):
        # Header of epoch
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1:2d}/{config['num_epochs']} ")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        
        # Print results
        print(f"\n RESULTS")
        print(f"{'─'*40}")
        print(f"Train   | Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"{'─'*40}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    # Save final model
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, 'final_model.pth')
    print(f" Final model saved!")
    
    return model
