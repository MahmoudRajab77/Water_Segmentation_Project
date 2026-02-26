"""
-----------------------------------------------------------{ Training script for water segmentation U-Net }---------------------------------------------------------------------------------
"""




# ----------------< Imports >-------------
import os
import sys
import argparse                          # For command-line arguments (so you can change settings without editing code)
import numpy as np
import torch
import torch.nn as nn                    # for neural network layers
import torch.optim as optim              # for optimizer
from torch.utils.data import DataLoader      # for patching our dataset 

from data_load import WaterDataset
from model import UNet

from sklearn.metrics import precision_score, recall_score, f1_score      # For calculating precision, recall, F1-score
from tqdm import tqdm              # For nice progress bars during training
import matplotlib.pyplot as plt    # For plotting training curves later









"""
    Combined Dice Loss and Binary Cross Entropy Loss
    Best for imbalanced segmentation (water pixels usually less than non-water)
"""
class DiceBCELoss(nn.Module):
    
    def __init__(self, weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.weight = weight  # وزن BCE (0.5 = نص ونص)
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
        inputs_sigmoid = torch.sigmoid(inputs)  # حول logits لـ probabilities
        
        # Flatten للتسهيل
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice_coef = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coef
        
        # 3. Combined Loss
        combined_loss = self.weight * bce_loss + (1 - self.weight) * dice_loss
        
        return combined_loss

#----------------------------------------------------------------------------------------------------------------------------------------------


"""
    Purpose: IoU measures overlap between predicted and actual water pixels

    How it works:

        Convert probabilities to binary using threshold (0.5)
        Flatten to 1D arrays for easy comparison
        Calculate intersection (pixels where both are 1)
        Calculate union (pixels where either is 1)
        Return intersection/union

    Range: 0 (no overlap) to 1 (perfect overlap)

"""

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
    
    # Handle case where union is 0 (no water pixels in target and no predictions)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


#------------------------------------------------------------------------------------------------

"""
    Train the model for one epoch.
        
    Args:
        model: U-Net model
        dataloader: Training data loader
        criterion: Loss function (BCEWithLogitsLoss)
        optimizer: Optimizer (Adam)
        device: Device to train on (cuda/cpu)
        
    Returns:
        float: Average loss for the epoch
        float: Average IoU for the epoch
"""

"""
    model.train()	: Sets model to training mode (enables dropout, batch norm updates)
    tqdm(dataloader)	: Creates progress bar showing training progress
    images.to(device)	: Moves data to GPU if available
    masks.unsqueeze(1)	: Adds channel dimension: (B, H, W) → (B, 1, H, W)
    optimizer.zero_grad()	: Clears gradients from previous step
    outputs = model(images)	: Forward pass through U-Net
    loss.backward()	: Computes gradients
    optimizer.step()	: Updates model weights
    torch.sigmoid(outputs)	: Converts logits to probabilities (0-1)
    calculate_iou()	: Measures overlap between predictions and ground truth

"""

def train_one_epoch(model, dataloader, criterion, optimizer, device):
   
    model.train()  # Set model to training mode
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc='Training', leave=False, ncols=80)
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Add channel dimension to masks if needed (from (B, H, W) to (B, 1, H, W))
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
        
        # Convert masks to float for loss calculation
        masks = masks.float()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Calculate IoU for this batch
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
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou


#----------------------------------------------------------------------------------------------------------------------

"""
    Validate the model on validation set.
        
    Args:
        model: U-Net model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        float: Average validation loss
        float: Average validation IoU
        float: Precision
        float: Recall
        float: F1-score
"""

"""
    model.eval() : Sets model to evaluation mode (no dropout, uses running stats for batch norm)
    with torch.no_grad() : Disables gradient calculation (saves memory and computation)
    Storing predictions	: Collects all predictions to calculate metrics at the end
    precision_score	: Of all pixels predicted as water, how many are actually water?
    recall_score : Of all actual water pixels, how many did we find?
    f1_score : Harmonic mean of precision and recall
    zero_division=0	: Handles cases where there are no predictions or no targets
"""

def validate(model, dataloader, criterion, device):
    
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_iou = 0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    # No gradient needed for validation
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating', leave=False, ncols=80)
        
        for images, masks in progress_bar:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Add channel dimension to masks if needed
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            masks = masks.float()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate IoU
            pred_probs = torch.sigmoid(outputs)
            batch_iou = calculate_iou(pred_probs, masks)
            
            # Store predictions and targets for precision/recall/f1
            pred_binary = (pred_probs > 0.5).float()
            all_preds.append(pred_binary.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            # Update metrics
            total_loss += loss.item()
            total_iou += batch_iou
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou:.4f}'
            })
    
    # Calculate average loss and IoU
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    # Calculate precision, recall, F1-score
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return avg_loss, avg_iou, precision, recall, f1

#--------------------------------------------------------------------------------------

"""
    Main training function.
        
    Args:
        config (dict): Configuration dictionary with training parameters
"""

def train_model(config):
   
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n" + "="*50)
    print("LOADING DATASETS")
    print("="*50)
    
    train_dataset = WaterDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        split='train'
    )
    
    val_dataset = WaterDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        split='val'
    )
    
    # Create data loaders
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
    print("CREATING MODEL")
    print("="*50)
    
    model = UNet(n_channels=12, n_classes=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = DiceBCELoss(weight=0.5)  # 0.5 means BCE and Dice are the same important 
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # OneCycleLR scheduler (best for segmentation)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],  # peak learning rate
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% of training for warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1000  # final_lr = initial_lr/1000
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
        # Header of epoh
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1:2d}/{config['num_epochs']} 🔥")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_iou, precision, recall, f1 = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # Print results مرتبة
        print(f"\n RESULTS")
        print(f"{'─'*40}")
        print(f"Train   | Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"Val     | Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")
        print(f"{'─'*40}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # Save best model
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
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_ious)
    
    return model

#------------------------------------------------------------------------------------------------------------

"""
    For Plotting training curves.
"""

def plot_training_curves(train_losses, val_losses, val_ious):  
  
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot IoU
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









