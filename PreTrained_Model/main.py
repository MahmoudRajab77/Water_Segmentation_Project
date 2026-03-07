
"""------------------------------------------------{Main entry point for Pretrained U-Net (ResNet34) water segmentation}----------------------------------------------------------------"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import train_model, DiceBCELoss, calculate_iou
from data_load import WaterDataset


def main():
    # Configuration
    config = {
        # Paths - update these for your system
        'images_dir': '/content/drive/MyDrive/satalite data/data/images',
        'masks_dir': '/content/drive/MyDrive/satalite data/data/labels',
        
        # Training hyperparameters
        'batch_size': 16,
        'learning_rate': 0.0001,  # Lower learning rate for pretrained
        'num_epochs': 100,
        'optimizer': 'AdamW',
        'weight_decay': 1e-4,
        'scheduler': 'CosineAnnealingWarmRestarts',
        'loss': 'DiceBCE',
        'augmentation': 'Heavy',
        'selected_bands': [1, 2, 3, 4, 5, 6, 8, 9, 11]  # 8 bands (Changed to 9 bands to add blue as the ImageNet normalization is applied to the first 3 channels which must be RGB)
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pretrained U-Net for Water Segmentation')
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['learning_rate'])
    parser.add_argument('--epochs', type=int, default=config['num_epochs'])
    parser.add_argument('--images_dir', type=str, default=config['images_dir'])
    parser.add_argument('--masks_dir', type=str, default=config['masks_dir'])
    
    args = parser.parse_args()
    
    # Update config
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['num_epochs'] = args.epochs
    config['images_dir'] = args.images_dir
    config['masks_dir'] = args.masks_dir
    
    # Print configuration
    print("="*60)
    print("PRETRAINED U-NET (ResNet34) WATER SEGMENTATION")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("="*60)
    
    # Check if data directories exist
    if not os.path.exists(config['images_dir']):
        print(f"ERROR: Images directory not found: {config['images_dir']}")
        sys.exit(1)
    
    if not os.path.exists(config['masks_dir']):
        print(f"ERROR: Masks directory not found: {config['masks_dir']}")
        sys.exit(1)
    
    # Train the model
    try:
        model = train_model(config)
        print("\n Training completed successfully!")
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== TEST ON TEST SET ==========
    print("\n" + "="*60)
    print(" TESTING ON TEST SET")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load('best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" Loaded best model from epoch {checkpoint['epoch']} with IoU: {checkpoint['train_iou']:.4f}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test dataset
    test_dataset = WaterDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        split='test',
        selected_bands=config['selected_bands']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Loss function
    criterion = DiceBCELoss(weight=0.5)
    
    # Test the model
    print("\n Evaluating on test set...")
    
    model.eval()
    test_loss = 0
    test_iou = 0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
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
            
            test_loss += loss.item()
            test_iou += batch_iou
            num_batches += 1
    
    # Calculate averages
    test_loss /= num_batches
    test_iou /= num_batches
    
    # Calculate precision, recall, f1
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    test_precision = precision_score(all_targets, all_preds, zero_division=0)
    test_recall = recall_score(all_targets, all_preds, zero_division=0)
    test_f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Print final results
    print("\n" + "="*60)
    print(" FINAL TEST RESULTS")
    print("="*60)
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test IoU:        {test_iou:.4f}")
    print(f"Test Precision:  {test_precision:.4f}")
    print(f"Test Recall:     {test_recall:.4f}")
    print(f"Test F1-Score:   {test_f1:.4f}")
    print("="*60)
    
    # Save results to file
    with open('test_results.txt', 'w') as f:
        f.write("PRETRAINED MODEL - FINAL TEST RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")
    
    print("\n Test results saved to 'test_results.txt'")

#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()








