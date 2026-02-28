
"""-----------------------------------------------------{ Main entry point for Water Segmentation project }--------------------------------------------------------------------------"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import train_model, validate, DiceBCELoss
from data_load import WaterDataset 

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import train_model








def main():
    # Configuration
    config = {
        # Paths - relative to project root
        'images_dir': '/content/drive/MyDrive/satalite data/data/images',
        'masks_dir': '/content/drive/MyDrive/satalite data/data/labels',
        
         # Training hyperparameters
        'batch_size': 16,                    
        'learning_rate': 0.0003,              
        'num_epochs': 120,                    
        'optimizer': 'AdamW',                  
        'weight_decay': 1e-4,                    
        'scheduler': 'CosineAnnealingWarmRestarts',  
        'loss': 'DiceBCE',                               
        'augmentation': 'Heavy'
        'selected_bands': [3, 4, 5, 6, 7, 9, 10, 12]
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Water Segmentation with U-Net')
    parser.add_argument('--batch_size', type=int, default=config['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=config['num_epochs'],
                        help='Number of epochs')
    parser.add_argument('--images_dir', type=str, default=config['images_dir'],
                        help='Path to images directory')
    parser.add_argument('--masks_dir', type=str, default=config['masks_dir'],
                        help='Path to masks directory')
    
    args = parser.parse_args()
    
    # Update config
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['num_epochs'] = args.epochs
    config['images_dir'] = args.images_dir
    config['masks_dir'] = args.masks_dir
    
    # Print configuration
    print("="*60)
    print("WATER SEGMENTATION PROJECT")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("="*60)
    
    # Check if data directories exist
    if not os.path.exists(config['images_dir']):
        print(f"ERROR: Images directory not found: {config['images_dir']}")
        print("Please update the path or create a symlink to your data.")
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
    print(f" Loaded best model from epoch {checkpoint['epoch']+1} with IoU: {checkpoint['val_iou']:.4f}")
    
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
    test_loss, test_iou, test_precision, test_recall, test_f1 = validate(
        model, test_loader, criterion, device
    )
    
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
        f.write("FINAL TEST RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")
    
    print("\n Test results saved to 'test_results.txt'")


if __name__ == "__main__":
    main()







