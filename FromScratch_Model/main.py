
"""-----------------------------------------------------{ Main entry point for Water Segmentation project }--------------------------------------------------------------------------"""

import os
import sys
import argparse

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
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs': 50,
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


if __name__ == "__main__":
    main()
