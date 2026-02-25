

"""-----------------------{ Data loading utilities: It handles loading of satellite images and masks }-----------------------------"""





# ------------< Imports >--------------
import os
import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split




class WaterDataset(Dataset):
    # Custom Dataset for water segmentation using satellite images and masks.
    
    def __init__(self, images_dir, masks_dir, split='train', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Args:
            images_dir (str): Path to directory with .tif satellite images
            masks_dir (str): Path to directory with .png labels files
            split (str): One of 'train', 'val', or 'test'
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            test_ratio (float): Proportion of data for testing
            random_seed (int): Random seed for reproducibility
        """

        
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.split = split
        
        # Get all image files
        all_image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith('.tif')])
        
        # Get all mask files
        all_mask_files = sorted([f for f in os.listdir(masks_dir) 
                                  if f.endswith('.png')])
        
        print(f"Total images found: {len(all_image_files)}")
        print(f"Total masks found: {len(all_mask_files)}")
        
        # Find which images have matching masks
        paired_images = []
        paired_masks = []
        
        for img_file in all_image_files:
            expected_mask = img_file.replace('.tif', '.png')
            if expected_mask in all_mask_files:
                paired_images.append(img_file)
                paired_masks.append(expected_mask)
        
        print(f"Paired images with masks: {len(paired_images)}")
        
        if len(paired_images) == 0:
            raise RuntimeError("No matching image-mask pairs found!")
        
        # Create train/val/test splits
        # First split: train and temp (val+test)
        train_files, temp_files = train_test_split(
            paired_images, 
            train_size=train_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val and test from temp
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            train_size=val_ratio_adjusted,
            random_state=random_seed,
            shuffle=True
        )
        
        # Store files based on split
        if split == 'train':
            self.image_files = train_files
        elif split == 'val':
            self.image_files = val_files
        elif split == 'test':
            self.image_files = test_files
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        # Get corresponding mask files
        self.mask_files = [f.replace('.tif', '.png') for f in self.image_files]
        
        print(f"Split '{split}': {len(self.image_files)} samples")
        print(f"  Train: {len(train_files)} samples")
        print(f"  Val: {len(val_files)} samples")
        print(f"  Test: {len(test_files)} samples")
        
        if len(self.image_files) == 0:
            raise RuntimeError("No matching image-mask pairs found!")
        
    #---------------------------------------------------------------------------------
    
    def __len__(self):
        # Return the total number of samples in the dataset.
        return len(self.image_files)

    #---------------------------------------------------------------------------------
    
    def __getitem__(self, idx):
        """
            Load and return a sample (image, mask) at the given index like dataset[0].
    
            Args:
                idx (int): Index of the sample to load
    
            Returns:
                tuple: (image_tensor, mask_tensor) where:
                    image_tensor: torch.Tensor of shape (12, 512, 512)
                    mask_tensor: torch.Tensor of shape (512, 512)
        """
        # Get filenames for this index
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        
        # Construct full paths
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load the 12-channel satellite image
        # Using tifffile to read multi-channel TIFF
        image = tifffile.imread(img_path)
        
        # Load the mask (binary PNG)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # Print shapes to understand what we're working with
        print(f"Image shape: {image.shape}")    
        print(f"Mask shape: {mask.shape}")
        print(f"Image dtype: {image.dtype}, Min: {image.min()}, Max: {image.max()}")
        print(f"Mask unique values: {np.unique(mask)}")
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        # Rearrange dimensions from (H, W, C) to (C, H, W) for PyTorch
        image_tensor = image_tensor.permute(2, 0, 1)  # (12, 128, 128)

        # Normalize image to [0, 1] range (simple min-max normalization)
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)
        
        return image_tensor, mask_tensor
    
    
    

# Test the dataset
if __name__ == "__main__":
    # Set paths - YOU NEED TO UPDATE THESE!
    images_dir = "/content/drive/MyDrive/satalite data/data/images"  # Adjust path as needed
    masks_dir = "/content/drive/MyDrive/satalite data/data/labels"    # Adjust path as needed
    
    # Test different splits
    print("="*50)
    print("TESTING TRAIN/VAL/TEST SPLITS")
    print("="*50)
    
    # Create train dataset
    print("\n1. Creating TRAIN dataset:")
    train_dataset = WaterDataset(images_dir, masks_dir, split='train')
    
    # Create validation dataset
    print("\n2. Creating VALIDATION dataset:")
    val_dataset = WaterDataset(images_dir, masks_dir, split='val')
    
    # Create test dataset
    print("\n3. Creating TEST dataset:")
    test_dataset = WaterDataset(images_dir, masks_dir, split='test')
    
    # Test loading one sample from each
    print("\n" + "="*50)
    print("TESTING SAMPLE LOADING")
    print("="*50)
    
    # Test train sample
    print("\nTrain sample:")
    image, mask = train_dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    
    # Test val sample
    print("\nValidation sample:")
    image, mask = val_dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    
    # Test test sample
    print("\nTest sample:")
    image, mask = test_dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    
    print("\n" + "="*50)
    print("✅ DATA SPLITTING WORKING CORRECTLY!")
    print("="*50)
