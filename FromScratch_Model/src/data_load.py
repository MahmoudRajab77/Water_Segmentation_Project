

"""-----------------------{ Data loading utilities: It handles loading of satellite images and masks }-----------------------------"""





# ------------< Imports >--------------
import os
import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset





class WaterDataset(Dataset):
    # Custom Dataset for water segmentation using satellite images and masks.
    
    def __init__(self, images_dir, masks_dir):
        """
        Args:
            images_dir (str): Path to directory with .tif satellite images
            masks_dir (str): Path to directory with .png labels files
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # Get all image files
        all_image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith('.tif')])
        
        # Get all mask files
        all_mask_files = sorted([f for f in os.listdir(masks_dir) 
                                  if f.endswith('.png')])
        
        print(f"Total images found: {len(all_image_files)}")
        print(f"Total masks found: {len(all_mask_files)}")
        
        # Find which images have matching masks
        self.image_files = []
        self.mask_files = []
        
        for img_file in all_image_files:
            # Convert .tif to .png to get expected mask name
            expected_mask = img_file.replace('.tif', '.png')
            
            if expected_mask in all_mask_files:
                self.image_files.append(img_file)
                self.mask_files.append(expected_mask)
        
        print(f"Paired images with masks: {len(self.image_files)}")
        
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
        # PyTorch expects (C, H, W) format
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor
    
    
    

# Test the dataset
if __name__ == "__main__":
    # Set paths - YOU NEED TO UPDATE THESE!
    images_dir = "../data/images"  # Adjust path as needed
    masks_dir = "../data/masks"    # Adjust path as needed
    
    # Create dataset
    print("Creating dataset...")
    dataset = WaterDataset(images_dir, masks_dir)
    
    # Test __len__
    print(f"\nDataset size: {len(dataset)}")
    
    # Test __getitem__ with first sample
    print("\nLoading first sample...")
    image, mask = dataset[0]
    
    print(f"\nFinal tensor shapes:")
    print(f"Image tensor shape: {image.shape}")
    print(f"Mask tensor shape: {mask.shape}")
    print(f"Image tensor dtype: {image.dtype}")
    print(f"Mask tensor dtype: {mask.dtype}")
    
    # Check value ranges
    print(f"\nImage tensor - Min: {image.min():.2f}, Max: {image.max():.2f}, Mean: {image.mean():.2f}")
    print(f"Mask tensor - Unique values: {torch.unique(mask)}")