"""--------------------------------------------------------Visualization utilities for multispectral satellite data----------------------------------------------------"""


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_load import WaterDataset










"""
    Visualize a single band from multispectral image.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) or (H, W, C)
        band_idx: Index of band to visualize (0-11)
        band_name: Name of the band (e.g., 'Coastal aerosol', 'Blue', etc.)
"""

def visualize_single_band(image_tensor, band_idx, band_name, save_path=None):
    
    # Handle different tensor shapes
    if len(image_tensor.shape) == 3:
        if image_tensor.shape[0] == 12:  # (C, H, W)
            band_data = image_tensor[band_idx].cpu().numpy()
        else:  # (H, W, C)
            band_data = image_tensor[:, :, band_idx].cpu().numpy()
    else:
        band_data = image_tensor.cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(band_data, cmap='gray')
    plt.colorbar(label='Reflectance Value')
    plt.title(f'Band {band_idx+1}: {band_name}')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

#----------------------------------------------------------------------------------------

"""
    Visualize all 12 bands in a grid.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) or (H, W, C)
        band_names: List of 12 band names
"""

def visualize_all_bands(image_tensor, band_names, save_dir='visualizations'):
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle tensor shape
    if len(image_tensor.shape) == 3:
        if image_tensor.shape[0] == 12:  # (C, H, W)
            bands = [image_tensor[i].cpu().numpy() for i in range(12)]
        else:  # (H, W, C)
            bands = [image_tensor[:, :, i].cpu().numpy() for i in range(12)]
    else:
        bands = [image_tensor.cpu().numpy()]
    
    # Create 3x4 grid for 12 bands
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(12):
        im = axes[i].imshow(bands[i], cmap='gray')
        axes[i].set_title(f'Band {i+1}: {band_names[i]}', fontsize=10)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Multispectral Data - All 12 Bands', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, 'all_bands_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Saved band visualization to {save_path}")

#--------------------------------------------------------------------------------------------------------
"""
    Create RGB composite from selected bands.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) or (H, W, C)
        rgb_indices: Tuple of (red_idx, green_idx, blue_idx) 0-based
        band_names: List of band names
"""

def create_rgb_composite(image_tensor, rgb_indices, band_names, title="RGB Composite", save_path=None):
    
    # Extract bands
    if len(image_tensor.shape) == 3:
        if image_tensor.shape[0] == 12:  # (C, H, W)
            r = image_tensor[rgb_indices[0]].cpu().numpy()
            g = image_tensor[rgb_indices[1]].cpu().numpy()
            b = image_tensor[rgb_indices[2]].cpu().numpy()
        else:  # (H, W, C)
            r = image_tensor[:, :, rgb_indices[0]].cpu().numpy()
            g = image_tensor[:, :, rgb_indices[1]].cpu().numpy()
            b = image_tensor[:, :, rgb_indices[2]].cpu().numpy()
    
    # Normalize for display
    def normalize(band):
        band = (band - band.min()) / (band.max() - band.min() + 1e-8)
        return band
    
    r_norm = normalize(r)
    g_norm = normalize(g)
    b_norm = normalize(b)
    
    # Stack to RGB
    rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(f'{title}\nR: {band_names[rgb_indices[0]]}, G: {band_names[rgb_indices[1]]}, B: {band_names[rgb_indices[2]]}')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------

"""
    Calculate and visualize water indices (NDWI, MNDWI).
"""

def visualize_water_indices(image_tensor, band_names, save_dir='visualizations'):
  
    # Band indices (based on your ordering)
    band_map = {
        'green': 2,      # Green band (index 2)
        'red': 3,        # Red band (index 3)
        'nir': 4,        # NIR band (index 4)
        'swir1': 5,      # SWIR1 band (index 5)
    }
    
    # Extract bands
    if image_tensor.shape[0] == 12:  # (C, H, W)
        green = image_tensor[band_map['green']].cpu().numpy()
        nir = image_tensor[band_map['nir']].cpu().numpy()
        swir1 = image_tensor[band_map['swir1']].cpu().numpy()
    else:
        green = image_tensor[:, :, band_map['green']].cpu().numpy()
        nir = image_tensor[:, :, band_map['nir']].cpu().numpy()
        swir1 = image_tensor[:, :, band_map['swir1']].cpu().numpy()
    
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir + 1e-8)
    
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    mndwi = (green - swir1) / (green + swir1 + 1e-8)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(ndwi, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('NDWI (Green - NIR)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(mndwi, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('MNDWI (Green - SWIR1)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Original RGB for reference
    rgb_indices = (3, 2, 1)  # Red, Green, Blue
    if image_tensor.shape[0] == 12:
        r = image_tensor[rgb_indices[0]].cpu().numpy()
        g = image_tensor[rgb_indices[1]].cpu().numpy()
        b = image_tensor[rgb_indices[2]].cpu().numpy()
    else:
        r = image_tensor[:, :, rgb_indices[0]].cpu().numpy()
        g = image_tensor[:, :, rgb_indices[1]].cpu().numpy()
        b = image_tensor[:, :, rgb_indices[2]].cpu().numpy()
    
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    rgb = np.stack([norm(r), norm(g), norm(b)], axis=2)
    
    axes[2].imshow(rgb)
    axes[2].set_title('RGB Reference')
    axes[2].axis('off')
    
    plt.suptitle('Water Indices Comparison')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'water_indices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

#-----------------------------------------------------------------------------------------------------

"""Main visualization function"""

def main():
  
    print("="*60)
    print("VISUALIZING MULTISPECTRAL DATA")
    print("="*60)
    
    # Band names (from your PDF)
    band_names = [
        "Coastal aerosol", "Blue", "Green", "Red", "NIR",
        "SWIR1", "SWIR2", "QA Band", "Merit DEM",
        "Copernicus DEM", "ESA world cover", "Water occurrence"
    ]
    
    # Load a sample image
    dataset = WaterDataset(
        images_dir='/content/drive/MyDrive/satalite data/data/images',
        masks_dir='/content/drive/MyDrive/satalite data/data/labels',
        split='train'
    )
    
    print(f"\n Loading sample image...")
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Visualize all 12 bands
    print("\n Visualizing all 12 bands...")
    visualize_all_bands(image, band_names)
    
    # 2. Visualize each band individually
    print("\n Visualizing individual bands...")
    for i in range(12):
        save_path = f'visualizations/band_{i+1:02d}_{band_names[i].replace(" ", "_")}.png'
        visualize_single_band(image, i, band_names[i], save_path)
    
    # 3. Create RGB composite (Natural color: Red, Green, Blue)
    print("\n Creating RGB composite (Natural color)...")
    create_rgb_composite(
        image, 
        rgb_indices=(3, 2, 1),  # Red, Green, Blue
        band_names=band_names,
        title="Natural Color RGB",
        save_path='visualizations/rgb_natural.png'
    )
    
    # 4. Create False color composite (NIR, Red, Green)
    print("\n Creating False color composite (NIR, Red, Green)...")
    create_rgb_composite(
        image,
        rgb_indices=(4, 3, 2),  # NIR, Red, Green
        band_names=band_names,
        title="False Color (NIR, Red, Green)",
        save_path='visualizations/rgb_false_color.png'
    )
    
    # 5. Visualize water indices
    print("\n Calculating and visualizing water indices...")
    visualize_water_indices(image, band_names)
    
    print("\n" + "="*60)
    print(" VISUALIZATION COMPLETE!")
    print(f" Visualizations saved in 'visualizations/' folder")
    print("="*60)


if __name__ == "__main__":
    main()
