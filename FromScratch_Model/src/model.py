"""--------------------------------------{ U-Net architecture for water segmentation }----------------------------------------------------
------------------------------------Implemented from scratch for 12-channel satellite images----------------------------------------------
"""



# -------------< Imports >--------------
import torch
import torch.nn as nn
import torch.nn.functional as F




#--------------------------------------< Code >----------------------------------------------

""" The basic building block of U-Net: two convolutional layers with batch norm and ReLU.
        in_channels: number of input channels (12 for first block)
        out_channels: number of output channels (64, 128, 256, etc.)
        padding=1 : ensures output size = input size (3x3 conv with padding=1 preserves dimensions)
        BatchNorm2d : normalizes activations for stable training
        ReLU : the activation function used 
"""
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------------------------------------------------------------------------------------

"""     
    MaxPool2d(2): Reduces image size by half (128×128 → 64×64)
    DoubleConv : is applied to extract features at this smaller scale
    This is the encoder path - capturing context and reducing spatial info

"""
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Halves the spatial dimensions (128→64, 64→32, etc.)
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

#----------------------------------------------------------------------------------------------------------------------------

"""
    Upsampling: Doubles the spatial dimensions (64×64 → 128×128)
    Skip connection: Concatenates with encoder features from the same level
    DoubleConv: Processes the concatenated features
"""
class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        # Upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Double convolution after concatenation with skip connection
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1: feature map from decoder (coming from below)
        # x2: feature map from encoder (skip connection)
        
        # Upsample x1
        x1 = self.up(x1)
        
        # Handle potential size mismatch (due to odd dimensions)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Pad x1 if needed to match x2's dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channels axis (skip connection)
        x = torch.cat([x2, x1], dim=1)
        
        # Apply double convolution
        return self.conv(x)

#---------------------------------------------------------------------------------------------------------------------

class OutConv(nn.Module):
    """Final 1x1 convolution to get desired number of output channels"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

#---------------------------------------------------------------------------------------------------------------------

class UNet(nn.Module):
    """
      U-Net architecture for water segmentation.
      Input: 12-channel satellite image (12, H, W)
      Output: 1-channel binary mask (1, H, W)
    """
    
    def __init__(self, n_channels=12, n_classes=1):
        """
        Args:
            n_channels (int): Number of input channels (12 for your data)
            n_classes (int): Number of output classes (1 for binary segmentation)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)      # 128x128 -> 128x128
        self.down1 = Down(64, 128)                  # 128x128 -> 64x64
        self.down2 = Down(128, 256)                 # 64x64 -> 32x32
        self.down3 = Down(256, 512)                 # 32x32 -> 16x16
        self.down4 = Down(512, 1024)                 # 16x16 -> 8x8
        
        # Decoder (Expanding Path)
        self.up1 = Up(1024, 512)                     # 8x8 -> 16x16
        self.up2 = Up(512, 256)                      # 16x16 -> 32x32
        self.up3 = Up(256, 128)                      # 32x32 -> 64x64
        self.up4 = Up(128, 64)                       # 64x64 -> 128x128
        
        # Output layer
        self.outc = OutConv(64, n_classes)           # 128x128 -> 128x128
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 12, 128, 128)
        
        Returns:
            Output tensor of shape (batch_size, 1, 128, 128)
        """
        # Encoder path (with skip connections stored)
        x1 = self.inc(x)      # Skip connection 1
        x2 = self.down1(x1)    # Skip connection 2
        x3 = self.down2(x2)    # Skip connection 3
        x4 = self.down3(x3)    # Skip connection 4
        x5 = self.down4(x4)    # Bottleneck
        
        # Decoder path (using skip connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer
        logits = self.outc(x)
        
        # For binary segmentation, we can apply sigmoid later in loss function
        return logits

#------------------------------------------------------------------------------------------------

# Test the model
"""
if __name__ == "__main__":
    # Create model instance
    model = UNet(n_channels=12, n_classes=1)
    
    # Print model architecture
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with random input (batch_size=2, channels=12, height=128, width=128)
    x = torch.randn(2, 12, 128, 128)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

"""

