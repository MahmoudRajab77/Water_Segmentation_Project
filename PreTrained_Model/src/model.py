"""
U-Net with pretrained ResNet34 encoder for water segmentation.
Adapted from 3-channel (RGB) to multi-spectral input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        # up takes the exact number of channels from below
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from below - has in_channels//2 channels
        # x2: skip connection - has in_channels//2 channels
        
        # Upsample
        x1 = self.up(x1)
        
        # Handle padding if sizes don't match
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class PretrainedUNet(nn.Module):
    """
    U-Net with ResNet34 encoder pretrained on ImageNet.
    First convolution layer adapted to handle arbitrary number of input channels.
    """
    
    def __init__(self, n_channels=12, n_classes=1):
        super(PretrainedUNet, self).__init__()
        
        # Load pretrained ResNet34
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        resnet = models.resnet34(weights=weights)
        
        # Extract encoder layers with adapted first conv
        self.inc = self._adapt_first_conv(resnet.conv1, n_channels)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Encoder blocks
        self.enc1 = resnet.layer1  # 64 channels
        self.enc2 = resnet.layer2  # 128 channels
        self.enc3 = resnet.layer3  # 256 channels
        self.enc4 = resnet.layer4  # 512 channels
        
        # Decoder with correct channel matching
        self.up1 = Up(512 + 256, 256)  # x5(512) + x4(256)
        self.up2 = Up(256 + 128, 128)  # up1 output(256) + x3(128)
        self.up3 = Up(128 + 64, 64)    # up2 output(128) + x2(64)
        self.up4 = Up(64 + 64, 64)     # up3 output(64) + x1(64)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
        
        # Store normalization parameters (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _adapt_first_conv(self, original_conv, new_in_channels):
        """
        Adapt first convolution layer from 3 channels to new_in_channels.
        Uses averaging strategy: average RGB weights and repeat.
        """
        # Create new conv layer
        new_conv = nn.Conv2d(
            new_in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Average over RGB channels (dim=1)
        with torch.no_grad():
            rgb_weight = original_conv.weight.mean(dim=1, keepdim=True)  # (out_channels, 1, k, k)
            new_weight = rgb_weight.repeat(1, new_in_channels, 1, 1) / 3  # (out_channels, new_in_channels, k, k)
            new_conv.weight.copy_(new_weight)
            
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)
        
        return new_conv

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, n_channels, H, W)
        Returns:
            Output tensor of shape (batch, n_classes, H, W)
        """
        # Split bands: first 3 for normalization, rest are kept as is
        x_rgb = x[:, :3, :, :]  # First 3 bands (for normalization)
        x_rest = x[:, 3:, :, :]  # Remaining bands
        
        # Normalize RGB part with ImageNet stats
        x_rgb = (x_rgb - self.mean) / self.std
        
        # Concatenate back
        x = torch.cat([x_rgb, x_rest], dim=1)
        
        # Encoder
        x = self.inc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # Skip connection 1 (after first conv + bn + relu) - 64 channels
        
        x = self.maxpool(x)
        x2 = self.enc1(x)  # Skip connection 2 - 64 channels
        
        x3 = self.enc2(x2)  # Skip connection 3 - 128 channels
        x4 = self.enc3(x3)  # Skip connection 4 - 256 channels
        x5 = self.enc4(x4)  # Bottleneck - 512 channels
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512 + 256 = 768 -> 256
        x = self.up2(x, x3)   # 256 + 128 = 384 -> 128
        x = self.up3(x, x2)   # 128 + 64 = 192 -> 64
        x = self.up4(x, x1)   # 64 + 64 = 128 -> 64
        
        # Output
        logits = self.outc(x)
        
        return logits


# Quick test
if __name__ == "__main__":
    model = PretrainedUNet(n_channels=8, n_classes=1)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    x = torch.randn(2, 8, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
