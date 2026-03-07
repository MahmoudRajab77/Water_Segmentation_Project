"""---------------------------------------------------U-Net with pretrained EfficientNet-b4 encoder for water segmentation-------------------------------------------------------------------------
-----------------------------------------------------------------Using segmentation_models_pytorch library------------------------------------------------------------------------------
"""



#-----------<Imports>---------------
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp









#------------------------------------------------------------------------------------------------------------------------------------
class PretrainedUNet(nn.Module):
    """
    U-Net with EfficientNet-b4 encoder pretrained on ImageNet.
    First convolution layer adapted to handle arbitrary number of input channels.
    """
    
    def __init__(self, n_channels=9, n_classes=1):                # Adjusted n_channels to be 9 instead of 8
        super().__init__()
        self.model = smp.Unet(
            encoder_name="efficientnet-b4",        # changed the ResNet34 encoder 
            encoder_weights="imagenet",
            in_channels=n_channels,
            classes=n_classes
        )
    #-------------------------------------------------------------------------------------------
    """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, n_channels, H, W)
        Returns:
            Output tensor of shape (batch, n_classes, H, W)
    """
    def forward(self, x):
        return self.model(x)

#------------------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":
    # Quick test
    model = PretrainedUNet(n_channels=8, n_classes=1)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    x = torch.randn(2, 8, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
"""





