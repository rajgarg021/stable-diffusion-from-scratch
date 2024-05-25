import torch 
import torch.nn as nn
import torch.nn.functional as F

class VAE_ResidualBlock(nn.Module):
    """
    Residual block for the encoder and decoder
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.tensor):
        """ x: (B, in_channels, H, W) """

        residue = x

        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        x = self.gn1(x)
        
        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        x = F.silu(x)
        
        # (B, in_channels, H, W) -> (B, out_channels, H, W)
        x = self.conv1(x)
        
        # (B, out_channels, H, W) -> (B, out_channels, H, W)
        x = self.gn2(x)
        
        # (B, out_channels, H, W) -> (B, out_channels, H, W)
        x = F.silu(x)
        
        # (B, out_channels, H, W) -> (B, out_channels, H, W)
        x = self.conv2(x)
        
        # (B, out_channels, H, W) -> (B, out_channels, H, W)
        return x + self.residual_layer(residue)

