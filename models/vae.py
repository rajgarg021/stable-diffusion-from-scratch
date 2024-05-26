import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


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


class VAE_AttentionBlock(nn.Module):
    """
    Attention block for the encoder and decoder
    """
     
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.tensor):
        """ x: (B, C, H, W) """

        residue = x 

        # (B, C, H, W) -> (B, C, H, W)
        x = self.gn(x)

        B, C, H, W = x.shape
        
        # (B, C, H, W) -> (B, C, H * W)
        x = x.view((B, C, H * W))
        
        # (B, C, H * W) -> (B, H * W, C). Each pixel becomes a feature of size "C", the sequence length is "H * W".
        x = x.transpose(-1, -2)
        
        # performing self-attention WITHOUT mask
        # (B, H * W, C) -> (B, H * W, C)
        x = self.attention(x)
        
        # (B, H * W, C) -> (B, C, H * W)
        x = x.transpose(-1, -2)
        
        # (B, C, H * W) -> (B, C, H, W)
        x = x.view((B, C, H, W))
        
        # (B, C, H, W) + (B, C, H, W) -> (B, C, H, W) 
        x += residue

        return x


class VAE_Encoder(nn.Sequential):
    """
    Encoder of the VAE
    The idea is to decrease the size of the image while increasing the number of 
    features/channels so that each pixel represents more information
    """

    def __init__(self):
        super.__init__(
            # (B, C, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # decreasing image size (B, 128, H, W) -> (B, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # increasing number of features (B, 128, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),

            # (B, 256, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),

            # decreasing image size (B, 256, H/2, W/2) -> (B, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            # increasing number of features (B, 256, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),

            # (B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),

            # decreasing image size (B, 512, H/4, W/4) -> (B, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_AttentionBlock(512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            nn.GroupNorm(num_groups=32, num_channels=512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            nn.SiLU(),

            # because the padding=1, it means the W and H will increase by 2
            # out_H = in_H + padding_top + padding_bottom
            # out_W = in_W + padding_left + padding_right
            # since padding=1 means padding_top = padding_bottom = padding_left = padding_right = 1
            # since the out_W = in_W + 2 (same for out_H), it will compensate for the kernel_size of 3
            # (B, 512, H/8, W/8) -> (B, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (B, 8, H/8, W/8) -> (B, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.tensor, noise: torch.tensor):
        """
        x: (B, C, H, W)
        noise: (B, 4, H/8, W/8)
        """

        for module in self:
            if getattr(module, "stride", None) == (2, 2): # padding at downsampling should be asymmetric
                # pad: (padding_left, padding_right, padding_top, padding_bottom)
                # pad with zeros on the right and bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (B, 8, H/8, W/8) -> two tensors of shape (B, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # clamping the log variance between -30 and 20, so that the variance is between apprx. 1e-14 and 1e8
        # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
        variance = log_variance.exp()

        # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
        stdev = variance.sqrt()

        # transforming N(0, 1) -> N(mean, stdev) 
        # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
        x = mean + stdev * noise
        
        # scaling by a constant
        # constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x


class VAE_Decoder(nn.Sequential):
    """
    Decoder of the VAE
    """

    def __init__(self):
        super().__init__(
            # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (B, 4, H/8, W/8) -> (B, 512, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), 
            
            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_AttentionBlock(512), 
            
            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), 
            
            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), 
            
            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), 
            
            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), 
            
            # repeating the rows and columns of the data by scale_factor (like when you resize an image by doubling its size)
            # (B, 512, H/8, W/8) -> (B, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),
            
            # (B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512), 
            
            # (B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512), 
            
            # (B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512), 
            
            # (B, 512, H/4, W/4) -> (B, 512, H/2, W/2)
            nn.Upsample(scale_factor=2), 
            
            # (B, 512, H/2, W/2) -> (B, 512, H/2, W/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (B, 512, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(512, 256), 
            
            # (B, 256, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256), 
            
            # (B, 256, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256), 
            
            # (B, 256, H/2, W/2) -> (B, 256, H, W)
            nn.Upsample(scale_factor=2), 
            
            # (B, 256, H, W) -> (B, 256, H, W)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # (B, 256, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(256, 128), 
            
            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128), 
            
            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128), 
            
            # (B, 128, H, W) -> (B, 128, H, W)
            nn.GroupNorm(32, 128), 
            
            # (B, 128, H, W) -> (B, 128, H, W)
            nn.SiLU(), 
            
            # (B, 128, H, W) -> (B, 3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        """ x: (B, 4, H/8, W/8) """
        
        # removing the scaling added by the Encoder
        x /= 0.18215

        for module in self:
            x = module(x)

        # (B, 3, H, W)
        return x
    