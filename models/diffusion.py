import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbeddings(nn.Module):
    """
    Module to generate time embeddings
    """

    def __init__(self, d_embed: int):
        super().__init__()
        self.ln1 = nn.Linear(d_embed, 4 * d_embed)
        self.ln2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.tensor):
        """ x: (1, 320) """

        x = self.ln1(x)
        x = F.silu(x)
        x = self.ln2(x)

        # (1, 1280)
        return x
        

class ResidualBlock(nn.Module):
    """
    Residual block for the UNet
    """

    def __init__(self, in_channels: int, out_channels: int, d_time=1280):
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear = nn.Linear(d_time, out_channels)
        
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, latents: torch.tensor, time: torch.tensor):
        """
        latents: (B, in_channels, H, W)
        time: (1, 1280)
        """

        residue = latents
        latents = self.gn1(latents)
        latents = F.silu(latents)
        latents = self.conv1(latents)

        time = F.silu(time)
        time = self.linear(time)

        merged = latents + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.gn2(merged)
        merged = F.silu(merged)
        merged = self.conv2(merged)

        return merged + self.residual_layer(residue)
    

class AttentionBlock():
    """
    Attention block for the UNet
    """

    def __init__(self, n_heads: int, d_embed: int, d_context=768):
        super().__init__()

        channels = n_heads * d_embed
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.ln1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.ln2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.ln3 = nn.LayerNorm(channels)
        self.geglu1  = nn.Linear(channels, 4 * channels * 2)
        self.geglu2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        """
        x: (B, C, H, W)
        context: (B, seq_len, d_context)
        """
        
        residue_long = x

        # (B, C, H, W) -> (B, C, H, W)
        x = self.gn(x)
        
        # (B, C, H, W) -> (B, C, H, W)
        x = self.conv_input(x)
        
        B, C, H, W = x.shape
        
        # (B, C, H, W) -> (B, C, H * W)
        x = x.view((B, C, H * W))
        
        # (B, C, H * W) -> (B, H * W, C)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (B, H * W, C)
        residue_short = x
        
        # (B, H * W, C) -> (B, H * W, C)
        x = self.ln1(x)
        
        # (B, H * W, C) -> (B, H * W, C)
        x = self.attention1(x)
        
        # (B, H * W, C) + (B, H * W, C) -> (B, H * W, C)
        x += residue_short
        
        # (B, H * W, C)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (B, H * W, C) -> (B, H * W, C)
        x = self.ln2(x)
        
        # (B, H * W, C) -> (B, H * W, C)
        x = self.attention2(x, context)
        
        # (B, H * W, C) + (B, H * W, C) -> (B, H * W, C)
        x += residue_short
        
        # (B, H * W, C)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (B, H * W, C) -> (B, H * W, C)
        x = self.ln3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (B, H * W, C) -> two tensors of shape (B, H * W, C * 4)
        x, gate = self.geglu1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (B, H * W, C * 4) * (B, H * W, C * 4) -> (B, H * W, C * 4)
        x = x * F.gelu(gate)
        
        # (B, H * W, C * 4) -> (B, H * W, C)
        x = self.geglu2(x)
        
        # (B, H * W, C) + (B, H * W, C) -> (B, H * W, C)
        x += residue_short
        
        # (B, H * W, C) -> (B, C, H * W)
        x = x.transpose(-1, -2)
        
        # (B, C, H * W) -> (B, C, H, W)
        x = x.view((B, C, H, W))

        # Final skip connection between initial input and output of the block
        # (B, C, H, W) + (B, C, H, W) -> (B, C, H, W)
        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    """
    Upsampling block
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (B, C, H, W) -> (B, C, H*2, W*2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """
    Applying parameters based on the layer
    """

    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    """
    neural network for the reverse diffusion process
    """

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (B, 4, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (B, 320, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            
            # (B, 320, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            
            # (B, 320, H/8, W/8) -> (B, 320, H/16, W/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (B, 320, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16)
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            
            # (B, 640, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16)
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            
            # (B, 640, H/16, W/16) -> (B, 640, H/32, W/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (B, 640, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32)
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            
            # (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32)
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            
            # (B, 1280, H/32, W/32) -> (B, 1280, H/64, W/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (B, 1280, H/64, W/64) -> (B, 1280, H/64, W/64)
            SwitchSequential(ResidualBlock(1280, 1280)),
            
            # (B, 1280, H/64, W/64) -> (B, 1280, H/64, W/64)
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (B, 1280, H/64, W/64) -> (B, 1280, H/64, W/64)
            ResidualBlock(1280, 1280), 
            
            # (B, 1280, H/64, W/64) -> (B, 1280, H/64, W/64)
            AttentionBlock(8, 160), 
            
            # (B, 1280, H/64, W/64) -> (B, 1280, H/64, W/64)
            ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # (B, 2560, H/64, W/64) -> (B, 1280, H/64, W/64)
            SwitchSequential(ResidualBlock(2560, 1280)),
            
            # (B, 2560, H/64, W/64) -> (B, 1280, H/64, W/64)
            SwitchSequential(ResidualBlock(2560, 1280)),
            
            # (B, 2560, H/64, W/64) -> (B, 1280, H/64, W/64) -> (B, 1280, H/32, W/32) 
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (B, 2560, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32)
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            
            # (B, 2560, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32)
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            
            # (B, 1920, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/32, W/32) -> (B, 1280, H/16, W/16)
            SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)),
            
            # (B, 1920, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16)
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            
            # (B, 1280, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16)
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            
            # (B, 960, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/16, W/16) -> (B, 640, H/8, W/8)
            SwitchSequential(ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)),
            
            # (B, 960, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            
            # (B, 640, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            
            # (B, 640, H/8, W/8) -> (B, 320, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        """
        x: (B, 4, H/8, W/8)
        context: (B, seq_len, d_embed) 
        time: (1, 1280)
        """

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # since we always concat with the skip connection of the encoder, the number of C increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNetFinalLayer():
    """
    Final output layer of our UNet
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """ x: (B, 320, H/8, W/8) """

        x = self.gn(x)
        x = F.silu(x)
        x = self.conv(x)

        # (B, 4, H/4, W/8)
        return x


class LDM(nn.Module):
    """
    Latent Diffusion Model
    """

    def __init__(self):
        super().__init__()
        self.time_embeddings = TimeEmbeddings(320)
        self.unet = UNet()
        self.final = UNetFinalLayer(320, 4)
    
    def forward(self, latent, context, time):
        """
        latent: (B, 4, H/8, W/8)
        context: (B, seq_len, d_embed)
        time: (1, 320)
        """

        # (1, 320) -> (1, 1280)
        time = self.time_embeddings(time)
        
        # (B, 4, H/8, W/8) -> (B, 320, H/8, W/8)
        output = self.unet(latent, context, time)
        
        # (B, 320, H/8, W/8) -> (B, 4, H/8, W/8)
        output = self.final(output)
        
        # (B, 4, H/ 8, W/8)
        return output
