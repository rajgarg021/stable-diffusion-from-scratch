import torch
import torch.nn as nn
import torch.nn.functional as F


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

        return x
        

class ResidualBlock():
    def __init__(self):
        raise NotImplementedError
    

class AttentionBlock():
    def __init__(self):
        raise NotImplementedError
    

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


class Unet(nn.Module):
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
            # since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNetFinalLayer():
    def __init__(self):
        raise NotImplementedError

        
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
