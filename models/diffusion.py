import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbeddings():
    def __init__(self):
        raise NotImplementedError
        

class UNet():
    def __init__(self):
        raise NotImplementedError


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
