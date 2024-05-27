import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    Multi-Head Self Attention
    """

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.head_size = d_embed // n_heads

    def forward(self, x: torch.tensor, causal_mask=False):
        """ x: (batch_size, seq_len, channels) """

        input_shape = x.shape
        B, seq_len, C = input_shape

        # (B, seq_len, C) -> (B, seq_len, C * 3) -> 3 tensors of shape (B, seq_len, C)
        q, k ,v = self.in_proj(x).chunk(3, dim=-1)

        interim_shape = (B, seq_len, self.n_heads, self.head_size)

        # (B, seq_len, C) -> (B, seq_len, n_heads, head_size) -> (B, n_heads, seq_len, head_size)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # computing attention scores
        scores = q @ k.transpose(-1, -2) # (B, n_heads, seq_len, seq_len)

        if causal_mask:
            # creating a mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            # filling the masked out values with -inf so they become zero when we apply softmax
            scores.masked_fill(mask, -torch.inf)

        scores /= math.sqrt(self.head_size)
        scores = F.softmax(scores, dim=-1)
        
        # (B, n_heads, seq_len, seq_len) @ (B, n_heads, seq_len, head_size) -> (B, n_heads, seq_len, head_size)
        output = scores @ v

        # (B, n_heads, seq_len, head_size) -> (B, seq_len, n_heads, head_size) 
        output = output.transpose(1, 2)

        # (B, seq_len, n_heads, head_size) -> (B, seq_len, C)
        output = output.reshape(input_shape)

        # (B, seq_len, C) -> (B, seq_len, C)
        output = self.out_proj(output)

        return output
    