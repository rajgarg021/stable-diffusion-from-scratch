import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbeddings(nn.Module):
    """
    Module to generate embeddings for CLIP
    """

    def __init__(self, n_vocab: int, d_embed: int, n_tokens: int):
        super().__init__()

        self.token_embeddings = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_embed)

        # a learnable weight matrix encodes the position information for each token
        self.position_embedddings = nn.Parameter(torch.zeros(n_tokens, d_embed))

    def forward(self, tokens):
        
        # (B, seq_len) -> (B, seq_len, d_embed)
        embeddings = self.token_embeddings(tokens)
        embeddings += self.position_embedddings

        return embeddings


class CLIPLayer(nn.Module):
    """
    Single encoder layer of CLIP
    """

    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()

        # self.n_head = n_head
        # self.head_size = d_embed // n_head

        self.norm1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_heads, d_embed)
        self.norm2 = nn.LayerNorm(d_embed)
        self.ln1 = nn.Linear(d_embed, 4 * d_embed)
        self.ln2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.tensor):
        """ x: (B, seq_len, d_embed) """

        # SELF ATTENTION
        residue = x
        x = self.norm1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        # FEED FORWARD
        residue = x
        x = self.norm2(x)
        x = self.ln1(x)
        x = x * torch.sigmoid(1.702 * x) # quick GELU activation function
        x = self.ln2(x)
        x += residue

        return x
    