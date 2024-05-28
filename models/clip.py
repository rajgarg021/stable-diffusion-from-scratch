import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """
    Module to generate embeddings for CLIP
    """

    def __init__(self, n_vocab: int, d_embed: int, n_tokens: int):
        super().__init__()

        self.token_embeddings = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_embed)

        # a learnable weight matrix encodes the position information for each token
        self.position_embedddings = nn.Parameter(torch.zeros(n_tokens, d_embed))

    def forward(self, tokens):
        
        # (B, seq_len) -> (B, seq_len, Dim)
        x = self.token_embeddings(tokens)
        x += self.position_embedddings

        return x
