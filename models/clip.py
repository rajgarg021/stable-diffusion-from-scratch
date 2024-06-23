import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPTextEmbeddings(nn.Module):
    """
    Module to generate text embeddings for CLIP
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
    Single text encoder layer of CLIP
    """

    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()

        # self.n_head = n_head
        # self.head_size = d_embed // n_head

        self.ln1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_heads, d_embed)
        self.ln2 = nn.LayerNorm(d_embed)
        self.linear1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.tensor):
        """ x: (B, seq_len, d_embed) """

        ## SELF ATTENTION
        residue = x
        x = self.ln1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        ## FEED FORWARD
        residue = x
        x = self.ln2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x) # quick GELU activation function
        x = self.linear2(x)
        x += residue

        return x
    

class CLIP(nn.Module):
    """
    CLIP Model
    """

    def __init__(self):
        self.embeddings = CLIPTextEmbeddings(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(n_heads=12, d_embed=768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor):
        tokens = tokens.type(torch.long)

        # (B, seq_len) -> (B, seq_len, d_embed)
        state = self.embeddings(tokens)

        for layer in self.layers:
            state = layer(state)

        # (B, seq_len, d_embed)
        output = self.layernorm(state)

        return output
    