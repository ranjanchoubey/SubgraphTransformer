import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        batch_size, num_nodes, embed_dim = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, num_nodes, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_nodes, 3 * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, num_nodes, embed_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output