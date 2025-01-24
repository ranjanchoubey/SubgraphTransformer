import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for processing subgraph relationships.
    
    Features:
    - Parallel attention heads for diverse feature learning
    - Scaled dot-product attention
    - Linear projections for queries, keys, and values
    
    Purpose:
    - Captures relationships between subgraphs
    - Enables learning different types of interactions
    - Maintains computational efficiency
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of parallel attention heads
            dropout: Dropout rate for regularization
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Forward pass of multi-head attention.
        
        Process:
        1. Project input to queries, keys, and values
        2. Split into multiple heads
        3. Compute scaled dot-product attention
        4. Combine heads and project output
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Attended features [batch_size, seq_len, embed_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        batch_size, num_nodes, embed_dim = x.size()
        qkv = self.qkv(x).reshape(batch_size, num_nodes, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_nodes, 3 * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, num_nodes, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output