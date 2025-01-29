import torch
import torch.nn as nn
import numpy as np
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
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_bias=True):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of parallel attention heads
            dropout: Dropout rate for regularization
            use_bias: Whether to use bias in linear projections
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate Q,K,V projections like in the DGL implementation
        if use_bias:
            self.Q = nn.Linear(embed_dim, embed_dim, bias=True)
            self.K = nn.Linear(embed_dim, embed_dim, bias=True)
            self.V = nn.Linear(embed_dim, embed_dim, bias=True)
        else:
            self.Q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.V = nn.Linear(embed_dim, embed_dim, bias=False)
            
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Add normalization layers
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Add attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Xavier initialization for attention
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Improved attention computation with better numerical stability
        Q = self.q_norm(self.Q(x))
        K = self.k_norm(self.K(x))
        V = self.v_norm(self.V(x))
        
        # Scale Q instead of scores for better gradient flow
        Q = Q * self.scale
        
        # Reshape and compute attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores with improved stability
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores.clamp(-10, 10)  # Increased clamp range
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention and dropout
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.embed_dim)
        out = self.proj_dropout(self.out_proj(out))
        out = self.output_norm(out)
        
        return out