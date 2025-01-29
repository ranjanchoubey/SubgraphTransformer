import torch
import torch.nn as nn
from src.layers.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import FeedForward

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder implementing the standard architecture:
    MultiHeadAttention -> LayerNorm -> FeedForward -> LayerNorm
    
    Features:
    - Self-attention mechanism for capturing subgraph relationships
    - Layer normalization for training stability
    - Feed-forward networks for non-linear transformations
    - Residual connections for better gradient flow
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1,
                 layer_norm=True, batch_norm=True, residual=True):
        """
        Initialize transformer encoder layer.
        
        Args:
            embed_dim: Dimension of the input embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            dropout: Dropout rate for regularization
            layer_norm: Flag to apply layer normalization
            batch_norm: Flag to apply batch normalization
            residual: Flag to apply residual connections
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.residual = residual
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        
        # Conditional normalization layers
        if layer_norm:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.input_norm = nn.LayerNorm(embed_dim)
            self.output_norm = nn.LayerNorm(embed_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.input_norm = nn.Identity()
            self.output_norm = nn.Identity()
            
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(embed_dim)
            self.batch_norm2 = nn.BatchNorm1d(embed_dim)
        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()
            
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the transformer encoder layer.
        
        Process:
        1. Self-attention on input
        2. Add & Norm with residual connection
        3. Feed-forward transformation
        4. Add & Norm with residual connection
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Processed tensor of same shape as input
        """
        # Input normalization
        x = self.input_norm(x)
        
        # First sublayer: Multi-head attention with residual
        attn_input = self.norm1(x)
        attn_output = self.self_attn(attn_input)
        if self.residual:
            x = x + self.dropout1(attn_output)
        else:
            x = attn_output
        
        # Second sublayer: Feed-forward with residual
        ff_input = self.norm2(x)
        ff_output = self.ffn(ff_input)
        if self.residual:
            x = x + self.dropout2(ff_output)
        else:
            x = ff_output
        
        # Output normalization
        x = self.output_norm(x)
        return x