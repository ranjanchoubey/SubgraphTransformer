import torch
import torch.nn as nn
from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward import FeedForward

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
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Initialize transformer encoder layer.
        
        Args:
            embed_dim: Dimension of the input embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            dropout: Dropout rate for regularization
        """
        super(TransformerEncoderLayer, self).__init__()
        # Self-attention layer for processing subgraph relationships
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network for transformation
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

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
        # Self-attention block
        attn_output = self.self_attn(x)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization

        # Feed-forward block
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        return x