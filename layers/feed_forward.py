import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Feed-forward network used in Transformer encoder layers.
    
    Architecture:
    - Two linear transformations with GELU activation
    - Dropout for regularization
    
    Purpose:
    - Processes attention outputs
    - Introduces non-linearity
    - Transforms features while maintaining dimensionality
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Input and output dimension
            ff_dim: Hidden layer dimension (usually 4x embed_dim)
            dropout: Dropout rate for regularization
        """
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Transformed tensor of same shape as input
        """
        return self.net(x)