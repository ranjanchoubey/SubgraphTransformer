import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=16, num_heads=4, num_layers=2, dropout=0.1, num_classes=7):
        """
        Graph Transformer model for node classification.
        Args:
            input_dim: Input feature size (e.g., 1433 for Cora).
            embed_dim: Size of the embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            num_classes: Number of classes for classification.
        """
        super(GraphTransformer, self).__init__()
        
        # Adjust input projection layer to accommodate concatenated input
        # self.input_proj = nn.Linear(input_dim, embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Ensures inputs are (batch, seq_len, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding, lpe_embedding):
        """
        Forward pass of the transformer.
        Args:
            token_embedding: [batch_size, input_dim]
            lpe_embedding: [batch_size, lpe_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        # Concatenate token embedding and LPE
        # x = torch.cat([token_embedding, lpe_embedding], dim=1)  # Shape: [batch_size, input_dim + lpe_dim]
        # x = self.input_proj(x)  # Project to embedding dimension
        # Add token embedding and LPE
        x = token_embedding + lpe_embedding  # Shape: [batch_size, input_dim]

        
        # Add a sequence length dimension for transformer (sequence length = 1 in our case)
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
        
        # Transformer forward pass
        x = self.transformer(x)  # Shape: [batch_size, 1, embed_dim]
        x = x.squeeze(1)  # Shape: [batch_size, embed_dim]
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)  # Shape: [batch_size, num_classes]
        return logits