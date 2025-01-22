import torch
import torch.nn as nn
from src.transformer_encoder import TransformerEncoder

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=16, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1, num_classes=7):
        """
        Graph Transformer model for subgraph classification.
        Args:
            input_dim: Input feature size (e.g., 1433 for Cora).
            embed_dim: Size of the embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward network dimension.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            num_classes: Number of classes for classification.
        """
        super(GraphTransformer, self).__init__()
        
        # Adjust input projection layer to accommodate input_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Custom transformer encoder layers
        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, subgraph_embedding, lpe_embedding):
        """
        Forward pass of the transformer.
        Args:
            subgraph_embedding: [batch_size, embed_dim]
            lpe_embedding: [batch_size, embed_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        # Ensure input tensors have the correct shape
        if subgraph_embedding.dim() == 1:
            subgraph_embedding = subgraph_embedding.unsqueeze(0)
        if lpe_embedding.dim() == 1:
            lpe_embedding = lpe_embedding.unsqueeze(0)

        print(f"Subgraph Embedding Shape: {subgraph_embedding.shape}")
        print(f"LPE Embedding Shape: {lpe_embedding.shape}")

        # Project subgraph embeddings to the embedding dimension
        subgraph_embedding = self.input_proj(subgraph_embedding)  # Shape: [batch_size, embed_dim]
        
        # Add subgraph embedding and LPE
        x = subgraph_embedding + lpe_embedding  # Shape: [batch_size, embed_dim]
        print(f"Combined Embedding Shape: {x.shape}")

        # Transformer forward pass
        x = self.transformer(x.unsqueeze(1)).squeeze(1)  # Shape: [batch_size, embed_dim]
        print(f"Transformer Output Shape: {x.shape}")

        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)  # Shape: [batch_size, num_classes]
        print(f"Logits Shape: {logits.shape}")

        return logits