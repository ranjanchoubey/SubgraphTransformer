import torch
import torch.nn as nn
from src.transformer_encoder import TransformerEncoder

class GraphTransformer(nn.Module):
    """
    A Graph Transformer model that operates on subgraphs to reduce attention complexity.
    Instead of computing attention over all nodes (N×N), it computes attention over subgraphs (K×K).
    
    Key Features:
    - Uses subgraph-level attention to reduce computational complexity
    - Combines GCN embeddings with positional encodings
    - Propagates subgraph predictions to all nodes within each subgraph
    """
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

    def forward(self, subgraph_embeddings, lpe_embeddings, num_nodes):
        """
        Processes subgraph embeddings through the transformer and propagates predictions to nodes.
        
        Process Flow:
        1. Project subgraph embeddings to common dimension
        2. Add positional encodings to maintain structural information
        3. Apply transformer attention over subgraphs
        4. Generate subgraph-level predictions
        5. Propagate predictions to all nodes in each subgraph
        
        Args:
            subgraph_embeddings: Encoded subgraph features [batch_size, embedding_dim]
            lpe_embeddings: Laplacian positional encodings [batch_size, embedding_dim]
            num_nodes: Number of nodes in each subgraph [batch_size]
            
        Returns:
            Node-level predictions for all nodes [total_num_nodes, num_classes]
        """
        # Ensure input tensors have the correct shape
        if subgraph_embeddings.dim() == 1:
            subgraph_embeddings = subgraph_embeddings.unsqueeze(0)
        if lpe_embeddings.dim() == 1:
            lpe_embeddings = lpe_embeddings.unsqueeze(0)

        print(f"Subgraph Embedding Shape: {subgraph_embeddings.shape}")
        print(f"LPE Embedding Shape: {lpe_embeddings.shape}")

        # Project subgraph embeddings to the embedding dimension
        subgraph_embeddings = self.input_proj(subgraph_embeddings)  # Shape: [batch_size, embed_dim]
        
        # Add subgraph embedding and LPE
        x = subgraph_embeddings + lpe_embeddings  # Shape: [batch_size, embed_dim]
        print(f"Combined Embedding Shape: {x.shape}")

        # Transformer forward pass
        x = self.transformer(x.unsqueeze(1)).squeeze(1)  # Shape: [batch_size, embed_dim]
        print(f"Transformer Output Shape: {x.shape}")

        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)  # Shape: [batch_size, num_classes]
        print(f"Subgraph Logits Shape: {logits.shape}")

        # Move num_nodes to the same device as logits
        num_nodes = num_nodes.to(logits.device)

        # Repeat subgraph-level predictions for each node in the subgraph
        logits = logits.repeat_interleave(num_nodes, dim=0)  # Shape: [total_num_nodes, num_classes]
        print(f"Node-level Logits Shape: {logits},{logits.shape}")

        return logits  # Shape: [total_num_nodes, num_classes]