import torch
import torch.nn as nn
from models.transformer_encoder import TransformerEncoder

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
        
        # Store num_classes as instance variable
        self.num_classes = num_classes
        
        # Adjust input projection layer to accommodate input_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Custom transformer encoder layers
        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, subgraph_embeddings, lpe_embeddings, node_counts, node_indices):  # Changed parameter name
        device = self.classifier.weight.device

        try:
            # Ensure input tensors have the correct shape
            if subgraph_embeddings.dim() == 1:
                subgraph_embeddings = subgraph_embeddings.unsqueeze(0)
            if lpe_embeddings.dim() == 1:
                lpe_embeddings = lpe_embeddings.unsqueeze(0)

            # print(f"Subgraph Embedding Shape: {subgraph_embeddings.shape}")
            # print(f"LPE Embedding Shape: {lpe_embeddings.shape}")

            # Project subgraph embeddings to the embedding dimension
            subgraph_embeddings = self.input_proj(subgraph_embeddings)  # Shape: [batch_size, embed_dim]
            
            # Add subgraph embedding and LPE
            x = subgraph_embeddings + lpe_embeddings  # Shape: [batch_size, embed_dim]
            # print(f"Combined Embedding Shape: {x.shape}")

            # Transformer forward pass
            x = self.transformer(x.unsqueeze(1)).squeeze(1)  # Shape: [batch_size, embed_dim]
            # print(f"Transformer Output Shape: {x.shape}")

            # Classification
            x = self.dropout(x)
            subgraph_predictions = self.classifier(x)  # [batch_size, num_classes]
            device = subgraph_predictions.device

            # Initialize lists to store predictions and indices for ALL nodes
            node_predictions = []
            node_indices_out = []
            
            start_idx = 0
            for i, (pred, count) in enumerate(zip(subgraph_predictions, node_counts)):  # Changed variable name
                end_idx = start_idx + count
                curr_indices = node_indices[i].to(device)
                
                # Expand predictions to all nodes in the subgraph
                node_predictions.append(pred.expand(len(curr_indices), -1))
                node_indices_out.append(curr_indices)
                
                start_idx = end_idx

            # Stack predictions and indices
            node_predictions = torch.cat(node_predictions, dim=0)
            node_indices_out = torch.cat(node_indices_out, dim=0)

            return node_predictions, node_indices_out
        
        except Exception as e:
            print(f"\nError in transformer forward: {str(e)}")
            print(f"Error location: {e.__traceback__.tb_lineno}")
            raise e