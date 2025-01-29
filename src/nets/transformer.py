import torch
import torch.nn as nn
import torch.nn.init as init
from src.nets.transformer_encoder import TransformerEncoder
from src.nets.mlp_readout import MLPReadout

class GraphTransformer(nn.Module):
    """
    A Graph Transformer model that operates on subgraphs to reduce attention complexity.
    Instead of computing attention over all nodes (N×N), it computes attention over subgraphs (K×K).
    
    Key Features:
    - Uses subgraph-level attention to reduce computational complexity
    - Combines GCN embeddings with positional encodings
    - Propagates subgraph predictions to all nodes within each subgraph
    """
    def __init__(self, input_dim, embed_dim=16, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1, num_classes=7, layer_norm=True, batch_norm=True, residual=True):
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
            layer_norm: Whether to use layer normalization.
            batch_norm: Whether to use batch normalization.
            residual: Whether to use residual connections.
        """
        super(GraphTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Transformer layers
        self.transformer = TransformerEncoder(
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_layers, 
            dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            residual=residual
        )
        
        # MLP readout
        self.mlp_head = MLPReadout(embed_dim, num_classes)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        # Better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, subgraph_embeddings, lpe_embeddings, node_counts, node_indices):
        try:
            print("Initial shapes:")
            print(f"subgraph_embeddings: {subgraph_embeddings.shape}")
            
            # Initial projection with skip connection
            x = self.input_proj(subgraph_embeddings)
            x = self.norm(x + lpe_embeddings)  # Improved residual connection
            
            # Add batch dimension
            x = x.unsqueeze(0)
            print(f"Transformer input shape: {x.shape}")
            
            # Apply transformer
            x = self.transformer(x)
            
            # Readout
            x = self.dropout(x)
            x = x.squeeze(0)
            subgraph_predictions = self.mlp_head(x)
            print(f"Subgraph predictions shape: {subgraph_predictions.shape}")

            # Node predictions
            node_predictions = []
            node_indices_out = []
            
            for i, count in enumerate(node_counts):
                curr_indices = node_indices[i]
                expanded_pred = subgraph_predictions[i].expand(len(curr_indices), -1)
                node_predictions.append(expanded_pred)
                node_indices_out.append(curr_indices)

            node_predictions = torch.cat(node_predictions, dim=0)
            node_indices_out = torch.cat(node_indices_out, dim=0)
            print(f"Final node predictions shape: {node_predictions.shape}")

            return node_predictions, node_indices_out

        except Exception as e:
            print(f"Error in transformer forward: {str(e)}")
            raise e
            
    def loss(self, pred, label):
        # Weighted cross-entropy loss for unbalanced classes
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.num_classes).long().to(pred.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        criterion = nn.CrossEntropyLoss(weight=weight)
        return criterion(pred, label)