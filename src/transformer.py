# src/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=32, num_heads=4, num_layers=2, dropout=0.1, num_classes=7, lpe_dim=16):
        """
        Graph Transformer model for node classification.
        Args:
            input_dim: Input feature size (e.g., 1433 for Cora).
            embed_dim: Size of the embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            num_classes: Number of classes for classification.
            lpe_dim: Dimension of Laplacian Positional Embedding.
        """
        super(GraphTransformer, self).__init__()
        
        # Adjust input projection layer to accommodate concatenated input
        self.input_proj = nn.Linear(input_dim + lpe_dim, embed_dim)
        
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
        x = torch.cat([token_embedding, lpe_embedding], dim=1)  # Shape: [batch_size, input_dim + lpe_dim]
        x = self.input_proj(x)  # Project to embedding dimension
        
        # Add a sequence length dimension for transformer (sequence length = 1 in our case)
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
        
        # Transformer forward pass
        x = self.transformer(x)  # Shape: [batch_size, 1, embed_dim]
        x = x.squeeze(1)  # Shape: [batch_size, embed_dim]
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)  # Shape: [batch_size, num_classes]
        return logits






# uncomment and run the code for testing transformer.py

# from data_processing import load_cora_data, partition_graph
# from embedding import mean_pooling, compute_laplacian_positional_embedding
# from transformer import GraphTransformer
# import torch

# def main():
#     # Step 1: Load the Cora dataset
#     print("Loading Cora dataset...")
#     graph = load_cora_data()
#     print(f"Graph Info:\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}, Features: {graph.num_node_features}")

#     # Step 2: Partition the graph into subgraphs
#     print("\nPartitioning the graph...")
#     num_parts = 10
#     cluster_data = partition_graph(graph, num_parts=num_parts)
#     print(f"Partitioning completed. Number of subgraphs: {len(cluster_data)}")

#     # Step 3: Compute embeddings for all subgraphs
#     token_embeddings = []
#     lpe_embeddings = []
#     for i in range(num_parts):
#         subgraph = cluster_data[i]
#         token_embeddings.append(mean_pooling(subgraph))
#         lpe_embeddings.append(compute_laplacian_positional_embedding(subgraph, embedding_dim=16))

#     token_embeddings = torch.stack(token_embeddings)  # Shape: [num_parts, input_dim]
#     lpe_embeddings = torch.stack(lpe_embeddings)  # Shape: [num_parts, embedding_dim]

#     # Step 4: Initialize transformer model
#     print("\nInitializing Graph Transformer...")
#     input_dim = token_embeddings.shape[1]
#     # Initialize Graph Transformer with adjusted dimensions
#     model = GraphTransformer(input_dim=1433, embed_dim=32, num_heads=4, num_layers=2, num_classes=7, lpe_dim=16)


#     # Forward pass
#     logits = model(token_embeddings, lpe_embeddings)
#     print(f"Logits Shape: {logits.shape}")

# if __name__ == "__main__":
#     main()
