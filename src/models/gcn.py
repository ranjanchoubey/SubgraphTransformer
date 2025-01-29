import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for generating node embeddings.
    
    Architecture:
    - Two GCN layers with ReLU activation
    - First layer: input_dim → hidden_dim
    - Second layer: hidden_dim → output_dim
    
    Purpose:
    - Learns node representations by aggregating neighborhood information
    - Reduces input dimension while preserving graph structure
    - Generates embeddings suitable for downstream tasks
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize GCN layers.
        
        Args:
            input_dim: Original feature dimension (e.g., 1433 for Cora)
            hidden_dim: Intermediate layer dimension (e.g., 64)
            output_dim: Final embedding dimension (e.g., 16)
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of GCN.
        
        Process:
        1. First convolution + ReLU
        2. Second convolution
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x