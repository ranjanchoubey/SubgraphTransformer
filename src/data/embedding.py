import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
from torch.linalg import eigh

def mean_pooling(embeddings):
    """
    Computes mean-pooled token embedding for a subgraph.
    Args:
        embeddings: Tensor of node embeddings.
    Returns:
        token_embedding: Mean-pooled feature vector.
    """
    # If embeddings is 1D, compute its mean and return as a tensor.
    if len(embeddings.shape) == 1:
        return embeddings.mean().unsqueeze(0)
    # Otherwise, compute mean along the node dimension.
    return torch.mean(embeddings, dim=0)

def compute_laplacian_positional_embedding(subgraph, embedding_dim=16):
    """
    Computes Laplacian positional embeddings for a subgraph using DGL.
    Args:
        subgraph: DGL subgraph object.
        embedding_dim: Dimension of the positional embedding.
    Returns:
        lpe: Laplacian positional embedding of size [num_nodes, embedding_dim].
    """
    # Get adjacency matrix
    adj = subgraph.adj().to_dense()
    
    # Compute degree matrix
    degrees = subgraph.in_degrees().float()
    D = torch.diag(degrees)
    
    # Compute normalized Laplacian
    L = D - adj
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(L)
    lpe = eigvecs[:, :embedding_dim]
    
    return lpe

class DGLGraphConv(nn.Module):
    """Standard 2-layer GCN for generating embeddings with dropout regularization."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(DGLGraphConv, self).__init__()
        # First graph convolution layer: from input to hidden representation.
        self.conv1 = GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
        # Second graph convolution layer: from hidden to output representation.
        self.conv2 = GraphConv(hidden_dim, output_dim, allow_zero_in_degree=True)
        # Dropout rate as used in many SOTA GCN implementations.
        self.dropout = dropout

    def forward(self, g, features):
        # First layer: convolution, followed by ReLU activation.
        h = self.conv1(g, features)
        h = F.relu(h)
        # Apply dropout after the activation.
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Second layer: compute final embeddings.
        h = self.conv2(g, h)
        return h

def compute_gcn_embeddings(subgraph, input_dim, hidden_dim, output_dim, dropout=0.5):
    """
    Computes GCN embeddings for a subgraph using the updated 2-layer GCN.
    Args:
        subgraph: DGL subgraph object.
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension.
        dropout: Dropout probability.
    Returns:
        gcn_embeddings: GCN embeddings of size [num_nodes, output_dim].
    """
    # Add self-loops to handle isolated nodes.
    subgraph = dgl.add_self_loop(subgraph)
    
    # Instantiate the updated GCN model with dropout.
    model = DGLGraphConv(input_dim, hidden_dim, output_dim, dropout=dropout)
    model.eval()  # Use evaluation mode as we're computing fixed embeddings.
    with torch.no_grad():
        features = subgraph.ndata['feat']
        gcn_embeddings = model(subgraph, features)
    return gcn_embeddings
