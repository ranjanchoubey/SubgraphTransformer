import torch
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
        token_embedding: Mean-pooled feature vector of size [num_features].
    """
    # This function remains the same as it works with regular tensors
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

class DGLGraphConv(torch.nn.Module):
    """Simple GCN model for generating embeddings"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGLGraphConv, self).__init__()
        # Add allow_zero_in_degree=True to handle isolated nodes
        self.conv1 = GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, output_dim, allow_zero_in_degree=True)

    def forward(self, g, features):
        h = torch.relu(self.conv1(g, features))
        h = self.conv2(g, h)
        return h

def compute_gcn_embeddings(subgraph, input_dim, hidden_dim, output_dim):
    """
    Computes GCN embeddings for a subgraph using DGL.
    Args:
        subgraph: DGL subgraph object.
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension.
    Returns:
        gcn_embeddings: GCN embeddings of size [num_nodes, output_dim].
    """
    # Add self-loops to handle isolated nodes
    subgraph = dgl.add_self_loop(subgraph)
    
    model = DGLGraphConv(input_dim, hidden_dim, output_dim)
    model.eval()
    with torch.no_grad():
        features = subgraph.ndata['feat']
        gcn_embeddings = model(subgraph, features)
    return gcn_embeddings