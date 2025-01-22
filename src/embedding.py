import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch.linalg import eigh
from src.gcn import GCN  # Ensure GCN is imported correctly

def mean_pooling(embeddings):
    """
    Computes mean-pooled token embedding for a subgraph.
    Args:
        embeddings: Tensor of node embeddings.
    Returns:
        token_embedding: Mean-pooled feature vector of size [num_features].
    """
    token_embedding = torch.mean(embeddings, dim=0)  # Mean pooling over nodes
    return token_embedding

def compute_laplacian_positional_embedding(subgraph, embedding_dim=16):
    """
    Computes Laplacian positional embeddings for a subgraph.
    Args:
        subgraph: PyG subgraph object.
        embedding_dim: Dimension of the positional embedding.
    Returns:
        lpe: Laplacian positional embedding of size [num_nodes, embedding_dim].
    """
    adj = to_dense_adj(subgraph.edge_index)[0]
    D = torch.diag(adj.sum(dim=1))
    L = D - adj
    eigvals, eigvecs = eigh(L)
    lpe = eigvecs[:, :embedding_dim]
    return lpe

def compute_gcn_embeddings(subgraph, input_dim, hidden_dim, output_dim):
    """
    Computes GCN embeddings for a subgraph.
    Args:
        subgraph: PyG subgraph object.
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension.
    Returns:
        gcn_embeddings: GCN embeddings of size [num_nodes, output_dim].
    """
    model = GCN(input_dim, hidden_dim, output_dim)
    model.eval()
    with torch.no_grad():
        gcn_embeddings = model(subgraph.x, subgraph.edge_index)
    return gcn_embeddings