# src/embedding.py
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch.linalg import eigh

def mean_pooling(subgraph):
    """
    Computes mean-pooled token embedding for a subgraph.
    Args:
        subgraph: PyG subgraph object.
    Returns:
        token_embedding: Mean-pooled feature vector of size [num_features].
    """
    token_embedding = torch.mean(subgraph.x, dim=0)  # Mean pooling over nodes
    return token_embedding

def compute_laplacian_positional_embedding(subgraph, embedding_dim=16):
    """
    Computes Laplacian positional embedding (LPE) for a subgraph.
    Args:
        subgraph: PyG subgraph object.
        embedding_dim: Size of the positional embedding.
    Returns:
        lpe: Laplacian positional embedding vector of size [embedding_dim].
    """
    # Get the adjacency matrix in dense format
    adj = to_dense_adj(subgraph.edge_index)[0]
    degree_matrix = torch.diag(adj.sum(dim=1))
    laplacian = degree_matrix - adj  # Laplacian matrix

    # Compute the eigenvectors of the Laplacian
    eigenvalues, eigenvectors = eigh(laplacian)
    lpe = eigenvectors[:, 1:embedding_dim+1]  # Skip the first eigenvector (constant vector)
    lpe_mean = torch.mean(lpe, dim=0)  # Mean-pool for the subgraph-level LPE

    return lpe_mean


# # Run the below code for testing embedding.py
# from data_processing import load_cora_data, partition_graph
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

#     # Step 3: Compute embeddings for the first subgraph
#     subgraph = cluster_data[0]  # First subgraph
#     token_embedding = mean_pooling(subgraph)
#     lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=16)

#     print(f"\nToken Embedding (Mean-Pooled): {token_embedding.shape}")
#     print(f"Laplacian Positional Embedding (Subgraph-Level): {lpe.shape}")

# if __name__ == "__main__":
#     main()
