import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch.linalg import eigh
from src.gcn import GCN  # Ensure GCN is imported correctly


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
    adj = to_dense_adj(subgraph.edge_index)[0]
    degree_matrix = torch.diag(adj.sum(dim=1))
    laplacian = degree_matrix - adj  # Laplacian matrix

    eigenvalues, eigenvectors = eigh(laplacian)
    lpe = eigenvectors[:, 1:embedding_dim+1]  # Skip the first eigenvector (constant vector)
    lpe_mean = torch.mean(lpe, dim=0)  # Mean-pool for the subgraph-level LPE

    return lpe_mean

def compute_gcn_embeddings(subgraph, input_dim, hidden_dim, output_dim):
    """
    Computes GCN embeddings for a subgraph.
    Args:
        subgraph: PyG subgraph object.
        input_dim: Input feature size.
        hidden_dim: Hidden layer size.
        output_dim: Output feature size.
    Returns:
        gcn_embeddings: GCN embeddings of size [num_nodes, output_dim].
    """
    gcn = GCN(input_dim, hidden_dim, output_dim)
    gcn_embeddings = gcn(subgraph.x, subgraph.edge_index)


    return gcn_embeddings


















# Run the below code for testing embedding.py
# from gcn import GCN
# from data_processing import load_cora_data, partition_graph
# def main():
#     # Step 1: Load the Cora dataset
#     print("Loading Cora dataset...")
#     graph = load_cora_data()
#     print(f"Graph Info:\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}, Features: {graph.num_node_features}")

#     # Step 2: Partition the graph into subgraphs
#     print("\nPartitioning the graph...")
#     num_parts = 100
#     cluster_data = partition_graph(graph, num_parts=num_parts)
#     print(f"Partitioning completed. Number of subgraphs: {len(cluster_data)}")

    # Step 3: Compute embeddings for the first subgraph
    
    # for i in range(num_parts):
    #     subgraph = cluster_data[i]
    #     print("subgraph {} Size : ",i,subgraph)
    
    #     subgraph = cluster_data[i]  # First subgraph
    #     gcn_embeddings = compute_gcn_embeddings(subgraph, input_dim=1433, hidden_dim=64, output_dim=32)
    #     token_embedding = torch.mean(gcn_embeddings, dim=0)  # Mean pooling
    #     print(i,token_embedding.shape)
        # lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=16)

        # print(f"\nToken Embedding (Mean-Pooled): {token_embedding.shape}")
        # print(f"Laplacian Positional Embedding (Subgraph-Level): {lpe.shape}")

# if __name__ == "__main__":
#     main()