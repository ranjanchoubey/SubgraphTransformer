import dgl
import torch

def load_cora_data():
    """
    Loads the Cora dataset using DGL.
    """
    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0]
    print(f"Dataset Loaded: Cora, Number of nodes: {graph.number_of_nodes()}")
    return graph

def partition_graph(graph, num_parts=4):
    """
    Partitions the graph into subgraphs using DGL's METIS implementation.
    Args:
        graph: DGL graph object
        num_parts: Number of partitions
    Returns:
        list of subgraphs
    """
    # Get partition assignments for each node
    partition_labels = dgl.metis_partition_assignment(graph, num_parts)
    
    # Add partition labels as node features
    graph.ndata['part_id'] = partition_labels
    
    # Create subgraphs
    subgraphs = []
    for part_id in range(num_parts):
        node_mask = graph.ndata['part_id'] == part_id
        subgraph = dgl.node_subgraph(graph, node_mask)
        subgraphs.append(subgraph)
    
    print(f"Graph partitioned into {num_parts} subgraphs.")
    return subgraphs

def test_embeddings(graph, subgraphs):
    """
    Test different embedding methods on the graph and subgraphs.
    """
    from embedding import (mean_pooling, compute_laplacian_positional_embedding,
                         compute_gcn_embeddings)
    
    print("\nTesting Embedding Methods:")
    
    # Test on first subgraph
    test_subg = subgraphs[0]
    
    # 1. Test mean pooling
    print("\n1. Testing Mean Pooling:")
    features = test_subg.ndata['feat']
    pooled = mean_pooling(features)
    print(f"Mean pooled shape: {pooled.shape}")
    
    # 2. Test Laplacian PE
    print("\n2. Testing Laplacian Positional Embedding:")
    lpe = compute_laplacian_positional_embedding(test_subg, embedding_dim=16)
    print(f"LPE shape: {lpe.shape}")
    
    # 3. Test GCN embeddings
    print("\n3. Testing GCN Embeddings:")
    input_dim = test_subg.ndata['feat'].shape[1]
    hidden_dim = 32
    output_dim = 16
    gcn_emb = compute_gcn_embeddings(test_subg, input_dim, hidden_dim, output_dim)
    print(f"GCN embeddings shape: {gcn_emb.shape}")

def main():
    # Load the dataset
    graph = load_cora_data()
    
    # Partition the graph
    num_partitions = 100
    subgraphs = partition_graph(graph, num_partitions)
    
    # Print statistics about the partitions
    print("\nPartition Statistics:")
    for i, subgraph in enumerate(subgraphs):
        print(f"Partition {i}:")
        print(f"  Nodes: {subgraph.number_of_nodes()}")
        print(f"  Edges: {subgraph.number_of_edges()}")
        print(f"  Features shape: {subgraph.ndata['feat'].shape}")
    
    # Test embedding methods
    test_embeddings(graph, subgraphs)

if __name__ == "__main__":
    main()


