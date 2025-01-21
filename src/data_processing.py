# src/data_processing.py
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import ClusterData
from torch_geometric.transforms import NormalizeFeatures

def load_cora_data(data_path="data/cora/"):
    """
    Loads the Cora dataset and normalizes features.
    """
    dataset = Planetoid(root=data_path, name='Cora', transform=NormalizeFeatures())
    print(f"Dataset Loaded: {dataset.name}, Number of Graphs: {len(dataset)}")
    return dataset[0]  # Return the graph object

def partition_graph(graph, num_parts=10):
    """
    Partitions the graph into subgraphs using ClusterData (METIS).
    Args:
        graph: PyG graph object.
        num_parts: Number of subgraphs to partition.
    Returns:
        cluster_data: Partitioned subgraph data.
    """
    cluster_data = ClusterData(graph, num_parts=num_parts, recursive=False)
    print(f"Graph partitioned into {num_parts} subgraphs.")
    return cluster_data

# Example usage in main.py: check uncomment below code to check this data_processing.py
def main():
    # Step 1: Load the Cora dataset
    print("Loading Cora dataset...")
    graph = load_cora_data()  # Returns the full graph
    print(f"Graph Info:\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}, Features: {graph.num_node_features}")

    # Step 2: Partition the graph into subgraphs
    print("\nPartitioning the graph...")
    num_parts = 100  # Number of subgraphs
    cluster_data = partition_graph(graph, num_parts=num_parts)
    print(f"Partitioning completed. Number of subgraphs: {len(cluster_data)}")
    
    for i in range(num_parts):
        subgraph = cluster_data[i]
        print("subgraph {} Size : ",i,subgraph)

if __name__ == "__main__":
    main()
