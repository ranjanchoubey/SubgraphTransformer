# src/data_processing.py
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import ClusterData
from torch_geometric.transforms import NormalizeFeatures

def load_cora_data(data_path="data/cora/"):
    """
    Loads the Cora dataset and normalizes features.
    """
    dataset = Planetoid(root=data_path, name='Cora',
    
    # The `transform=NormalizeFeatures()` argument in
    # the `Planetoid` dataset initialization is
    # applying a transformation to normalize the node
    # features of the dataset. This transformation
    # normalizes the node features of the graph data,
    # typically by subtracting the mean and dividing
    # by the standard deviation of each feature
    # across all nodes in the graph. Normalizing
    # features can help improve the training process
    # and model performance by ensuring that the
    # features are on a similar scale and have a
    # consistent distribution.
    
    transform=NormalizeFeatures())
    
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


