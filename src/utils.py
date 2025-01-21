import torch
from collections import Counter

def assign_subgraph_labels(cluster_data):
    """
    Assigns labels to subgraphs based on the majority label of nodes in each subgraph.
    Args:
        cluster_data: Partitioned subgraph data from ClusterData.
    Returns:
        subgraph_labels: Tensor of labels for each subgraph.
    """
    subgraph_labels = []
    
    for i in range(len(cluster_data)):
        subgraph = cluster_data[i]
        node_labels = subgraph.y  # Get node labels directly from the subgraph
        majority_label = Counter(node_labels.tolist()).most_common(1)[0][0]  # Most common label
        subgraph_labels.append(majority_label)
    
    return torch.tensor(subgraph_labels)
