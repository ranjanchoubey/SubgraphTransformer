    
import dgl

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
    
    # print(f"Graph partitioned into {num_parts} subgraphs.")
    return subgraphs

