import dgl
"""
    File to load dataset based on user control from main file
"""
def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'Cora':  
        graph = dgl.data.CoraGraphDataset()
        dataset = graph[0]
        # print(f"Dataset Loaded: Cora, Number of nodes: {dataset.number_of_nodes()}")
        
        return dataset
    
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
