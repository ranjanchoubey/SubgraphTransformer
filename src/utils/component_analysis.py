import torch
import dgl
import networkx as nx

def get_component_info(subgraph):
    """
    Returns a list of tuples for each component:
       (sorted_component_indices, component_size)
    The indices are relative to the node ordering in the subgraph.
    """
    nx_graph = subgraph.to_networkx().to_undirected()
    components = list(nx.connected_components(nx_graph))
    
    # Assume subgraph.nodes() returns nodes in the order they appear in the subgraph.
    node_list = list(subgraph.nodes())
    component_info = []
    for comp in components:
        # Find indices of nodes (in node_list) that belong to this component.
        comp_indices = [node_list.index(n) for n in comp]
        comp_indices = sorted(comp_indices)  # sort if needed
        component_info.append((comp_indices, len(comp_indices)))
    
    return component_info
