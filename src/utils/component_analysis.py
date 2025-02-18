import torch
import dgl
import networkx as nx

def get_component_info(subgraph):
    """
    Get sizes of connected components in a subgraph
    Args:
        subgraph: DGL subgraph object
    Returns:
        component_sizes: List of component sizes [n1, n2, n3, ...] where ni is number of nodes in component i
    """
    # Convert to networkx for component analysis
    nx_graph = subgraph.to_networkx().to_undirected()
    
    # Get connected components
    components = list(nx.connected_components(nx_graph))
    
    # Get size of each component
    component_sizes = [len(component) for component in components]
    
    # For debugging
    print(f"Found {len(component_sizes)} components with sizes: {component_sizes}")
    
    return component_sizes
