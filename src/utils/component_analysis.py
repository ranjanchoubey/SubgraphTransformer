import torch
import dgl
import networkx as nx

def get_component_info(subgraph):
    """
    Analyze connected components in a subgraph
    Returns dict with component scores and node mappings
    """
    # Convert to networkx for component analysis
    nx_graph = dgl.to_networkx(subgraph).to_undirected()
    components = list(nx.connected_components(nx_graph))
    
    component_info = []
    for comp_nodes in components:
        comp_subgraph = nx_graph.subgraph(comp_nodes)
        
        # Calculate density score
        num_nodes = len(comp_nodes)
        num_edges = comp_subgraph.number_of_edges()
        max_possible_edges = (num_nodes * (num_nodes - 1)) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        component_info.append({
            'nodes': sorted(list(comp_nodes)),  # nodes in this component
            'score': density,                   # component density score
            'size': num_nodes                   # size of component
        })
    
    # Sort components by score (highest first)
    component_info.sort(key=lambda x: (x['score'], x['size']), reverse=True)
    
    return component_info
