import torch
import dgl
import networkx as nx

def propagate_labels_with_components(logits, subgraphs, config):
    """
    Propagate labels using connected components
    Args:
        logits: Model predictions
        subgraphs: List of subgraphs
        config: Label propagation config containing:
            - enabled: bool
            - top_k: int (number of top labels to consider)
            - min_component_size: int
    """
    all_node_predictions = []
    
    for subgraph_idx, (subgraph, subgraph_logits) in enumerate(zip(subgraphs, logits)):
        # Get softmax probabilities
        probs = torch.softmax(subgraph_logits, dim=0)
        
        # Get components
        nx_graph = dgl.to_networkx(subgraph).to_undirected()
        components = list(nx.connected_components(nx_graph))
        
        # Use top_k from config
        values, indices = torch.sort(probs, descending=True)
        top_k_labels = indices[:config['top_k']]  # Use config's top_k
        
        # Initialize predictions
        num_nodes = subgraph.number_of_nodes()
        num_classes = len(probs)
        node_predictions = torch.zeros((num_nodes, num_classes), device=logits.device)
        
        # Assign labels to components
        for i, comp_nodes in enumerate(components):
            if len(comp_nodes) >= config['min_component_size']:
                label_idx = i % config['top_k']  # Cycle through top-k labels
                chosen_label = top_k_labels[label_idx]
                
                # Create one-hot prediction
                pred = torch.zeros_like(probs)
                pred[chosen_label] = 1.0
                
                # Assign to all nodes in this component
                for node_idx in comp_nodes:
                    node_predictions[node_idx] = pred
                    
        all_node_predictions.append(node_predictions)
        
    return torch.cat(all_node_predictions, dim=0)
