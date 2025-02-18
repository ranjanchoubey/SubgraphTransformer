"""
    Utility file to select GraphNN model as
    selected by the user
"""

from src.models.networks.graph_transformer_net import GraphTransformerNet

def GraphTransformer(net_params,subgraph_components):
    return GraphTransformerNet(net_params,subgraph_components)

def gnn_model(MODEL_NAME, net_params,subgraph_components):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params,subgraph_components)