from src.models.networks.graph_transformer_net import GraphTransformerNet
from src.models.networks.gcn_baseline import GCNBaseline

def GraphTransformer(net_params, subgraph_components):
    return GraphTransformerNet(net_params, subgraph_components)

def GCN2Layer(net_params, subgraph_components):
    # For the baseline, we ignore subgraph_components.
    return GCNBaseline(net_params)

def gnn_model(MODEL_NAME, net_params, subgraph_components):
    models = {
        'GraphTransformer': GraphTransformer,
        'GCNBaseline': GCN2Layer
    }
    return models[MODEL_NAME](net_params, subgraph_components)
