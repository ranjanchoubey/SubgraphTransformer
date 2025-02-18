"""
    VIEWING MODEL CONFIG AND PARAMS
"""
import numpy as np
from src.models.networks.load_net import gnn_model


def view_model_param(MODEL_NAME, net_params,subgraph_components):
    model = gnn_model(MODEL_NAME, net_params,subgraph_components)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    # print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param