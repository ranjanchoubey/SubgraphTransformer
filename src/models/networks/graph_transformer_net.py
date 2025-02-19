import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Graph Transformer
    
"""
from src.models.layers.graph_transformer_layer import GraphTransformerLayer
from src.models.layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params,subgraph_components=None):
        super().__init__()
        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 100 
        self.num_subgraph =  net_params['num_subgraph']

        
        
        ######### Regularization :  Component handling parameters #########
        self.reg_lambda = net_params.get('reg_lambda')  # Reduce from 0.01 to 0.0001
        self.reg_enabled = True


        self.list_weight =  nn.ParameterList()
        for i in range(self.num_subgraph):
            self.component_weights = nn.Parameter(torch.ones(len(subgraph_components[i]), device=self.device))/len(subgraph_components[i])
            # print(f"Component weights for subgraph {i}: {self.component_weights}")
            self.list_weight.append(self.component_weights)




        self.input_proj = nn.Linear(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        



    def forward(self, features,subgraph_components):
        h = self.input_proj(features)  # [num_subgraphs, hidden_dim]
        h = self.in_feat_dropout(h)
        for layer in self.layers:
            h = layer(None, h)
        logits = self.MLP_layer(h)  # [num_subgraphs, num_classes]

        return self.propagate_labels(logits, subgraph_components)  # [total_nodes, num_classes]


    def propagate_labels(self, subgraph_logits, subgraph_components):
        predictions = []

        for i in range(len(subgraph_components)):

            print("\n list_weight : ",self.list_weight[i])

            for j in range(len(subgraph_components[i])):
                comp_size = int(subgraph_components[i][j])
                comp_pred = subgraph_logits[i] * self.list_weight[i][j]
                node_preds = comp_pred.repeat(comp_size, 1)
                predictions.append(node_preds)

        result = torch.cat(predictions, dim=0) # size [total_nodes, num_classes]
        return result


    def compute_reg_loss(self, subgraph_components): 
        reg_loss = 0.0

        for i in range(self.num_subgraph):
            # Convert to tensor and normalize
            components_tensor = torch.tensor(subgraph_components[i], 
                                          dtype=torch.float32, 
                                          device=self.device)
            # Compute dot product with weights
            product = torch.dot(self.list_weight[i], components_tensor)
            
            # Use quadratic penalty for better stability
            reg_loss += product - 1.0
            
            # # Optional: Add debugging information
            # if self.training:
            #     print(f"\nSubgraph {i} statistics:")
            #     print(f"Raw components: {components_tensor}")
            #     print(f"Weights: {self.list_weight[i]}")
            #     print(f"Product: {product:.4f}")
            #     print(f"Reg loss term: {torch.square(product - 1.0):.4f}")

        return reg_loss
    

    def loss(self, pred, label, subgraph_components=None):
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        criterion = nn.CrossEntropyLoss(weight=weight)
        base_loss = criterion(pred, label)

        reg_loss = self.compute_reg_loss(subgraph_components)
        reg_term = self.reg_lambda * reg_loss
        total_loss = base_loss + reg_term
        
        return total_loss, base_loss.item(), reg_term.item()




