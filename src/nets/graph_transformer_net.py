import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""
from src.layers.graph_transformer_layer import GraphTransformerLayer
from src.layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        print("===============")
        print("GraphTransformerNet init : net_params", net_params)
        print("===============")
        
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        # Input projection if needed
        self.input_proj = nn.Linear(16, hidden_dim)  # Project from GCN output dim to transformer hidden dim
        
        # Add layer normalization for input
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        
        # Add classifier normalization
        self.final_norm = nn.LayerNorm(out_dim)
        
        self.MLP_layer = MLPReadout(out_dim, n_classes)


    def forward(self, h):
        print("\n=== GraphTransformerNet Forward Pass ===")
        print(f"Input embedding size: {h.shape}")  # Should be [100, 16]
        
        # Project and normalize input
        h = self.input_proj(h)
        h = self.input_norm(h)
        print(f"After input projection: {h.shape}")  # Should be [100, 64]
        
        # Process through transformer layers
        for i, conv in enumerate(self.layers):
            print(f"\nTransformer Layer {i+1}:")
            h = conv(h)
        
        print("\nBefore MLP classification:")
        print(f"Feature size: {h.shape}")  # Should be [100, 64]
        
        # Normalize before classification
        h = self.final_norm(h)
        h_out = self.MLP_layer(h)
        print(f"Final output size: {h_out.shape}")  # Should be [100, 7] for 7 classes
        
        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss




