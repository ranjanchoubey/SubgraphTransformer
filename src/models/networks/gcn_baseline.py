import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCNBaseline(nn.Module):
    def __init__(self, net_params):
        super(GCNBaseline, self).__init__()
        # in_dim will be set in main.py from the graph's features.
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']

        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(g, x)
        return x

    def loss(self, logits, labels, mask):
        # Cross-entropy loss computed only on masked nodes.
        return F.cross_entropy(logits[mask], labels[mask])
