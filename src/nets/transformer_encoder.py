import torch
import torch.nn as nn
from src.layers.transformer_encoder_layer import TransformerEncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1,
                 layer_norm=True, batch_norm=True, residual=True):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                residual=residual
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x