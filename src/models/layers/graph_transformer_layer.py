import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
'''Dot Product Attention Score Calculation: The attention score is computed 
as the dot product between keys and queries:'''
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

'''Scaled Exponential: The scores are scaled and exponentiated for numerical stability:'''
def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    '''Initialization: Q, K, V: These are linear transformations of the input features
      to get the query, key, and value matrices.'''
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                
    def forward(self, g, h):
        # h: [K, d_h]
        Q = self.Q(h)  # [K, M*d_k]
        K = self.K(h)
        V = self.V(h)
        K_tokens = h.shape[0]  # number of subgraphs = K
        d_k = self.out_dim      # assume self.out_dim is d_k

        # Reshape to [K, M, d_k]
        Q = Q.view(K_tokens, self.num_heads, d_k)
        K = K.view(K_tokens, self.num_heads, d_k)
        V = V.view(K_tokens, self.num_heads, d_k)

        # Permute to [M, K, d_k] so that the attention is computed across subgraphs
        Q = Q.transpose(0, 1)  # [M, K, d_k]
        K = K.transpose(0, 1)  # [M, K, d_k]
        V = V.transpose(0, 1)  # [M, K, d_k]

        # Compute attention scores: shape [M, K, K]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        attention_probs = F.softmax(scores, dim=-1)  # [M, K, K]

        # Aggregate heads to form a final KxK attention matrix (for analysis or downstream use)
        effective_attention = attention_probs.mean(dim=0)  # [K, K]

        # Compute head outputs: shape [M, K, d_k]
        out_heads = torch.matmul(attention_probs, V)
        # Option: concatenate heads
        out = out_heads.transpose(0, 1).reshape(K_tokens, self.num_heads * d_k)
        # Alternatively, average over heads:
        # out = out_heads.mean(dim=0)  # [K, d_k]

        # Optionally, you can store effective_attention for later analysis:
        self.last_attention = effective_attention

        return out


'''Graph Transformer Layer: This layer combines multi-head attention with additional processing steps
 like feed-forward networks,layer normalization, and batch normalization.'''
class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    '''Initialization : This layer initializes components for multi-head attention,
      feed-forward networks, and normalization layers.'''
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h):
        h_in1 = h # for first residual connection
        
        '''Multi-Head Attention: The input features are processed through the multi-head attention layer.'''
        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)
        
        '''Dropout: A dropout layer is applied to prevent overfitting.'''
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        '''First Residual Connection: The output of the attention mechanism is added to the original input (residual connection).'''
        if self.residual:
            h = h_in1 + h # residual connection
        
        '''First Normalization: A layer normalization is applied to stabilize training. '''
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        '''Feed-Forward Network (FFN): A two-layer feed-forward network is applied:'''
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        '''Second Residual Connection: The output of the FFN is added to the input of the FFN.'''
        if self.residual:
            h = h_in2 + h # residual connection
        

        '''Final Normalization: Another round of normalization is applied.'''
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                            self.in_channels,
                                            self.out_channels, self.num_heads, self.residual)