
# Detailed Architecture Documentation

## 1. Data Processing Pipeline

### Graph Partitioning
The partitioning process is handled by `src/data/partitioning.py`:
- Uses METIS algorithm through DGL
- Ensures balanced partition sizes (±5% nodes)
- Minimizes edge cuts to preserve local structure
- Each partition becomes a subgraph for independent processing

```python
def partition_graph(graph, num_parts=4):
    # METIS partitioning creates roughly equal-sized subgraphs
    partition_labels = dgl.metis_partition_assignment(graph, num_parts)
    # Each subgraph maintains original node features and structure
    subgraphs = [dgl.node_subgraph(graph, mask) for mask in partition_masks]
```

### Feature Engineering

The embedding generation (`src/data/embedding.py`) involves:

1. **GCN Embeddings**:
   - 2-layer GraphConv network
   - Layer 1: Input → Hidden (ReLU activation)
   - Layer 2: Hidden → Output
   - Captures local topology and features
   - Parameters:
     - Input dim: Original node features
     - Hidden dim: Usually 2-4x input
     - Output dim: Final embedding size

2. **Laplacian Positional Encoding**:
   - Computes normalized graph Laplacian
   - Extracts top-k eigenvectors
   - Provides structural node positions
   - Helps distinguish similar subgraphs
   ```python
   L = D - A  # Laplacian matrix
   eigvals, eigvecs = eigh(L)
   lpe = eigvecs[:, :embedding_dim]
   ```

3. **Mean Pooling**:
   - Aggregates node embeddings per subgraph
   - Creates fixed-size subgraph representations
   - Maintains permutation invariance

## 2. Transformer Architecture

### Multi-Head Attention (`src/models/layers/graph_transformer_layer.py`)

1. **Query, Key, Value Projections**:
   ```python
   Q = self.Q(h)  # [num_subgraphs, num_heads * head_dim]
   K = self.K(h)  # [num_subgraphs, num_heads * head_dim]
   V = self.V(h)  # [num_subgraphs, num_heads * head_dim]
   ```

2. **Attention Computation**:
   - Scaled dot-product attention per head
   - Parallel computation across heads
   - Softmax normalization for attention weights
   ```python
   attention = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(head_dim)
   attention = F.softmax(attention, dim=-1)
   output = torch.matmul(attention, V)
   ```

3. **Multi-Head Aggregation**:
   - Concatenates outputs from all heads
   - Projects to final output dimension
   - Applies dropout for regularization

### Transformer Layer Components

1. **Layer Architecture**:
   ```
   Input
     ↓
   Multi-Head Attention
     ↓
   Add & Normalize
     ↓
   Feed Forward
     ↓
   Add & Normalize
     ↓
   Output
   ```

2. **Feed-Forward Network**:
   - Two linear transformations
   - ReLU activation in between
   - Dimension expansion and contraction
   ```python
   h = self.FFN_layer1(h)  # Expand
   h = F.relu(h)
   h = self.FFN_layer2(h)  # Contract
   ```

3. **Normalization Layers**:
   - Layer normalization after each block
   - Batch normalization option
   - Helps with training stability

## 3. Training Process

### Loss Computation

1. **Node Label Expansion**:
   ```python
   def expand_subgraph_predictions(subgraph_scores, node_counts):
       # Expands subgraph predictions to node level
       node_prediction = torch.repeat_interleave(
           subgraph_scores, 
           node_counts, 
           dim=0
       )
       return node_prediction
   ```

2. **Weighted Cross-Entropy**:
   - Handles class imbalance
   - Weights based on class frequencies
   - Only computed for labeled nodes

### Training Pipeline

1. **Optimization**:
   - Adam optimizer
   - Learning rate scheduling
   - Weight decay regularization
   - Early stopping on validation loss

2. **Batch Processing**:
   - Single graph as one batch
   - Forward pass through transformer
   - Backward pass with accumulated gradients
   - Parameter updates

3. **Monitoring**:
   - Training/validation loss curves
   - Accuracy metrics per epoch
   - Learning rate adjustments
   - Model checkpointing

## 4. Performance Considerations

### Memory Management

1. **Efficient Implementations**:
   - Sparse matrix operations
   - In-place operations where possible
   - GPU memory optimization
   ```python
   # Example of memory-efficient attention
   attention = torch.baddbmm(
       alpha=1.0 / sqrt(head_dim),
       batch1=Q,
       batch2=K.transpose(-2, -1)
   )
   ```

2. **Batch Processing**:
   - Dynamic batch sizing
   - Gradient accumulation
   - Memory-aware scheduling

### Computational Optimization

1. **Parallel Processing**:
   - Multi-GPU support
   - Distributed training options
   - Batch parallelization

2. **Caching Strategies**:
   - Feature preprocessing
   - Partition information
   - Intermediate computations

This architecture provides a scalable and efficient approach to graph learning by combining the benefits of graph partitioning with transformer-based processing.
