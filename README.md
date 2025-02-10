# SubgraphTransformer

A novel approach to graph representation learning that combines graph partitioning with transformer-based architectures for enhanced node classification.

## Overview

SubgraphTransformer introduces a novel architecture that processes large graphs by:
1. Partitioning them into meaningful subgraphs
2. Computing rich subgraph embeddings
3. Using transformer layers to model subgraph interactions
4. Propagating subgraph predictions back to individual nodes

This approach offers several advantages:
- Reduces computational complexity for large graphs
- Captures both local and global graph structure
- Leverages modern transformer architectures for graph data
- Maintains interpretability through explicit subgraph representations

## Architecture Details

### 1. Graph Partitioning
The system begins by partitioning the input graph into smaller subgraphs using METIS:
- Balances subgraph sizes
- Minimizes edge cuts between partitions
- Creates manageable units for processing

### 2. Feature Generation
For each subgraph, we compute two types of embeddings:
- **GCN Embeddings**: Capture node features and local topology
  - Uses a 2-layer GraphConv network
  - Aggregates neighborhood information
  - Produces learned representations

- **Laplacian Positional Embeddings**: Encode structural information
  - Based on the graph Laplacian eigendecomposition
  - Captures topological position within subgraph
  - Provides critical structural context

### 3. Transformer Architecture
The core transformer architecture consists of:

#### Multi-Head Attention
- Processes subgraph-to-subgraph relationships
- Computes attention scores between all subgraph pairs
- Uses multiple attention heads for diverse relationship capture
```python
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.out_dim)
attention_probs = F.softmax(attention_scores, dim=-1)
out = torch.matmul(attention_probs, V)
```

#### Transformer Layers
Each transformer layer includes:
- Multi-head attention mechanism
- Feed-forward neural networks
- Layer normalization
- Residual connections
- Dropout for regularization

### 4. Node Classification
The final prediction process:
1. Transformer outputs subgraph-level predictions
2. Predictions are expanded to all nodes within each subgraph
3. Loss is computed only for labeled nodes
4. Cross-entropy loss with class weight balancing

## Implementation Details

### Key Components

#### 1. Training Pipeline
- Handles the main training loop
- Manages model optimization
- Tracks metrics and checkpoints
- Implements early stopping

#### 2. Graph Transformer Layer
```python
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, 
                 layer_norm=False, batch_norm=True, residual=True):
        # Initialize attention, normalization, and FFN components
        # ...

    def forward(self, g, h):
        # Multi-head attention
        # Normalization
        # Feed-forward network
        # Residual connections
```

#### 3. Feature Processing
- GCN-based feature extraction
- Laplacian positional encoding
- Mean pooling for subgraph embeddings

### Configuration System

The system uses a flexible JSON configuration system:
```json
{
    "gpu": {"use": true, "id": 0},
    "model": "GraphTransformer",
    "dataset": "CoraFull",
    "params": {
        "epochs": 800,
        "batch_size": 1,
        "init_lr": 0.0007
        // ...
    },
    "net_params": {
        "L": 4,
        "n_heads": 8,
        "hidden_dim": 512
        // ...
    }
}
```

## Usage

1. **Installation**
```bash
git clone https://github.com/username/SubgraphTransformer.git
cd SubgraphTransformer
pip install -r requirements.txt
```

2. **Running Training**
```bash
python main.py --config configs/cora_full.json
```

3. **Configuration**
- Modify `configs/cora_full.json` for different settings
- Key parameters:
  - `num_parts`: Number of subgraphs
  - `hidden_dim`: Model capacity
  - `n_heads`: Number of attention heads
  - `L`: Number of transformer layers

## Performance Optimization

### Memory Efficiency
- Batch processing of subgraphs
- Efficient attention computation
- Smart memory management for large graphs

### Training Speed
- GPU acceleration
- Optimized data loading
- Parallel subgraph processing

### Model Architecture
- Residual connections for gradient flow
- Layer normalization for stability
- Dropout for regularization

## Results Visualization

The system includes visualization tools for:
- Training/validation curves
- Attention patterns
- Subgraph predictions
- Node classification results

## Future Directions

Potential improvements and extensions:
1. Dynamic partitioning strategies
2. Enhanced positional encodings
3. Hierarchical attention mechanisms
4. Edge feature incorporation
5. Self-supervised pretraining

## Technical Requirements

- Python 3.7+
- PyTorch 1.7+
- DGL 0.6+
- CUDA-capable GPU (recommended)

## Citation

If you use this code in your research, please cite:
```bibtex
@article{subgraphtransformer2023,
  title={SubgraphTransformer: A Partition-based Approach to Graph Classification},
  author={[Authors]},
  journal={[Journal]},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
