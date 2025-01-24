# Graph Transformer with Subgraph-Level Attention

## Project Overview
This project aims to solve node classification in large graphs by reducing the quadratic attention complexity O(N²) to O(K²) where K is the number of subgraphs.

## Detailed Technical Specifications

### 1. Cora Dataset Details
- Total Nodes: 2708
- Edge Count: 5429
- Features Per Node: 1433 (sparse bag-of-words)
- Classes: 7 (different research paper topics)
- Class Distribution:
  - Training: ~140 nodes per class (~20%)
  - Validation: ~500 nodes (~18.5%)
  - Test: ~1000 nodes (~37%)
  - Unlabeled: Remaining nodes (~24.5%)

### 2. Graph Partitioning Process
- Algorithm: METIS
- Number of Partitions: 100
- Partition Characteristics:
  - Min Nodes Per Subgraph: ~15
  - Max Nodes Per Subgraph: ~40
  - Average: 27 nodes
  - Standard Deviation: ±8 nodes
- Edge Cut Minimization:
  - Internal Edges (preserved): ~80%
  - External Edges (cut): ~20%

### 3. Embedding Generation Pipeline

#### A. GCN Embeddings
- Input Layer: [N, 1433] → ReLU → [N, 64]
- Hidden Layer: [N, 64] → ReLU → [N, 32]
- Output Layer: [N, 32] → [N, 16]
- Architecture Details:
  - Layer Normalization after each conv
  - Dropout rate: 0.1
  - Skip connections

#### B. Laplacian Positional Encoding
1. Adjacency Matrix Construction:
   - Sparse format for efficiency
   - Self-loops added
2. Normalized Laplacian:
   - L = I - D^(-1/2)AD^(-1/2)
3. Eigenvector Computation:
   - Top-16 smallest eigenvalues
   - Output shape: [N, 16]

#### C. Subgraph Embedding Process
1. Node Feature Aggregation:
   - Mean pooling across nodes
   - Max pooling (alternative)
2. Dimension Retention:
   - Preserves 16-dim structure
   - Maintains spatial information

### 4. Transformer Architecture Details

#### A. Input Processing
1. Embedding Combination:
   ```
   Subgraph_Final = Subgraph_GCN + LPE
   [100, 16] = [100, 16] + [100, 16]
   ```

#### B. Multi-Head Attention
1. Attention Head Breakdown:
   - Number of Heads: 8
   - Dimension per Head: 2
   - Query/Key Size: 16
2. Attention Computation:
   ```
   Q = XWq, K = XWk, V = XWv
   Attention = softmax(QK^T/√d)V
   ```
3. Matrix Sizes:
   - Q, K, V: [100, 8, 2]
   - Attention Matrix: [8, 100, 100]
   - Output: [100, 16]

#### C. Feed-Forward Network
1. Layer Architecture:
   ```
   Linear[16→64] → GELU → Dropout(0.1) → Linear[64→16]
   ```
2. Layer Normalization:
   - Pre-norm architecture
   - Epsilon: 1e-5

### 5. Label Propagation Mechanism

#### A. Subgraph to Node Mapping
1. Forward Process:
   ```
   Subgraph Prediction [100, 7]
   → Node Mapping [2708, 7]
   Using repeat_interleave with num_nodes
   ```

2. Label Assignment:
   ```python
   for each subgraph i:
       nodes_in_subgraph = num_nodes[i]
       nodes[start:end] = subgraph_prediction[i]
   ```

### 6. Training Protocol

#### A. Optimization Details
- Optimizer: Adam
  - Learning Rate: 0.001
  - Betas: (0.9, 0.999)
  - Weight Decay: 1e-4
- Loss: CrossEntropy
  - Label Smoothing: 0.1

#### B. Training Schedule
- Epochs: 500
- Early Stopping:
  - Patience: 100
  - Delta: 1e-4
- Learning Rate Schedule:
  - Warm-up: 10 epochs
  - Decay: Cosine annealing

### 7. Memory Usage Analysis

#### A. Traditional Transformer
- Attention Matrix: 2708 × 2708 = 7,333,264 elements
- Memory Required: ~29.3MB (float32)

#### B. Our Approach
- Attention Matrix: 100 × 100 = 10,000 elements
- Memory Required: ~40KB (float32)
- Memory Reduction: 99.86%

### 8. Evaluation Metrics

#### A. Node-Level Metrics
- Accuracy
- Macro F1-Score
- Micro F1-Score
- Per-Class Precision/Recall

#### B. Subgraph-Level Analysis
- Label Consistency within Subgraphs
- Edge Cut Impact Analysis
- Cross-Subgraph Error Propagation

### 9. Implementation Notes

#### A. Device Management
- GPU Memory Usage: ~1.2GB
- CPU Memory Usage: ~4GB
- Mixed Precision Training
  - FP16 for attention computation
  - FP32 for gradients

#### B. Data Pipeline
- Lazy Loading
- Sparse Matrix Operations
- Efficient Memory Management
