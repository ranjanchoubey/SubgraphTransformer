## Summary

### Data Processing

- Loading the Cora dataset.
- Partitioning the graph into subgraphs using METIS.

### Embedding Computation

- Computing GCN embeddings for each subgraph.
- Computing Laplacian positional embeddings for each subgraph.
- Using mean pooling to get subgraph-level embeddings.

### Model Architecture

- Implementing a Graph Transformer model that takes token embeddings and Laplacian positional embeddings as input.
- Adding the token embeddings and Laplacian positional embeddings instead of concatenating them.

### Training and Evaluation

- Training the Graph Transformer model using the computed embeddings.
- Evaluating the model on the test dataset.



### NOTE :  For each subgraph We will get one prediction value; Now We will populate  this single prediction value to each of the node in the subgraph to calculate loss (ytrue - yhat) ad node level


