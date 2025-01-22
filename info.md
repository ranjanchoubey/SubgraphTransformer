## Summary

### Data Processing

- **Loading the Cora dataset**: The dataset is loaded using `torch_geometric.datasets.Planetoid`.
- **Partitioning the graph into subgraphs using METIS**: The graph is partitioned into subgraphs using `torch_geometric.loader.ClusterData`.

### Embedding Computation

- **Computing GCN embeddings for each subgraph**: GCN embeddings are computed for each subgraph using a GCN model.
- **Computing Laplacian positional embeddings for each subgraph**: Laplacian positional embeddings are computed for each subgraph.
- **Using mean pooling to get subgraph-level embeddings**: Mean pooling is applied to get subgraph-level embeddings.

### Model Architecture

- **Implementing a Graph Transformer model**: The model takes token embeddings and Laplacian positional embeddings as input.
- **Adding the token embeddings and Laplacian positional embeddings instead of concatenating them**: The embeddings are added together.

### Training and Evaluation

- **Training the Graph Transformer model using the computed embeddings**: The model is trained using the computed embeddings.
- **Evaluating the model on the test dataset**: The model is evaluated on the test dataset.

### NOTE: For each subgraph, we will get one prediction value; now we will populate this single prediction value to each of the nodes in the subgraph to calculate loss (ytrue - yhat) at the node level.

## Detailed Explanation

### Data Processing

1. **Loading the Cora Dataset**:
    - The Cora dataset is loaded using the `torch_geometric.datasets.Planetoid` class. This dataset contains citation networks where nodes represent documents and edges represent citations between documents. Each node has a feature vector and a label.

2. **Partitioning the Graph into Subgraphs using METIS**:
    - The graph is partitioned into subgraphs using the METIS algorithm, which is implemented in `torch_geometric.loader.ClusterData`. This algorithm partitions the graph into a specified number of subgraphs, ensuring that each subgraph has a balanced number of nodes and edges.

### Embedding Computation

1. **Computing GCN Embeddings for Each Subgraph**:
    - A Graph Convolutional Network (GCN) is used to compute embeddings for each node in the subgraph. The GCN model is defined in `src/gcn.py` and consists of two convolutional layers. The embeddings are computed by passing the node features and edge indices of the subgraph through the GCN model.

2. **Computing Laplacian Positional Embeddings for Each Subgraph**:
    - Laplacian positional embeddings are computed for each subgraph to capture the structural information of the graph. The Laplacian matrix is computed from the adjacency matrix of the subgraph, and its eigenvectors corresponding to the smallest eigenvalues are used as positional embeddings.

3. **Using Mean Pooling to Get Subgraph-Level Embeddings**:
    - Mean pooling is applied to the node embeddings to obtain a single embedding vector for the entire subgraph. This is done by averaging the embeddings of all nodes in the subgraph.

### Model Architecture

1. **Implementing a Graph Transformer Model**:
    - The Graph Transformer model is implemented in `src/transformer.py`. It takes token embeddings (GCN embeddings) and Laplacian positional embeddings as input. The model consists of an input projection layer, multiple transformer encoder layers, and a classification head.

2. **Adding the Token Embeddings and Laplacian Positional Embeddings Instead of Concatenating Them**:
    - The token embeddings and Laplacian positional embeddings are added together element-wise instead of concatenating them. This combined embedding is then passed through the transformer encoder layers.

### Training and Evaluation

1. **Training the Graph Transformer Model Using the Computed Embeddings**:
    - The model is trained using the computed embeddings. The training loop is implemented in `src/trainer.py`. For each batch of subgraphs, the model outputs a single prediction for each subgraph. This prediction is then propagated to each node in the subgraph, and the loss is calculated at the node level using the propagated predictions and the true node labels.

2. **Evaluating the Model on the Test Dataset**:
    - The model is evaluated on the test dataset using the same process as training. The evaluation loop is also implemented in `src/trainer.py`. The accuracy is calculated at the node level using the propagated predictions and the true node labels.

### Detailed Steps for Loss Calculation

1. **Model Forward Pass**:
    - The `GraphTransformer` model takes `token_embeddings` and `lpe_embeddings` as input.
    - It computes the subgraph-level embedding using mean pooling.
    - It then makes a single prediction for the subgraph.
    - This prediction is propagated to each node in the subgraph.

2. **Propagating Predictions**:
    - The subgraph-level prediction is repeated for each node in the subgraph.

3. **Loss Calculation**:
    - The loss is calculated using `nn.CrossEntropyLoss()`, which compares the propagated predictions with the true node labels.

### Code for Loss Calculation

Here is the relevant part of the training loop in `trainer.py`:

```python
# filepath: /ranjan/graphtransformer/my_project/src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    """
    Trains the GraphTransformer model.
    Args:
        model: The GraphTransformer model.
        dataloader: DataLoader for the training data.
        num_epochs: Number of epochs to train.
        learning_rate: Learning rate for the optimizer.
    Returns:
        train_accuracy: Training accuracy after the final epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            token_embeddings, lpe_embeddings, labels = batch

            # Move to GPU if available
            if torch.cuda.is_available():
                token_embeddings = token_embeddings.cuda()
                lpe_embeddings = lpe_embeddings.cuda()
                labels = labels.cuda()
                model = model.cuda()

            # Ensure input tensors have the correct shape
            if token_embeddings.dim() == 2:
                token_embeddings = token_embeddings.unsqueeze(0)
            if lpe_embeddings.dim() == 2:
                lpe_embeddings = lpe_embeddings.unsqueeze(0)

            # Forward pass
            outputs = model(token_embeddings, lpe_embeddings)  # Shape: [batch_size, num_nodes, num_classes]
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten to [batch_size * num_nodes, num_classes]
            labels = labels.view(-1)  # Flatten to [batch_size * num_nodes]
            loss = criterion(outputs, labels)  # Node-level loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {train_accuracy:.2f}%")

    return train_accuracy

def evaluate_model(model, dataloader):
    """
    Evaluates the model on the test dataset.
    Args:
        model: The trained Graph Transformer model.
        dataloader: DataLoader for test data.
    Returns:
        Accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            token_embeddings, lpe_embeddings, labels = batch

            if torch.cuda.is_available():
                token_embeddings = token_embeddings.cuda()
                lpe_embeddings = lpe_embeddings.cuda()
                labels = labels.cuda()
                model = model.cuda()

            # Ensure input tensors have the correct shape
            if token_embeddings.dim() == 2:
                token_embeddings = token_embeddings.unsqueeze(0)
            if lpe_embeddings.dim() == 2:
                lpe_embeddings = lpe_embeddings.unsqueeze(0)

            outputs = model(token_embeddings, lpe_embeddings)  # Shape: [batch_size, num_nodes, num_classes]
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten to [batch_size * num_nodes, num_classes]
            labels = labels.view(-1)  # Flatten to [batch_size * num_nodes]
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


