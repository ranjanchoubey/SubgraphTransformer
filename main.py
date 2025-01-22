import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.data_processing import load_cora_data, partition_graph
from src.embedding import mean_pooling, compute_laplacian_positional_embedding, compute_gcn_embeddings
from src.transformer import GraphTransformer
from src.trainer import train_model, evaluate_model
from src.utils import assign_subgraph_labels
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Set the seed for reproducibility
    set_seed(42)

    # Step 1: Load the Cora dataset
    print("Loading Cora dataset...")
    graph = load_cora_data()
    print(f"Graph Info:\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}, Features: {graph.num_node_features}")

    # Step 2: Partition the graph into subgraphs
    num_parts = 100  # More partitions for larger training set
    cluster_data = partition_graph(graph, num_parts=num_parts)

    # Step 3: Compute embeddings and assign real labels
    subgraph_embeddings = []
    lpe_embeddings = []
    node_labels = []

    for i in range(num_parts):
        subgraph = cluster_data[i]
        print(f"Subgraph {i} - Number of nodes: {subgraph.num_nodes}")
        print(f"Subgraph {i} - Feature vector size: {subgraph.x.size(1)}")
        
        # Compute GCN embeddings
        gcn_embeddings = compute_gcn_embeddings(subgraph, input_dim=1433, hidden_dim=64, output_dim=16)
        
        # Compute Laplacian positional embeddings
        lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=16)
        
        # Compute subgraph-level embeddings using mean pooling
        subgraph_embedding = mean_pooling(gcn_embeddings)
        
        # Append subgraph-level embeddings and labels
        subgraph_embeddings.append(subgraph_embedding)
        lpe_embeddings.append(lpe.mean(dim=0))  # Mean pooling for LPE as well
        node_labels.append(subgraph.y)

        # Debugging prints to check tensor sizes
        print(f"Subgraph {i} - GCN Embeddings Size: {gcn_embeddings.size()}")
        print(f"Subgraph {i} - LPE Size: {lpe.size()}")
        print(f"Subgraph {i} - Subgraph Embedding Size: {subgraph_embedding.size()}")
        print(f"Subgraph {i} - Node Labels Size: {subgraph.y.size()}")

    subgraph_embeddings = torch.stack(subgraph_embeddings)
    lpe_embeddings = torch.stack(lpe_embeddings)
    node_labels = torch.cat(node_labels, dim=0)

    # Debugging prints to check tensor sizes
    print(f"Subgraph Embeddings Size: {subgraph_embeddings.size()}")
    print(f"LPE Embeddings Size: {lpe_embeddings.size()}")
    print(f"Node Labels Size: {node_labels.size()}")

    # Split data into training and testing sets
    train_indices, test_indices = train_test_split(np.arange(subgraph_embeddings.size(0)), test_size=0.2, random_state=42)
    train_dataset = TensorDataset(subgraph_embeddings[train_indices], lpe_embeddings[train_indices], node_labels[train_indices])
    test_dataset = TensorDataset(subgraph_embeddings[test_indices], lpe_embeddings[test_indices], node_labels[test_indices])

    batch_size = len(subgraph_embeddings)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 4: Initialize and train the model
    input_dim = 16  # Fixed embedding size
    model = GraphTransformer(input_dim=input_dim, embed_dim=16, num_heads=4, num_layers=2, ff_dim=64, dropout=0.1, num_classes=7)
    train_accuracy = train_model(model, train_dataloader, num_epochs=50, learning_rate=0.001)

    # Step 5: Evaluate the model
    print("\nEvaluating the model...")
    test_accuracy = evaluate_model(model, test_dataloader)

    # Print train and test accuracy
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()