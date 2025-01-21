import random
import numpy as np
import torch
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
    token_embeddings = []
    lpe_embeddings = []

    # Get actual node labels from the graph
    labels = graph.y

    # Assign labels to subgraphs
    subgraph_labels = assign_subgraph_labels(cluster_data)

    for i in range(num_parts):
        subgraph = cluster_data[i]
        # print(subgraph)

        
        # Compute GCN embeddings
        gcn_embeddings = compute_gcn_embeddings(subgraph, input_dim=1433, hidden_dim=64, output_dim=16)
        
        # Apply mean pooling to get subgraph-level embeddings
        token_embedding = torch.mean(gcn_embeddings, dim=0)
        token_embeddings.append(token_embedding)
        
        # Compute Laplacian positional embeddings
        lpe_embeddings.append(compute_laplacian_positional_embedding(subgraph, embedding_dim=16))

    token_embeddings = torch.stack(token_embeddings)
    print( "token_embeddings : ",token_embeddings.shape)
    lpe_embeddings = torch.stack(lpe_embeddings)
    print("lpe_embeddings : ",lpe_embeddings.shape)

    # Step 4: Create DataLoader
    dataset = TensorDataset(token_embeddings, lpe_embeddings, subgraph_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Step 5: Initialize and train the model
    input_dim = token_embeddings.shape[1] + lpe_embeddings.shape[1]  # Adjust input_dim
    model = GraphTransformer(input_dim=input_dim, embed_dim=16, num_heads=4, num_layers=2, num_classes=7)
    train_accuracy = train_model(model, dataloader, num_epochs=50, learning_rate=0.001)

    # Step 6: Evaluate the model
    print("\nEvaluating the model...")
    test_accuracy = evaluate_model(model, dataloader)

    # Print train and test accuracy
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()