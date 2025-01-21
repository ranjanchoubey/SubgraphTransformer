from src.data_processing import load_cora_data, partition_graph
from src.embedding import mean_pooling, compute_laplacian_positional_embedding
from src.transformer import GraphTransformer
from src.trainer import train_model, evaluate_model
from src.utils import assign_subgraph_labels
import torch
from torch.utils.data import DataLoader, TensorDataset

def main():
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
        token_embeddings.append(mean_pooling(subgraph))
        lpe_embeddings.append(compute_laplacian_positional_embedding(subgraph, embedding_dim=16))

    token_embeddings = torch.stack(token_embeddings)
    lpe_embeddings = torch.stack(lpe_embeddings)

    # Step 4: Create DataLoader
    dataset = TensorDataset(token_embeddings, lpe_embeddings, subgraph_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Step 5: Initialize and train the model
    input_dim = 1433
    model = GraphTransformer(input_dim=input_dim, embed_dim=32, num_heads=4, num_layers=2, num_classes=7)
    train_model(model, dataloader, num_epochs=10, learning_rate=0.001)

    # Step 6: Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, dataloader)

if __name__ == "__main__":
    main()
