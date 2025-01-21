# src/trainer.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import os

def train_model(model, dataloader, num_epochs=10, learning_rate=0.001, checkpoint_dir="outputs/checkpoints"):
    """
    Training loop for the Graph Transformer.
    Args:
        model: The Graph Transformer model.
        dataloader: DataLoader for training data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        checkpoint_dir: Directory to save model checkpoints.
    """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
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

            # Forward pass
            outputs = model(token_embeddings, lpe_embeddings)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Compute epoch metrics
        accuracy = correct / total
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    print("Training completed!")

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

            outputs = model(token_embeddings, lpe_embeddings)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy






# uncomment and run the code for testing trainer.py
# from data_processing import load_cora_data, partition_graph
# from embedding import mean_pooling, compute_laplacian_positional_embedding
# from transformer import GraphTransformer
# from trainer import train_model, evaluate_model
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# def main():
#     # Step 1: Load the Cora dataset
#     print("Loading Cora dataset...")
#     graph = load_cora_data()
#     print(f"Graph Info:\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}, Features: {graph.num_node_features}")

#     # Step 2: Partition the graph into subgraphs
#     num_parts = 100  # More partitions for larger training set
#     cluster_data = partition_graph(graph, num_parts=num_parts)

#     # Step 3: Compute embeddings and labels for all subgraphs
#     token_embeddings = []
#     lpe_embeddings = []
#     labels = torch.randint(0, 7, (num_parts,))  # Simulated labels (replace with actual data)

#     for i in range(num_parts):
#         subgraph = cluster_data[i]
#         token_embeddings.append(mean_pooling(subgraph))
#         lpe_embeddings.append(compute_laplacian_positional_embedding(subgraph, embedding_dim=16))

#     token_embeddings = torch.stack(token_embeddings)
#     lpe_embeddings = torch.stack(lpe_embeddings)

#     # Step 4: Create DataLoader
#     dataset = TensorDataset(token_embeddings, lpe_embeddings, labels)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#     # Step 5: Initialize and train the model
#     input_dim = 1433
#     model = GraphTransformer(input_dim=input_dim, embed_dim=32, num_heads=4, num_layers=2, num_classes=7)
#     train_model(model, dataloader, num_epochs=5, learning_rate=0.001)

#     # Step 6: Evaluate the model
#     print("\nEvaluating the model...")
#     evaluate_model(model, dataloader)

# if __name__ == "__main__":
#     main()
