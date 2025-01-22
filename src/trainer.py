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
            subgraph_embeddings, lpe_embeddings, labels = batch

            # Move to GPU if available
            if torch.cuda.is_available():
                subgraph_embeddings = subgraph_embeddings.cuda()
                lpe_embeddings = lpe_embeddings.cuda()
                labels = labels.cuda()
                model = model.cuda()

            # print(f"Subgraph Embeddings Shape: {subgraph_embeddings.shape}")
            # print(f"LPE Embeddings Shape: {lpe_embeddings.shape}")
            # print(f"Labels Shape: {labels.shape}")

            # Forward pass
            outputs = model(subgraph_embeddings, lpe_embeddings)  # Shape: [batch_size, num_classes]
            # print(f"Model Outputs Shape: {outputs.shape}")
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
            subgraph_embeddings, lpe_embeddings, labels = batch

            if torch.cuda.is_available():
                subgraph_embeddings = subgraph_embeddings.cuda()
                lpe_embeddings = lpe_embeddings.cuda()
                labels = labels.cuda()
                model = model.cuda()

            # print(f"Subgraph Embeddings Shape: {subgraph_embeddings.shape}")
            # print(f"LPE Embeddings Shape: {lpe_embeddings.shape}")
            # print(f"Labels Shape: {labels.shape}")

            outputs = model(subgraph_embeddings, lpe_embeddings)  # Shape: [batch_size, num_classes]
            # print(f"Model Outputs Shape: {outputs.shape}")
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy