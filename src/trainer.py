import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, F1, precision, and recall metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    return {
        'accuracy': 100. * accuracy,
        'f1_macro': 100. * f1_macro,
        'f1_micro': 100. * f1_micro,
        'precision_macro': 100. * precision_macro,
        'precision_micro': 100. * precision_micro,
        'recall_macro': 100. * recall_macro,
        'recall_micro': 100. * recall_micro
    }

def train_model(model, train_dataloader, val_dataloader, num_epochs=500, learning_rate=0.001):
    """Train model with validation-based model selection."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_f1 = 0.0
    best_model_state = None
    best_train_metrics = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_preds, train_labels = [], []
        train_loss = 0

        for batch in train_dataloader:
            subgraph_embeddings, lpe_embeddings, labels, num_nodes = [b.to(device) for b in batch]
            
            # Forward pass and loss calculation
            outputs = model(subgraph_embeddings, lpe_embeddings, num_nodes)
            repeated_labels = torch.repeat_interleave(labels, num_nodes)
            loss = F.cross_entropy(outputs, repeated_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store predictions and labels
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(repeated_labels.cpu().numpy())
            train_loss += loss.item()

        # Calculate metrics
        train_metrics = calculate_metrics(np.array(train_labels), np.array(train_preds))
        val_metrics = evaluate_model(model, val_dataloader)
        
        # Model selection based on validation F1
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            best_train_metrics = train_metrics

        # Logging
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}]')
            print(f'Train Loss: {train_loss/len(train_dataloader):.4f}')
            print('Training Metrics:')
            for metric, value in train_metrics.items():
                print(f'  {metric}: {value:.2f}%')
            print('Validation Metrics:')
            for metric, value in val_metrics.items():
                print(f'  {metric}: {value:.2f}%')

    # Restore best model
    model.load_state_dict(best_model_state)
    return best_train_metrics

def evaluate_model(model, dataloader):
    """Evaluate model performance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            subgraph_embeddings, lpe_embeddings, labels, num_nodes = [b.to(device) for b in batch]
            outputs = model(subgraph_embeddings, lpe_embeddings, num_nodes)
            repeated_labels = torch.repeat_interleave(labels, num_nodes)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(repeated_labels.cpu().numpy())

    return calculate_metrics(np.array(all_labels), np.array(all_preds))