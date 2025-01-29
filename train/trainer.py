import torch
import torch.nn.functional as F
from utils.utils import calculate_metrics, print_metrics, calculate_masked_metrics
from tqdm import tqdm

def train_model(*, model, subgraph_embeddings, lpe_embeddings, node_labels, node_counts, 
                train_mask, val_mask, node_indices, num_epochs=100, learning_rate=0.001):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0
    best_model_state = None
    best_train_metrics = None

    # Standard progress bar format
    progress_bar = tqdm(
        range(num_epochs),
        desc='Training',
        ncols=80,
        leave=True,
        bar_format='{desc} {n_fmt}/{total_fmt}: {percentage:3.0f}%|{bar}| [{postfix}]'
    )

    for epoch in progress_bar:
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        try:
            # Get predictions for all nodes
            train_predictions, _ = model(subgraph_embeddings, lpe_embeddings, node_counts, node_indices)
            
            # Get training predictions and loss
            train_node_predictions = train_predictions[train_mask]
            train_node_labels = node_labels[train_mask]
            loss = F.cross_entropy(train_node_predictions, train_node_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                train_metrics = calculate_masked_metrics(train_node_predictions, train_node_labels)
                val_predictions, _ = model(subgraph_embeddings, lpe_embeddings, node_counts, node_indices)
                val_node_predictions = val_predictions[val_mask]
                val_node_labels = node_labels[val_mask]
                val_metrics = calculate_masked_metrics(val_node_predictions, val_node_labels)

            # Update progress bar with standard format
            progress_bar.set_postfix_str(
                f"loss: {loss.item():.4f}, val_acc: {val_metrics['accuracy']:.1f}%"
            )

            # Save best model based on validation accuracy
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
                best_train_metrics = train_metrics

        except Exception as e:
            print(f"\nError in train_model: {str(e)}")
            print(f"Error location: {e.__traceback__.tb_lineno}")
            raise e

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return best_train_metrics

def evaluate_model(*, model, subgraph_embeddings, lpe_embeddings, node_labels, node_counts, mask, node_indices):
    """Evaluate model using mask."""
    device = next(model.parameters()).device 
    model.eval()
    
    with torch.no_grad():
        # Get predictions for all nodes
        predictions, node_indices_out = model(subgraph_embeddings, lpe_embeddings, node_counts, node_indices)
        
        # Ensure predictions are properly aligned with node indices
        eval_node_predictions = predictions[mask]
        eval_node_labels = node_labels[mask]
        
        # Evaluation statistics
        print("\nModel Evaluation:")
        print("═══════════════")
        print("├── Data Summary:")
        print(f"│   ├── Total Nodes: {len(node_labels)}")
        print(f"│   ├── Evaluated Nodes: {mask.sum().item()}")
        print(f"│   └── Prediction Shape: {eval_node_predictions.shape}")
        
        # Compute metrics
        metrics = calculate_masked_metrics(eval_node_predictions, eval_node_labels)
        
        print("└── Results:")
        print(f"    └── Test Accuracy: {metrics['accuracy']:.2f}%")
        
        return metrics
        