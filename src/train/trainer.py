"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl


from src.train.metrics import accuracy_Cora as accuracy

def train_epoch(model, optimizer, device, data_loader, epoch,train_mask):
    
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    
    # for iter, (batch_graphs, batch_labels) in enumerate(data_loader):    

        # optimizer.zero_grad()

            
        # batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
    
    
    
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data_loader, epoch,test_mask):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    
    return epoch_test_loss, epoch_test_acc


























































# import torch
# import torch.nn.functional as F
# from src.utils.utils import calculate_metrics, print_metrics, calculate_masked_metrics
# import os
# import time
# from torch.utils.tensorboard import SummaryWriter
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# def train_model(*, model, subgraph_embeddings, lpe_embeddings, node_labels, node_counts, 
#                 train_mask, val_mask, node_indices, num_epochs=100, learning_rate=0.001):
#     device = next(model.parameters()).device
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
#     best_val_acc = 0
#     best_model_state = None
#     best_train_metrics = None
    
#     print(f"Starting training for {num_epochs} epochs...")
    
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
        
#         try:
#             # Forward pass
#             train_predictions, _ = model(subgraph_embeddings, lpe_embeddings, node_counts, node_indices)
            
#             # Get masked predictions and compute loss
#             train_node_predictions = train_predictions[train_mask]
#             train_node_labels = node_labels[train_mask]
            
#             loss = model.loss(train_node_predictions, train_node_labels)  # Use weighted loss

#             # Backward pass with gradient clipping
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             # Validation
#             with torch.no_grad():
#                 val_predictions, _ = model(subgraph_embeddings, lpe_embeddings, node_counts, node_indices)
#                 val_node_predictions = val_predictions[val_mask]
#                 val_node_labels = node_labels[val_mask]
#                 val_metrics = calculate_masked_metrics(val_node_predictions, val_node_labels)

#             print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f} Val Accuracy: {val_metrics["accuracy"]:.1f}%')

#             if val_metrics['accuracy'] > best_val_acc:
#                 best_val_acc = val_metrics['accuracy']
#                 best_model_state = model.state_dict().copy()
#                 print(f'New best validation accuracy: {best_val_acc:.1f}%')

#             # Update learning rate
#             scheduler.step(val_metrics['accuracy'])

#         except Exception as e:
#             print(f"\nError in training: {str(e)}")
#             raise e

#     print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.1f}%")
#     model.load_state_dict(best_model_state)
#     return {'val_acc': best_val_acc}

# def evaluate_model(*, model, subgraph_embeddings, lpe_embeddings, node_labels, node_counts, mask, node_indices):
#     """Evaluate model using mask."""
#     device = next(model.parameters()).device 
#     model.eval()
    
#     with torch.no_grad():
#         # Get predictions for all nodes
#         predictions, node_indices_out = model(subgraph_embeddings, lpe_embeddings, node_counts, node_indices)
        
#         # Ensure predictions are properly aligned with node indices
#         eval_node_predictions = predictions[mask]
#         eval_node_labels = node_labels[mask]
        
#         # Evaluation statistics
#         print("\nModel Evaluation:")
#         print("═══════════════")
#         print("├── Data Summary:")
#         print(f"│   ├── Total Nodes: {len(node_labels)}")
#         print(f"│   ├── Evaluated Nodes: {mask.sum().item()}")
#         print(f"│   └── Prediction Shape: {eval_node_predictions.shape}")
        
#         # Compute metrics
#         metrics = calculate_masked_metrics(eval_node_predictions, eval_node_labels)
        
#         print("└── Results:")
#         print(f"    └── Test Accuracy: {metrics['accuracy']:.2f}%")
        
#         return metrics
