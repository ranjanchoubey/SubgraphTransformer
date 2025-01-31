"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl


from src.train.metrics import accuracy_Cora as accuracy

def train_epoch(model, optimizer, device, data_loader, epoch, train_mask, node_counts, graph=None):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    
    for iter, batch_graphs in enumerate(data_loader):
        print("\n=== Training Step ===")
        print(f"Input batch size: {batch_graphs.shape}")  # [num_subgraphs, embedding_dim]
        
        batch_graphs = batch_graphs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with embeddings directly
        batch_scores = model(batch_graphs)
        print(f"Model output (subgraph predictions): {batch_scores.shape}")  # [num_subgraphs, num_classes]
        
        # Map predictions back to nodes using node_counts
        node_predictions = map_subgraph_to_node_predictions(batch_scores, node_counts)
        print(f"Node-level predictions: {node_predictions.shape}")  # [total_num_nodes, num_classes]
        print(f"Train mask shape: {train_mask.shape}")  # [total_num_nodes]
        print(f"Masked predictions: {node_predictions[train_mask].shape}")  # [num_train_nodes, num_classes]
        
        # Calculate loss and accuracy using mapped predictions
        class_counts = torch.bincount(graph.ndata['label'][train_mask])
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        
        loss = nn.functional.cross_entropy(
            node_predictions[train_mask], 
            graph.ndata['label'][train_mask],
            weight=class_weights.to(device)
        )
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(node_predictions[train_mask], graph.ndata['label'][train_mask])
        
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data_loader, epoch, eval_mask, node_counts, graph=None):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            
            # Forward pass with embeddings
            batch_scores = model(batch_graphs)
            
            # Map predictions to nodes
            node_predictions = map_subgraph_to_node_predictions(batch_scores, node_counts)
            print("\n=== Evaluation Step ===")
            print(f"Node-level predictions (before mask): {node_predictions.shape}")  # [total_num_nodes, num_classes]
            
            # Apply mask
            masked_predictions = node_predictions[eval_mask]
            print(f"Node-level predictions (after mask): {masked_predictions.shape}")  # [num_eval_nodes, num_classes]
            
            # Get predicted classes and true labels
            predicted_classes = torch.argmax(masked_predictions, dim=1)
            print(f"Predicted classes: {predicted_classes.shape}")  # [num_eval_nodes]
            true_labels = graph.ndata['label'][eval_mask]
            
            # Print comparison for mismatched predictions
            print("\n=== Prediction Analysis ===")
            print("Format: Node_idx | Predicted -> True Label")
            mismatches = (predicted_classes != true_labels).nonzero().squeeze()
            
            for idx in mismatches[:130]:  # Show first 20 mismatches
                node_idx = eval_mask.nonzero()[idx]
                pred = predicted_classes[idx].item()
                true = true_labels[idx].item()
                print(f"Node {node_idx.item():4d} | Predicted: {pred} -> True: {true}")
            
            if len(mismatches) > 130:
                print(f"... and {len(mismatches)-20} more mismatches")
                
            print(f"\nAccuracy Stats:")
            print(f"Total nodes evaluated: {len(true_labels)}")
            print(f"Correct predictions: {(predicted_classes == true_labels).sum().item()}")
            print(f"Wrong predictions: {len(mismatches)}")
            
            # Calculate class-wise accuracy
            print("\nClass-wise Accuracy:")
            for class_idx in range(7):  # Cora has 7 classes
                class_mask = (true_labels == class_idx)
                if class_mask.sum() > 0:
                    class_acc = (predicted_classes[class_mask] == true_labels[class_mask]).float().mean()
                    print(f"Class {class_idx}: {class_acc:.4f}")
            
            loss = model.loss(masked_predictions, true_labels)
            epoch_test_loss += loss.item()
            epoch_test_acc += accuracy(masked_predictions, true_labels)
            
    epoch_test_loss /= (iter + 1)
    epoch_test_acc /= (iter + 1)
    
    return epoch_test_loss, epoch_test_acc

def map_subgraph_to_node_predictions(subgraph_predictions, node_counts):
    print("\n=== Node Mapping Debug ===")
    print(f"Subgraph predictions shape: {subgraph_predictions.shape}")  # [num_subgraphs, num_classes]
    print(f"Node counts shape: {node_counts.shape}")  # [num_subgraphs]
    
    """Maps subgraph predictions back to node-level predictions"""
    node_predictions = []
    start_idx = 0
    
    for i, count in enumerate(node_counts):
        count = count.item()
        print(f"Subgraph {i}: {count} nodes")
        # Repeat the subgraph prediction for each node in that subgraph
        repeated_pred = subgraph_predictions[i:i+1].repeat(count, 1)
        print(f"Repeated prediction shape: {repeated_pred.shape}")  # [num_nodes_in_subgraph, num_classes]
        node_predictions.append(repeated_pred)
        start_idx += count
        
    final_predictions = torch.cat(node_predictions, dim=0)
    print(f"Final node predictions shape: {final_predictions.shape}")  # [total_num_nodes, num_classes]
    print(f"Final node predictions shape: {final_predictions.shape}")  # [total_num_nodes, num_classes]
    return final_predictions


