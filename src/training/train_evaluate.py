import torch
import torch.nn as nn
import math
import dgl
from src.utils.label_propagation import propagate_labels_with_components
from src.utils.metrics import  accuracy
from torch.utils.data import DataLoader
import dgl
import torch


def collate_graphs(batch):
    """Simply return the features tensor."""
    features, = batch  # Only one item since dataset length is 1
    return features

def expand_subgraph_predictions(subgraph_scores, node_counts):
    """
    Expands subgraph-level predictions to all nodes in the graph 
    Args:
        subgraph_scores: Subgraph-level predictions (shape: [num_subgraphs, num_classes]).
        node_counts: Number of nodes in each subgraph (shape: [num_subgraphs]).
        phase: Current phase ('train', 'test', or 'val').
        epoch: Current epoch number.
    
    Returns:
        node_prediction: Node-level predictions (shape: [num_nodes, num_classes]).
    """
    node_prediction = torch.repeat_interleave(subgraph_scores, node_counts, dim=0)
    return node_prediction




def train_epoch(model, optimizer, device, data_loader, epoch, train_mask, node_labels, node_counts, subgraphs=None, label_prop_config=None):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0

    for iter, batch_features in enumerate(data_loader):
        batch_features = batch_features.to(device)
        batch_labels = node_labels.to(device)

        optimizer.zero_grad()
        
        # Get logits from model
        subgraph_logits = model.forward(batch_features)
        
        # Use label propagation during training if enabled
        if label_prop_config and label_prop_config['enabled']:
            node_predictions = propagate_labels_with_components(
                subgraph_logits,
                subgraphs,
                label_prop_config
            )
        else:
            node_predictions = expand_subgraph_predictions(subgraph_logits, node_counts)
        
        # Compute loss
        loss = model.loss(node_predictions[train_mask], batch_labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc, _ = accuracy(node_predictions[train_mask], batch_labels[train_mask], phase="train", epoch=epoch)
        epoch_train_acc += acc
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data_loader, epoch, test_mask, node_labels, 
                    node_counts, phase, compareSubgraph=False, subgraphs=None, 
                    label_prop_config=None, subgraph_components=None):  # Added missing parameters
    """
    Evaluate network with optional label propagation
    """
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_labels = node_labels.to(device)

            # Get logits from model
            subgraph_logits = model.forward(batch_graphs)
            
            # Use label propagation if enabled
            if label_prop_config and label_prop_config['enabled']:
                from src.utils.label_propagation import propagate_labels_with_components
                node_predictions = propagate_labels_with_components(
                    subgraph_logits,
                    subgraphs,
                    label_prop_config
                )
            else:
                node_predictions = expand_subgraph_predictions(subgraph_logits, node_counts)
            
            if compareSubgraph:
                if label_prop_config and label_prop_config['enabled']:
                    from src.utils.label_propagation import propagate_labels_with_components
                    node_predictions = propagate_labels_with_components(
                        subgraph_logits,
                        subgraphs,
                        label_prop_config
                    )
                    return node_predictions, batch_labels
                else:
                    # Currently defaulting to this, which gives same label to all nodes
                    node_predictions = expand_subgraph_predictions(subgraph_logits, node_counts)
                    return node_predictions, batch_labels
            
            # Compute loss
            loss = model.loss(node_predictions[test_mask], batch_labels[test_mask])
            
            # Calculate accuracy
            acc, _ = accuracy(node_predictions[test_mask], batch_labels[test_mask], phase=phase, epoch=epoch)
            epoch_test_acc += acc
            epoch_test_loss += loss.detach().item()

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc