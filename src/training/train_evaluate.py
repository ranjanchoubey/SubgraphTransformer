import torch
import torch.nn as nn
import math
import dgl
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




def train_epoch(model, optimizer, device, data_loader, epoch, train_mask, node_labels, node_counts):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0

    for iter, batch_features in enumerate(data_loader):
        batch_features = batch_features.to(device)
        batch_labels = node_labels.to(device)

        optimizer.zero_grad()
        
        # Get logits from model
        subgraph_logits = model.forward(batch_features)
        
        # Expand logits to node level
        node_logits = expand_subgraph_predictions(subgraph_logits, node_counts)
        
        # Compute loss using logits
        loss = model.loss(node_logits[train_mask], batch_labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Get accuracy and probabilities
        acc, probs = accuracy(node_logits[train_mask], batch_labels[train_mask], phase="train", epoch=epoch)
        epoch_train_acc += acc
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data_loader, epoch, test_mask, node_labels, node_counts, phase, compareSubgraph=False):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_labels = node_labels.to(device)

            # Get logits from model
            subgraph_logits = model.forward(batch_graphs)
            
            # Expand logits to node level
            node_logits = expand_subgraph_predictions(subgraph_logits, node_counts)
            
            if compareSubgraph:
                # Return logits for visualization/comparison
                return node_logits, batch_labels
            
            # Compute loss using logits
            loss = model.loss(node_logits[test_mask], batch_labels[test_mask])
            
            # Get accuracy and probabilities
            acc, probs = accuracy(node_logits[test_mask], batch_labels[test_mask], phase=phase, epoch=epoch)
            epoch_test_acc += acc
            epoch_test_loss += loss.detach().item()

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc