"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
# from src.utils.utils import plot_subgraph_comparison
import torch
import torch.nn as nn
import math
import dgl
from src.train.metrics import  accuracy
from torch.utils.data import DataLoader
import dgl
import torch


def collate_graphs(batch):
    """Simply return the features tensor."""
    features, = batch  # Only one item since dataset length is 1
    return features

def expand_subgraph_predictions(subgraph_scores, node_counts,phase = None):
    """
    Expands subgraph-level predictions to all nodes in the graph.
    
    Args:
        subgraph_scores: Subgraph-level predictions (shape: [num_subgraphs, num_classes]).
        node_counts: Number of nodes in each subgraph (shape: [num_subgraphs]).
    
    Returns:
        node_prediction: Node-level predictions (shape: [num_nodes, num_classes]).
    """
    # Repeat subgraph predictions for each node in the subgraph
    # print("\nbefore populating subgraph prediction: ",subgraph_scores.shape)
    
    node_prediction = torch.repeat_interleave(subgraph_scores, node_counts, dim=0) # The `repeat_interleave` function in PyTorch is used to repeat elements
    # of a tensor along a specified dimension.
    # print("After populating subgraph prediction : ",node_prediction.shape)
    return node_prediction




def train_epoch(model, optimizer, device, data_loader, epoch, train_mask, node_labels, node_counts):
    """
    Trains the model for one epoch.
    
    Args:
        model: The model to train.
        optimizer: The optimizer for updating model parameters.
        device: The device (CPU/GPU) to use.
        data_loader: DataLoader for the training dataset.
        epoch: Current epoch number.
        train_mask: Mask for training nodes.
        node_labels: Labels for all nodes in the graph.
        node_counts: Number of nodes in each subgraph.
    
    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_train_acc: Average accuracy for the epoch.
        optimizer: The optimizer after updating parameters.
    """
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0

    # Since the dataset is a single graph, the data_loader will have only one batch
    for iter, batch_features in enumerate(data_loader):
        # Move data to the device
        batch_features = batch_features.to(device)
        batch_labels = node_labels.to(device)  # Labels for all nodes

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Get subgraph-level predictions
        subgraph_scores = model.forward(batch_features)  # Shape: [num_subgraphs, num_classes]

        # visulaize_subgraph_embedding(subgraph_scores,phase='During_Training')
        # Expand subgraph predictions to all nodes
        node_prediction = expand_subgraph_predictions(subgraph_scores, node_counts,phase = 'train')  # Shape: [num_nodes, num_classes]

        # print("\n node_prediction : ",node_prediction,node_prediction.shape)
        
        # Compute loss only for training nodes
        loss = model.loss(node_prediction[train_mask], batch_labels[train_mask])
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(node_prediction[train_mask], batch_labels[train_mask], phase="train", epoch=epoch)

    # Normalize loss and accuracy
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)  # Average accuracy across batches
    print()
    return epoch_loss, epoch_train_acc, optimizer

def  evaluate_network(model, device, data_loader, epoch,test_mask, node_labels,node_counts, phase="val", CompareSubgraphFlag = False):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            # batch_x = batch_graphs.ndata['feat'].to(device)  # Node features
            batch_labels = node_labels.to(device)  # Labels for all nodes

            # Forward pass: Get subgraph-level predictions
            batch_scores = model.forward(batch_graphs)  # Shape: [num_subgraphs, num_classes]
                    
            # Expand subgraph predictions to all nodes
            node_prediction = expand_subgraph_predictions(batch_scores, node_counts)  # Shape: [num_nodes, num_classes]
            # print("\n node_prediction : ",node_prediction.shape)
            # Compute loss only for training nodes
            loss = model.loss(node_prediction[test_mask], batch_labels[test_mask])
            
            epoch_test_loss += loss.detach().item()
            # Compute accuracy only for test nodes
            epoch_test_acc += accuracy(node_prediction[test_mask], batch_labels[test_mask], phase=phase, epoch=epoch)
            
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)  # Average accuracy across batches
        
        if CompareSubgraphFlag == True: # for plotting subgraph comparison
            return node_prediction[test_mask],batch_labels[test_mask]

    return epoch_test_loss, epoch_test_acc