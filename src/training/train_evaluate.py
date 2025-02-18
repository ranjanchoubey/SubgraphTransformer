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






def train_epoch(model, optimizer, device, data_loader, epoch, train_mask, node_labels, node_counts, subgraphs=None, subgraph_components=None, label_prop_config=None):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    n_iters = 0

    for iter, batch_features in enumerate(data_loader):
        batch_features = batch_features.to(device)
        batch_labels = node_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with components
        # batch_features shape: [80, 32] (num_subgraphs, input_dim)
        # subgraph_components: list of 80 lists, each containing component sizes
        # Example: [[5,3,2], [4,2], [6], ...] - varying number of components per subgraph
        logits = model.forward(batch_features, subgraph_components) # size [num_subgraphs, num_classes]
        # logits shape: [2708, 7] (total_nodes, num_classes)
        # where 2708 = sum of all component sizes across all subgraphs
        # and 7 = number of classes
        
        # When accessing with mask:
        # logits[train_mask] shape: [140, 7] (num_train_nodes, num_classes)
        # logits[val_mask] shape: [500, 7] (num_val_nodes, num_classes)
        # logits[test_mask] shape: [1000, 7] (num_test_nodes, num_classes)
        
        # Loss computation
        loss = model.loss(logits[train_mask], batch_labels[train_mask], subgraph_components)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy (use logits directly)
        acc, _ = accuracy(logits[train_mask], batch_labels[train_mask], phase="train", epoch=epoch)
        epoch_train_acc += acc
        epoch_loss += loss.detach().item()
        n_iters += 1

    return epoch_loss/n_iters, epoch_train_acc/n_iters, optimizer

def evaluate_network(model, device, data_loader, epoch, test_mask, node_labels, node_counts, subgraph_components, phase, compareSubgraph=False, subgraphs=None, label_prop_config=None):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_labels = node_labels.to(device)

            # Forward pass with components
            logits = model.forward(batch_graphs, subgraph_components) # size [num_subgraphs, num_classes]
            
            # Compute loss with components
            loss = model.loss(logits[test_mask], batch_labels[test_mask], subgraph_components)
            
            # Calculate accuracy
            acc, _ = accuracy(logits[test_mask], batch_labels[test_mask], phase=phase, epoch=epoch)
            epoch_test_acc += acc
            epoch_test_loss += loss.detach().item()

            if compareSubgraph:
                return logits, batch_labels

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc