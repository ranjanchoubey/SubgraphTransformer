import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate
import os

def save_prediction_analysis(true_classes,pred_classes, phase, epoch, out_dir="out/predictions"):
    """
        Helper function to save prediction analysis to file
    """
    if epoch==199:
        phase_dir = os.path.join(out_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)
        filepath = os.path.join(phase_dir, f"predictions_epoch_{epoch}.txt")
        with open(filepath, 'w') as f:
            for i in range(len(pred_classes)):
                f.write(f"Node {i}: True: {true_classes[i]}, Pred: {pred_classes[i]}\n")
                           

def accuracy(pred, labels, phase, epoch):
    """
    Compute classification accuracy and save prediction analysis to file.
    
    Args:
        pred: Model predictions/logits [num_nodes, num_classes]
        labels: Ground truth labels [num_nodes]
        phase: Current phase ('train', 'val', or 'test')
        epoch: Current epoch number
    Returns:
        accuracy: Classification accuracy as percentage (0-100)
    """
    with torch.no_grad():
        pred_classes = torch.argmax(pred, dim=1)

        # Move tensors to CPU and convert to numpy
        pred_classes = pred_classes.cpu().numpy()
        true_classes = labels.cpu().numpy()
        acc = accuracy_score(true_classes, pred_classes)*100.0
        
        save_prediction_analysis(true_classes,pred_classes, phase, epoch)
    return acc 




















