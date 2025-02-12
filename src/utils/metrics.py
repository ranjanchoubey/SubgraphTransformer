import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate
import os

def save_prediction_analysis(true_classes,pred_classes, phase, epoch, out_dir="out/raw_predictions"):
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
                           

def accuracy(logits, labels, phase="train", epoch=None):
    """
    Computes accuracy from logits and labels
    Args:
        logits: Model logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        phase: Training phase ('train', 'val', or 'test')
        epoch: Current epoch number
    Returns:
        accuracy: Classification accuracy
        probs: Softmax probabilities for further use
    """
    # Move tensors to CPU before numpy conversion
    if torch.is_tensor(logits):
        logits = logits.detach().cpu()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu()

    probs = torch.softmax(logits, dim=1)
    preds = probs.max(1)[1]
    correct = preds.eq(labels)
    acc = correct.sum().item() / len(labels)
    
    # Return CPU tensors
    return acc, probs.cpu()

def get_predictions_from_logits(logits):
    """
    Convert logits to probabilities and predictions
    Args:
        logits: Raw model outputs [batch_size, num_classes]
    Returns:
        probs: Softmax probabilities [batch_size, num_classes]
        preds: Predicted class indices [batch_size]
    """
    probs = torch.softmax(logits, dim=1)
    preds = probs.max(1)[1]
    return probs, preds




















