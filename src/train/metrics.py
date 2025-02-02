import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate
import os

def save_prediction_analysis(pred_classes, true_classes, phase, epoch, out_dir="out/predictions"):
    """Helper function to save prediction analysis to file"""
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{phase}_predictions_epoch_{epoch}.txt")
    
    with open(filepath, 'w') as f:
        # Header
        f.write(f"=== {phase} Prediction Analysis - Epoch {epoch} ===\n\n")
        
        # All predictions table
        f.write("Complete Prediction Analysis:\n")
        comparison = []
        for i in range(len(pred_classes)):
            is_correct = true_classes[i] == pred_classes[i]
            comparison.append([i, true_classes[i], pred_classes[i], is_correct])
            
            # Write in batches to avoid memory issues
            if len(comparison) >= 1000 or i == len(pred_classes) - 1:
                f.write(tabulate(comparison, 
                               headers=['Node', 'True Label', 'Predicted', 'Correct?'],
                               tablefmt='grid'))
                f.write('\n')
                comparison = []  # Clear the batch
        
        # Summary statistics
        f.write("\n\nClass Distribution:\n")
        unique, counts = np.unique(true_classes, return_counts=True)
        total_correct = 0
        total_nodes = len(true_classes)
        
        for cls, count in zip(unique, counts):
            correct = np.sum((true_classes == cls) & (pred_classes == cls))
            total_correct += correct
            f.write(f"Class {cls}: {correct}/{count} correct ({(correct/count)*100:.2f}%)\n")
            
        # Overall accuracy
        f.write(f"\nOverall Accuracy: {total_correct}/{total_nodes} "
                f"({(total_correct/total_nodes)*100:.2f}%)\n")

def accuracy(pred, labels, phase="train", epoch=0):
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
    print("\npred shape: ",pred[0],pred.shape)
    # Convert logits to predictions
    pred_classes = torch.argmax(pred, dim=1)
    
    # Move tensors to CPU and convert to numpy
    pred_classes = pred_classes.cpu().numpy()
    true_classes = labels.cpu().numpy()
    
    # print("\pred_classes : ",pred_classes,pred_classes.shape)
    # print("\ntrue_classes : ",true_classes,true_classes.shape)
    
    # Calculate accuracy
    acc = accuracy_score(true_classes, pred_classes)
    
    # Save complete analysis to file
    save_prediction_analysis(pred_classes, true_classes, phase, epoch)
    
    return acc * 100.0















