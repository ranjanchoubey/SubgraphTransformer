import numpy as np
import networkx as nx
import torch
import os
from matplotlib import pyplot as plt
import shutil


def plot_train_val_curves(loss_data, output_path):
    plt.figure(figsize=(15, 5))
    
    # Plot total losses
    plt.subplot(1, 2, 1)
    train_line, = plt.plot(loss_data['train_loss'], label='Train Loss', color='blue')
    val_line, = plt.plot(loss_data['val_loss'], label='Validation Loss', color='red')
    
    # Annotate min/max points
    train_min_idx = np.argmin(loss_data['train_loss'])
    train_min = loss_data['train_loss'][train_min_idx]
    val_min_idx = np.argmin(loss_data['val_loss'])
    val_min = loss_data['val_loss'][val_min_idx]
    
    plt.annotate(f'Min: {train_min:.3f}', 
                xy=(train_min_idx, train_min),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'Min: {val_min:.3f}', 
                xy=(val_min_idx, val_min),
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot loss components
    plt.subplot(1, 2, 2)
    class_line, = plt.plot(loss_data['train_class_loss'], label='Classification Loss', color='green')
    reg_line, = plt.plot(loss_data['train_reg_loss'], label='Regularization Term', color='purple')
    
    # Annotate final values
    final_class = loss_data['train_class_loss'][-1]
    final_reg = loss_data['train_reg_loss'][-1]
    
    plt.annotate(f'Final: {final_class:.3f}', 
                xy=(len(loss_data['train_class_loss'])-1, final_class),
                xytext=(-10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'Final: {final_reg:.3f}', 
                xy=(len(loss_data['train_reg_loss'])-1, final_reg),
                xytext=(-10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='plum', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss Components')
    plt.legend()
    plt.title('Classification vs Regularization Loss')
    plt.grid(True, alpha=0.3)
    
    # Add overall title and adjust layout
    plt.suptitle('Training Loss Analysis', fontsize=14, y=1.05)
    plt.tight_layout()
    
    # Save with high DPI for better quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()






def visualize_node_predictions(node_logits, node_labels, node_counts, subgraphs):
    """
    Visualize prediction accuracy per component
    """
    # Move tensors to CPU before converting to numpy
    node_logits = node_logits.detach().cpu()
    node_labels = node_labels.cpu()
    node_counts = node_counts.cpu()
    
    # Calculate accuracies
    predictions = torch.argmax(node_logits, dim=1)
    accuracies = []
    
    start_idx = 0
    for count in node_counts:
        count = int(count)
        component_preds = predictions[start_idx:start_idx + count]
        component_labels = node_labels[start_idx:start_idx + count]
        acc = (component_preds == component_labels).float().mean().item()
        accuracies.append(acc)
        start_idx += count
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, acc in enumerate(accuracies):
        plt.scatter(i, acc, alpha=0.5, label=f'Component {i}')
    
    plt.xlabel('Component Index')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy per Component')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('component_accuracies.png')
    plt.close()
