import numpy as np
import networkx as nx
import torch
import os
from matplotlib import pyplot as plt
import shutil  # For removing directories


import os
import matplotlib.pyplot as plt

def plot_train_val_curves(epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs):
    """
    Plot the training and validation loss and accuracy curves,
    and annotate the final loss and accuracy values on the plots.
    
    Args:
        epoch_train_losses (list): List of training losses for each epoch.
        epoch_val_losses (list): List of validation losses for each epoch.
        epoch_train_accs (list): List of training accuracies for each epoch.
        epoch_val_accs (list): List of validation accuracies for each epoch.
    """
    # Ensure all tensors are on CPU and converted to numpy
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return [to_numpy(i) for i in x]
        return x
    
    train_losses = to_numpy(epoch_train_losses)
    val_losses = to_numpy(epoch_val_losses)
    train_accs = to_numpy(epoch_train_accs)
    val_accs = to_numpy(epoch_val_accs)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with two subplots.
    plt.figure(figsize=(12, 5))
    
    # Plot Loss Curves.
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs. Validation Loss')
    ax1.legend()
    # Annotate the final loss values.
    ax1.text(epochs[-1], train_losses[-1],
             f'{train_losses[-1]:.4f}', fontsize=10, color='blue', ha='right', va='bottom')
    ax1.text(epochs[-1], val_losses[-1],
             f'{val_losses[-1]:.4f}', fontsize=10, color='red', ha='right', va='bottom')
    
    # Plot Accuracy Curves.
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train vs. Validation Accuracy')
    ax2.legend()
    # Annotate the final accuracy values (formatted as percentages).
    ax2.text(epochs[-1], train_accs[-1],
            f'{train_accs[-1]:.2f}%', fontsize=10, color='blue', ha='right', va='bottom')
    ax2.text(epochs[-1], val_accs[-1],
            f'{val_accs[-1]:.2f}%', fontsize=10, color='red', ha='right', va='bottom')


    
    plt.tight_layout()
    
    # Define the output directory and filename.
    out_dir = os.path.abspath(os.path.join(os.getcwd(), 'out'))
    os.makedirs(out_dir, exist_ok=True)
    
    # Use a constant filename to overwrite the same image each time.
    plot_save_path = os.path.join(out_dir, 'train_val_curves.png')
    
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
    plt.show()




import matplotlib.pyplot as plt
import torch
import numpy as np

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
