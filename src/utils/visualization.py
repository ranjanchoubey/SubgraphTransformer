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
    epochs = range(1, len(epoch_train_losses) + 1)
    
    # Create a figure with two subplots.
    plt.figure(figsize=(12, 5))
    
    # Plot Loss Curves.
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, epoch_train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, epoch_val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs. Validation Loss')
    ax1.legend()
    # Annotate the final loss values.
    ax1.text(epochs[-1], epoch_train_losses[-1],
             f'{epoch_train_losses[-1]:.4f}', fontsize=10, color='blue', ha='right', va='bottom')
    ax1.text(epochs[-1], epoch_val_losses[-1],
             f'{epoch_val_losses[-1]:.4f}', fontsize=10, color='red', ha='right', va='bottom')
    
    # Plot Accuracy Curves.
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, epoch_train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, epoch_val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train vs. Validation Accuracy')
    ax2.legend()
    # Annotate the final accuracy values (formatted as percentages).
    ax2.text(epochs[-1], epoch_train_accs[-1],
            f'{epoch_train_accs[-1]:.2f}%', fontsize=10, color='blue', ha='right', va='bottom')
    ax2.text(epochs[-1], epoch_val_accs[-1],
            f'{epoch_val_accs[-1]:.2f}%', fontsize=10, color='red', ha='right', va='bottom')


    
    plt.tight_layout()
    
    # Define the output directory and filename.
    out_dir = os.path.abspath(os.path.join(os.getcwd(), 'out'))
    os.makedirs(out_dir, exist_ok=True)
    
    # Use a constant filename to overwrite the same image each time.
    plot_save_path = os.path.join(out_dir, 'train_val_curves.png')
    
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
    plt.show()




def visualize_subgraph(node_prediction, node_labels, node_counts, subgraphs):
    """
    Visualize the subgraph comparison between predicted and actual labels.
    
    Args:
        node_prediction (torch.Tensor): Predicted labels for nodes in subgraphs 
            (shape: [num_nodes, num_classes]).
        node_labels (torch.Tensor): Actual labels for nodes in subgraphs (shape: [num_nodes]).
        node_counts (list or tensor): Number of nodes in each subgraph.
        subgraphs (list): List of DGL subgraphs.
    """
    # Get predicted classes for each node.
    prediction = torch.argmax(node_prediction, dim=1)
    labels = node_labels
    num_plots = min(10, len(subgraphs))  # visualize at most 10 subgraphs

    distinct_colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#a65628', '#756bb1', '#636363'
    ]
    num_colors = len(distinct_colors)
    
    # Define the directory to save visualizations.
    save_dir = os.path.abspath(os.path.join(os.getcwd(), 'out', 'visualize_subgraph'))
    os.makedirs(save_dir, exist_ok=True)
    
    # Flush the folder: remove all files and subdirectories in save_dir.
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    for i in range(num_plots):
        start_idx = sum(node_counts[:i].cpu().numpy())
        end_idx = start_idx + node_counts[i].cpu().numpy()
        
        subgraph_predictions = prediction[start_idx:end_idx]
        subgraph_true_labels = labels[start_idx:end_idx]
    
        plt.figure(figsize=(15, 6))
        subgraph = subgraphs[i]
        G = subgraph.to_networkx().to_undirected()
        pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50, seed=42)
    
        # Plot original labels.
        ax1 = plt.subplot(1, 2, 1)
        colors = [distinct_colors[int(lab) % num_colors] for lab in subgraph_true_labels.cpu().numpy()]
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='black', alpha=0.2, width=0.9)
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=100, node_color=colors)
        ax1.set_title('Original Labels')
        ax1.set_xticks([])
        ax1.set_yticks([])
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
    
        # Plot predicted labels.
        ax2 = plt.subplot(1, 2, 2)
        colors = [distinct_colors[int(lab) % num_colors] for lab in subgraph_predictions.cpu().numpy()]
        nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='black', alpha=0.2, width=0.9)
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=100, node_color=colors)
        ax2.set_title('Predicted Labels')
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
    
        # Compute the union of unique labels from both true and predicted values.
        unique_labels = torch.unique(torch.cat((subgraph_true_labels, subgraph_predictions)))
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=distinct_colors[int(label) % num_colors],
                       label=f'Class {int(label)}',
                       markersize=10)
            for label in unique_labels
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
        accuracy_val = (subgraph_predictions == subgraph_true_labels).float().mean().item()
        # Get the number of nodes in the subgraph. If node_counts[i] is a tensor, convert it to a scalar.
        num_nodes_in_subgraph = node_counts[i].item() if hasattr(node_counts[i], 'item') else node_counts[i]
        plt.suptitle(f'Subgraph {i} Comparison\nAccuracy: {accuracy_val:.2%} | Nodes: {num_nodes_in_subgraph}')
        plt.tight_layout()
    
        # Save the figure with a .png extension.
        save_path = os.path.join(save_dir, f'subgraph{i}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    print("\n✓ Subgraph Plotting Complete !!! \n")
    print(f"Visualizations saved in {save_dir}")
