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
    num_plots = min(20, len(subgraphs))  # visualize at most 10 subgraphs

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