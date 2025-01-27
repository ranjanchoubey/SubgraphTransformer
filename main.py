import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from src.data_processing import load_cora_data, partition_graph
from src.embedding import mean_pooling, compute_laplacian_positional_embedding, compute_gcn_embeddings
from src.transformer import GraphTransformer
from src.trainer import train_model, evaluate_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NodeLevelDataset(Dataset):
    """
    Custom dataset for handling subgraph-based node classification.
    
    Key Features:
    - Maintains mapping between subgraphs and nodes
    - Handles mask-based data splitting
    - Computes subgraph labels from node labels
    
    Process:
    1. Takes subgraph embeddings and node labels
    2. Computes subgraph labels using mode of node labels
    3. Maintains mapping for label propagation
    4. Handles masked node selection for train/val/test splits
    """
    def __init__(self, subgraph_embeddings, lpe_embeddings, node_labels, num_nodes_list, mask):
        self.subgraph_embeddings = subgraph_embeddings
        self.lpe_embeddings = lpe_embeddings
        self.node_labels = node_labels
        self.num_nodes_list = num_nodes_list
        self.mask = mask

    def __len__(self):
        return len(self.subgraph_embeddings)

    def __getitem__(self, idx):
        return (
            self.subgraph_embeddings[idx],
            self.lpe_embeddings[idx],

            self.node_labels[idx],  # Return original node labels
            self.num_nodes_list[idx]
        )

def custom_collate_fn(batch):
    """
    Custom collation function for batching subgraph data.
    
    Features:
    - Handles variable-sized subgraphs
    - Stacks embeddings and labels properly
    - Maintains node count information
    
    Process:
    1. Collects batch elements
    2. Stacks embeddings and labels
    3. Converts node counts to tensor
    """
    subgraph_embeddings, lpe_embeddings, labels, num_nodes = zip(*batch)
    subgraph_embeddings = torch.stack(subgraph_embeddings)
    lpe_embeddings = torch.stack(lpe_embeddings)
    labels = torch.stack(labels)  # Changed to handle subgraph-level labels
    num_nodes = torch.tensor(num_nodes)
    return subgraph_embeddings, lpe_embeddings, labels, num_nodes

def main():
    """
    Main execution function implementing the complete pipeline.
    
    Pipeline Steps:
    1. Data Loading and Preprocessing:
       - Loads Cora dataset
       - Partitions graph into subgraphs
       - Computes embeddings and masks
    
    2. Dataset Creation:
       - Creates train/val/test datasets
       - Handles mask-based splitting
       - Sets up data loaders
    
    3. Model Training:
       - Initializes GraphTransformer
       - Trains with validation monitoring
       - Tracks multiple metrics
    
    4. Evaluation:
       - Evaluates on validation set
       - Performs final test set evaluation
       - Reports comprehensive metrics
    """
    # Set the seed for reproducibility
    set_seed(42)

    # Step 1: Load the Cora dataset
    print("Loading Cora dataset...")
    graph = load_cora_data()
    print(f"Graph Info:\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}, Features: {graph.num_node_features}")

    # Step 2: Partition the graph into subgraphs
    num_parts = 100  # More partitions for larger training set
    cluster_data = partition_graph(graph, num_parts=num_parts)

    # Step 3: Compute embeddings and track mask information
    subgraph_embeddings = []
    lpe_embeddings = []
    node_labels = []
    num_nodes_list = []
    train_masks = []
    val_masks = []
    test_masks = []

    for i in range(num_parts):
        subgraph = cluster_data[i]
        print("="*100)
        print(f"Subgraph {i} - Number of nodes: {subgraph.num_nodes}")
        print(f"Subgraph {i} - Feature vector size: {subgraph.x.size(1)}")
        
        # Compute GCN embeddings
        gcn_embeddings = compute_gcn_embeddings(subgraph, input_dim=1433, hidden_dim=64, output_dim=16)
        
        # Compute Laplacian positional embeddings
        lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=16)
        
        # Compute subgraph-level embeddings using mean pooling
        subgraph_embedding = mean_pooling(gcn_embeddings)
        
        # Append subgraph-level embeddings and labels
        subgraph_embeddings.append(subgraph_embedding)
        lpe_embeddings.append(lpe.mean(dim=0))  # Mean pooling for LPE as well
        node_labels.append(subgraph.y)
        num_nodes_list.append(subgraph.num_nodes)

        # Store the masks for each subgraph
        train_masks.append(subgraph.train_mask)
        val_masks.append(subgraph.val_mask)
        test_masks.append(subgraph.test_mask)

        # Debugging prints to check tensor sizes
        print(f"Subgraph {i} - GCN Embeddings Size: {gcn_embeddings.size()}")
        print(f"Subgraph {i} - LPE Size: {lpe.size()}")
        print(f"Subgraph {i} - Subgraph Embedding Size: {subgraph_embedding.size()}")
        print(f"Subgraph {i} - Node Labels Size: {subgraph.y.size()}")
        print("="*100)

    subgraph_embeddings = torch.stack(subgraph_embeddings)
    lpe_embeddings = torch.stack(lpe_embeddings)
    node_labels = torch.cat(node_labels, dim=0)
    num_nodes_list = torch.tensor(num_nodes_list)

    # Stack all tensors
    train_mask = torch.cat(train_masks)
    val_mask = torch.cat(val_masks)
    test_mask = torch.cat(test_masks)

    # Debugging prints to check tensor sizes
    print(f"Subgraph Embeddings Size: {subgraph_embeddings.size()}")
    print(f"LPE Embeddings Size: {lpe_embeddings.size()}")
    print(f"Node Labels Size: {node_labels.size()}")
    print(f"Num Nodes List Size: {num_nodes_list.size()}")

    # Ensure all tensors have the same first dimension
    assert subgraph_embeddings.size(0) == lpe_embeddings.size(0) == num_nodes_list.size(0), "Size mismatch between tensors"

    # Create datasets using masks
    train_dataset = NodeLevelDataset(
        subgraph_embeddings,
        lpe_embeddings,
        node_labels,
        num_nodes_list,
        train_mask
    )
    
    val_dataset = NodeLevelDataset(
        subgraph_embeddings,
        lpe_embeddings,
        node_labels,
        num_nodes_list,
        val_mask
    )
    
    test_dataset = NodeLevelDataset(
        subgraph_embeddings,
        lpe_embeddings,
        node_labels,
        num_nodes_list,
        test_mask
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=len(train_dataset),
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    print(f"Number of nodes in train/val/test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # Step 4: Initialize and train the model
    input_dim = 16  # Fixed embedding size
    model = GraphTransformer(input_dim=input_dim, embed_dim=16, num_heads=16, num_layers=8, ff_dim=64, dropout=0.1, num_classes=7)
    
    # Train the model
    train_metrics = train_model(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        num_epochs=2,
        learning_rate=0.001
    )
    print("\nFinal Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.2f}%")

    # Evaluate on validation set
    val_metrics = evaluate_model(model, val_dataloader)
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.2f}%")

    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_dataloader)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.2f}%")

if __name__ == "__main__":
    main()