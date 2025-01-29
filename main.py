import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from data.data_processing import load_cora_data, partition_graph
from data.embedding import mean_pooling, compute_laplacian_positional_embedding, compute_gcn_embeddings
from src.models.transformer import GraphTransformer
from train.trainer import train_model, evaluate_model
from utils.utils import set_seed
from config.config import load_config

def main():
    print("\n" + "="*50)
    print("Step 1: Loading Configuration")
    print("="*50)
    config = load_config('config/default_config.yaml')
    set_seed(config.training.seed)
    print("✓ Configuration loaded successfully")

    print("\n" + "="*50)
    print("Step 2: Loading Cora Dataset")
    print("="*50)
    graph = load_cora_data(config.data.data_path)
    print("✓ Dataset loaded successfully")

    print("\n" + "="*50)
    print("Step 3: Partitioning Graph")
    print("="*50)
    cluster_data = partition_graph(graph, num_parts=config.data.num_parts)
    print(f"✓ Graph partitioned into {config.data.num_parts} subgraphs")

    print("\n" + "="*50)
    print("Step 4: Computing Embeddings")
    print("="*50)
    subgraph_embeddings = []
    lpe_embeddings = []
    node_labels = []
    node_counts = []  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    node_indices = []
    start_idx = 0

    # Get masks directly from the original graph
    train_mask = graph.train_mask.to(device)
    val_mask = graph.val_mask.to(device)
    test_mask = graph.test_mask.to(device)

    for i in range(config.data.num_parts):
        subgraph = cluster_data[i]
        num_nodes = subgraph.num_nodes
        node_indices.append(torch.arange(start_idx, start_idx + num_nodes, device=device))
        start_idx += num_nodes
        
        # Update GCN embedding computation with config values
        gcn_embeddings = compute_gcn_embeddings(
            subgraph, 
            input_dim=1433,  # Original Cora feature dimension
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim
        )
        lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=config.model.embed_dim)
        subgraph_embedding = mean_pooling(gcn_embeddings)
        
        subgraph_embeddings.append(subgraph_embedding)
        lpe_embeddings.append(lpe.mean(dim=0))
        node_labels.append(subgraph.y)
        # `node_counts` is a list that stores the number of nodes in each subgraph. It is used to keep
        # track of the number of nodes in each partitioned subgraph of the original graph. This
        # information is important for various computations and operations within the code, such as
        # computing embeddings, handling masks, and ensuring consistency in the sizes of tensors
        # during processing.
        node_counts.append(subgraph.num_nodes)  # Changed from num_nodes_list

        if i % 20 == 0:
            print(f"✓ Processed {i}/{config.data.num_parts} subgraphs")

    subgraph_embeddings = torch.stack(subgraph_embeddings).to(device)
    lpe_embeddings = torch.stack(lpe_embeddings).to(device)
    node_labels = torch.cat(node_labels, dim=0).to(device)
    node_counts = torch.tensor(node_counts).to(device)  # Changed from num_nodes_list

    print(f"Subgraph Embeddings Size: {subgraph_embeddings.size()}")
    print(f"LPE Embeddings Size: {lpe_embeddings.size()}")
    print(f"Node Labels Size: {node_labels.size()}")

    assert subgraph_embeddings.size(0) == lpe_embeddings.size(0) == node_counts.size(0), "Size mismatch between tensors"

    print("\n" + "="*50)
    print("Step 5: Model Initialization")
    print("="*50)
    model = GraphTransformer(
        input_dim=config.model.output_dim,  # This should match GCN output_dim
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        ff_dim=config.model.ff_dim,
        dropout=config.model.dropout,
        num_classes=config.model.num_classes
    ).to(device)
    print("✓ Model initialized successfully")

    print("\n" + "="*50)
    print("Step 6: Training Phase")
    print("="*50)
    train_metrics = train_model(
        model=model,
        subgraph_embeddings=subgraph_embeddings,
        lpe_embeddings=lpe_embeddings,
        node_labels=node_labels,
        node_counts=node_counts,
        train_mask=train_mask,
        val_mask=val_mask,  # Add validation mask
        node_indices=node_indices,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate
    )

    print("\n" + "="*50)
    print("Step 7: Final Evaluation")
    print("="*50)
    
    # Only run test set evaluation at the very end
    print("\n📊 Test Set Performance:")
    test_metrics = evaluate_model(
        model=model,
        subgraph_embeddings=subgraph_embeddings,
        lpe_embeddings=lpe_embeddings,
        node_labels=node_labels,
        node_counts=node_counts,
        mask=test_mask,
        node_indices=node_indices
    )
    for metric, value in test_metrics.items():
        print(f"✓ {metric}: {value:.2f}%")

    print("\n" + "="*50)
    print("🎉 Completed !")
    print("="*50)

if __name__ == "__main__":
    main()
