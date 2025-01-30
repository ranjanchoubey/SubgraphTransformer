"""
    IMPORTING LIBS
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import os
import time
import torch
import argparse
from tqdm import tqdm

"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from src.data.data_processing import load_cora_data, partition_graph
from src.data.embedding import mean_pooling, compute_laplacian_positional_embedding, compute_gcn_embeddings
from src.nets.transformer import GraphTransformer
from src.train.trainer import train_model, evaluate_model
from src.utils.utils import set_seed
from src.configs.config import load_config

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if device.type == "cuda":
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
    else:
        print('cuda not available, using CPU')
    return device

def main():    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/configs/default_config.json')
    parser.add_argument('--gpu_id', type=int, help="GPU ID")
    parser.add_argument('--dataset', default="Cora", help="Dataset name")
    parser.add_argument('--out_dir', default="out/", help="Output directory")
    args = parser.parse_args()
    
    # Load and setup configuration
    config = load_config(args.config)
    if args.gpu_id is not None:
        config.gpu.id = args.gpu_id
        
    device = gpu_setup(config.gpu.use, config.gpu.id)
    
    # Create output directories
    timestamp = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    out_dir = os.path.join(args.out_dir, f"Cora_GraphTransformer_GPU{config.gpu.id}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Step 1: Loading Configuration")
    print("="*50)
    set_seed(config.training.seed)
    print("✓ Configuration loaded successfully")

    print("\n" + "="*50)
    print("Step 2: Loading Dataset")
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
    
    # Initialize lists for storing embeddings and metadata
    subgraph_embeddings, lpe_embeddings = [], []
    node_labels, node_counts, node_indices = [], [], []
    start_idx = 0

    # Get masks
    train_mask = graph.train_mask.to(device)
    val_mask = graph.val_mask.to(device)
    test_mask = graph.test_mask.to(device)

    # Process each subgraph
    for i in range(config.data.num_parts):
        subgraph = cluster_data[i]
        num_nodes = subgraph.num_nodes
        node_indices.append(torch.arange(start_idx, start_idx + num_nodes, device=device))
        start_idx += num_nodes
        
        # Compute embeddings
        gcn_embeddings = compute_gcn_embeddings(
            subgraph, 
            input_dim=config.gcn.input_dim,
            hidden_dim=config.gcn.hidden_dim,
            output_dim=config.gcn.output_dim
        )
        lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=config.model.embed_dim)
        
        # Store results
        subgraph_embeddings.append(mean_pooling(gcn_embeddings))
        lpe_embeddings.append(lpe.mean(dim=0))
        
        node_labels.append(subgraph.y)
        node_counts.append(num_nodes)

    # Stack and move to device
    subgraph_embeddings = torch.stack(subgraph_embeddings).to(device)
    lpe_embeddings = torch.stack(lpe_embeddings).to(device)
    
    node_labels = torch.cat(node_labels, dim=0).to(device)
    node_counts = torch.tensor(node_counts).to(device)

    print("\n" + "="*50)
    print("Step 5: Model Training")
    print("="*50)
    
    model = GraphTransformer(
        input_dim=config.gcn.output_dim,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        ff_dim=config.model.ff_dim,
        dropout=config.model.dropout,
        num_classes=config.model.num_classes
    ).to(device)

    train_metrics = train_model(
        model=model,
        subgraph_embeddings=subgraph_embeddings,
        lpe_embeddings=lpe_embeddings,
        node_labels=node_labels,
        node_counts=node_counts,
        train_mask=train_mask,
        val_mask=val_mask,
        node_indices=node_indices,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate
    )

    print("\n" + "="*50)
    print("Step 6: Final Evaluation")
    print("="*50)
    
    test_metrics = evaluate_model(
        model=model,
        subgraph_embeddings=subgraph_embeddings,
        lpe_embeddings=lpe_embeddings,
        node_labels=node_labels,
        node_counts=node_counts,
        mask=test_mask,
        node_indices=node_indices
    )

    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    main()