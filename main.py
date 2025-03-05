# main.py
import sys
sys.dont_write_bytecode = True
import os
import time
import torch
import glob
import argparse
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.utils.args import parse_arguments
from src.utils.config import load_config, update_config_with_args
from src.utils.misc import gpu_setup, set_seed
from src.data.data_loader import LoadData
from src.data.embedding import mean_pooling, compute_laplacian_positional_embedding, compute_gcn_embeddings
from src.utils.supergraph import create_feature_dataset
from src.models.model_definition import view_model_param
from src.training.training_pipeline import train_val_pipeline
from src.utils.component_analysis import get_component_info
from src.models.networks.load_net import gnn_model

def main():
    # Step 1: Load configuration.
    print("\n" + "="*50, "\n Step 1: Loading Configuration", "\n" + "="*50)
    
    args = parse_arguments()
    config = load_config(args.config)
    config = update_config_with_args(config, args)

    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    params = config['params']
    set_seed(params['seed'])
    out_dir = config['out_dir']
    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']
    print(f"\n Dataset: {DATASET_NAME}\n")
    
    # Setup network parameters.
    net_params = config.get('net_params', {})
    net_params['dataset'] = DATASET_NAME
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    
    # Setup output directories.
    time_str = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_log_dir = os.path.join(out_dir, 'logs', f"{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    root_ckpt_dir = os.path.join(out_dir, 'checkpoints', f"{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    write_file_name = os.path.join(out_dir, 'results', f"result_{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    write_config_file = os.path.join(out_dir, 'configs', f"config_{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'configs'), exist_ok=True)
    print("✓ Configuration loaded successfully")
    
    # Step 2: Load dataset.
    print("\n" + "="*50, "\n Step 2: Loading Dataset", "\n" + "="*50)
    graph = LoadData(DATASET_NAME)
    print("✓ Dataset loaded successfully")
    
    # Check if we're running the GCN baseline.
    if MODEL_NAME == "GCNBaseline":
        # For the baseline, we use the entire graph.
        net_params['in_dim'] = graph.ndata['feat'].shape[1]
        net_params['n_classes'] = config['data']['num_classes']
        
        train_mask = graph.ndata['train_mask'].to(device)
        val_mask = graph.ndata['val_mask'].to(device)
        test_mask = graph.ndata['test_mask'].to(device)
        node_labels = graph.ndata['label'].to(device)
        
        # Wrap the graph in a list for consistency.
        dataset = [graph]
        subgraphs = None
        subgraph_components = None
        node_counts = torch.tensor([graph.number_of_nodes()]).to(device)
    else:
        # For the transformer-based model, perform partitioning.
        print("\n" + "="*50, "\n Step 3: Partitioning Graph and Analyzing Components", "\n" + "="*50)
        from src.data.partitioning import partition_graph
        subgraphs = partition_graph(graph, num_parts=net_params.get('num_subgraph', 4))
        
        subgraph_components = []
        # We'll also collect masks from each subgraph.
        train_masks, val_masks, test_masks = [], [], []
        node_labels_list = []
        node_counts_list = []
        
        print("Processing subgraphs ...")
        for i, subgraph in enumerate(subgraphs):
            component_info = get_component_info(subgraph)
            subgraph_components.append(component_info)
            
            # Collect masks and labels.
            train_masks.append(subgraph.ndata['train_mask'])
            val_masks.append(subgraph.ndata['val_mask'])
            test_masks.append(subgraph.ndata['test_mask'])
            node_labels_list.append(subgraph.ndata['label'])
            node_counts_list.append(subgraph.number_of_nodes())
        
        # Concatenate masks and labels across subgraphs.
        train_mask = torch.cat(train_masks, dim=0).to(device)
        val_mask = torch.cat(val_masks, dim=0).to(device)
        test_mask = torch.cat(test_masks, dim=0).to(device)
        node_labels = torch.cat(node_labels_list, dim=0).to(device)
        node_counts = torch.tensor(node_counts_list).to(device)
        
        print("\n" + "="*50, "\n Step 4: Computing Embeddings", "\n" + "="*50)
        subgraph_embeddings, lpe_embeddings = [], []
        
        for subgraph in subgraphs:
            gcn_embeddings = compute_gcn_embeddings(
                subgraph, 
                input_dim=config['gcn']['input_dim'],
                hidden_dim=config['gcn']['hidden_dim'],
                output_dim=config['gcn']['output_dim']
            )
            lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=config['gcn']['output_dim'])
            subgraph_embeddings.append(mean_pooling(gcn_embeddings))
            lpe_embeddings.append(mean_pooling(lpe))
        
        subgraph_embeddings = torch.stack(subgraph_embeddings).to(device)
        lpe_embeddings = torch.stack(lpe_embeddings).to(device)
        
        from src.utils.supergraph import create_feature_dataset
        combined_embedding = subgraph_embeddings + lpe_embeddings
        dataset = create_feature_dataset(combined_embedding)
        
        # Set in_dim for transformer based on the GCN output dimension.
        net_params['in_dim'] = config['gcn']['output_dim']
        net_params['n_classes'] = config['data']['num_classes']
    
    # Step 5: Final statistics and training.
    print("\n" + "="*50, "\n Step 5: Final statistics and training ", "\n" + "="*50)
    if MODEL_NAME == "GCNBaseline":
        print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        print(f"Feature dimension: {net_params['in_dim']}")
    else:
        print(f"Total number of subgraphs: {len(subgraphs)}")
        print(f"GCN embedding shape: {subgraph_embeddings.shape}")
        print(f"LPE embedding shape: {lpe_embeddings.shape}")
        print(f"Average nodes per subgraph: {torch.mean(node_counts.float()):.2f}")
    
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, subgraph_components)
    train_val_pipeline(
        MODEL_NAME,
        dataset,
        params,
        net_params,
        dirs=(root_log_dir, root_ckpt_dir, write_file_name, write_config_file),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        node_labels=node_labels,
        node_counts=node_counts,
        subgraphs=subgraphs,
        subgraph_components=subgraph_components
    )

if __name__ == "__main__":
    main()
