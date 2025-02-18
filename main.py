# importing libraries
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
from torch.utils.data import DataLoader
from src.utils.args import parse_arguments
from src.utils.config import load_config, update_config_with_args
from src.utils.misc import gpu_setup, set_seed
from src.data.data_loader import LoadData
from src.data.partitioning import partition_graph
from src.data.embedding import mean_pooling, compute_laplacian_positional_embedding, compute_gcn_embeddings
from src.utils.supergraph import create_feature_dataset
from src.models.model_definition import view_model_param
from src.training.training_pipeline import train_val_pipeline
from src.utils.component_analysis import get_component_info



def main():
    
    # Step 1: Load configuration.
    print("\n" + "="*50,"\n Step 1: Loading Configuration","\n"+"="*50)
    
    # Parse arguments and load configuration.
    args = parse_arguments()
    config = load_config(args.config)
    config = update_config_with_args(config, args)

    # Setup device, seed, and retrieve common settings.
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    params = config['params']
    set_seed(params['seed'])
    out_dir = config['out_dir']
    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']
    print(f"\n Dataset: {DATASET_NAME}\n")
    # Setup network parameters.
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    net_params['in_dim'] = config['gcn']['output_dim']  # Example: 16 (from GCN output)
    net_params['n_classes'] = config['data']['num_classes']  # For example, 7 in cora small dataset

    

    # Setup output directories for logs, checkpoints, results, and configs.
    time_str = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_log_dir = os.path.join(out_dir, 'logs', f"{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    root_ckpt_dir = os.path.join(out_dir, 'checkpoints', f"{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    write_file_name = os.path.join(out_dir, 'results', f"result_{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    write_config_file = os.path.join(out_dir, 'configs', f"config_{MODEL_NAME}_{DATASET_NAME}_GPU{config['gpu']['id']}_{time_str}")
    dirs = (root_log_dir, root_ckpt_dir, write_file_name, write_config_file)
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'configs'), exist_ok=True)
    print("✓ Configuration loaded successfully")
    
    
    # Step 2: Load and partition the dataset.
    print("\n" + "="*50,"\n Step 2: Loading Dataset","\n"+"="*50)
    graph = LoadData(DATASET_NAME)
    print("✓ Dataset loaded successfully")
    
    
    # Step 3: Partition the graph and analyze components
    print("\n" + "="*50,"\n Step 3: Partitioning Graph and Analyzing Components","\n"+"="*50)
    subgraphs = partition_graph(graph, num_parts=config['data']['num_parts'])
    
    # Store component information during preprocessing[[9,6],[5,4,3],...] - varying number of components per
    subgraph_components = []
    for i, subgraph in enumerate(subgraphs):
        component_info = get_component_info(subgraph)
        subgraph_components.append(component_info)
        # print(f"Subgraph {i}: {len(component_info)} components")
    print("subgraph_components",subgraph_components)    

    print(f"✓ Graph partitioned into {config['data']['num_parts']} subgraphs with component analysis")
    
    # Step 4: Compute embeddings, node label and mask for each subgraph.
    print("\n" + "="*50,"\n Step 4: Computing Embeddings","\n"+"="*50)
    subgraph_embeddings, lpe_embeddings = [], []
    node_labels, node_counts, node_indices = [], [], []
    train_mask,val_mask,test_mask = [],[],[]
    start_idx = 0
    
    for i, subgraph in enumerate(subgraphs):
        num_nodes = subgraph.number_of_nodes()
        node_indices.append(torch.arange(start_idx, start_idx + num_nodes, device=device))
        start_idx += num_nodes
        
        gcn_embeddings = compute_gcn_embeddings(subgraph,input_dim=config['gcn']['input_dim'],hidden_dim=config['gcn']['hidden_dim'],output_dim=config['gcn']['output_dim'])
        lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=config['gcn']['output_dim'])
        
        subgraph_embeddings.append(mean_pooling(gcn_embeddings))
        lpe_embeddings.append(mean_pooling(lpe))
        node_labels.append(subgraph.ndata['label'])
        node_counts.append(num_nodes)
        train_mask.append(subgraph.ndata['train_mask'])
        val_mask.append(subgraph.ndata['val_mask'])
        test_mask.append(subgraph.ndata['test_mask'])
    print("✓ Embeddings computed successfully")
    
    subgraph_embeddings = torch.stack(subgraph_embeddings).to(device)
    lpe_embeddings = torch.stack(lpe_embeddings).to(device)
    node_labels = torch.cat(node_labels, dim=0).to(device)
    node_counts = torch.tensor(node_counts).to(device)
    train_mask = torch.cat(train_mask, dim=0).to(device)
    val_mask = torch.cat(val_mask, dim=0).to(device)
    test_mask = torch.cat(test_mask, dim=0).to(device)
    
    # Step 5: Final statistics and training.
    print("\n" + "="*50,"\n Step 4: Final statistics and training ","\n"+"="*50)
    print(f"Total number of subgraphs: {len(subgraphs)}")
    print(f"GCN embedding shape: {subgraph_embeddings.shape}")
    print(f"LPE embedding shape: {lpe_embeddings.shape}")
    print(f"Average nodes per subgraph: {torch.mean(node_counts.float()):.2f}\n")
    
    # Combine the subgraph embeddings  and laplacian embedding and create the dataset.
    combined_embedding = subgraph_embeddings + lpe_embeddings
    dataset = create_feature_dataset(combined_embedding)
    
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params,subgraph_components)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, train_mask,val_mask,test_mask, node_labels, node_counts, subgraphs,subgraph_components)  # Added subgraph_components

if __name__ == "__main__":
    main()



