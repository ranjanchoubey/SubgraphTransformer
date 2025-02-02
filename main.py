"""
    IMPORTING LIBS
"""
import random
import sys
sys.dont_write_bytecode = True

import numpy as np
import os
import time
import torch
import glob
import torch.optim as optim
import argparse
import dgl 
from tqdm import tqdm
import json
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from src.data.data import LoadData, partition_graph
from src.data.embedding import mean_pooling, compute_laplacian_positional_embedding, compute_gcn_embeddings
from src.nets.load_net import gnn_model 

from src.train.trainer import collate_graphs, evaluate_network, train_epoch
from src.utils.supergraph import  create_DGLSupergraph
from src.configs.config import load_config

from torch.utils.data import DataLoader
import dgl
import torch



def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if device.type == "cuda":
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
    else:
        print('cuda not available, using CPU')
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""
def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs,graph,node_labels,node_counts):

    start0 = time.time()
    per_epoch_time = []
    
    DATASET_NAME = 'Cora'

    # Extract the masks
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    print("train_mask : ",train_mask.shape)
    
    trainset = dataset
    valset = dataset
    testset = dataset

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)    
    
    
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=params['lr_reduce_factor'],
                                                    patience=params['lr_schedule_patience'],
                                                    verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], [] 

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_graphs)
    # print("******** train_loader *********",len(train_loader)) # it is 1

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:   
                
                t.set_description('Epoch %d' % epoch)
            
                start = time.time()
                
                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, train_mask,node_labels,node_counts)                

                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch,  val_mask, node_labels, node_counts, phase="val")
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, test_mask, node_labels, node_counts, phase="test")                    

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)                

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)                
                
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                            train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                            train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                            test_acc=epoch_test_acc)         
                
                per_epoch_time.append(time.time()-start)
                
                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))
    
                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)
    
                scheduler.step(epoch_val_loss)
    
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                    break    
                
                # Stop training after params['max_time'] hours
                if time.time()-start0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    _, test_acc = evaluate_network(model, device, test_loader, epoch,test_mask, node_labels,node_counts)
    _, train_acc = evaluate_network(model, device, train_loader, epoch,train_mask, node_labels,node_counts)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-start0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_acc, train_acc, epoch, (time.time()-start0)/3600, np.mean(per_epoch_time)))



def main():
    """
        USER CONTROLS
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/configs/default_config.json')
    parser.add_argument('--gpu_id', type=int, help="GPU ID")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', default="Cora", help="Dataset name")
    parser.add_argument('--out_dir', default="out/", help="Output directory")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--lap_pos_enc', help="Please give a value for lap_pos_enc")
    parser.add_argument('--wl_pos_enc', help="Please give a value for wl_pos_enc")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    
        # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id']) 
    
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
        
    # parameters
    
    params = config['params'] 
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)    
    
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.lap_pos_enc is not None:
        net_params['lap_pos_enc'] = True if args.pos_enc=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.wl_pos_enc is not None:
        net_params['wl_pos_enc'] = True if args.pos_enc=='True' else False

    # Cora
    net_params['in_dim'] = config['gcn']['output_dim']  # This is 16 from GCN output
    net_params['n_classes'] = 7
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file
    
    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')


    print("\n" + "="*50)
    print("Step 1: Loading Configuration")
    print("="*50)
    # set_seed(config.training.seed)
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
        
    print("✓ Configuration loaded successfully")
    
    
    print("\n" + "="*50)
    print("Step 1: Loading Dataset")
    print("="*50)
    graph = LoadData(DATASET_NAME)  # Now using DGL data loading

    print("✓ Dataset loaded successfully")
    
    print("\n" + "="*50)
    print("Step 3: Partitioning Graph")
    print("="*50)
    subgraphs = partition_graph(graph, num_parts=config['data']['num_parts'])
    print(f"✓ Graph partitioned into {config['data']['num_parts']} subgraphs")
    

    print("\n" + "="*50)
    print("Step 4: Computing Embeddings")
    print("="*50)
    
    # Initialize lists for storing embeddings and metadata
    subgraph_embeddings, lpe_embeddings = [], []
    node_labels, node_counts, node_indices = [], [], []
    start_idx = 0

    # Process each subgraph
    for i, subgraph in enumerate(subgraphs):
        num_nodes = subgraph.number_of_nodes()
        node_indices.append(torch.arange(start_idx, start_idx + num_nodes, device=device))
        start_idx += num_nodes
        
        # Compute embeddings
        gcn_embeddings = compute_gcn_embeddings(
            subgraph, 
            input_dim=config['gcn']['input_dim'],
            hidden_dim=config['gcn']['hidden_dim'],
            output_dim=config['gcn']['output_dim']
        )
        lpe = compute_laplacian_positional_embedding(subgraph, embedding_dim=config['gcn']['output_dim'])
        
        # Store results
        subgraph_embeddings.append(mean_pooling(gcn_embeddings))
        lpe_embeddings.append(mean_pooling(lpe))
        
        # Get labels from DGL graph
        node_labels.append(subgraph.ndata['label'])
        node_counts.append(num_nodes)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(subgraphs)} subgraphs")

    # Stack and move to device
    subgraph_embeddings = torch.stack(subgraph_embeddings).to(device)
    lpe_embeddings = torch.stack(lpe_embeddings).to(device)
    
    node_labels = torch.cat(node_labels, dim=0).to(device)
    node_counts = torch.tensor(node_counts).to(device)
    
    print("\n" + "="*50)
    print("Step 5: Final Statistics")
    print("="*50)
    print(f"Total number of subgraphs: {len(subgraphs)}")
    print(f"GCN embedding shape: {subgraph_embeddings.shape}")
    print(f"LPE embedding shape: {lpe_embeddings.shape}")
    print(f"Average nodes per subgraph: {torch.mean(node_counts.float()):.2f}")
    
    # Fix the function call
    combined_embedding = subgraph_embeddings + lpe_embeddings
    supergraph = create_DGLSupergraph(combined_embedding)
    

    
    dataset = supergraph
     # size = [subgraph * embedding dimension]
    
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs,graph,node_labels,node_counts)


#     # print("\n🎉 Training Complete!")

if __name__ == "__main__":
    main()