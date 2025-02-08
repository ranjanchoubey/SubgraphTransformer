import glob
import os
import time
import torch
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
from src.models.networks.load_net import gnn_model
from src.training.train_evaluate import collate_graphs, evaluate_network, train_epoch
from src.utils.visualization import visualize_subgraph


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs,train_mask,val_mask,test_mask, node_labels,node_counts,subgraphs):

    start0 = time.time()
    per_epoch_time = []
    
    DATASET_NAME = 'Cora'

    
    trainset = dataset
    valset = dataset
    testset = dataset

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)    
    
    
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=params['lr_reduce_factor'],
                                                    patience=params['lr_schedule_patience'])
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], [] 

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_graphs)


    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:   
                
                t.set_description('Epoch %d' % epoch)
            
                start = time.time()
                
                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, train_mask,node_labels,node_counts)                

                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch,  val_mask, node_labels, node_counts,phase="val")
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, test_mask, node_labels, node_counts,phase="test")                    

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

    _, test_acc = evaluate_network(model, device, test_loader, epoch,test_mask, node_labels,node_counts,phase="test")
    _, train_acc = evaluate_network(model, device, train_loader, epoch,train_mask, node_labels,node_counts,phase="train")
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


    print("\n *** Plotting Subgraph Comparison After Model Training .... ***\n")
    node_prediction, node_labels = evaluate_network(model, device, test_loader, epoch,test_mask, node_labels,node_counts,phase="test",comapreSubgraph=True)
    
    visualize_subgraph(node_prediction, node_labels,node_counts,subgraphs)
    

