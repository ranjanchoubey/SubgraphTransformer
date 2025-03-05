# training_pipeline.py
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
from src.utils.visualization import plot_train_val_curves, visualize_node_predictions, visualize_subgraph

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, 
                       train_mask, val_mask, test_mask, node_labels, 
                       node_counts, subgraphs, subgraph_components=None):
    start0 = time.time()
    per_epoch_time = []
    DATASET_NAME = net_params['dataset']
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    with open(write_config_file + '.txt', 'w') as f:
        f.write(f"Dataset: {DATASET_NAME}\nModel: {MODEL_NAME}\nparams={params}\nnet_params={net_params}\nTotal Parameters: {net_params.get('total_param', 'N/A')}\n")
        
    # If running the GCN baseline, use a simplified training loop.
    if MODEL_NAME == "GCNBaseline":
        writer = SummaryWriter(log_dir=os.path.join(root_log_dir, "RUN_0"))
        model = gnn_model(MODEL_NAME, net_params, subgraph_components=None).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        
        graph = dataset[0].to(device)
        features = graph.ndata['feat']
        
        epoch_train_losses = []
        epoch_train_accs = []
        epoch_val_losses = []
        epoch_val_accs = []
        
        for epoch in range(params['epochs']):
            model.train()
            logits = model(graph, features)
            loss = model.loss(logits, node_labels, train_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                logits = model(graph, features)
                pred = logits.argmax(dim=1)
                train_acc = (pred[train_mask] == node_labels[train_mask]).float().mean()
                val_acc   = (pred[val_mask]   == node_labels[val_mask]).float().mean()
                test_acc  = (pred[test_mask]  == node_labels[test_mask]).float().mean()
            
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(train_acc.item())
            epoch_val_losses.append(loss.item())
            epoch_val_accs.append(val_acc.item())
            
            writer.add_scalar('train/loss', loss.item(), epoch)
            writer.add_scalar('train/acc', train_acc.item(), epoch)
            writer.add_scalar('val/acc', val_acc.item(), epoch)
            writer.add_scalar('test/acc', test_acc.item(), epoch)
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {train_acc.item():.4f} | Val Acc: {val_acc.item():.4f} | Test Acc: {test_acc.item():.4f}")
            
            per_epoch_time.append(time.time() - start0)
            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_0")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pkl"))
            
            if time.time()-start0 > params['max_time']*3600:
                print("Max training time reached, stopping.")
                break
        
        writer.close()
        print(f"\nFinal Test Accuracy: {test_acc.item():.4f}")
        print(f"Total training time: {time.time()-start0:.2f}s")
        return

    # Otherwise, use your transformer-based training loop.
    writer = SummaryWriter(log_dir=os.path.join(root_log_dir, "RUN_0"))
    model = gnn_model(MODEL_NAME, net_params, subgraph_components=subgraph_components).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=params['lr_reduce_factor'],
                                                    patience=params['lr_schedule_patience'])
    
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_graphs)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []
    train_class_losses, train_reg_losses = [], []
    
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')
                start = time.time()
                total_loss, class_loss, reg_loss, epoch_train_acc, optimizer = train_epoch(
                    model, optimizer, device, train_loader, epoch,
                    train_mask, node_labels, node_counts, 
                    subgraphs=subgraphs,
                    subgraph_components=subgraph_components
                )
                epoch_train_losses.append(total_loss)
                train_class_losses.append(class_loss)
                train_reg_losses.append(reg_loss)
                epoch_train_accs.append(epoch_train_acc)
                
                epoch_val_loss, epoch_val_acc = evaluate_network(
                    model, device, val_loader, epoch, val_mask,
                    node_labels, node_counts, subgraph_components, phase="val"
                )
                _, epoch_test_acc = evaluate_network(
                    model, device, test_loader, epoch, test_mask,
                    node_labels, node_counts, subgraph_components, phase="test"
                )
                epoch_val_losses.append(epoch_val_loss)
                epoch_val_accs.append(epoch_val_acc)
                
                writer.add_scalar('train/loss', total_loss, epoch)
                writer.add_scalar('val/loss', epoch_val_loss, epoch)
                writer.add_scalar('train/acc', epoch_train_acc, epoch)
                writer.add_scalar('val/acc', epoch_val_acc, epoch)
                writer.add_scalar('test/acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=total_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)
                
                per_epoch_time.append(time.time()-start)
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_0")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pkl"))
                
                scheduler.step(epoch_val_loss)
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("Learning rate below minimum threshold, stopping.")
                    break
                if time.time()-start0 > params['max_time']*3600:
                    print("Max training time reached, stopping.")
                    break
                    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    _, test_acc = evaluate_network(model, device, test_loader, epoch,test_mask, node_labels,node_counts,subgraph_components,phase="test")
    _, train_acc = evaluate_network(model, device, train_loader, epoch,train_mask, node_labels,node_counts,subgraph_components,phase="train")
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

    # Plot training and validation curves.
    loss_data = {
        'train_loss': epoch_train_losses,
        'val_loss': epoch_val_losses,
        'train_class_loss': train_class_losses,
        'train_reg_loss': train_reg_losses
    }
    # Use the log directory for saving the plot
    plot_path = os.path.join('train_summary.png')
    plot_train_val_curves(loss_data, plot_path)
    
    # visualize subgraph comparison
    print("\nPlotting Subgraph ....\n")
    
    # Get label propagation config safely with defaults
    label_prop_config = params.get('label_propagation')

    print("\nprediction...")
    node_logits, node_labels = evaluate_network(
        model, device, test_loader, epoch, test_mask, 
        node_labels, node_counts,subgraph_components, phase="test", 
        compareSubgraph=True,
        subgraphs=subgraphs,
        label_prop_config=label_prop_config  # Pass the config
    )
    # visualize_node_predictions(node_logits, node_labels, node_counts, subgraphs)

    visualize_subgraph(node_logits, node_labels,node_counts,subgraphs)


