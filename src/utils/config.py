import json

def load_config(config_file='src/configs/default_config.json'):
    with open(config_file) as f:
        config = json.load(f)
    
    # Add use_partition  config if not present
    if 'use_partition' not in config:
        config['use_partition'] = True
    
    return config

def update_config_with_args(config, args):
    """
    Override configuration values with command-line arguments if provided.
    
    This function updates keys in the configuration dictionary based on the
    attributes in the 'args' object. It handles general settings (model, dataset,
    out_dir), training parameters, and network parameters.
    
    Parameters:
        config (dict): The configuration dictionary.
        args (Namespace): The parsed command-line arguments.
        
    Returns:
        dict: Updated configuration dictionary.
    """
    # --- GPU Configuration ---
    if hasattr(args, 'gpu_id') and args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True

    # --- General Settings: model, dataset, out_dir ---
    config['model'] = args.model if hasattr(args, 'model') and args.model is not None else config.get('model', 'default_model')
    config['dataset'] = args.dataset if hasattr(args, 'dataset') and args.dataset is not None else config.get('dataset', 'Cora')
    config['out_dir'] = args.out_dir if hasattr(args, 'out_dir') and args.out_dir is not None else config.get('out_dir', 'out/')

    # --- Training Parameters ---
    if 'params' in config:
        if hasattr(args, 'seed') and args.seed is not None:
            config['params']['seed'] = int(args.seed)
        if hasattr(args, 'epochs') and args.epochs is not None:
            config['params']['epochs'] = int(args.epochs)
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            config['params']['batch_size'] = int(args.batch_size)
        if hasattr(args, 'init_lr') and args.init_lr is not None:
            config['params']['init_lr'] = float(args.init_lr)
        if hasattr(args, 'lr_reduce_factor') and args.lr_reduce_factor is not None:
            config['params']['lr_reduce_factor'] = float(args.lr_reduce_factor)
        if hasattr(args, 'lr_schedule_patience') and args.lr_schedule_patience is not None:
            config['params']['lr_schedule_patience'] = int(args.lr_schedule_patience)
        if hasattr(args, 'min_lr') and args.min_lr is not None:
            config['params']['min_lr'] = float(args.min_lr)
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            config['params']['weight_decay'] = float(args.weight_decay)
        if hasattr(args, 'print_epoch_interval') and args.print_epoch_interval is not None:
            config['params']['print_epoch_interval'] = int(args.print_epoch_interval)
        if hasattr(args, 'max_time') and args.max_time is not None:
            config['params']['max_time'] = float(args.max_time)

    # --- Network Parameters ---
    if 'net_params' in config:
        if hasattr(args, 'L') and args.L is not None:
            config['net_params']['L'] = int(args.L)
        if hasattr(args, 'hidden_dim') and args.hidden_dim is not None:
            config['net_params']['hidden_dim'] = int(args.hidden_dim)
        if hasattr(args, 'out_dim') and args.out_dim is not None:
            config['net_params']['out_dim'] = int(args.out_dim)
        if hasattr(args, 'residual') and args.residual is not None:
            config['net_params']['residual'] = True if args.residual == 'True' else False
        if hasattr(args, 'edge_feat') and args.edge_feat is not None:
            config['net_params']['edge_feat'] = True if args.edge_feat == 'True' else False
        if hasattr(args, 'readout') and args.readout is not None:
            config['net_params']['readout'] = args.readout
        if hasattr(args, 'n_heads') and args.n_heads is not None:
            config['net_params']['n_heads'] = int(args.n_heads)
        if hasattr(args, 'in_feat_dropout') and args.in_feat_dropout is not None:
            config['net_params']['in_feat_dropout'] = float(args.in_feat_dropout)
        if hasattr(args, 'dropout') and args.dropout is not None:
            config['net_params']['dropout'] = float(args.dropout)
        if hasattr(args, 'layer_norm') and args.layer_norm is not None:
            config['net_params']['layer_norm'] = True if args.layer_norm == 'True' else False
        if hasattr(args, 'batch_norm') and args.batch_norm is not None:
            config['net_params']['batch_norm'] = True if args.batch_norm == 'True' else False
        if hasattr(args, 'self_loop') and args.self_loop is not None:
            config['net_params']['self_loop'] = True if args.self_loop == 'True' else False
        if hasattr(args, 'lap_pos_enc') and args.lap_pos_enc is not None:
            config['net_params']['lap_pos_enc'] = True if args.lap_pos_enc == 'True' else False
        if hasattr(args, 'pos_enc_dim') and args.pos_enc_dim is not None:
            config['net_params']['pos_enc_dim'] = int(args.pos_enc_dim)
        if hasattr(args, 'wl_pos_enc') and args.wl_pos_enc is not None:
            config['net_params']['wl_pos_enc'] = True if args.wl_pos_enc == 'True' else False

    return config
