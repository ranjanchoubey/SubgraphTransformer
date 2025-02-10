import dgl
import os
import torch
import numpy as np

from src.data.train_test_splitting import create_mask_splits

"""
    File to load dataset based on user control from main file
"""


def LoadData(DATASET_NAME):
    """
    Load dataset and store it in organized subdirectories
    """    
    # Set base datasets directory
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets')
    
    # Create dataset-specific directory
    dataset_dir = os.path.join(base_dir, DATASET_NAME.lower())
    os.makedirs(dataset_dir, exist_ok=True)
    
    if DATASET_NAME == 'CoraSmall':  
        graph = dgl.data.CoraGraphDataset(raw_dir=dataset_dir)
        dataset = graph[0]        
        return dataset
    elif DATASET_NAME == 'CoraFull':
        graph = dgl.data.CoraFullDataset(raw_dir=dataset_dir)
        dataset = graph[0]
        
        # Create and add masks for CoraFull
        num_nodes = dataset.number_of_nodes()
        
        # cora Full Dataset download does not have masks so we create them
        train_mask, val_mask, test_mask = create_mask_splits(num_nodes)
        
        # Add masks to the graph
        dataset.ndata['train_mask'] = train_mask
        dataset.ndata['val_mask'] = val_mask
        dataset.ndata['test_mask'] = test_mask
        
        return dataset
