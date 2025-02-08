import os
import random
import torch
import numpy as np


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if device.type == "cuda":
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
    else:
        print('cuda not available, using CPU')
    return device


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False