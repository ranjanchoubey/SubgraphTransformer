import torch
from dgl.data import DGLDataset

class SuperGraphDataset(DGLDataset):
    def __init__(self, features):
        """Initialize with just the features matrix.
        Args:
            features: Combined embeddings tensor [num_subgraphs, feature_dim]
        """
        self.features = features
        super().__init__(name='SuperGraph')

    def process(self):
        pass

    def __getitem__(self, _):
        return self.features

    def __len__(self):
        return 1

def create_feature_dataset(combined_embedding):
    """Create dataset from combined embeddings without graph structure."""
    print(f"\n ------ Feature Dataset Statistics:------------")
    print(f"Number of subgraphs: {combined_embedding.shape[0]}")
    print(f"Feature dimension: {combined_embedding.shape[1]}")
    print(f"Device: {combined_embedding.device}")
    
    dataset = SuperGraphDataset(combined_embedding)
    return dataset


