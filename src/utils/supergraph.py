import dgl
import torch
from dgl.data import DGLDataset

class SuperGraphDataset(DGLDataset):
    def __init__(self, graph, labels=None):
        self.graph = graph
        self.labels = labels if labels is not None else torch.zeros(graph.num_nodes())
        super().__init__(name='SuperGraph')

    def process(self):
        pass

    def __getitem__(self, _):  # idx parameter not used as we always return the same graph
        return self.graph, self.labels

    def __len__(self):
        return 1

def create_DGLSupergraph(combined_embedding):
    """Create a complete DGL supergraph with the combined embeddings."""
    num_nodes = combined_embedding.shape[0]
    device = combined_embedding.device
    
    # Create source and destination indices for complete graph
    src_indices = []
    dst_indices = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                src_indices.append(i)
                dst_indices.append(j)
    
    # Convert to tensors and move to correct device
    src = torch.tensor(src_indices, device=device)
    dst = torch.tensor(dst_indices, device=device)
    
    # Create graph directly on the correct device
    supergraph = dgl.graph((src, dst), device=device)
    
    # Add node features
    supergraph.ndata['feat'] = combined_embedding
    
    # Print supergraph information
    print(f"\n ------ Supergraph Statistics:------------")
    print(f"Number of nodes: {supergraph.num_nodes()}")
    print(f"Number of edges: {supergraph.num_edges()}")
    print(f"Node feature shape: {supergraph.ndata['feat'].shape}")
    print(f"Device: {supergraph.device}")
    
    # Convert to dataset object
    dataset = SuperGraphDataset(supergraph)
    print(f"Dataset object created: {dataset}")
    
    return dataset


