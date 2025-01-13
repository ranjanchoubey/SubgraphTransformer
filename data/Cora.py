import time
import numpy as np
import dgl
import torch
from scipy import sparse as sp
import networkx as nx
import hashlib
from dgl.data import CoraGraphDataset


class load_CorasDataSetDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, name=None, split=None):
        """
        Adapted to load the Cora dataset directly using DGL's built-in function.
        """
        self.split = split
        self.is_test = split.lower() in ['test', 'val']


        dataset = CoraGraphDataset(reverse_edge=True)
        self.dataset = dataset  # Cora only has one graph
        self.num_classes = dataset.num_classes

        self.node_labels = []
        self.graph_lists = []
        # print("self.n_samples  = ",self.n_samples )
        self._prepare()

    def _prepare(self):
        """
        Prepare the graph for the current split (train/val/test).
        """
        g = self.dataset[0]

        print(f"Preparing graph for the {self.split.upper()} set...")

        node_features = g.ndata['feat']
        edge_list = g.edges()

        # DGL Graph directly from Cora dataset
        g.ndata['feat'] = node_features
        g.ndata['label'] = g.ndata['label']

        self.graph_lists.append(g)
        self.node_labels.append(g.ndata['label'])

    def __len__(self):
        """Return the number of graphs (always 1 for Cora)."""
        return len(self.graph_lists)

    def __getitem__(self, idx):
        """
        Get the idx^th sample.
        """
        return self.graph_lists[idx], self.node_labels[idx]


class CorasDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name):
        """
        Dataset wrapper for loading Cora without pickle files.
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name

        self.train = load_CorasDataSetDGL(name="Cora", split='train')
        self.val = load_CorasDataSetDGL(name="Cora", split='val')
        self.test = load_CorasDataSetDGL(name="Cora", split='test')

        print("[I] Finished loading.")
        print("Here : ",len(self.train))
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

        print("Here : ",len(self.train))

def self_loop(g):
    """
    Add self-loops to the graph.
    """
    new_g = dgl.add_self_loop(g)
    return new_g





def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Use adjacency_matrix instead of adjacency_matrix_scipy
    A = g.adjacency_matrix().to_dense().numpy().astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Ensure L is a scipy.sparse matrix and convert it to a dense array
    L_dense = L.toarray() if sp.issparse(L) else L

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L_dense)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float() 
    
    return g


def wl_positional_encoding(g):
    """
    WL-based absolute positional embedding adapted from Graph-Bert.
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # Initialize
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            new_color_dict[node] = hash_object.hexdigest()

        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]

        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict

        iteration_count += 1

    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


class CorasDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        """
        Loading Cora dataset.
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        # Directly load Cora instead of pickle
        self.train = load_CorasDataSetDGL(name="Cora", split='train')
        self.val = load_CorasDataSetDGL(name="Cora", split='val')
        self.test = load_CorasDataSetDGL(name="Cora", split='test')

        print(f"train, test, val sizes: {len(self.train)}, {len(self.test)}, {len(self.val)}")
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        """
        Form a mini-batch from a given list of (graph, label) pairs.
        """
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _add_self_loops(self):
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_wl_positional_encodings(self):
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]
