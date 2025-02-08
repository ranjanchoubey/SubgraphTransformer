def visualize_graph(graph):
    plt.figure(figsize=(15, 10))
    
    # Create networkx graph from DGL graph
    G = graph.to_networkx().to_undirected()
    
    # Get node labels - for DGL graph
    labels = graph.ndata['label'].numpy()
    unique_labels = np.unique(labels)
    
    # Strong colors for nodes
    distinct_colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
        '#ff7f00', '#a65628', '#756bb1', '#636363'
    ]
    colors = (distinct_colors * (len(unique_labels) // len(distinct_colors) + 1))[:len(unique_labels)]
    node_colors = [colors[label] for label in labels]
    
    # Compute layout with more spacing
    pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=100, seed=42)
    
    # Draw edges FIRST with darker color
    nx.draw_networkx_edges(G, pos, edge_color='#000000', alpha=0.20, width=0.9)
    
    # Draw nodes SECOND
    nodes = nx.draw_networkx_nodes(G, pos, node_size=15, node_color=node_colors, edgecolors='white', linewidths=0.2)
    
    # Legend
    legend_elements = [plt.Line2D([0], [0],
                                marker='o',
                                color='w',
                                markerfacecolor=colors[i],
                                label=f'Class {i}',
                                markersize=10,
                                markeredgecolor='white') for i in range(len(unique_labels))]
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, facecolor='white')
    
    plt.title(f'Cora Citation Network'
             f'\nNodes: {graph.number_of_nodes()}, '
             f'\nEdges: {graph.number_of_edges()}'
             f'\nFeatures per node: {graph.ndata["feat"].shape[1]}, '
             f'\nNumber of classes: {len(torch.unique(graph.ndata["label"]))}'
             f'\ndata format: {graph}')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_subgraphs(cluster_data, num_plots=10):
    """Visualize multiple subgraphs in a grid layout with borders"""
    
    num_plots = min(num_plots, len(cluster_data))
    n_cols = min(5, num_plots)
    n_rows = (num_plots - 1) // n_cols + 1
    
    # Increase figure size and spacing
    fig = plt.figure(figsize=(4.5*n_cols, 4.5*n_rows))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    for idx in range(num_plots):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        subgraph = cluster_data[idx]
        
        # Add subplot border and background
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_facecolor('#FFFFFF')  # Light gray background
        
        # Create networkx graph
        G = nx.Graph()
        edge_index = subgraph.edge_index.numpy()
        nodes = range(subgraph.x.size(0))
        G.add_nodes_from(nodes)
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        
        # Node color mapping
        labels = subgraph.y.numpy()
        unique_labels = np.unique(labels)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_mapping[l] for l in labels])
        
        distinct_colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
            '#ff7f00', '#a65628', '#756bb1', '#636363'
        ]
        num_classes = len(label_mapping)
        colors = (distinct_colors * ((num_classes - 1) // len(distinct_colors) + 1))[:num_classes]
        node_colors = [colors[int(l)] for l in mapped_labels]
        
        # Layout and draw
        pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50, seed=42)
        nx.draw_networkx_edges(G, pos, edge_color='#000000', alpha=0.2, width=0.9)
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=list(G.nodes()),
                             node_size=30,
                             node_color=node_colors,
                             edgecolors='white',
                             linewidths=0.2)
        
        plt.title(f'Subgraph {idx}\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}',
                 pad=10)  # Add padding to title
        plt.axis('on')  # Turn axis on to show borders
        ax.set_xticks([])  # Remove ticks
        ax.set_yticks([])
    
    plt.tight_layout(pad=3.0)
    plt.show()