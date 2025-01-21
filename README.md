# Graph Transformer Project

This repository implements a Graph Transformer model designed for graph-structured data, particularly for node classification tasks on datasets such as Cora. The project provides tools for data processing, model architecture, training, evaluation, and visualization of results.

## Folder Structure

The project directory is structured as follows:

```
graph_transformer_project/
├── data/                      # For datasets
│   └── cora/                  # Cora dataset (store .pt or raw files)
├── src/                       # Source code folder
│   ├── __init__.py            # Makes src a module
│   ├── data_processing.py     # Code for loading and partitioning the graph
│   ├── embedding.py           # Code for mean-pooling and Laplacian positional embedding
│   ├── transformer.py         # Code for the transformer model architecture
│   ├── trainer.py             # Training loop, loss function, evaluation
│   └── utils.py               # Helper functions (e.g., metrics, plotting, etc.)
├── notebooks/                 # Jupyter notebooks for experiments
├── configs/                   # Configuration files for training parameters
│   └── config.yaml            # YAML file with hyperparameters and settings
├── outputs/                   # Output folder for storing results
│   ├── logs/                  # Logs from runs
│   ├── checkpoints/           # Model checkpoints
│   └── metrics/               # Evaluation metrics and plots
├── main.py                    # Main script to run the entire pipeline
└── README.md                  # Description and setup instructions
```

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- PyTorch
- PyTorch Geometric
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/graph_transformer_project.git
    cd graph_transformer_project
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Dataset

The Cora dataset will be automatically downloaded and processed when you run the main script.

## Usage

### Running the Pipeline

To run the entire pipeline, execute the [main.py](http://_vscodecontentref_/10) script:
```sh
python main.py
