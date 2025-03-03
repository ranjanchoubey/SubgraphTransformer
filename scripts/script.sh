# clone repo
git clone https://github.com/ranjanchoubey/SubgraphTransformer.git

# go to folder 
cd SubgraphTransformer

# creat venv
python3 -m venv venv

# activate venv
source venv/bin/activate

# install dependencies
pip install -e ".[dev]"

# Run main.py file (entry point of code small dataset)
python main.py  --dataset CoraSmall   --config 'configs/cora_small.json' --gpu_id 0

# for CoraFull dataset
python main.py  --dataset CoraFull   --config 'configs/cora_full.json' --gpu_id 0

# for citeseer dataset
python main.py  --dataset Citeseer   --config 'configs/citeseer.json' --gpu_id 0

# for Pubmed dataset
python main.py  --dataset Pubmed   --config 'configs/pubmed.json' --gpu_id 0

# for Chameleon dataset
python main-hetero.py --dataset Chameleon --config 'configs/chameleon.json' --gpu_id 0

# Experiment Tracker link
https://docs.google.com/spreadsheets/d/1JUpCwrU6YX0tHYqXLDpRhz7GeOKgkvfStAXD7AqgojE/edit?gid=0#gid=0


# Additional Scripts 

# to print directory structure
tree -I 'out|dataset|venv'

# Remove all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -r {} +
