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

# Run main.py file (entry point of code)
python main.py  --dataset Cora   --config 'src/configs/default_config.json' --gpu_id 0





# Additional Scripts 

# to print directory structure
tree -I 'out|dataset|venv'

# Remove all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -r {} +
