# Remove all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -r {} +


# to run main.py file (entry point of code)
python main.py  --dataset Cora   --config 'src/configs/default_config.json' --gpu_id 0


# I am using other project env for now, to activate use below command
source  /ranjan/graphtransformer/env/bin/activate

# to print directory structure
tree -I 'out|dataset'
