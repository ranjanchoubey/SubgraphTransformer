import json
from types import SimpleNamespace

def load_config(config_path):
    """Load config from JSON file and convert to namespace object."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        # return config_dict
        # Convert nested dictionaries to SimpleNamespace
        return json.loads(json.dumps(config_dict), 
                        object_hook=lambda d: SimpleNamespace(**d))
        

