import yaml
from types import SimpleNamespace

def load_config(config_path):
    """Load YAML config file and return nested SimpleNamespace object."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    def dict_to_namespace(d):
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace
    
    return dict_to_namespace(config_dict)
