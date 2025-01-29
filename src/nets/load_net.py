"""
    Utility file to select GraphNN model as
    selected by the user
"""

from src.nets.transformer import GraphTransformer

def create_model(config):
    """
    Create a GraphTransformer model from config parameters.
    Args:
        config: Configuration object containing model parameters
    Returns:
        GraphTransformer model instance
    """
    try:
        model = GraphTransformer(
            input_dim=config.gcn.output_dim,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            ff_dim=config.model.ff_dim,
            dropout=config.model.dropout,
            num_classes=config.model.num_classes
        )
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise e