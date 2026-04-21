"""FlowRefiner model."""

from .flowrefiner_model import FlowRefiner3D

__all__ = ['FlowRefiner3D', 'get_model']


def get_model(config):
    """Instantiate FlowRefiner with the given config."""
    return FlowRefiner3D(config)
