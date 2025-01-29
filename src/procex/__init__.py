"""ProceX: A Python package for preprocessing medical images."""

import importlib.metadata

from .main import process_images
from .transforms import ToTensor

__version__ = importlib.metadata.version(__name__)
__all__ = [
    "ToTensor",
    "process_images",
]
