"""ProceX: A Python package for preprocessing medical images."""

import importlib.metadata

from .main import process_images

__version__ = importlib.metadata.version(__name__)
__all__ = ["process_images"]
