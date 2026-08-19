"""ASTRA scientific engine.

Importing this package applies the cache redirects before any astronomy
library has a chance to resolve its own default paths.
"""

from . import config

__version__ = "0.1.0"

config.apply_cache_redirects()

__all__ = ["config", "__version__"]
