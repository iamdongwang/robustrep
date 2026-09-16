"""robustrep: robust, transport-agnostic reputation scoring for AI agents."""
from .config import Config
from .pipeline import score

__all__ = ["Config", "score"]
from importlib import metadata as _metadata

try:
    # Single source of truth: pyproject.toml, via the installed distribution.
    __version__ = _metadata.version("robustrep")
except _metadata.PackageNotFoundError:  # running from a bare checkout without an install
    __version__ = "0.0.0+unknown"
