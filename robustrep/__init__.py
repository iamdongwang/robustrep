"""robustrep: robust, transport-agnostic reputation scoring for AI agents."""
from .config import Config
from .pipeline import score

__all__ = ["Config", "score"]
__version__ = "0.1.0"
