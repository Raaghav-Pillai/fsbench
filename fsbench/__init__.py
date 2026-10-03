"""FS-EntropyBench: measure how filesystem structure affects agent task performance."""

from fsbench.config import FSConfig, PRESETS
from fsbench.generate import generate_env

__all__ = ["FSConfig", "PRESETS", "generate_env"]
__version__ = "0.1.0"
