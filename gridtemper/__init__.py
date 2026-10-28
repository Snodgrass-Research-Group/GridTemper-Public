"""GridTemper — TX-2030 synthetic-grid case refinement pipeline.

Python conversion of the TX-2030-Case-Updates MATLAB codebase. See README.md
for architecture and usage. The detailed conversion manual (per-function
algorithm docs, deviations log, status tracker) that the module docstrings
reference under ``conversion_manual/`` lives in the source TX-2030-Case-Updates
repository.
"""

from .config import CaseConfig, TemperOptions
from .grid_model import GridModel, SingularNetworkError

__all__ = ["CaseConfig", "TemperOptions", "GridModel", "SingularNetworkError"]
__version__ = "0.1.0"
