"""Paper-ready plotting modules and shared styling helpers."""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mpl-cache")
)

from .style import PAPER_TICK_EDGE_PADDING_FRACTION, despine, use_paper_style

__all__ = [
    "despine",
    "PAPER_TICK_EDGE_PADDING_FRACTION",
    "use_paper_style",
]
