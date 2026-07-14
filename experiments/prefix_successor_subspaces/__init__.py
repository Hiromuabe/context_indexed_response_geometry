"""Prefix-specific one-step successor subspace experiments.

The repository uses a ``src/`` layout without packaging metadata.  Module-style
commands are required to work directly from a fresh checkout, so expose that
existing source tree to this process before stage modules import the reusable
``prefix_displacement`` helpers.  This changes no environment or installation.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
