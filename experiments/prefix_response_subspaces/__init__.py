"""Leakage-safe prefix-specific one-step response-subspace experiments.

The legacy repository keeps its reusable package under ``src/`` without an
installed project. Make that checked-in package importable for the documented
``python -m experiments...`` interface; no environment installation occurs.
"""
from pathlib import Path
import sys

_SOURCE_ROOT = str(Path(__file__).resolve().parents[2] / "src")
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)
