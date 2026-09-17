"""
Root conftest.py

Ensures the project root is on sys.path so that all test modules can import
project packages (models, simulation, sensors, digital_twin, dashboard) using
absolute imports without per-file sys.path hacks.
"""

import sys
from pathlib import Path

# Add project root to sys.path (idempotent).
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
