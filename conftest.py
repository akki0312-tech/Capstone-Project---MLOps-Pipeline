"""
conftest.py - pytest configuration for the MLOps project.
Ensures the src/ directory is importable from tests/.
"""

import os
import sys

# Add src/ to path so tests can import from it directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
