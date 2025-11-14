# tests/conftest.py

import sys
from pathlib import Path

# Get project root: one level up from tests/
ROOT = Path(__file__).resolve().parents[1]

# Add project root to sys.path if not already there
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
